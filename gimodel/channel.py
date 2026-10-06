"""Radial channel solves (port of ``sector_solver.jl``, ``channel_solver.jl``,
``fixed_channel_solver.jl`` and the nonperturbative-contact helpers of
``contact_hyperfine.jl``).

``channel_solution`` solves the spin-independent central problem for one ``L``;
``fixed_channel_solution`` diagonalises the complete fixed-``(L,S,J)``
Hamiltonian (central + contact + symmetric spin-orbit + diagonal tensor).
The solver chooses the representation; FD implementations live here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import fd
from .constants import L_SYMBOLS
from .hamiltonian import central_eigenbasis, nonrelativistic_hamiltonian, relativistic_hamiltonian
from .parameters import GIParameters
from .quarks import ConstituentMasses
from .solvers import FiniteDifferenceSolver, RadialSolver, SpinTerms
from .spin import (
    ContactHyperfine,
    FineStructureMultiplet,
    contact_eigenbasis,
    contact_matrix,
    fine_structure_eigenbasis,
    fine_structure_grid_matrices,
)
from .waves import MeshWave, RadialWave, physically_normalized_waves

__all__ = [
    "RadialChannelKey",
    "ChannelRadialSolution",
    "SectorComputation",
    "channel_solution",
    "fixed_channel_solution",
    "fixed_channel_matrices",
    "resummed_channel_solution",
    "contact_hyperfine_nonperturbative_states",
    "solve_sector",
    "MIN_POINTS_ACROSS_STATE",
]

MIN_POINTS_ACROSS_STATE = 9


def _sig12(x: float) -> float:
    return float(f"{float(x):.12g}")


@dataclass(frozen=True)
class RadialChannelKey:
    """Cache key: masses rounded to 12 significant digits, ``L`` label, and
    ``(multiplicity, J)`` -- ``(0, 0)`` marks a central spin-independent channel."""

    m1_GeV: float
    m2_GeV: float
    L_label: str
    multiplicity: int = 0
    J: int = 0

    @classmethod
    def of(cls, masses: ConstituentMasses, L_label: str, multiplicity: int = 0, J: int = 0) -> RadialChannelKey:
        return cls(_sig12(masses.m1_GeV), _sig12(masses.m2_GeV), str(L_label), int(multiplicity), int(J))


class ChannelRadialSolution:
    """Lowest radial eigenvalues (GeV) and their phase-fixed native waves."""

    def __init__(self, eigenvalues_GeV, waves, *, convergence: Any = None) -> None:
        waves = list(waves)
        if len(eigenvalues_GeV) != len(waves):
            raise ValueError("ChannelRadialSolution: eigenvalue/wave counts differ")
        values = np.array(eigenvalues_GeV, dtype=float)
        values.flags.writeable = False
        self.eigenvalues_GeV: np.ndarray = values
        self.waves: tuple[RadialWave, ...] = tuple(w.fix_outer_phase() for w in waves)
        self.convergence = convergence

    @classmethod
    def from_mesh(cls, eigenvalues_GeV, vectors: np.ndarray, r: np.ndarray) -> ChannelRadialSolution:
        h = r[1] - r[0] if len(r) > 1 else 1.0
        return cls(eigenvalues_GeV, [MeshWave(vectors[:, n], r, h) for n in range(vectors.shape[1])])

    def radial_wave(self, radial_level: int) -> RadialWave:
        """Wave of radial level ``n`` (1-based, as in spectroscopic notation)."""
        if not 1 <= radial_level <= len(self.waves):
            raise ValueError(f"radial_level={radial_level} outside stored range 1:{len(self.waves)}")
        return self.waves[radial_level - 1]

    def __repr__(self) -> str:
        vals = ", ".join(f"{e:.4f}" for e in self.eigenvalues_GeV)
        kind = type(self.waves[0]).__name__ if self.waves else "RadialWave"
        return f"ChannelRadialSolution{{{kind}}}({vals} GeV)"


@dataclass
class SectorComputation:
    """Parameters, solver and the cache of native radial solves of one spectrum."""

    params: GIParameters
    solver: RadialSolver
    channel_cache: dict[RadialChannelKey, ChannelRadialSolution]


# --- FD implementations ------------------------------------------------------------


def _warn_if_underresolved(values, vectors, r, h, masses: ConstituentMasses, L: int) -> None:
    if len(values) == 0:
        return
    u = vectors[:, 0]
    nrm = np.sum(u**2) * h
    if not nrm > 0:
        return
    rms = np.sqrt(np.sum(u**2 * r**2) * h / nrm)
    pts = rms / h
    if pts >= MIN_POINTS_ACROSS_STATE:
        return
    warnings.warn(
        f"Radial grid too coarse for this state: its RMS radius spans only {pts:.1f} grid points "
        f"(want >= {MIN_POINTS_ACROSS_STATE}). Expect several MeV of error on the eigenvalues and much "
        f"worse on spin-dependent quantities. Increase `ngrid` (or reduce `rmax`). "
        f"(m1={masses.m1_GeV}, m2={masses.m2_GeV}, L={L}, rms_radius={rms}, h={h})",
        RuntimeWarning,
        stacklevel=3,
    )


def fd_central_solution(
    solver: FiniteDifferenceSolver, params: GIParameters, masses: ConstituentMasses, L: int, nlevels: int
) -> ChannelRadialSolution:
    if solver.kinetic == "relativistic":
        r, h = fd.radial_grid(solver.ngrid, solver.rmax)
        values, y = fd.lowest_eigenpairs(central_eigenbasis(params, masses, L, r, h), nlevels, solver.eigensolver)
        vectors = fd.p2_eigen(L, r, h)[1] @ y
        offset = 0.0
    else:
        H, r = nonrelativistic_hamiltonian(params, masses, L, ngrid=solver.ngrid, rmax=solver.rmax)
        offset = masses.m1_GeV + masses.m2_GeV
        values, vectors = fd.lowest_eigenpairs(H, nlevels, solver.eigensolver)
    h = r[1] - r[0] if len(r) > 1 else 0.0
    _warn_if_underresolved(values, vectors, r, h, masses, L)
    waves = physically_normalized_waves(vectors, h if h > 0 else 1.0)
    return ChannelRadialSolution.from_mesh(values + offset, waves, r)


def fixed_channel_matrices(
    solver: FiniteDifferenceSolver,
    params: GIParameters,
    masses: ConstituentMasses,
    multiplet: FineStructureMultiplet,
    terms: SpinTerms = SpinTerms(),
) -> dict[str, np.ndarray]:
    """Dense r-space FD operators of one fixed sector: ``central``, ``contact``, the
    fine-structure parts and their sum ``total`` (plus the mesh ``r``).

    Diagnostic view, assembled directly as ``B K B`` products; the solver itself
    assembles the orthogonally similar operator in the ``p^2`` eigenbasis."""
    if solver.kinetic != "relativistic":
        raise ValueError(
            "the complete GI fixed-sector Hamiltonian is relativistic; use channel_solution with "
            "this nonrelativistic solver as a central-only comparator"
        )
    L = multiplet.L
    central, r = relativistic_hamiltonian(params, masses, L, ngrid=solver.ngrid, rmax=solver.rmax)
    h = r[1] - r[0]
    n = len(r)
    zero = np.zeros((n, n))
    contact = (
        contact_matrix(ContactHyperfine(params, masses, multiplet.multiplicity), L, r)
        if terms.contact_hyperfine
        else zero
    )
    if terms.fine_structure:
        fine = fine_structure_grid_matrices(
            params, masses, multiplet.J, r, h, L=L, multiplicity=multiplet.multiplicity
        )
        fine_parts = fine._asdict()
    else:
        fine_parts = dict(spin_orbit_vector=zero, spin_orbit_thomas=zero, spin_orbit=zero, tensor=zero, total=zero)
    total = central + contact + fine_parts["total"]
    total = 0.5 * (total + total.T)
    return dict(
        central=central,
        contact=contact,
        spin_orbit_vector=fine_parts["spin_orbit_vector"],
        spin_orbit_thomas=fine_parts["spin_orbit_thomas"],
        spin_orbit=fine_parts["spin_orbit"],
        tensor=fine_parts["tensor"],
        fine_structure=fine_parts["total"],
        total=total,
        r=r,
    )


def fd_fixed_channel_solution(
    solver: FiniteDifferenceSolver,
    params: GIParameters,
    masses: ConstituentMasses,
    multiplet: FineStructureMultiplet,
    terms: SpinTerms,
    nlevels: int,
) -> ChannelRadialSolution:
    if solver.kinetic != "relativistic":
        raise ValueError(
            "the complete GI fixed-sector Hamiltonian is relativistic; use channel_solution with "
            "this nonrelativistic solver as a central-only comparator"
        )
    L = multiplet.L
    r, h = fd.radial_grid(solver.ngrid, solver.rmax)
    total = central_eigenbasis(params, masses, L, r, h)
    if terms.contact_hyperfine:
        total = total + contact_eigenbasis(ContactHyperfine(params, masses, multiplet.multiplicity), L, r, h)
    if terms.fine_structure:
        fine = fine_structure_eigenbasis(params, masses, multiplet.J, r, h, L=L, multiplicity=multiplet.multiplicity)
        if fine is not None:
            total = total + fine
    total = 0.5 * (total + total.T)
    values, y = fd.lowest_eigenpairs(total, nlevels, solver.eigensolver)
    vectors = fd.p2_eigen(L, r, h)[1] @ y
    waves = physically_normalized_waves(vectors, r[1] - r[0])
    return ChannelRadialSolution.from_mesh(values, waves, r)


# --- public entry points ----------------------------------------------------------


def channel_solution(
    params: GIParameters,
    masses: ConstituentMasses,
    L: int,
    *,
    solver: RadialSolver | None = None,
    nlevels: int | None = None,
) -> ChannelRadialSolution:
    """Spin-independent central solve for one orbital channel."""
    solver = FiniteDifferenceSolver() if solver is None else solver
    nlevels = solver.nlevels_per_channel if nlevels is None else int(nlevels)
    return solver.solve_central(params, masses, int(L), nlevels)


def fixed_channel_solution(
    params: GIParameters,
    masses: ConstituentMasses,
    multiplet: FineStructureMultiplet,
    *,
    solver: RadialSolver | None = None,
    terms: SpinTerms = SpinTerms(),
    nlevels: int | None = None,
) -> ChannelRadialSolution:
    """Diagonalise the complete radial Hamiltonian for one fixed ``(L,S,J)`` sector."""
    solver = FiniteDifferenceSolver() if solver is None else solver
    nlevels = solver.nlevels_per_channel if nlevels is None else int(nlevels)
    return solver.solve_fixed_channel(params, masses, multiplet, terms, nlevels)


def resummed_channel_solution(
    params: GIParameters,
    masses: ConstituentMasses,
    L: int,
    V: np.ndarray,
    *,
    solver: FiniteDifferenceSolver | None = None,
    nlevels: int | None = None,
) -> ChannelRadialSolution:
    """FD diagnostic: diagonalise the central Hamiltonian plus a dense mesh operator ``V``."""
    solver = FiniteDifferenceSolver() if solver is None else solver
    if not isinstance(solver, FiniteDifferenceSolver):
        raise ValueError("resummed_channel_solution accepts an FD mesh operator only")
    nlevels = solver.nlevels_per_channel if nlevels is None else int(nlevels)
    H, r = relativistic_hamiltonian(params, masses, L, ngrid=solver.ngrid, rmax=solver.rmax)
    if V.shape[0] != len(r):
        raise ValueError("V must live on the (ngrid, rmax) mesh")
    total = H + V
    values, vectors = fd.lowest_eigenpairs(0.5 * (total + total.T), nlevels, solver.eigensolver)
    return ChannelRadialSolution.from_mesh(values, physically_normalized_waves(vectors, r[1] - r[0]), r)


def contact_hyperfine_nonperturbative_states(
    params: GIParameters,
    masses: ConstituentMasses,
    L_label: str,
    multiplicity: int,
    nlevels: int,
    *,
    r: np.ndarray | None = None,
    solver: RadialSolver | None = None,
) -> ChannelRadialSolution:
    """Fixed-sector solve with the contact term resummed and fine structure off.

    Passing a mesh ``r`` (FD only) re-derives ``(ngrid, rmax)`` from it.
    """
    ContactHyperfine(params, masses, multiplicity)  # validate before solving
    solver = FiniteDifferenceSolver() if solver is None else solver
    if r is not None:
        if not isinstance(solver, FiniteDifferenceSolver):
            raise ValueError("the overload carrying r is FD-only; omit r for native solver dispatch")
        r = np.asarray(r, dtype=float)
        if len(r) < 2:
            raise ValueError("contact solve requires at least two mesh points")
        h = r[1] - r[0]
        rmax = h * (len(r) + 1)
        rebuilt, _ = fd.radial_grid(len(r), rmax)
        if not np.allclose(r, rebuilt, rtol=1e-10, atol=1e-12):
            raise ValueError("contact solve requires the interior Dirichlet grid r[i] = i*h")
        solver = FiniteDifferenceSolver(
            ngrid=len(r), rmax=rmax, kinetic=solver.kinetic, eigensolver=solver.eigensolver,
            nlevels_per_channel=solver.nlevels_per_channel,
        )
    orbital = L_SYMBOLS[str(L_label)]
    J = orbital if multiplicity == 1 else orbital + 1
    return fixed_channel_solution(
        params,
        masses,
        FineStructureMultiplet(str(L_label), multiplicity, J),
        solver=solver,
        terms=SpinTerms(contact_hyperfine=True, fine_structure=False, same_j_spin_orbit=False, tensor=False),
        nlevels=nlevels,
    )


def solve_sector(
    params: GIParameters, equal_mass_GeV: float, *, maxn: int = 6, solver: RadialSolver | None = None
) -> dict[tuple[int, str], float]:
    """Equal-mass diagnostic sweep of central eigenvalues over every orbital letter."""
    m = float(equal_mass_GeV)
    masses = ConstituentMasses(m, m)
    results: dict[tuple[int, str], float] = {}
    for symbol, L in L_SYMBOLS.items():
        levels = channel_solution(params, masses, L, nlevels=maxn, solver=solver).eigenvalues_GeV
        for n, value in enumerate(levels, start=1):
            results[(n, symbol)] = float(value)
    return results
