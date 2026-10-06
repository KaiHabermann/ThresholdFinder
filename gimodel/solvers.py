"""Solver options (port of ``solver_options.jl``).

Two kinds of knob, kept apart because the distinction is physical:

* :class:`RadialSolver` -- *how* the radial problem is solved (method and
  resolution). Changing it must not move a mass beyond discretisation error.
* :class:`SpinTerms` -- *what* is in the Hamiltonian; each switch is a paper
  equation, so changing it is meant to move masses.

Solver dispatch
---------------
Every solver implements two hooks, which is all the spectrum machinery calls:

* ``solve_central(params, masses, L, nlevels) -> ChannelRadialSolution``
* ``solve_fixed_channel(params, masses, multiplet, terms, nlevels) -> ChannelRadialSolution``

The returned waves must implement :class:`gimodel.waves.RadialWave`; all
contributions, mixing elements and observables are computed through that wave
interface, so a new representation plugs in without touching the spectrum code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .channel import ChannelRadialSolution
    from .parameters import GIParameters
    from .quarks import ConstituentMasses
    from .spin import FineStructureMultiplet

__all__ = ["RadialSolver", "FiniteDifferenceSolver", "OscillatorSolver", "SpinTerms"]


@dataclass(frozen=True)
class SpinTerms:
    """Which spin-dependent terms enter the mass (one switch per paper equation).

    * ``contact_hyperfine`` -- smeared spin-spin contact term;
    * ``fine_structure`` -- spin-orbit (vector + Thomas) and tensor (also gated by
      ``params.fine_structure.enabled``);
    * ``same_j_spin_orbit`` -- same-J antisymmetric spin-orbit mixing (unequal flavours);
    * ``tensor`` -- same-J tensor mixing (``L = J -/+ 1``).
    """

    contact_hyperfine: bool = True
    fine_structure: bool = True
    same_j_spin_orbit: bool = True
    tensor: bool = True


class RadialSolver(ABC):
    """Abstract radial solver; see the module docstring for the two hooks."""

    nlevels_per_channel: int

    @abstractmethod
    def solve_central(
        self, params: GIParameters, masses: ConstituentMasses, L: int, nlevels: int
    ) -> ChannelRadialSolution:
        """Spin-independent central solve for orbital ``L``."""

    @abstractmethod
    def solve_fixed_channel(
        self,
        params: GIParameters,
        masses: ConstituentMasses,
        multiplet: FineStructureMultiplet,
        terms: SpinTerms,
        nlevels: int,
    ) -> ChannelRadialSolution:
        """Diagonalise central + contact + diagonal fine structure in one ``(L,S,J)`` sector."""


@dataclass(frozen=True)
class FiniteDifferenceSolver(RadialSolver):
    """Uniform-mesh solve: ``ngrid`` interior points ``r_i = i rmax/(ngrid+1)``.

    ``kinetic`` is ``"relativistic"`` (production) or ``"nonrelativistic"``
    (central-only comparator); ``eigensolver`` is ``"full"`` (dense) or ``"krylov"``.
    """

    ngrid: int = 450
    rmax: float = 24.0
    kinetic: str = "relativistic"
    eigensolver: str = "full"
    nlevels_per_channel: int = 6

    def __post_init__(self) -> None:
        if self.ngrid < 2:
            raise ValueError(f"FiniteDifferenceSolver: ngrid must be ≥ 2, got {self.ngrid}")
        if not self.rmax > 0:
            raise ValueError(f"FiniteDifferenceSolver: rmax must be positive, got {self.rmax}")
        if self.kinetic not in ("relativistic", "nonrelativistic"):
            raise ValueError(
                f"FiniteDifferenceSolver: kinetic must be 'relativistic' or 'nonrelativistic', got `{self.kinetic}`"
            )
        if self.eigensolver not in ("full", "krylov"):
            raise ValueError(
                f"FiniteDifferenceSolver: eigensolver must be 'full' or 'krylov', got `{self.eigensolver}`"
            )
        if self.nlevels_per_channel < 1:
            raise ValueError(
                f"FiniteDifferenceSolver: nlevels_per_channel must be ≥ 1, got {self.nlevels_per_channel}"
            )
        object.__setattr__(self, "ngrid", int(self.ngrid))
        object.__setattr__(self, "rmax", float(self.rmax))

    @property
    def h(self) -> float:
        return self.rmax / (self.ngrid + 1)

    def solve_central(self, params, masses, L, nlevels):
        from .channel import fd_central_solution

        return fd_central_solution(self, params, masses, L, nlevels)

    def solve_fixed_channel(self, params, masses, multiplet, terms, nlevels):
        from .channel import fd_fixed_channel_solution

        return fd_fixed_channel_solution(self, params, masses, multiplet, terms, nlevels)

    def __str__(self) -> str:
        return (
            f"FiniteDifferenceSolver: ngrid = {self.ngrid}, rmax = {self.rmax} GeV^-1 "
            f"(h = {self.h:.5f}), {self.kinetic}, {self.eigensolver}, "
            f"{self.nlevels_per_channel} levels/channel"
        )


HO_DEFAULT_NBASIS = 24
HO_DEFAULT_MAX_NBASIS = 80
HO_DEFAULT_BASIS_STEP = 8
HO_DEFAULT_ENERGY_TOLERANCE_GEV = 1.0e-4
HO_DEFAULT_BETA_TOLERANCE_GEV = 2.0e-3
HO_BETA_GRID: tuple[float, ...] = tuple(round(0.25 + 0.10 * k, 10) for k in range(22))  # 0.25:0.10:2.35


@dataclass(frozen=True)
class OscillatorSolver(RadialSolver):
    """Harmonic-oscillator expansion (GI Eq. (A17)); see :mod:`gimodel.ho`.

    Field defaults and validation mirror GIModel.jl. ``max_nbasis=None`` means
    ``max(80, nbasis + 2 basis_step)``. One ``beta`` per sector minimises the highest
    requested level (grid scan + golden section to ``beta_tolerance_GeV``; an endpoint
    optimum is an error). With ``converge=True`` the basis grows by ``basis_step`` until
    two consecutive refinements move every requested level by at most
    ``energy_tolerance_GeV``; ``converge=False`` is one unchecked fixed-size solve.
    """

    nbasis: int = HO_DEFAULT_NBASIS
    max_nbasis: int | None = None
    basis_step: int = HO_DEFAULT_BASIS_STEP
    energy_tolerance_GeV: float = HO_DEFAULT_ENERGY_TOLERANCE_GEV
    beta_grid: tuple[float, ...] = field(default=HO_BETA_GRID)
    beta_tolerance_GeV: float = HO_DEFAULT_BETA_TOLERANCE_GEV
    converge: bool = True
    nlevels_per_channel: int = 6

    def __post_init__(self) -> None:
        if self.max_nbasis is None:
            object.__setattr__(
                self, "max_nbasis", max(HO_DEFAULT_MAX_NBASIS, self.nbasis + 2 * self.basis_step)
            )
        grid = tuple(float(b) for b in self.beta_grid)
        object.__setattr__(self, "beta_grid", grid)
        if self.nbasis < 1:
            raise ValueError(f"OscillatorSolver: nbasis must be ≥ 1, got {self.nbasis}")
        if self.basis_step < 1:
            raise ValueError(f"OscillatorSolver: basis_step must be ≥ 1, got {self.basis_step}")
        if self.max_nbasis < self.nbasis:
            raise ValueError(
                f"OscillatorSolver: max_nbasis={self.max_nbasis} is below nbasis={self.nbasis}"
            )
        if not self.energy_tolerance_GeV > 0:
            raise ValueError("OscillatorSolver: energy_tolerance_GeV must be positive")
        if len(grid) == 0:
            raise ValueError("OscillatorSolver: beta_grid must not be empty")
        if len(grid) == 2:
            raise ValueError(
                "OscillatorSolver: beta_grid needs one fixed value or at least three bracket points"
            )
        if not all(b > 0 for b in grid):
            raise ValueError(f"OscillatorSolver: every beta must be positive, got {list(grid)}")
        if not all(b > a for a, b in zip(grid, grid[1:])):
            raise ValueError("OscillatorSolver: beta_grid must be strictly increasing")
        if not self.beta_tolerance_GeV > 0:
            raise ValueError("OscillatorSolver: beta_tolerance_GeV must be positive")
        if self.nlevels_per_channel < 1:
            raise ValueError(
                f"OscillatorSolver: nlevels_per_channel must be ≥ 1, got {self.nlevels_per_channel}"
            )

    def solve_central(self, params, masses, L, nlevels):
        from .ho import oscillator_channel_solution

        return oscillator_channel_solution(self, params, masses, int(L), int(nlevels))

    def solve_fixed_channel(self, params, masses, multiplet, terms, nlevels):
        from .ho import oscillator_fixed_channel_solution

        return oscillator_fixed_channel_solution(self, params, masses, multiplet, terms, int(nlevels))

    def __str__(self) -> str:
        basis = (
            f"nbasis = {self.nbasis}:{self.basis_step}:{self.max_nbasis} adaptive, "
            f"ΔE ≤ {1000 * self.energy_tolerance_GeV} MeV"
            if self.converge
            else f"nbasis = {self.nbasis} unchecked"
        )
        return (
            f"OscillatorSolver: {basis}, beta in [{self.beta_grid[0]}, {self.beta_grid[-1]}] GeV "
            f"({len(self.beta_grid)} bracket points, refined to {self.beta_tolerance_GeV} GeV), "
            f"{self.nlevels_per_channel} levels/channel"
        )
