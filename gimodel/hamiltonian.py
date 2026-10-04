"""Finite-difference central Hamiltonians (port of ``hamiltonian.jl``).

Relativised central Hamiltonian on the mesh::

    H0 = sqrt(p^2 + m1^2) + sqrt(p^2 + m2^2) + A(p) G~(r) A(p) + S~(r),
    A(p) = sqrt(1 + p^2/(E1 E2)),

for the production ``appendix_a_momentum_sandwich`` method (other methods use a
diagonal potential). All momentum functions are matrix functions of the
L-dependent mesh ``p^2``.
"""
from __future__ import annotations

import numpy as np

from . import fd
from .parameters import CentralPotentialMethod, GIParameters
from .quarks import ConstituentMasses
from .smearing import central_potential_values, smeared_confinement_S_closed, smeared_coulomb_G_closed

__all__ = [
    "relativistic_hamiltonian",
    "nonrelativistic_hamiltonian",
    "appendix_a_momentum_sandwich_matrix",
    "potential_diagonal",
]


def potential_diagonal(params: GIParameters, masses: ConstituentMasses, r: np.ndarray) -> np.ndarray:
    """Diagonal central potential for ``params.central``."""
    return central_potential_values(params, masses.m1_GeV, masses.m2_GeV, r)


def appendix_a_momentum_sandwich_matrix(
    params: GIParameters, masses: ConstituentMasses, L: int, r: np.ndarray, h: float
) -> np.ndarray:
    """``A(p) G~ A(p) + S~`` on the mesh."""
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    lam, V = fd.p2_eigen(L, r, h)
    lam = np.maximum(lam, 0.0)
    e1 = np.sqrt(lam + m1**2)
    e2 = np.sqrt(lam + m2**2)
    A = fd.matrix_function(np.sqrt(1 + lam / (e1 * e2)), V)
    g = np.asarray(smeared_coulomb_G_closed(params, m1, m2, r))
    s = np.asarray(smeared_confinement_S_closed(params, m1, m2, r))
    out = (A * g) @ A
    out[np.diag_indices_from(out)] += s
    return out


def relativistic_hamiltonian(
    params: GIParameters, masses: ConstituentMasses, L: int, *, ngrid: int = 450, rmax: float = 24.0
) -> tuple[np.ndarray, np.ndarray]:
    """Dense relativised central Hamiltonian ``H0`` and its mesh ``r`` (cached, read-only)."""
    r, h = fd.radial_grid(ngrid, rmax)
    key = ("H0", params, masses.m1_GeV, masses.m2_GeV, int(L), int(ngrid), float(rmax))

    def factory():
        m1, m2 = masses.m1_GeV, masses.m2_GeV
        lam, V = fd.p2_eigen(L, r, h)
        lam = np.maximum(lam, 0.0)
        kinetic = fd.matrix_function(np.sqrt(lam + m1**2), V) + fd.matrix_function(np.sqrt(lam + m2**2), V)
        if params.central is CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH:
            potential = appendix_a_momentum_sandwich_matrix(params, masses, L, r, h)
        else:
            potential = np.diag(potential_diagonal(params, masses, r))
        H = kinetic + potential
        return (0.5 * (H + H.T),)

    return fd._MATRIX_CACHE.get(key, factory)[0], r


def nonrelativistic_hamiltonian(
    params: GIParameters, masses: ConstituentMasses, L: int, *, ngrid: int = 450, rmax: float = 24.0
) -> tuple[np.ndarray, np.ndarray]:
    """Tridiagonal ``p^2/2mu + V`` comparator (dense), without the rest masses."""
    mu = masses.reduced_mass
    r, h = fd.radial_grid(ngrid, rmax)
    v = potential_diagonal(params, masses, r)
    diagonal = 1 / (mu * h**2) + L * (L + 1) / (2 * mu * r**2) + v
    off = np.full(ngrid - 1, -1 / (2 * mu * h**2))
    H = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    return H, r


# --- p^2-eigenbasis assembly (used by the FD solves) ---------------------------------
#
# Every operator of the model is ``f(p^2)`` or ``F(p^2) K(r) F(p^2)``. In the
# eigenbasis of the mesh ``p^2`` (``p^2 = V diag(lambda) V^T``) these become
# ``diag(f)`` and ``(F F^T) o (V^T K V)``. The Hamiltonian is orthogonally
# similar to its r-space form, so eigenvalues agree and waves are ``u = V y``,
# while each local kernel costs one matrix product instead of three.


def central_eigenbasis(params: GIParameters, masses: ConstituentMasses, L: int, r: np.ndarray, h: float) -> np.ndarray:
    """``H0`` in the eigenbasis of the mesh ``p^2``."""
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    lam = np.maximum(fd.p2_eigen(L, r, h)[0], 0.0)
    e1 = np.sqrt(lam + m1**2)
    e2 = np.sqrt(lam + m2**2)
    if params.central is CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH:
        a = np.sqrt(1 + lam / (e1 * e2))
        MG = fd.projected_diagonal(
            L, r, h, ("G~", params, m1, m2), lambda: smeared_coulomb_G_closed(params, m1, m2, r)
        )
        MS = fd.projected_diagonal(
            L, r, h, ("S~", params, m1, m2), lambda: smeared_confinement_S_closed(params, m1, m2, r)
        )
        H = np.outer(a, a) * MG + MS
    else:
        H = fd.projected_diagonal(
            L, r, h, ("V", params, params.central, m1, m2), lambda: potential_diagonal(params, masses, r)
        ).copy()
    H[np.diag_indices_from(H)] += e1 + e2
    return H
