"""Spin-dependent interactions: smeared contact hyperfine, spin-orbit (vector +
Thomas) and tensor (port of ``contact_hyperfine.jl`` and ``spin_fine_structure.jl``).

All spin operators use the post-(A14) GI prescription ``B K(r) B`` with
``B = (m1 m2/(E1 E2))^(1/2 + epsilon_i)`` a function of ``p^2``. The ``pair11`` /
``pair22`` terms use ``(m1, m1)`` / ``(m2, m2)`` both for ``B`` and for the
smearing width of the kernel.

Expectations and cross elements go through the :class:`~gimodel.waves.RadialWave`
interface and are therefore representation-independent; only
:func:`contact_matrix` and :func:`fine_structure_grid_matrices` build mesh operators.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import fd
from .coupling import alpha_s_r, tensor_kernel_coulomb_running
from .constants import L_SYMBOLS
from .parameters import GIParameters
from .quarks import ConstituentMasses
from .smearing import (
    smeared_confinement_S_prime_closed,
    smeared_contact_kernel,
    smeared_coulomb_G_prime_closed,
    tensor_kernel_smeared_coulomb,
)
from .waves import RadialWave

__all__ = [
    "FineStructureMultiplet",
    "spin_dot",
    "l_dot_s",
    "tensor_triplet_lj",
    "tensor_triplet_offdiag_same_j",
    "ContactHyperfine",
    "contact_matrix",
    "contact_expectation",
    "contact_hyperfine_shift",
    "contact_hyperfine_shift_momentum_sandwich",
    "contact_hyperfine_shift_active",
    "diagonal_fine_structure_active",
    "fine_structure_radial_kernels",
    "fine_structure_components",
    "fine_structure_split",
    "fine_structure_grid_matrices",
    "spin_orbit_mixing_components",
    "tensor_mixing_components",
    "FineStructureComponents",
    "FineStructureMatrices",
]


@dataclass(frozen=True)
class FineStructureMultiplet:
    """Spectroscopic labels ``(L_label, multiplicity = 2S+1, J)`` of one fixed sector."""

    L_label: str
    multiplicity: int
    J: int

    @property
    def L(self) -> int:
        return L_SYMBOLS[self.L_label]


# --- angular algebra ------------------------------------------------------------


def spin_dot(multiplicity) -> float:
    """``<S1.S2>``: -3/4 for the singlet (1), +1/4 for the triplet (3)."""
    if isinstance(multiplicity, FineStructureMultiplet):
        multiplicity = multiplicity.multiplicity
    S = (multiplicity - 1) / 2
    return 0.5 * (S * (S + 1) - 1.5)


def l_dot_s(L: int, S: int, J: int) -> float:
    """Eigenvalue ``[J(J+1) - L(L+1) - S(S+1)]/2`` of ``L.S``."""
    return 0.5 * (J * (J + 1) - L * (L + 1) - S * (S + 1))


def tensor_triplet_lj(L: int, J: int, S: int) -> float:
    """Diagonal ``<S12>`` in ``^3L_J``: ``-2(L+1)/(2L-1)``, ``2``, ``-2L/(2L+3)`` for ``J = L-1, L, L+1``."""
    if S != 1 or L <= 0:
        return 0.0
    if J == L - 1:
        return -2.0 * (L + 1) / (2 * L - 1)
    if J == L:
        return 2.0
    if J == L + 1:
        return -2.0 * L / (2 * L + 3)
    return 0.0


def tensor_triplet_offdiag_same_j(J: int, S: int) -> float:
    """Off-diagonal ``S12`` between ``^3(J-1)_J`` and ``^3(J+1)_J``: ``6 sqrt(J(J+1))/(2J+1)``."""
    if S != 1 or J <= 0:
        return 0.0
    return 6.0 * math.sqrt(J * (J + 1.0)) / (2 * J + 1)


# --- contact hyperfine ------------------------------------------------------------


@dataclass(frozen=True)
class ContactHyperfine:
    """The smeared spin-spin contact interaction (A15); acts in every ``L``.

    ``sandwich=None`` follows ``params.factors.contact_momentum_sandwich``.
    """

    params: GIParameters
    masses: ConstituentMasses
    multiplicity: int
    sandwich: bool | None = None

    def __post_init__(self) -> None:
        if self.multiplicity not in (1, 3):
            raise ValueError("contact hyperfine requires q-qbar spin multiplicity 1 or 3")
        if self.sandwich is None:
            object.__setattr__(self, "sandwich", bool(self.params.factors.contact_momentum_sandwich))

    def kernel(self, r):
        return smeared_contact_kernel(self.params, self.masses, r)

    @property
    def strength(self) -> float:
        m = self.masses
        factor = 1.0 if self.sandwich else 1 + self.params.factors.epsilon_c
        return (32 * math.pi / (9 * m.m1_GeV * m.m2_GeV)) * spin_dot(self.multiplicity) * factor

    @property
    def exponent(self) -> float:
        return fd.gi_spin_dependent_side_exponent(self.params.factors.epsilon_c)


def _check_uniform(r: np.ndarray) -> float:
    if len(r) < 2:
        raise ValueError("operator requires at least two mesh points")
    h = r[1] - r[0]
    d = np.diff(r)
    if not (np.all(d > 0) and np.allclose(d, h, rtol=1e-10, atol=1e-12)):
        raise ValueError("non-uniform r mesh")
    return float(h)


def _radial_sandwich_matrix(L: int, r: np.ndarray, h: float, m1: float, m2: float, exponent: float, k: np.ndarray):
    B = fd.relativization_matrix(L, r, h, m1, m2, exponent)
    return (B * k) @ B


def contact_matrix(op: ContactHyperfine, L: int, r: np.ndarray) -> np.ndarray:
    """Dense mesh contact operator ``strength * B_c K B_c`` (or ``strength * K`` locally)."""
    if L < 0:
        raise ValueError("contact matrix requires L >= 0")
    r = np.asarray(r, dtype=float)
    h = _check_uniform(r)
    k = np.asarray(op.kernel(r))
    if not op.sandwich:
        return op.strength * np.diag(k)
    m = op.masses
    key = ("contact", op.params, m.m1_GeV, m.m2_GeV, int(L), len(r), h, float(r[0]))

    def factory():
        return (_radial_sandwich_matrix(L, r, h, m.m1_GeV, m.m2_GeV, op.exponent, k),)

    return op.strength * fd._MATRIX_CACHE.get(key, factory)[0]


def contact_expectation(op: ContactHyperfine, L: int, wave: RadialWave) -> float:
    """``<contact>`` in a wave (Rayleigh quotient), for any wave representation."""
    m = op.masses
    if op.sandwich:
        radial = wave.contact_sandwich_expectation(m.m1_GeV, m.m2_GeV, L, op.exponent, op.kernel)
    else:
        radial = wave.radial_expect(op.kernel)
    return op.strength * radial


def contact_hyperfine_shift(
    params: GIParameters, masses: ConstituentMasses, multiplet: FineStructureMultiplet, wave: RadialWave
) -> float:
    """First-order *local* contact shift with the historical ``(1 + epsilon_c)`` factor."""
    op = ContactHyperfine(params, masses, multiplet.multiplicity, sandwich=False)
    return contact_expectation(op, multiplet.L, wave)


def contact_hyperfine_shift_momentum_sandwich(
    params: GIParameters, masses: ConstituentMasses, multiplet: FineStructureMultiplet, wave: RadialWave
) -> float:
    """Contact shift with the GI momentum-factor sandwich."""
    op = ContactHyperfine(params, masses, multiplet.multiplicity, sandwich=True)
    return contact_expectation(op, multiplet.L, wave)


def contact_hyperfine_shift_active(
    params: GIParameters, masses: ConstituentMasses, multiplet: FineStructureMultiplet, wave: RadialWave
) -> float:
    """Contact shift with the prescription configured in ``params``."""
    op = ContactHyperfine(params, masses, multiplet.multiplicity)
    return contact_expectation(op, multiplet.L, wave)


# --- fine-structure kernels -------------------------------------------------------


def _pair(m: float) -> ConstituentMasses:
    return ConstituentMasses(float(m), float(m))


def _vector_so_kernel(params: GIParameters, pair: ConstituentMasses, r):
    r0 = np.maximum(np.asarray(r, dtype=float), 1e-8)
    if params.factors.fine_structure_smeared_kernels:
        return np.asarray(smeared_coulomb_G_prime_closed(params, pair, r0)) / r0
    return (4 / 3) * np.asarray(alpha_s_r(r0)) / r0**3


def _scalar_so_kernel(params: GIParameters, pair: ConstituentMasses, r):
    r0 = np.maximum(np.asarray(r, dtype=float), 1e-8)
    if params.factors.fine_structure_smeared_kernels:
        return np.asarray(smeared_confinement_S_prime_closed(params, pair, r0)) / r0
    return params.potential.b / r0


def _tensor_kernel(params: GIParameters, masses: ConstituentMasses, r):
    if params.factors.fine_structure_smeared_kernels:
        return np.asarray(tensor_kernel_smeared_coulomb(params, masses, r))
    return np.asarray(tensor_kernel_coulomb_running(r))


class FineStructureKernels(NamedTuple):
    vector_11: np.ndarray
    vector_22: np.ndarray
    vector_12: np.ndarray
    scalar_11: np.ndarray
    scalar_22: np.ndarray
    tensor_12: np.ndarray


def fine_structure_radial_kernels(params: GIParameters, masses: ConstituentMasses, r) -> FineStructureKernels:
    """The six local A15-A16 kernels (middle operators of the ``B K B`` sandwiches)."""
    p11, p22 = _pair(masses.m1_GeV), _pair(masses.m2_GeV)
    return FineStructureKernels(
        vector_11=_vector_so_kernel(params, p11, r),
        vector_22=_vector_so_kernel(params, p22, r),
        vector_12=_vector_so_kernel(params, masses, r),
        scalar_11=_scalar_so_kernel(params, p11, r),
        scalar_22=_scalar_so_kernel(params, p22, r),
        tensor_12=_tensor_kernel(params, masses, r),
    )


def diagonal_fine_structure_active(params: GIParameters, L: int, multiplicity: int) -> bool:
    """Diagonal ``L.S`` and ``S12`` vanish for ``L = 0`` or ``S = 0``."""
    return bool(params.fine_structure.enabled) and L > 0 and multiplicity == 3


def _fine_structure_algebra(masses: ConstituentMasses, L: int, J: int, G11, G22, G12, S11, S22, T12):
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    ls = l_dot_s(int(L), 1, int(J))
    vector = ls * (G11 / (4 * m1**2) + G22 / (4 * m2**2) + G12 / (m1 * m2))
    thomas = -ls * (S11 / (4 * m1**2) + S22 / (4 * m2**2))
    tensor = tensor_triplet_lj(int(L), int(J), 1) * T12 / (12 * m1 * m2)
    return vector, thomas, vector + thomas, tensor, vector + thomas + tensor


# --- wave-generic spin expectations -------------------------------------------------


def _spin_expectation(params, pair: ConstituentMasses, L: int, wave: RadialWave, epsilon: float, kernel) -> float:
    if params.factors.fine_structure_momentum_sandwich:
        return wave.sandwich_expectation(
            pair.m1_GeV, pair.m2_GeV, L, fd.gi_spin_dependent_side_exponent(epsilon), kernel
        )
    return (1 + epsilon) * wave.radial_expect(kernel)


def _spin_cross_expectation(
    params, pair: ConstituentMasses, L_left: int, left: RadialWave, L_right: int, right: RadialWave, epsilon, kernel
) -> float:
    if params.factors.fine_structure_momentum_sandwich:
        return left.sandwich_cross(
            right, pair.m1_GeV, pair.m2_GeV, L_left, L_right, fd.gi_spin_dependent_side_exponent(epsilon), kernel
        )
    return (1 + epsilon) * left.radial_overlap(right, kernel)


def _spin_orbit_radial_integrals(params, masses, L, left, right) -> dict[str, float]:
    p11, p22 = _pair(masses.m1_GeV), _pair(masses.m2_GeV)
    f = params.factors

    def cross(pair, eps, kernel):
        return _spin_cross_expectation(params, pair, L, left, L, right, eps, kernel)

    return {
        "vector_11": cross(p11, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p11, r)),
        "vector_22": cross(p22, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p22, r)),
        "scalar_11": cross(p11, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p11, r)),
        "scalar_22": cross(p22, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p22, r)),
    }


class FineStructureComponents(NamedTuple):
    I_vector_11: float = 0.0
    I_vector_22: float = 0.0
    I_vector_12: float = 0.0
    I_scalar_11: float = 0.0
    I_scalar_22: float = 0.0
    I_tk: float = 0.0
    spin_orbit_vector: float = 0.0
    spin_orbit_thomas: float = 0.0
    spin_orbit: float = 0.0
    tensor: float = 0.0
    total: float = 0.0


def fine_structure_components(
    params: GIParameters,
    masses: ConstituentMasses,
    multiplet: FineStructureMultiplet,
    wave: RadialWave,
    *,
    enabled: bool = True,
) -> FineStructureComponents:
    """Diagonal vector/Thomas spin-orbit and tensor contributions in one wave."""
    L = multiplet.L
    if not enabled or not diagonal_fine_structure_active(params, L, multiplet.multiplicity):
        return FineStructureComponents()
    f = params.factors
    p11, p22 = _pair(masses.m1_GeV), _pair(masses.m2_GeV)

    def expect(pair, eps, kernel):
        return _spin_expectation(params, pair, L, wave, eps, kernel)

    G11 = expect(p11, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p11, r))
    G22 = expect(p22, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p22, r))
    S11 = expect(p11, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p11, r))
    S22 = expect(p22, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p22, r))
    G12 = expect(masses, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, masses, r))
    Tk = expect(masses, f.epsilon_t, lambda r: _tensor_kernel(params, masses, r))
    vector, thomas, so, tensor, total = _fine_structure_algebra(
        masses, L, multiplet.J, G11, G22, G12, S11, S22, Tk
    )
    return FineStructureComponents(G11, G22, G12, S11, S22, Tk, vector, thomas, so, tensor, total)


def fine_structure_split(
    params: GIParameters,
    masses: ConstituentMasses,
    multiplet: FineStructureMultiplet,
    wave: RadialWave,
    *,
    enabled: bool = True,
) -> float:
    """Total diagonal fine-structure shift (spin-orbit + tensor)."""
    return fine_structure_components(params, masses, multiplet, wave, enabled=enabled).total


class FineStructureMatrices(NamedTuple):
    spin_orbit_vector: np.ndarray
    spin_orbit_thomas: np.ndarray
    spin_orbit: np.ndarray
    tensor: np.ndarray
    total: np.ndarray


def _zero_fine_matrices(n: int) -> FineStructureMatrices:
    z = np.zeros((n, n))
    return FineStructureMatrices(z, z, z, z, z)


def fine_structure_grid_matrices(
    params: GIParameters,
    masses: ConstituentMasses,
    J: int,
    r: np.ndarray,
    h: float,
    *,
    L: int = 1,
    multiplicity: int = 3,
) -> FineStructureMatrices:
    """Triplet ``^3L_J`` spin-orbit and tensor operators as dense mesh matrices (A15-A16).

    Requires the smeared momentum-sandwich prescription.
    """
    r = np.asarray(r, dtype=float)
    if not diagonal_fine_structure_active(params, L, multiplicity):
        return _zero_fine_matrices(len(r))
    f = params.factors
    if not (f.fine_structure_momentum_sandwich and f.fine_structure_smeared_kernels):
        raise ValueError(
            "fine_structure_grid_matrices requires the Appendix-A smeared momentum-sandwich path"
        )
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    key = ("fine", params, m1, m2, int(L), len(r), float(h), float(r[0]))

    def factory():
        p11, p22 = _pair(m1), _pair(m2)

        def sandwich(pair, eps, kernel_values):
            return _radial_sandwich_matrix(
                L, r, h, pair.m1_GeV, pair.m2_GeV, fd.gi_spin_dependent_side_exponent(eps), kernel_values
            )

        return (
            sandwich(p11, f.epsilon_so_vector, _vector_so_kernel(params, p11, r)),
            sandwich(p22, f.epsilon_so_vector, _vector_so_kernel(params, p22, r)),
            sandwich(masses, f.epsilon_so_vector, _vector_so_kernel(params, masses, r)),
            sandwich(p11, f.epsilon_so_scalar, _scalar_so_kernel(params, p11, r)),
            sandwich(p22, f.epsilon_so_scalar, _scalar_so_kernel(params, p22, r)),
            sandwich(masses, f.epsilon_t, np.asarray(tensor_kernel_smeared_coulomb(params, masses, r))),
        )

    G11, G22, G12, S11, S22, T12 = fd._MATRIX_CACHE.get(key, factory)
    return FineStructureMatrices(*_fine_structure_algebra(masses, L, J, G11, G22, G12, S11, S22, T12))


# --- same-J mixing elements -------------------------------------------------------


class SpinOrbitMixingComponents(NamedTuple):
    I_vector_11: float
    I_vector_22: float
    I_scalar_11: float
    I_scalar_22: float
    vector: float
    thomas: float
    total: float
    angular: float


def spin_orbit_mixing_components(
    params: GIParameters,
    masses: ConstituentMasses,
    L_label: str,
    radial_left: RadialWave,
    radial_right: RadialWave | None = None,
    *,
    enabled: bool = True,
) -> SpinOrbitMixingComponents:
    """Antisymmetric spin-orbit element between ``|^1L_L>`` (left) and ``|^3L_L>`` (right).

    Angular convention ``<^1L_L| L.(S1-S2) |^3L_L> = sqrt(L(L+1))``.
    """
    if radial_right is None:
        radial_right = radial_left
    L = L_SYMBOLS[str(L_label)]
    if not enabled or L == 0:
        return SpinOrbitMixingComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    t = _spin_orbit_radial_integrals(params, masses, L, radial_left, radial_right)
    angular = math.sqrt(L * (L + 1.0))
    vector = angular * (t["vector_11"] / (4 * m1**2) - t["vector_22"] / (4 * m2**2))
    thomas = angular * (-t["scalar_11"] / (4 * m1**2) + t["scalar_22"] / (4 * m2**2))
    return SpinOrbitMixingComponents(
        t["vector_11"], t["vector_22"], t["scalar_11"], t["scalar_22"], vector, thomas, vector + thomas, angular
    )


class TensorMixingComponents(NamedTuple):
    I_tk: float
    angular: float
    total: float


def tensor_mixing_components(
    params: GIParameters,
    masses: ConstituentMasses,
    radial_left: RadialWave,
    radial_right: RadialWave,
    J: int,
    *,
    enabled: bool = True,
) -> TensorMixingComponents:
    """Tensor element between ``^3(J-1)_J`` (left) and ``^3(J+1)_J`` (right):
    ``I_tk * 6 sqrt(J(J+1))/(2J+1) / (12 m1 m2)``."""
    J = int(J)
    if not enabled or J <= 0:
        return TensorMixingComponents(0.0, 0.0, 0.0)

    def kernel(r):
        return _tensor_kernel(params, masses, r)

    if params.factors.fine_structure_momentum_sandwich:
        Itk = radial_left.sandwich_cross(
            radial_right,
            masses.m1_GeV,
            masses.m2_GeV,
            J - 1,
            J + 1,
            fd.gi_spin_dependent_side_exponent(params.factors.epsilon_t),
            kernel,
        )
    else:
        Itk = radial_left.radial_overlap(radial_right, kernel)
    angular = tensor_triplet_offdiag_same_j(J, 1)
    return TensorMixingComponents(Itk, angular, Itk * angular / (12 * masses.m1_GeV * masses.m2_GeV))


# --- p^2-eigenbasis operators (used by the FD solves) --------------------------------


def _eigenbasis_sandwich(L, r, h, tag, m1, m2, exponent, kernel_values) -> np.ndarray:
    lam = fd.p2_eigen(L, r, h)[0]
    b = fd.relativization_values(lam, m1, m2, exponent)
    M = fd.projected_diagonal(L, r, h, tag, kernel_values)
    return np.outer(b, b) * M


def contact_eigenbasis(op: ContactHyperfine, L: int, r: np.ndarray, h: float) -> np.ndarray:
    """:func:`contact_matrix` in the eigenbasis of the mesh ``p^2``."""
    m = op.masses
    tag = ("contact", op.params, m.m1_GeV, m.m2_GeV)
    if not op.sandwich:
        return op.strength * fd.projected_diagonal(L, r, h, tag, lambda: op.kernel(r))
    return op.strength * _eigenbasis_sandwich(L, r, h, tag, m.m1_GeV, m.m2_GeV, op.exponent, lambda: op.kernel(r))


def fine_structure_eigenbasis(
    params: GIParameters, masses: ConstituentMasses, J: int, r: np.ndarray, h: float, *, L: int, multiplicity: int
) -> np.ndarray | None:
    """Total fine-structure operator of :func:`fine_structure_grid_matrices` in the
    eigenbasis of the mesh ``p^2`` (``None`` when inactive)."""
    if not diagonal_fine_structure_active(params, L, multiplicity):
        return None
    f = params.factors
    if not (f.fine_structure_momentum_sandwich and f.fine_structure_smeared_kernels):
        raise ValueError("fine-structure operators require the Appendix-A smeared momentum-sandwich path")
    m1, m2 = masses.m1_GeV, masses.m2_GeV
    p11, p22 = _pair(m1), _pair(m2)

    def sandwich(kind, pair, eps, kernel):
        tag = (kind, params, pair.m1_GeV, pair.m2_GeV)
        return _eigenbasis_sandwich(
            L, r, h, tag, pair.m1_GeV, pair.m2_GeV, fd.gi_spin_dependent_side_exponent(eps), lambda: kernel(r)
        )

    G11 = sandwich("vso", p11, f.epsilon_so_vector, lambda x: _vector_so_kernel(params, p11, x))
    G22 = sandwich("vso", p22, f.epsilon_so_vector, lambda x: _vector_so_kernel(params, p22, x))
    G12 = sandwich("vso", masses, f.epsilon_so_vector, lambda x: _vector_so_kernel(params, masses, x))
    S11 = sandwich("sso", p11, f.epsilon_so_scalar, lambda x: _scalar_so_kernel(params, p11, x))
    S22 = sandwich("sso", p22, f.epsilon_so_scalar, lambda x: _scalar_so_kernel(params, p22, x))
    T12 = sandwich("tensor", masses, f.epsilon_t, lambda x: tensor_kernel_smeared_coulomb(params, masses, x))
    return _fine_structure_algebra(masses, L, J, G11, G22, G12, S11, S22, T12)[4]
