"""Running coupling and the pointwise (unsmeared) Coulomb kernels (port of
``running_coupling.jl`` and the running-coupling helpers of ``spin_fine_structure.jl``).

All functions accept a float or a NumPy array of radii (GeV^-1) and return the
same shape. Small-r clamps are those of the Julia source.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf

from .constants import ALPHA_COEFFS, ALPHA_GAMMAS

__all__ = [
    "alpha_s_r",
    "alpha_s_q",
    "alpha_s_prime_r",
    "alpha_s_second_r",
    "coulomb_G_running",
    "coulomb_G_prime_running",
    "coulomb_G_second_running",
    "tensor_kernel_coulomb_running",
    "central_potential",
    "static_coulomb_G",
    "static_confinement_S",
    "dV_coul_central_dr",
]

_SQRT_PI = np.sqrt(np.pi)


def _out(value: np.ndarray, like: ArrayLike):
    return float(value) if np.ndim(like) == 0 else value


def erf_prime(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 2 / _SQRT_PI * np.exp(-(x**2))


def erf_second(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return -4 * x / _SQRT_PI * np.exp(-(x**2))


def alpha_s_r(r: ArrayLike):
    """Coordinate-space coupling ``alpha_s(r) = sum_k alpha_k erf(gamma_k r)``."""
    x = np.asarray(r, dtype=float)
    s = sum(a * erf(g * x) for a, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS))
    return _out(s, r)


def alpha_s_q(Q: ArrayLike):
    """GI running coupling ``alpha_s(Q^2) = sum_k alpha_k exp(-Q^2 / 4 gamma_k^2)`` (Eq. 12)."""
    x = np.asarray(Q, dtype=float)
    s = sum(a * np.exp(-(x**2) / (4 * g**2)) for a, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS))
    return _out(s, Q)


def alpha_s_prime_r(r: ArrayLike):
    x = np.asarray(r, dtype=float)
    s = sum(a * g * erf_prime(g * x) for a, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS))
    return _out(s, r)


def alpha_s_second_r(r: ArrayLike):
    x = np.asarray(r, dtype=float)
    s = sum(a * g**2 * erf_second(g * x) for a, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS))
    return _out(s, r)


def coulomb_G_running(r: ArrayLike):
    """``G(r) = -4 alpha_s(r) / (3 r)`` with the source clamp ``r >= 1e-9``."""
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-9)
    return _out(-(4.0 / 3.0) * np.asarray(alpha_s_r(ri)) / ri, r)


def coulomb_G_prime_running(r: ArrayLike):
    """``dG/dr = (4/3) [alpha_s/r^2 - alpha_s'/r]``."""
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-9)
    a = np.asarray(alpha_s_r(ri))
    ap = np.asarray(alpha_s_prime_r(ri))
    return _out((4.0 / 3.0) * (a / ri**2 - ap / ri), r)


def coulomb_G_second_running(r: ArrayLike):
    """``d2G/dr2 = (4/3) [2 alpha_s'/r^2 - 2 alpha_s/r^3 - alpha_s''/r]``."""
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-9)
    a = np.asarray(alpha_s_r(ri))
    ap = np.asarray(alpha_s_prime_r(ri))
    app = np.asarray(alpha_s_second_r(ri))
    return _out((4.0 / 3.0) * (2.0 * ap / ri**2 - 2.0 * a / ri**3 - app / ri), r)


def tensor_kernel_coulomb_running(r: ArrayLike):
    """Tensor kernel ``(1/r) dG/dr - d2G/dr2`` of the unsmeared running Coulomb potential."""
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-9)
    val = (1.0 / ri) * np.asarray(coulomb_G_prime_running(ri)) - np.asarray(coulomb_G_second_running(ri))
    return _out(val, r)


def central_potential(r: ArrayLike, params):
    """Pointwise ``V = b r - 4 alpha_s(r)/(3 r) + c`` (diagnostic baseline)."""
    x = np.asarray(r, dtype=float)
    val = params.potential.b * x - (4 / 3) * np.asarray(alpha_s_r(x)) / x + params.potential.c
    return _out(val, r)


def static_coulomb_G(r: ArrayLike, params=None):
    """Coulomb piece ``G(r) = -4 alpha_s / (3 r)`` with clamp ``r >= 1e-12``."""
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-12)
    return _out(-(4 / 3) * np.asarray(alpha_s_r(ri)) / ri, r)


def static_confinement_S(r: ArrayLike, params):
    """Confinement piece ``S(r) = b r + c``."""
    x = np.asarray(r, dtype=float)
    return _out(params.potential.b * x + params.potential.c, r)


def dV_coul_central_dr(r: ArrayLike, params=None):
    """Derivative of the running Coulomb potential ``G(r)``."""
    return coulomb_G_prime_running(r)
