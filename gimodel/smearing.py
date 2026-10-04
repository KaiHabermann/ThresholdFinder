"""Appendix-A Gaussian smearing in closed form (port of ``smearing_appendix_a.jl``,
the smeared-kernel helpers of ``spin_fine_structure.jl``, the contact regulator
of ``contact_hyperfine.jl`` and ``central_potential_dispatch.jl``).

The universal width (A9) is

    sigma^2 = sigma0^2 (1/2 + 1/2 [4 m1 m2/(m1+m2)^2]^4) + s^2 (2 m1 m2/(m1+m2))^2

and each Gaussian of the running coupling is smeared to the width
``tau_k = 1/sqrt(1/sigma^2 + 1/gamma_k^2)`` (A12)-(A14). Only the production
closed-form path is ported; the diagnostic convolution variants are not.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf

from .constants import ALPHA_COEFFS, ALPHA_GAMMAS
from .coupling import central_potential
from .parameters import CentralPotentialMethod, GIParameters
from .quarks import ConstituentMasses

__all__ = [
    "contact_smearing_sigma",
    "smeared_coulomb_G_closed",
    "smeared_confinement_S_closed",
    "smeared_coulomb_G_prime_closed",
    "smeared_coulomb_G_second_closed",
    "smeared_confinement_S_prime_closed",
    "tensor_kernel_smeared_coulomb",
    "delta_sigma_3d",
    "smeared_contact_kernel",
    "central_potential_values",
    "appendix_a_closed_central_values",
]

_SQRT_PI = np.sqrt(np.pi)


def _masses(m1, m2=None) -> tuple[float, float]:
    if isinstance(m1, ConstituentMasses):
        return m1.m1_GeV, m1.m2_GeV
    return float(m1), float(m2)


def _out(value, like):
    return float(value) if np.ndim(like) == 0 else value


def contact_smearing_sigma(params: GIParameters, m1, m2=None) -> float:
    """Appendix A (A9) universal smearing width ``sigma(m1, m2)`` in GeV.

    Accepts ``(params, m1, m2)`` or ``(params, ConstituentMasses)``.
    """
    a, b = _masses(m1, m2)
    mass_factor = 4 * a * b / (a + b) ** 2
    reduced_twice = 2 * a * b / (a + b)
    return float(
        np.sqrt(
            params.smearing.sigma0**2 * (0.5 + 0.5 * mass_factor**4)
            + params.smearing.s**2 * reduced_twice**2
        )
    )


def _taus(params: GIParameters, m1: float, m2: float) -> list[float]:
    sigma = max(contact_smearing_sigma(params, m1, m2), 1.0e-12)
    return [1 / np.sqrt(1 / sigma**2 + 1 / g**2) for g in ALPHA_GAMMAS]


def smeared_coulomb_G_closed(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """Closed-form smeared Coulomb ``G~(r) = -sum_k 4 alpha_k erf(tau_k r)/(3 r)``.

    Call as ``(params, m1, m2, r)`` or ``(params, masses, r)``.
    """
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    x = np.asarray(r, dtype=float)
    taus = _taus(params, a, b)
    small = np.abs(x) < 1.0e-8
    xs = np.where(small, 1.0, x)
    limit = -sum(8 * al * t / (3 * _SQRT_PI) for al, t in zip(ALPHA_COEFFS, taus))
    val = -sum(4 * al * erf(t * xs) / (3 * xs) for al, t in zip(ALPHA_COEFFS, taus))
    return _out(np.where(small, limit, val), r)


def smeared_confinement_S_closed(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """Closed-form smeared linear confinement ``S~(r)``."""
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    x = np.asarray(r, dtype=float)
    sigma = max(contact_smearing_sigma(params, a, b), 1.0e-12)
    small = np.abs(x) < 1.0e-8
    xs = np.where(small, 1.0, x)
    z = sigma * xs
    bracket = np.exp(-(z**2)) / (_SQRT_PI * z) + (1 + 1 / (2 * z**2)) * erf(z)
    val = params.potential.b * xs * bracket + params.potential.c
    limit = 2 * params.potential.b / (_SQRT_PI * sigma) + params.potential.c
    return _out(np.where(small, limit, val), r)


def smeared_coulomb_G_prime_closed(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """``dG~/dr`` with the source clamp ``r >= 1e-7``."""
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-7)
    s = np.zeros_like(ri)
    for al, t in zip(ALPHA_COEFFS, _taus(params, a, b)):
        e = erf(t * ri)
        ep = 2 * t / _SQRT_PI * np.exp(-((t * ri) ** 2))
        s = s + (-(4 * al / 3) * (ep / ri - e / ri**2))
    return _out(s, r)


def smeared_coulomb_G_second_closed(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """``d2G~/dr2`` with the source clamp ``r >= 1e-7``."""
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-7)
    s = np.zeros_like(ri)
    for al, t in zip(ALPHA_COEFFS, _taus(params, a, b)):
        e = erf(t * ri)
        ep = 2 * t / _SQRT_PI * np.exp(-((t * ri) ** 2))
        fpp = -2 * t**2 * ep - 2 * ep / ri**2 + 2 * e / ri**3
        s = s + (-(4 * al / 3) * fpp)
    return _out(s, r)


def tensor_kernel_smeared_coulomb(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """Smeared tensor kernel ``(1/r) dG~/dr - d2G~/dr2`` (clamp ``r >= 1e-7``)."""
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    ri = np.maximum(np.asarray(r, dtype=float), 1.0e-7)
    val = (1.0 / ri) * np.asarray(smeared_coulomb_G_prime_closed(params, a, b, ri)) - np.asarray(
        smeared_coulomb_G_second_closed(params, a, b, ri)
    )
    return _out(val, r)


def smeared_confinement_S_prime_closed(params: GIParameters, m1, m2=None, r: ArrayLike | None = None):
    """``dS~/dr`` (zero for ``|r| < 1e-7``)."""
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    x = np.asarray(r, dtype=float)
    sigma = max(contact_smearing_sigma(params, a, b), 1.0e-12)
    small = np.abs(x) < 1.0e-7
    ri = np.where(small, 1.0, x)
    z = sigma * ri
    expz = np.exp(-(z**2))
    h = ri + 1 / (2 * sigma**2 * ri)
    hp = 1 - 1 / (2 * sigma**2 * ri**2)
    val = params.potential.b * (
        (-2 * sigma * ri / _SQRT_PI) * expz + hp * erf(z) + h * (2 * sigma / _SQRT_PI) * expz
    )
    return _out(np.where(small, 0.0, val), r)


def delta_sigma_3d(r: ArrayLike, sigma: float):
    """3D-normalised Gaussian ``sigma^3/pi^(3/2) exp(-sigma^2 r^2)``; ``4 pi int r^2 delta dr = 1``."""
    sigma = float(sigma)
    x = np.asarray(r, dtype=float)
    if not sigma > 0:
        return _out(np.zeros_like(x), r)
    return _out(sigma**3 / np.pi**1.5 * np.exp(-((sigma * x) ** 2)), r)


def smeared_contact_kernel(params: GIParameters, masses: ConstituentMasses, r: ArrayLike):
    """Contact density ``sum_k alpha_k delta_{tau_k}(r)`` from the Laplacian of ``G~`` (A15)."""
    sigma = contact_smearing_sigma(params, masses)
    x = np.asarray(r, dtype=float)
    val = sum(
        al * np.asarray(delta_sigma_3d(x, 1 / np.sqrt(1 / sigma**2 + 1 / g**2)))
        for al, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS)
    )
    return _out(val, r)


def appendix_a_closed_central_values(params: GIParameters, m1: float, m2: float, r: ArrayLike) -> np.ndarray:
    """Closed-form diagonal ``G~(r) + S~(r)``."""
    x = np.asarray(r, dtype=float)
    return np.asarray(smeared_coulomb_G_closed(params, m1, m2, x)) + np.asarray(
        smeared_confinement_S_closed(params, m1, m2, x)
    )


def central_potential_values(
    params: GIParameters,
    m1,
    m2=None,
    r: ArrayLike | None = None,
    *,
    method: CentralPotentialMethod | str | None = None,
) -> np.ndarray:
    """Diagonal central potential on ``r`` for the given method (default ``params.central``).

    ``APPENDIX_A_MOMENTUM_SANDWICH`` returns the same closed-form diagonal as
    ``APPENDIX_A_CLOSED_FORM``; the nonlocal ``A(p) G~ A(p)`` part lives in the
    Hamiltonian builder.
    """
    if isinstance(m1, ConstituentMasses):
        r, m2 = m2, None
    a, b = _masses(m1, m2)
    chosen = params.central if method is None else CentralPotentialMethod(method)
    x = np.asarray(r, dtype=float)
    if chosen is CentralPotentialMethod.POINTWISE:
        return np.asarray(central_potential(x, params), dtype=float)
    if chosen in (
        CentralPotentialMethod.APPENDIX_A_CLOSED_FORM,
        CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH,
    ):
        return appendix_a_closed_central_values(params, a, b, x)
    raise NotImplementedError(
        f"central potential method `{chosen.value}` is a GIModel.jl research comparator "
        "that is not ported; use appendix_a_momentum_sandwich, appendix_a_closed_form or pointwise"
    )
