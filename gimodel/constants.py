"""Model constants shared by every module (port of GIModel.jl ``constants.jl``).

The Godfrey–Isgur running coupling is a fixed sum of three Gaussians in
momentum space (``alpha_s(Q^2) = sum_k alpha_k exp(-Q^2 / 4 gamma_k^2)``),
whose coordinate-space Coulomb kernel is ``sum_k alpha_k erf(gamma_k r)``.
"""
from __future__ import annotations

import math

ALPHA_COEFFS: tuple[float, float, float] = (0.25, 0.15, 0.20)
ALPHA_GAMMAS: tuple[float, float, float] = (0.5, math.sqrt(10.0) / 2, math.sqrt(1000.0) / 2)

L_SYMBOLS: dict[str, int] = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}
L_LABELS: dict[int, str] = {v: k for k, v in L_SYMBOLS.items()}


def orbital_angular_momentum(label: str) -> int:
    """Return the orbital angular momentum encoded by a spectroscopic letter."""
    key = str(label)
    if key not in L_SYMBOLS:
        raise ValueError(f"unsupported orbital label `{label}`")
    return L_SYMBOLS[key]


def orbital_label(L: int) -> str:
    """Return the spectroscopic orbital letter for angular momentum ``L``."""
    value = int(L)
    if value not in L_LABELS:
        raise ValueError(f"unsupported orbital angular momentum `{L}`")
    return L_LABELS[value]
