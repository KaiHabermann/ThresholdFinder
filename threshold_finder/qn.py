"""Quantum number coupling utilities."""
from typing import Iterator


def j_range(j1: float, j2: float, L: int) -> Iterator[float]:
    """Yield all total J values from coupling j1, j2, and orbital L."""
    # First couple j1 + j2 -> s, then s + L -> J
    s_min = abs(j1 - j2)
    s_max = j1 + j2
    s = s_min
    while s <= s_max + 1e-9:
        j_lo = abs(s - L)
        j_hi = s + L
        j = j_lo
        while j <= j_hi + 1e-9:
            yield j
            j += 1.0
        s += 1.0


def parity(p1: int, p2: int, L: int) -> int:
    return p1 * p2 * ((-1) ** L)


def identical_bosons_L_allowed(L: int) -> bool:
    """For two identical bosons, spatial wave function must be symmetric: L even."""
    return L % 2 == 0


def identical_fermions_L_allowed(s_total: float, L: int) -> bool:
    """For two identical fermions, total wave function antisymmetric.
    Spin-singlet (s=0) is antisymmetric -> spatial must be symmetric (L even).
    Spin-triplet (s=1) is symmetric -> spatial must be antisymmetric (L odd).
    We can't know s from outside, so we allow all L (caller handles if needed).
    """
    return True


def can_produce(j1: float, p1: int, j2: float, p2: int,
                J_target: float, P_target: int,
                L: int, identical: bool, both_bosons: bool) -> bool:
    """Return True if these two particles in orbital L can produce J^P = J_target^P_target."""
    if identical and both_bosons and not identical_bosons_L_allowed(L):
        return False
    if parity(p1, p2, L) != P_target:
        return False
    return any(abs(j - J_target) < 1e-9 for j in j_range(j1, j2, L))
