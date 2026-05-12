"""Quantum number coupling utilities."""
from functools import reduce, lru_cache
from itertools import product
from typing import Iterator


def j_range(j1: float, j2: float, L: int) -> Iterator[float]:
    """Yield all total J values from coupling j1, j2, and orbital L."""
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


def _couple_two(j_set: set[float], j2: float) -> set[float]:
    """Return all J values reachable by coupling any j in j_set with j2."""
    result: set[float] = set()
    for j1 in j_set:
        lo = abs(j1 - j2)
        hi = j1 + j2
        j = lo
        while j <= hi + 1e-9:
            result.add(round(j * 2) / 2)
            j += 1.0
    return result


def j_range_nbody(spins: list[float], L: int) -> set[float]:
    """Return the set of total J values reachable by coupling all particle spins and orbital L."""
    if not spins:
        return set()
    j_set: set[float] = {round(spins[0] * 2) / 2}
    for s in spins[1:]:
        j_set = _couple_two(j_set, s)
    j_set = _couple_two(j_set, float(L))
    return j_set


def parity(p1: int, p2: int, L: int) -> int:
    return p1 * p2 * ((-1) ** L)


def parity_nbody(parities: list[int], L: int) -> int:
    """Parity of an n-body state: product of intrinsic parities times (-1)^L."""
    return reduce(lambda a, b: a * b, parities, 1) * ((-1) ** L)


def identical_bosons_L_allowed(L: int) -> bool:
    """For two identical bosons, spatial wave function must be symmetric: L even."""
    return L % 2 == 0


def identical_fermions_L_allowed(s_total: float, L: int) -> bool:
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


def can_produce_nbody(
    spins: list[float],
    parities: list[int],
    J_target: float,
    P_target: int,
    L: int,
) -> bool:
    """Return True if n particles with these spins/parities in total orbital L can produce J^P."""
    if parity_nbody(parities, L) != P_target:
        return False
    reachable = j_range_nbody(spins, L)
    return any(abs(j - J_target) < 1e-9 for j in reachable)


# ---------------------------------------------------------------------------
# Sequential coupling scheme for n-body states
# ---------------------------------------------------------------------------

def _couple_step(
    states: set[tuple[float, int]],
    j_next: float,
    p_next: int,
    L: int,
) -> set[tuple[float, int]]:
    """Couple each (J, P) intermediate state with the next particle at orbital L.

    Returns the set of reachable (J_total, P_total) after this step.
    """
    result: set[tuple[float, int]] = set()
    p_factor = p_next * ((-1) ** L)
    for j_cur, p_cur in states:
        p_new = p_cur * p_factor
        j_lo = abs(j_cur - j_next)
        j_hi = j_cur + j_next
        # then add L via triangle
        j_lo2 = abs(j_lo - L)  # lower bound after adding L is conservative; reuse _couple_two
        # couple (j_cur + j_next) first, then add L
        s = j_lo
        while s <= j_hi + 1e-9:
            jl = abs(s - L)
            jh = s + L
            jj = jl
            while jj <= jh + 1e-9:
                result.add((round(jj * 2) / 2, p_new))
                jj += 1.0
            s += 1.0
    return result


def couple_sequential(
    spins: list[float],
    parities: list[int],
    L_tuple: tuple[int, ...],
) -> set[tuple[float, int]]:
    """Couple n particles sequentially using the given L values.

    L_tuple has length n-1: L_tuple[i] is the orbital AM between particle i+1
    and the (0..i) subsystem.

    Returns the set of (J_total, P_total) reachable by this coupling scheme.
    """
    n = len(spins)
    assert len(L_tuple) == n - 1
    # seed: first particle contributes its own J and P, with no orbital term yet
    states: set[tuple[float, int]] = {(round(spins[0] * 2) / 2, parities[0])}
    for i in range(1, n):
        states = _couple_step(states, spins[i], parities[i], L_tuple[i - 1])
    return states


def _lex_L_tuples(n_L: int, L_max: int) -> Iterator[tuple[int, ...]]:
    """Yield all tuples of length n_L with values in [0, L_max] in lex order."""
    yield from product(range(L_max + 1), repeat=n_L)


@lru_cache(maxsize=512)
def _prefix_states(
    spins: tuple[float, ...],
    parities: tuple[int, ...],
    L_max: int,
) -> list[dict[tuple[int, ...], frozenset[tuple[float, int]]]]:
    """Pre-compute intermediate coupling states for all L-prefixes.

    Returns a list of length n-1 where entry k maps each L-prefix tuple of
    length k+1 to the set of (J, P) states reachable after coupling particles
    0..k+1 with those L values.

    This avoids recomputing the same prefix many times when iterating over
    all L-tuples.
    """
    n = len(spins)
    n_L = n - 1
    seed: frozenset[tuple[float, int]] = frozenset({(round(spins[0] * 2) / 2, parities[0])})

    # level[k] maps L-prefix of length k to the states after coupling particles 0..k
    # level[0] = {(): seed}  (before any coupling step)
    levels: list[dict[tuple[int, ...], frozenset[tuple[float, int]]]] = [
        {(): seed}
    ]
    for step in range(n_L):
        prev = levels[step]
        curr: dict[tuple[int, ...], frozenset[tuple[float, int]]] = {}
        j_next = spins[step + 1]
        p_next = parities[step + 1]
        for prefix, states in prev.items():
            for L in range(L_max + 1):
                new_states = frozenset(_couple_step(set(states), j_next, p_next, L))
                curr[prefix + (L,)] = new_states
        levels.append(curr)

    return levels[1:]  # drop the seed level; entries 0..n_L-1 correspond to steps 1..n_L


def first_L_tuple(
    spins: list[float],
    parities: list[int],
    J_target: float,
    P_target: int,
    L_max: int,
) -> tuple[int, ...] | None:
    """Return the lexicographically first L-tuple that lets these particles produce J^P.

    Uses cached prefix states to avoid redundant coupling work across calls with
    the same spin/parity signature.
    Returns None if no such tuple exists within L_max.
    """
    n_L = len(spins) - 1
    if n_L == 0:
        return None
    levels = _prefix_states(tuple(spins), tuple(parities), L_max)
    final = levels[-1]
    for L_tuple in _lex_L_tuples(n_L, L_max):
        states = final[L_tuple]
        if any(abs(j - J_target) < 1e-9 and p == P_target for j, p in states):
            return L_tuple
    return None


def all_L_tuples(
    spins: list[float],
    parities: list[int],
    J_target: float,
    P_target: int,
    L_max: int,
) -> list[tuple[int, ...]]:
    """Return all L-tuples (up to L_max per component) that produce J^P.

    Uses cached prefix states to avoid redundant coupling work across calls with
    the same spin/parity signature.
    """
    n_L = len(spins) - 1
    if n_L == 0:
        return []
    levels = _prefix_states(tuple(spins), tuple(parities), L_max)
    final = levels[-1]
    return [
        L_tuple for L_tuple in _lex_L_tuples(n_L, L_max)
        if any(abs(j - J_target) < 1e-9 and p == P_target for j, p in final[L_tuple])
    ]
