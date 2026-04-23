"""Particle lookup by name with fuzzy suggestions on failure."""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Optional

from particle import Particle, ParticleNotFound
from particle.pdgid import is_hadron

from .flavor import parse_quark_content, FLAVORS
from .qn import j_range, parity


@lru_cache(maxsize=1)
def _hadron_list() -> list[Particle]:
    return [
        p for p in Particle.findall()
        if is_hadron(p.pdgid) and p.mass is not None and p.J is not None and p.P is not None
    ]


def suggest_particles(query: str, n: int = 5) -> list[str]:
    """Return up to n PDG particle names most similar to query."""
    hadrons = _hadron_list()
    scored: dict[str, float] = {}
    for p in hadrons:
        for candidate in (p.name, p.programmatic_name):
            r = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            scored[p.name] = max(scored.get(p.name, 0.0), r)
    return sorted(scored, key=lambda x: -scored[x])[:n]


def _resolve(name: str) -> Particle:
    """Resolve particle name, raising LookupError with suggestions on failure."""
    try:
        return Particle.from_name(name)
    except (ParticleNotFound, KeyError):
        pass
    suggestions = suggest_particles(name, n=5)
    raise LookupError(f"PARTICLE_NOT_FOUND:{name}:{','.join(suggestions)}")


def lowest_jp_combinations(
    name1: str,
    name2: str,
    n: int = 3,
    max_L: int = 10,
) -> list[tuple[float, int, int]]:
    """Return the n lowest J^P combinations (J, P, min_L) the pair can produce.

    Sorted by min_L first, then J.
    """
    p1 = _resolve(name1)
    p2 = _resolve(name2)
    both_bosons = (float(p1.J) % 1.0 < 1e-9) and (float(p2.J) % 1.0 < 1e-9)
    identical = (int(p1.pdgid) == int(p2.pdgid))

    seen: dict[tuple[float, int], int] = {}
    for L in range(0, max_L + 1):
        if identical and both_bosons and L % 2 != 0:
            continue
        P_tot = parity(int(p1.P), int(p2.P), L)
        for J_tot in j_range(float(p1.J), float(p2.J), L):
            key = (J_tot, P_tot)
            if key not in seen:
                seen[key] = L

    results = sorted(seen.items(), key=lambda x: (x[1], x[0][0]))
    return [(J, P, L) for (J, P), L in results[:n]]


def pair_can_produce(
    name1: str,
    name2: str,
    J_target: float,
    P_target: int,
    max_L: Optional[int] = None,
) -> bool:
    """Return True if the pair can produce J_target^P_target at any L <= max_L."""
    p1 = _resolve(name1)
    p2 = _resolve(name2)
    both_bosons = (float(p1.J) % 1.0 < 1e-9) and (float(p2.J) % 1.0 < 1e-9)
    identical = (int(p1.pdgid) == int(p2.pdgid))
    L_cap = max_L if max_L is not None else int(J_target + float(p1.J) + float(p2.J)) + 4

    for L in range(0, L_cap + 1):
        if identical and both_bosons and L % 2 != 0:
            continue
        if parity(int(p1.P), int(p2.P), L) != P_target:
            continue
        if any(abs(j - J_target) < 1e-9 for j in j_range(float(p1.J), float(p2.J), L)):
            return True
    return False


def resolve_flavor_filter_from_particles(
    name1: str,
    name2: str,
    mass_min: float,
    mass_max: float,
    J: Optional[float],
    P: Optional[int],
    extra_cli_args: str = "",
) -> tuple[dict[str, int], list[str]]:
    """Resolve two particle names to a combined flavor dict.

    Returns (flavor_dict, warnings).
    Raises ValueError if quark content is ambiguous for either particle.
    """
    p1 = _resolve(name1)
    p2 = _resolve(name2)

    qc1 = parse_quark_content(p1.quarks) if p1.quarks else None
    qc2 = parse_quark_content(p2.quarks) if p2.quarks else None

    unclear = []
    if qc1 is None:
        unclear.append((name1, p1.quarks or "unknown"))
    if qc2 is None:
        unclear.append((name2, p2.quarks or "unknown"))

    if unclear:
        clear_flavors: dict[str, int] = {}
        if qc1 is not None:
            for f, v in qc1.items():
                clear_flavors[f] = clear_flavors.get(f, 0) + v
        if qc2 is not None:
            for f, v in qc2.items():
                clear_flavors[f] = clear_flavors.get(f, 0) + v

        unclear_names = [n for n, _ in unclear]
        clear_names = [name1, name2]
        for n in unclear_names:
            clear_names.remove(n)

        msg_lines = ["Quark content is ambiguous (mixed/superposition state) for:"]
        for n, qs in unclear:
            msg_lines.append(f"  {n}  (PDG quarks string: '{qs}')")
        if clear_flavors:
            msg_lines.append(f"Determined from {clear_names}: " +
                             ", ".join(f"{f}={v:+d}" for f, v in sorted(clear_flavors.items())))
        msg_lines.append("")
        msg_lines.append("Set the remaining flavor flags manually. Example command:")
        cmd = _build_command(mass_min, mass_max, J, P, clear_flavors, extra_cli_args,
                             name1, name2)
        msg_lines.append(f"  {cmd}")
        raise ValueError("\n".join(msg_lines))

    combined: dict[str, int] = {}
    for qc in (qc1, qc2):
        for f, v in qc.items():
            combined[f] = combined.get(f, 0) + v

    warnings = []
    threshold = float(p1.mass) + float(p2.mass)
    if threshold > mass_max:
        warnings.append(
            f"WARNING: threshold of {name1} + {name2} = {threshold:.1f} MeV "
            f"is above mass_max = {mass_max:.1f} MeV"
        )
    elif threshold < mass_min:
        warnings.append(
            f"WARNING: threshold of {name1} + {name2} = {threshold:.1f} MeV "
            f"is below mass_min = {mass_min:.1f} MeV"
        )

    return combined, warnings


def _build_command(
    mass_min: float,
    mass_max: float,
    J: Optional[float],
    P: Optional[int],
    known_flavors: dict[str, int],
    extra: str,
    p1_name: Optional[str] = None,
    p2_name: Optional[str] = None,
) -> str:
    parts = ["threshold-finder", str(mass_min), str(mass_max)]
    if J is not None and P is not None:
        p_str = "+1" if P > 0 else "-1"
        parts += [str(J), p_str]
    for f in FLAVORS:
        if f in known_flavors:
            parts.append(f"--{f} {known_flavors[f]}")
        else:
            parts.append(f"--{f} ???")
    if p1_name and p2_name:
        parts.append(f"--particles '{p1_name}' '{p2_name}'")
    if extra:
        parts.append(extra)
    return " ".join(parts)
