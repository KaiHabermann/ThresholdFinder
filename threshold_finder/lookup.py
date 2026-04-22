"""Particle lookup by name with fuzzy suggestions on failure."""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Optional

from particle import Particle, ParticleNotFound
from particle.pdgid import is_hadron

from .flavor import parse_quark_content, FLAVORS


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


def lookup_particle(name: str) -> Particle:
    """Look up a particle by PDG name. Raises LookupError with suggestions on failure."""
    try:
        return Particle.from_name(name)
    except (ParticleNotFound, KeyError):
        pass
    suggestions = suggest_particles(name, n=5)
    raise LookupError(suggestions)


def resolve_flavor_filter_from_particles(
    name1: str,
    name2: str,
    mass_min: float,
    mass_max: float,
    J: float,
    P: int,
    extra_cli_args: str = "",
) -> tuple[dict[str, int], list[str]]:
    """Resolve two particle names to a combined flavor dict.

    Returns (flavor_dict, warnings).
    Raises LookupError if a particle is not found (with suggestions embedded).
    Raises ValueError if quark content is ambiguous for either particle.
    """
    p1 = _resolve(name1)
    p2 = _resolve(name2)

    qc1 = parse_quark_content(p1.quarks) if p1.quarks else None
    qc2 = parse_quark_content(p2.quarks) if p2.quarks else None

    # Report which particles have unclear quark content and fail with a helpful command
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
        cmd = _build_command(mass_min, mass_max, J, P, clear_flavors, extra_cli_args)
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


def _resolve(name: str) -> Particle:
    """Resolve particle name, raising a formatted LookupError with suggestions."""
    try:
        return Particle.from_name(name)
    except (ParticleNotFound, KeyError):
        pass
    suggestions = suggest_particles(name, n=5)
    raise LookupError(f"PARTICLE_NOT_FOUND:{name}:{','.join(suggestions)}")


def _build_command(
    mass_min: float,
    mass_max: float,
    J: float,
    P: int,
    known_flavors: dict[str, int],
    extra: str,
) -> str:
    from .result import _fmt_J
    p_str = "+1" if P > 0 else "-1"
    parts = ["threshold-finder", str(mass_min), str(mass_max), str(J), p_str]
    for f in FLAVORS:
        if f in known_flavors:
            parts.append(f"--{f} {known_flavors[f]}")
        else:
            parts.append(f"--{f} ???")
    if extra:
        parts.append(extra)
    return " ".join(parts)
