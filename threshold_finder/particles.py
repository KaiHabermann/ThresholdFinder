"""Particle loading and filtering from the PDG via the `particle` package."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from particle import Particle
from particle.pdgid import is_hadron


@dataclass(frozen=True)
class ParticleInfo:
    name: str
    mass: float       # MeV
    charge: float
    J: float
    P: int
    pdgid: int
    is_self_conjugate: bool

    @property
    def antiparticle_pdgid(self) -> int:
        return -self.pdgid


@lru_cache(maxsize=1)
def load_hadrons(
    max_mass: float,
    status_filter: frozenset[int] = frozenset({0}),
) -> list[ParticleInfo]:
    """Return hadrons with known mass, J, P below max_mass with given status codes.

    Status codes (PDG):
        0 = established (R in PDG tables)
        1 = evidence, but not confirmed
        2 = omitted from summary tables
    """
    result = []
    seen_pairs: set[tuple[int, int]] = set()  # avoid double-counting p/pbar etc.

    for p in Particle.findall():
        if p.mass is None or p.J is None or p.P is None:
            continue
        if not is_hadron(p.pdgid):
            continue
        if p.mass > max_mass:
            continue
        if int(p.status) not in status_filter:
            continue

        pdgid = int(p.pdgid)
        anti_id = -pdgid

        # Avoid adding both particle and antiparticle as separate entries
        # when they are distinct; we track pairs.
        pair = (min(pdgid, anti_id), max(pdgid, anti_id))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        is_self_conj = (pdgid == anti_id) or (p.invert().pdgid == p.pdgid)

        result.append(ParticleInfo(
            name=p.name,
            mass=float(p.mass),
            charge=float(p.charge),
            J=float(p.J),
            P=int(p.P),
            pdgid=pdgid,
            is_self_conjugate=is_self_conj,
        ))

    return result


def get_particle_pairs(
    particles: list[ParticleInfo],
    total_charge: Optional[float] = None,
) -> list[tuple[ParticleInfo, ParticleInfo, bool]]:
    """Generate all unordered pairs (p1, p2, identical).

    For non-self-conjugate particles we also include (p, pbar) pairs.
    If total_charge is given, only pairs with that combined charge are returned.
    """
    # Build full list including antiparticles
    full: list[ParticleInfo] = []
    for p in particles:
        full.append(p)
        if not p.is_self_conjugate:
            # Create antiparticle entry
            try:
                anti = Particle.from_pdgid(-p.pdgid)
                if anti.mass is not None:
                    full.append(ParticleInfo(
                        name=anti.name,
                        mass=float(anti.mass),
                        charge=float(anti.charge),
                        J=float(anti.J),
                        P=int(anti.P),
                        pdgid=int(anti.pdgid),
                        is_self_conjugate=False,
                    ))
            except Exception:
                pass

    # Deduplicate by pdgid
    seen: dict[int, ParticleInfo] = {}
    for p in full:
        seen[p.pdgid] = p
    full = list(seen.values())

    pairs = []
    for i, p1 in enumerate(full):
        for j, p2 in enumerate(full):
            if j < i:
                continue
            q_sum = p1.charge + p2.charge
            if total_charge is not None and abs(q_sum - total_charge) > 1e-9:
                continue
            identical = (p1.pdgid == p2.pdgid)
            pairs.append((p1, p2, identical))

    return pairs
