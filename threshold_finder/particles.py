"""Particle loading and filtering from the PDG via the `particle` package."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Optional

from particle import Particle
from particle.pdgid import is_hadron

from .flavor import parse_quark_content


@dataclass(frozen=True)
class ParticleInfo:
    name: str
    mass: float       # MeV
    charge: float
    J: float
    P: int
    pdgid: int
    is_self_conjugate: bool
    quark_content: Optional[dict[str, int]]  # net quark numbers; None for mixed states

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
        qc = parse_quark_content(p.quarks) if p.quarks else None

        result.append(ParticleInfo(
            name=p.name,
            mass=float(p.mass),
            charge=float(p.charge),
            J=float(p.J),
            P=int(p.P),
            pdgid=pdgid,
            is_self_conjugate=is_self_conj,
            quark_content=qc,
        ))

    return result


def _build_full_list(particles: list[ParticleInfo]) -> list[ParticleInfo]:
    """Expand particle list to include antiparticles, deduplicated by pdgid."""
    full: list[ParticleInfo] = []
    for p in particles:
        full.append(p)
        if not p.is_self_conjugate:
            try:
                anti = Particle.from_pdgid(-p.pdgid)
                if anti.mass is not None:
                    anti_qc = parse_quark_content(anti.quarks) if anti.quarks else None
                    full.append(ParticleInfo(
                        name=anti.name,
                        mass=float(anti.mass),
                        charge=float(anti.charge),
                        J=float(anti.J),
                        P=int(anti.P),
                        pdgid=int(anti.pdgid),
                        is_self_conjugate=False,
                        quark_content=anti_qc,
                    ))
            except Exception:
                pass
    seen: dict[int, ParticleInfo] = {}
    for p in full:
        seen[p.pdgid] = p
    return list(seen.values())


def get_particle_pairs(
    particles: list[ParticleInfo],
    total_charge: Optional[float] = None,
) -> list[tuple[ParticleInfo, ParticleInfo, bool]]:
    """Generate all unordered pairs (p1, p2, identical).

    For non-self-conjugate particles we also include (p, pbar) pairs.
    If total_charge is given, only pairs with that combined charge are returned.
    """
    full = _build_full_list(particles)
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


def get_particle_combinations(
    particles: list[ParticleInfo],
    n: int,
    total_charge: Optional[float] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
) -> list[tuple[ParticleInfo, ...]]:
    """Generate all unordered n-tuples of particles (with repetition allowed).

    For non-self-conjugate particles antiparticles are included.
    Charge and mass pruning are applied during generation so invalid tuples
    are never fully constructed.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    full = _build_full_list(particles)
    # Sort by mass ascending so running mass bounds are tight
    full_sorted = sorted(full, key=lambda p: p.mass)
    masses = [p.mass for p in full_sorted]
    charges = [p.charge for p in full_sorted]
    result: list[tuple[ParticleInfo, ...]] = []
    _combine(
        full_sorted, masses, charges, n,
        total_charge, mass_min, mass_max,
        0, [], 0.0, 0.0, result,
    )
    return result


def _combine(
    pool: list[ParticleInfo],
    masses: list[float],
    charges: list[float],
    n: int,
    total_charge: Optional[float],
    mass_min: Optional[float],
    mass_max: Optional[float],
    start: int,
    current: list[ParticleInfo],
    mass_so_far: float,
    charge_so_far: float,
    result: list[tuple[ParticleInfo, ...]],
) -> None:
    remaining = n - len(current)

    if remaining == 0:
        if total_charge is not None and abs(charge_so_far - total_charge) > 1e-9:
            return
        if mass_min is not None and mass_so_far < mass_min - 1e-9:
            return
        if mass_max is not None and mass_so_far > mass_max + 1e-9:
            return
        result.append(tuple(current))
        return

    # Mass pruning: pool sorted by mass ascending, so masses[start] is the minimum
    # mass any remaining pick can contribute.
    if mass_max is not None and mass_so_far > mass_max + 1e-9:
        return
    m_min_per = masses[start] if start < len(masses) else 0.0
    m_max_per = masses[-1] if masses else 0.0
    mass_lo = mass_so_far + remaining * m_min_per
    mass_hi = mass_so_far + remaining * m_max_per
    if mass_max is not None and mass_lo > mass_max + 1e-9:
        return
    if mass_min is not None and mass_hi < mass_min - 1e-9:
        return

    # Charge pruning: charges are not monotone with the mass-sorted pool,
    # so we can't use tight bounds here — but we can still prune at the leaf.
    # (Charge-sorted pruning would require a different pool ordering.)

    for i in range(start, len(pool)):
        p = pool[i]
        current.append(p)
        _combine(
            pool, masses, charges, n,
            total_charge, mass_min, mass_max,
            i, current,
            mass_so_far + p.mass, charge_so_far + p.charge,
            result,
        )
        current.pop()
