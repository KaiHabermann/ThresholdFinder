"""Core threshold finder logic."""
from __future__ import annotations

from typing import Optional

from .flavor import FlavorFilter
from .particles import load_hadrons, get_particle_pairs, ParticleInfo
from .qn import can_produce, parity, j_range
from .result import CombinationResult, ThresholdResult


class ThresholdFinder:
    """Find two-body hadronic thresholds that can produce given J^P quantum numbers.

    Parameters
    ----------
    mass_min, mass_max:
        Mass range in MeV to search thresholds within.
    J_target:
        Target total angular momentum (half-integer or integer).
    P_target:
        Target parity (+1 or -1).
    max_L:
        Maximum orbital angular momentum to consider. None = unlimited
        (in practice capped at J_target + J1 + J2 + 5 to keep it finite).
    total_charge:
        If given, only pairs with this total charge are considered.
        Defaults to 0 (neutral resonances).
    flavor_filter:
        Optional FlavorFilter specifying required net quark numbers for the pair.
        Only flavors that are set (not None) are enforced. Pairs involving
        particles with undefined quark content (mixed states) are excluded
        when any flavor constraint is active.
    status_filter:
        PDG status codes to include. 0 = well-established only (default).
    """

    def __init__(
        self,
        mass_min: float,
        mass_max: float,
        J_target: float,
        P_target: int,
        max_L: Optional[int] = None,
        total_charge: float = 0.0,
        flavor_filter: Optional[FlavorFilter] = None,
        status_filter: tuple[int, ...] = (0,),
    ):
        if P_target not in (1, -1):
            raise ValueError("P_target must be +1 or -1")
        if J_target < 0:
            raise ValueError("J_target must be >= 0")
        if mass_min >= mass_max:
            raise ValueError("mass_min must be less than mass_max")

        self.mass_min = mass_min
        self.mass_max = mass_max
        self.J_target = J_target
        self.P_target = P_target
        self.max_L = max_L
        self.total_charge = total_charge
        self.flavor_filter = flavor_filter or FlavorFilter()
        self.status_filter = frozenset(status_filter)

    def _effective_max_L(self, j1: float, j2: float) -> int:
        if self.max_L is not None:
            return self.max_L
        # Sensible cap: to couple j1+j2 to J_target we need L <= J_target + j1 + j2
        return int(self.J_target + j1 + j2) + 4

    def run(self) -> ThresholdResult:
        hadrons = load_hadrons(
            max_mass=self.mass_max,
            status_filter=self.status_filter,
        )

        pairs = get_particle_pairs(hadrons, total_charge=self.total_charge)

        combinations: list[CombinationResult] = []

        for p1, p2, identical in pairs:
            threshold = p1.mass + p2.mass
            if threshold < self.mass_min or threshold > self.mass_max:
                continue
            if not self.flavor_filter.check(p1.quark_content, p2.quark_content):
                continue

            L_max = self._effective_max_L(p1.J, p2.J)
            both_bosons = (p1.J % 1.0 < 1e-9) and (p2.J % 1.0 < 1e-9)

            for L in range(0, L_max + 1):
                if can_produce(
                    p1.J, p1.P, p2.J, p2.P,
                    self.J_target, self.P_target,
                    L, identical, both_bosons,
                ):
                    combinations.append(CombinationResult(
                        particle1=p1.name,
                        particle2=p2.name,
                        mass1=p1.mass,
                        mass2=p2.mass,
                        threshold=threshold,
                        charge1=p1.charge,
                        charge2=p2.charge,
                        J1=p1.J,
                        J2=p2.J,
                        P1=p1.P,
                        P2=p2.P,
                        L=L,
                        J_total=self.J_target,
                        P_total=self.P_target,
                        identical=identical,
                    ))

        return ThresholdResult(
            J_target=self.J_target,
            P_target=self.P_target,
            mass_min=self.mass_min,
            mass_max=self.mass_max,
            max_L=self.max_L,
            flavor_filter=self.flavor_filter,
            combinations=combinations,
        )
