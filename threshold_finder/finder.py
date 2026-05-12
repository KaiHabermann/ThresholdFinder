"""Core threshold finder logic."""
from __future__ import annotations

from typing import Optional

from .flavor import FlavorFilter
from .particles import load_hadrons, get_particle_pairs, get_particle_combinations, ParticleInfo
from .qn import can_produce, first_L_tuple
from .result import CombinationResult, ThresholdResult


class ThresholdFinder:
    """Find n-body hadronic thresholds that can produce given J^P quantum numbers.

    Parameters
    ----------
    mass_min, mass_max:
        Mass range in MeV to search thresholds within.
    J_target:
        Target total angular momentum (half-integer or integer).
    P_target:
        Target parity (+1 or -1).
    n_body:
        Number of particles in the final state (>= 2). Default is 2.
    max_L:
        Maximum orbital angular momentum to consider. None = unlimited
        (in practice capped at J_target + sum(Ji) + 5 to keep it finite).
    total_charge:
        If given, only combinations with this total charge are considered.
        Defaults to 0 (neutral resonances).
    flavor_filter:
        Optional FlavorFilter specifying required net quark numbers for the combination.
        Only flavors that are set (not None) are enforced. Combinations involving
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
        n_body: int = 2,
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
        if n_body < 2:
            raise ValueError("n_body must be >= 2")

        self.mass_min = mass_min
        self.mass_max = mass_max
        self.J_target = J_target
        self.P_target = P_target
        self.n_body = n_body
        self.max_L = max_L
        self.total_charge = total_charge
        self.flavor_filter = flavor_filter or FlavorFilter()
        self.status_filter = frozenset(status_filter)

    def _effective_max_L(self, spins: list[float]) -> int:
        if self.max_L is not None:
            return self.max_L
        return int(self.J_target + sum(spins)) + 4

    def _check_flavor(self, combo: tuple[ParticleInfo, ...]) -> bool:
        if self.flavor_filter.is_empty():
            return True
        # All particles must have defined quark content
        if any(p.quark_content is None for p in combo):
            return False
        combined: dict[str, int] = {}
        for p in combo:
            for f, v in p.quark_content.items():  # type: ignore[union-attr]
                combined[f] = combined.get(f, 0) + v
        from .flavor import FLAVORS
        for flavor in FLAVORS:
            target = getattr(self.flavor_filter, flavor)
            if target is None:
                continue
            if combined.get(flavor, 0) != target:
                return False
        return True

    def run(self) -> ThresholdResult:
        hadrons = load_hadrons(
            max_mass=self.mass_max,
            status_filter=self.status_filter,
        )

        combinations: list[CombinationResult] = []

        if self.n_body == 2:
            from .particles import get_particle_pairs
            pairs = get_particle_pairs(hadrons, total_charge=self.total_charge)
            for p1, p2, identical in pairs:
                threshold = p1.mass + p2.mass
                if threshold < self.mass_min or threshold > self.mass_max:
                    continue
                if not self.flavor_filter.check(p1.quark_content, p2.quark_content):
                    continue
                L_max = self._effective_max_L([p1.J, p2.J])
                both_bosons = (p1.J % 1.0 < 1e-9) and (p2.J % 1.0 < 1e-9)
                for L in range(0, L_max + 1):
                    if can_produce(
                        p1.J, p1.P, p2.J, p2.P,
                        self.J_target, self.P_target,
                        L, identical, both_bosons,
                    ):
                        combinations.append(CombinationResult(
                            particles=(p1.name, p2.name),
                            masses=(p1.mass, p2.mass),
                            charges=(p1.charge, p2.charge),
                            spins=(p1.J, p2.J),
                            parities=(p1.P, p2.P),
                            threshold=threshold,
                            L=L,
                            J_total=self.J_target,
                            P_total=self.P_target,
                        ))
        else:
            combos = get_particle_combinations(
                hadrons,
                n=self.n_body,
                total_charge=self.total_charge,
                mass_min=self.mass_min,
                mass_max=self.mass_max,
            )
            for combo in combos:
                threshold = sum(p.mass for p in combo)
                if threshold < self.mass_min or threshold > self.mass_max:
                    continue
                if not self._check_flavor(combo):
                    continue
                spins = [p.J for p in combo]
                parities_list = [p.P for p in combo]
                L_max = self._effective_max_L(spins)
                L_tuple = first_L_tuple(
                    spins, parities_list,
                    self.J_target, self.P_target,
                    L_max,
                )
                if L_tuple is not None:
                    combinations.append(CombinationResult(
                        particles=tuple(p.name for p in combo),
                        masses=tuple(p.mass for p in combo),
                        charges=tuple(p.charge for p in combo),
                        spins=tuple(spins),
                        parities=tuple(parities_list),
                        threshold=threshold,
                        L=L_tuple,
                        J_total=self.J_target,
                        P_total=self.P_target,
                    ))

        return ThresholdResult(
            J_target=self.J_target,
            P_target=self.P_target,
            mass_min=self.mass_min,
            mass_max=self.mass_max,
            max_L=self.max_L,
            n_body=self.n_body,
            flavor_filter=self.flavor_filter,
            combinations=combinations,
        )
