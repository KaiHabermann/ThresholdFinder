from dataclasses import dataclass, field
from typing import Optional

from .flavor import FlavorFilter


@dataclass(frozen=True)
class CombinationResult:
    particle1: str
    particle2: str
    mass1: float          # MeV
    mass2: float          # MeV
    threshold: float      # MeV = mass1 + mass2
    charge1: float
    charge2: float
    J1: float
    J2: float
    P1: int
    P2: int
    L: int                # orbital angular momentum
    J_total: float        # total spin achieved
    P_total: int          # resulting parity
    identical: bool       # whether the two particles are identical

    @property
    def total_charge(self) -> float:
        return self.charge1 + self.charge2

    def __str__(self) -> str:
        same = " [identical]" if self.identical else ""
        return (
            f"{self.particle1} + {self.particle2}{same}  "
            f"threshold={self.threshold:.1f} MeV  "
            f"L={self.L}  "
            f"J^P={self.J_total:.0f}^{'+' if self.P_total > 0 else '-'}"
        )


@dataclass
class ThresholdResult:
    J_target: float
    P_target: int
    mass_min: float       # MeV
    mass_max: float       # MeV
    max_L: Optional[int]
    flavor_filter: FlavorFilter = field(default_factory=FlavorFilter)
    combinations: list[CombinationResult] = field(default_factory=list)

    def __str__(self) -> str:
        flavor_str = f"  flavor: {self.flavor_filter}" if not self.flavor_filter.is_empty() else ""
        lines = [
            f"Thresholds for J^P = {self.J_target:.0f}^{'+' if self.P_target > 0 else '-'}  "
            f"in [{self.mass_min:.1f}, {self.mass_max:.1f}] MeV"
            f"  (max L = {'∞' if self.max_L is None else self.max_L})"
            f"{flavor_str}",
            f"Found {len(self.combinations)} combination(s):",
        ]
        for c in sorted(self.combinations, key=lambda x: (x.threshold, x.L)):
            lines.append(f"  {c}")
        return "\n".join(lines)
