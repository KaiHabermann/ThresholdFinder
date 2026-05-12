from dataclasses import dataclass, field
from typing import Optional, Union

from .flavor import FlavorFilter


def _fmt_J(j: float) -> str:
    """Format J as '1', '2', '1/2', '3/2', etc."""
    twice = round(j * 2)
    if twice % 2 == 0:
        return str(twice // 2)
    return f"{twice}/2"


def _fmt_L(L: Union[int, tuple[int, ...]]) -> str:
    if isinstance(L, int):
        return str(L)
    return "[" + ", ".join(str(l) for l in L) + "]"


@dataclass(frozen=True)
class CombinationResult:
    particles: tuple[str, ...]
    masses: tuple[float, ...]
    charges: tuple[float, ...]
    spins: tuple[float, ...]
    parities: tuple[int, ...]
    threshold: float
    L: Union[int, tuple[int, ...]]   # int for 2-body, tuple for n-body
    J_total: float
    P_total: int

    # two-body convenience aliases
    @property
    def particle1(self) -> str:
        return self.particles[0]

    @property
    def particle2(self) -> str:
        return self.particles[1]

    @property
    def mass1(self) -> float:
        return self.masses[0]

    @property
    def mass2(self) -> float:
        return self.masses[1]

    @property
    def charge1(self) -> float:
        return self.charges[0]

    @property
    def charge2(self) -> float:
        return self.charges[1]

    @property
    def J1(self) -> float:
        return self.spins[0]

    @property
    def J2(self) -> float:
        return self.spins[1]

    @property
    def P1(self) -> int:
        return self.parities[0]

    @property
    def P2(self) -> int:
        return self.parities[1]

    @property
    def identical(self) -> bool:
        return len(set(self.particles)) == 1

    @property
    def total_charge(self) -> float:
        return sum(self.charges)

    def __str__(self) -> str:
        state = " + ".join(self.particles)
        return (
            f"{state}  "
            f"threshold={self.threshold:.1f} MeV  "
            f"L={_fmt_L(self.L)}  "
            f"J^P={_fmt_J(self.J_total)}^{'+' if self.P_total > 0 else '-'}"
        )


@dataclass
class ThresholdResult:
    J_target: float
    P_target: int
    mass_min: float
    mass_max: float
    max_L: Optional[int]
    n_body: int = 2
    flavor_filter: FlavorFilter = field(default_factory=FlavorFilter)
    combinations: list[CombinationResult] = field(default_factory=list)

    def __str__(self) -> str:
        flavor_str = f"  flavor: {self.flavor_filter}" if not self.flavor_filter.is_empty() else ""
        n_str = f"  ({self.n_body}-body)" if self.n_body != 2 else ""
        lines = [
            f"Thresholds for J^P = {_fmt_J(self.J_target)}^{'+' if self.P_target > 0 else '-'}  "
            f"in [{self.mass_min:.1f}, {self.mass_max:.1f}] MeV"
            f"  (max L = {'∞' if self.max_L is None else self.max_L})"
            f"{n_str}{flavor_str}",
            f"Found {len(self.combinations)} combination(s):",
        ]
        for c in sorted(self.combinations, key=lambda x: (x.threshold, str(x.L))):
            lines.append(f"  {c}")
        return "\n".join(lines)
