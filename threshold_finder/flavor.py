"""Flavor quantum number definitions and conservation check."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

FLAVORS = ("u", "d", "s", "c", "b")


def parse_quark_content(quarks_str: str) -> Optional[dict[str, int]]:
    """Parse the PDG quarks string into net quark numbers.

    Lowercase letter = quark (+1), uppercase = antiquark (-1).
    Returns None for superposition / mixed states (e.g. '(uU-dD)/sqrt(2)').
    """
    s = quarks_str.strip()
    if any(ch in s for ch in ("(", "/", "+", "-", "x", "y")):
        return None
    counts: dict[str, int] = {}
    for ch in s:
        if ch.isalpha():
            flavor = ch.lower()
            counts[flavor] = counts.get(flavor, 0) + (1 if ch.islower() else -1)
    return counts


@dataclass(frozen=True)
class FlavorFilter:
    """Specifies required net quark numbers for the two-particle system.

    Each field is Optional[int]. If None, that flavor is unconstrained.
    Net quark number = #quark - #antiquark for each flavor.
    """
    u: Optional[int] = None
    d: Optional[int] = None
    s: Optional[int] = None
    c: Optional[int] = None
    b: Optional[int] = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in FLAVORS)

    def check(
        self,
        qc1: Optional[dict[str, int]],
        qc2: Optional[dict[str, int]],
    ) -> bool:
        """Return True if the combined quark content of two particles satisfies all constraints.

        If either particle has undefined quark content (mixed state), the pair is
        excluded whenever any flavor constraint is active.
        """
        if self.is_empty():
            return True

        if qc1 is None or qc2 is None:
            return False  # can't verify conservation for mixed states

        for flavor in FLAVORS:
            target = getattr(self, flavor)
            if target is None:
                continue
            combined = qc1.get(flavor, 0) + qc2.get(flavor, 0)
            if combined != target:
                return False
        return True

    def __str__(self) -> str:
        parts = []
        for f in FLAVORS:
            v = getattr(self, f)
            if v is not None:
                parts.append(f"{f}={v:+d}")
        return ", ".join(parts) if parts else "unconstrained"
