"""Constituent masses, quark types and meson channels (port of ``model_objects.jl``
(masses), ``quark.jl`` and ``meson.jl``).

In the GI model the medium is flavor-blind: the constituent mass is the only
dynamical input. Flavor identity carries electric charge and decides whether a
channel is self-conjugate (:func:`is_equal_flavor`), which gates the same-J
antisymmetric spin-orbit mixing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

__all__ = [
    "ConstituentMasses",
    "reduced_mass",
    "LightQuark",
    "StrangeQuark",
    "HeavyQuark",
    "charge",
    "flavor_symbol",
    "mass_GeV",
    "Meson",
    "is_equal_flavor",
    "flavor_label",
    "canonical_flavor",
]


@dataclass(frozen=True)
class ConstituentMasses:
    """Constituent quark masses in GeV for one radial channel."""

    m1_GeV: float
    m2_GeV: float

    def __post_init__(self) -> None:
        for m in (self.m1_GeV, self.m2_GeV):
            if not (math.isfinite(m) and m > 0):
                raise ValueError(
                    f"ConstituentMasses: masses must be positive and finite, "
                    f"got ({self.m1_GeV}, {self.m2_GeV}) GeV"
                )
        object.__setattr__(self, "m1_GeV", float(self.m1_GeV))
        object.__setattr__(self, "m2_GeV", float(self.m2_GeV))

    @property
    def reduced_mass(self) -> float:
        return self.m1_GeV * self.m2_GeV / (self.m1_GeV + self.m2_GeV)


# --- quarks -----------------------------------------------------------------


def _check_mass(mass: float) -> float:
    if not mass > 0:
        raise ValueError(f"quark mass must be positive, got {mass}")
    return float(mass)


@dataclass(frozen=True)
class LightQuark:
    """An up/down constituent quark (``m_u = m_d``); charge is deliberately undefined."""

    mass_GeV: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mass_GeV", _check_mass(self.mass_GeV))

    @property
    def flavor(self) -> str:
        return "q"


@dataclass(frozen=True)
class StrangeQuark:
    """The strange constituent quark (charge -1/3)."""

    mass_GeV: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mass_GeV", _check_mass(self.mass_GeV))

    @property
    def flavor(self) -> str:
        return "s"


@dataclass(frozen=True)
class HeavyQuark:
    """A heavy quark; ``isospin`` is ``"up"`` (charge +2/3) or ``"down"`` (-1/3)."""

    mass_GeV: float
    name: str
    isospin: str

    def __post_init__(self) -> None:
        if self.isospin not in ("up", "down"):
            raise ValueError(
                f"HeavyQuark tag must be 'up' or 'down' (weak-isospin class), got `{self.isospin}`"
            )
        object.__setattr__(self, "mass_GeV", _check_mass(self.mass_GeV))

    @property
    def flavor(self) -> str:
        return self.name


Quark = LightQuark | StrangeQuark | HeavyQuark


def mass_GeV(q: Quark) -> float:
    """Constituent mass in GeV."""
    return q.mass_GeV


def flavor_symbol(q: Quark) -> str:
    """Flavor label stored by a :class:`Meson` (``q`` for light, ``s``, or the heavy name)."""
    return q.flavor


def charge(q: Quark) -> Fraction:
    """Electric charge in units of e. Raises ``TypeError`` for :class:`LightQuark`."""
    if isinstance(q, StrangeQuark):
        return Fraction(-1, 3)
    if isinstance(q, HeavyQuark):
        return Fraction(2, 3) if q.isospin == "up" else Fraction(-1, 3)
    raise TypeError("charge is undefined for LightQuark: the model has no isospin resolution")


# --- mesons -----------------------------------------------------------------

_MESON_FLAVOR_KEYS = {"u": "u", "d": "d", "s": "s", "c": "c", "b": "b", "q": "q"}


def canonical_flavor(flavor: str) -> str:
    """``n`` is an alias for the light average ``q``."""
    flavor = str(flavor)
    return "q" if flavor == "n" else flavor


def _flavor_mass(quark_masses: Mapping[str, float], flavor: str) -> float:
    key = _MESON_FLAVOR_KEYS.get(flavor)
    if key is None:
        raise ValueError(
            f"unknown quark flavor `{flavor}`; expected one of {sorted(_MESON_FLAVOR_KEYS)} "
            "(or the alias n for q)"
        )
    if key not in quark_masses:
        raise ValueError(f"quark mass table has no entry for flavor `{flavor}` (key `{key}`)")
    return float(quark_masses[key])


@dataclass(frozen=True)
class Meson:
    """A ``q1 q2bar`` channel with constituent masses.

    Build from a mass table with :meth:`from_table` (``Meson.from_table(mq, "c", "u")``),
    from two quarks with :meth:`from_quarks`, or directly from explicit masses.
    """

    flavor1: str
    flavor2: str
    constituent_masses: ConstituentMasses

    def __post_init__(self) -> None:
        object.__setattr__(self, "flavor1", canonical_flavor(self.flavor1))
        object.__setattr__(self, "flavor2", canonical_flavor(self.flavor2))

    @classmethod
    def from_table(cls, quark_masses: Mapping[str, float], flavor1: str, flavor2: str) -> Meson:
        f1, f2 = canonical_flavor(flavor1), canonical_flavor(flavor2)
        masses = ConstituentMasses(_flavor_mass(quark_masses, f1), _flavor_mass(quark_masses, f2))
        return cls(f1, f2, masses)

    @classmethod
    def from_quarks(cls, q1: Quark, q2: Quark) -> Meson:
        return cls(flavor_symbol(q1), flavor_symbol(q2), ConstituentMasses(q1.mass_GeV, q2.mass_GeV))

    @property
    def flavors(self) -> tuple[str, str]:
        return (self.flavor1, self.flavor2)

    def __str__(self) -> str:
        m = self.constituent_masses
        return (
            f"Meson: {self.flavor1} {self.flavor2}bar   (m1 = {m.m1_GeV}, m2 = {m.m2_GeV} GeV, "
            f"reduced = {m.reduced_mass:.4f} GeV)"
        )


def is_equal_flavor(meson: Meson) -> bool:
    """``True`` for self-conjugate flavor content (``c cbar``, ``q qbar``, ...)."""
    return meson.flavor1 == meson.flavor2


def flavor_label(meson: Meson) -> str:
    """Compact flavor-pair label, e.g. ``"cb"``."""
    return f"{meson.flavor1}{meson.flavor2}"


def reduced_mass(obj: ConstituentMasses | Meson) -> float:
    """Reduced mass ``m1 m2 / (m1 + m2)`` in GeV."""
    masses = obj.constituent_masses if isinstance(obj, Meson) else obj
    return masses.reduced_mass
