"""PDG meson data for ``gi-spectrum --match``, read with the official PDG Python API.

Thin loader layer: :func:`load_entries` walks the meson listings of the ``pdg`` package
(https://pdgapi.lbl.gov/doc/, offline SQLite) and returns :class:`gimodel.pdg_match.PDGEntry`
objects; everything else in this module is pure parsing (quantum-number strings, flavour
family from the PDG naming scheme) and is unit-tested without the database.

Imported only by :mod:`gimodel.cli` (when ``--match`` is given).

PDG API objects used: ``pdg.connect()`` -> ``PdgApi`` (``edition``, ``citation``,
``get_particles()``); each ``PdgParticleList`` is one PDG listing (``pdgid``,
``data_flags`` with ``'M'`` for mesons) and iterates over its ``PdgParticle`` charge states
(``name``, ``mcid``, ``charge``, ``quantum_I/J/P/C``, ``masses()``, ``widths()``,
``has_width_entry``, ``best(...)`` -> ``best_summary()`` with ``value``, ``error_*``,
``is_limit``, ``is_upper_limit``, ``is_lower_limit``, ``in_summary_table``).

The API has no quark content and no "established / further state" status per listing; see
:func:`family_from_name` and :func:`status_of` for how these are derived.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

from .pdg_match import FLAG_AMBIGUOUS, LIGHT_ISOSCALAR, PDGEntry

FLAG_UNCERTAIN = "uncertain J^P in PDG"
FLAG_MASS_RANGE = "PDG mass is a range; midpoint used"
FLAG_FAMILY_FROM_SECTION = "flavour family from PDG section (X state)"
FLAG_ISOSPIN_UNKNOWN = "isospin unknown"

# PDG naming scheme for mesons -> family. "light" = light unflavoured (q qbar / s sbar),
# "section" = X(...) states, whose family is that of the PDG section they are listed in,
# None = four-quark T states (never a q qbar partner).
_NAME_RULES: tuple[tuple[re.Pattern, object], ...] = (
    (re.compile(r"^(eta_c|J/psi|psi|chi_c|h_c)"), ("c", "c")),
    (re.compile(r"^(eta_b|Upsilon|chi_b|h_b)"), ("b", "b")),
    (re.compile(r"^D_s"), ("s", "c")),
    (re.compile(r"^D"), ("q", "c")),
    (re.compile(r"^B_s"), ("s", "b")),
    (re.compile(r"^B_c"), ("c", "b")),
    (re.compile(r"^B"), ("q", "b")),
    (re.compile(r"^K"), ("q", "s")),
    (re.compile(r"^(pi|rho|omega|phi|eta|f|a|b|h)(?![A-Za-z])"), "light"),
    (re.compile(r"^X\("), "section"),
    (re.compile(r"^T"), None),
)
_ANTI_PREFIXES = ("Dbar", "Kbar", "Bbar")


# --- quantum-number strings ---------------------------------------------------------------


@dataclass(frozen=True)
class Quantum:
    values: tuple            # parsed alternatives; empty = unknown
    uncertain: bool = False  # parenthesised or several alternatives in the PDG string


def _parse_one(s: str, kind: str):
    s = s.strip()
    if kind == "J":
        m = re.fullmatch(r"(\d+)(?:[+-]{1,2})?", s)  # "2++" in "2++ or 4" carries P C
        return int(m.group(1)) if m else None
    if kind == "I":
        try:
            num, _, den = s.partition("/")
            return int(num) / int(den or 1)
        except ValueError:
            return None
    return {"+": 1, "-": -1}.get(s)


def parse_quantum(text: Optional[str], kind: str) -> Quantum:
    """Parse a PDG quantum-number string (``kind`` in J, P, C, I).

    ``'1'`` -> certain; ``'(1)'`` or ``'2++ or 4'`` -> uncertain alternatives; ``'?'``,
    ``None`` or anything unparseable -> unknown (no values).
    """
    if text is None:
        return Quantum(())
    s = str(text).strip()
    if s in ("", "?", "None"):
        return Quantum(())
    uncertain = False
    if s.startswith("(") and s.endswith(")"):
        s, uncertain = s[1:-1], True
    parts = [p for p in re.split(r"\s+or\s+", s)]
    values = tuple(_parse_one(p, kind) for p in parts)
    if any(v is None for v in values):
        return Quantum(())
    return Quantum(values, uncertain or len(values) > 1)


def resolve(q: Quantum, include_uncertain: bool) -> tuple:
    """Values to use: certain values always; uncertain ones only with ``include_uncertain``."""
    if not q.values or (q.uncertain and not include_uncertain):
        return ()
    return q.values


# --- flavour family ---------------------------------------------------------------------------


def family_from_name(name: str):
    """Family from the PDG meson naming scheme: a flavour pair, "light", "section" or None."""
    for pattern, family in _NAME_RULES:
        if pattern.match(name):
            return family
    return None


def assign_families(names_in_order: Sequence[str]) -> list:
    """Families for listings in PDG order; X(...) states take the family of their section.

    PDG lists mesons by section (light unflavoured, strange, charmed, ..., c cbar, b bbar), so
    an X state belongs to the section of the nearest preceding listing with a definite name.
    """
    out, current = [], None
    for name in names_in_order:
        fam = family_from_name(name)
        if fam == "section":
            out.append(("section", current))
            continue
        if fam is not None:
            current = fam
        out.append(fam)
    return out


def flavour_pairs(family, isospin: Optional[float]) -> tuple[frozenset, tuple[str, ...]]:
    """Model-flavour pairs and flags for a family; light isoscalars fit q qbar and s sbar."""
    flags: tuple[str, ...] = ()
    if isinstance(family, tuple) and family and family[0] == "section":
        family = family[1]
        flags = (FLAG_FAMILY_FROM_SECTION,)
    if family is None or family == "section":
        return frozenset(), flags
    if family == "light":
        if isospin == 1:
            return frozenset({("q", "q")}), flags
        extra = (FLAG_ISOSPIN_UNKNOWN,) if isospin is None else ()
        return LIGHT_ISOSCALAR, (FLAG_AMBIGUOUS,) + extra + flags
    return frozenset({family}), flags


# --- status -----------------------------------------------------------------------------------


def status_of(mass_in_summary_table: bool) -> int:
    """threshold-finder status code for a PDG API listing.

    The PDG API marks every value that appears in the Summary Tables (``in_summary_table``)
    but has no per-listing status. Listings whose best mass is in the Summary Tables count as
    established (0); listings with a mass that is not, as omitted from the summary tables (2).
    No listing maps to 1 (evidence only): the API has no such category. The light-meson
    "Further States" (listing M300) carry no mass, J or P data and cannot be matched.
    """
    return 0 if mass_in_summary_table else 2


# --- loader -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    edition: str
    citation: str
    package_version: str

    @property
    def note(self) -> str:
        return f"PDG data: PDG Python API (pdg {self.package_version}), {self.edition} edition"


def source_info() -> Source:
    import pdg

    api = _api()
    return Source(str(api.edition), str(api.citation), getattr(pdg, "__version__", "?"))


@lru_cache(maxsize=1)
def _api():
    import pdg

    return pdg.connect()


def _best_summary(p, props):
    try:
        return p.best(props, p.name).best_summary()
    except Exception:  # PdgNoDataError, ambiguous best values, ...
        return None


def _value_GeV(summary) -> tuple[Optional[float], bool]:
    """(value in GeV, is_range). Ranges ("400 to 800") give their midpoint; limits give None."""
    if summary is None:
        return None, False
    if summary.is_limit:
        if summary.is_upper_limit or summary.is_lower_limit:
            return None, False
        return summary.get_value("GeV"), True
    return summary.get_value("GeV"), False


@dataclass(frozen=True)
class RawState:
    """One charge state as read from the API, before J/P/C resolution."""

    name: str
    listing: str
    mcid: Optional[int]
    charge: float
    mass: Optional[float]   # MeV
    mass_is_range: bool
    width: Optional[float]  # MeV
    I: Optional[str]
    J: Optional[str]
    P: Optional[str]
    C: Optional[str]
    in_summary_table: bool
    family: object


@lru_cache(maxsize=1)
def read_raw_states() -> tuple[RawState, ...]:
    """All meson charge states (particles, not antiparticles) of the installed PDG edition."""
    api = _api()
    listings = [pl for pl in api.get_particles() if "M" in (pl.data_flags or "")]
    families = assign_families([_clean_name(pl[0].name) if len(pl) else "" for pl in listings])
    out = []
    for pl, family in zip(listings, families):
        for p in pl:
            name = _clean_name(p.name)
            if (p.mcid is not None and p.mcid < 0) or p.charge < 0 or name.startswith(_ANTI_PREFIXES):
                continue
            summary = _best_summary(p, p.masses())
            mass, is_range = _value_GeV(summary)
            width = None
            if p.has_width_entry:
                width, _ = _value_GeV(_best_summary(p, p.widths()))
            out.append(RawState(
                name=name,
                listing=str(pl.pdgid).split("/")[0],
                mcid=p.mcid,
                charge=float(p.charge),
                mass=None if mass is None else 1000.0 * mass,
                mass_is_range=is_range,
                width=None if width is None else 1000.0 * width,
                I=p.quantum_I, J=p.quantum_J, P=p.quantum_P, C=p.quantum_C,
                in_summary_table=bool(summary is not None and summary.in_summary_table),
                family=family,
            ))
    return tuple(out)


def _clean_name(name: str) -> str:
    return name.replace("()", "")  # API names such as "D_s()+" / "B_c()+"


def entries_from_raw(raw: Sequence[RawState], include_uncertain: bool = False) -> list[PDGEntry]:
    """Resolve J, P, C, I and flavour; uncertain J/P are dropped unless include_uncertain.

    Several J alternatives ("2++ or 4") give one entry per alternative. Entries with unknown
    J or P keep J/P = None and are skipped by the matcher.
    """
    out = []
    for r in raw:
        if r.mass is None:
            continue
        Jq, Pq = parse_quantum(r.J, "J"), parse_quantum(r.P, "P")
        Js = resolve(Jq, include_uncertain) or (None,)
        Ps = resolve(Pq, include_uncertain) or (None,)
        Cs = resolve(parse_quantum(r.C, "C"), include_uncertain)
        Is = resolve(parse_quantum(r.I, "I"), include_uncertain)
        isospin = Is[0] if len(Is) == 1 else None
        pairs, flags = flavour_pairs(r.family, isospin)
        if Jq.uncertain or Pq.uncertain:
            flags = (f"{FLAG_UNCERTAIN} (J={r.J}, P={r.P})",) + flags
        if r.mass_is_range:
            flags = flags + (FLAG_MASS_RANGE,)
        for J in Js:
            for P in Ps:
                out.append(PDGEntry(
                    name=r.name, listing=r.listing, mcid=r.mcid, mass=r.mass, width=r.width,
                    J=J, P=P, C=Cs[0] if len(Cs) == 1 else None, isospin=isospin, charge=r.charge,
                    flavours=pairs, flags=flags, status=status_of(r.in_summary_table),
                ))
    return out


def load_entries(status: Sequence[int] = (0,), include_uncertain: bool = False) -> list[PDGEntry]:
    wanted = set(status)
    return [e for e in entries_from_raw(read_raw_states(), include_uncertain) if e.status in wanted]
