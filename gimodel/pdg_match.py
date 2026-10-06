"""Pair Godfrey–Isgur predictions with existing PDG mesons (``gi-spectrum --match``).

Imported only by :mod:`gimodel.cli`; the gimodel core never imports this module or any
PDG data package.

This module is data-source agnostic. A loader (:mod:`gimodel.pdg_data`, built on the
official PDG Python API) turns every PDG charge state into a :class:`PDGEntry` that already
carries its compatible model-flavour pairs; :func:`collapse_multiplets` merges isospin/charge
multiplets, and the pairing logic (:func:`find_candidates`, :func:`assign`,
:func:`unassigned_states`) works on plain :class:`Prediction` / :class:`PDGState`
dataclasses, so it can be tested with synthetic candidates.

Masses and widths are in MeV. Δ = m_pred − m_PDG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Model flavours: u and d are the isospin-averaged light quark q.
_MODEL = {"u": "q", "d": "q", "q": "q", "s": "s", "c": "c", "b": "b"}
_ORDER = "qscbt"
_PDG_DIGIT_FLAVOUR = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t"}

# K(S)0 / K(L)0 are weak eigenstates of the K0 / K0bar system already represented by K0.
WEAK_EIGENSTATES = frozenset({130, 310})

FLAG_AMBIGUOUS = "flavour-ambiguous"
FLAG_EXOTIC = "exotic J^PC (not q qbar)"

BIG_COST = 1e9


def _pair(f1: str, f2: str) -> tuple[str, str]:
    """Canonical (unordered) model-flavour pair: charge conjugates compare equal."""
    a, b = _MODEL[f1], _MODEL[f2]
    return (a, b) if _ORDER.index(a) <= _ORDER.index(b) else (b, a)


LIGHT_ISOSCALAR = frozenset({("q", "q"), ("s", "s")})


# --- data -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PDGEntry:
    """One PDG charge state, as produced by a loader (antiparticles are not listed)."""

    name: str
    listing: str                    # PDG listing identifier (charge states of one listing share it)
    mcid: Optional[int]             # Monte-Carlo particle ID, if the PDG assigns one
    mass: float                     # MeV
    width: Optional[float]          # MeV
    J: Optional[int]
    P: Optional[int]
    C: Optional[int]
    isospin: Optional[float]
    charge: float
    flavours: frozenset[tuple[str, str]]  # canonical model-flavour pairs (see SystemFlavour)
    flags: tuple[str, ...] = ()
    status: int = 0


@dataclass(frozen=True)
class PDGState:
    """A PDG meson after isospin/charge-multiplet collapse."""

    name: str                       # representative name (charge suffix dropped for multiplets)
    mass: float                     # MeV; mean over the multiplet members
    width: Optional[float]          # MeV; mean over the members with a known width
    J: int
    P: int
    C: Optional[int]
    isospin: Optional[float]
    flavours: frozenset[tuple[str, str]]  # canonical model-flavour pairs the state can be
    flags: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    pdgid: str = ""  # PDG listing identifier of the representative
    status: int = 0

    def __post_init__(self):
        if not self.members:
            object.__setattr__(self, "members", (self.name,))


@dataclass(frozen=True)
class Prediction:
    label: str
    J: int
    P: int
    C: Optional[int]
    mass: float  # MeV


@dataclass(frozen=True)
class MatchSettings:
    mass_tol: float = 50.0
    width_frac: float = 0.5
    status: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class SystemFlavour:
    """The model system as seen by the matcher."""

    pair: tuple[str, str]
    isovector_only: bool = False  # u dbar / d ubar: only I = 1 states

    @classmethod
    def from_quarks(cls, quark: str, antiquark: str) -> "SystemFlavour":
        return cls(_pair(quark, antiquark), {quark, antiquark} == {"u", "d"})

    @property
    def self_conjugate(self) -> bool:
        return self.pair[0] == self.pair[1]


@dataclass(frozen=True)
class Candidate:
    state: PDGState
    delta: float          # m_pred - m_PDG
    tolerance: float
    tolerance_source: str  # "mass_tol" or "width"
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnassignedState:
    state: PDGState
    tolerance: float
    tolerance_source: str
    flags: tuple[str, ...] = ()


@dataclass
class MatchResult:
    candidates: list[list[Candidate]]
    assigned: list[Optional[Candidate]]
    unassigned: list[UnassignedState] = field(default_factory=list)


# --- flavour classification -------------------------------------------------------------------


def _pdgid_quark_digits(pdgid: int) -> Optional[tuple[str, str]]:
    """Quark flavours from a standard meson PDG ID (digits n_q2, n_q3), else None."""
    a = abs(pdgid)
    nq1, nq2, nq3 = (a // 1000) % 10, (a // 100) % 10, (a // 10) % 10
    if nq1 != 0 or nq2 not in _PDG_DIGIT_FLAVOUR or nq3 not in _PDG_DIGIT_FLAVOUR:
        return None
    return _PDG_DIGIT_FLAVOUR[nq2], _PDG_DIGIT_FLAVOUR[nq3]


# --- multiplet collapse -----------------------------------------------------------------------

_CHARGE_SUFFIX = re.compile(r"(~?0|\+\+|--|\+|-)$")


def _isospin_family_id(pdgid: int) -> int:
    """PDG ID with light quark digits u (2) mapped to d (1): isospin partners coincide."""
    a = abs(pdgid)
    if _pdgid_quark_digits(a) is None:
        return a
    nq2, nq3 = (a // 100) % 10, (a // 10) % 10
    nq2, nq3 = (1 if nq2 == 2 else nq2), (1 if nq3 == 2 else nq3)
    return a - ((a // 100) % 10) * 100 - ((a // 10) % 10) * 10 + nq2 * 100 + nq3 * 10


def multiplet_key(entry: PDGEntry) -> tuple:
    """Isospin partners share the key: same PDG listing, or MC IDs equal up to u <-> d digits."""
    family = ("mcid", _isospin_family_id(entry.mcid)) if entry.mcid else ("listing", entry.listing)
    return (family, entry.isospin, entry.J, entry.P)


def base_name(name: str) -> str:
    """PDG name without its charge suffix: ``D*(2010)+`` -> ``D*(2010)``, ``K0`` -> ``K``."""
    stripped = _CHARGE_SUFFIX.sub("", name)
    return stripped or name


def collapse_multiplets(entries: Iterable[PDGEntry]) -> list[PDGState]:
    """Group charge states of one isospin multiplet into one :class:`PDGState`.

    Members share a PDG listing or, across listings (D*(2007)0 / D*(2010)+, D0 / D+), the
    isospin family of their MC ID (u <-> d digits), plus isospin, J and P. Entries with
    unknown J or P, K(S)0/K(L)0 and entries without a flavour assignment are skipped. The
    representative is the neutral member (else the lowest MC ID); for multiplets its name
    loses the charge suffix. Mass and width are the means over the members (the model uses
    isospin-averaged light quarks).
    """
    groups: dict[tuple, list[PDGEntry]] = {}
    for e in entries:
        if e.J is None or e.P not in (1, -1) or not e.flavours:
            continue
        if e.mcid is not None and abs(e.mcid) in WEAK_EIGENSTATES:
            continue
        groups.setdefault(multiplet_key(e), []).append(e)

    states: list[PDGState] = []
    for members in groups.values():
        members.sort(key=lambda e: (abs(e.charge) > 0, abs(e.mcid or 0), e.name))
        rep = members[0]
        pairs: set[tuple[str, str]] = set()
        flags: list[str] = []
        for m in members:
            pairs |= m.flavours
            flags += [f for f in m.flags if f not in flags]
        widths = [m.width for m in members if m.width is not None]
        C = next((m.C for m in members if m.C in (1, -1)), None)
        states.append(PDGState(
            name=base_name(rep.name) if len(members) > 1 else rep.name,
            mass=sum(m.mass for m in members) / len(members),
            width=sum(widths) / len(widths) if widths else None,
            J=rep.J,
            P=rep.P,
            C=C,
            isospin=rep.isospin,
            flavours=frozenset(pairs),
            flags=tuple(flags),
            members=tuple(m.name for m in members),
            pdgid=rep.listing,
            status=min(m.status for m in members),
        ))
    states.sort(key=lambda s: (s.mass, s.name))
    return states


def effective_tolerance(state: PDGState, settings: MatchSettings) -> tuple[float, str]:
    """max(mass_tol, width_frac * Γ); mass_tol alone when Γ is unknown."""
    if state.width is not None and settings.width_frac * state.width > settings.mass_tol:
        return settings.width_frac * state.width, "width"
    return settings.mass_tol, "mass_tol"


def quantum_numbers_match(J: int, P: int, C: Optional[int], state: PDGState) -> bool:
    """J and P equal; C equal when both are defined."""
    if state.J != J or state.P != P:
        return False
    return C is None or state.C is None or state.C == C


def flavour_match(state: PDGState, system: SystemFlavour) -> bool:
    if system.pair not in state.flavours:
        return False
    return not system.isovector_only or state.isospin == 1


def find_candidates(
    pred: Prediction, states: Iterable[PDGState], system: SystemFlavour, settings: MatchSettings
) -> list[Candidate]:
    """All compatible PDG states within tolerance of one prediction, ranked by |Δm|."""
    out = []
    for s in states:
        if not quantum_numbers_match(pred.J, pred.P, pred.C, s) or not flavour_match(s, system):
            continue
        tol, source = effective_tolerance(s, settings)
        delta = pred.mass - s.mass
        if abs(delta) <= tol:
            out.append(Candidate(s, delta, tol, source, s.flags))
    out.sort(key=lambda c: (abs(c.delta), c.state.name))
    return out


def assign(predictions: Sequence[Prediction], candidates: Sequence[Sequence[Candidate]]) -> list[Optional[Candidate]]:
    """Best one-to-one assignment within each J^P(C) group (minimal total |Δm|).

    Pairs outside tolerance get cost BIG_COST, so the solver first maximises the number of
    matched pairs and then minimises the total |Δm|; BIG_COST pairs are dropped afterwards.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    result: list[Optional[Candidate]] = [None] * len(predictions)
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(predictions):
        groups.setdefault((p.J, p.P, p.C), []).append(i)
    for idx in groups.values():
        states: list[PDGState] = []
        for i in idx:
            for c in candidates[i]:
                if c.state not in states:
                    states.append(c.state)
        if not states:
            continue
        cost = np.full((len(idx), len(states)), BIG_COST)
        lookup: dict[tuple[int, int], Candidate] = {}
        for r, i in enumerate(idx):
            for c in candidates[i]:
                k = states.index(c.state)
                cost[r, k] = abs(c.delta)
                lookup[(r, k)] = c
        rows, cols = linear_sum_assignment(cost)
        for r, k in zip(rows, cols):
            if cost[r, k] < BIG_COST:
                result[idx[r]] = lookup[(r, k)]
    return result


def qqbar_allowed(J: int, P: int, C: Optional[int]) -> bool:
    """Whether a q qbar state (P = (-1)^(L+1), C = (-1)^(L+S)) can have this J^P(C)."""
    for L in range(max(0, J - 1), J + 2):
        for S in (0, 1):
            if not abs(L - S) <= J <= L + S:
                continue
            if (-1) ** (L + 1) == P and (C is None or (-1) ** (L + S) == C):
                return True
    return False


def unassigned_states(
    states: Iterable[PDGState],
    assigned: Iterable[Optional[Candidate]],
    window: tuple[float, float],
    system: SystemFlavour,
    settings: MatchSettings,
    jpc_filter=None,
) -> list[UnassignedState]:
    """Compatible PDG states in [mass_min - tol, mass_max + tol] that no prediction took.

    ``jpc_filter(J, P, C) -> bool`` applies the user's J^P / C filter. States whose J^PC no
    q qbar state can have are kept and flagged as exotic.
    """
    taken = {c.state for c in assigned if c is not None}
    out = []
    for s in states:
        if s in taken or not flavour_match(s, system):
            continue
        C = s.C if system.self_conjugate else None
        if jpc_filter is not None and not jpc_filter(s.J, s.P, C):
            continue
        tol, source = effective_tolerance(s, settings)
        if not window[0] - tol <= s.mass <= window[1] + tol:
            continue
        flags = s.flags + (() if qqbar_allowed(s.J, s.P, C) else (FLAG_EXOTIC,))
        out.append(UnassignedState(s, tol, source, flags))
    out.sort(key=lambda u: (u.state.mass, u.state.name))
    return out


def match(
    predictions: Sequence[Prediction],
    states: Sequence[PDGState],
    system: SystemFlavour,
    settings: MatchSettings,
    window: tuple[float, float],
    jpc_filter=None,
) -> MatchResult:
    cands = [find_candidates(p, states, system, settings) for p in predictions]
    assigned = assign(predictions, cands)
    unassigned = unassigned_states(states, assigned, window, system, settings, jpc_filter)
    return MatchResult(cands, assigned, unassigned)


# --- serialisation ----------------------------------------------------------------------------


def state_dict(state: PDGState) -> dict:
    return {
        "name": state.name,
        "members": list(state.members),
        "pdg_listing": state.pdgid,
        "status": state.status,
        "mass_MeV": state.mass,
        "width_MeV": state.width,
        "J": state.J,
        "P": state.P,
        "C": state.C,
        "isospin": state.isospin,
    }


def candidate_dict(c: Candidate, assigned_to: Optional[str] = None) -> dict:
    return {
        **state_dict(c.state),
        "delta_MeV": c.delta,
        "tolerance_MeV": c.tolerance,
        "tolerance_source": c.tolerance_source,
        "flags": list(c.flags),
        "assigned_to": assigned_to,
    }


def unassigned_dict(u: UnassignedState) -> dict:
    return {
        **state_dict(u.state),
        "tolerance_MeV": u.tolerance,
        "tolerance_source": u.tolerance_source,
        "flags": list(u.flags),
    }
