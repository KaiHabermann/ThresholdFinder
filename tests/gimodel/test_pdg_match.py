"""Unit tests for gimodel.pdg_match with synthetic PDG states (no PDG table access)."""
from __future__ import annotations

import pytest

from gimodel.pdg_match import (
    FLAG_AMBIGUOUS,
    FLAG_EXOTIC,
    MatchSettings,
    PDGEntry,
    PDGState,
    Prediction,
    SystemFlavour,
    assign,
    base_name,
    collapse_multiplets,
    effective_tolerance,
    find_candidates,
    flavour_match,
    match,
    qqbar_allowed,
    quantum_numbers_match,
    unassigned_states,
)

CC = frozenset({("c", "c")})
QQ = frozenset({("q", "q")})
SETTINGS = MatchSettings(mass_tol=50.0, width_frac=0.5)
CHARMONIUM = SystemFlavour.from_quarks("c", "c")


def state(name, mass, J=1, P=-1, C=-1, width=None, flavours=CC, isospin=0.0, flags=()):
    return PDGState(name=name, mass=mass, width=width, J=J, P=P, C=C, isospin=isospin,
                    flavours=flavours, flags=flags)


def pred(label, mass, J=1, P=-1, C=-1):
    return Prediction(label, J, P, C, mass)


def entry(name, mcid, mass, J=0, P=-1, C=None, I=0.5, charge=0.0, flavours=frozenset({("q", "c")}),
          width=None, status=0, listing=None, flags=()):
    return PDGEntry(name=name, listing=listing or f"L{mcid}", mcid=mcid, mass=mass, width=width, J=J, P=P,
                    C=C, isospin=I, charge=charge, flavours=flavours, flags=flags, status=status)


# --- tolerance --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width, expected",
    [(None, (50.0, "mass_tol")), (10.0, (50.0, "mass_tol")), (100.0, (50.0, "mass_tol")),
     (300.0, (150.0, "width"))],
)
def test_effective_tolerance(width, expected):
    assert effective_tolerance(state("X", 1000.0, width=width), SETTINGS) == expected


def test_effective_tolerance_follows_settings():
    s = state("X", 1000.0, width=40.0)
    assert effective_tolerance(s, MatchSettings(mass_tol=5.0, width_frac=0.5)) == (20.0, "width")
    assert effective_tolerance(s, MatchSettings(mass_tol=5.0, width_frac=0.0)) == (5.0, "mass_tol")


def test_width_widens_window():
    p = pred("2^3S_1", 1300.0)
    broad = state("rho-like", 1450.0, width=400.0)       # tol = 200
    narrow = state("narrow", 1450.0, width=20.0)         # tol = 50
    unknown = state("unknown-width", 1450.0, width=None)  # tol = 50
    found = find_candidates(p, [broad, narrow, unknown], CHARMONIUM, SETTINGS)
    assert [c.state.name for c in found] == ["rho-like"]
    c = found[0]
    assert c.delta == pytest.approx(-150.0)
    assert (c.tolerance, c.tolerance_source) == (200.0, "width")


def test_boundary_is_inclusive():
    found = find_candidates(pred("x", 1050.0), [state("X", 1000.0)], CHARMONIUM, SETTINGS)
    assert len(found) == 1


# --- quantum numbers --------------------------------------------------------------------------


def test_jp_must_match():
    s = state("X", 1000.0, J=1, P=-1, C=-1)
    assert quantum_numbers_match(1, -1, -1, s)
    assert not quantum_numbers_match(1, +1, -1, s)
    assert not quantum_numbers_match(2, -1, -1, s)


def test_c_compared_only_when_both_defined():
    assert not quantum_numbers_match(1, 1, 1, state("h", 1000.0, J=1, P=1, C=-1))
    assert quantum_numbers_match(1, 1, None, state("h", 1000.0, J=1, P=1, C=-1))  # open flavour prediction
    assert quantum_numbers_match(1, 1, 1, state("K1", 1000.0, J=1, P=1, C=None))


def test_c_parity_separates_chi_c1_and_h_c():
    chi = state("chi(c1)(1P)", 3510.7, J=1, P=1, C=1)
    hc = state("h(c)(1P)", 3525.4, J=1, P=1, C=-1)
    found = find_candidates(pred("1^1P_1", 3515.1, J=1, P=1, C=-1), [chi, hc], CHARMONIUM, SETTINGS)
    assert [c.state.name for c in found] == ["h(c)(1P)"]


def test_candidates_ranked_by_abs_delta():
    states = [state("A", 1040.0), state("B", 990.0), state("C", 1020.0)]
    found = find_candidates(pred("x", 1000.0), states, CHARMONIUM, SETTINGS)
    assert [c.state.name for c in found] == ["B", "C", "A"]


# --- flavour ----------------------------------------------------------------------------------


def test_flavour_open_pair_is_unordered():
    ds = state("D(s)+", 1968.0, J=0, P=-1, C=None, flavours=frozenset({("s", "c")}))
    assert flavour_match(ds, SystemFlavour.from_quarks("c", "s"))
    assert flavour_match(ds, SystemFlavour.from_quarks("s", "c"))
    assert not flavour_match(ds, SystemFlavour.from_quarks("c", "u"))


def test_flavour_u_dbar_requires_isovector():
    rho = state("rho", 775.0, flavours=QQ, isospin=1.0)
    omega = state("omega", 783.0, flavours=frozenset({("q", "q"), ("s", "s")}), isospin=0.0)
    ud = SystemFlavour.from_quarks("u", "d")
    uu = SystemFlavour.from_quarks("u", "u")
    assert flavour_match(rho, ud) and not flavour_match(omega, ud)
    assert flavour_match(rho, uu) and flavour_match(omega, uu)
    assert flavour_match(omega, SystemFlavour.from_quarks("s", "s"))
    assert not flavour_match(rho, SystemFlavour.from_quarks("s", "s"))


# --- multiplet collapse -----------------------------------------------------------------------


def test_collapse_charge_states_across_listings():
    entries = [
        entry("D0", 421, 1864.8, listing="S032"),
        entry("D+", 411, 1869.7, charge=1.0, listing="S031"),
        entry("D^*(2007)0", 423, 2006.9, J=1, width=2.0, listing="M061"),
        entry("D^*(2010)+", 413, 2010.3, J=1, charge=1.0, width=0.1, listing="M062"),
        entry("D_s+", 431, 1968.3, I=0.0, charge=1.0, flavours=frozenset({("s", "c")})),
    ]
    states = {s.name: s for s in collapse_multiplets(entries)}
    assert set(states) == {"D", "D^*(2007)", "D_s+"}  # singletons keep their full PDG name
    d = states["D"]
    assert d.members == ("D0", "D+")
    assert d.mass == pytest.approx((1864.8 + 1869.7) / 2)
    assert d.width is None and d.pdgid == "S032"
    dstar = states["D^*(2007)"]
    assert dstar.members == ("D^*(2007)0", "D^*(2010)+")
    assert dstar.width == pytest.approx(1.05)


def test_collapse_by_listing_without_mcid():
    entries = [
        entry("D_0(2550)+", None, 2549.0, charge=1.0, listing="M198"),
        entry("D_0(2550)0", None, 2549.0, listing="M198"),
        entry("D_1^*(2600)0", None, 2627.4, J=1, listing="M199"),
    ]
    states = {s.name: s for s in collapse_multiplets(entries)}
    assert set(states) == {"D_0(2550)", "D_1^*(2600)0"}
    assert states["D_0(2550)"].members == ("D_0(2550)0", "D_0(2550)+")


def test_collapse_keeps_isospin_partners_apart_from_isoscalars():
    QS = frozenset({("q", "q"), ("s", "s")})
    entries = [
        entry("pi0", 111, 135.0, C=1, I=1.0, flavours=QQ),
        entry("pi+", 211, 139.6, I=1.0, charge=1.0, flavours=QQ),
        entry("eta", 221, 547.9, C=1, I=0.0, flavours=QS, flags=(FLAG_AMBIGUOUS,)),
        entry("rho(770)+", 213, 775.1, J=1, I=1.0, charge=1.0, flavours=QQ, listing="M009"),
        entry("rho(770)0", 113, 775.3, J=1, C=-1, I=1.0, flavours=QQ, listing="M009"),
    ]
    states = {s.name: s for s in collapse_multiplets(entries)}
    assert set(states) == {"pi", "eta", "rho(770)"}
    assert states["rho(770)"].C == -1          # C from the neutral member
    assert states["rho(770)"].members[0] == "rho(770)0"
    assert states["pi"].flavours == QQ
    assert FLAG_AMBIGUOUS in states["eta"].flags


def test_collapse_skips_unknown_jp_flavour_and_weak_eigenstates():
    entries = [
        entry("D_s^*+", 433, 2112.2, J=1, P=None, I=0.0, charge=1.0),
        entry("Jless", None, 4000.0, J=None),
        entry("X(3940)", None, 3942.0, J=None, P=None),
        entry("no-flavour", None, 1000.0, flavours=frozenset()),
        entry("K(S)0", 310, 497.6, flavours=frozenset({("q", "s")})),
        entry("K0", 311, 497.6, flavours=frozenset({("q", "s")})),
    ]
    assert [s.name for s in collapse_multiplets(entries)] == ["K0"]


def test_base_name():
    assert base_name("D^*(2010)+") == "D^*(2010)"
    assert base_name("K0") == "K"
    assert base_name("D*(2007)~0") == "D*(2007)"
    assert base_name("a(0)(980)-") == "a(0)(980)"
    assert base_name("chi(c1)(1P)") == "chi(c1)(1P)"


# --- one-to-one assignment --------------------------------------------------------------------


def test_assignment_resolves_competition():
    # A's nearest state is X, but X is B's only option: the optimum is A->Y, B->X.
    preds = [pred("A", 1000.0), pred("B", 1010.0)]
    states = [state("X", 1008.0), state("Y", 960.0)]
    settings = MatchSettings(mass_tol=45.0)
    cands = [find_candidates(p, states, CHARMONIUM, settings) for p in preds]
    assert [c.state.name for c in cands[0]] == ["X", "Y"]
    assert [c.state.name for c in cands[1]] == ["X"]
    got = assign(preds, cands)
    assert [c.state.name for c in got] == ["Y", "X"]


def test_assignment_minimises_total_delta():
    preds = [pred("A", 1000.0), pred("B", 1030.0)]
    states = [state("X", 1010.0), state("Y", 1035.0)]
    got = assign(preds, [find_candidates(p, states, CHARMONIUM, SETTINGS) for p in preds])
    assert [c.state.name for c in got] == ["X", "Y"]


def test_assignment_leaves_surplus_prediction_unmatched():
    preds = [pred("A", 1000.0), pred("B", 1004.0)]
    got = assign(preds, [find_candidates(p, [state("X", 1003.0)], CHARMONIUM, SETTINGS) for p in preds])
    assert got[0] is None and got[1].state.name == "X"


def test_assignment_is_per_jpc_group():
    preds = [pred("vector", 1000.0, J=1, P=-1, C=-1), pred("pseudoscalar", 1000.0, J=0, P=-1, C=1)]
    states = [state("V", 1001.0, J=1, P=-1, C=-1), state("PS", 1001.0, J=0, P=-1, C=1)]
    got = assign(preds, [find_candidates(p, states, CHARMONIUM, SETTINGS) for p in preds])
    assert [c.state.name for c in got] == ["V", "PS"]


def test_out_of_tolerance_pairs_never_assigned():
    preds = [pred("A", 1000.0)]
    got = assign(preds, [find_candidates(preds[0], [state("far", 1200.0)], CHARMONIUM, SETTINGS)])
    assert got == [None]


# --- unassigned section -----------------------------------------------------------------------


@pytest.mark.parametrize("J, P, C, ok", [(1, -1, -1, True), (0, 1, 1, True), (1, 1, -1, True),
                                         (1, -1, 1, False), (0, -1, -1, False), (0, 1, -1, False),
                                         (2, 1, -1, False), (1, -1, None, True)])
def test_qqbar_allowed(J, P, C, ok):
    assert qqbar_allowed(J, P, C) is ok


def test_unassigned_window_and_flags():
    preds = [pred("1^3S_1", 3091.0)]
    jpsi = state("J/psi", 3096.9)
    inside = state("psi-in", 3620.0)                       # below mass_max
    widened = state("psi-wide", 3640.0)                    # within mass_max + 50
    broad = state("psi-broad", 3800.0, width=500.0)        # within mass_max + 250
    outside = state("psi-out", 3700.0, width=10.0)
    exotic = state("pi1-like", 3500.0, J=1, P=-1, C=1)
    light = state("rho", 3500.0, flavours=QQ, isospin=1.0)
    result = match(preds, [jpsi, inside, widened, broad, outside, exotic, light], CHARMONIUM, SETTINGS,
                   (3000.0, 3600.0))
    assert result.assigned[0].state is jpsi
    names = [u.state.name for u in result.unassigned]
    assert names == ["pi1-like", "psi-in", "psi-wide", "psi-broad"]
    assert FLAG_EXOTIC in result.unassigned[0].flags
    assert result.unassigned[3].tolerance_source == "width"


def test_unassigned_respects_jpc_filter():
    states = [state("V", 3500.0), state("S", 3500.0, J=0, P=1, C=1)]
    got = unassigned_states(states, [], (3000.0, 3600.0), CHARMONIUM, SETTINGS,
                            jpc_filter=lambda J, P, C: J == 0)
    assert [u.state.name for u in got] == ["S"]
