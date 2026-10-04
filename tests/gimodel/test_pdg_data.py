"""Tests for gimodel.pdg_data: pure parsing with synthetic rows, plus a smoke test of the PDG API."""
from __future__ import annotations

import pytest

from gimodel.pdg_data import (
    FLAG_FAMILY_FROM_SECTION,
    FLAG_MASS_RANGE,
    FLAG_UNCERTAIN,
    Quantum,
    RawState,
    assign_families,
    entries_from_raw,
    family_from_name,
    flavour_pairs,
    parse_quantum,
    resolve,
    status_of,
)
from gimodel.pdg_match import FLAG_AMBIGUOUS, LIGHT_ISOSCALAR


@pytest.mark.parametrize(
    "text, kind, expected",
    [
        ("1", "J", Quantum((1,))),
        ("(1)", "J", Quantum((1,), True)),
        ("2++ or 4", "J", Quantum((2, 4), True)),
        ("?", "J", Quantum(())),
        (None, "J", Quantum(())),
        ("+", "P", Quantum((1,))),
        ("(-)", "P", Quantum((-1,), True)),
        ("?", "P", Quantum(())),
        ("None", "C", Quantum(())),
        ("1/2", "I", Quantum((0.5,))),
        ("0", "I", Quantum((0.0,))),
        ("even", "J", Quantum(())),
    ],
)
def test_parse_quantum(text, kind, expected):
    assert parse_quantum(text, kind) == expected


def test_resolve_uncertain():
    assert resolve(Quantum((1,), True), include_uncertain=False) == ()
    assert resolve(Quantum((1,), True), include_uncertain=True) == (1,)
    assert resolve(Quantum((2, 4), True), include_uncertain=True) == (2, 4)
    assert resolve(Quantum(()), include_uncertain=True) == ()


@pytest.mark.parametrize(
    "name, family",
    [
        ("pi+", "light"), ("rho(770)0", "light"), ("phi(1020)0", "light"), ("eta^'(958)0", "light"),
        ("f_2^'(1525)0", "light"), ("b_1(1235)+", "light"), ("h_1(1170)0", "light"),
        ("K^*(892)+", ("q", "s")), ("D^*(2010)+", ("q", "c")), ("D_s1^*(2700)+", ("s", "c")),
        ("B0", ("q", "b")), ("B_s2^*(5840)0", ("s", "b")), ("B_c+", ("c", "b")),
        ("eta_c(1S)", ("c", "c")), ("J/psi(1S)", ("c", "c")), ("chi_c1(3872)", ("c", "c")),
        ("psi(4230)", ("c", "c")), ("h_c(1P)", ("c", "c")), ("Upsilon(10753)", ("b", "b")),
        ("h_b(1P)", ("b", "b")), ("X(3940)", "section"), ("T_{c cbar 1}(3900)+", None),
        ("T_cc(3875)+", None),
    ],
)
def test_family_from_name(name, family):
    assert family_from_name(name) == family


def test_x_states_take_their_section():
    fams = assign_families(["rho(1700)0", "X(1750)0", "chi_c2(3930)", "X(3940)", "psi(4040)"])
    assert fams[1] == ("section", "light") and fams[3] == ("section", ("c", "c"))
    pairs, flags = flavour_pairs(fams[3], 0.0)
    assert pairs == {("c", "c")} and flags == (FLAG_FAMILY_FROM_SECTION,)


def test_flavour_pairs_light():
    assert flavour_pairs("light", 1.0) == (frozenset({("q", "q")}), ())
    pairs, flags = flavour_pairs("light", 0.0)
    assert pairs == LIGHT_ISOSCALAR and flags == (FLAG_AMBIGUOUS,)
    pairs, flags = flavour_pairs("light", None)
    assert pairs == LIGHT_ISOSCALAR and "isospin unknown" in flags
    assert flavour_pairs(None, 1.0) == (frozenset(), ())


def test_status_mapping():
    assert status_of(True) == 0 and status_of(False) == 2


def raw(name, J="1", P="-", C=None, I="0", mass=1000.0, family="light", in_st=True, rng=False):
    return RawState(name=name, listing="M999", mcid=None, charge=0.0, mass=mass, mass_is_range=rng,
                    width=None, I=I, J=J, P=P, C=C, in_summary_table=in_st, family=family)


def test_entries_uncertain_jp():
    rows = [raw("f_J(2220)0", J="2++ or 4", P="+", C="+"), raw("Xq", J="?", P="?"), raw("Xp", J="(1)", P="-")]
    certain = entries_from_raw(rows)
    assert [(e.name, e.J, e.P) for e in certain] == [("f_J(2220)0", None, 1), ("Xq", None, None), ("Xp", None, -1)]
    loose = entries_from_raw(rows, include_uncertain=True)
    assert [(e.name, e.J) for e in loose] == [("f_J(2220)0", 2), ("f_J(2220)0", 4), ("Xq", None), ("Xp", 1)]
    assert loose[0].flags[0].startswith(FLAG_UNCERTAIN) and loose[0].C == 1
    assert not any(f.startswith(FLAG_UNCERTAIN) for f in loose[2].flags)


def test_entries_range_mass_status_and_missing_mass():
    rows = [raw("f_0(500)0", J="0", P="+", mass=600.0, rng=True), raw("nomass", mass=None),
            raw("listing-only", in_st=False)]
    out = entries_from_raw(rows)
    assert [e.name for e in out] == ["f_0(500)0", "listing-only"]
    assert FLAG_MASS_RANGE in out[0].flags
    assert [e.status for e in out] == [0, 2]


def test_real_pdg_api_smoke():
    pytest.importorskip("pdg")
    from gimodel import pdg_data
    from gimodel.pdg_match import collapse_multiplets

    src = pdg_data.source_info()
    assert src.edition.isdigit() and int(src.edition) >= 2024
    states = {s.name: s for s in collapse_multiplets(pdg_data.load_entries())}
    assert states["J/psi(1S)"].C == -1 and states["J/psi(1S)"].flavours == {("c", "c")}
    assert states["D_s^*+"].P == -1                      # P unknown in the `particle` package
    assert states["D_s1^*(2700)+"].J == 1                # no MC ID, absent from `particle`
    assert states["D^*(2007)"].members == ("D^*(2007)0", "D^*(2010)+")
    assert states["rho(770)"].members == ("rho(770)0", "rho(770)+")
    assert "X(3940)" not in states                       # J = '?'
