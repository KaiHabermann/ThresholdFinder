"""Tests for flavor quantum number filtering."""
import pytest
from threshold_finder.flavor import FlavorFilter, parse_quark_content
from threshold_finder import ThresholdFinder


# --- parse_quark_content ---

def test_parse_pi_plus():
    assert parse_quark_content("uD") == {"u": 1, "d": -1}

def test_parse_pi_minus():
    assert parse_quark_content("Ud") == {"u": -1, "d": 1}

def test_parse_kaon_plus():
    assert parse_quark_content("uS") == {"u": 1, "s": -1}

def test_parse_D_plus():
    assert parse_quark_content("cD") == {"c": 1, "d": -1}

def test_parse_B_plus():
    assert parse_quark_content("uB") == {"u": 1, "b": -1}

def test_parse_proton():
    assert parse_quark_content("uud") == {"u": 2, "d": 1}

def test_parse_omega_minus():
    assert parse_quark_content("sss") == {"s": 3}

def test_parse_mixed_returns_none():
    assert parse_quark_content("(uU-dD)/sqrt(2)") is None
    assert parse_quark_content("x(uU+dD)+y(sS)") is None


# --- FlavorFilter.check ---

def test_empty_filter_always_passes():
    f = FlavorFilter()
    assert f.check({"u": 1, "d": -1}, {"u": -1, "d": 1})
    assert f.check(None, None)

def test_filter_rejects_wrong_flavor():
    f = FlavorFilter(s=0)
    # pi+pi- has no strangeness -> passes
    assert f.check({"u": 1, "d": -1}, {"u": -1, "d": 1})
    # K+pi- has net s=-1 -> fails
    assert not f.check({"u": 1, "s": -1}, {"u": -1, "d": 1})

def test_filter_mixed_state_excluded_when_constrained():
    f = FlavorFilter(u=0)
    assert not f.check(None, {"u": 1})

def test_filter_mixed_state_allowed_when_unconstrained():
    f = FlavorFilter()
    assert f.check(None, None)

def test_partial_filter_only_checks_set_flavors():
    # Only constrain s=0, leave u/d free
    f = FlavorFilter(s=0)
    assert f.check({"u": 2, "d": 1}, {"u": -2, "d": -1})   # no strangeness -> passes
    assert not f.check({"u": 1, "s": -1}, {"u": -1, "d": 1})  # s=-1 net -> fails


# --- Integration: ThresholdFinder with FlavorFilter ---

def test_strangeness_zero_includes_kaon_pairs():
    # K+K- has net s = (-1)+(+1) = 0, so it passes an s=0 filter
    result = ThresholdFinder(
        970, 1010, 1, -1,
        flavor_filter=FlavorFilter(s=0),
    ).run()
    pair_names = {tuple(sorted([c.particle1, c.particle2])) for c in result.combinations}
    assert ("K+", "K-") in pair_names

def test_strangeness_one_excludes_pipi():
    # pi+pi- has net s=0, does not pass an s=1 filter
    result = ThresholdFinder(
        250, 300, 1, -1,
        flavor_filter=FlavorFilter(s=1),
    ).run()
    pair_names = {tuple(sorted([c.particle1, c.particle2])) for c in result.combinations}
    assert ("pi+", "pi-") not in pair_names

def test_strangeness_zero_allows_pipi_in_range():
    result = ThresholdFinder(
        250, 300, 1, -1,
        flavor_filter=FlavorFilter(u=0, d=0, s=0, c=0, b=0),
    ).run()
    pair_names = {tuple(sorted([c.particle1, c.particle2])) for c in result.combinations}
    assert ("pi+", "pi-") in pair_names

def test_nonzero_strangeness_selects_kaon_pairs():
    # K+pi- has net s=-1; look for 0^- channels near K+pi- threshold
    result = ThresholdFinder(
        600, 700, 0, -1,
        total_charge=0.0,
        flavor_filter=FlavorFilter(s=-1),
    ).run()
    # K+ + K- has s=0, should be absent; K pi has s=-1 and Q=0 only for K0 pi0 etc.
    for c in result.combinations:
        qc1 = c.particle1
        qc2 = c.particle2
        # Every result should involve a strange particle
        names = {qc1, qc2}
        has_kaon = any("K" in n for n in names)
        assert has_kaon, f"Expected kaon in pair: {c}"
