"""Tests for the sequential coupling scheme and qn-options command."""
import pytest
from threshold_finder.qn import (
    couple_sequential,
    first_L_tuple,
    all_L_tuples,
)
from threshold_finder.lookup import qn_options_for_channel
from threshold_finder.cli import qn_options_main


# ---------------------------------------------------------------------------
# couple_sequential
# ---------------------------------------------------------------------------

def test_couple_sequential_two_pions_L1():
    # pi+ pi-: J=0 P=-1 each, L1=1 -> J=1, P=(-1)(-1)(-1)^1 = -1  -> 1^-
    states = couple_sequential([0.0, 0.0], [-1, -1], (1,))
    assert (1.0, -1) in states


def test_couple_sequential_two_pions_L0():
    # L=0: J=0, P=+1 -> 0^+
    states = couple_sequential([0.0, 0.0], [-1, -1], (0,))
    assert (0.0, 1) in states
    assert (1.0, -1) not in states


def test_couple_sequential_three_pions_L00_gives_0minus():
    # Three pi (J=0, P=-1), L1=0 L2=0: P=(-1)^3*(-1)^0*(-1)^0 = -1, J=0 -> 0^-
    states = couple_sequential([0.0, 0.0, 0.0], [-1, -1, -1], (0, 0))
    assert (0.0, -1) in states


def test_couple_sequential_three_pions_L10_gives_1plus():
    # L1=1 L2=0: P=(-1)^3*(-1)^1*(-1)^0=+1, J reachable includes 1 -> 1^+
    states = couple_sequential([0.0, 0.0, 0.0], [-1, -1, -1], (1, 0))
    assert (1.0, 1) in states


def test_couple_sequential_wrong_length_raises():
    with pytest.raises((AssertionError, TypeError)):
        couple_sequential([0.0, 0.0, 0.0], [-1, -1, -1], (0,))  # need 2 L values for 3 particles


def test_couple_sequential_two_nucleons_L0():
    # Two protons: J=1/2, P=+1 each, L=0 -> S=0 or S=1, P=+1 -> {0^+, 1^+}
    states = couple_sequential([0.5, 0.5], [1, 1], (0,))
    assert (0.0, 1) in states
    assert (1.0, 1) in states


# ---------------------------------------------------------------------------
# first_L_tuple
# ---------------------------------------------------------------------------

def test_first_L_tuple_two_pions_1minus():
    # Two J=0 particles: total J == L, so only L=1 gives J=1
    result = first_L_tuple([0.0, 0.0], [-1, -1], 1.0, -1, L_max=4)
    assert result == (1,)


def test_first_L_tuple_three_pions_0minus():
    result = first_L_tuple([0.0, 0.0, 0.0], [-1, -1, -1], 0.0, -1, L_max=4)
    assert result == (0, 0)


def test_first_L_tuple_impossible_returns_none():
    # Two spin-1/2 fermions can never produce J=5 at any L<=2
    result = first_L_tuple([0.5, 0.5], [1, 1], 5.0, 1, L_max=2)
    assert result is None


def test_first_L_tuple_is_lex_smallest():
    # For three pions -> 1^+: parity needs (-1)^3 * (-1)^(L1+L2) = +1
    # so L1+L2 must be odd. Lex-first odd-sum tuple is (0,1).
    result = first_L_tuple([0.0, 0.0, 0.0], [-1, -1, -1], 1.0, 1, L_max=4)
    assert result == (0, 1)


# ---------------------------------------------------------------------------
# all_L_tuples
# ---------------------------------------------------------------------------

def test_all_L_tuples_two_pions_1minus():
    results = all_L_tuples([0.0, 0.0], [-1, -1], 1.0, -1, L_max=3)
    # Two J=0 particles: J==L, so only L=1 gives J=1 with P=(-1)(-1)(-1)^1=-1
    assert (1,) in results
    assert (0,) not in results
    assert (2,) not in results
    assert (3,) not in results


def test_all_L_tuples_empty_when_impossible():
    results = all_L_tuples([0.5, 0.5], [1, 1], 5.0, 1, L_max=2)
    assert results == []


def test_all_L_tuples_three_pions_0minus():
    results = all_L_tuples([0.0, 0.0, 0.0], [-1, -1, -1], 0.0, -1, L_max=2)
    # Need P=-1: (-1)^3 * (-1)^(L1+L2) = -1 -> L1+L2 even
    # and J=0: spins all 0 so J from orbital coupling only, need |L1-L2|..L1+L2 to include 0
    # -> L1==L2. So (0,0),(1,1),(2,2).
    assert (0, 0) in results
    assert (1, 1) in results
    assert (2, 2) in results
    for tup in results:
        assert tup[0] == tup[1], f"Expected L1==L2, got {tup}"


# ---------------------------------------------------------------------------
# qn_options_for_channel
# ---------------------------------------------------------------------------

def test_qn_options_for_channel_pipi_1minus():
    # Two J=0 particles: J==L, only L=1 achieves J^P=1^-
    results = qn_options_for_channel(["pi+", "pi-"], 1.0, -1, L_max=3)
    assert (1,) in results
    assert len(results) == 1


def test_qn_options_for_channel_three_pions_0minus():
    results = qn_options_for_channel(["pi+", "pi-", "pi0"], 0.0, -1, L_max=2)
    assert (0, 0) in results


def test_qn_options_for_channel_unknown_particle_raises():
    with pytest.raises(LookupError):
        qn_options_for_channel(["pi+", "DOESNOTEXIST"], 1.0, -1, L_max=2)


# ---------------------------------------------------------------------------
# qn_options_main CLI
# ---------------------------------------------------------------------------

def test_qn_options_cli_two_pions(capsys):
    qn_options_main(["1", "-1", "pi+", "pi-", "--max-L", "3"])
    out = capsys.readouterr().out
    assert "1^-" in out
    assert "1" in out  # L=1 appears


def test_qn_options_cli_three_pions_0minus(capsys):
    qn_options_main(["0", "-1", "pi+", "pi-", "pi0", "--max-L", "2"])
    out = capsys.readouterr().out
    assert "0^-" in out
    assert "0" in out


def test_qn_options_cli_no_solution(capsys):
    qn_options_main(["5", "+1", "pi+", "pi-", "--max-L", "1"])
    out = capsys.readouterr().out
    assert "No L combination" in out


def test_qn_options_cli_unknown_particle(capsys):
    with pytest.raises(SystemExit):
        qn_options_main(["1", "-1", "NOPE", "pi-"])
    err = capsys.readouterr().err
    assert "Unknown particle" in err


def test_qn_options_cli_warning_on_many_results(capsys):
    # Three protons (J=1/2) with large L_max produces > 25 combinations -> triggers warning
    qn_options_main(["0.5", "-1", "p", "p", "p", "--max-L", "15"])
    err = capsys.readouterr().err
    assert "WARNING" in err
