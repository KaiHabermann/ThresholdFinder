"""Integration tests for ThresholdFinder."""
import pytest
from threshold_finder import ThresholdFinder


def test_pipi_rho_channel():
    # pi+pi- threshold at ~279 MeV, should give 1^- with L=1
    result = ThresholdFinder(250, 300, 1, -1).run()
    names = {(c.particle1, c.particle2) for c in result.combinations}
    assert ("pi+", "pi-") in names or ("pi-", "pi+") in names


def test_pipi_L1_only_for_1minus():
    # In the pi+pi- threshold region, L must be 1 for 1^-
    result = ThresholdFinder(250, 300, 1, -1, max_L=0).run()
    assert len(result.combinations) == 0


def test_kk_1minus_appears():
    result = ThresholdFinder(970, 1010, 1, -1).run()
    pair_names = {tuple(sorted([c.particle1, c.particle2])) for c in result.combinations}
    assert ("K+", "K-") in pair_names


def test_no_results_wrong_parity():
    # pi0 pi0 (identical bosons) can only have even L -> only even parity P = (+1)(+1)(-1)^L
    # For 1^- we need P=-1 and J=1. With two pi0: L=1 is odd -> forbidden; L=3 gives J=3 not 1.
    result = ThresholdFinder(250, 290, 1, -1, max_L=3).run()
    # pi0pi0 should NOT appear (identical boson constraint)
    pi0_pi0 = [c for c in result.combinations if c.particle1 == "pi0" and c.particle2 == "pi0"]
    assert len(pi0_pi0) == 0


def test_max_L_respected():
    result_L0 = ThresholdFinder(250, 300, 1, -1, max_L=0).run()
    result_L1 = ThresholdFinder(250, 300, 1, -1, max_L=1).run()
    assert len(result_L0.combinations) < len(result_L1.combinations)
