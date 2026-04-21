"""Tests for quantum number coupling utilities."""
import pytest
from threshold_finder.qn import j_range, parity, can_produce, identical_bosons_L_allowed


def test_j_range_two_scalars():
    # Two J=0 particles, L=1 -> J=1 only
    assert list(j_range(0, 0, 1)) == [1.0]


def test_j_range_scalar_vector():
    # J=0 + J=1, L=0 -> J=1
    assert list(j_range(0, 1, 0)) == [1.0]


def test_j_range_two_vectors_L1():
    # J=1 + J=1, L=1: s in {0,1,2}, then |s-1|..s+1
    result = sorted(set(j_range(1, 1, 1)))
    assert 0.0 in result
    assert 1.0 in result
    assert 2.0 in result


def test_parity_pi_pi_L1():
    assert parity(-1, -1, 1) == -1


def test_parity_pi_pi_L0():
    assert parity(-1, -1, 0) == 1


def test_identical_bosons():
    assert identical_bosons_L_allowed(0) is True
    assert identical_bosons_L_allowed(1) is False
    assert identical_bosons_L_allowed(2) is True


def test_can_produce_rho_from_pipi():
    # pi+pi- in L=1 -> 1^-
    assert can_produce(0, -1, 0, -1, 1, -1, 1, identical=False, both_bosons=True)


def test_pipi_1minus_forbidden_for_pi0pi0():
    # pi0 pi0 are identical bosons, L=1 is odd -> forbidden
    assert not can_produce(0, -1, 0, -1, 1, -1, 1, identical=True, both_bosons=True)


def test_pipi_0plus_L0_forbidden():
    # pi pi L=0: P = (-1)(-1)(+1) = +1, J=0 -> 0^+ allowed for non-identical
    assert can_produce(0, -1, 0, -1, 0, 1, 0, identical=False, both_bosons=True)
    # identical bosons L=0 is even -> allowed
    assert can_produce(0, -1, 0, -1, 0, 1, 0, identical=True, both_bosons=True)
