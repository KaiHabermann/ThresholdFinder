"""Harmonic-oscillator basis matrices against GIModel.jl (tests/reference/ho_matrices.json)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from gimodel import ConstituentMasses, smearing
from gimodel import ho
from gimodel.fd import gi_spin_dependent_side_exponent
from gimodel.solvers import HO_BETA_GRID, OscillatorSolver

from conftest import load_reference

ATOL = 1e-10


@pytest.fixture(scope="module")
def ref():
    return load_reference("ho_matrices.json")


def _close(actual, expected, atol=ATOL, rtol=0.0):
    expected = np.array(expected, dtype=float)
    assert actual.shape == expected.shape
    err = np.max(np.abs(actual - expected))
    assert err <= atol + rtol * np.max(np.abs(expected)), err


def test_beta_grid_matches_julia(ref):
    assert OscillatorSolver().beta_grid == HO_BETA_GRID
    expected = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45,
                1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35]
    assert list(HO_BETA_GRID) == expected


def test_reduced_radial_and_dvr_nodes(ref):
    for e in ref["ho_reduced_radial"]:
        u = ho.ho_reduced_radial(e["n"], e["L"], e["beta"], np.array(e["r"]))
        assert np.allclose(u, e["u"], rtol=1e-13, atol=1e-15)
    for e in ref["gauss_laguerre_dvr"]:
        sqrt_x, Z = ho.gauss_laguerre_dvr(e["L"], e["nbasis"], e["nq"])
        assert np.allclose(sqrt_x, e["sqrt_x"], rtol=1e-13, atol=1e-14)
        assert Z.shape == (e["nbasis"], e["nq"])


def test_exact_matrices_self_checks():
    for L in (0, 1, 2):
        beta = 0.7
        mu = 0.8
        # p^2/2mu + beta^4 r^2/2mu is the diagonal oscillator Hamiltonian.
        H = ho.ho_p2_matrix(L, beta, 12) / (2 * mu) + beta**4 * ho.ho_r2_matrix(L, beta, 12) / (2 * mu)
        assert np.allclose(H - np.diag(np.diag(H)), 0.0, atol=1e-13)
        assert np.allclose(ho.ho_operator_matrix(L, beta, 12, lambda r: np.ones_like(r)), np.eye(12), atol=1e-13)
        assert np.allclose(ho.ho_operator_matrix(L, beta, 12, lambda r: r**2), ho.ho_r2_matrix(L, beta, 12), atol=1e-11)


def test_position_momentum_and_central_matrices(ref, params, mq):
    cc = ConstituentMasses(mq["c"], mq["c"])
    cu = ConstituentMasses(mq["c"], mq["u"])
    mc = cc.m1_GeV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Julia also hits the nq cap here
        for e in ref["entries"]:
            L, beta, N = e["L"], e["beta"], e["nbasis"]
            _close(ho.ho_r2_matrix(L, beta, N), e["r2"], atol=1e-13)
            _close(ho.ho_p2_matrix(L, beta, N), e["p2"], atol=1e-13)
            op = lambda g: ho.ho_operator_matrix(L, beta, N, g)  # noqa: E731
            _close(op(lambda r: r), e["op_r"])
            _close(op(lambda r: np.exp(-(r**2))), e["op_exp_minus_r2"])
            _close(op(lambda r: smearing.smeared_coulomb_G_closed(params, mc, mc, r)), e["op_G_tilde_cc"])
            _close(op(lambda r: smearing.smeared_confinement_S_closed(params, mc, mc, r)), e["op_S_tilde_cc"])
            _close(op(lambda r: smearing.smeared_coulomb_G_closed(params, cu.m1_GeV, cu.m2_GeV, r)), e["op_G_tilde_cu"])
            mom = lambda f: ho.ho_momentum_operator_matrix(L, beta, N, f)  # noqa: E731
            _close(mom(lambda p: np.sqrt(p**2 + mc**2)), e["mom_sqrt_p2_plus_mc2"])
            _close(mom(lambda p: np.sqrt(p**2 + cu.m2_GeV**2)), e["mom_sqrt_p2_plus_mu2"])
            _close(mom(lambda p: np.sqrt(1 + p**2 / np.sqrt((p**2 + mc**2) * (p**2 + mc**2)))), e["mom_A_cc"])
            _close(ho.oscillator_central_matrix(params, cc, L, beta, N), e["central_matrix_cc"])
            _close(ho.oscillator_central_matrix(params, cu, L, beta, N), e["central_matrix_cu"])


def test_spin_matrices(ref, params):
    for e in ref["spin_matrices"]:
        masses = ConstituentMasses(*e["masses_GeV"])
        beta, N = e["beta"], e["nbasis"]
        for key, mat in e["contact_hyperfine"].items():
            L, mult = int(key[1]), int(key[-1])
            _close(ho.ho_contact_hyperfine_matrix(params, masses, L, mult, beta, N), mat, rtol=1e-10)
        sandwich = ho.ho_momentum_sandwich_matrix(
            0, beta, N, masses.m1_GeV, masses.m2_GeV,
            gi_spin_dependent_side_exponent(params.factors.epsilon_c),
            lambda r: smearing.smeared_contact_kernel(params, masses, r), rtol=1e-8,
        )
        _close(sandwich, e["momentum_sandwich_contact_kernel_L0_eps_c"], rtol=1e-10)
        for f in e["fine_structure"]:
            mats = ho.ho_fine_structure_matrices(params, masses, f["L"], f["multiplicity"], f["J"], beta, N)
            for name in ("spin_orbit_vector", "spin_orbit_thomas", "spin_orbit", "tensor", "total"):
                _close(getattr(mats, name), f[name], rtol=1e-10)


def test_quadgk_port():
    value, err = ho.quadgk(lambda x: np.exp(-x) * np.cos(5 * x), 0.0, 20.0, rtol=1e-12)
    exact = (1 - np.exp(-20) * (np.cos(100) - 5 * np.sin(100))) / 26
    assert value == pytest.approx(exact, rel=1e-12)


def test_nq_cap_warning_like_julia():
    """GIModel.jl reaches nq_max for g(r) = r at L = 0 (sqrt(x) is not polynomial) and warns."""
    ho._WARNED_CAP.clear()
    with pytest.warns(RuntimeWarning, match="did not reach rtol"):
        ho.ho_operator_matrix(0, 0.5, 8, lambda r: r)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ho.ho_operator_matrix(1, 0.5, 8, lambda r: r)  # converges before the cap
