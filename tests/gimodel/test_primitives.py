"""Unit tests of the model primitives (mirrors GIModel.jl test/spin_kernels.jl,
central_potentials.jl, parameters_and_core.jl, fine_structure.jl)."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from gimodel import (
    ALPHA_COEFFS,
    ALPHA_GAMMAS,
    BasisState,
    CentralPotentialMethod,
    ConstituentMasses,
    FiniteDifferenceSolver,
    HeavyQuark,
    LightQuark,
    Meson,
    MixingBlock,
    MixingResult,
    OscillatorSolver,
    StrangeQuark,
    alpha_s_q,
    alpha_s_r,
    central_potential_method,
    charge,
    contact_smearing_sigma,
    diagonalize_mixing_block,
    is_equal_flavor,
    l_dot_s,
    load_parameters,
    orbital_angular_momentum,
    orbital_label,
    reduced_mass,
    same_j_mixing,
    spin_dot,
    tensor_triplet_lj,
    tensor_triplet_offdiag_same_j,
)
from gimodel import coupling, fd, smearing
from gimodel.parameters import default_parameters_path


# --- parameters ---------------------------------------------------------------


def test_parameter_loading(params, mq):
    assert mq["c"] == pytest.approx(1.628)
    assert mq["b"] == pytest.approx(4.977)
    assert mq["u"] == mq["d"] == mq["q"] == pytest.approx(0.220)
    assert params.potential.b == pytest.approx(0.18)
    assert params.potential.c == pytest.approx(-0.253)
    assert params.central is CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH
    f = params.factors
    assert (f.epsilon_c, f.epsilon_t, f.epsilon_so_vector, f.epsilon_so_scalar) == pytest.approx(
        (-0.168, 0.025, -0.035, 0.055)
    )
    assert f.contact_momentum_sandwich and f.fine_structure_momentum_sandwich and f.fine_structure_smeared_kernels
    assert params.fine_structure.enabled
    assert params.annihilation.s1_A == pytest.approx(2.5)


def test_parameter_variations_by_replace(params):
    varied = replace(params, potential=replace(params.potential, b=0.19), smearing=replace(params.smearing, s=1.6))
    assert varied.potential.b == 0.19 and varied.potential.c == params.potential.c
    assert varied.smearing.s == 1.6 and varied.central is params.central
    assert hash(varied) != hash(params)


def test_strict_parameter_files(tmp_path):
    text = default_parameters_path().read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("[fine_structure]", "[fine_structure]\nk_tensor = 1.0"))
    with pytest.raises(ValueError, match="unknown or inactive"):
        load_parameters(bad)
    bad.write_text(text.replace("contact_momentum_sandwich = true", "contact_momentum_sandwich = 1"))
    with pytest.raises(ValueError, match="boolean"):
        load_parameters(bad)
    bad.write_text(text + "\n[extra]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown parameter section"):
        load_parameters(bad)


def test_central_method_names():
    assert central_potential_method("appendix_a_momentum_sandwich") is CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH
    assert central_potential_method("pointwise") is CentralPotentialMethod.POINTWISE
    with pytest.raises(ValueError):
        central_potential_method("appendix_a_typo")


def test_orbital_labels():
    assert [orbital_angular_momentum(x) for x in "SPDFG"] == [0, 1, 2, 3, 4]
    assert orbital_label(2) == "D"
    with pytest.raises(ValueError):
        orbital_label(7)


# --- quarks / mesons -------------------------------------------------------------


def test_mesons_and_quarks(mq):
    cc = Meson.from_table(mq, "c", "c")
    assert is_equal_flavor(cc) and reduced_mass(cc) == pytest.approx(mq["c"] / 2)
    bu = Meson.from_table(mq, "b", "u")
    assert not is_equal_flavor(bu) and bu.constituent_masses.m2_GeV == pytest.approx(mq["u"])
    assert Meson.from_table(mq, "n", "n") == Meson.from_table(mq, "q", "q")
    with pytest.raises(ValueError):
        Meson.from_table(mq, "t", "t")
    with pytest.raises(ValueError):
        ConstituentMasses(-1.0, 1.0)
    with pytest.raises(ValueError):
        ConstituentMasses(0.2, math.inf)
    c = HeavyQuark(1.628, "c", "up")
    assert charge(c) == pytest.approx(2 / 3) and charge(StrangeQuark(0.419)) == pytest.approx(-1 / 3)
    with pytest.raises(TypeError):
        charge(LightQuark(0.22))
    assert Meson.from_quarks(c, c) == Meson("c", "c", ConstituentMasses(1.628, 1.628))


# --- coupling and smearing --------------------------------------------------------


def test_alpha_profiles():
    assert alpha_s_r(0.0) == 0.0
    assert alpha_s_r(1e6) == pytest.approx(sum(ALPHA_COEFFS))
    assert alpha_s_q(0.0) == pytest.approx(sum(ALPHA_COEFFS))
    r = np.array([0.1, 1.0, 3.0])
    assert np.allclose(alpha_s_r(r), [alpha_s_r(x) for x in r])


@pytest.mark.parametrize("r0", [0.08, 0.15, 0.4, 1.2, 3.0])
def test_running_coulomb_derivatives(r0):
    d = 1e-5 * max(1.0, r0)
    G = coupling.coulomb_G_running
    num1 = (G(r0 + d) - G(r0 - d)) / (2 * d)
    num2 = (G(r0 + d) - 2 * G(r0) + G(r0 - d)) / d**2
    assert coupling.coulomb_G_prime_running(r0) == pytest.approx(num1, rel=1e-7)
    assert coupling.coulomb_G_second_running(r0) == pytest.approx(num2, rel=1e-4)
    a = alpha_s_r
    assert coupling.alpha_s_prime_r(r0) == pytest.approx((a(r0 + d) - a(r0 - d)) / (2 * d), rel=1e-7)
    assert coupling.tensor_kernel_coulomb_running(r0) == pytest.approx(
        coupling.coulomb_G_prime_running(r0) / r0 - coupling.coulomb_G_second_running(r0)
    )


def test_pointwise_central_is_G_plus_S(params):
    for r0 in (0.15, 0.4, 1.2, 3.0):
        assert coupling.central_potential(r0, params) == pytest.approx(
            coupling.static_coulomb_G(r0) + coupling.static_confinement_S(r0, params), rel=1e-12
        )


def test_contact_sigma_formula(params, mq):
    for m1, m2 in [(mq["c"], mq["c"]), (mq["c"], mq["u"]), (mq["b"], mq["s"])]:
        mf = 4 * m1 * m2 / (m1 + m2) ** 2
        expected = math.sqrt(1.8**2 * (0.5 + 0.5 * mf**4) + 1.55**2 * (2 * m1 * m2 / (m1 + m2)) ** 2)
        assert contact_smearing_sigma(params, m1, m2) == pytest.approx(expected, rel=1e-14)
        assert contact_smearing_sigma(params, ConstituentMasses(m1, m2)) == contact_smearing_sigma(params, m2, m1)


def test_smeared_closed_forms_small_r_and_derivatives(params, mq):
    mc = mq["c"]
    G = lambda x: smearing.smeared_coulomb_G_closed(params, mc, mc, x)
    S = lambda x: smearing.smeared_confinement_S_closed(params, mc, mc, x)
    assert G(0.0) == pytest.approx(G(1e-10), rel=1e-12)
    assert S(0.0) == pytest.approx(S(1e-10), rel=1e-6)
    r0, d = 1.4, 1e-4
    assert smearing.smeared_coulomb_G_prime_closed(params, mc, mc, r0) == pytest.approx(
        (G(r0 + d) - G(r0 - d)) / (2 * d), rel=1e-6
    )
    assert smearing.smeared_coulomb_G_second_closed(params, mc, mc, r0) == pytest.approx(
        (G(r0 + d) - 2 * G(r0) + G(r0 - d)) / d**2, rel=1e-5
    )
    assert smearing.smeared_confinement_S_prime_closed(params, mc, mc, r0) == pytest.approx(
        (S(r0 + d) - S(r0 - d)) / (2 * d), rel=1e-7
    )
    assert smearing.smeared_confinement_S_prime_closed(params, mc, mc, 0.0) == 0.0
    assert smearing.tensor_kernel_smeared_coulomb(params, mc, mc, r0) == pytest.approx(
        smearing.smeared_coulomb_G_prime_closed(params, mc, mc, r0) / r0
        - smearing.smeared_coulomb_G_second_closed(params, mc, mc, r0)
    )
    # vectorised and scalar evaluation agree; masses-object call form works
    r = np.array([0.0, 0.3, 2.0])
    assert np.allclose(G(r), [G(x) for x in r])
    assert np.allclose(smearing.smeared_coulomb_G_closed(params, ConstituentMasses(mc, mc), r), G(r))


def test_closed_form_coulomb_matches_3d_convolution(params, mq):
    m = mq["c"]
    sigma = contact_smearing_sigma(params, m, m)

    def convolved(R):
        total = 0.0
        for a, g in zip(ALPHA_COEFFS, ALPHA_GAMMAS):
            def integrand(rp):
                gv = -8 * a * g / (3 * math.sqrt(math.pi)) if rp == 0 else -4 * a * erf(g * rp) / (3 * rp)
                pre = sigma / (math.sqrt(math.pi) * R)
                return pre * rp * (math.exp(-((sigma * (R - rp)) ** 2)) - math.exp(-((sigma * (R + rp)) ** 2))) * gv

            total += quad(integrand, 0.0, R + 40 / sigma, epsabs=1e-13, epsrel=1e-11, limit=200)[0]
        return total

    for R in (0.2, 0.8, 2.0):
        assert smearing.smeared_coulomb_G_closed(params, m, m, R) == pytest.approx(convolved(R), rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("sigma", [0.2, 0.5, 1.0, 2.0, 5.0])
def test_contact_regulator_is_3d_normalised(sigma):
    rmax, n = 12.0 / sigma, 4000
    h = rmax / n
    r = (np.arange(1, n + 1) - 0.5) * h
    assert np.sum(4 * np.pi * r**2 * smearing.delta_sigma_3d(r, sigma) * h) == pytest.approx(1.0, abs=2e-6)


def test_contact_kernel_is_laplacian_of_smeared_coulomb(params, mq):
    """A15: the contact density is the Laplacian of the smeared Coulomb potential,
    lap G~ = (16 pi/3) sum_k alpha_k delta_tau_k."""
    masses = ConstituentMasses(mq["c"], mq["u"])
    G = lambda x: smearing.smeared_coulomb_G_closed(params, masses, x)
    for r0 in (0.3, 0.9, 1.7):
        d = 1e-3
        lap = (G(r0 + d) - 2 * G(r0) + G(r0 - d)) / d**2 + (2 / r0) * (G(r0 + d) - G(r0 - d)) / (2 * d)
        assert lap == pytest.approx(16 * math.pi / 3 * smearing.smeared_contact_kernel(params, masses, r0), rel=1e-5)


# --- spin algebra -------------------------------------------------------------------


def test_spin_algebra():
    assert l_dot_s(1, 1, 0) == -2.0 and l_dot_s(1, 1, 1) == -1.0 and l_dot_s(1, 1, 2) == 1.0
    assert spin_dot(1) == -0.75 and spin_dot(3) == 0.25
    for L in range(1, 5):
        js = range(L - 1, L + 2)
        assert sum((2 * J + 1) * l_dot_s(L, 1, J) for J in js) == pytest.approx(0, abs=1e-12)
        assert sum((2 * J + 1) * tensor_triplet_lj(L, J, 1) for J in js) == pytest.approx(0, abs=1e-12)
    assert (tensor_triplet_lj(1, 0, 1), tensor_triplet_lj(1, 1, 1), tensor_triplet_lj(1, 2, 1)) == pytest.approx(
        (-4.0, 2.0, -0.4)
    )
    assert tensor_triplet_offdiag_same_j(1, 1) == pytest.approx(6 * math.sqrt(2) / 3)
    assert tensor_triplet_offdiag_same_j(0, 1) == 0.0 and tensor_triplet_offdiag_same_j(1, 0) == 0.0


# --- mesh -----------------------------------------------------------------------------


def test_radial_grid_and_p2():
    r, h = fd.radial_grid(12, 3.0)
    assert h == pytest.approx(3.0 / 13) and np.allclose(r, h * np.arange(1, 13))
    lam, V = fd.p2_eigen(1, r, h)
    d, e = fd.p2_operator(1, r, h)
    P = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    assert np.allclose(V @ np.diag(lam) @ V.T, P)
    assert np.all(np.diff(lam) > 0)


# --- basis states and mixing blocks ------------------------------------------------------


def test_basis_state_validation():
    assert BasisState(1, "P", 3, 0).label == "1^3P_0"
    for bad in [(0, "S", 3, 1), (1, "S", 2, 1), (1, "S", 3, -1), (1, "P", 3, 3), (1, "S", 3, 0), (1, "P", 1, 0), (1, "s", 1, 0)]:
        with pytest.raises(ValueError):
            BasisState(*bad)
    assert BasisState(1, "S", 1, 0, flavors=("n", "c")).flavors == ("q", "c")


def test_mixing_blocks():
    basis = (BasisState(1, "S", 3, 1), BasisState(1, "D", 3, 1), BasisState(2, "S", 3, 1))
    block = MixingBlock("smoke", basis, [[3.0, 0.01, 0.0], [0.01, 3.2, 0.02], [0.0, 0.02, 3.6]], mechanism="test")
    res = diagonalize_mixing_block(block)
    assert np.all(np.diff(res.masses) > 0)
    assert np.allclose(res.vectors.T @ res.vectors, np.eye(3), atol=1e-12)
    assert np.all(np.diag(res.vectors) > 0)  # assigned anchors are positive
    with pytest.raises(ValueError):
        MixingBlock("empty", (), np.zeros((0, 0)))
    with pytest.raises(ValueError):
        MixingBlock("asym", basis[:2], [[1.0, 0.2], [0.1, 2.0]])
    with pytest.raises(ValueError):
        MixingBlock("dup", (BasisState(1, "S", 3, 1, label="a"), BasisState(1, "S", 3, 1, label="b")), np.eye(2))
    with pytest.raises(ValueError):
        MixingResult(block, res.masses[::-1], res.vectors)
    with pytest.raises(ValueError):
        MixingResult(block, res.masses, 2 * res.vectors)
    mix = same_j_mixing(3.5, 3.6, 0.0)
    assert mix.masses == pytest.approx([3.5, 3.6]) and mix.theta_deg == pytest.approx(0.0, abs=1e-12)


# --- solver options ----------------------------------------------------------------------


def test_solver_options():
    with pytest.raises(ValueError):
        FiniteDifferenceSolver(ngrid=1)
    with pytest.raises(ValueError):
        FiniteDifferenceSolver(kinetic="fast")
    assert FiniteDifferenceSolver() == FiniteDifferenceSolver()
    ho = OscillatorSolver()
    assert ho.max_nbasis == 80 and len(ho.beta_grid) == 22 and ho.beta_grid[-1] == pytest.approx(2.35)
    assert OscillatorSolver() == OscillatorSolver() and hash(OscillatorSolver()) == hash(OscillatorSolver())
    assert OscillatorSolver() != OscillatorSolver(nbasis=32)
    with pytest.raises(ValueError):
        OscillatorSolver(beta_grid=(0.5, 0.7))
