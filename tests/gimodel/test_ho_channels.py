"""HO channel solves against GIModel.jl (tests/reference/channels.json, solver ``ho_default``).

Eigenvalues at 1e-7 GeV; converged beta, nbasis and refinement count identical;
native wave coefficients (outer-lobe-positive phase) at 1e-8. The full fixed-sector
sweep is ``slow``; one representative sector per system runs by default.
"""
from __future__ import annotations

import numpy as np
import pytest

from gimodel import (
    ConstituentMasses,
    FineStructureMultiplet,
    OscillatorSolver,
    OscillatorWave,
    channel_solution,
    fixed_channel_solution,
    momentum_expect,
    momentum_overlap,
    wave_mean_squares,
)
from gimodel import ho
from gimodel.ho import OscillatorConvergence

from conftest import load_reference

EIG_TOL = 1e-7
WAVE_TOL = 1e-8
FAST_FIXED = {("cc", "^3P_1"), ("cu", "^1P_1")}


@pytest.fixture(scope="module")
def ref():
    return load_reference("channels.json")["solvers"]["ho_default"]


def _check(sol, ref_solution):
    assert np.max(np.abs(sol.eigenvalues_GeV - ref_solution["eigenvalues_GeV"])) <= EIG_TOL
    cert, rc = sol.convergence, ref_solution["convergence"]
    assert isinstance(cert, OscillatorConvergence)
    assert cert.status == rc["status"]
    assert cert.beta_GeV == rc["beta_GeV"]
    assert cert.nbasis == rc["nbasis"]
    assert cert.refinements == rc["refinements"]
    assert cert.energy_delta_GeV == pytest.approx(rc["energy_delta_GeV"], abs=1e-10)
    assert cert.max_wave_overlap_defect == pytest.approx(rc["max_wave_overlap_defect"], abs=1e-10)
    for wave, rw in zip(sol.waves, ref_solution["waves_native"]):
        assert isinstance(wave, OscillatorWave)
        assert wave.L == rw["L"] and wave.beta == rw["beta"]
        assert np.max(np.abs(wave.coefficients - np.array(rw["coefficients"]))) <= WAVE_TOL


def test_solver_settings_match(ref):
    s = OscillatorSolver()
    r = ref["solver"]
    assert (s.nbasis, s.max_nbasis, s.basis_step) == (r["nbasis"], r["max_nbasis"], r["basis_step"])
    assert s.energy_tolerance_GeV == r["energy_tolerance_GeV"] and s.beta_tolerance_GeV == r["beta_tolerance_GeV"]
    assert list(s.beta_grid) == r["beta_grid"] and s.nlevels_per_channel == r["nlevels_per_channel"]


@pytest.mark.parametrize("system", ["cc", "cu"])
def test_central_channels(ref, params, system):
    entry = ref["systems"][system]
    masses = ConstituentMasses(*entry["masses_GeV"])
    for c in entry["central"]:
        _check(channel_solution(params, masses, c["L"], solver=OscillatorSolver()), c["solution"])


def _fixed_cases(fast: bool):
    cases = []
    for system in ("cc", "cu"):
        for label in ("^1S_0", "^3S_1", "^1P_1", "^3P_0", "^3P_1", "^3P_2", "^1D_2", "^3D_1", "^3D_2", "^3D_3"):
            if ((system, label) in FAST_FIXED) == fast:
                cases.append((system, label))
    return cases


def _run_fixed(ref, params, system, label):
    entry = ref["systems"][system]
    fixed = next(f for f in entry["fixed"] if f["label"] == label)
    multiplet = FineStructureMultiplet(fixed["L_label"], fixed["multiplicity"], fixed["J"])
    sol = fixed_channel_solution(params, ConstituentMasses(*entry["masses_GeV"]), multiplet, solver=OscillatorSolver())
    _check(sol, fixed["solution"])


@pytest.mark.parametrize("system,label", _fixed_cases(True))
def test_fixed_channel(ref, params, system, label):
    _run_fixed(ref, params, system, label)


@pytest.mark.slow
@pytest.mark.parametrize("system,label", _fixed_cases(False))
def test_fixed_channel_full(ref, params, system, label):
    _run_fixed(ref, params, system, label)


def test_unchecked_solver_and_errors(params, mq):
    masses = ConstituentMasses(mq["c"], mq["c"])
    sol = channel_solution(params, masses, 0, solver=OscillatorSolver(converge=False), nlevels=2)
    assert sol.convergence.status == "unchecked" and sol.convergence.nbasis == 24
    with pytest.raises(ValueError):
        channel_solution(params, masses, 0, solver=OscillatorSolver(nbasis=4, max_nbasis=80), nlevels=6)
    with pytest.raises(ArithmeticError, match="endpoint"):
        channel_solution(params, masses, 0, solver=OscillatorSolver(beta_grid=(0.25, 0.3, 0.35)), nlevels=1)


def test_wave_interface_self_consistency():
    rng = np.random.default_rng(1)
    w = OscillatorWave(1, 0.6, rng.normal(size=12) * 0.5 ** np.arange(12))
    assert w.wave_norm() == pytest.approx(1.0, abs=1e-14)
    # Same wave at another beta: quadrature path of radial_overlap.
    other = OscillatorWave(1, 0.61, w.coefficients)
    assert w.radial_overlap(w, lambda r: 1.0) == pytest.approx(1.0, abs=1e-12)
    assert abs(w.radial_overlap(other, lambda r: 1.0)) < 1.0
    assert w.radial_expect(lambda r: r**2) == pytest.approx(
        float(w.coefficients @ ho.ho_r2_matrix(1, 0.6, 12) @ w.coefficients), abs=1e-11
    )
    r2, p2 = wave_mean_squares(w, 1)
    assert p2 == pytest.approx(float(w.coefficients @ ho.ho_p2_matrix(1, 0.6, 12) @ w.coefficients), abs=1e-12)
    mw = w.momentum_wave(1)
    assert momentum_expect(mw, lambda p: 1.0) == pytest.approx(1.0, abs=1e-12)
    mo = momentum_overlap(mw, other.momentum_wave(), lambda p: 1.0)
    assert mo == pytest.approx(w.radial_overlap(other, lambda r: 1.0), abs=1e-8)
    norm_q, _ = ho.quadgk(lambda p: p**2 * mw(p) ** 2, 0.0, w._momentum_cutoff(), rtol=1e-12)
    assert norm_q == pytest.approx(1.0, abs=1e-10)
    flipped = OscillatorWave(1, 0.6, -w.coefficients).fix_outer_phase()
    assert np.allclose(flipped.fix_outer_phase().coefficients, w.fix_outer_phase().coefficients)
    # sandwich with B = 1 (exponent 0) reduces to the plain radial expectation.
    kernel = lambda r: np.exp(-0.3 * r)  # noqa: E731
    assert w.sandwich_expectation(1.0, 1.0, 1, 0.0, kernel) == pytest.approx(w.radial_expect(kernel), rel=1e-8)
