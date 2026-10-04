"""End-to-end spectrum tests: documented GIModel.jl values (FD default solver) and the
structural checks of GIModel.jl test/spectrum.jl."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gimodel import (
    BasisState,
    ConstituentMasses,
    FiniteDifferenceSolver,
    Meson,
    OscillatorSolver,
    SpinTerms,
    add_intra_meson_mixing,
    central_spectrum,
    channel_solution,
    compute_spectrum,
    contact_hyperfine_nonperturbative_states,
    fixed_spectrum,
    physical_components,
    physical_state_amplitude,
    radial_expect,
    radial_overlap,
    radial_wave,
    solve_sector,
    spectrum_levels,
    spectrum_state,
    wave_mean_squares,
    wave_norm,
)

CHARMONIUM_DOC = {
    "1^1S_0": 2.9667, "1^3S_1": 3.0910, "2^1S_0": 3.6254, "2^3S_1": 3.6788,
    "1^1P_1": 3.5151, "1^3P_0": 3.4428, "1^3P_1": 3.5082, "1^3P_2": 3.5481,
    "2^1P_1": 3.9561, "2^3P_0": 3.9163, "2^3P_1": 3.9531, "2^3P_2": 3.9794,
    "1^1D_2": 3.8366, "1^3D_1": 3.8182, "1^3D_2": 3.8373, "1^3D_3": 3.8477,
    "2^1D_2": 4.2075, "2^3D_1": 4.1941, "2^3D_2": 4.2081, "2^3D_3": 4.2165,
}


def test_spectrum_levels():
    levels = spectrum_levels(2)
    assert len(levels) == 2 * (2 + 4 + 4)
    assert all(l.J == 1 for l in levels if l.L_label == "S" and l.multiplicity == 3)
    assert sorted(l.J for l in levels if l.L_label == "P" and l.multiplicity == 3 and l.n == 1) == [0, 1, 2]
    assert len(spectrum_levels(3, L_labels=("S",))) == 6
    with pytest.raises(ValueError):
        spectrum_levels(0)
    with pytest.raises(ValueError):
        spectrum_levels(2, L_labels=("X",))


# --- documented values ------------------------------------------------------------


def test_charmonium_table(charmonium):
    assert [s.label for s in charmonium.states] == list(CHARMONIUM_DOC)
    for label, mass in CHARMONIUM_DOC.items():
        assert round(spectrum_state(charmonium, label).mass_GeV, 4) == pytest.approx(mass, abs=1e-12), label
    eta_c = spectrum_state(charmonium, "1^1S_0")
    assert round(eta_c.central_GeV, 4) == pytest.approx(3.0784)
    assert round(eta_c.contact_shift_GeV, 4) == pytest.approx(-0.1117)
    assert sum(1 for s in charmonium.states if s.mixings) == 4


def test_chi_c_decomposition(charmonium):
    chi1 = spectrum_state(charmonium, "1^3P_1")
    assert (chi1.n, chi1.L, chi1.multiplicity, chi1.J) == (1, "P", 3, 1)
    assert chi1.mass_GeV == pytest.approx(3.5081502109538754, abs=1e-9)
    assert chi1.central_GeV == pytest.approx(3.5238043169507294, abs=1e-9)
    assert chi1.spin_orbit_shift_GeV == pytest.approx(-0.028732806502261746, abs=1e-9)
    assert chi1.tensor_shift_GeV == pytest.approx(0.010230059107934618, abs=1e-9)
    assert chi1.contact_shift_GeV == pytest.approx(0.002848641397473027, abs=1e-9)
    chi2 = spectrum_state(charmonium, "1^3P_2")
    assert chi2.mass_GeV == pytest.approx(3.548136774754569, abs=1e-9)
    assert chi2.spin_orbit_vector_shift_GeV == pytest.approx(0.03362190128723508, abs=1e-9)
    assert chi2.spin_orbit_thomas_shift_GeV == pytest.approx(-0.01070969941038476, abs=1e-9)
    assert chi2.tensor_shift_GeV == pytest.approx(-0.0017649734822792074, abs=1e-9)


def test_jpsi_components_and_eta_c_wave(charmonium):
    comps = {c.basis.label: c.coefficient for c in physical_components(charmonium, "1^3S_1")}
    assert list(comps) == ["1^3S_1", "2^3S_1", "1^3D_1", "2^3D_1"]
    assert round(comps["1^3S_1"], 4) == pytest.approx(0.9999)
    assert round(comps["1^3D_1"], 4) == pytest.approx(-0.0133)
    assert round(comps["2^3D_1"], 4) == pytest.approx(0.0086)
    assert sum(c**2 for c in comps.values()) == pytest.approx(1.0, abs=1e-12)
    eta_c = radial_wave(charmonium, "1^1S_0")
    assert wave_norm(eta_c) == pytest.approx(1.0, abs=1e-12)
    r2, p2 = wave_mean_squares(eta_c, 0)
    assert r2 == pytest.approx(2.1229942234941013, rel=1e-9)
    assert p2 == pytest.approx(1.2899911969232778, rel=1e-9)
    with pytest.raises(ValueError):
        radial_wave(charmonium, "1^3S_1")


def test_open_charm_p_wave_mixing(params, mq):
    """Documented cu 1P example (spectrum_levels(1), P only: a 2x2 same-J block)."""
    D = Meson.from_table(mq, "c", "u")
    fixed = fixed_spectrum(params, D, levels=spectrum_levels(1, L_labels=("P",)))
    unmixed = {s.label: round(s.mass_GeV, 4) for s in fixed.states}
    assert unmixed == {"1^1P_1": 2.4574, "1^3P_0": 2.3948, "1^3P_1": 2.4634, "1^3P_2": 2.5020}
    mixed = add_intra_meson_mixing(fixed)
    d1 = spectrum_state(mixed, "1^1P_1")
    (m,) = d1.mixings
    assert m.mechanism == "antisymmetric_spin_orbit"
    assert m.block_label == "same-J antisymmetric spin-orbit"
    assert m.partner_labels == ["1^1P_1", "1^3P_1"]
    assert m.offdiag_GeV == pytest.approx(0.003742140815267442, abs=1e-11)
    assert np.allclose(m.components, [0.9005661482910976, -0.43471900413041176], atol=1e-9)
    assert np.allclose(m.partner_masses_GeV, [2.4556181687426726, 2.465176803815588], atol=1e-11)
    assert round(spectrum_state(mixed, "1^3P_1").mass_GeV, 4) == pytest.approx(2.4652)
    # contact strength: -4/3 <contact> of the singlet, in MeV
    assert -4 / 3 * d1.contact_shift_GeV * 1000 == pytest.approx(25.064, abs=0.02)
    assert d1.fine_structure_mass_convention == "unequal_mass_same_j_mixed"
    assert d1.corrected.fine_structure_mass_convention == "unequal_mass_equal_share_LdotS"


# --- structure (GIModel.jl test/spectrum.jl) -------------------------------------------

COARSE = FiniteDifferenceSolver(ngrid=120, rmax=12.0)


def test_breakdown_and_tensor_block(params, mq):
    meson = Meson.from_table(mq, "c", "c")
    levels = [BasisState(1, "S", 3, 1), BasisState(2, "S", 3, 1), BasisState(1, "D", 3, 1), BasisState(1, "P", 3, 2)]
    spec = compute_spectrum(params, meson, levels=levels, solver=COARSE)
    for s in spec.states:
        if not s.mixings:
            assert s.mass_GeV == pytest.approx(s.central_GeV + s.contact_shift_GeV + s.fine_structure_shift_GeV, abs=1e-12)
        assert s.fine_structure_shift_GeV == pytest.approx(s.spin_orbit_shift_GeV + s.tensor_shift_GeV, abs=1e-12)
    mixed = [s for s in spec.states if s.mixings]
    assert len(mixed) == 3
    assert all(m.mechanism == "tensor_mixing" for s in mixed for m in s.mixings)
    assert sum(s.mass_GeV for s in mixed) == pytest.approx(sum(s.mixings[-1].unmixed_GeV for s in mixed), abs=1e-10)
    assert all(s.mixings[-1].result is mixed[0].mixings[-1].result for s in mixed)
    assert all(len(physical_components(spec, s)) == 3 for s in mixed)
    comps = physical_components(spec, mixed[0])
    coherent = sum(
        a.coefficient * b.coefficient * radial_overlap(a.wave, b.wave, lambda r: r**2)
        for a in comps
        for b in comps
        if (a.basis.L_label, a.basis.multiplicity, a.basis.J) == (b.basis.L_label, b.basis.multiplicity, b.basis.J)
    )
    assert radial_expect(spec, mixed[0], lambda r: r**2) == pytest.approx(coherent, abs=1e-12)
    assert spectrum_state(spec, 1, "P", 3, 2).label == "1^3P_2"
    with pytest.raises(ValueError):
        spectrum_state(spec, 3, "S", 1, 0)
    with pytest.raises(ValueError):
        compute_spectrum(params, meson, levels=[BasisState(7, "S", 1, 0)], solver=FiniteDifferenceSolver(ngrid=80, rmax=8.0))


def test_same_j_mixing_gated_by_flavor(params, mq):
    pair = [BasisState(1, "P", 1, 1), BasisState(1, "P", 3, 1)]
    us = compute_spectrum(params, Meson.from_table(mq, "u", "s"), levels=pair, solver=COARSE)
    so = [s for s in us.states if any(m.mechanism == "antisymmetric_spin_orbit" for m in s.mixings)]
    assert len(so) == 2
    assert sum(s.mass_GeV for s in so) == pytest.approx(sum(s.mixings[-1].unmixed_GeV for s in so), abs=1e-10)
    assert any(abs(s.mixings[-1].offdiag_GeV) > 0 for s in so)
    cc = compute_spectrum(params, Meson.from_table(mq, "c", "c"), levels=pair, solver=COARSE)
    assert all(not s.mixings for s in cc.states)


def test_central_diagnostic_and_stages(params, mq):
    us = Meson.from_table(mq, "u", "s")
    levels = spectrum_levels(2, L_labels=("S", "P"))
    solver = FiniteDifferenceSolver(ngrid=250, rmax=16.0)
    central = central_spectrum(params, us, levels=levels, solver=solver)
    assert all(k.multiplicity == 0 for k in central.computation.channel_cache)
    corrected = fixed_spectrum(params, us, levels=levels, solver=solver)
    assert all(k.multiplicity != 0 for k in corrected.computation.channel_cache)
    for s in corrected.states:
        assert s.mass_GeV == pytest.approx(s.central_GeV + s.contact_shift_GeV + s.fine_structure_shift_GeV)
    mixed = add_intra_meson_mixing(corrected)
    direct = compute_spectrum(params, us, levels=levels, solver=solver)
    assert [s.mass_GeV for s in direct.states] == pytest.approx([s.mass_GeV for s in mixed.states], abs=1e-13)
    for s in mixed.states:
        if s.mixings:
            assert s.mixings[-1].unmixed_GeV == s.corrected.mass_GeV
    bare = add_intra_meson_mixing(
        fixed_spectrum(params, us, levels=levels, solver=solver, terms=SpinTerms(contact_hyperfine=False, fine_structure=False))
    )
    assert all(s.mass_GeV == s.central_GeV and not s.mixings for s in bare.states)
    assert all(s.fine_structure_mass_convention == "disabled" for s in bare.states)
    with pytest.raises(TypeError):
        add_intra_meson_mixing(central)


def test_central_spectrum_degenerate_multiplet(params, mq):
    spec = central_spectrum(params, Meson.from_table(mq, "c", "c"), levels=spectrum_levels(1, L_labels=("P",)))
    values = {round(s.central_GeV, 4) for s in spec.states}
    assert values == {3.5230}


def test_nonperturbative_contact_states(params, mq):
    masses = ConstituentMasses(mq["q"], mq["q"])
    s1 = contact_hyperfine_nonperturbative_states(params, masses, "S", 1, 2)
    s3 = contact_hyperfine_nonperturbative_states(params, masses, "S", 3, 2)
    assert radial_expect(s1.radial_wave(1), lambda r: r**2) < radial_expect(s3.radial_wave(1), lambda r: r**2)
    assert s1.eigenvalues_GeV[0] < s3.eigenvalues_GeV[0]


def test_accessors(charmonium):
    level = BasisState(1, "S", 1, 0)
    assert np.array_equal(radial_wave(charmonium, level).u, radial_wave(charmonium, "1^1S_0").u)
    assert physical_state_amplitude(lambda c: 1.0, charmonium, level) == 1.0
    w1, w2 = radial_wave(charmonium, "1^1S_0"), radial_wave(charmonium, "2^1S_0")
    assert abs(radial_overlap(w1, w2, lambda r: 1.0)) < 1e-12
    # outermost lobe positive
    for s in charmonium.states:
        u = radial_wave(charmonium, s.corrected).u
        assert u[np.nonzero(np.abs(u) > 0.2 * np.abs(u).max())[0][-1]] > 0


def test_solve_sector_ordering(params, mq):
    cc = solve_sector(params, mq["c"], maxn=4, solver=FiniteDifferenceSolver(ngrid=250, rmax=20.0))
    assert cc[(1, "S")] < cc[(2, "S")] < cc[(3, "S")]
    assert cc[(1, "S")] < cc[(1, "P")] < cc[(1, "D")]


def test_krylov_matches_full(params, mq):
    m = ConstituentMasses(mq["c"], mq["c"])
    for kinetic in ("relativistic", "nonrelativistic"):
        full = channel_solution(params, m, 0, nlevels=3, solver=FiniteDifferenceSolver(ngrid=120, rmax=16.0, kinetic=kinetic))
        kry = channel_solution(
            params, m, 0, nlevels=3, solver=FiniteDifferenceSolver(ngrid=120, rmax=16.0, kinetic=kinetic, eigensolver="krylov")
        )
        assert np.allclose(kry.eigenvalues_GeV, full.eigenvalues_GeV, rtol=1e-10, atol=1e-10)


def test_oscillator_solver_is_ported(params, mq):
    # Full HO validation against GIModel.jl lives in tests/test_ho_*.py.
    from gimodel import OscillatorWave, channel_solution

    meson = Meson.from_table(mq, "c", "c")
    sol = channel_solution(params, meson.constituent_masses, 0, solver=OscillatorSolver(converge=False), nlevels=1)
    assert isinstance(sol.radial_wave(1), OscillatorWave)
    assert sol.convergence.status == "unchecked"


def test_runtime_one_meson(params, mq):
    from gimodel import fd

    fd.clear_caches()
    t0 = time.perf_counter()
    compute_spectrum(params, Meson.from_table(mq, "b", "s"))
    assert time.perf_counter() - t0 < 10.0
