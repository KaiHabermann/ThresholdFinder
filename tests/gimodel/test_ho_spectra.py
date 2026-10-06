"""HO spectra against GIModel.jl (tests/reference/spectra_ho.json, spot_checks.json).

``compute_spectrum(params, Meson(mq, a, b); levels=spectrum_levels(2), solver=OscillatorSolver())``.
Every contribution is compared at 1e-6 GeV, mixing components at 1e-6, and every
cached fixed-sector certificate (beta, nbasis) exactly. The full set of 11 systems
is marked ``slow``; cc, cs and bc run by default.
"""
from __future__ import annotations

import numpy as np
import pytest

from gimodel import Meson, OscillatorSolver, compute_spectrum, physical_components, spectrum_levels, spectrum_state

from conftest import load_reference

GEV_TOL = 1e-6
COMPONENT_TOL = 1e-6
CONTRIBUTIONS = (
    "mass_GeV",
    "corrected_mass_GeV",
    "central_GeV",
    "contact_shift_GeV",
    "spin_orbit_vector_shift_GeV",
    "spin_orbit_thomas_shift_GeV",
    "spin_orbit_shift_GeV",
    "tensor_shift_GeV",
    "fine_structure_shift_GeV",
)
FAST_SYSTEMS = ("cc", "cs", "bc")
ALL_SYSTEMS = ("cc", "bb", "cu", "uc", "cs", "us", "qq", "ss", "bu", "bs", "bc")


@pytest.fixture(scope="module")
def ref():
    return load_reference("spectra_ho.json")


def _value(state, name):
    if name == "corrected_mass_GeV":
        return state.corrected.mass_GeV
    return getattr(state, name)


def compare_spectrum(spec, entry) -> dict[str, float]:
    """Max deviations of a Python HO spectrum from one ``spectra_ho.json`` system entry."""
    dev = {"contributions": 0.0, "components": 0.0, "offdiag": 0.0, "certificate_mismatches": 0, "beta_ties": 0}
    states = {s.label: s for s in spec.states}
    assert [s.label for s in spec.states] == [s["label"] for s in entry["states"]]
    for rs in entry["states"]:
        state = states[rs["label"]]
        for name in CONTRIBUTIONS:
            dev["contributions"] = max(dev["contributions"], abs(_value(state, name) - rs[name]))
        assert state.fine_structure_mass_convention == rs["fine_structure_mass_convention"]
        assert len(state.mixings) == len(rs["mixings"])
        for mx, rmx in zip(state.mixings, rs["mixings"]):
            assert mx.partner_labels == rmx["partner_labels"]
            dev["components"] = max(dev["components"], float(np.max(np.abs(mx.components - rmx["components"]))))
            dev["contributions"] = max(
                dev["contributions"], float(np.max(np.abs(mx.partner_masses_GeV - rmx["partner_masses_GeV"])))
            )
            dev["offdiag"] = max(dev["offdiag"], abs(mx.offdiag_GeV - rmx["offdiag_GeV"]))
        comps = physical_components(spec, state)
        assert [c.basis.label for c in comps] == [c["label"] for c in rs["physical_components"]]
        for c, rc in zip(comps, rs["physical_components"]):
            dev["components"] = max(dev["components"], abs(c.coefficient - rc["coefficient"]))
    cache = spec.computation.channel_cache
    for rsec in entry["sector_solutions"]:
        sol = next(
            v for k, v in cache.items()
            if (k.L_label, k.multiplicity, k.J) == (rsec["L_label"], rsec["multiplicity"], rsec["J"])
        )
        cert, rcert = sol.convergence, rsec["certificate"]
        if cert.nbasis != rcert["nbasis"] or cert.status != rcert["status"] or cert.refinements != rcert["refinements"]:
            dev["certificate_mismatches"] += 1
        elif cert.beta_GeV != rcert["beta_GeV"]:
            # Rounding tie: the golden-section search visits the same beta sequence, but the
            # last two objectives can be equal to ~1e-15 GeV, so either point may win (seen for
            # bb ^3D_3). Accept only a tie: both betas within the search tolerance and every
            # eigenvalue equal far below the energy tolerance.
            tie = abs(cert.beta_GeV - rcert["beta_GeV"]) <= OscillatorSolver().beta_tolerance_GeV and float(
                np.max(np.abs(sol.eigenvalues_GeV - rsec["eigenvalues_GeV"]))
            ) <= 1e-12
            dev["beta_ties"] += 1
            if not tie:
                dev["certificate_mismatches"] += 1
        dev["contributions"] = max(
            dev["contributions"], float(np.max(np.abs(sol.eigenvalues_GeV - rsec["eigenvalues_GeV"])))
        )
    return dev


def _run(ref, params, mq, system):
    entry = ref["systems"][system]
    assert entry["error"] is None
    meson = Meson.from_table(mq, *entry["flavors"])
    spec = compute_spectrum(params, meson, levels=spectrum_levels(2), solver=OscillatorSolver())
    dev = compare_spectrum(spec, entry)
    assert dev["certificate_mismatches"] == 0, dev
    assert dev["contributions"] <= GEV_TOL, dev
    assert dev["offdiag"] <= GEV_TOL, dev
    assert dev["components"] <= COMPONENT_TOL, dev
    return spec


@pytest.mark.parametrize("system", FAST_SYSTEMS)
def test_ho_spectrum(ref, params, mq, system):
    _run(ref, params, mq, system)


@pytest.mark.slow
@pytest.mark.parametrize("system", [s for s in ALL_SYSTEMS if s not in FAST_SYSTEMS])
def test_ho_spectrum_full(ref, params, mq, system):
    _run(ref, params, mq, system)


def test_ho_spot_checks(params, mq):
    """Documented GIModel.jl HO numbers (README): channel_solution and 1S masses."""
    spot = {c["name"]: c for c in load_reference("spot_checks.json")["checks"]}
    meson = Meson.from_table(mq, "c", "c")
    from gimodel import channel_solution

    sol = channel_solution(params, meson.constituent_masses, 0, solver=OscillatorSolver())
    for i, documented in enumerate((3.0645236910656135, 3.666192134948986, 4.090957384190697)):
        computed = spot[f"cc HO channel_solution L=0 eigenvalue {i + 1}"]["computed"]
        assert sol.eigenvalues_GeV[i] == pytest.approx(documented, abs=1e-10)
        assert sol.eigenvalues_GeV[i] == pytest.approx(computed, abs=1e-12)
    spec = compute_spectrum(params, meson, levels=spectrum_levels(1, L_labels=("S",)), solver=OscillatorSolver())
    assert round(spectrum_state(spec, "1^1S_0").mass_GeV, 4) == 2.9671
    assert round(spectrum_state(spec, "1^3S_1").mass_GeV, 4) == 3.0914
