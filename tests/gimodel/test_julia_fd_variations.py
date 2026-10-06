"""Layer f (tests/reference/variations.json): FD spectra (spectrum_levels(1), and the
S/P/D/F run) for cc and cs under one-at-a-time parameter, quark-mass and SpinTerms
variations. Every run is rebuilt from its *stored* full parameter set (via
dataclasses.replace on the defaults), stored masses and stored SpinTerms, recomputed
and compared state by state at the spectrum tolerances.

The baseline runs and the S/P/D/F run are in the default selection; the other runs are
marked ``slow`` (each recomputes a full FD spectrum, ~0.7 s).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gimodel import FiniteDifferenceSolver, SpinTerms

import julia_ref as jr
from julia_ref import ENERGY, check

REF = jr.load_or_none("variations.json")
FAST_RUNS = {"baseline", "L_labels=SPDF"}


def _run_id(run) -> str:
    return f"{run['name']}-{run['system']}"


def _marks(run):
    return [] if run["name"] in FAST_RUNS else [pytest.mark.slow]


RUNS = [pytest.param(k, id=_run_id(run), marks=_marks(run)) for k, run in enumerate(REF["runs"])] if REF else []
STATE_CASES = (
    [
        pytest.param(k, i, id=f"{_run_id(run)}-{s['label']}", marks=_marks(run))
        for k, run in enumerate(REF["runs"])
        for i, s in enumerate(run["states"])
    ]
    if REF
    else []
)


@pytest.fixture(scope="module")
def ref():
    return jr.load("variations.json")


def _flavors(system: str) -> tuple[str, str]:
    return system[0], system[1]


def _inputs(k: int):
    ref = jr.load("variations.json")
    run = ref["runs"][k]
    params = jr.params_from_dict(run["parameters"])
    meson = jr.meson_from_system(_flavors(run["system"]), run["masses_GeV"])
    levels = jr.levels_from_spec(run["levels_spec"])
    solver = jr.solver_from_dict(ref["solver"])
    terms = jr.terms_from_dict(run["terms"])
    return run, params, meson, levels, solver, terms


def _spectrum(k: int):
    run, params, meson, levels, solver, terms = _inputs(k)
    return run, jr.cached_spectrum(params, meson, levels, solver, terms)


# --- the stored inputs describe exactly one change relative to the defaults -------------------


def _diff_fields(a, b, prefix=""):
    out = []
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if dataclasses.is_dataclass(x):
            out += _diff_fields(x, y, f"{prefix}{f.name}.")
        elif x != y:
            out.append(f"{prefix}{f.name}")
    return out


EXPECTED_CHANGE = {
    "baseline": [],
    "b=0.20": ["potential.b"],
    "c=-0.3": ["potential.c"],
    "sigma0=1.6": ["smearing.sigma0"],
    "s=1.7": ["smearing.s"],
    "eps_c=-0.1": ["factors.epsilon_c"],
    "eps_t=0.05": ["factors.epsilon_t"],
    "eps_so_v=-0.02": ["factors.epsilon_so_vector"],
    "eps_so_s=0.08": ["factors.epsilon_so_scalar"],
}


@pytest.mark.parametrize("k", [pytest.param(k, id=_run_id(r)) for k, r in enumerate(REF["runs"])] if REF else [])
def test_run_inputs(ref, k):
    run, params, meson, levels, solver, terms = _inputs(k)
    base, mq = jr.defaults()
    assert solver == FiniteDifferenceSolver()
    assert len(levels) == len(run["states"])
    changed = _diff_fields(params, base)
    name = run["name"]
    if name in EXPECTED_CHANGE:
        assert changed == EXPECTED_CHANGE[name]
        assert terms == SpinTerms()
    elif name.startswith("terms:"):
        assert changed == []
        field, value = name[len("terms:"):].split("=")
        assert terms == dataclasses.replace(SpinTerms(), **{field: value == "true"})
    else:
        assert changed == [] and terms == SpinTerms()
    default_masses = [mq[f] for f in _flavors(run["system"])]
    if name.startswith("m_"):
        flavor, value = name[2:].split("=")
        expect = [float(value) if f == flavor else mq[f] for f in _flavors(run["system"])]
        assert run["masses_GeV"] == expect
    else:
        assert run["masses_GeV"] == default_masses


@pytest.mark.parametrize("k", RUNS)
def test_run_structure(k):
    run, spec = _spectrum(k)
    assert [s.label for s in spec.states] == [s["label"] for s in run["states"]]
    assert [len(s.mixings) for s in spec.states] == [len(s["mixings"]) for s in run["states"]]
    assert [[m.mechanism for m in s.mixings] for s in spec.states] == [
        [m["mechanism"] for m in s["mixings"]] for s in run["states"]
    ]


@pytest.mark.parametrize("k,i", STATE_CASES)
def test_state(ref, k, i):
    run, spec = _spectrum(k)
    state, sref = spec.states[i], run["states"][i]
    what = f"{_run_id(run)} {sref['label']}"
    jr.compare_contributions(state, sref, what)
    jr.compare_mixings(state, sref, what)
    jr.compare_physical_components(spec, state, sref, what)
    jr.compare_precursor_wave(spec, state, sref, ref["wave_sample_r"], what)


# --- every variation actually changes something, and the switches act as documented ------------


@pytest.mark.parametrize(
    "k",
    [pytest.param(k, id=_run_id(r), marks=_marks(r)) for k, r in enumerate(REF["runs"]) if r["name"] != "baseline"]
    if REF
    else [],
)
def test_variation_moves_masses_like_julia(ref, k):
    """The Python mass shift relative to the baseline equals Julia's (to 1e-9 GeV) and is non-zero."""
    run, spec = _spectrum(k)
    base_k = next(
        j for j, r in enumerate(ref["runs"]) if r["name"] == "baseline" and r["system"] == run["system"]
    )
    base_run, base_spec = _spectrum(base_k)
    base_by_label = {s.label: s for s in base_spec.states}
    base_ref = {s["label"]: s for s in base_run["states"]}
    shifts, ref_shifts = [], []
    for s, sref in zip(spec.states, run["states"]):
        if s.label in base_by_label:
            shifts.append(s.mass_GeV - base_by_label[s.label].mass_GeV)
            ref_shifts.append(sref["mass_GeV"] - base_ref[s.label]["mass_GeV"])
    check(shifts, ref_shifts, ENERGY, f"{_run_id(run)} shifts vs baseline", record=False)
    name = run["name"]
    inert = (name == "terms:same_j_spin_orbit=false" and run["system"] == "cc") or (  # equal flavor
        name.startswith("m_") and name[2] not in run["system"]  # varied quark absent
    )
    if inert:
        assert np.max(np.abs(shifts)) == 0.0
    elif run["name"] != "L_labels=SPDF":
        assert np.max(np.abs(shifts)) > 1e-6


@pytest.mark.parametrize(
    "k",
    [pytest.param(k, id=_run_id(r), marks=_marks(r)) for k, r in enumerate(REF["runs"]) if r["name"].startswith("terms:")]
    if REF
    else [],
)
def test_switched_off_terms_vanish(k):
    run, spec = _spectrum(k)
    terms = jr.terms_from_dict(run["terms"])
    for s in spec.states:
        if not terms.contact_hyperfine:
            assert s.contact_shift_GeV == 0.0
        if not terms.fine_structure:
            assert s.fine_structure_shift_GeV == 0.0 and s.fine_structure_mass_convention == "disabled"
            assert not s.mixings  # mixing requires stage-1 fine structure
        mechanisms = {m.mechanism for m in s.mixings}
        if not terms.tensor:
            assert "tensor_mixing" not in mechanisms
        if not terms.same_j_spin_orbit:
            assert "antisymmetric_spin_orbit" not in mechanisms
