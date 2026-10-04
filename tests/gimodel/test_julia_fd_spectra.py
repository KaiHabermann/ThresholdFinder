"""Layer e (tests/reference/spectra_fd.json): compute_spectrum with the default
FiniteDifferenceSolver and spectrum_levels(2) for 11 flavor systems, compared state by
state with GIModel.jl.

Per state: every mass contribution (1e-9 GeV), every mixing step (mechanism, block,
partner labels, eigenvector column; block matrix 1e-10; components and vectors 1e-8
with exact signs), the flattened physical components (1e-8, exact signs) and the
pre-mixing radial wave sampled on the 81-point plotting grid (1e-7, same phase).
"""
from __future__ import annotations

import numpy as np
import pytest

from gimodel import is_equal_flavor, spectrum_state
from gimodel.mixing import ANTISYMMETRIC_SPIN_ORBIT, TENSOR_MIXING

import julia_ref as jr
from julia_ref import ENERGY, MIXING, check

REF = jr.load_or_none("spectra_fd.json")
SYSTEMS = list(REF["systems"]) if REF else []
STATE_CASES = (
    [pytest.param(sys, i, id=f"{sys}-{s['label']}") for sys in SYSTEMS for i, s in enumerate(REF["systems"][sys]["states"])]
    if REF
    else []
)


@pytest.fixture(scope="module")
def ref():
    return jr.load("spectra_fd.json")


def _spectrum(sys: str):
    ref = jr.load("spectra_fd.json")
    entry = ref["systems"][sys]
    params = jr.params_from_dict(ref["metadata"]["parameters"])
    meson = jr.meson_from_system(entry["flavors"], entry["masses_GeV"])
    return jr.cached_spectrum(
        params,
        meson,
        jr.levels_from_dicts(ref["levels"]),
        jr.solver_from_dict(ref["solver"]),
        jr.terms_from_dict(ref["terms"]),
    )


def _state(sys: str, i: int):
    ref = jr.load("spectra_fd.json")
    sref = ref["systems"][sys]["states"][i]
    spec = _spectrum(sys)
    return spec, spec.states[i], sref, f"{sys} {sref['label']}"


# --- file-level structure ------------------------------------------------------------------


def test_reference_setup(ref):
    from gimodel import FiniteDifferenceSolver, SpinTerms, spectrum_levels

    assert jr.solver_from_dict(ref["solver"]) == FiniteDifferenceSolver()
    assert jr.terms_from_dict(ref["terms"]) == SpinTerms()
    assert [l.label for l in jr.levels_from_dicts(ref["levels"])] == [l.label for l in spectrum_levels(2)]
    assert all(l["flavors"] is None for l in ref["levels"])
    assert np.allclose(ref["wave_sample_r"], np.linspace(0.0, 20.0, 81), rtol=0, atol=1e-15)


@pytest.mark.parametrize("sys", SYSTEMS)
def test_system_structure(ref, sys):
    entry = ref["systems"][sys]
    assert entry["error"] is None
    params, mq = jr.defaults()
    from gimodel import Meson

    meson = Meson.from_table(mq, *entry["flavors"])
    m = meson.constituent_masses
    assert [m.m1_GeV, m.m2_GeV] == entry["masses_GeV"]
    assert is_equal_flavor(meson) == entry["is_equal_flavor"]
    spec = _spectrum(sys)
    assert [s.label for s in spec.states] == [s["label"] for s in entry["states"]]
    assert [s.basis.flavors for s in spec.states] == [tuple(s["basis"]["flavors"]) for s in entry["states"]]
    # same set of mixed states, and the same number of mixing steps each
    assert [len(s.mixings) for s in spec.states] == [len(s["mixings"]) for s in entry["states"]]


# --- per state -----------------------------------------------------------------------------


@pytest.mark.parametrize("sys,i", STATE_CASES)
def test_state_contributions(sys, i):
    spec, state, sref, what = _state(sys, i)
    jr.compare_contributions(state, sref, what)


@pytest.mark.parametrize("sys,i", STATE_CASES)
def test_state_mixings(sys, i):
    spec, state, sref, what = _state(sys, i)
    jr.compare_mixings(state, sref, what)


@pytest.mark.parametrize("sys,i", STATE_CASES)
def test_state_physical_components(sys, i):
    spec, state, sref, what = _state(sys, i)
    jr.compare_physical_components(spec, state, sref, what)


@pytest.mark.parametrize("sys,i", STATE_CASES)
def test_state_precursor_wave(ref, sys, i):
    spec, state, sref, what = _state(sys, i)
    jr.compare_precursor_wave(spec, state, sref, ref["wave_sample_r"], what)


# --- cross-system structure ---------------------------------------------------------------


@pytest.mark.parametrize("sys", SYSTEMS)
def test_mixing_mechanisms_by_flavor(ref, sys):
    """Equal flavor: tensor mixing only; unequal flavor: antisymmetric spin-orbit as well."""
    entry = ref["systems"][sys]
    spec = _spectrum(sys)
    got = {m.mechanism for s in spec.states for m in s.mixings}
    exp = {m["mechanism"] for s in entry["states"] for m in s["mixings"]}
    assert got == exp
    if entry["is_equal_flavor"]:
        assert got == {TENSOR_MIXING}
    else:
        assert got == {TENSOR_MIXING, ANTISYMMETRIC_SPIN_ORBIT}


def _so_blocks(spec):
    """Antisymmetric spin-orbit blocks keyed by their partner labels."""
    out = {}
    for s in spec.states:
        for mx in s.mixings:
            if mx.mechanism == ANTISYMMETRIC_SPIN_ORBIT:
                out[tuple(mx.partner_labels)] = mx
    return out


def _ref_so_blocks(entry):
    out = {}
    for s in entry["states"]:
        for mx in s["mixings"]:
            if mx["mechanism"] == ANTISYMMETRIC_SPIN_ORBIT:
                out[tuple(mx["partner_labels"])] = mx
    return out


@pytest.mark.parametrize("pair", [("cu", "uc")])
def test_spin_orbit_sign_flip_under_quark_exchange(ref, pair):
    """Swapping m1 <-> m2 leaves every mass unchanged but flips the sign of the
    antisymmetric spin-orbit coupling (the singlet-triplet off-diagonal elements),
    hence of the triplet components of the mixed eigenvectors -- in Julia and Python alike."""
    a, b = pair
    spec_a, spec_b = _spectrum(a), _spectrum(b)
    for sa, sb in zip(spec_a.states, spec_b.states):
        assert sa.label == sb.label
        check(sa.mass_GeV, sb.mass_GeV, ENERGY, f"{a}/{b} {sa.label} mass symmetric", record=False)
    blocks_a, blocks_b = _so_blocks(spec_a), _so_blocks(spec_b)
    ref_a, ref_b = _ref_so_blocks(ref["systems"][a]), _ref_so_blocks(ref["systems"][b])
    assert set(blocks_a) == set(blocks_b) == set(ref_a) == set(ref_b) and blocks_a
    for labels, mxa in blocks_a.items():
        mxb = blocks_b[labels]
        mat_a, mat_b = np.array(mxa.result.block.matrix), np.array(mxb.result.block.matrix)
        n = len(labels)
        off = ~np.eye(n, dtype=bool)
        singlet = np.array([l.split("^")[1].startswith("1") for l in labels])
        cross = singlet[:, None] != singlet[None, :]
        # Python: diagonal identical, off-diagonal sign flipped
        check(np.diag(mat_b), np.diag(mat_a), ENERGY, f"{labels} diagonal", record=False)
        check(mat_b[off], -mat_a[off], ENERGY, f"{labels} offdiag flips", record=False)
        assert np.all(np.abs(mat_a[cross]) > 1e-6), "singlet-triplet couplings must be non-trivial"
        assert np.all(mat_a[off & ~cross] == 0.0), "no same-spin couplings in a spin-orbit block"
        # Julia shows the same flip
        ja, jb = np.array(ref_a[labels]["block_matrix_GeV"]), np.array(ref_b[labels]["block_matrix_GeV"])
        check(jb[off], -ja[off], ENERGY, f"{labels} Julia offdiag flips", record=False)
        # eigenvectors: v_b = D v_a per column (D = +1 singlet rows, -1 triplet rows), up to
        # the column phase fixed by the assigned-state convention -- Python and Julia alike
        D = np.where(singlet, 1.0, -1.0)
        for name, va, vb in (
            ("python", np.array(mxa.result.vectors), np.array(mxb.result.vectors)),
            ("julia", np.array(ref_a[labels]["block_vectors"]), np.array(ref_b[labels]["block_vectors"])),
        ):
            dva = D[:, None] * va
            anchor = np.argmax(np.abs(va), axis=0)
            phase = np.sign(vb[anchor, range(n)] * dva[anchor, range(n)])
            check(vb, dva * phase, MIXING, f"{labels} {name} vectors flip", record=False)


@pytest.mark.parametrize("sys", SYSTEMS)
def test_spectrum_state_lookup(sys):
    spec = _spectrum(sys)
    for s in spec.states:
        assert spectrum_state(spec, s.label) is s
