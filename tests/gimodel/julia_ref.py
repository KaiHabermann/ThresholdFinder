"""Shared helpers for the GIModel.jl cross-validation suite (tests/test_julia_fd_*.py).

* :func:`load` -- cached JSON reference loading (skips when a file is absent);
* :func:`params_from_dict`, :func:`terms_from_dict`, :func:`solver_from_dict`,
  :func:`meson_from_system` -- rebuild the Python objects from the stored metadata;
* :func:`sample_wave` -- port of GIModel.jl ``sample_wave(::MeshWave, r)``;
* :func:`check` / :func:`check_sign` -- layered-tolerance comparisons that also
  record the largest deviation seen per layer (printed in the terminal summary).

Tolerance layers (see the module constants): primitives ``1e-12`` relative
(absolute near zero), matrices ``1e-10``, energies ``1e-9`` GeV, mixing and
physical-component coefficients ``1e-8``, sampled/native waves ``1e-7``.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gimodel import (
    CentralPotentialMethod,
    ConstituentMasses,
    FiniteDifferenceSolver,
    GIParameters,
    Meson,
    SpinTerms,
    load_parameters_and_quark_masses,
)

REFERENCE_DIR = Path(__file__).parent / "reference"

# --- tolerance layers -----------------------------------------------------------------

PRIMITIVE = "primitive"
MATRIX = "matrix"
ENERGY = "energy"
MIXING = "mixing"
WAVE = "wave"

TOL = {
    PRIMITIVE: dict(rtol=1e-12, atol=1e-12),  # relative, absolute only near zero (|x| < 1)
    MATRIX: dict(rtol=0.0, atol=1e-10),
    ENERGY: dict(rtol=0.0, atol=1e-9),
    MIXING: dict(rtol=0.0, atol=1e-8),
    WAVE: dict(rtol=0.0, atol=1e-7),
}

EPS = float(np.finfo(float).eps)
NOISE_ULPS = 16.0
CANCELLATION = "primitive (cancellation, ulps of scale)"

#: largest |got - expected| per layer (and, for primitives, the largest relative deviation)
DEVIATIONS: dict[str, dict[str, float | str]] = {}


def _record(layer: str, abs_dev: float, rel_dev: float, what: str) -> None:
    entry = DEVIATIONS.setdefault(layer, {"abs": 0.0, "rel": 0.0, "where": ""})
    if abs_dev > entry["abs"]:
        entry["abs"] = abs_dev
        entry["where"] = what
    entry["rel"] = max(entry["rel"], rel_dev)


def check(got, expected, layer: str, what: str = "", *, noise=None, record: bool = True) -> None:
    """Assert ``got ~= expected`` elementwise within the tolerance of ``layer``.

    ``None`` in the reference (non-finite in Julia) must be non-finite in Python.
    ``noise`` (primitives only) is a per-element rounding floor for closed forms that
    subtract nearly equal terms: ``NOISE_ULPS * eps * (sum of |cancelling terms|)``.
    Where it exceeds the 1e-12 limit, the deviation is recorded in units of
    ``eps * scale`` under ``"primitive (cancellation, ulps of scale)"`` instead.
    """
    tol = TOL[layer]
    exp = np.atleast_1d(np.array(expected, dtype=float))  # None -> nan
    g = np.atleast_1d(np.array(got, dtype=float))
    assert g.shape == exp.shape, f"{what}: shape {g.shape} != reference {exp.shape}"
    finite = np.isfinite(exp)
    assert np.array_equal(np.isfinite(g), finite), f"{what}: finiteness pattern differs"
    if not finite.any():
        return
    diff = np.abs(g[finite] - exp[finite])
    scale = np.abs(exp[finite])
    if layer == PRIMITIVE:
        limit = np.maximum(tol["atol"], tol["rtol"] * scale)
    else:
        limit = tol["atol"] + tol["rtol"] * scale
    plain = np.ones_like(diff, dtype=bool)
    if noise is not None:
        floor = np.broadcast_to(np.asarray(noise, dtype=float), exp.shape)[finite]
        plain = floor <= limit
        if (~plain).any() and record:
            ulps = diff[~plain] / (EPS * (floor[~plain] / (NOISE_ULPS * EPS)))
            _record(CANCELLATION, float(ulps.max()), 0.0, what)
        limit = np.maximum(limit, floor)
    if plain.any() and record:
        d, sc = diff[plain], scale[plain]
        rel_dev = float(np.where(sc > 1.0, d / np.maximum(sc, 1e-300), d).max())
        _record(layer, float(d.max()), rel_dev, what)
    bad = diff > limit
    if bad.any():
        idx = np.argwhere(finite)[bad][:5]
        detail = "; ".join(
            f"[{','.join(map(str, i))}] got {g[tuple(i)]!r} expected {exp[tuple(i)]!r}" for i in idx
        )
        raise AssertionError(
            f"{what}: {int(bad.sum())} element(s) outside {layer} tolerance {tol} "
            f"(max |diff| {float(diff.max()):.3e}): {detail}"
        )


def check_sign(got, expected, what: str = "", floor: float = 1e-8) -> None:
    """Exact sign agreement for every reference entry with ``|x| > floor``."""
    g = np.atleast_1d(np.array(got, dtype=float))
    e = np.atleast_1d(np.array(expected, dtype=float))
    mask = np.abs(e) > floor
    mism = mask & (np.sign(g) != np.sign(e))
    assert not mism.any(), f"{what}: sign mismatch at {np.argwhere(mism).ravel().tolist()} (got {g[mism]}, expected {e[mism]})"


# --- reference loading ----------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _load(name: str):
    with open(REFERENCE_DIR / name) as handle:
        return json.load(handle)


def load(name: str):
    if not (REFERENCE_DIR / name).exists():
        pytest.skip(f"Julia reference file {name} not available")
    return _load(name)


def load_or_none(name: str):
    """For collection-time parametrisation (no skip outside a test)."""
    return _load(name) if (REFERENCE_DIR / name).exists() else None


# --- building Python objects from the metadata ----------------------------------------


@functools.lru_cache(maxsize=1)
def defaults() -> tuple[GIParameters, dict]:
    return load_parameters_and_quark_masses()


def _central_method(name: str) -> CentralPotentialMethod:
    """Julia ``AppendixAMomentumSandwich`` -> ``appendix_a_momentum_sandwich``."""
    snake = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()
    return CentralPotentialMethod(snake)


def params_from_dict(d: dict, base: GIParameters | None = None) -> GIParameters:
    """Rebuild :class:`GIParameters` from a stored ``params_dict`` via ``dataclasses.replace``."""
    base = defaults()[0] if base is None else base
    f = d["factors"]
    a = d["annihilation"]
    return replace(
        base,
        potential=replace(base.potential, b=d["potential"]["b_GeV2"], c=d["potential"]["c_GeV"]),
        central=_central_method(d["central"]),
        smearing=replace(base.smearing, sigma0=d["smearing"]["sigma0_GeV"], s=d["smearing"]["s"]),
        factors=replace(
            base.factors,
            epsilon_c=f["epsilon_c"],
            epsilon_t=f["epsilon_t"],
            epsilon_so_vector=f["epsilon_so_vector"],
            epsilon_so_scalar=f["epsilon_so_scalar"],
            contact_momentum_sandwich=f["contact_momentum_sandwich"],
            fine_structure_momentum_sandwich=f["fine_structure_momentum_sandwich"],
            fine_structure_smeared_kernels=f["fine_structure_smeared_kernels"],
        ),
        fine_structure=replace(base.fine_structure, enabled=d["fine_structure_enabled"]),
        annihilation=replace(base.annihilation, **a),
    )


def terms_from_dict(d: dict) -> SpinTerms:
    return SpinTerms(**d)


def solver_from_dict(d: dict) -> FiniteDifferenceSolver:
    assert d["type"] == "FiniteDifferenceSolver", d
    return FiniteDifferenceSolver(
        ngrid=d["ngrid"],
        rmax=d["rmax"],
        kinetic=d["kinetic"],
        eigensolver=d["eigensolver"],
        nlevels_per_channel=d["nlevels_per_channel"],
    )


def meson_from_system(flavors, masses_GeV) -> Meson:
    """``Meson(f1, f2, ConstituentMasses(m1, m2))`` with the stored (possibly varied) masses."""
    return Meson(flavors[0], flavors[1], ConstituentMasses(*masses_GeV))


# --- waves ----------------------------------------------------------------------------


def sample_wave(wave, r) -> np.ndarray:
    """Port of GIModel.jl ``sample_wave(::MeshWave, r).u``: linear interpolation with
    ``u(0) = 0`` and ``u = 0`` from ``r_last + h`` on, renormalised so ``sum(u^2) dr = 1``."""
    r = np.asarray(r, dtype=float)
    nodes = np.concatenate(([0.0], wave.r, [wave.r[-1] + wave.h]))
    values = np.concatenate(([0.0], wave.u, [0.0]))
    out = np.zeros_like(r)
    for k, ri in enumerate(r):
        if ri <= 0 or ri >= nodes[-1]:
            continue
        i = int(np.clip(np.searchsorted(nodes, ri, side="right") - 1, 0, len(nodes) - 2))
        t = (ri - nodes[i]) / (nodes[i + 1] - nodes[i])
        out[k] = (1 - t) * values[i] + t * values[i + 1]
    dr = r[1] - r[0]
    return out / np.sqrt(np.sum(out**2) * dr)


def ids_of(items, key) -> list[str]:
    return [key(x) for x in items]


# --- spectra ------------------------------------------------------------------------------


def levels_from_spec(spec: str):
    """Map the stored ``levels_spec`` string to the Python level list."""
    from gimodel import spectrum_levels

    known = {
        "spectrum_levels(1)": lambda: spectrum_levels(1),
        "spectrum_levels(2)": lambda: spectrum_levels(2),
        'spectrum_levels(1; L_labels=("S","P","D","F"))': lambda: spectrum_levels(1, L_labels=("S", "P", "D", "F")),
    }
    if spec not in known:
        raise KeyError(f"unknown levels_spec {spec!r}")
    return tuple(known[spec]())


def levels_from_dicts(levels) -> tuple:
    from gimodel import BasisState

    return tuple(BasisState(l["n"], l["L_label"], l["multiplicity"], l["J"]) for l in levels)


@functools.lru_cache(maxsize=None)
def cached_spectrum(params, meson, levels: tuple, solver, terms):
    from gimodel import compute_spectrum

    return compute_spectrum(params, meson, levels=list(levels), solver=solver, terms=terms)


def assert_basis(basis, ref: dict, what: str) -> None:
    got = (basis.label, basis.n, basis.L_label, basis.multiplicity, basis.J)
    exp = (ref["label"], ref["n"], ref["L_label"], ref["multiplicity"], ref["J"])
    assert got == exp, f"{what}: basis {got} != {exp}"
    exp_flavors = None if ref["flavors"] is None else tuple(ref["flavors"])
    assert basis.flavors == exp_flavors, f"{what}: flavors {basis.flavors} != {exp_flavors}"


CONTRIBUTIONS = (
    "central_GeV",
    "contact_shift_GeV",
    "spin_orbit_vector_shift_GeV",
    "spin_orbit_thomas_shift_GeV",
    "spin_orbit_shift_GeV",
    "tensor_shift_GeV",
    "fine_structure_shift_GeV",
)


def compare_contributions(state, ref: dict, what: str) -> None:
    assert_basis(state.basis, ref["basis"], what)
    assert state.label == ref["label"], what
    check(state.mass_GeV, ref["mass_GeV"], ENERGY, f"{what} mass_GeV")
    check(state.corrected.mass_GeV, ref["corrected_mass_GeV"], ENERGY, f"{what} corrected_mass_GeV")
    for name in CONTRIBUTIONS:
        check(getattr(state.corrected, name), ref[name], ENERGY, f"{what} {name}")
    assert state.fine_structure_mass_convention == ref["fine_structure_mass_convention"], what
    assert state.corrected.fine_structure_mass_convention == ref["corrected_fine_structure_mass_convention"], what
    # internal consistency mirrored from the reference notes
    c = state.corrected
    total = c.central_GeV + c.contact_shift_GeV + c.fine_structure_shift_GeV
    assert abs(total - c.mass_GeV) < 1e-12, f"{what}: contributions do not sum to the corrected mass"


def compare_mixings(state, ref: dict, what: str) -> None:
    got, exp = state.mixings, ref["mixings"]
    assert [m.mechanism for m in got] == [m["mechanism"] for m in exp], f"{what}: mixing mechanisms"
    for k, (mx, mref) in enumerate(zip(got, exp)):
        w = f"{what} mixing[{k}] ({mref['mechanism']})"
        assert mx.block_label == mref["block_label"], w
        assert list(mx.partner_labels) == mref["partner_labels"], f"{w}: partner labels"
        assert mx.eigenstate + 1 == mref["eigenstate"], f"{w}: eigenstate column (Julia 1-based)"
        for b in mx.result.block.basis:
            assert b.flavors == state.basis.flavors, f"{w}: partner {b.label} flavors"
        check(mx.unmixed_GeV, mref["unmixed_GeV"], ENERGY, f"{w} unmixed_GeV")
        check(mx.offdiag_GeV, mref["offdiag_GeV"], ENERGY, f"{w} offdiag_GeV")
        check(mx.partner_masses_GeV, mref["partner_masses_GeV"], ENERGY, f"{w} partner_masses_GeV")
        check(mx.result.block.matrix, mref["block_matrix_GeV"], MATRIX, f"{w} block_matrix_GeV")
        check(mx.components, mref["components"], MIXING, f"{w} components")
        check_sign(mx.components, mref["components"], f"{w} components")
        check(mx.result.vectors, mref["block_vectors"], MIXING, f"{w} block_vectors")
        check_sign(mx.result.vectors, mref["block_vectors"], f"{w} block_vectors")


def compare_physical_components(spec, state, ref: dict, what: str) -> None:
    from gimodel import physical_components

    comps = physical_components(spec, state)
    exp = ref["physical_components"]
    assert [c.basis.label for c in comps] == [c["label"] for c in exp], f"{what}: component labels"
    for c, cref in zip(comps, exp):
        assert_basis(c.basis, cref["basis"], f"{what} component {cref['label']}")
    coeffs = [c.coefficient for c in comps]
    expected = [c["coefficient"] for c in exp]
    check(coeffs, expected, MIXING, f"{what} physical coefficients")
    check_sign(coeffs, expected, f"{what} physical coefficients")


def compare_precursor_wave(spec, state, ref: dict, sample_r, what: str) -> None:
    from gimodel import radial_wave

    u = sample_wave(radial_wave(spec, state.corrected), sample_r)
    check(u, ref["precursor_wave_sampled"], WAVE, f"{what} precursor_wave_sampled")
    check_sign(u, ref["precursor_wave_sampled"], f"{what} precursor_wave_sampled", floor=1e-6)
