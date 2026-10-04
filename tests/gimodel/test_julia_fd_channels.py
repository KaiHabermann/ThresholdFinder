"""Layer d (tests/reference/channels.json), FD solvers only: channel_solution (central,
L = 0..2) and fixed_channel_solution (10 (L,S,J) multiplets) for cc and cu with the
default FiniteDifferenceSolver (ngrid 450) and the toy grid (ngrid 20).

Per channel: all 6 eigenvalues (1e-9 GeV), the native mesh waves (mesh metadata exact,
u within 1e-7 with the same outer-lobe-positive phase) and the waves sampled on the
81-point plotting grid via ``sample_wave`` (1e-7, same phase).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from gimodel import FineStructureMultiplet, Meson, MeshWave, channel_solution, fixed_channel_solution

import julia_ref as jr
from julia_ref import ENERGY, WAVE, check, check_sign

REF = jr.load_or_none("channels.json")
FD_SOLVERS = [k for k in ("fd_default", "fd_toy")] if REF else []


def _channel_ids(ref):
    out = []
    for sname in FD_SOLVERS:
        for sys, entry in ref["solvers"][sname]["systems"].items():
            for i, c in enumerate(entry["central"]):
                out.append(pytest.param(sname, sys, "central", i, id=f"{sname}-{sys}-central-L{c['L']}"))
            for i, c in enumerate(entry["fixed"]):
                out.append(pytest.param(sname, sys, "fixed", i, id=f"{sname}-{sys}-fixed-{c['label']}"))
    return out


CASES = _channel_ids(REF) if REF else []


@pytest.fixture(scope="module")
def ref():
    return jr.load("channels.json")


@functools.lru_cache(maxsize=None)
def _solve(sname: str, sys: str, kind: str, i: int):
    ref = jr.load("channels.json")
    block = ref["solvers"][sname]
    entry = block["systems"][sys]
    params = jr.params_from_dict(ref["metadata"]["parameters"])
    solver = jr.solver_from_dict(block["solver"])
    from gimodel import ConstituentMasses

    masses = ConstituentMasses(*entry["masses_GeV"])
    c = entry[kind][i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # toy grid is deliberately under-resolved
        if kind == "central":
            sol = channel_solution(params, masses, c["L"], solver=solver)
        else:
            sol = fixed_channel_solution(
                params, masses, FineStructureMultiplet(c["L_label"], c["multiplicity"], c["J"]), solver=solver
            )
    return sol, c["solution"], f"{sname} {sys} {kind} {c.get('label', c.get('L'))}"


def test_reference_setup(ref):
    from gimodel import FiniteDifferenceSolver

    assert jr.solver_from_dict(ref["solvers"]["fd_default"]["solver"]) == FiniteDifferenceSolver()
    assert jr.solver_from_dict(ref["solvers"]["fd_toy"]["solver"]) == FiniteDifferenceSolver(ngrid=20, rmax=10.0)
    params, mq = jr.defaults()
    for sname in FD_SOLVERS:
        for sys, entry in ref["solvers"][sname]["systems"].items():
            m = Meson.from_table(mq, sys[0], sys[1]).constituent_masses
            assert [m.m1_GeV, m.m2_GeV] == entry["masses_GeV"]
            assert [c["L"] for c in entry["central"]] == [0, 1, 2]
            assert len(entry["fixed"]) == 10


@pytest.mark.parametrize("sname,sys,kind,i", CASES)
def test_eigenvalues(sname, sys, kind, i):
    sol, sref, what = _solve(sname, sys, kind, i)
    assert sref["convergence"] is None and sol.convergence is None
    assert len(sol.eigenvalues_GeV) == len(sref["eigenvalues_GeV"]) == 6
    check(sol.eigenvalues_GeV, sref["eigenvalues_GeV"], ENERGY, f"{what} eigenvalues")


@pytest.mark.parametrize("sname,sys,kind,i", CASES)
def test_waves_native(sname, sys, kind, i):
    sol, sref, what = _solve(sname, sys, kind, i)
    assert len(sol.waves) == len(sref["waves_native"])
    for n, (w, wref) in enumerate(zip(sol.waves, sref["waves_native"]), start=1):
        assert isinstance(w, MeshWave) and wref["type"] == "MeshWave"
        assert len(w.u) == wref["n"]
        assert w.h == pytest.approx(wref["h"], rel=1e-15)
        assert w.r[0] == pytest.approx(wref["r_first"], rel=1e-15)
        assert np.sum(w.u**2) * w.h == pytest.approx(1.0, abs=1e-12)
        check(w.u, wref["u"], WAVE, f"{what} native wave n={n}")
        check_sign(w.u, wref["u"], f"{what} native wave n={n}", floor=1e-6)


@pytest.mark.parametrize("sname,sys,kind,i", CASES)
def test_waves_sampled(ref, sname, sys, kind, i):
    sol, sref, what = _solve(sname, sys, kind, i)
    r = ref["wave_sample_r"]
    for n, (w, uref) in enumerate(zip(sol.waves, sref["waves_sampled"]), start=1):
        u = jr.sample_wave(w, r)
        check(u, uref, WAVE, f"{what} sampled wave n={n}")
        check_sign(u, uref, f"{what} sampled wave n={n}", floor=1e-6)
