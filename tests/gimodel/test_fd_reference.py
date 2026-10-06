"""Toy-grid (ngrid=20, rmax=10) FD operators against GIModel.jl (tests/reference/fd_matrices.json)."""
from __future__ import annotations

import numpy as np
import pytest

from gimodel import (
    ConstituentMasses,
    FineStructureMultiplet,
    FiniteDifferenceSolver,
    fixed_channel_matrices,
    fixed_channel_solution,
    spin_orbit_mixing_components,
    tensor_mixing_components,
)
from gimodel import fd
from gimodel.hamiltonian import relativistic_hamiltonian

from conftest import load_reference

ATOL = 1e-11


@pytest.fixture(scope="module")
def ref():
    return load_reference("fd_matrices.json")


@pytest.fixture(scope="module")
def toy(ref):
    return FiniteDifferenceSolver(ngrid=ref["solver"]["ngrid"], rmax=ref["solver"]["rmax"])


def _masses(entry) -> ConstituentMasses:
    return ConstituentMasses(*entry["masses_GeV"])


def test_grid_and_p2(ref, toy):
    r, h = fd.radial_grid(toy.ngrid, toy.rmax)
    assert np.allclose(r, ref["r"], rtol=0, atol=1e-14) and h == pytest.approx(ref["h"], rel=1e-15)
    for key, mat in ref["p2"].items():
        d, e = fd.p2_operator(int(key[1:]), r, h)
        P = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        assert np.allclose(P, np.array(mat), rtol=1e-14, atol=1e-12)


def test_central_hamiltonians(ref, toy, params):
    for entry in ref["central"]:
        H, _ = relativistic_hamiltonian(params, _masses(entry), entry["L"], ngrid=toy.ngrid, rmax=toy.rmax)
        assert np.allclose(H, np.array(entry["H0"]), rtol=0, atol=ATOL)
        assert np.allclose(np.linalg.eigvalsh(H), entry["eigenvalues"], rtol=0, atol=ATOL)


def test_fixed_channel_terms_and_eigenbasis_solve(ref, toy, params):
    for entry in ref["fixed"]:
        multiplet = FineStructureMultiplet(entry["L_label"], entry["multiplicity"], entry["J"])
        mats = fixed_channel_matrices(toy, params, _masses(entry), multiplet)
        for name in ("central", "contact", "spin_orbit_vector", "spin_orbit_thomas", "tensor", "fine_structure", "total"):
            assert np.allclose(mats[name], np.array(entry[name]), rtol=0, atol=ATOL), name
        sol = fixed_channel_solution(params, _masses(entry), multiplet, solver=toy, nlevels=6)
        assert np.allclose(sol.eigenvalues_GeV, entry["eigenvalues_total"][:6], rtol=0, atol=ATOL)


def test_mixing_elements(ref, toy, params, mq):
    def ground(m1, m2, L, mult, J):
        masses = ConstituentMasses(m1, m2)
        return masses, fixed_channel_solution(
            params, masses, FineStructureMultiplet(L, mult, J), solver=toy, nlevels=1
        ).radial_wave(1)

    els = ref["mixing_elements"]
    masses, w1 = ground(mq["c"], mq["u"], "P", 1, 1)
    _, w3 = ground(mq["c"], mq["u"], "P", 3, 1)
    so = spin_orbit_mixing_components(params, masses, "P", w1, w3)
    for key, value in els["cu_1P1_3P1_spin_orbit"].items():
        assert getattr(so, key) == pytest.approx(value, rel=1e-9, abs=1e-13), key
    for name, (m1, m2) in {"cc": (mq["c"], mq["c"]), "cu": (mq["c"], mq["u"])}.items():
        masses, ws = ground(m1, m2, "S", 3, 1)
        _, wd = ground(m1, m2, "D", 3, 1)
        t = tensor_mixing_components(params, masses, ws, wd, 1)
        for key, value in els[f"{name}_3S1_3D1_tensor"].items():
            assert getattr(t, key) == pytest.approx(value, rel=1e-9, abs=1e-13), key
