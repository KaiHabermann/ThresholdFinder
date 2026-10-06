"""Layer c (tests/reference/fd_matrices.json), toy grid ngrid=20, rmax=10: the operators
that tests/test_fd_reference.py does not reach, parametrized per entry at the matrix
tolerance (1e-10): the kinetic/potential split of the central Hamiltonian, the summed
spin-orbit matrix, the full 20-level spectrum of every fixed-sector Hamiltonian (both
the r-space operator and the p^2-eigenbasis operator the solver diagonalises), and the
toy-grid fixed-channel waves feeding the mixing elements.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from gimodel import ConstituentMasses, FineStructureMultiplet, FiniteDifferenceSolver, SpinTerms, fixed_channel_matrices
from gimodel import fd
from gimodel.hamiltonian import appendix_a_momentum_sandwich_matrix, central_eigenbasis
from gimodel.spin import ContactHyperfine, contact_eigenbasis, fine_structure_eigenbasis

import julia_ref as jr
from julia_ref import ENERGY, MATRIX, check

REF = jr.load_or_none("fd_matrices.json")
CENTRAL = (
    [pytest.param(i, id=f"{e['system']}-L{e['L']}") for i, e in enumerate(REF["central"])] if REF else []
)
FIXED = (
    [pytest.param(i, id=f"{e['system']}-^{e['multiplicity']}{e['L_label']}_{e['J']}") for i, e in enumerate(REF["fixed"])]
    if REF
    else []
)


@pytest.fixture(scope="module")
def ref():
    return jr.load("fd_matrices.json")


@pytest.fixture(scope="module")
def params(ref):
    return jr.params_from_dict(ref["metadata"]["parameters"])


@pytest.fixture(scope="module")
def toy(ref):
    solver = jr.solver_from_dict(ref["solver"])
    assert solver == FiniteDifferenceSolver(ngrid=20, rmax=10.0)
    return solver


def _masses(entry) -> ConstituentMasses:
    return ConstituentMasses(*entry["masses_GeV"])


@pytest.mark.parametrize("i", CENTRAL)
def test_central_kinetic_and_potential(ref, params, toy, i):
    entry = ref["central"][i]
    m = _masses(entry)
    r, h = fd.radial_grid(toy.ngrid, toy.rmax)
    lam, V = fd.p2_eigen(entry["L"], r, h)
    lam = np.maximum(lam, 0.0)
    kinetic = fd.matrix_function(np.sqrt(lam + m.m1_GeV**2), V) + fd.matrix_function(np.sqrt(lam + m.m2_GeV**2), V)
    potential = appendix_a_momentum_sandwich_matrix(params, m, entry["L"], r, h)
    what = f"{entry['system']} L={entry['L']}"
    check(kinetic, entry["kinetic"], MATRIX, f"{what} kinetic")
    check(potential, entry["potential"], MATRIX, f"{what} potential")
    check(kinetic + potential, entry["H0"], MATRIX, f"{what} kinetic+potential")


@pytest.mark.parametrize("i", CENTRAL)
def test_central_eigenbasis_spectrum(ref, params, toy, i):
    """The solver's p^2-eigenbasis H0 is orthogonally similar to the r-space H0."""
    entry = ref["central"][i]
    r, h = fd.radial_grid(toy.ngrid, toy.rmax)
    H = central_eigenbasis(params, _masses(entry), entry["L"], r, h)
    check(np.linalg.eigvalsh(0.5 * (H + H.T)), entry["eigenvalues"], ENERGY, f"{entry['system']} L={entry['L']} eigenbasis")


@pytest.mark.parametrize("i", FIXED)
def test_fixed_spin_orbit_and_full_spectrum(ref, params, toy, i):
    entry = ref["fixed"][i]
    mult = FineStructureMultiplet(entry["L_label"], entry["multiplicity"], entry["J"])
    what = f"{entry['system']} ^{entry['multiplicity']}{entry['L_label']}_{entry['J']}"
    mats = fixed_channel_matrices(toy, params, _masses(entry), mult)
    check(mats["spin_orbit"], entry["spin_orbit"], MATRIX, f"{what} spin_orbit")
    check(mats["spin_orbit_vector"] + mats["spin_orbit_thomas"], entry["spin_orbit"], MATRIX, f"{what} so parts")
    check(np.linalg.eigvalsh(mats["total"]), entry["eigenvalues_total"], ENERGY, f"{what} all eigenvalues")


@pytest.mark.parametrize("i", FIXED)
def test_fixed_eigenbasis_spectrum(ref, params, toy, i):
    """Assemble the solver's eigenbasis operator exactly as fd_fixed_channel_solution does."""
    entry = ref["fixed"][i]
    m = _masses(entry)
    L = {"S": 0, "P": 1, "D": 2, "F": 3}[entry["L_label"]]
    r, h = fd.radial_grid(toy.ngrid, toy.rmax)
    total = central_eigenbasis(params, m, L, r, h)
    total = total + contact_eigenbasis(ContactHyperfine(params, m, entry["multiplicity"]), L, r, h)
    fine = fine_structure_eigenbasis(params, m, entry["J"], r, h, L=L, multiplicity=entry["multiplicity"])
    if fine is not None:
        total = total + fine
    what = f"{entry['system']} ^{entry['multiplicity']}{entry['L_label']}_{entry['J']}"
    check(np.linalg.eigvalsh(0.5 * (total + total.T)), entry["eigenvalues_total"], ENERGY, f"{what} eigenbasis")


@pytest.mark.parametrize("i", FIXED)
def test_fixed_terms_toggle(ref, params, toy, i):
    """SpinTerms switches remove exactly the corresponding reference operators."""
    entry = ref["fixed"][i]
    mult = FineStructureMultiplet(entry["L_label"], entry["multiplicity"], entry["J"])
    what = f"{entry['system']} ^{entry['multiplicity']}{entry['L_label']}_{entry['J']}"
    no_contact = fixed_channel_matrices(toy, params, _masses(entry), mult, SpinTerms(contact_hyperfine=False))
    no_fine = fixed_channel_matrices(toy, params, _masses(entry), mult, SpinTerms(fine_structure=False))
    central, contact, fine = (np.array(entry[k]) for k in ("central", "contact", "fine_structure"))
    check(no_contact["total"], central + fine, MATRIX, f"{what} total without contact")
    check(no_fine["total"], central + contact, MATRIX, f"{what} total without fine structure")
