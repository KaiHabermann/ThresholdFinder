"""tests/reference/spot_checks.json, FD entries: the documented GIModel.jl numbers. Each
Python value must match the Julia run's ``computed`` value at the layer tolerance (energies
1e-9 GeV, mixing components 1e-8 with exact sign) and the ``documented`` value within the
entry's own tolerance. The HO entries belong to the oscillator suite.
"""
from __future__ import annotations

import pytest

from gimodel import FiniteDifferenceSolver, Meson, SpinTerms, spectrum_levels, spectrum_state

import julia_ref as jr
from julia_ref import ENERGY, MIXING, check, check_sign

REF = jr.load_or_none("spot_checks.json")
FD_CHECKS = [c for c in REF["checks"] if " FD " in c["name"]] if REF else []


def _cc():
    params, mq = jr.defaults()
    return jr.cached_spectrum(
        params, Meson.from_table(mq, "c", "c"), tuple(spectrum_levels(2)), FiniteDifferenceSolver(), SpinTerms()
    )


def _cu_p():
    params, mq = jr.defaults()
    return jr.cached_spectrum(
        params,
        Meson.from_table(mq, "c", "u"),
        tuple(spectrum_levels(1, L_labels=("P",))),
        FiniteDifferenceSolver(),
        SpinTerms(),
    )


def _cu_mixing():
    (mx,) = spectrum_state(_cu_p(), "1^1P_1").mixings
    return mx


VALUES = {
    "cc FD 1^1S_0 mass (4 d.p.)": (ENERGY, lambda: spectrum_state(_cc(), "1^1S_0").mass_GeV),
    "cc FD 1^3S_1 mass (4 d.p.)": (ENERGY, lambda: spectrum_state(_cc(), "1^3S_1").mass_GeV),
    "cc FD 1^3P_0 mass (4 d.p.)": (ENERGY, lambda: spectrum_state(_cc(), "1^3P_0").mass_GeV),
    "cc FD chi_c1 1^3P_1 mass": (ENERGY, lambda: spectrum_state(_cc(), "1^3P_1").mass_GeV),
    "cc FD chi_c2 1^3P_2 mass": (ENERGY, lambda: spectrum_state(_cc(), "1^3P_2").mass_GeV),
    "cu FD 1P mixed low (4 d.p.)": (ENERGY, lambda: spectrum_state(_cu_p(), "1^1P_1").mass_GeV),
    "cu FD 1P mixed high (4 d.p.)": (ENERGY, lambda: spectrum_state(_cu_p(), "1^3P_1").mass_GeV),
    "cu FD 1P offdiag": (ENERGY, lambda: _cu_mixing().offdiag_GeV),
    "cu FD 1P partner mass low": (ENERGY, lambda: _cu_mixing().partner_masses_GeV[0]),
    "cu FD 1P partner mass high": (ENERGY, lambda: _cu_mixing().partner_masses_GeV[1]),
    "cu FD 1P components[1]": (MIXING, lambda: _cu_mixing().components[0]),
    "cu FD 1P components[2]": (MIXING, lambda: _cu_mixing().components[1]),
}


def test_every_fd_check_is_mapped():
    assert {c["name"] for c in FD_CHECKS} == set(VALUES)
    assert all(c["pass"] for c in FD_CHECKS)


@pytest.mark.parametrize("entry", FD_CHECKS, ids=[c["name"] for c in FD_CHECKS])
def test_spot_check(entry):
    layer, value = VALUES[entry["name"]]
    got = float(value())
    check(got, entry["computed"], layer, f"spot {entry['name']}")
    if layer == MIXING:
        check_sign(got, entry["computed"], f"spot {entry['name']}")
    assert abs(got - entry["documented"]) <= entry["tolerance"], (got, entry["documented"])
