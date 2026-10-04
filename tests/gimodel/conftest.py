"""Shared fixtures for the gimodel test-suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gimodel import Meson, compute_spectrum, load_parameters_and_quark_masses, spectrum_levels

REFERENCE_DIR = Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def params_mq():
    return load_parameters_and_quark_masses()


@pytest.fixture(scope="session")
def params(params_mq):
    return params_mq[0]


@pytest.fixture(scope="session")
def mq(params_mq):
    return params_mq[1]


@pytest.fixture(scope="session")
def charmonium(params, mq):
    """Default FD charmonium spectrum, spectrum_levels(2) (the documented example)."""
    return compute_spectrum(params, Meson.from_table(mq, "c", "c"), levels=spectrum_levels(2))


def load_reference(name: str):
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Julia reference file {name} not available")
    with open(path) as handle:
        return json.load(handle)


def pytest_terminal_summary(terminalreporter):
    """Largest deviation per tolerance layer of the GIModel.jl cross-validation (julia_ref)."""
    import sys

    julia_ref = sys.modules.get("julia_ref")
    if julia_ref is None or not julia_ref.DEVIATIONS:
        return
    terminalreporter.section("GIModel.jl cross-validation: max deviation per layer")
    for layer, d in sorted(julia_ref.DEVIATIONS.items()):
        terminalreporter.write_line(f"{layer:42s} max {d['abs']:.3e}  (rel/abs-near-0 {d['rel']:.3e})  at {d['where']}")
