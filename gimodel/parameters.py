"""GI model parameters and the strict TOML loader (port of ``parameters.jl`` and
``quark_mass_table.jl``).

Parameters are grouped by model aspect, one frozen dataclass per TOML section.
Variations are made with :func:`dataclasses.replace`, e.g.::

    from dataclasses import replace
    varied = replace(params, potential=replace(params.potential, b=0.19))

The TOML stores quark masses and the constant ``c`` in MeV; they are converted
to GeV on load. Everything inside the package is in GeV / GeV^-1.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ConfinementPotential",
    "RelativisticSmearing",
    "RelativisticFactors",
    "FineStructure",
    "AnnihilationAmplitudes",
    "CentralPotentialMethod",
    "central_potential_method",
    "GIParameters",
    "QuarkMassTable",
    "default_parameters_path",
    "load_parameters",
    "load_quark_masses",
    "load_parameters_and_quark_masses",
]

QuarkMassTable = dict[str, float]
"""Flavor-keyed constituent masses in GeV (keys ``u``, ``d``, ``q``, ``s``, ``c``, ``b``)."""


def default_parameters_path() -> Path:
    """Path to the Godfrey–Isgur parameter set shipped with the package."""
    return Path(str(resources.files("gimodel").joinpath("data", "parameters.provisional.toml")))


@dataclass(frozen=True)
class ConfinementPotential:
    """Linear-plus-constant confinement: slope ``b`` in GeV^2 and offset ``c`` in GeV."""

    b: float
    c: float


@dataclass(frozen=True)
class RelativisticSmearing:
    """Appendix A (A9) smearing inputs: ``sigma0`` in GeV and dimensionless ``s``."""

    sigma0: float
    s: float


class CentralPotentialMethod(str, Enum):
    """Construction of the spin-independent central potential (TOML ``[potential].central``).

    Only ``POINTWISE``, ``APPENDIX_A_CLOSED_FORM`` and ``APPENDIX_A_MOMENTUM_SANDWICH``
    (the production path) are implemented in this port; the remaining research
    comparators are accepted by the loader but raise ``NotImplementedError`` when used.
    """

    POINTWISE = "pointwise"
    COULOMB_1D_SMEAR = "coulomb_1d_smear"
    APPENDIX_A_SMEARING_3D = "appendix_a_smearing"
    APPENDIX_A_DERIVATIVE_G = "appendix_a_derivative_g"
    APPENDIX_A_CLOSED_FORM = "appendix_a_closed_form"
    APPENDIX_A_MOMENTUM_SANDWICH = "appendix_a_momentum_sandwich"


def central_potential_method(name: str | CentralPotentialMethod) -> CentralPotentialMethod:
    """Resolve a ``central`` TOML key to its :class:`CentralPotentialMethod`."""
    if isinstance(name, CentralPotentialMethod):
        return name
    try:
        return CentralPotentialMethod(str(name))
    except ValueError:
        options = ", ".join(sorted(m.value for m in CentralPotentialMethod))
        raise ValueError(
            f"unknown central potential method `{name}`; expected one of: {options}"
        ) from None


@dataclass(frozen=True)
class RelativisticFactors:
    """Post-(A14) relativistic epsilon factors and the switches choosing how they apply."""

    epsilon_c: float = 0.0
    epsilon_t: float = 0.0
    epsilon_so_vector: float = 0.0
    epsilon_so_scalar: float = 0.0
    contact_momentum_sandwich: bool = False
    fine_structure_momentum_sandwich: bool = False
    fine_structure_smeared_kernels: bool = False


@dataclass(frozen=True)
class FineStructure:
    """Master switch for the Appendix-A spin-orbit and tensor operators."""

    enabled: bool = True


@dataclass(frozen=True)
class AnnihilationAmplitudes:
    """Table III annihilation constants (carried for completeness; unused by this port)."""

    p1_A_np: float = 0.5
    p1_m_eta: float = 0.548
    p2_A_np: float = 0.55
    p2_M0: float = 1.17
    s1_A: float = 2.5
    a_3p2: float = -0.8


@dataclass(frozen=True)
class GIParameters:
    """The model: one field per TOML section. How it is solved lives on the solver."""

    potential: ConfinementPotential
    central: CentralPotentialMethod
    smearing: RelativisticSmearing
    factors: RelativisticFactors
    fine_structure: FineStructure
    annihilation: AnnihilationAmplitudes

    def __post_init__(self) -> None:
        object.__setattr__(self, "central", central_potential_method(self.central))

    def __str__(self) -> str:
        f, a = self.factors, self.annihilation
        return "\n".join(
            [
                "GIParameters",
                f"  potential       b = {self.potential.b} GeV², c = {self.potential.c} GeV",
                f"  central         {self.central.value}",
                f"  smearing        σ₀ = {self.smearing.sigma0} GeV, s = {self.smearing.s}",
                f"  ε factors       contact {f.epsilon_c}, tensor {f.epsilon_t}, spin-orbit vector "
                f"{f.epsilon_so_vector}, scalar {f.epsilon_so_scalar}",
                f"  sandwiches      contact {f.contact_momentum_sandwich}, fine structure "
                f"{f.fine_structure_momentum_sandwich}, smeared kernels {f.fine_structure_smeared_kernels}",
                f"  fine structure  {'enabled' if self.fine_structure.enabled else 'disabled'}",
                f"  annihilation    P1 A = {a.p1_A_np}, P2 A = {a.p2_A_np}, A(³S₁) = {a.s1_A}, "
                f"A(³P₂) = {a.a_3p2}",
            ]
        )


# --- strict TOML validation -------------------------------------------------

_BOOL_KEYS = frozenset(
    {
        "enabled",
        "contact_momentum_sandwich",
        "fine_structure_momentum_sandwich",
        "fine_structure_smeared_kernels",
    }
)


def _validate_section(
    raw: Mapping[str, Any], name: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if name not in raw:
        raise ValueError(f"missing parameter section [{name}]")
    section = raw[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"[{name}] must be a table")
    unknown = sorted(set(section) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"unknown or inactive [{name}] key(s): {', '.join(unknown)}")
    absent = [key for key in required if key not in section]
    if absent:
        raise ValueError(f"missing [{name}] key(s): {', '.join(absent)}")
    for key, value in section.items():
        if key == "central":
            continue
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"[{name}].{key} must be a boolean")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"[{name}].{key} must be a finite number")
    return section


def _validate_model_file(raw: Mapping[str, Any]) -> None:
    allowed = {
        "metadata",
        "masses",
        "potential",
        "relativistic_smearing",
        "relativistic_factors",
        "fine_structure",
        "annihilation",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown parameter section(s): {', '.join(unknown)}")
    _validate_section(raw, "potential", ("b_GeV2", "c_MeV", "central"))
    _validate_section(raw, "relativistic_smearing", ("sigma0_GeV", "s"))
    _validate_section(
        raw,
        "relativistic_factors",
        (
            "epsilon_c",
            "epsilon_t",
            "epsilon_so_vector",
            "epsilon_so_scalar",
            "contact_momentum_sandwich",
            "fine_structure_momentum_sandwich",
            "fine_structure_smeared_kernels",
        ),
    )
    _validate_section(raw, "fine_structure", ("enabled",))
    if "masses" in raw:
        _validate_section(raw, "masses", ("m_ud_avg_MeV", "m_s_MeV", "m_c_MeV", "m_b_MeV"))
    if "annihilation" in raw:
        _validate_section(
            raw, "annihilation", ("p1_A_np", "p1_m_eta_GeV", "p2_A_np", "p2_M0_GeV", "s1_A", "a_3p2")
        )


def _parameters_from_raw(raw: Mapping[str, Any]) -> GIParameters:
    _validate_model_file(raw)
    pot = raw["potential"]
    rf = raw.get("relativistic_factors", {})
    fs = raw.get("fine_structure", {})
    ann = raw.get("annihilation", {})
    legacy = [key for key in ("k_spin_orbit", "k_tensor") if key in fs]
    if legacy:
        raise ValueError(
            f"removed non-paper fine-structure setting(s): {', '.join(legacy)}; "
            "A15-A16 strengths are fixed by the paper"
        )
    return GIParameters(
        potential=ConfinementPotential(b=float(pot["b_GeV2"]), c=pot["c_MeV"] / 1000),
        central=central_potential_method(pot.get("central", "pointwise")),
        smearing=RelativisticSmearing(
            sigma0=float(raw["relativistic_smearing"]["sigma0_GeV"]),
            s=float(raw["relativistic_smearing"]["s"]),
        ),
        factors=RelativisticFactors(
            epsilon_c=float(rf.get("epsilon_c", 0.0)),
            epsilon_t=float(rf.get("epsilon_t", 0.0)),
            epsilon_so_vector=float(rf.get("epsilon_so_vector", 0.0)),
            epsilon_so_scalar=float(rf.get("epsilon_so_scalar", 0.0)),
            contact_momentum_sandwich=bool(rf.get("contact_momentum_sandwich", False)),
            fine_structure_momentum_sandwich=bool(rf.get("fine_structure_momentum_sandwich", False)),
            fine_structure_smeared_kernels=bool(rf.get("fine_structure_smeared_kernels", False)),
        ),
        fine_structure=FineStructure(enabled=bool(fs.get("enabled", True))),
        annihilation=AnnihilationAmplitudes(
            p1_A_np=float(ann.get("p1_A_np", 0.5)),
            p1_m_eta=float(ann.get("p1_m_eta_GeV", 0.548)),
            p2_A_np=float(ann.get("p2_A_np", 0.55)),
            p2_M0=float(ann.get("p2_M0_GeV", 1.17)),
            s1_A=float(ann.get("s1_A", 2.5)),
            a_3p2=float(ann.get("a_3p2", -0.8)),
        ),
    )


def _quark_masses_from_raw(raw: Mapping[str, Any]) -> QuarkMassTable:
    m = _validate_section(raw, "masses", ("m_ud_avg_MeV", "m_s_MeV", "m_c_MeV", "m_b_MeV"))
    if not all(value > 0 for value in m.values()):
        raise ValueError("constituent masses must be positive")
    light = m["m_ud_avg_MeV"] / 1000
    return {
        "u": light,
        "d": light,
        "q": light,
        "s": m["m_s_MeV"] / 1000,
        "c": m["m_c_MeV"] / 1000,
        "b": m["m_b_MeV"] / 1000,
    }


def _read_toml(path: str | Path | None) -> dict[str, Any]:
    target = default_parameters_path() if path is None else Path(path)
    with open(target, "rb") as handle:
        return tomllib.load(handle)


def load_parameters(path: str | Path | None = None) -> GIParameters:
    """Read model parameters from a TOML file (default: the bundled GI preset).

    All sections and keys are required; unknown keys are errors.
    """
    return _parameters_from_raw(_read_toml(path))


def load_quark_masses(path: str | Path | None = None) -> QuarkMassTable:
    """Read the ``[masses]`` block of a parameter TOML file, converting MeV to GeV."""
    return _quark_masses_from_raw(_read_toml(path))


def load_parameters_and_quark_masses(
    path: str | Path | None = None,
) -> tuple[GIParameters, QuarkMassTable]:
    """Read model parameters and constituent masses from one TOML file."""
    raw = _read_toml(path)
    return _parameters_from_raw(raw), _quark_masses_from_raw(raw)
