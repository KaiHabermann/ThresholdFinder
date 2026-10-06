"""Tests for the ``gi-spectrum`` command-line interface (FD solver only, small nmax)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gimodel import cli
from gimodel.cli import main

FAST = ["--nmax", "1", "--max-L", "1"]


def run(capsys, argv):
    main(argv)
    return capsys.readouterr()


def run_json(capsys, argv):
    return json.loads(run(capsys, argv + ["--json"]).out)


def fails(capsys, argv, code=1):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == code
    return capsys.readouterr()


def masses(data):
    return {s["label"]: s["mass_MeV"] for s in data["states"]}


# --- charmonium reference run ----------------------------------------------------------------


def test_charmonium_default_run(capsys):
    out = run(capsys, ["2900", "3200", "--quarks", "c", "c"]).out
    assert "GI spectrum of c c̄" in out
    assert "solver = fd" in out
    assert "parameters.provisional.toml" in out
    assert "Found 2 state(s):" in out
    assert "1^1S_0  J^PC=0^-+  mass=2966.7 MeV" in out
    assert "1^3S_1  J^PC=1^--  mass=3091.0 MeV" in out
    assert "annihilation" not in out  # heavy hidden flavour: no light-isoscalar note


def test_charmonium_reference_masses_json(capsys):
    # Same levels as the FD reference (spectrum_levels(2), S P D): J/psi includes 3S1-3D1 mixing.
    data = run_json(capsys, ["2900", "3200", "--quarks", "c", "c", "--nmax", "2", "--max-L", "2"])
    m = masses(data)
    assert m["1^1S_0"] == pytest.approx(2966.7, abs=0.1)
    assert m["1^3S_1"] == pytest.approx(3091.0, abs=0.1)


# --- flavour input paths ---------------------------------------------------------------------


def test_quarks_open_flavour(capsys):
    data = run_json(capsys, ["1800", "2100", "--quarks", "c", "u"] + FAST)
    assert (data["system"]["quark"], data["system"]["antiquark"]) == ("c", "u")
    assert data["system"]["model_flavors"] == ["c", "q"]
    assert data["system"]["self_conjugate"] is False
    assert all(s["C"] is None for s in data["states"])
    assert any("isospin-averaged" in n for n in data["notes"])


def test_quarks_reject_unknown_letter(capsys):
    err = fails(capsys, ["0", "5000", "--quarks", "c", "t"], code=2).err
    assert "Invalid quark 't'" in err


def test_net_flags(capsys):
    data = run_json(capsys, ["1900", "2200", "--c", "1", "--s", "-1"] + FAST)
    assert data["system"]["label"] == "c s̄"
    assert data["system"]["source"] == "the flavour flags"


def test_net_flags_order_quark_is_flavour_one(capsys):
    data = run_json(capsys, ["1900", "2200", "--s", "-1", "--c", "1"] + FAST)
    assert data["system"]["model_flavors"] == ["c", "s"]
    data = run_json(capsys, ["1900", "2200", "--s", "1", "--c", "-1"] + FAST)
    assert data["system"]["model_flavors"] == ["s", "c"]


def test_net_flags_hidden_flavour_suggests_quarks(capsys):
    err = fails(capsys, ["2900", "3200", "--c", "0"]).err
    assert "hidden flavour" in err
    assert "gi-spectrum 2900.0 3200.0 --quarks c c" in err


@pytest.mark.parametrize(
    "flags",
    [["--c", "1", "--u", "-1", "--s", "1"], ["--c", "2", "--u", "-1"], ["--c", "1"], ["--c", "1", "--s", "1"]],
)
def test_net_flags_not_one_quark_antiquark(capsys, flags):
    err = fails(capsys, ["2000", "3000"] + flags).err
    assert "not one quark (+1) plus one antiquark (-1)" in err
    assert "--quarks" in err


def test_flavour_is_mandatory(capsys):
    err = fails(capsys, ["2900", "3200"], code=2).err
    assert "flavour content is required" in err


@pytest.mark.parametrize(
    "extra", [["--c", "1"], ["--particles", "D0"]],
)
def test_quarks_cannot_be_mixed(capsys, extra):
    err = fails(capsys, ["2900", "3200", "--quarks", "c", "c"] + extra, code=2).err
    assert "--quarks cannot be combined" in err


def test_particles_d0_piplus_gives_c_dbar(capsys):
    out = run(capsys, ["1800", "2100", "--particles", "D0", "pi+"] + FAST).out
    assert "Reference particles: 'D0' + 'pi+'  ->  c d̄" in out
    assert "GI spectrum of c d̄" in out
    assert "J^P=0^-" in out and "J^PC" not in out


def test_particles_ds_gives_c_sbar(capsys):
    data = run_json(capsys, ["1900", "2200", "--particles", "D(s)+"] + FAST)
    assert (data["system"]["quark"], data["system"]["antiquark"]) == ("c", "s")


def test_particles_jpsi_gives_c_cbar(capsys):
    out = run(capsys, ["2900", "3200", "--particles", "J/psi(1S)", "--nmax", "2", "--max-L", "2"]).out
    assert "1^1S_0  J^PC=0^-+  mass=2966.7 MeV" in out
    assert "'J/psi(1S)'  ->  c c̄" in out
    assert "1^3S_1  J^PC=1^--  mass=3091.0 MeV" in out


def test_particles_upsilon_gives_b_bbar(capsys):
    data = run_json(capsys, ["9000", "10000", "--particles", "Upsilon(1S)", "--nmax", "1", "--max-L", "0"])
    assert data["system"]["model_flavors"] == ["b", "b"]
    assert data["system"]["self_conjugate"] is True


def test_particles_pi0_gives_light_qqbar_with_annihilation_note(capsys):
    out = run(capsys, ["600", "900", "--particles", "pi0"] + FAST).out
    assert "GI spectrum of q q̄" in out
    assert "light superposition" in out
    assert "annihilation mixing" in out
    assert "J^PC=1^--" in out


def test_ssbar_has_annihilation_note(capsys):
    data = run_json(capsys, ["900", "1100", "--quarks", "s", "s"] + FAST)
    assert any("annihilation" in n for n in data["notes"])


@pytest.mark.parametrize("name", ["eta", "eta'(958)", "phi(1020)"])
def test_particles_ambiguous_superposition(capsys, name):
    err = fails(capsys, ["500", "1100", "--particles", name]).err
    assert "ambiguous" in err
    assert "--quarks q q" in err and "--quarks s s" in err


def test_particles_kshort_ambiguous(capsys):
    err = fails(capsys, ["400", "600", "--particles", "K(S)0"]).err
    assert "ambiguous" in err and "--quarks ??? ???" in err


def test_particles_net_zero_sum_is_error(capsys):
    err = fails(capsys, ["3700", "3800", "--particles", "D0", "D~0"]).err
    assert "hidden flavour" in err and "--quarks c c" in err


def test_particles_baryon_is_error(capsys):
    err = fails(capsys, ["900", "1000", "--particles", "p"]).err
    assert "not one quark (+1) plus one antiquark (-1)" in err


def test_particles_neutral_superposition_in_sum_carries_no_flavour(capsys):
    data = run_json(capsys, ["1800", "2100", "--particles", "D0", "pi0"] + FAST)
    assert (data["system"]["quark"], data["system"]["antiquark"]) == ("c", "u")


def test_particles_unknown_name_suggestions(capsys):
    err = fails(capsys, ["2900", "3200", "--particles", "Jpsi"]).err
    assert "ERROR: Unknown particle 'Jpsi'" in err
    assert "Did you mean one of these?" in err
    assert "gi-spectrum 2900.0 3200.0 --particles 'J/psi(1S)'" in err


def test_particles_flags_override(capsys):
    """As in threshold-finder, explicit flavour flags override the derived values."""
    out = run(capsys, ["2000", "2200", "--particles", "D0", "pi+", "--d", "0", "--s", "-1"] + FAST).out
    assert "->  c s̄  (overridden by --d 0, --s -1)" in out


# --- J^P, C, window, output -------------------------------------------------------------------


def test_jp_filter(capsys):
    data = run_json(capsys, ["2000", "3200", "1", "+1", "--quarks", "c", "s", "--nmax", "1", "--max-L", "2"])
    assert {s["JPC"] for s in data["states"]} == {"1^+"}
    assert len(data["states"]) == 2  # the two mixed 1^+ P-wave states
    assert data["filter"] == {"J": 1, "P": 1, "C": None}


def test_jp_must_be_given_together(capsys):
    err = fails(capsys, ["2900", "3200", "1", "--quarks", "c", "c"], code=2).err
    assert "both J and P" in err


@pytest.mark.parametrize("J", ["0.5", "3/2"])
def test_half_integer_J_rejected(capsys, J):
    err = fails(capsys, ["2900", "3200", J, "+1", "--quarks", "c", "c"], code=2).err
    assert "integer J" in err


def test_bad_parity_rejected(capsys):
    err = fails(capsys, ["2900", "3200", "1", "2", "--quarks", "c", "c"], code=2).err
    assert "Invalid parity" in err


def test_mass_window_and_sorting(capsys):
    data = run_json(capsys, ["3000", "3600", "--quarks", "c", "c"] + FAST)
    ms = [s["mass_MeV"] for s in data["states"]]
    assert ms == sorted(ms)
    assert all(3000 <= m <= 3600 for m in ms)
    assert "1^1S_0" not in masses(data)  # eta_c at 2967 is below the window


def test_c_parity_display_and_filter(capsys):
    data = run_json(capsys, ["3000", "3700", "--quarks", "c", "c"] + FAST)
    jpc = {s["label"]: s["JPC"] for s in data["states"]}
    assert jpc["1^3P_0"] == "0^++"
    assert jpc["1^1P_1"] == "1^+-"
    assert jpc["1^3P_1"] == "1^++"
    data = run_json(capsys, ["3000", "3700", "--quarks", "c", "c", "--C", "-1"] + FAST)
    assert {s["label"] for s in data["states"]} == {"1^3S_1", "1^1P_1"}


def test_c_filter_requires_self_conjugate(capsys):
    err = fails(capsys, ["1800", "2100", "--quarks", "c", "u", "--C", "+1"]).err
    assert "self-conjugate" in err


def test_unequal_flavour_mixture_marked(capsys):
    out = run(capsys, ["2500", "2600", "1", "+1", "--quarks", "c", "s", "--nmax", "1", "--max-L", "1"]).out
    assert "J^P=1^+" in out
    assert out.count("(mixture:") == 2
    assert "% 1^1P_1" in out and "% 1^3P_1" in out


def test_details(capsys):
    out = run(capsys, ["2500", "2600", "1", "+1", "--quarks", "c", "s", "--nmax", "1", "--max-L", "1", "--details"]).out
    assert out.count("contributions [MeV]: central=") == 2
    assert "spin-orbit=" in out and "tensor=" in out and "contact=" in out
    assert out.count("components: ") == 2


def test_json_structure(capsys):
    data = run_json(capsys, ["2900", "3200", "--quarks", "c", "c", "--nmax", "1", "--max-L", "0"])
    assert set(data) >= {"system", "solver", "parameters", "nmax", "max_L", "window_MeV", "filter",
                         "notes", "warnings", "states"}
    assert data["solver"] == "fd" and data["nmax"] == 1 and data["max_L"] == 0
    state = data["states"][1]
    assert state["label"] == "1^3S_1" and state["JPC"] == "1^--"
    assert (state["n"], state["L"], state["S"], state["J"], state["P"], state["C"]) == (1, 0, 1, 1, -1, -1)
    contrib = state["contributions_MeV"]
    assert set(contrib) == {"central", "contact", "spin_orbit", "tensor", "mixing"}
    assert sum(contrib.values()) == pytest.approx(state["mass_MeV"], abs=1e-6)
    assert state["components"][0] == {"label": "1^3S_1", "coefficient": 1.0}


def test_completeness_warning(capsys):
    res = run(capsys, ["2900", "3800", "--quarks", "c", "c", "--nmax", "1", "--max-L", "0"])
    assert "WARNING" in res.err and "S (L=0)" in res.err and "--nmax" in res.err
    res = run(capsys, ["2900", "3200", "--quarks", "c", "c", "--nmax", "2", "--max-L", "0"])
    assert "WARNING" not in res.err


def test_completeness_warning_respects_jp_filter(capsys):
    # Only 0^- (S-wave singlet) matters; its n = 2 level (~3.6 GeV) is above mass_max.
    res = run(capsys, ["2900", "3500", "0", "-1", "--quarks", "c", "c", "--nmax", "2", "--max-L", "1"])
    assert "WARNING" not in res.err


def test_custom_params_path(capsys):
    from gimodel import default_parameters_path

    data = run_json(capsys, ["2900", "3200", "--quarks", "c", "c", "--nmax", "1", "--max-L", "0",
                             "--params", str(default_parameters_path())])
    assert data["parameters"] == str(default_parameters_path())


def test_missing_params_file(capsys, tmp_path):
    err = fails(capsys, ["2900", "3200", "--quarks", "c", "c", "--params", str(tmp_path / "nope.toml")]).err
    assert "parameter file not found" in err


def test_ho_not_implemented_is_clean_error(capsys, monkeypatch):
    from gimodel.solvers import RadialSolver

    class Unavailable(RadialSolver):
        nlevels_per_channel = 6

        def solve_central(self, *a):
            raise NotImplementedError("HO solver pending")

        solve_fixed_channel = solve_central

    monkeypatch.setattr(cli, "_make_solver", lambda name, nmax: Unavailable())
    err = fails(capsys, ["2900", "3200", "--quarks", "c", "c", "--solver", "ho"]).err
    assert "ERROR: the 'ho' solver is not available" in err
    assert "HO solver pending" in err and "--solver fd" in err
    assert "Traceback" not in err


def test_core_does_not_import_threshold_finder():
    import subprocess

    code = ("import sys, gimodel, gimodel.spectrum, gimodel.solvers; "
            "sys.exit(any(m in sys.modules for m in ('threshold_finder', 'particle', 'pdg', 'gimodel.pdg_match', 'gimodel.pdg_data')))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- PDG matching (--match) -------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "reference" / "cli_golden"


def _golden(name: str) -> str:
    from gimodel import default_parameters_path

    return (GOLDEN / name).read_text().replace("@PARAMS@", str(default_parameters_path()))


@pytest.mark.parametrize(
    "argv, golden",
    [
        (["2900", "3600", "--quarks", "c", "c", "--nmax", "2", "--max-L", "1", "--details"], "charmonium_details.txt"),
        (["2900", "3600", "--quarks", "c", "c", "--nmax", "2", "--max-L", "1", "--json"], "charmonium.json"),
        (["1800", "2100", "--particles", "D0", "pi+", "--nmax", "1", "--max-L", "1"], "d_mesons.txt"),
    ],
)
def test_output_without_match_is_unchanged(capsys, argv, golden):
    """Golden files were captured before --match existed: without it, output is byte-identical."""
    assert run(capsys, argv).out == _golden(golden)


CHARMONIUM = ["--quarks", "c", "c", "--nmax", "2", "--max-L", "2", "--match"]
needs_pdg = pytest.mark.skipif(__import__("importlib").util.find_spec("pdg") is None,
                               reason="PDG Python API (pdg) not installed")


def assigned(data):
    return {s["label"]: (s["match"]["assigned"] or {}).get("name") for s in data["states"]}


@needs_pdg
def test_match_charmonium_window(capsys):
    data = run_json(capsys, ["2900", "3600"] + CHARMONIUM)
    assert assigned(data) == {
        "1^1S_0": "eta_c(1S)",
        "1^3S_1": "J/psi(1S)",
        "1^3P_0": "chi_c0(1P)",
        "1^3P_1": "chi_c1(1P)",
        "1^1P_1": "h_c(1P)",
        "1^3P_2": "chi_c2(1P)",
    }
    by_label = {s["label"]: s for s in data["states"]}
    chi_c1 = by_label["1^3P_1"]["match"]["assigned"]
    assert chi_c1["delta_MeV"] == pytest.approx(3508.2 - 3510.7, abs=0.1)
    assert chi_c1["tolerance_MeV"] == 50.0 and chi_c1["tolerance_source"] == "mass_tol"
    # C separates h_c (1+-) from chi_c1 (1++), 7 MeV apart
    assert [c["name"] for c in by_label["1^1P_1"]["match"]["candidates"]] == ["h_c(1P)"]
    # the 2S predictions (3625, 3679) are above the window; eta_c(2S) at 3637.8 lies within
    # mass_max + 50 MeV and is reported as unassigned, psi(2S) at 3686.1 does not
    assert [u["name"] for u in data["unassigned_pdg"]] == ["eta_c(2S)"]
    settings = data["match_settings"]
    assert (settings["mass_tol_MeV"], settings["width_frac"], settings["status"]) == (50.0, 0.5, [0])
    assert settings["include_uncertain"] is False
    assert settings["source"]["api"] == "pdg" and int(settings["source"]["edition"]) >= 2024
    # existing keys and order are unchanged; the match keys are appended
    assert list(data)[:11] == ["system", "solver", "solver_description", "parameters", "nmax", "max_L",
                               "window_MeV", "filter", "notes", "warnings", "states"]


@needs_pdg
def test_match_charmonium_with_2s(capsys):
    data = run_json(capsys, ["2900", "3700"] + CHARMONIUM)
    got = assigned(data)
    assert got["2^1S_0"] == "eta_c(2S)" and got["2^3S_1"] == "psi(2S)"
    assert all(got.values()) and len(got) == 8
    assert data["unassigned_pdg"] == []


@needs_pdg
def test_match_charmonium_text(capsys):
    out = run(capsys, ["2900", "3600"] + CHARMONIUM).out
    assert "PDG match: |m_pred - m_PDG| <= max(50 MeV, 0.5·Γ_PDG), PDG status 0" in out
    assert "PDG data: PDG Python API (pdg " in out and " edition; only listings with known mass, J and P" in out
    assert "mass=3508.2 MeV  → chi_c1(1P) 3510.7 MeV (Δ = -2.5)" in out
    assert "→ chi_c0(1P) 3415.5 MeV (Δ = +27.3)" in out
    assert "PDG states in window without assigned prediction" in out
    assert "  eta_c(2S)  J^PC=0^-+  mass=3637.8 MeV  Γ=11.6 MeV  tol = 50.0" in out


@needs_pdg
def test_match_details_lists_candidates(capsys):
    out = run(capsys, ["3400", "3600", "1", "+1", "--C", "+1", "--quarks", "c", "c", "--nmax", "1", "--max-L", "1",
                       "--match", "--details"]).out
    assert out.count("PDG candidates (by |Δm|):") == 1
    assert "chi_c1(1P)  mass=3510.7 MeV  Γ=0.8 MeV  Δ=-2.5  tol = 50.0  assigned" in out
    assert "h_c(1P)" not in out  # C = -1: neither candidate nor in the C = + filtered window


@needs_pdg
def test_match_d_mesons_collapse_charge_states(capsys):
    argv = ["1800", "2100", "--particles", "D0", "pi+"] + FAST + ["--match"]
    data = run_json(capsys, argv)
    by_label = {s["label"]: s["match"] for s in data["states"]}
    d, dstar = by_label["1^1S_0"], by_label["1^3S_1"]
    assert [c["name"] for c in d["candidates"]] == ["D"]
    assert d["assigned"]["members"] == ["D0", "D+"]
    assert d["assigned"]["mass_MeV"] == pytest.approx((1864.84 + 1869.66) / 2, abs=0.05)
    assert [c["name"] for c in dstar["candidates"]] == ["D^*(2007)"]
    assert dstar["assigned"]["members"] == ["D^*(2007)0", "D^*(2010)+"]
    assert data["unassigned_pdg"] == []
    out = run(capsys, argv).out
    assert "→ D 1867.3 MeV (Δ = +6.2)" in out and "→ D^*(2007) 2008.6 MeV (Δ = +29.5)" in out
    assert "D+" not in out.split("Found")[1]  # no separate row per charge state


@needs_pdg
def test_match_ds_states(capsys):
    """D_s* (P unknown in the `particle` package) and D_s2*(2573) are matched."""
    data = run_json(capsys, ["2050", "2650", "--quarks", "c", "s", "--match"])
    got = assigned(data)
    assert got["1^3S_1"] == "D_s^*+"
    assert got["1^3P_2"] == "D_s2^*(2573)+"


@needs_pdg
def test_match_ds1_2700_without_mc_id(capsys):
    """D_s1*(2700) has no Monte-Carlo ID; the PDG API lists it."""
    data = run_json(capsys, ["2650", "3000", "--particles", "D~0", "K-", "--match", "--match-mass-tol", "80"])
    by_label = {s["label"]: s for s in data["states"]}
    assert by_label["2^3S_1"]["mass_MeV"] == pytest.approx(2734.3, abs=0.1)
    assert by_label["2^3S_1"]["match"]["assigned"]["name"] == "D_s1^*(2700)+"
    assert by_label["1^3D_1"]["match"]["assigned"]["name"] == "D_s1^*(2860)+"
    assert by_label["1^3D_3"]["match"]["assigned"]["name"] == "D_s3^*(2860)+"


@needs_pdg
def test_match_light_qqbar_isovector_and_isoscalar(capsys):
    data = run_json(capsys, ["700", "850", "1", "-1", "--quarks", "u", "u"] + FAST + ["--match"])
    (rho,) = data["states"]
    cands = {c["name"]: c for c in rho["match"]["candidates"]}
    assert set(cands) == {"rho(770)", "omega(782)0"}
    assert cands["rho(770)"]["isospin"] == 1.0 and cands["rho(770)"]["members"] == ["rho(770)0", "rho(770)+"]
    assert cands["omega(782)0"]["isospin"] == 0.0 and "flavour-ambiguous" in cands["omega(782)0"]["flags"]
    assert rho["match"]["assigned"]["name"] == "rho(770)"
    assert cands["rho(770)"]["tolerance_source"] == "width"  # 0.5 * 148 MeV > 50 MeV
    assert [u["name"] for u in data["unassigned_pdg"]] == ["omega(782)0"]
    # u dbar has isospin 1: the isoscalar is not a candidate
    data = run_json(capsys, ["700", "850", "1", "-1", "--quarks", "u", "d"] + FAST + ["--match"])
    assert [c["name"] for c in data["states"][0]["match"]["candidates"]] == ["rho(770)"]
    assert data["unassigned_pdg"] == []


@needs_pdg
def test_match_ssbar_flags_ambiguous(capsys):
    out = run(capsys, ["950", "1100", "--quarks", "s", "s"] + FAST + ["--match"]).out
    assert "→ phi(1020)0 1019.5 MeV (Δ = -2.4)  [flavour-ambiguous]" in out


@needs_pdg
def test_match_exotic_flagged(capsys):
    out = run(capsys, ["1500", "1700", "1", "-1", "--C", "+1", "--quarks", "u", "u"] + FAST + ["--match"]).out
    assert "Found 0 state(s):" in out
    assert "pi_1(1600)  J^PC=1^-+" in out and "exotic J^PC" in out


@needs_pdg
def test_match_tolerance_flags_change_outcome(capsys):
    base = ["3400", "3480", "0", "+1", "--quarks", "c", "c", "--nmax", "1", "--max-L", "1", "--match"]
    chi_c0 = lambda data: data["states"][0]["match"]["assigned"]  # noqa: E731
    first = chi_c0(run_json(capsys, base))
    assert first["name"] == "chi_c0(1P)" and first["delta_MeV"] == pytest.approx(27.3, abs=0.1)
    data = run_json(capsys, base + ["--match-mass-tol", "20"])
    assert chi_c0(data) is None and data["states"][0]["match"]["candidates"] == []
    assert [u["name"] for u in data["unassigned_pdg"]] == ["chi_c0(1P)"]
    # width ~12.4 MeV: 3 * Γ > 27.3 widens the window again
    got = chi_c0(run_json(capsys, base + ["--match-mass-tol", "20", "--match-width-frac", "3"]))
    assert got["name"] == "chi_c0(1P)" and got["tolerance_source"] == "width"
    assert got["tolerance_MeV"] == pytest.approx(3 * got["width_MeV"])
    out = run(capsys, base + ["--match-mass-tol", "20", "--match-width-frac", "3"]).out
    assert "(Δ = +27.3, tol = 3·Γ = " in out


@needs_pdg
def test_match_include_uncertain(capsys):
    """f_J(2220) has J = '2++ or 4' in the PDG data: used only with --match-include-uncertain."""
    base = ["2150", "2300", "4", "+1", "--quarks", "s", "s", "--nmax", "1", "--max-L", "3", "--match"]
    data = run_json(capsys, base)
    names = [c["name"] for c in data["states"][0]["match"]["candidates"]]
    assert "f_J(2220)0" not in names and data["states"][0]["match"]["assigned"]["name"] == "f_4(2300)0"
    data = run_json(capsys, base + ["--match-include-uncertain"])
    got = data["states"][0]["match"]["assigned"]
    assert got["name"] == "f_J(2220)0" and any(f.startswith("uncertain J^P") for f in got["flags"])
    assert data["match_settings"]["include_uncertain"] is True


@needs_pdg
def test_match_status_filter(capsys):
    base = ["2900", "3200", "--quarks", "c", "c", "--nmax", "1", "--max-L", "0", "--match"]
    assert all(assigned(run_json(capsys, base)).values())
    # every PDG API listing with a mass has it in the Summary Tables (status 0); nothing maps to 1
    data = run_json(capsys, base + ["--match-status", "1", "2"])
    assert not any(assigned(data).values()) and data["unassigned_pdg"] == []
    assert data["match_settings"]["status"] == [1, 2]


@pytest.mark.parametrize("flag", [["--match-mass-tol", "10"], ["--match-width-frac", "1"], ["--match-status", "1"],
                                  ["--match-include-uncertain"]])
def test_match_options_require_match(capsys, flag):
    err = fails(capsys, ["2900", "3200", "--quarks", "c", "c"] + flag, code=2).err
    assert "requires --match" in err


def test_match_negative_tolerance_rejected(capsys):
    err = fails(capsys, ["2900", "3200", "--quarks", "c", "c", "--match", "--match-mass-tol", "-1"], code=2).err
    assert ">= 0" in err


@pytest.mark.slow
def test_ho_solver_runs(capsys):
    data = run_json(capsys, ["2900", "3200", "--quarks", "c", "c", "--solver", "ho", "--nmax", "1", "--max-L", "0"])
    assert data["solver"] == "ho"
    assert [s["label"] for s in data["states"]] == ["1^1S_0", "1^3S_1"]
