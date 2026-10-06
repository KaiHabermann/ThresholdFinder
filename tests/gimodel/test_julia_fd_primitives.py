"""Layer a (tests/reference/primitives.json): running coupling, smearing widths, smeared
kernels and their derivatives for 10 mass pairs, and the angular tables, against
GIModel.jl at the primitive tolerance (1e-12 relative, absolute near zero).

Also covers the solver-independent parameter metadata of every reference file.
"""
from __future__ import annotations

import numpy as np
import pytest

from gimodel import (
    ALPHA_COEFFS,
    ALPHA_GAMMAS,
    ConstituentMasses,
    Meson,
    alpha_s_q,
    alpha_s_r,
    contact_smearing_sigma,
    l_dot_s,
    spin_dot,
    tensor_triplet_lj,
    tensor_triplet_offdiag_same_j,
)
from gimodel import coupling, fd, smearing
from gimodel.spin import fine_structure_radial_kernels

import julia_ref as jr
from julia_ref import EPS, NOISE_ULPS, PRIMITIVE, check
from scipy.special import erf

REF = jr.load_or_none("primitives.json")
PAIRS = list(REF["pairs"]) if REF else []
PAIR_ARRAYS = [
    "G_tilde",
    "S_tilde",
    "central_closed_G_plus_S",
    "G_tilde_prime",
    "G_tilde_second",
    "S_tilde_prime",
    "tensor_kernel_smeared",
    "contact_kernel_smeared",
    "delta_sigma_3d_at_sigma",
]
FS_KERNELS = ["vector_11", "vector_22", "vector_12", "scalar_11", "scalar_22", "tensor_12"]
RUNNING = list(REF["running"]) if REF else []


@pytest.fixture(scope="module")
def ref():
    return jr.load("primitives.json")


@pytest.fixture(scope="module")
def p(ref):
    return jr.params_from_dict(ref["metadata"]["parameters"])


@pytest.fixture(scope="module")
def r(ref):
    return np.array(ref["r_grid"])


# --- rounding floors of the closed forms that cancel at small r ------------------------------
#
# G~', G~'', the tensor kernel and S~' subtract terms of size ~1/r^k that cancel to O(r) or O(1)
# as r -> 0 (and the running forms likewise). Below r ~ 0.01 GeV^-1 the result is pure rounding
# noise in *both* implementations (Julia: openlibm erf; Python: cephes erf), so the comparison
# floor there is NOISE_ULPS * eps * (sum of |cancelling terms|). The measured deviations are
# <= ~1 ulp of that scale everywhere, i.e. the port reproduces Julia to rounding.

_SQRT_PI = np.sqrt(np.pi)


def _gp_scale(p, m1, m2, r):
    ri = np.maximum(np.asarray(r, dtype=float), 1e-7)
    s = 0.0
    for al, t in zip(ALPHA_COEFFS, smearing._taus(p, m1, m2)):
        e = np.abs(erf(t * ri))
        ep = 2 * t / _SQRT_PI * np.exp(-((t * ri) ** 2))
        s = s + (4 * al / 3) * (ep / ri + e / ri**2)
    return s


def _gpp_scale(p, m1, m2, r):
    ri = np.maximum(np.asarray(r, dtype=float), 1e-7)
    s = 0.0
    for al, t in zip(ALPHA_COEFFS, smearing._taus(p, m1, m2)):
        e = np.abs(erf(t * ri))
        ep = 2 * t / _SQRT_PI * np.exp(-((t * ri) ** 2))
        s = s + (4 * al / 3) * (2 * t * t * ep + 2 * ep / ri**2 + 2 * e / ri**3)
    return s


def _tensor_scale(p, m1, m2, r):
    ri = np.maximum(np.asarray(r, dtype=float), 1e-7)
    return _gp_scale(p, m1, m2, r) / ri + _gpp_scale(p, m1, m2, r)


def _sp_scale(p, m1, m2, r):
    x = np.asarray(r, dtype=float)
    sig = contact_smearing_sigma(p, m1, m2)
    small = np.abs(x) < 1e-7
    ri = np.where(small, 1.0, x)
    z = sig * ri
    ez = np.exp(-z * z)
    val = p.potential.b * (
        2 * sig * ri / _SQRT_PI * ez
        + (1 + 1 / (2 * sig**2 * ri**2)) * np.abs(erf(z))
        + (ri + 1 / (2 * sig**2 * ri)) * 2 * sig / _SQRT_PI * ez
    )
    return np.where(small, 0.0, val)


def _running_scales(r):
    ri = np.maximum(np.asarray(r, dtype=float), 1e-9)
    a = np.abs(coupling.alpha_s_r(ri))
    ap = np.abs(coupling.alpha_s_prime_r(ri))
    app = np.abs(coupling.alpha_s_second_r(ri))
    gp = (4 / 3) * (a / ri**2 + ap / ri)
    gpp = (4 / 3) * (2 * ap / ri**2 + 2 * a / ri**3 + app / ri)
    return {
        "coulomb_G_prime_running": gp,
        "coulomb_G_second_running": gpp,
        "tensor_kernel_coulomb_running": gp / ri + gpp,
    }


def noise_floor(scale):
    return NOISE_ULPS * EPS * np.asarray(scale, dtype=float)


def pair_noise(p, entry, name, r):
    m1, m2 = entry["m1_GeV"], entry["m2_GeV"]
    if name == "G_tilde_prime":
        return noise_floor(_gp_scale(p, m1, m2, r))
    if name == "G_tilde_second":
        return noise_floor(_gpp_scale(p, m1, m2, r))
    if name == "tensor_kernel_smeared":
        return noise_floor(_tensor_scale(p, m1, m2, r))
    if name == "S_tilde_prime":
        return noise_floor(_sp_scale(p, m1, m2, r))
    return None


def fs_noise(p, entry, kernel, r):
    m1, m2 = entry["m1_GeV"], entry["m2_GeV"]
    r0 = np.maximum(np.asarray(r, dtype=float), 1e-8)
    pair = {"11": (m1, m1), "22": (m2, m2), "12": (m1, m2)}[kernel[-2:]]
    if kernel.startswith("vector"):
        return noise_floor(_gp_scale(p, *pair, r0) / r0)
    if kernel.startswith("scalar"):
        return noise_floor(_sp_scale(p, *pair, r0) / r0)
    return noise_floor(_tensor_scale(p, m1, m2, r))


# --- metadata ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["primitives.json", "fd_matrices.json", "channels.json", "spectra_fd.json", "variations.json", "spot_checks.json"],
)
def test_metadata_matches_default_parameters(name):
    meta = jr.load(name)["metadata"]
    params, mq = jr.defaults()
    assert jr.params_from_dict(meta["parameters"]) == params
    assert meta["gimodel_worktree_dirty"] is False
    for flavor, mass in meta["quark_masses_GeV"].items():
        assert mq[flavor] == mass, flavor
        assert Meson.from_table(mq, flavor, flavor).constituent_masses.m1_GeV == mass


def test_constants(ref):
    check(ALPHA_COEFFS, ref["constants"]["ALPHA_COEFFS"], PRIMITIVE, "ALPHA_COEFFS")
    check(ALPHA_GAMMAS, ref["constants"]["ALPHA_GAMMAS"], PRIMITIVE, "ALPHA_GAMMAS")


# --- running coupling -------------------------------------------------------------------------


def test_alpha_s_r(ref, r):
    check(alpha_s_r(r), ref["alpha_s_r"], PRIMITIVE, "alpha_s_r")
    check([alpha_s_r(float(x)) for x in r], ref["alpha_s_r"], PRIMITIVE, "alpha_s_r scalar")


def test_alpha_s_q(ref):
    q = np.array(ref["q_grid"])
    check(alpha_s_q(q), ref["alpha_s_q"], PRIMITIVE, "alpha_s_q")


@pytest.mark.parametrize("name", RUNNING)
def test_running_forms(ref, p, r, name):
    fn = getattr(coupling, name)
    takes_params = name in ("static_coulomb_G", "static_confinement_S")
    got = fn(r, p) if takes_params else fn(r)
    noise = _running_scales(r).get(name)
    noise = None if noise is None else noise_floor(noise)
    check(got, ref["running"][name], PRIMITIVE, f"running.{name}", noise=noise)
    # Julia evaluates pointwise; the scalar path must agree as well
    scalar = [fn(float(x), p) if takes_params else fn(float(x)) for x in r]
    check(scalar, ref["running"][name], PRIMITIVE, f"running.{name} scalar", noise=noise)


# --- smeared kernels per mass pair ------------------------------------------------------------


def _masses(entry) -> ConstituentMasses:
    return ConstituentMasses(entry["m1_GeV"], entry["m2_GeV"])


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_masses_and_widths(ref, p, pair):
    entry = ref["pairs"][pair]
    params, mq = jr.defaults()
    m = Meson.from_table(mq, pair[0], pair[1]).constituent_masses
    assert (m.m1_GeV, m.m2_GeV) == (entry["m1_GeV"], entry["m2_GeV"])
    sigma = contact_smearing_sigma(p, entry["m1_GeV"], entry["m2_GeV"])
    check(sigma, entry["contact_smearing_sigma_GeV"], PRIMITIVE, f"{pair} sigma")
    check(contact_smearing_sigma(p, m), entry["contact_smearing_sigma_GeV"], PRIMITIVE, f"{pair} sigma(masses)")
    check(smearing._taus(p, m.m1_GeV, m.m2_GeV), entry["tau_k_GeV"], PRIMITIVE, f"{pair} tau_k")


def _pair_array(p, entry, name, r):
    m1, m2 = entry["m1_GeV"], entry["m2_GeV"]
    if name == "G_tilde":
        return smearing.smeared_coulomb_G_closed(p, m1, m2, r)
    if name == "S_tilde":
        return smearing.smeared_confinement_S_closed(p, m1, m2, r)
    if name == "central_closed_G_plus_S":
        return smearing.appendix_a_closed_central_values(p, m1, m2, r)
    if name == "G_tilde_prime":
        return smearing.smeared_coulomb_G_prime_closed(p, m1, m2, r)
    if name == "G_tilde_second":
        return smearing.smeared_coulomb_G_second_closed(p, m1, m2, r)
    if name == "S_tilde_prime":
        return smearing.smeared_confinement_S_prime_closed(p, m1, m2, r)
    if name == "tensor_kernel_smeared":
        return smearing.tensor_kernel_smeared_coulomb(p, m1, m2, r)
    if name == "contact_kernel_smeared":
        return smearing.smeared_contact_kernel(p, ConstituentMasses(m1, m2), r)
    if name == "delta_sigma_3d_at_sigma":
        return smearing.delta_sigma_3d(r, contact_smearing_sigma(p, m1, m2))
    raise KeyError(name)


@pytest.mark.parametrize("name", PAIR_ARRAYS)
@pytest.mark.parametrize("pair", PAIRS)
def test_pair_kernels(ref, p, r, pair, name):
    entry = ref["pairs"][pair]
    noise = pair_noise(p, entry, name, r)
    check(_pair_array(p, entry, name, r), entry[name], PRIMITIVE, f"{pair}.{name}", noise=noise)


@pytest.mark.parametrize("name", PAIR_ARRAYS)
@pytest.mark.parametrize("pair", ["cu", "bb"])
def test_pair_kernels_scalar_path(ref, p, r, pair, name):
    """Julia evaluates every kernel pointwise (scalar r): exercise the scalar branch too."""
    entry = ref["pairs"][pair]
    got = [_pair_array(p, entry, name, float(x)) for x in r]
    check(got, entry[name], PRIMITIVE, f"{pair}.{name} scalar", noise=pair_noise(p, entry, name, r))


@pytest.mark.parametrize("kernel", FS_KERNELS)
@pytest.mark.parametrize("pair", PAIRS)
def test_fine_structure_radial_kernels(ref, p, r, pair, kernel):
    entry = ref["pairs"][pair]
    got = getattr(fine_structure_radial_kernels(p, _masses(entry), r), kernel)
    noise = fs_noise(p, entry, kernel, r)
    check(got, entry["fine_structure_radial_kernels"][kernel], PRIMITIVE, f"{pair}.fs.{kernel}", noise=noise)


@pytest.mark.parametrize("pair", PAIRS)
def test_fine_structure_kernels_scalar_path(ref, p, r, pair):
    entry = ref["pairs"][pair]
    pts = [fine_structure_radial_kernels(p, _masses(entry), float(x)) for x in r]
    for kernel in FS_KERNELS:
        got = [float(getattr(k, kernel)) for k in pts]
        noise = fs_noise(p, entry, kernel, r)
        check(got, entry["fine_structure_radial_kernels"][kernel], PRIMITIVE, f"{pair}.fs.{kernel} scalar", noise=noise)


# --- angular tables -------------------------------------------------------------------------


def test_spin_dot(ref):
    for mult, value in ref["spin_dot"].items():
        assert spin_dot(int(mult)) == value, mult


def test_ldots_table(ref):
    for row in ref["LdotS"]:
        assert l_dot_s(row["L"], row["S"], row["J"]) == pytest.approx(row["LdotS"], rel=1e-15, abs=1e-15), row


def test_tensor_triplet_lj_table(ref):
    for row in ref["tensor_triplet_LJ"]:
        got = tensor_triplet_lj(row["L"], row["J"], row["S"])
        check(got, row["tensor_triplet_LJ"], PRIMITIVE, f"tensor_triplet_LJ{(row['L'], row['S'], row['J'])}")


def test_tensor_offdiag_table(ref):
    for row in ref["tensor_triplet_offdiag_sameJ"]:
        check(tensor_triplet_offdiag_same_j(row["J"], 1), row["value"], PRIMITIVE, f"offdiag J={row['J']}")


def test_side_exponent(ref):
    for row in ref["gi_spin_dependent_side_exponent"]:
        assert fd.gi_spin_dependent_side_exponent(row["epsilon"]) == row["value"], row
