"""Harmonic-oscillator (HO) radial expansion -- GI Appendix A, Eq. (A17).

Port of ``harmonic_oscillator_basis.jl`` plus the HO parts of
``sector_solver.jl``, ``channel_solver.jl``, ``fixed_channel_solver.jl``,
``contact_hyperfine.jl`` and ``spin_fine_structure.jl`` (GIModel.jl eb68b4b).

There is no spatial mesh on this path:

* ``p^2`` and ``r^2`` have exact tridiagonal matrix elements;
* position operators ``<a|g(r)|b>`` use generalised Gauss-Laguerre quadrature in
  Golub-Welsch/DVR form (``Z diag(g) Z^T``; the weights are never formed), with
  the quadrature size doubled from ``max(64, 2N)`` until the matrix moves by at
  most ``rtol * max(max|M|, 1)`` (cap ``nq_max = 8192``, with a warning);
* continuum momentum functions use the same rule with ``beta -> 1/beta`` and
  ``(-1)^n`` signs (Fourier-Bessel self-duality of the basis);
* momentum sandwiches ``B K B`` keep an auxiliary basis of ``2N + 32``.

The variational scale ``beta`` is chosen per sector by a grid scan plus golden
section (objective = highest requested level), and the basis is grown
``24 -> 80`` in steps of 8 until two consecutive refinements move every
requested level by at most ``energy_tolerance_GeV``.

QuadGK integrals (wave overlaps and the converged-wave momentum sandwich) use
:func:`quadgk`, a vectorised port of QuadGK.jl's order-7 Gauss-Kronrod
h-adaptive algorithm (same rule constants, same error/termination logic).
"""
from __future__ import annotations

import heapq
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.linalg.blas import dsyrk
from scipy.special import gammaln

from .parameters import CentralPotentialMethod, GIParameters
from .quarks import ConstituentMasses
from .smearing import (
    smeared_confinement_S_closed,
    smeared_contact_kernel,
    smeared_coulomb_G_closed,
    tensor_kernel_smeared_coulomb,
)

__all__ = [
    "generalized_laguerre",
    "ho_reduced_radial",
    "ho_r2_matrix",
    "ho_p2_matrix",
    "gauss_laguerre_dvr",
    "ho_operator_matrix",
    "ho_momentum_operator_matrix",
    "ho_momentum_sandwich_matrix",
    "oscillator_central_matrix",
    "ho_contact_hyperfine_matrix",
    "ho_fine_structure_matrices",
    "OscillatorConvergence",
    "oscillator_channel_solution",
    "oscillator_fixed_channel_matrices",
    "oscillator_fixed_channel_solution",
    "quadgk",
    "ho_cross_sandwich",
    "ho_expansion_values",
    "oscillator_tail_rho",
    "clear_ho_caches",
]

HO_REQUIRED_CONVERGED_REFINEMENTS = 2


# ---------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------


def _sym_upper(M: np.ndarray) -> np.ndarray:
    """Julia ``Symmetric(M)`` (default ``uplo = :U``): mirror the upper triangle."""
    U = np.triu(M)
    return U + np.triu(M, 1).T


def _eval(f: Callable, x: np.ndarray) -> np.ndarray:
    """Evaluate ``f`` on an array, vectorised when it supports it."""
    try:
        v = np.asarray(f(x), dtype=float)
    except (TypeError, ValueError):
        v = None
    if v is not None:
        if v.shape == x.shape:
            return v
        if v.ndim == 0:
            return np.full(x.shape, float(v))
    return np.array([float(f(float(t))) for t in x])


class _LRU:
    """Small bounded memo (exact: same key -> same computation -> same bits)."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.data: OrderedDict = OrderedDict()

    def get(self, key, factory):
        try:
            value = self.data[key]
            self.data.move_to_end(key)
            return value
        except KeyError:
            pass
        except TypeError:  # unhashable key: no memo
            return factory()
        value = factory()
        self.data[key] = value
        if len(self.data) > self.maxsize:
            self.data.popitem(last=False)
        return value

    def clear(self) -> None:
        self.data.clear()


# ---------------------------------------------------------------------------------
# basis functions and exact matrices
# ---------------------------------------------------------------------------------


def generalized_laguerre(n: int, alpha: float, x):
    """``L_n^alpha(x)`` by the three-term recurrence (vectorised in ``x``)."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x) if x.ndim else 1.0
    lm2 = np.ones_like(x)
    lm1 = 1.0 + alpha - x
    for k in range(2, n + 1):
        lk = ((2 * k - 1 + alpha - x) * lm1 - (k - 1 + alpha) * lm2) / k
        lm2, lm1 = lm1, lk
    return lm1 if x.ndim else float(lm1)


def _ho_norm(nr: int, L: int, beta: float) -> float:
    return math.sqrt(2 * beta * math.exp(gammaln(nr + 1) - gammaln(nr + L + 1.5)))


def ho_reduced_radial(nr: int, L: int, beta: float, r):
    """Normalised reduced radial HO function ``u_{nr,L}(r)`` (``int u^2 dr = 1``)."""
    r = np.asarray(r, dtype=float)
    rho = beta * r
    x = rho**2
    val = _ho_norm(nr, L, beta) * rho ** (L + 1) * np.exp(-0.5 * x) * generalized_laguerre(nr, L + 0.5, x)
    return val if r.ndim else float(val)


def _ho_basis_values(L: int, beta: float, n: int, r: np.ndarray) -> np.ndarray:
    """``U[i, k] = ho_reduced_radial(k, L, beta, r_i)`` for ``k < n`` (one recurrence pass)."""
    r = np.asarray(r, dtype=float)
    rho = beta * r
    x = rho**2
    alpha = L + 0.5
    env = rho ** (L + 1) * np.exp(-0.5 * x)
    U = np.empty((len(r), n))
    lm2 = np.ones_like(x)
    U[:, 0] = _ho_norm(0, L, beta) * env * lm2
    if n > 1:
        lm1 = 1.0 + alpha - x
        U[:, 1] = _ho_norm(1, L, beta) * env * lm1
        for k in range(2, n):
            lk = ((2 * k - 1 + alpha - x) * lm1 - (k - 1 + alpha) * lm2) / k
            lm2, lm1 = lm1, lk
            U[:, k] = _ho_norm(k, L, beta) * env * lk
    return U


def _ho_basis_derivatives(L: int, beta: float, n: int, r: np.ndarray) -> np.ndarray:
    """``d/dr ho_reduced_radial(k, L, beta, r)`` for ``k < n`` (Julia formula, incl. ``r = 0``)."""
    r = np.asarray(r, dtype=float)
    rho = beta * r
    x = rho**2
    alpha = L + 0.5
    env = rho ** (L + 1) * np.exp(-0.5 * x)
    zero = r == 0
    rs = np.where(zero, 1.0, r)
    out = np.empty((len(r), n))
    for k in range(n):
        nrm = _ho_norm(k, L, beta)
        lag = np.asarray(generalized_laguerre(k, alpha, x), dtype=float)
        dlag = 0.0 if k == 0 else -np.asarray(generalized_laguerre(k - 1, alpha + 1, x), dtype=float)
        val = nrm * env * (((L + 1) / rs - beta**2 * rs) * lag + 2 * beta**2 * rs * dlag)
        at0 = nrm * beta * float(generalized_laguerre(k, alpha, 0.0)) if L == 0 else 0.0
        out[:, k] = np.where(zero, at0, val)
    return out


def _r2_tridiagonal(L: int, beta: float, nbasis: int) -> tuple[np.ndarray, np.ndarray]:
    ib2 = 1 / float(beta) ** 2
    n = np.arange(nbasis, dtype=float)
    d = ib2 * (2 * n + L + 1.5)
    k = np.arange(nbasis - 1, dtype=float)
    e = -ib2 * np.sqrt((k + 1) * (k + L + 1.5))
    return d, e


def _tridiag_dense(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


def ho_r2_matrix(L: int, beta: float, nbasis: int) -> np.ndarray:
    """Exact ``<n|r^2|m>`` (tridiagonal, returned dense)."""
    if nbasis < 1:
        raise ValueError("ho_r2_matrix: nbasis must be ≥ 1")
    if not beta > 0:
        raise ValueError("ho_r2_matrix: β must be positive")
    return _tridiag_dense(*_r2_tridiagonal(L, beta, nbasis))


def ho_p2_matrix(L: int, beta: float, nbasis: int) -> np.ndarray:
    """Exact ``<n|p^2|m>`` (tridiagonal, returned dense): ``beta^2 (2n+L+3/2)`` and ``beta^2 sqrt((n+1)(n+L+3/2))``."""
    if nbasis < 1:
        raise ValueError("ho_p2_matrix: nbasis must be ≥ 1")
    if not beta > 0:
        raise ValueError("ho_p2_matrix: β must be positive")
    b2 = float(beta) ** 2
    n = np.arange(nbasis, dtype=float)
    k = np.arange(nbasis - 1, dtype=float)
    return _tridiag_dense(b2 * (2 * n + L + 1.5), b2 * np.sqrt((k + 1) * (k + L + 1.5)))


# ---------------------------------------------------------------------------------
# Golub-Welsch DVR position operators
# ---------------------------------------------------------------------------------

# Rule store keyed by (L, nq): sqrt of the nodes and the leading rows of the
# Jacobi eigenvectors. The rule is beta-independent (x = (beta r)^2), so it is
# computed once per (L, nq). Only the leading rows are kept (all of them when
# nq is small), which bounds memory for the 8192+ node rules.
_DVR: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_DVR_MAX_BYTES = 512 * 2**20
_DVR_MIN_ROWS = 256


def _dvr_bytes() -> int:
    return sum(v[0].nbytes + v[1].nbytes for v in _DVR.values())


def gauss_laguerre_dvr(L: int, nbasis: int, nq: int) -> tuple[np.ndarray, np.ndarray]:
    """``(sqrt_x, Z)`` of the generalised Gauss-Laguerre rule with weight ``x^(L+1/2) e^-x``.

    ``Z = V[:nbasis, :]`` are the leading rows of the eigenvectors of
    ``ho_r2_matrix(L, 1, nq)``: entries ``sqrt(w_i) p_n(x_i)``; the weights are
    never formed. Signs are LAPACK-arbitrary and cancel in every use.
    """
    key = (int(L), int(nq))
    rows = min(int(nbasis), int(nq))
    entry = _DVR.get(key)
    if entry is None or entry[1].shape[0] < rows:
        if _dvr_bytes() > _DVR_MAX_BYTES:
            _DVR.clear()
        d, e = _r2_tridiagonal(L, 1.0, nq)
        # Tridiagonal MRRR (LAPACK stemr), as Julia's eigen(::SymTridiagonal) (stegr).
        w, v = eigh_tridiagonal(d, e, lapack_driver="stemr")
        keep = min(int(nq), max(rows, _DVR_MIN_ROWS))
        entry = (np.sqrt(np.maximum(w, 0.0)), np.ascontiguousarray(v[:keep, :]))
        del v
        _DVR[key] = entry
    return entry[0], entry[1][:rows, :]


# Columns (nodes) whose leading-row eigenvector entries are all below this bound
# contribute below 1e-50 |g| to any element -- far beneath the eigensolver's own
# absolute accuracy (~1e-16) on those entries -- and are skipped. For the large
# rules that diffuse beta needs (nq = 8192+) this removes ~85% of the work.
_DVR_COLUMN_FLOOR = 1e-25
_DVR_COLUMNS: dict[tuple[int, int, int], int] = {}


def _dvr_active(L: int, nbasis: int, nq: int) -> tuple[np.ndarray, np.ndarray]:
    sqrt_x, Z = gauss_laguerre_dvr(L, nbasis, nq)
    key = (int(L), int(nbasis), int(nq))
    k = _DVR_COLUMNS.get(key)
    if k is None:
        significant = np.nonzero(np.max(np.abs(Z), axis=0) > _DVR_COLUMN_FLOOR)[0]
        k = int(significant[-1]) + 1 if len(significant) else Z.shape[1]
        _DVR_COLUMNS[key] = k
    return sqrt_x[:k], Z[:, :k]


def _ho_operator_matrix_at(L: int, beta: float, nbasis: int, g: Callable, nq: int) -> np.ndarray:
    """Upper triangle of ``Z diag(g(r_i)) Z^T`` (the lower triangle is left zero).

    Julia forms the full product and keeps the upper triangle (``Symmetric``);
    here a symmetric rank-k update (``dsyrk``) on ``Z sqrt|g|`` builds only that
    triangle -- the same sums up to rounding at half the cost.
    """
    sqrt_x, Z = _dvr_active(L, nbasis, nq)
    r = sqrt_x / float(beta)
    gv = _eval(g, r)
    pos = gv >= 0
    if pos.all():
        return dsyrk(1.0, Z * np.sqrt(gv))
    if not pos.any():
        return dsyrk(-1.0, Z * np.sqrt(-gv))
    C = dsyrk(1.0, Z[:, pos] * np.sqrt(gv[pos]))
    return dsyrk(-1.0, Z[:, ~pos] * np.sqrt(-gv[~pos]), beta=1.0, c=C, overwrite_c=True)


_WARNED_CAP: set = set()


def ho_operator_matrix(
    L: int,
    beta: float,
    nbasis: int,
    g: Callable,
    *,
    rtol: float = 1e-10,
    nq: int = 64,
    nq_max: int = 8192,
) -> np.ndarray:
    """``<a|g(r)|b>`` in the 3D HO basis by Golub-Welsch quadrature (no mesh).

    ``nq`` starts at ``max(nq, 2 nbasis)`` and doubles while ``nq < nq_max`` until
    ``max|cur - prev| <= rtol * max(max|cur|, 1)``; otherwise the last (largest-nq)
    matrix is returned with a warning, exactly as GIModel.jl does.
    """
    if nbasis < 1:
        raise ValueError("ho_operator_matrix: nbasis must be ≥ 1")
    if not beta > 0:
        raise ValueError("ho_operator_matrix: β must be positive")
    n = max(int(nq), 2 * int(nbasis))
    prev = _ho_operator_matrix_at(L, beta, nbasis, g, n)
    while n < nq_max:
        n *= 2
        cur = _ho_operator_matrix_at(L, beta, nbasis, g, n)
        scale = max(float(np.max(np.abs(cur))), 1.0)
        if float(np.max(np.abs(cur - prev))) <= rtol * scale:
            return _sym_upper(cur)
        prev = cur
    if "cap" not in _WARNED_CAP:  # Julia: @warn ... maxlog = 1
        _WARNED_CAP.add("cap")
        warnings.warn(
            f"ho_operator_matrix: quadrature did not reach rtol={rtol} by nq={nq_max} "
            f"(L={L}, β={beta}, nbasis={nbasis}). Diffuse β needs the most nodes; the result is "
            "the largest-nq value, not a converged one.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _sym_upper(prev)


# Memo of operator matrices by an explicit, hashable description of g. Exact.
_OPS = _LRU(4096)


def _cached_operator(key: Hashable | None, L, beta, nbasis, g, rtol) -> np.ndarray:
    if key is None:
        return ho_operator_matrix(L, beta, nbasis, g, rtol=rtol)
    full = ("pos", key, int(L), float(beta), int(nbasis), float(rtol))
    return _OPS.get(full, lambda: ho_operator_matrix(L, beta, nbasis, g, rtol=rtol))


def _signs(n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 1, -1.0, 1.0)


def ho_momentum_operator_matrix(
    L: int, beta: float, nbasis: int, f: Callable, *, rtol: float = 1e-10, key: Hashable | None = None
) -> np.ndarray:
    """Continuum momentum function ``<a|f(p)|b>``: ``S ho_operator_matrix(L, 1/beta, N, f) S``,
    ``S = diag((-1)^n)`` (Julia ``_ho_momentum_operator_matrix``)."""

    def build():
        M = ho_operator_matrix(L, 1.0 / float(beta), nbasis, f, rtol=rtol)
        s = _signs(nbasis)
        return s[:, None] * M * s[None, :]

    if key is None:
        return build()
    return _OPS.get(("mom", key, int(L), float(beta), int(nbasis), float(rtol)), build)


def side_factor(m1: float, m2: float, exponent: float) -> Callable:
    """``B(p) = (m1 m2 / sqrt((p^2+m1^2)(p^2+m2^2)))^exponent``."""

    def B(p):
        return (m1 * m2 / np.sqrt((p**2 + m1**2) * (p**2 + m2**2))) ** exponent

    return B


def ho_momentum_sandwich_matrix(
    L: int,
    beta: float,
    nbasis: int,
    m1: float,
    m2: float,
    exponent: float,
    kernel: Callable,
    *,
    rtol: float = 1e-10,
    kernel_key: Hashable | None = None,
) -> np.ndarray:
    """``B kernel(r) B`` with ``B = (m1 m2/(E1 E2))^exponent`` (``exponent = 1/2 + epsilon``).

    ``B`` (default ``rtol``) and the kernel (``rtol``) live on a ``2N + 32`` auxiliary
    basis; the product is projected back onto ``N`` functions.
    """
    work = 2 * int(nbasis) + 32
    m1, m2, exponent = float(m1), float(m2), float(exponent)
    B = ho_momentum_operator_matrix(L, beta, work, side_factor(m1, m2, exponent), key=("B", m1, m2, exponent))
    K = _cached_operator(kernel_key, L, beta, work, kernel, rtol)
    C = B[:, :nbasis]
    return _sym_upper((C.T @ K) @ C)


def _cached_sandwich(L, beta, nbasis, m1, m2, exponent, kernel, rtol, kernel_key):
    def build():
        return ho_momentum_sandwich_matrix(
            L, beta, nbasis, m1, m2, exponent, kernel, rtol=rtol, kernel_key=kernel_key
        )

    if kernel_key is None:
        return build()
    key = ("sandwich", kernel_key, int(L), float(beta), int(nbasis), float(m1), float(m2), float(exponent), float(rtol))
    return _OPS.get(key, build)


# ---------------------------------------------------------------------------------
# Hamiltonian pieces
# ---------------------------------------------------------------------------------


def oscillator_central_matrix(
    params: GIParameters, masses: ConstituentMasses, L: int, beta: float, nbasis: int
) -> np.ndarray:
    """Native Eq. (A17) spin-independent matrix ``K(p) + A G~ A + S~`` (``A``, ``G~`` on ``2N+32``)."""
    if params.central != CentralPotentialMethod.APPENDIX_A_MOMENTUM_SANDWICH:
        raise ValueError(
            "OscillatorSolver implements only the native Appendix-A momentum-sandwich central "
            f"Hamiltonian; use FiniteDifferenceSolver for `{params.central.value}`"
        )
    m1, m2 = float(masses.m1_GeV), float(masses.m2_GeV)
    key = ("central", params, m1, m2, int(L), float(beta), int(nbasis))

    def build():
        kinetic = ho_momentum_operator_matrix(
            L, beta, nbasis, lambda p: np.sqrt(p**2 + m1**2) + np.sqrt(p**2 + m2**2), key=("kin", m1, m2)
        )
        work = 2 * nbasis + 32
        A = ho_momentum_operator_matrix(
            L, beta, work, lambda p: np.sqrt(1 + p**2 / np.sqrt((p**2 + m1**2) * (p**2 + m2**2))), key=("A", m1, m2)
        )
        G = _cached_operator(
            ("G", params, m1, m2), L, beta, work, lambda r: smeared_coulomb_G_closed(params, m1, m2, r), 1e-10
        )
        S = _cached_operator(
            ("S", params, m1, m2), L, beta, nbasis, lambda r: smeared_confinement_S_closed(params, m1, m2, r), 1e-10
        )
        C = A[:, :nbasis]
        return _sym_upper(kinetic + (C.T @ G) @ C + S)

    return _OPS.get(key, build)


def _contact_radial_matrix(params, masses: ConstituentMasses, L: int, beta: float, nbasis: int, sandwich: bool):
    from .fd import gi_spin_dependent_side_exponent

    m1, m2 = float(masses.m1_GeV), float(masses.m2_GeV)
    kernel_key = ("contact", params, m1, m2)

    def kernel(r):
        return smeared_contact_kernel(params, masses, r)

    if sandwich:
        exponent = gi_spin_dependent_side_exponent(params.factors.epsilon_c)
        return _cached_sandwich(L, beta, nbasis, m1, m2, exponent, kernel, 1e-8, kernel_key)
    return _cached_operator(kernel_key, L, beta, nbasis, kernel, 1e-8)


def ho_contact_hyperfine_matrix(
    params: GIParameters, masses: ConstituentMasses, L: int, multiplicity: int, beta: float, nbasis: int
) -> np.ndarray:
    """``strength * B_c K B_c`` (or ``strength * K`` locally), kernel at ``rtol = 1e-8``."""
    from .spin import ContactHyperfine

    if L < 0:
        raise ValueError("contact matrix requires L >= 0")
    op = ContactHyperfine(params, masses, multiplicity)
    radial = _contact_radial_matrix(params, masses, L, beta, nbasis, bool(op.sandwich))
    return _sym_upper(op.strength * radial)


def _fine_sandwiches(params, masses: ConstituentMasses, L: int, beta: float, nbasis: int):
    """The six J-independent radial sandwiches ``G11, G22, G12, S11, S22, T12``."""
    from .fd import gi_spin_dependent_side_exponent as side
    from .spin import _pair, _scalar_so_kernel, _vector_so_kernel

    f = params.factors
    m1, m2 = float(masses.m1_GeV), float(masses.m2_GeV)
    p11, p22 = _pair(m1), _pair(m2)

    def sw(kind, pair, eps, kernel, rtol=1e-10):
        a, b = float(pair.m1_GeV), float(pair.m2_GeV)
        return _cached_sandwich(L, beta, nbasis, a, b, side(eps), kernel, rtol, (kind, params, a, b))

    G11 = sw("vso", p11, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p11, r))
    G22 = sw("vso", p22, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, p22, r))
    G12 = sw("vso", masses, f.epsilon_so_vector, lambda r: _vector_so_kernel(params, masses, r))
    S11 = sw("sso", p11, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p11, r))
    S22 = sw("sso", p22, f.epsilon_so_scalar, lambda r: _scalar_so_kernel(params, p22, r))
    # Narrow tensor kernel: certified at rtol = 1e-8 (as GIModel.jl).
    T12 = sw("tensor", masses, f.epsilon_t, lambda r: tensor_kernel_smeared_coulomb(params, masses, r), 1e-8)
    return G11, G22, G12, S11, S22, T12


def ho_fine_structure_matrices(
    params: GIParameters, masses: ConstituentMasses, L: int, multiplicity: int, J: int, beta: float, nbasis: int
):
    """Native HO vector/Thomas spin-orbit and tensor matrices of one fixed ``(L,S,J)`` sector."""
    from .spin import FineStructureMatrices, _fine_structure_algebra, _zero_fine_matrices, diagonal_fine_structure_active

    if not diagonal_fine_structure_active(params, L, multiplicity):
        return _zero_fine_matrices(nbasis)
    f = params.factors
    if not (f.fine_structure_momentum_sandwich and f.fine_structure_smeared_kernels):
        raise ValueError("native HO fine structure requires smeared Appendix-A momentum sandwiches")
    parts = _fine_structure_algebra(masses, L, J, *_fine_sandwiches(params, masses, L, beta, nbasis))
    return FineStructureMatrices(*(_sym_upper(p) for p in parts))


# ---------------------------------------------------------------------------------
# QuadGK port (order-7 Gauss-Kronrod, h-adaptive, vectorised evaluation)
# ---------------------------------------------------------------------------------

# QuadGK.cachedrule(Float64, 7), dumped from Julia (exact values).
_GK_X = np.array([
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691, -0.7415311855993945,
    -0.5860872354676911, -0.4058451513773972, -0.20778495500789848, 0.0,
])
_GK_W = np.array([
    0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592,
    0.1690047266392679, 0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
])
_GK_G = np.array([0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694])
# Evaluation order: centre, x[7] pair, then for i = 1..3 the x[2i] and x[2i-1] pairs.
_GK_PAIRS = [6, 1, 0, 3, 2, 5, 4]  # 0-based indices into _GK_X, in Julia's order


def _gk_points(a: float, b: float) -> tuple[float, np.ndarray]:
    s = 0.5 * (b - a)
    pts = [a + s]
    for j in _GK_PAIRS:
        pts.append(a + (1 + _GK_X[j]) * s)
        pts.append(a + (1 - _GK_X[j]) * s)
    return s, np.array(pts)


def _gk_combine(a: float, b: float, s: float, fv: np.ndarray):
    """Julia ``evalrule`` arithmetic (odd branch, n = 7) on precomputed values."""
    x, w, wg = _GK_X, _GK_W, _GK_G
    f0 = fv[0]
    Ig = f0 * wg[3]
    Ik = f0 * w[7] + (fv[1] + fv[2]) * w[6]
    k = 3
    for i in range(1, 4):  # Julia i = 1:3
        fg = fv[k] + fv[k + 1]  # x[2i]
        fk = fv[k + 2] + fv[k + 3]  # x[2i-1]
        k += 4
        Ig += fg * wg[i - 1]
        Ik += fg * w[2 * i - 1] + fk * w[2 * i - 2]
    Ik_s, Ig_s = Ik * s, Ig * s
    E = abs(Ik_s - Ig_s)
    if not math.isfinite(E):
        raise FloatingPointError(f"integrand produced {E} in the interval ({a}, {b})")
    return Ik_s, E


def _endpoint_roundoff(a: float, b: float) -> bool:
    c = 0.5 * (b - a)
    return a == a + (1 + _GK_X[0]) * c or b == a + (1 - _GK_X[0]) * c


def quadgk(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, *, rtol: float | None = None,
           atol: float | None = None, maxevals: int = 10**7) -> tuple[float, float]:
    """Port of QuadGK.jl ``quadgk(f, a, b)`` (order 7) for a real scalar integrand.

    ``f`` is called with arrays of abscissae (15 or 30 at a time). Termination,
    error estimate and largest-error-first bisection follow QuadGK 2.11.
    """
    a, b = float(a), float(b)
    n = 7
    s, pts = _gk_points(a, b)
    I, E = _gk_combine(a, b, s, np.asarray(f(pts), dtype=float))
    numevals = 2 * n + 1
    atol_ = 0.0 if atol is None else float(atol)
    rtol_ = (math.sqrt(np.finfo(float).eps) if atol_ == 0 else 0.0) if rtol is None else float(rtol)
    if numevals >= maxevals or E <= atol_ or E <= rtol_ * abs(I):
        return I, E
    heap = [(-E, 0, a, b, I)]
    counter = 1
    while E > atol_ and E > rtol_ * abs(I) and numevals < maxevals:
        negE, _, sa, sb, sI = heapq.heappop(heap)
        mid = (sa + sb) / 2
        if _endpoint_roundoff(sa, mid) or _endpoint_roundoff(mid, sb):
            heapq.heappush(heap, (negE, counter, sa, sb, sI))
            break
        s1, p1 = _gk_points(sa, mid)
        s2, p2 = _gk_points(mid, sb)
        fv = np.asarray(f(np.concatenate([p1, p2])), dtype=float)
        I1, E1 = _gk_combine(sa, mid, s1, fv[:15])
        I2, E2 = _gk_combine(mid, sb, s2, fv[15:])
        I = (I - sI) + I1 + I2
        E = (E - (-negE)) + E1 + E2
        numevals += 4 * n + 2
        heapq.heappush(heap, (-E1, counter, sa, mid, I1))
        heapq.heappush(heap, (-E2, counter + 1, mid, sb, I2))
        counter += 2
    # re-sum (QuadGK resum)
    I = heap[0][4]
    E = -heap[0][0]
    for item in heap[1:]:
        I += item[4]
        E += -item[0]
    return I, E


# ---------------------------------------------------------------------------------
# converged-wave momentum sandwiches (Julia _ho_cross_sandwich)
# ---------------------------------------------------------------------------------


def oscillator_tail_rho(L: int, nbasis: int) -> float:
    """Dimensionless integration cutoff: ten units beyond the classical turning radius."""
    return math.sqrt(4 * (nbasis - 1) + 2 * L + 3) + 10


_RECURRENCE: dict[tuple[int, int], tuple[list, list, list]] = {}


def _recurrence_constants(L: int, n: int):
    key = (int(L), int(n))
    consts = _RECURRENCE.get(key)
    if consts is None:
        alpha = L + 0.5
        a = [2 * k + alpha + 1 for k in range(n)]
        b = [math.sqrt(k * (k + alpha)) for k in range(n)]
        d = [math.sqrt((k + 1) * (k + alpha + 1)) for k in range(n)]
        consts = _RECURRENCE[key] = (a, b, d)
    return consts


def ho_expansion_values(L: int, beta: float, coefficients: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``sum_n c_n u_{n,L}(r)`` by one normalised-Laguerre recurrence with 1e100 rescaling
    (vectorised port of Julia ``_ho_expansion_value``; safe for thousands of functions)."""
    r = np.asarray(r, dtype=float)
    c = [float(v) for v in coefficients]
    rho = beta * r
    x = rho**2
    previous = np.zeros_like(x)
    current = math.sqrt(2 * beta / math.gamma(L + 1.5)) * rho ** (L + 1)
    value = c[0] * current
    logscale = None
    a, b, d = _recurrence_constants(L, len(c))
    for n in range(len(c) - 1):
        following = ((a[n] - x) * current - b[n] * previous) / d[n]
        previous, current = current, following
        value = value + c[n + 1] * current
        # Julia tests max(|previous|, |current|, |value|) > 1e100 per point; |previous|
        # already passed that test on the previous step, so two reductions suffice here.
        if np.abs(current).max() > 1e100 or np.abs(value).max() > 1e100:
            m = np.maximum(np.maximum(np.abs(previous), np.abs(current)), np.abs(value)) > 1e100
            previous = np.where(m, previous * 1e-100, previous)
            current = np.where(m, current * 1e-100, current)
            value = np.where(m, value * 1e-100, value)
            step = np.where(m, math.log(1e100), 0.0)
            logscale = step if logscale is None else logscale + step
    if logscale is None:
        return value * np.exp(-x / 2)
    return value * np.exp(logscale - x / 2)


def _ho_momentum_transformed(wave, m1: float, m2: float, exponent: float, nbasis: int) -> np.ndarray:
    """Coefficients of ``B(p) |wave>`` on ``nbasis`` functions via the ``nq = nbasis`` DVR rule."""
    nodes, Z = gauss_laguerre_dvr(wave.L, nbasis, nbasis)
    lam = (wave.beta * nodes) ** 2
    factors = (m1 * m2 / np.sqrt((lam + m1**2) * (lam + m2**2))) ** exponent
    signs = _signs(nbasis)
    n = len(wave.coefficients)
    projected = Z[:n, :].T @ (signs[:n] * wave.coefficients)
    return signs * (Z @ (factors * projected))


def ho_cross_sandwich(left, right, m1: float, m2: float, exponent: float, kernel: Callable) -> float:
    """``<left| B kernel(r) B |right> / (|left||right|)`` for two OscillatorWaves.

    Fixed input coefficients, auxiliary space ``nextpow2(max(64, N))`` doubled up to
    2048 until two consecutive integrals agree (``rtol = 1e-6, atol = 1e-9``); each
    integral by :func:`quadgk` (``rtol = 1e-9, atol = 1e-12``) on ``[0, r_tail]``.
    """
    m1, m2, exponent = float(m1), float(m2), float(exponent)
    nmax = max(64, len(left.coefficients), len(right.coefficients))
    nbasis = 1 << (nmax - 1).bit_length()
    previous = None
    consecutive = 0
    delta = math.inf
    same = left is right
    while nbasis <= 2048:
        c_left = _ho_momentum_transformed(left, m1, m2, exponent, nbasis)
        c_right = c_left if same else _ho_momentum_transformed(right, m1, m2, exponent, nbasis)
        rmax = max(oscillator_tail_rho(left.L, nbasis) / left.beta, oscillator_tail_rho(right.L, nbasis) / right.beta)

        def integrand(r):
            vl = ho_expansion_values(left.L, left.beta, c_left, r)
            vr = vl if same else ho_expansion_values(right.L, right.beta, c_right, r)
            return vl * vr * _eval(kernel, r)

        value, _ = quadgk(integrand, 0.0, rmax, rtol=1e-9, atol=1e-12)
        if previous is not None:
            delta = abs(value - previous)
            close = delta <= max(1e-9, 1e-6 * max(abs(value), abs(previous)))
            consecutive = consecutive + 1 if close else 0
            if consecutive >= 2:
                return value / math.sqrt(left.wave_norm() * right.wave_norm())
        previous = value
        nbasis *= 2
    raise ArithmeticError(
        "HO momentum sandwich did not converge through 2048 auxiliary functions "
        f"(beta={left.beta}, {right.beta}, exponent={exponent}, last change={delta}, value={previous})"
    )


# ---------------------------------------------------------------------------------
# beta search and basis growth
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class OscillatorConvergence:
    """Numerical certificate of an HO solve (``status`` ``"converged"`` or ``"unchecked"``)."""

    status: str
    beta_GeV: float
    nbasis: int
    energy_delta_GeV: float | None
    max_wave_overlap_defect: float | None
    tolerance_GeV: float
    refinements: int

    def __post_init__(self) -> None:
        if self.status not in ("converged", "unchecked"):
            raise ValueError("OscillatorConvergence: status must be converged or unchecked")
        if not self.beta_GeV > 0:
            raise ValueError("OscillatorConvergence: beta must be positive")
        if self.nbasis < 1:
            raise ValueError("OscillatorConvergence: nbasis must be positive")
        if self.status == "converged" and (self.energy_delta_GeV is None or self.max_wave_overlap_defect is None):
            raise ValueError("OscillatorConvergence: converged status requires final deltas")


class _Eval:
    __slots__ = ("values", "vectors", "extra")

    def __init__(self, values, vectors, extra=None):
        self.values = values
        self.vectors = vectors
        self.extra = extra


def _objective(result: _Eval) -> float:
    return float(result.values[-1])


def _cached_evaluation(cache: dict, evaluate, beta: float) -> _Eval:
    beta_f = float(beta)
    if beta_f not in cache:
        result = evaluate(beta_f)
        if not np.all(np.isfinite(result.values)):
            raise ArithmeticError(f"oscillator beta search produced non-finite eigenvalues at beta={beta_f}")
        cache[beta_f] = result
    return cache[beta_f]


def _best_cached(cache: dict) -> tuple[float, _Eval]:
    betas = sorted(cache)
    best_beta, best = betas[0], cache[betas[0]]
    for beta in betas[1:]:
        if _objective(cache[beta]) < _objective(best):
            best_beta, best = beta, cache[beta]
    return best_beta, best


def _local_beta_grid_index(cache, evaluate, grid, seed) -> int:
    """0-based port of ``_local_beta_grid_index!``."""
    n = len(grid)
    if seed is None:
        for beta in grid:
            _cached_evaluation(cache, evaluate, beta)
        objectives = [_objective(cache[beta]) for beta in grid]
        return int(np.argmin(objectives))
    index = int(np.argmin(np.abs(np.asarray(grid) - seed)))
    index = min(max(index, 1), n - 2)
    while True:
        for i in range(index - 1, min(index + 1, n - 1) + 1):
            _cached_evaluation(cache, evaluate, grid[i])
        center = _objective(cache[grid[index]])
        left = _objective(cache[grid[index - 1]])
        right = _objective(cache[grid[index + 1]])
        if left < center and left <= right:
            index -= 1
        elif right < center and right < left:
            index += 1
        else:
            return index
        if not 0 < index < n - 1:
            return index


def _refine_oscillator_beta(evaluate, solver, seed=None) -> tuple[float, _Eval]:
    grid = solver.beta_grid
    first_beta = grid[0] if seed is None else min(max(float(seed), grid[0]), grid[-1])
    first = evaluate(first_beta)
    if not np.all(np.isfinite(first.values)):
        raise ArithmeticError(f"oscillator beta search produced non-finite eigenvalues at beta={first_beta}")
    cache = {float(first_beta): first}
    if len(grid) == 1:
        return float(first_beta), first
    index = _local_beta_grid_index(cache, evaluate, grid, seed)
    if index == 0 or index == len(grid) - 1:
        edge = "lower" if index == 0 else "upper"
        raise ArithmeticError(
            f"oscillator beta optimum reached the {edge} beta_grid endpoint ({grid[index]} GeV); "
            "widen the declared beta bracket"
        )
    a, b = grid[index - 1], grid[index + 1]
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inverse_phi * (b - a)
    d = a + inverse_phi * (b - a)
    fc = _objective(_cached_evaluation(cache, evaluate, c))
    fd = _objective(_cached_evaluation(cache, evaluate, d))
    while b - a > solver.beta_tolerance_GeV:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - inverse_phi * (b - a)
            fc = _objective(_cached_evaluation(cache, evaluate, c))
        else:
            a, c, fc = c, d, fd
            d = a + inverse_phi * (b - a)
            fd = _objective(_cached_evaluation(cache, evaluate, d))
    return _best_cached(cache)


def _oscillator_solution_search(solver, L: int, nlevels: int, evaluate):
    from .waves import OscillatorWave

    if nlevels < 1:
        raise ValueError("oscillator solve requires at least one eigenlevel")
    if nlevels > solver.nbasis:
        raise ValueError(
            f"oscillator solve requested {nlevels} levels from an initial basis of {solver.nbasis}; "
            "raise nbasis to at least nlevels"
        )
    beta_seed = None
    previous_values = None
    previous_waves = None
    consecutive = 0
    refinements = 0
    nbasis = solver.nbasis
    while True:
        best_beta, best = _refine_oscillator_beta(lambda beta: evaluate(beta, nbasis), solver, seed=beta_seed)
        waves = [OscillatorWave(L, best_beta, best.vectors[:, n]).fix_outer_phase() for n in range(nlevels)]
        if not solver.converge:
            cert = OscillatorConvergence("unchecked", best_beta, nbasis, None, None, solver.energy_tolerance_GeV, 0)
            return best_beta, best, waves, cert
        if previous_values is not None:
            refinements += 1
            energy_delta = float(np.max(np.abs(best.values - previous_values)))
            overlap_defect = float(max(
                max(0.0, 1.0 - min(1.0, abs(previous_waves[n].radial_overlap(waves[n], lambda _: 1.0))))
                for n in range(nlevels)
            ))
            consecutive = consecutive + 1 if energy_delta <= solver.energy_tolerance_GeV else 0
            if consecutive >= HO_REQUIRED_CONVERGED_REFINEMENTS:
                cert = OscillatorConvergence(
                    "converged", best_beta, nbasis, energy_delta, overlap_defect,
                    solver.energy_tolerance_GeV, refinements,
                )
                return best_beta, best, waves, cert
        if nbasis == solver.max_nbasis:
            break
        previous_values = best.values.copy()
        previous_waves = waves
        beta_seed = best_beta
        nbasis = min(nbasis + solver.basis_step, solver.max_nbasis)
    delta = "not measured" if previous_values is None else f"{1000 * float(np.max(np.abs(best.values - previous_values)))} MeV"
    raise ArithmeticError(
        f"oscillator basis did not converge through max_nbasis={solver.max_nbasis}; final requested-level "
        f"change was {delta}, tolerance is {1000 * solver.energy_tolerance_GeV} MeV"
    )


def _lowest_eigenpairs(H: np.ndarray, nlevels: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigh(H)  # LAPACK syevr, as Julia eigen(Symmetric)
    return values[:nlevels].copy(), vectors[:, :nlevels].copy()


def oscillator_channel_solution(solver, params: GIParameters, masses: ConstituentMasses, L: int, nlevels: int):
    """Central spin-independent HO solve for orbital ``L``."""
    from .channel import ChannelRadialSolution

    def evaluate(beta, nbasis):
        H = oscillator_central_matrix(params, masses, L, beta, nbasis)
        return _Eval(*_lowest_eigenpairs(H, nlevels))

    _, best, waves, cert = _oscillator_solution_search(solver, int(L), int(nlevels), evaluate)
    return ChannelRadialSolution(best.values, waves, convergence=cert)


def oscillator_fixed_channel_matrices(
    params: GIParameters, masses: ConstituentMasses, multiplet, beta: float, terms, nbasis: int
) -> dict[str, np.ndarray]:
    """Native HO matrices of one fixed ``(L,S,J)`` sector (Julia ``_fixed_channel_matrices``)."""
    from .spin import _zero_fine_matrices

    L = multiplet.L
    central = oscillator_central_matrix(params, masses, L, beta, nbasis)
    zero = np.zeros((nbasis, nbasis))
    contact = (
        ho_contact_hyperfine_matrix(params, masses, L, multiplet.multiplicity, beta, nbasis)
        if terms.contact_hyperfine
        else zero
    )
    fine = (
        ho_fine_structure_matrices(params, masses, L, multiplet.multiplicity, multiplet.J, beta, nbasis)
        if terms.fine_structure
        else _zero_fine_matrices(nbasis)
    )
    return dict(
        central=central,
        contact=contact,
        spin_orbit_vector=fine.spin_orbit_vector,
        spin_orbit_thomas=fine.spin_orbit_thomas,
        spin_orbit=fine.spin_orbit,
        tensor=fine.tensor,
        fine_structure=fine.total,
        total=_sym_upper(central + contact + fine.total),
    )


def oscillator_fixed_channel_solution(
    solver, params: GIParameters, masses: ConstituentMasses, multiplet, terms, nlevels: int
):
    """Diagonalise central + contact + diagonal fine structure in one ``(L,S,J)`` sector."""
    from .channel import ChannelRadialSolution

    def evaluate(beta, nbasis):
        total = oscillator_fixed_channel_matrices(params, masses, multiplet, beta, terms, nbasis)["total"]
        return _Eval(*_lowest_eigenpairs(total, nlevels))

    _, best, waves, cert = _oscillator_solution_search(solver, multiplet.L, int(nlevels), evaluate)
    return ChannelRadialSolution(best.values, waves, convergence=cert)


def clear_ho_caches() -> None:
    """Drop the DVR rules and memoised HO operator matrices (costs time only)."""
    _DVR.clear()
    _DVR_COLUMNS.clear()
    _OPS.clear()
