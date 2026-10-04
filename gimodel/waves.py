"""Radial waves and the representation-independent wave interface (port of the
``RadialWave``/``MeshWave`` part of ``model_objects.jl`` and the mesh part of
``momentum_waves.jl``).

Consumers ask a :class:`RadialWave` for operations and never touch its
representation. :class:`MeshWave` (finite-difference samples) and
:class:`OscillatorWave` (harmonic-oscillator expansion, port of the wave part of
``harmonic_oscillator_basis.jl``) implement the same abstract methods.

Conventions: ``u`` is the reduced radial wave with ``int u^2 dr = 1``; every
wave produced by a solve has its outermost lobe positive (the last sample with
``|u| > 0.2 max|u|`` is positive).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from . import fd

__all__ = [
    "RadialWave",
    "MeshWave",
    "MeshMomentumWave",
    "OscillatorWave",
    "OscillatorMomentumWave",
    "physically_normalized_waves",
    "wave_norm",
    "radial_expect",
    "radial_overlap",
    "radial_derivative_overlap",
    "fix_outer_phase",
    "momentum_wave",
    "momentum_expect",
    "momentum_overlap",
    "momentum_functional",
    "wave_mean_squares",
    "WaveMeanSquares",
    "evaluate_on_grid",
]

RadialFunction = Callable[[np.ndarray], "np.ndarray | float"]


def evaluate_on_grid(f: RadialFunction, r: np.ndarray) -> np.ndarray:
    """Evaluate ``f`` on ``r``: vectorised when possible, pointwise otherwise."""
    try:
        values = np.asarray(f(r), dtype=float)
    except (TypeError, ValueError):
        values = None
    if values is not None:
        if values.shape == r.shape:
            return values
        if values.ndim == 0:
            return np.full(r.shape, float(values))
    return np.array([float(f(float(x))) for x in r])


class RadialWave(ABC):
    """One radial eigenlevel, however it was computed.

    Required operations (all with physically normalised waves):

    * :meth:`wave_norm` -- ``int u^2 dr`` (1 for solver output);
    * :meth:`radial_expect` -- ``int u^2 f(r) dr``;
    * :meth:`radial_overlap` -- ``int u_x u_y f(r) dr``;
    * :meth:`fix_outer_phase` -- same state with outermost lobe positive;
    * :meth:`sandwich_expectation` -- ``<u| B K B |u>/<u|u>`` with
      ``B = (m1 m2/(E1 E2))^exponent`` a function of the ``L``-dependent ``p^2``;
    * :meth:`sandwich_cross` -- ``<u_l| B_l K B_r |u_r>/(|u_l||u_r|)`` with ``B_l``
      built for ``L_left`` and ``B_r`` for ``L_right``.

    :meth:`contact_sandwich_expectation` defaults to :meth:`sandwich_expectation`;
    a representation whose reference implementation evaluates the contact
    operator differently (GIModel.jl's oscillator path) may override it.
    """

    @abstractmethod
    def wave_norm(self) -> float: ...

    @abstractmethod
    def radial_expect(self, f: RadialFunction) -> float: ...

    @abstractmethod
    def radial_overlap(self, other: RadialWave, f: RadialFunction) -> float: ...

    @abstractmethod
    def fix_outer_phase(self) -> RadialWave: ...

    @abstractmethod
    def sandwich_expectation(
        self, m1: float, m2: float, L: int, exponent: float, kernel: RadialFunction
    ) -> float: ...

    @abstractmethod
    def sandwich_cross(
        self,
        other: RadialWave,
        m1: float,
        m2: float,
        L_left: int,
        L_right: int,
        exponent: float,
        kernel: RadialFunction,
    ) -> float: ...

    def contact_sandwich_expectation(
        self, m1: float, m2: float, L: int, exponent: float, kernel: RadialFunction
    ) -> float:
        return self.sandwich_expectation(m1, m2, L, exponent, kernel)

    def radial_derivative_overlap(self, other: RadialWave, f: RadialFunction) -> float:
        raise NotImplementedError(f"{type(self).__name__} has no radial_derivative_overlap")

    def momentum_wave(self, L: int, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no momentum_wave")


@dataclass(frozen=True, eq=False)
class MeshWave(RadialWave):
    """Reduced radial wave ``u`` sampled on a uniform interior mesh ``r`` with spacing ``h``."""

    u: np.ndarray
    r: np.ndarray
    h: float

    def __init__(self, u, r, h: float | None = None) -> None:
        u_arr = np.array(u, dtype=float)
        r_arr = np.array(r, dtype=float)
        if u_arr.shape != r_arr.shape or u_arr.ndim != 1:
            raise ValueError("MeshWave: length(u) != length(r)")
        if len(r_arr) < 1:
            raise ValueError("MeshWave: empty r")
        if h is None:
            if len(r_arr) < 2:
                raise ValueError("MeshWave(u, r): need at least two points to infer h")
            h = r_arr[1] - r_arr[0]
        hf = float(h)
        if not (np.isfinite(hf) and hf > 0):
            raise ValueError(f"MeshWave: invalid mesh spacing h={h}")
        if len(r_arr) >= 2:
            hinfer = r_arr[1] - r_arr[0]
            if not np.isclose(hinfer, hf, rtol=1e-10, atol=1e-12):
                raise ValueError(f"MeshWave: h={hf} inconsistent with r spacing {hinfer}")
        u_arr.flags.writeable = False
        r_arr.flags.writeable = False
        object.__setattr__(self, "u", u_arr)
        object.__setattr__(self, "r", r_arr)
        object.__setattr__(self, "h", hf)

    def __repr__(self) -> str:
        return f"MeshWave({len(self.u)} points, h = {self.h:.4g} GeV⁻¹, r ≤ {self.r[-1]:.4g} GeV⁻¹)"

    # --- guards ---------------------------------------------------------------
    def _require_uniform(self) -> None:
        if len(self.r) >= 2:
            d = np.diff(self.r)
            if not (np.all(d > 0) and np.allclose(d, self.h, rtol=1e-10, atol=1e-12)):
                raise ValueError("MeshWave: non-uniform r mesh or h mismatch")

    def _require_shared_mesh(self, other: MeshWave, what: str) -> None:
        if not isinstance(other, MeshWave):
            raise TypeError(f"{what}: both waves must be MeshWave")
        if len(self.r) != len(other.r):
            raise ValueError(f"{what}: waves live on different meshes")
        if not np.isclose(self.h, other.h, rtol=1e-10):
            raise ValueError(f"{what}: mesh spacings differ")
        if not np.allclose(self.r, other.r, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{what}: mesh points differ")

    # --- interface --------------------------------------------------------------
    def wave_norm(self) -> float:
        return float(np.sum(self.u**2) * self.h)

    def radial_expect(self, f: RadialFunction) -> float:
        nrm = self.wave_norm()
        if not nrm > 0:
            raise ValueError("radial_expect: zero-norm wave")
        return float(np.sum(self.u**2 * evaluate_on_grid(f, self.r)) * self.h / nrm)

    def radial_overlap(self, other: RadialWave, f: RadialFunction) -> float:
        self._require_shared_mesh(other, "radial_overlap")
        nx, ny = self.wave_norm(), other.wave_norm()
        if not (nx > 0 and ny > 0):
            raise ValueError("radial_overlap: zero-norm wave")
        s = np.sum(self.u * other.u * evaluate_on_grid(f, self.r))
        return float(s * self.h / np.sqrt(nx * ny))

    def radial_derivative_overlap(self, other: RadialWave, f: RadialFunction) -> float:
        """``int (du_x/dr) u_y f dr`` with Dirichlet endpoints and centred differences."""
        if len(self.r) < 2:
            raise ValueError("radial_derivative_overlap: need at least two mesh points")
        self._require_shared_mesh(other, "radial_derivative_overlap")
        nx, ny = self.wave_norm(), other.wave_norm()
        if not (nx > 0 and ny > 0):
            raise ValueError("radial_derivative_overlap: zero-norm wave")
        padded = np.concatenate(([0.0], self.u, [0.0]))
        du = (padded[2:] - padded[:-2]) / (2 * self.h)
        s = np.sum(du * other.u * evaluate_on_grid(f, self.r))
        return float(s * self.h / np.sqrt(nx * ny))

    def fix_outer_phase(self) -> MeshWave:
        peak = np.max(np.abs(self.u))
        idx = np.nonzero(np.abs(self.u) > 0.2 * peak)[0]
        if len(idx) == 0 or self.u[idx[-1]] >= 0:
            return self
        return MeshWave(-self.u, self.r, self.h)

    def sandwich_expectation(self, m1, m2, L, exponent, kernel) -> float:
        if len(self.r) < 2:
            return 0.0
        self._require_uniform()
        norm2 = float(self.u @ self.u)
        if norm2 <= 0.0:
            return 0.0
        bu = fd.apply_relativization(self.u, L, self.r, self.h, m1, m2, exponent)
        return float(bu @ (evaluate_on_grid(kernel, self.r) * bu)) / norm2

    def sandwich_cross(self, other, m1, m2, L_left, L_right, exponent, kernel) -> float:
        self._require_shared_mesh(other, "radial cross expectation")
        self._require_uniform()
        nl = np.sqrt(self.u @ self.u)
        nr = np.sqrt(other.u @ other.u)
        if nl == 0.0 or nr == 0.0:
            return 0.0
        bl = fd.apply_relativization(self.u, L_left, self.r, self.h, m1, m2, exponent)
        br = fd.apply_relativization(other.u, L_right, self.r, self.h, m1, m2, exponent)
        return float(bl @ (evaluate_on_grid(kernel, self.r) * br)) / (nl * nr)

    def momentum_wave(self, L: int, *, pmax: float = 30.0, npoints: int = 1501) -> MeshMomentumWave:
        return mock_momentum_wave(self, L, pmax=pmax, npoints=npoints)


# --- oscillator-basis waves ------------------------------------------------------


def _quadgk(*args, **kwargs):
    from .ho import quadgk

    return quadgk(*args, **kwargs)


@dataclass(frozen=True, eq=False)
class OscillatorWave(RadialWave):
    """Normalised radial wave ``sum_n c_n u_{n,L}(r; beta)`` in HO basis functions.

    ``coefficients[n]`` multiplies :func:`gimodel.ho.ho_reduced_radial` ``(n, L, beta, r)``;
    the constructor normalises them. No mesh is stored.
    """

    L: int
    beta: float
    coefficients: np.ndarray

    def __init__(self, L: int, beta: float, coefficients) -> None:
        if L < 0:
            raise ValueError("OscillatorWave: L must be nonnegative")
        if not beta > 0:
            raise ValueError("OscillatorWave: beta must be positive")
        c = np.array(coefficients, dtype=float).ravel()
        if len(c) == 0:
            raise ValueError("OscillatorWave: empty coefficients")
        nrm = float(np.linalg.norm(c))
        if not nrm > 0:
            raise ValueError("OscillatorWave: zero-norm coefficients")
        c = c / nrm
        c.flags.writeable = False
        object.__setattr__(self, "L", int(L))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "coefficients", c)

    def __repr__(self) -> str:
        return f"OscillatorWave(L = {self.L}, β = {self.beta:.4f} GeV, {len(self.coefficients)} basis functions)"

    # --- evaluation ---------------------------------------------------------------
    def __call__(self, r):
        """``u(r)`` from the expansion (scalar or array ``r``)."""
        from .ho import _ho_basis_values

        x = np.asarray(r, dtype=float)
        v = _ho_basis_values(self.L, self.beta, len(self.coefficients), np.atleast_1d(x)) @ self.coefficients
        return v.reshape(x.shape) if x.ndim else float(v[0])

    def _derivative(self, r: np.ndarray) -> np.ndarray:
        from .ho import _ho_basis_derivatives

        return _ho_basis_derivatives(self.L, self.beta, len(self.coefficients), r) @ self.coefficients

    def _coordinate_cutoff(self) -> float:
        from .ho import oscillator_tail_rho

        return oscillator_tail_rho(self.L, len(self.coefficients)) / self.beta

    def _momentum_cutoff(self) -> float:
        from .ho import oscillator_tail_rho

        return oscillator_tail_rho(self.L, len(self.coefficients)) * self.beta

    def sample(self, r) -> MeshWave:
        """Plotting samples on a uniform grid (Julia ``sample_wave``), normalised on that grid."""
        r = np.asarray(r, dtype=float)
        samples = self(r)
        h = r[1] - r[0]
        samples = samples / np.sqrt(np.sum(samples**2) * h)
        return MeshWave(samples, r, h)

    # --- interface ----------------------------------------------------------------
    def wave_norm(self) -> float:
        return float(np.sum(self.coefficients**2))

    def radial_expect(self, f: RadialFunction) -> float:
        from .ho import ho_operator_matrix

        c = self.coefficients
        op = ho_operator_matrix(self.L, self.beta, len(c), f)
        return float(c @ (op @ c)) / self.wave_norm()

    def _require_oscillator(self, other, what: str) -> None:
        if not isinstance(other, OscillatorWave):
            raise TypeError(f"{what}: both waves must be OscillatorWave")

    def radial_overlap(self, other: RadialWave, f: RadialFunction) -> float:
        from .ho import ho_operator_matrix

        self._require_oscillator(other, "radial_overlap")
        nl, nr = np.sqrt(self.wave_norm()), np.sqrt(other.wave_norm())
        if self.L == other.L and self.beta == other.beta:
            n = max(len(self.coefficients), len(other.coefficients))
            cl = np.zeros(n)
            cl[: len(self.coefficients)] = self.coefficients
            cr = np.zeros(n)
            cr[: len(other.coefficients)] = other.coefficients
            op = ho_operator_matrix(self.L, self.beta, n, f)
            return float(cl @ (op @ cr)) / (nl * nr)
        rmax = max(self._coordinate_cutoff(), other._coordinate_cutoff())
        value, _ = _quadgk(lambda r: self(r) * evaluate_on_grid(f, r) * other(r), 0.0, rmax, rtol=1e-10)
        return value / (nl * nr)

    def radial_derivative_overlap(self, other: RadialWave, f: RadialFunction) -> float:
        self._require_oscillator(other, "radial_derivative_overlap")
        nl, nr = np.sqrt(self.wave_norm()), np.sqrt(other.wave_norm())
        rmax = max(self._coordinate_cutoff(), other._coordinate_cutoff())
        value, _ = _quadgk(lambda r: self._derivative(r) * evaluate_on_grid(f, r) * other(r), 0.0, rmax, rtol=1e-10)
        return value / (nl * nr)

    def fix_outer_phase(self) -> OscillatorWave:
        # The last significant antinode of the analytic expansion on a dimensionless
        # interval fixes the phase (not the sign of the tiny highest coefficient).
        rho_max = np.sqrt(4 * (len(self.coefficients) - 1) + 2 * self.L + 3) + 6
        values = self(np.linspace(0.0, rho_max, 2049) / self.beta)
        peak = np.max(np.abs(values))
        idx = np.nonzero(np.abs(values) > 0.2 * peak)[0]
        if len(idx) == 0 or values[idx[-1]] >= 0:
            return self
        return OscillatorWave(self.L, self.beta, -self.coefficients)

    def _check_L(self, L: int, what: str) -> None:
        if int(L) != self.L:
            raise ValueError(f"{what}: L={L} does not match wave L={self.L}")

    def sandwich_expectation(self, m1, m2, L, exponent, kernel) -> float:
        from .ho import ho_cross_sandwich

        self._check_L(L, "radial_expect_momentum_sandwich")
        return ho_cross_sandwich(self, self, m1, m2, exponent, kernel)

    def sandwich_cross(self, other, m1, m2, L_left, L_right, exponent, kernel) -> float:
        from .ho import ho_cross_sandwich

        self._require_oscillator(other, "radial cross expectation")
        if int(L_left) != self.L or int(L_right) != other.L:
            raise ValueError("radial_cross_expect_momentum_sandwich: orbital labels do not match waves")
        return ho_cross_sandwich(self, other, m1, m2, exponent, kernel)

    def contact_sandwich_expectation(self, m1, m2, L, exponent, kernel) -> float:
        """GIModel.jl evaluates the HO contact through its basis matrix
        (``ho_momentum_sandwich_matrix`` with the kernel at ``rtol = 1e-8``)."""
        from .ho import _cached_sandwich

        self._check_L(L, "contact expectation")
        kernel_key = None
        owner = getattr(kernel, "__self__", None)
        if owner is not None and type(owner).__name__ == "ContactHyperfine" and getattr(kernel, "__func__", None) is type(owner).kernel:
            kernel_key = ("contact", owner.params, float(owner.masses.m1_GeV), float(owner.masses.m2_GeV))
        c = self.coefficients
        M = _cached_sandwich(self.L, self.beta, len(c), float(m1), float(m2), float(exponent), kernel, 1e-8, kernel_key)
        norm2 = float(c @ c)
        return 0.0 if norm2 <= 0 else float(c @ (M @ c)) / norm2

    def momentum_wave(self, L: int | None = None, **kwargs) -> OscillatorMomentumWave:
        if L is not None and int(L) != self.L:
            raise ValueError(f"momentum_wave: requested L={L} for OscillatorWave with L={self.L}")
        return OscillatorMomentumWave(self)


@dataclass(frozen=True, eq=False)
class OscillatorMomentumWave:
    """Exact momentum-space view ``Phi_L(p)`` of an :class:`OscillatorWave`
    (``beta -> 1/beta`` and ``(-1)^n``); ``int p^2 Phi^2 dp = 1``. Call it with ``p``."""

    source: OscillatorWave

    def _values(self, p: np.ndarray) -> np.ndarray:
        from .ho import _ho_basis_values

        w = self.source
        p = np.asarray(p, dtype=float)
        pf = np.where(p == 0, np.sqrt(np.finfo(float).eps) * w.beta, p)
        n = len(w.coefficients)
        signs = np.where(np.arange(n) % 2 == 1, -1.0, 1.0)
        vals = (_ho_basis_values(w.L, 1.0 / w.beta, n, pf) @ (signs * w.coefficients)) / pf
        if w.L > 0:
            vals = np.where(p == 0, 0.0, vals)
        return vals

    def __call__(self, p):
        x = np.asarray(p, dtype=float)
        if not np.all(np.isfinite(x) & (x >= 0)):
            raise ValueError("momentum must be finite and nonnegative")
        v = self._values(np.atleast_1d(x))
        return v.reshape(x.shape) if x.ndim else float(v[0])


def _p2_function_matrix(L: int, beta: float, n: int, g: Callable) -> np.ndarray:
    """``V diag(g(sqrt(lambda))) V^T`` of the truncated exact ``p^2`` matrix (as GIModel.jl)."""
    from scipy.linalg import eigh

    from .ho import ho_p2_matrix

    lam, V = eigh(ho_p2_matrix(L, beta, n))
    values = evaluate_on_grid(g, np.sqrt(np.maximum(lam, 0.0)))
    return (V * values) @ V.T


def physically_normalized_waves(waves: np.ndarray, h: float) -> np.ndarray:
    """Scale each column to ``sum u_i^2 h = 1`` (zero columns are left alone)."""
    out = np.array(waves, dtype=float)
    if not h > 0:
        return out
    norms = np.sqrt(np.sum(out**2, axis=0) * h)
    nonzero = norms > 0
    out[:, nonzero] /= norms[nonzero]
    return out


# --- free-function interface ---------------------------------------------------


def wave_norm(w: RadialWave) -> float:
    """The physical norm ``int u^2 dr``."""
    return w.wave_norm()


def radial_expect(obj, *args):
    """``radial_expect(wave, f)`` -> ``int u^2 f dr``; ``radial_expect(spec, state, f)`` ->
    coherent expectation in a (possibly mixed) physical spectrum state."""
    if isinstance(obj, RadialWave):
        (f,) = args
        return obj.radial_expect(f)
    from .spectrum import spectrum_radial_expect

    return spectrum_radial_expect(obj, *args)


def radial_overlap(wx: RadialWave, wy: RadialWave, f: RadialFunction) -> float:
    """``int u_x u_y f dr``, each wave normalised."""
    return wx.radial_overlap(wy, f)


def radial_derivative_overlap(wx: RadialWave, wy: RadialWave, f: RadialFunction) -> float:
    """``int (du_x/dr) u_y f dr``, each wave normalised."""
    return wx.radial_derivative_overlap(wy, f)


def fix_outer_phase(w: RadialWave) -> RadialWave:
    return w.fix_outer_phase()


# --- momentum space (mesh only) ------------------------------------------------


@dataclass(frozen=True, eq=False)
class MeshMomentumWave:
    """A normalised momentum-space radial wave ``Phi_L(p)`` sampled on a p-grid."""

    p: np.ndarray
    phi: np.ndarray


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _spherical_bessel_j(L: int, x: np.ndarray) -> np.ndarray:
    """Vectorised port of the Julia helper (same small-x branches and upward recurrence)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        j0 = np.where(np.abs(x) < 1.0e-8, 1.0 - x**2 / 6, np.sin(x) / x)
        if L == 0:
            return j0
        small = np.abs(x) < 1.0e-4
        jm1 = j0
        j = np.sin(x) / x**2 - np.cos(x) / x
        for l in range(1, L):
            jp1 = (2 * l + 1) / x * j - jm1
            jm1, j = j, jp1
        dfact = float(np.prod(np.arange(1, 2 * L + 2, 2)))
        return np.where(small, x**L / dfact, j)


def mock_momentum_wave(radial: MeshWave, L: int, *, pmax: float = 30.0, npoints: int = 1501) -> MeshMomentumWave:
    """Spherical-Bessel transform ``Phi_L(p) = sqrt(2/pi) int r u(r) j_L(p r) dr``, normalised."""
    p = np.linspace(0.0, float(pmax), int(npoints))
    jl = _spherical_bessel_j(int(L), np.outer(p, radial.r))
    phi = np.sqrt(2 / np.pi) * (jl @ (radial.r * radial.u)) * radial.h
    nrm = np.sqrt(_trapz(p, p**2 * phi**2))
    if not nrm > 0:
        raise ValueError("mock_momentum_wave: zero-norm wave")
    return MeshMomentumWave(p, phi / nrm)


def momentum_wave(w: RadialWave, L: int, **kwargs) -> MeshMomentumWave:
    """Momentum-space radial wave ``Phi(p)``, normalised so ``int p^2 Phi^2 dp = 1``."""
    return w.momentum_wave(L, **kwargs)


def momentum_expect(mw, g: Callable) -> float:
    """``int p^2 Phi(p)^2 g(p) dp``."""
    if isinstance(mw, OscillatorMomentumWave):
        w = mw.source
        c = w.coefficients
        return float(c @ (_p2_function_matrix(w.L, w.beta, len(c), g) @ c)) / w.wave_norm()
    return _trapz(mw.p, mw.p**2 * mw.phi**2 * evaluate_on_grid(g, mw.p))


def momentum_overlap(left: MeshMomentumWave, right: MeshMomentumWave, g: Callable) -> float:
    """``int p^2 Phi_x Phi_y g dp`` with both waves normalised."""
    if isinstance(left, OscillatorMomentumWave) or isinstance(right, OscillatorMomentumWave):
        return _oscillator_momentum_overlap(left, right, g)
    if len(left.p) != len(right.p) or not np.allclose(left.p, right.p, rtol=1e-12, atol=1e-14):
        raise ValueError("momentum_overlap: momentum grids differ")
    nl = momentum_expect(left, lambda _: 1.0)
    nr = momentum_expect(right, lambda _: 1.0)
    if not (nl > 0 and nr > 0):
        raise ValueError("momentum_overlap: zero-norm wave")
    value = _trapz(left.p, left.p**2 * left.phi * right.phi * evaluate_on_grid(g, left.p))
    return value / np.sqrt(nl * nr)


class WaveMeanSquares(NamedTuple):
    r2: float
    p2: float


def wave_mean_squares(wave: RadialWave, L: int, **momentum_kwargs) -> WaveMeanSquares:
    """``(<r^2> in GeV^-2, <p^2> in GeV^2)`` for one radial wave of orbital ``L``."""
    if L < 0:
        raise ValueError("wave_mean_squares: L must be non-negative")
    r2 = wave.radial_expect(lambda r: r**2)
    p2 = momentum_expect(wave.momentum_wave(L, **momentum_kwargs), lambda p: p**2)
    return WaveMeanSquares(r2, p2)


def _oscillator_momentum_overlap(left, right, g: Callable) -> float:
    if not (isinstance(left, OscillatorMomentumWave) and isinstance(right, OscillatorMomentumWave)):
        raise TypeError("momentum_overlap: both momentum waves must be OscillatorMomentumWave")
    wl, wr = left.source, right.source
    norm = np.sqrt(wl.wave_norm() * wr.wave_norm())
    if wl.L == wr.L and wl.beta == wr.beta:
        n = max(len(wl.coefficients), len(wr.coefficients))
        cl = np.zeros(n)
        cl[: len(wl.coefficients)] = wl.coefficients
        cr = np.zeros(n)
        cr[: len(wr.coefficients)] = wr.coefficients
        return float(cl @ (_p2_function_matrix(wl.L, wl.beta, n, g) @ cr)) / norm
    pmax = max(wl._momentum_cutoff(), wr._momentum_cutoff())
    value, _ = _quadgk(lambda p: p**2 * left._values(p) * right._values(p) * evaluate_on_grid(g, p), 0.0, pmax, rtol=1e-9)
    return value / norm


def momentum_functional(mw, K: Callable) -> float:
    """``int p^2 Phi(p) K(p) dp`` -- linear in ``Phi``."""
    if isinstance(mw, OscillatorMomentumWave):
        pmax = mw.source._momentum_cutoff()
        value, _ = _quadgk(lambda p: p**2 * mw._values(p) * evaluate_on_grid(K, p), 0.0, pmax, rtol=1e-9)
        return value
    return _trapz(mw.p, mw.p**2 * mw.phi * evaluate_on_grid(K, mw.p))
