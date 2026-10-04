"""Finite-difference mesh primitives (port of ``radial_grid.jl`` and the matrix
helpers of ``hamiltonian.jl`` / ``harmonic_oscillator_basis.jl``).

The mesh is ``r_i = i h`` (``i = 1..ngrid``), ``h = rmax/(ngrid+1)`` with
Dirichlet boundaries at ``0`` and ``rmax``. ``p^2`` is the three-point
tridiagonal operator with diagonal ``2/h^2 + L(L+1)/r^2`` and off-diagonal
``-1/h^2``. Every function of momentum ``f(p^2)`` is a matrix function through
the eigendecomposition of that L-dependent ``p^2`` (eigenvalues clamped at 0).

Eigendecompositions and dense relativisation matrices are cached, since many
operators of one spectrum share them.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Hashable

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal

__all__ = [
    "radial_grid",
    "p2_operator",
    "p2_eigen",
    "matrix_function",
    "projected_diagonal",
    "relativization_values",
    "relativization_matrix",
    "apply_relativization",
    "gi_spin_dependent_side_exponent",
    "lowest_eigenpairs",
    "clear_caches",
]


class _ByteLRU:
    """Tiny LRU cache bounded by the total size of the stored NumPy arrays."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._data: OrderedDict[Hashable, tuple] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _size(value: tuple) -> int:
        return sum(v.nbytes for v in value if isinstance(v, np.ndarray))

    def get(self, key: Hashable, factory: Callable[[], tuple]) -> tuple:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        value = factory()
        for v in value:
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
        size = self._size(value)
        if size <= self.max_bytes:
            self._data[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, old = self._data.popitem(last=False)
                self._bytes -= self._size(old)
        return value

    def clear(self) -> None:
        self._data.clear()
        self._bytes = 0


_P2_CACHE = _ByteLRU(256 * 1024**2)
_MATRIX_CACHE = _ByteLRU(256 * 1024**2)


def clear_caches() -> None:
    """Drop all cached ``p^2`` eigendecompositions and relativisation matrices."""
    _P2_CACHE.clear()
    _MATRIX_CACHE.clear()


def radial_grid(ngrid: int, rmax: float) -> tuple[np.ndarray, float]:
    """Interior Dirichlet mesh ``r_i = i h`` and spacing ``h = rmax/(ngrid+1)``."""
    h = rmax / (ngrid + 1)
    return h * np.arange(1, ngrid + 1, dtype=float), h


def p2_operator(L: int, r: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal and off-diagonal of the tridiagonal mesh ``p^2`` for orbital ``L``."""
    r = np.asarray(r, dtype=float)
    diagonal = 2 / h**2 + L * (L + 1) / r**2
    offdiag = np.full(len(r) - 1, -1 / h**2)
    return diagonal, offdiag


def _mesh_key(r: np.ndarray, h: float) -> tuple[int, float, float]:
    return (len(r), float(h), float(r[0]))


def p2_eigen(L: int, r: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached eigendecomposition ``(lambda, V)`` of the mesh ``p^2`` (ascending, read-only)."""
    r = np.asarray(r, dtype=float)
    return _p2_eigen_full(int(L), r, h)[:2]


def _p2_eigen_full(L: int, r: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(L), *_mesh_key(r, h))

    def factory():
        d, e = p2_operator(int(L), r, h)
        values, vectors = eigh_tridiagonal(d, e)
        return values, np.ascontiguousarray(vectors), np.ascontiguousarray(vectors.T)

    return _P2_CACHE.get(key, factory)


def projected_diagonal(L: int, r: np.ndarray, h: float, tag: Hashable, values: Callable[[], np.ndarray]) -> np.ndarray:
    """Cached ``V^T diag(k) V``: a local kernel ``k(r)`` in the eigenbasis of the mesh ``p^2``.

    ``tag`` must identify the kernel uniquely (it is part of the cache key);
    ``values`` is only called on a cache miss.
    """
    r = np.asarray(r, dtype=float)
    key = ("proj", tag, int(L), *_mesh_key(r, h))

    def factory():
        _, V, VT = _p2_eigen_full(L, r, h)
        k = np.asarray(values(), dtype=float)
        M = (VT * k) @ V
        return (0.5 * (M + M.T),)

    return _MATRIX_CACHE.get(key, factory)[0]


def matrix_function(fvalues: np.ndarray, vectors: np.ndarray, vectors_t: np.ndarray | None = None) -> np.ndarray:
    """Dense ``V diag(f) V^T``."""
    vt = np.ascontiguousarray(vectors.T) if vectors_t is None else vectors_t
    return (vectors * fvalues) @ vt


def gi_spin_dependent_side_exponent(epsilon: float) -> float:
    """Exponent ``1/2 + epsilon`` of the GI momentum factor placed on each side."""
    return 0.5 + float(epsilon)


def relativization_values(lam: np.ndarray, m1: float, m2: float, exponent: float) -> np.ndarray:
    """``(m1 m2 / (E1 E2))^exponent`` on clamped ``p^2`` eigenvalues."""
    lam = np.maximum(lam, 0.0)
    e1 = np.sqrt(lam + m1**2)
    e2 = np.sqrt(lam + m2**2)
    return (m1 * m2 / (e1 * e2)) ** exponent


def relativization_matrix(L: int, r: np.ndarray, h: float, m1: float, m2: float, exponent: float) -> np.ndarray:
    """Cached dense ``B = (m1 m2/(E1 E2))^exponent`` as a matrix function of the mesh ``p^2``."""
    key = ("B", int(L), *_mesh_key(np.asarray(r), h), float(m1), float(m2), float(exponent))

    def factory():
        lam, V, VT = _p2_eigen_full(L, r, h)
        return (matrix_function(relativization_values(lam, m1, m2, exponent), V, VT),)

    return _MATRIX_CACHE.get(key, factory)[0]


def apply_relativization(
    u: np.ndarray, L: int, r: np.ndarray, h: float, m1: float, m2: float, exponent: float
) -> np.ndarray:
    """``B u`` without forming ``B`` (``O(n^2)``)."""
    lam, V, VT = _p2_eigen_full(L, r, h)
    return V @ (relativization_values(lam, m1, m2, exponent) * (VT @ u))


def lowest_eigenpairs(
    hamiltonian: np.ndarray, nlevels: int, eigensolver: str = "full"
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``nlevels`` eigenpairs of a dense symmetric matrix (ascending)."""
    n = hamiltonian.shape[0]
    nlevels = int(nlevels)
    if eigensolver == "krylov":
        from scipy.sparse.linalg import eigsh

        values, vectors = eigsh(hamiltonian, k=nlevels, which="SA")
        order = np.argsort(values)
        return values[order], vectors[:, order]
    if nlevels >= n:
        values, vectors = eigh(hamiltonian)
    else:
        values, vectors = eigh(hamiltonian, subset_by_index=[0, nlevels - 1])
    return values[:nlevels], vectors[:, :nlevels]
