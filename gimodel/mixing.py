"""Basis-state labels and generic mixing-block diagonalisation (port of
``state_mixing.jl`` and ``same_j_mixing`` from ``spin_fine_structure.jl``).

Phase convention: each eigenvector column has a positive component on its
*assigned* basis state, where ascending eigenvalues are assigned to basis states
ordered by ascending unmixed (diagonal) mass; a zero overlap falls back to the
largest component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
from scipy.linalg import eigh

from .constants import L_SYMBOLS
from .quarks import canonical_flavor

__all__ = [
    "BasisState",
    "MixingBlock",
    "MixingResult",
    "diagonalize_mixing_block",
    "same_j_mixing",
    "ANTISYMMETRIC_SPIN_ORBIT",
    "TENSOR_MIXING",
]

ANTISYMMETRIC_SPIN_ORBIT = "antisymmetric_spin_orbit"
TENSOR_MIXING = "tensor_mixing"


@dataclass(frozen=True)
class BasisState:
    """Quantum labels ``n ^(2S+1) L_J`` of a requested level or mixing component.

    ``label`` defaults to ``"{n}^{mult}{L}_{J}"`` (e.g. ``"1^3S_1"``); ``flavors`` is
    ``None`` for reusable level selections and ``(flavor1, flavor2)`` on solved states.
    """

    n: int
    L_label: str
    multiplicity: int
    J: int
    label: str = ""
    flavors: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        n, mult, J, L_label = int(self.n), int(self.multiplicity), int(self.J), str(self.L_label)
        if n < 1:
            raise ValueError("BasisState: n must be positive")
        if mult not in (1, 3):
            raise ValueError("BasisState: multiplicity must be 1 or 3")
        if J < 0:
            raise ValueError("BasisState: J must be non-negative")
        if L_label not in L_SYMBOLS:
            raise ValueError(f"BasisState: unknown orbital label `{L_label}`; expected one of S, P, D, F, G")
        L, S = L_SYMBOLS[L_label], (mult - 1) // 2
        if not abs(L - S) <= J <= L + S:
            raise ValueError(
                f"BasisState: J = {J} cannot be formed from L = {L} and S = {S} "
                f"(allowed: {abs(L - S)}:{L + S})"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "multiplicity", mult)
        object.__setattr__(self, "J", J)
        object.__setattr__(self, "L_label", L_label)
        if not self.label:
            object.__setattr__(self, "label", f"{n}^{mult}{L_label}_{J}")
        if self.flavors is not None:
            if len(self.flavors) != 2:
                raise ValueError("BasisState: `flavors` must contain exactly two symbols")
            object.__setattr__(
                self, "flavors", (canonical_flavor(self.flavors[0]), canonical_flavor(self.flavors[1]))
            )

    @property
    def L(self) -> int:
        return L_SYMBOLS[self.L_label]

    def identity(self) -> tuple:
        return (self.n, self.L_label, self.multiplicity, self.J, self.flavors)


@dataclass(frozen=True, eq=False)
class MixingBlock:
    """Symmetric mass matrix (GeV) over pure :class:`BasisState` entries."""

    name: str
    basis: tuple[BasisState, ...]
    matrix: np.ndarray
    mechanism: str = ""
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        basis = tuple(self.basis)
        n = len(basis)
        if n == 0:
            raise ValueError(f"MixingBlock {self.name}: basis must not be empty")
        if len({s.identity() for s in basis}) != n:
            raise ValueError(f"MixingBlock `{self.name}`: basis contains duplicate spectroscopic/flavor identities")
        mat = np.array(self.matrix, dtype=float)
        if mat.shape != (n, n):
            raise ValueError(
                f"MixingBlock `{self.name}`: matrix size {mat.shape} does not match basis length {n}"
            )
        if not np.allclose(mat, mat.T, rtol=1e-12, atol=1e-12):
            raise ValueError(f"MixingBlock `{self.name}`: matrix must be symmetric/Hermitian")
        mat.flags.writeable = False
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "matrix", mat)


@dataclass(frozen=True, eq=False)
class MixingResult:
    """Eigen-solution of a :class:`MixingBlock`: columns of ``vectors`` ordered by ascending ``masses``."""

    block: MixingBlock
    masses: np.ndarray
    vectors: np.ndarray
    pole_matrices: tuple[np.ndarray, ...] = field(default=())

    def __post_init__(self) -> None:
        n = len(self.block.basis)
        masses = np.array(self.masses, dtype=float)
        vectors = np.array(self.vectors, dtype=float)
        if masses.shape != (n,):
            raise ValueError("MixingResult: mass count does not match block basis")
        if not np.all(np.isfinite(masses)):
            raise ValueError("MixingResult: masses must be finite")
        if not np.all(np.diff(masses) >= 0):
            raise ValueError("MixingResult: masses must be ordered ascending with the vector columns")
        if vectors.shape != (n, n):
            raise ValueError("MixingResult: vector matrix size does not match block basis")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("MixingResult: vectors must be finite")
        if not np.allclose(np.sum(vectors**2, axis=0), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("MixingResult: every vector column must have unit norm")
        if self.pole_matrices and len(self.pole_matrices) != n:
            raise ValueError("MixingResult: pole_matrices must be empty or contain one matrix per pole")
        masses.flags.writeable = False
        vectors.flags.writeable = False
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "pole_matrices", tuple(np.asarray(m, dtype=float) for m in self.pole_matrices))


def _phase_fix_state_columns(vectors: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    anchors = np.argsort(diagonal, kind="stable")
    if len(anchors) != vectors.shape[1]:
        raise ValueError("phase basis size mismatch")
    for col, anchor in enumerate(anchors):
        index = int(np.argmax(np.abs(vectors[:, col]))) if vectors[anchor, col] == 0 else anchor
        if vectors[index, col] < 0:
            vectors[:, col] *= -1
    return vectors


def diagonalize_mixing_block(block: MixingBlock, *, phase_anchor: str | int = "assigned") -> MixingResult:
    """Diagonalise a block; ``phase_anchor="assigned"`` (default) or a 0-based common anchor row."""
    n = len(block.basis)
    if not (phase_anchor == "assigned" or (isinstance(phase_anchor, int) and 0 <= phase_anchor < n)):
        raise ValueError(f"phase_anchor={phase_anchor} outside 0:{n - 1}")
    values, vectors = eigh(np.array(block.matrix))
    vectors = np.array(vectors)
    if phase_anchor == "assigned":
        _phase_fix_state_columns(vectors, np.diag(block.matrix))
    else:
        for col in range(n):
            anchor = int(np.argmax(np.abs(vectors[:, col]))) if vectors[phase_anchor, col] == 0 else phase_anchor
            if vectors[anchor, col] < 0:
                vectors[:, col] *= -1
    return MixingResult(block, values, vectors)


class SameJMixing(NamedTuple):
    result: MixingResult
    block: MixingBlock
    matrix: np.ndarray
    masses: np.ndarray
    vectors: np.ndarray
    theta_rad: float
    theta_deg: float


def same_j_mixing(
    singlet_mass: float,
    triplet_mass: float,
    offdiag: float,
    *,
    basis: Sequence[BasisState] | None = None,
) -> SameJMixing:
    """Diagonalise the ``(^1L_L, ^3L_L)`` block; ``low = cos(theta) singlet + sin(theta) triplet``."""
    if basis is None:
        basis = (BasisState(1, "P", 1, 1, label="^1L_L"), BasisState(1, "P", 3, 1, label="^3L_L"))
    if len(basis) != 2:
        raise ValueError("same_j_mixing requires two basis states")
    block = MixingBlock(
        "same-J ^1L_L/^3L_L",
        tuple(basis),
        np.array([[float(singlet_mass), float(offdiag)], [float(offdiag), float(triplet_mass)]]),
        mechanism=ANTISYMMETRIC_SPIN_ORBIT,
    )
    result = diagonalize_mixing_block(block)
    low = np.array(result.vectors[:, 0])
    if low[0] < 0:
        low = -low
    theta = math.atan2(low[1], low[0])
    return SameJMixing(result, block, block.matrix, result.masses, result.vectors, theta, theta * 180 / math.pi)
