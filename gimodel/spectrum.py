"""Meson spectrum assembly (port of ``spectrum.jl``, without flavor annihilation).

Production path (:func:`compute_spectrum`):

1. :func:`fixed_spectrum` -- diagonalise the complete Hamiltonian in every
   requested fixed ``(L,S,J)`` sector; report the central, contact and
   fine-structure contributions as expectation values in the solved
   eigenstate (they sum to its eigenvalue).
2. :func:`add_intra_meson_mixing` -- same-J antisymmetric spin-orbit mixing
   (unequal flavours only) and triplet tensor ``L = J -/+ 1`` mixing on the
   stage-1 masses. Ascending eigenvalues go to members ordered by unmixed mass.

:func:`central_spectrum` is an independent spin-independent diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, NamedTuple, Sequence, TypeVar

import numpy as np

from .channel import (
    ChannelRadialSolution,
    RadialChannelKey,
    SectorComputation,
    channel_solution,
    fixed_channel_solution,
)
from .constants import L_SYMBOLS
from .mixing import (
    ANTISYMMETRIC_SPIN_ORBIT,
    TENSOR_MIXING,
    BasisState,
    MixingBlock,
    MixingResult,
    diagonalize_mixing_block,
)
from .parameters import GIParameters
from .quarks import ConstituentMasses, Meson, flavor_label, is_equal_flavor
from .solvers import FiniteDifferenceSolver, RadialSolver, SpinTerms
from .spin import (
    FineStructureMultiplet,
    contact_hyperfine_shift_active,
    diagonal_fine_structure_active,
    fine_structure_components,
    spin_orbit_mixing_components,
    tensor_mixing_components,
)
from .waves import RadialWave

__all__ = [
    "spectrum_levels",
    "StateMixing",
    "CentralState",
    "CorrectedState",
    "MixedState",
    "Spectrum",
    "PhysicalComponent",
    "central_spectrum",
    "fixed_spectrum",
    "add_intra_meson_mixing",
    "compute_spectrum",
    "spectrum_state",
    "parameters",
    "radial_wave",
    "physical_components",
    "physical_state_amplitude",
    "physical_transition_amplitude",
    "spectrum_radial_expect",
]


def spectrum_levels(nmax: int, L_labels: Sequence[str] = ("S", "P", "D")) -> list[BasisState]:
    """All ``n ^(2S+1) L_J`` multiplets for ``n = 1..nmax``: singlet ``J = L`` then triplets."""
    if nmax < 1:
        raise ValueError(f"spectrum_levels: nmax must be ≥ 1, got {nmax}")
    levels: list[BasisState] = []
    for L_label in L_labels:
        if str(L_label) not in L_SYMBOLS:
            raise ValueError(f"spectrum_levels: unknown orbital label `{L_label}`")
        L = L_SYMBOLS[str(L_label)]
        for n in range(1, nmax + 1):
            levels.append(BasisState(n, L_label, 1, L))
            for J in ([1] if L == 0 else range(L - 1, L + 2)):
                levels.append(BasisState(n, L_label, 3, J))
    return levels


# --- state types ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StateMixing:
    """One mixing step of a state: the shared block solution, this state's eigenvector
    column ``eigenstate`` (0-based) and its pre-mixing mass."""

    result: MixingResult
    eigenstate: int
    unmixed_GeV: float

    def __post_init__(self) -> None:
        if not 0 <= self.eigenstate < len(self.result.masses):
            raise ValueError(f"StateMixing: eigenstate column {self.eigenstate} is outside the mixing result")
        if not np.isfinite(self.unmixed_GeV):
            raise ValueError("StateMixing: unmixed mass must be finite")

    @property
    def mechanism(self) -> str:
        return self.result.block.mechanism

    @property
    def block_label(self) -> str:
        return self.result.block.name

    @property
    def partner_labels(self) -> list[str]:
        return [s.label for s in self.result.block.basis]

    @property
    def components(self) -> np.ndarray:
        return self.result.vectors[:, self.eigenstate]

    @property
    def partner_masses_GeV(self) -> np.ndarray:
        return self.result.masses

    @property
    def offdiag_GeV(self) -> float:
        m = self.result.block.matrix
        n = m.shape[0]
        if n <= 1:
            return 0.0
        return float(np.max(np.abs(m[np.triu_indices(n, 1)])))

    def __repr__(self) -> str:
        parts = ", ".join(f"{l} => {c:+.4f}" for l, c in zip(self.partner_labels, self.components))
        return f"StateMixing({self.mechanism}: {parts})"


class _Labelled:
    basis: BasisState

    @property
    def n(self) -> int:
        return self.basis.n

    @property
    def L(self) -> str:
        return self.basis.L_label

    @property
    def multiplicity(self) -> int:
        return self.basis.multiplicity

    @property
    def J(self) -> int:
        return self.basis.J

    @property
    def label(self) -> str:
        return self.basis.label


@dataclass(frozen=True)
class CentralState(_Labelled):
    """One level after the spin-independent central solve only."""

    basis: BasisState
    central_GeV: float

    def __repr__(self) -> str:
        return f"CentralState({self.label}, {self.central_GeV:.4f} GeV)"


@dataclass(frozen=True)
class CorrectedState(_Labelled):
    """One eigenstate of the complete fixed-sector Hamiltonian; contributions sum to ``mass_GeV``."""

    basis: BasisState
    central_GeV: float
    contact_shift_GeV: float
    spin_orbit_vector_shift_GeV: float
    spin_orbit_thomas_shift_GeV: float
    spin_orbit_shift_GeV: float
    tensor_shift_GeV: float
    fine_structure_shift_GeV: float
    fine_structure_mass_convention: str
    mass_GeV: float

    def __repr__(self) -> str:
        return f"CorrectedState({self.label}, {self.mass_GeV:.4f} GeV)"


@dataclass(frozen=True)
class MixedState(_Labelled):
    """A level of a mixed spectrum. Unknown attributes forward to ``corrected``."""

    corrected: CorrectedState
    mixings: tuple[StateMixing, ...]
    fine_structure_mass_convention: str
    mass_GeV: float

    @property
    def basis(self) -> BasisState:  # type: ignore[override]
        return self.corrected.basis

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "corrected":
            raise AttributeError(name)
        return getattr(self.corrected, name)

    def _with_mixing(self, mixing: StateMixing, mass_GeV: float, convention: str | None = None) -> MixedState:
        return MixedState(
            self.corrected,
            self.mixings + (mixing,),
            self.fine_structure_mass_convention if convention is None else convention,
            float(mass_GeV),
        )

    def __repr__(self) -> str:
        return f"MixedState({self.label}, {self.mass_GeV:.4f} GeV)"


State = TypeVar("State", CentralState, CorrectedState, MixedState)


@dataclass(eq=False)
class Spectrum(Generic[State]):
    """Staged model spectrum: flavor ``channels``, ``states`` in request order, and the
    :class:`~gimodel.channel.SectorComputation` holding the cached radial solves."""

    channels: list[Meson]
    states: list[State]
    computation: SectorComputation
    nonstrange_isoscalar: bool = False

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("Spectrum needs at least one flavor channel")
        self.channels = list(self.channels)
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Spectrum flavor channels must be unique")
        flavors = {m.flavors for m in self.channels}
        if len(flavors) != len(self.channels):
            raise ValueError("Spectrum channels must have distinct flavor identities")
        self.states = list(self.states)
        for s in self.states:
            if s.basis.flavors is None:
                raise ValueError(f"Spectrum state {s.label} must carry explicit flavor identity")
            if s.basis.flavors not in flavors:
                raise ValueError(
                    f"Spectrum state {s.label} belongs to {s.basis.flavors}, which is absent from the spectrum channels"
                )

    @property
    def stage(self) -> str:
        if not self.states:
            return "Spectrum"
        return {CentralState: "CentralSpectrum", CorrectedState: "CorrectedSpectrum", MixedState: "MixedSpectrum"}[
            type(self.states[0])
        ]

    def __str__(self) -> str:
        stage = self.stage
        chan = " + ".join(flavor_label(m) for m in self.channels)
        lines = [f"{stage}: {chan}, {len(self.states)} levels — all values in GeV"]
        if stage == "CentralSpectrum":
            cols = ("central",)
        elif stage == "CorrectedSpectrum":
            cols = ("central", "contact", "fine str", "mass")
        else:
            cols = ("central", "contact", "fine str", "mixing", "mass")
        width = max([5] + [len(s.label) for s in self.states])
        lines.append("  " + "level".ljust(width) + "".join(c.rjust(11) for c in cols))
        for s in self.states:
            if isinstance(s, CentralState):
                vals = (s.central_GeV,)
            elif isinstance(s, CorrectedState):
                vals = (s.central_GeV, s.contact_shift_GeV, s.fine_structure_shift_GeV, s.mass_GeV)
            else:
                vals = (
                    s.central_GeV,
                    s.contact_shift_GeV,
                    s.fine_structure_shift_GeV,
                    s.mass_GeV - s.corrected.mass_GeV,
                    s.mass_GeV,
                )
            lines.append("  " + s.label.ljust(width) + "".join(f"{v:11.4f}" for v in vals))
        if stage == "MixedSpectrum":
            nmix = sum(1 for s in self.states if s.mixings)
            if nmix:
                lines.append(f"  ({nmix} levels carry mixing; see `spec.states[i].mixings`)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        chan = "+".join(flavor_label(m) for m in self.channels)
        return f"{self.stage}({chan}, {len(self.states)} levels)"


def parameters(spec: Spectrum) -> GIParameters:
    """The parameters that produced ``spec``."""
    return spec.computation.params


def _single_channel(spec: Spectrum) -> Meson:
    if len(spec.channels) != 1:
        raise ValueError(f"operation requires a single-channel spectrum; got {len(spec.channels)} channels")
    return spec.channels[0]


def _channel_for(spec: Spectrum, basis: BasisState) -> Meson:
    if basis.flavors is None:
        return _single_channel(spec)
    for meson in spec.channels:
        if meson.flavors == basis.flavors:
            return meson
    raise ValueError(f"spectrum has no flavor channel {basis.flavors} for {basis.label}")


def _with_flavors(level: BasisState, meson: Meson) -> BasisState:
    return BasisState(level.n, level.L_label, level.multiplicity, level.J, label=level.label, flavors=meson.flavors)


def _validate_levels(levels: Sequence[BasisState], meson: Meson, solver: RadialSolver, who: str) -> None:
    if not levels:
        raise ValueError(f"{who}: empty `levels`")
    keys = [(l.n, l.L_label, l.multiplicity, l.J) for l in levels]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{who}: duplicate requested spectroscopic levels")
    for level in levels:
        if level.flavors is not None and level.flavors != meson.flavors:
            raise ValueError(f"{who}: level {level.label} belongs to {level.flavors}, not {flavor_label(meson)}")
        if level.L_label not in L_SYMBOLS:
            raise ValueError(f"{who}: unknown orbital label `{level.L_label}`")
        if not 1 <= level.n <= solver.nlevels_per_channel:
            raise ValueError(
                f"{who}: level {level.label} has n={level.n} outside 1:{solver.nlevels_per_channel} "
                "(raise nlevels_per_channel)"
            )


# --- stages ------------------------------------------------------------------------------


def central_spectrum(
    params: GIParameters,
    meson: Meson,
    *,
    levels: Sequence[BasisState] | None = None,
    solver: RadialSolver | None = None,
) -> Spectrum[CentralState]:
    """Diagnostic: one spin-independent solve per distinct orbital in ``levels``."""
    levels = spectrum_levels(2) if levels is None else list(levels)
    solver = FiniteDifferenceSolver() if solver is None else solver
    _validate_levels(levels, meson, solver, "central_spectrum")
    masses = meson.constituent_masses
    cache: dict[RadialChannelKey, ChannelRadialSolution] = {}
    for L_label in sorted({l.L_label for l in levels}):
        requested = max(l.n for l in levels if l.L_label == L_label)
        cache[RadialChannelKey.of(masses, L_label)] = channel_solution(
            params, masses, L_SYMBOLS[L_label], nlevels=requested, solver=solver
        )
    states = []
    for level in levels:
        sol = cache[RadialChannelKey.of(masses, level.L_label)]
        if level.n > len(sol.eigenvalues_GeV):
            raise ValueError(
                f"central_spectrum: channel `{level.L_label}` returned only {len(sol.eigenvalues_GeV)} levels; "
                f"requested n={level.n}"
            )
        states.append(CentralState(_with_flavors(level, meson), float(sol.eigenvalues_GeV[level.n - 1])))
    return Spectrum([meson], states, SectorComputation(params, solver, cache))


def _mass_convention(masses: ConstituentMasses) -> str:
    return "equal_mass" if masses.m1_GeV == masses.m2_GeV else "unequal_mass_equal_share_LdotS"


def fixed_spectrum(
    params: GIParameters,
    meson: Meson,
    *,
    levels: Sequence[BasisState] | None = None,
    solver: RadialSolver | None = None,
    terms: SpinTerms = SpinTerms(),
) -> Spectrum[CorrectedState]:
    """Stage 1: solve every requested fixed ``(L,S,J)`` sector once and decompose its masses."""
    levels = spectrum_levels(2) if levels is None else list(levels)
    solver = FiniteDifferenceSolver() if solver is None else solver
    _validate_levels(levels, meson, solver, "fixed_spectrum")
    masses = meson.constituent_masses
    required: dict[tuple[str, int, int], int] = {}
    for level in levels:
        sector = (level.L_label, level.multiplicity, level.J)
        required[sector] = max(required.get(sector, 0), level.n)
    cache: dict[RadialChannelKey, ChannelRadialSolution] = {}
    states: list[CorrectedState] = []
    for level in levels:
        multiplet = FineStructureMultiplet(level.L_label, level.multiplicity, level.J)
        key = RadialChannelKey.of(masses, level.L_label, level.multiplicity, level.J)
        if key not in cache:
            cache[key] = fixed_channel_solution(
                params,
                masses,
                multiplet,
                solver=solver,
                terms=terms,
                nlevels=required[(level.L_label, level.multiplicity, level.J)],
            )
        sol = cache[key]
        wave = sol.radial_wave(level.n)
        mass = float(sol.eigenvalues_GeV[level.n - 1])
        contact = contact_hyperfine_shift_active(params, masses, multiplet, wave) if terms.contact_hyperfine else 0.0
        so_vector = so_thomas = so_total = tensor = fs_total = 0.0
        convention = "disabled"
        requested_fs = terms.fine_structure and params.fine_structure.enabled and level.L_label != "S"
        active_fs = terms.fine_structure and diagonal_fine_structure_active(
            params, L_SYMBOLS[level.L_label], level.multiplicity
        )
        if active_fs:
            comp = fine_structure_components(params, masses, multiplet, wave, enabled=True)
            so_vector, so_thomas = comp.spin_orbit_vector, comp.spin_orbit_thomas
            so_total = so_vector + so_thomas
            tensor = comp.tensor
            fs_total = comp.total
            convention = _mass_convention(masses)
        elif requested_fs:
            convention = _mass_convention(masses)
        states.append(
            CorrectedState(
                _with_flavors(level, meson),
                mass - contact - fs_total,
                contact,
                so_vector,
                so_thomas,
                so_total,
                tensor,
                fs_total,
                convention,
                mass,
            )
        )
    return Spectrum([meson], states, SectorComputation(params, solver, cache))


def _cached_state_wave(cache: dict, masses: ConstituentMasses, state) -> RadialWave:
    fixed = RadialChannelKey.of(masses, state.L, state.multiplicity, state.J)
    key = fixed if fixed in cache else RadialChannelKey.of(masses, state.L)
    return cache[key].radial_wave(state.n)


def _spectrum_state_wave(spec: Spectrum, state) -> RadialWave:
    meson = _channel_for(spec, state.basis)
    return _cached_state_wave(spec.computation.channel_cache, meson.constituent_masses, state)


def _assign_block_members(
    states: list[MixedState], members: list[int], result: MixingResult, convention: str | None = None
) -> None:
    if len(members) != len(result.block.basis):
        raise ValueError("mixing member count does not match the result basis")
    ordered = sorted(members, key=lambda i: states[i].mass_GeV)
    for rank, i in enumerate(ordered):
        mixing = StateMixing(result, rank, states[i].mass_GeV)
        states[i] = states[i]._with_mixing(mixing, float(result.masses[rank]), convention)


def _apply_same_j_spin_orbit_mixing(states: list[MixedState], params, masses, cache) -> None:
    groups: dict[tuple[str, int], list[int]] = {}
    for i, s in enumerate(states):
        if s.L not in L_SYMBOLS:
            continue
        L = L_SYMBOLS[s.L]
        if L > 0 and s.J == L and s.multiplicity in (1, 3):
            groups.setdefault((s.L, s.J), []).append(i)
    for indices in groups.values():
        singlets = [i for i in indices if states[i].multiplicity == 1]
        triplets = [i for i in indices if states[i].multiplicity == 3]
        if not singlets or not triplets:
            continue
        ordered = sorted(singlets, key=lambda i: states[i].n) + sorted(triplets, key=lambda i: states[i].n)
        matrix = np.diag([states[i].mass_GeV for i in ordered])
        for i in singlets:
            for j in triplets:
                left = _cached_state_wave(cache, masses, states[i])
                right = _cached_state_wave(cache, masses, states[j])
                element = spin_orbit_mixing_components(params, masses, states[i].L, left, right, enabled=True).total
                row, col = ordered.index(i), ordered.index(j)
                matrix[row, col] = matrix[col, row] = element
        block = MixingBlock(
            "same-J antisymmetric spin-orbit",
            tuple(states[i].basis for i in ordered),
            matrix,
            mechanism=ANTISYMMETRIC_SPIN_ORBIT,
            source="Godfrey-Isgur Eqs. (6)-(7)",
        )
        _assign_block_members(states, ordered, diagonalize_mixing_block(block), "unequal_mass_same_j_mixed")


def _apply_tensor_mixing(states: list[MixedState], params, masses, cache) -> None:
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(states):
        if s.multiplicity != 3 or s.L not in L_SYMBOLS or s.J <= 0:
            continue
        L = L_SYMBOLS[s.L]
        if L in (s.J - 1, s.J + 1):
            groups.setdefault(s.J, []).append(i)
    for indices in groups.values():
        low = [i for i in indices if L_SYMBOLS[states[i].L] == states[i].J - 1]
        high = [i for i in indices if L_SYMBOLS[states[i].L] == states[i].J + 1]
        if not low or not high:
            continue
        ordered = sorted(low, key=lambda i: states[i].n) + sorted(high, key=lambda i: states[i].n)
        matrix = np.diag([states[i].mass_GeV for i in ordered])
        for i in low:
            for j in high:
                low_wave = _cached_state_wave(cache, masses, states[i])
                high_wave = _cached_state_wave(cache, masses, states[j])
                element = tensor_mixing_components(params, masses, low_wave, high_wave, states[i].J).total
                row, col = ordered.index(i), ordered.index(j)
                matrix[row, col] = matrix[col, row] = element
        block = MixingBlock(
            "same-J tensor triplet L/L'",
            tuple(states[i].basis for i in ordered),
            matrix,
            mechanism=TENSOR_MIXING,
        )
        _assign_block_members(states, ordered, diagonalize_mixing_block(block))


def add_intra_meson_mixing(spec: Spectrum[CorrectedState], *, terms: SpinTerms = SpinTerms()) -> Spectrum[MixedState]:
    """Stage 2: same-J antisymmetric spin-orbit mixing (unequal flavour), then tensor mixing.

    Mixing acts only when stage 1 applied fine structure (some convention != ``"disabled"``).
    """
    if spec.states and not isinstance(spec.states[0], CorrectedState):
        raise TypeError("add_intra_meson_mixing requires a CorrectedSpectrum (from fixed_spectrum)")
    params = parameters(spec)
    meson = _single_channel(spec)
    masses = meson.constituent_masses
    cache = spec.computation.channel_cache
    states = [MixedState(s, (), s.fine_structure_mass_convention, s.mass_GeV) for s in spec.states]
    applied = any(s.fine_structure_mass_convention != "disabled" for s in spec.states)
    if terms.same_j_spin_orbit and applied and not is_equal_flavor(meson):
        _apply_same_j_spin_orbit_mixing(states, params, masses, cache)
    if terms.tensor and applied:
        _apply_tensor_mixing(states, params, masses, cache)
    return Spectrum(spec.channels, states, spec.computation, nonstrange_isoscalar=spec.nonstrange_isoscalar)


def compute_spectrum(
    params: GIParameters,
    meson: Meson,
    *,
    levels: Sequence[BasisState] | None = None,
    solver: RadialSolver | None = None,
    terms: SpinTerms = SpinTerms(),
) -> Spectrum[MixedState]:
    """Production stages: :func:`fixed_spectrum` then :func:`add_intra_meson_mixing`."""
    corrected = fixed_spectrum(params, meson, levels=levels, solver=solver, terms=terms)
    return add_intra_meson_mixing(corrected, terms=terms)


# --- accessors ------------------------------------------------------------------------------


def spectrum_state(spec: Spectrum, *selector):
    """Look up one state by label (``"1^3S_1"``), :class:`BasisState`, or ``(n, L_label, multiplicity, J)``."""
    if len(selector) == 1 and isinstance(selector[0], str):
        label = selector[0]
        hits = [s for s in spec.states if s.label == label]
        if not hits:
            available = ", ".join(s.label for s in spec.states)
            raise ValueError(f"spectrum has no state `{label}`; available: {available}")
        if len(hits) > 1:
            raise ValueError(f"state label `{label}` is flavor-ambiguous; use a BasisState with explicit flavors")
        return hits[0]
    if len(selector) == 1 and isinstance(selector[0], BasisState):
        b = selector[0]
        hits = [
            s
            for s in spec.states
            if s.n == b.n
            and s.L == b.L_label
            and s.multiplicity == b.multiplicity
            and s.J == b.J
            and (b.flavors is None or s.basis.flavors == b.flavors)
        ]
        if not hits:
            raise ValueError(f"spectrum has no state matching {b.label} with flavors={b.flavors}")
        if len(hits) > 1:
            raise ValueError(f"state {b.label} is flavor-ambiguous; set `flavors` on BasisState")
        return hits[0]
    if len(selector) == 4:
        n, L_label, mult, J = selector
        hits = [s for s in spec.states if s.n == n and s.L == str(L_label) and s.multiplicity == mult and s.J == J]
        if not hits:
            raise ValueError(f"spectrum has no state {n}^{mult}{L_label}_{J}")
        if len(hits) > 1:
            raise ValueError(f"state {n}^{mult}{L_label}_{J} is flavor-ambiguous; pass a BasisState with flavors")
        return hits[0]
    raise TypeError("spectrum_state(spec, label | BasisState | n, L_label, multiplicity, J)")


def _resolve(spec: Spectrum, selector):
    if isinstance(selector, (CentralState, CorrectedState, MixedState)):
        return selector
    return spectrum_state(spec, selector)


def radial_wave(obj, selector) -> RadialWave:
    """``radial_wave(spec, state | label | BasisState)`` -> native wave of an *unmixed* level;
    ``radial_wave(solution, n)`` -> radial level ``n`` (1-based) of a channel solution."""
    if isinstance(obj, ChannelRadialSolution):
        return obj.radial_wave(selector)
    state = _resolve(obj, selector)
    if isinstance(state, MixedState) and state.mixings:
        raise ValueError(
            f"{state.label} is a mixed physical state; use physical_components instead of "
            "selecting one precursor radial wave"
        )
    return _spectrum_state_wave(obj, state)


class PhysicalComponent(NamedTuple):
    basis: BasisState
    coefficient: float
    wave: RadialWave


def physical_components(spec: Spectrum, selector) -> list[PhysicalComponent]:
    """Flattened signed composition of a physical state: ``(basis, coefficient, wave)`` entries."""
    state = _resolve(spec, selector)
    if not isinstance(state, MixedState) or not state.mixings:
        return [PhysicalComponent(state.basis, 1.0, _spectrum_state_wave(spec, state))]
    mixing = state.mixings[-1]
    result = mixing.result
    coefficients = result.vectors[:, mixing.eigenstate]
    components: list[PhysicalComponent] = []
    for i, basis in enumerate(result.block.basis):
        component_state = spectrum_state(spec, basis)
        sources = [k for k, m in enumerate(component_state.mixings) if m.result is result]
        if len(sources) != 1:
            raise ValueError(f"{basis.label} does not carry the shared {result.block.name} result")
        source = sources[0]
        prior = MixedState(
            component_state.corrected,
            component_state.mixings[:source],
            component_state.fine_structure_mass_convention,
            component_state.mixings[source].unmixed_GeV,
        )
        for comp in physical_components(spec, prior):
            components.append(PhysicalComponent(comp.basis, float(coefficients[i]) * comp.coefficient, comp.wave))
    merged: list[PhysicalComponent] = []
    positions: dict[BasisState, int] = {}
    for comp in components:
        pos = positions.get(comp.basis)
        if pos is None:
            positions[comp.basis] = len(merged)
            merged.append(comp)
        else:
            prior_comp = merged[pos]
            merged[pos] = PhysicalComponent(prior_comp.basis, prior_comp.coefficient + comp.coefficient, prior_comp.wave)
    return merged


def physical_state_amplitude(kernel: Callable[[PhysicalComponent], float], spec: Spectrum, selector) -> float:
    """``sum_c coefficient_c * kernel(c)`` over the physical components of a state."""
    return sum(c.coefficient * kernel(c) for c in physical_components(spec, selector))


def physical_transition_amplitude(
    kernel: Callable[[PhysicalComponent, PhysicalComponent], float], spec: Spectrum, left, right
) -> float:
    """Coherent bilinear ``sum a.coefficient b.coefficient kernel(a, b)``."""
    lc = physical_components(spec, left)
    rc = physical_components(spec, right)
    return sum(a.coefficient * b.coefficient * kernel(a, b) for a in lc for b in rc)


def spectrum_radial_expect(spec: Spectrum, selector, f) -> float:
    """``<f(r)>`` in a physical state, keeping radial interference within one ``(L,S,J)`` sector."""
    comps = physical_components(spec, selector)
    return sum(
        a.coefficient * b.coefficient * a.wave.radial_overlap(b.wave, f)
        for a in comps
        for b in comps
        if a.basis.L_label == b.basis.L_label
        and a.basis.multiplicity == b.basis.multiplicity
        and a.basis.J == b.basis.J
        and a.basis.flavors == b.basis.flavors
    )
