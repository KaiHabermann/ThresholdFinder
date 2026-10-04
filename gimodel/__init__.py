"""gimodel -- Python port of the core of GIModel.jl (Godfrey–Isgur relativized quark model).

Quick start::

    from gimodel import load_parameters_and_quark_masses, Meson, compute_spectrum, spectrum_state
    params, mq = load_parameters_and_quark_masses()
    spec = compute_spectrum(params, Meson.from_table(mq, "c", "c"))
    spectrum_state(spec, "1^3P_1").mass_GeV

Units: GeV and GeV^-1 throughout.
"""
from __future__ import annotations

from .channel import (
    ChannelRadialSolution,
    RadialChannelKey,
    SectorComputation,
    channel_solution,
    contact_hyperfine_nonperturbative_states,
    fixed_channel_matrices,
    fixed_channel_solution,
    resummed_channel_solution,
    solve_sector,
)
from .constants import ALPHA_COEFFS, ALPHA_GAMMAS, L_SYMBOLS, orbital_angular_momentum, orbital_label
from .coupling import alpha_s_q, alpha_s_r
from .mixing import BasisState, MixingBlock, MixingResult, diagonalize_mixing_block, same_j_mixing
from .parameters import (
    AnnihilationAmplitudes,
    CentralPotentialMethod,
    ConfinementPotential,
    FineStructure,
    GIParameters,
    QuarkMassTable,
    RelativisticFactors,
    RelativisticSmearing,
    central_potential_method,
    default_parameters_path,
    load_parameters,
    load_parameters_and_quark_masses,
    load_quark_masses,
)
from .quarks import (
    ConstituentMasses,
    HeavyQuark,
    LightQuark,
    Meson,
    StrangeQuark,
    charge,
    flavor_label,
    flavor_symbol,
    is_equal_flavor,
    mass_GeV,
    reduced_mass,
)
from .smearing import central_potential_values, contact_smearing_sigma
from .solvers import FiniteDifferenceSolver, OscillatorSolver, RadialSolver, SpinTerms
from .spectrum import (
    CentralState,
    CorrectedState,
    MixedState,
    PhysicalComponent,
    Spectrum,
    StateMixing,
    add_intra_meson_mixing,
    central_spectrum,
    compute_spectrum,
    fixed_spectrum,
    parameters,
    physical_components,
    physical_state_amplitude,
    physical_transition_amplitude,
    radial_wave,
    spectrum_levels,
    spectrum_state,
)
from .spin import (
    ContactHyperfine,
    FineStructureMultiplet,
    contact_hyperfine_shift,
    contact_hyperfine_shift_active,
    contact_hyperfine_shift_momentum_sandwich,
    fine_structure_components,
    fine_structure_grid_matrices,
    fine_structure_radial_kernels,
    fine_structure_split,
    l_dot_s,
    spin_dot,
    spin_orbit_mixing_components,
    tensor_mixing_components,
    tensor_triplet_lj,
    tensor_triplet_offdiag_same_j,
)
from .ho import OscillatorConvergence
from .waves import (
    MeshMomentumWave,
    MeshWave,
    OscillatorMomentumWave,
    OscillatorWave,
    RadialWave,
    WaveMeanSquares,
    momentum_expect,
    momentum_functional,
    momentum_overlap,
    momentum_wave,
    radial_derivative_overlap,
    radial_expect,
    radial_overlap,
    wave_mean_squares,
    wave_norm,
)

__version__ = "0.1.0"

__all__ = [name for name in dir() if not name.startswith("_") and name not in {"annotations"}]
