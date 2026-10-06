# Generate layered reference data from GIModel.jl for validating the Python port.
#
# Usage (see README.md):
#   GIMODEL_JL_PATH=/path/to/GIModel.jl julia generate_reference.jl [targets...]
#
# targets: primitives ho_matrices fd_matrices channels spectra_fd spectra_ho
#          variations spot_checks   (default: all)
#
# Output directory: ../../tests/gimodel/reference (override with GIMODEL_REFERENCE_OUT).

import Pkg
const HERE = @__DIR__
Pkg.activate(HERE; io = devnull)
let gipath = get(ENV, "GIMODEL_JL_PATH", "")
    if !isempty(gipath)
        Pkg.develop(path = abspath(gipath); io = devnull)
    end
    Pkg.instantiate(; io = devnull)
end

using GIModel
using LinearAlgebra
using Dates

const OUTDIR = abspath(get(ENV, "GIMODEL_REFERENCE_OUT", joinpath(HERE, "..", "..", "tests", "gimodel", "reference")))
mkpath(OUTDIR)

# =============================================================================
# Minimal deterministic JSON writer.
#
# Floats are written with Julia's shortest round-trip representation (`repr`),
# which reproduces the exact IEEE-754 double when parsed by Python's float()
# (equivalent to, and never less precise than, %.17g). Non-finite values -> null.
# Matrices are written row-major as a list of rows. NamedTuples keep key order;
# Dicts are written with sorted keys.
# =============================================================================

jstr(io, s::AbstractString) = begin
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base = 16, pad = 4))
        else
            print(io, c)
        end
    end
    print(io, '"')
end

jwrite(io, x::Nothing) = print(io, "null")
jwrite(io, x::Bool) = print(io, x ? "true" : "false")
jwrite(io, x::Integer) = print(io, x)
jwrite(io, x::Rational) = jwrite(io, float(x))
jwrite(io, x::AbstractFloat) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
jwrite(io, x::AbstractString) = jstr(io, x)
jwrite(io, x::Symbol) = jstr(io, String(x))
function jwrite(io, x::Union{AbstractVector,Tuple})
    print(io, '[')
    first = true
    for v in x
        first || print(io, ',')
        first = false
        jwrite(io, v)
    end
    print(io, ']')
end
function jwrite(io, x::AbstractMatrix)
    print(io, '[')
    for i in axes(x, 1)
        i == first(axes(x, 1)) || print(io, ',')
        jwrite(io, collect(view(x, i, :)))
    end
    print(io, ']')
end
function _jobject(io, kvs)
    print(io, '{')
    first = true
    for (k, v) in kvs
        first || print(io, ',')
        first = false
        jstr(io, string(k))
        print(io, ':')
        jwrite(io, v)
    end
    print(io, '}')
end
jwrite(io, x::NamedTuple) = _jobject(io, pairs(x))
jwrite(io, x::AbstractDict) = _jobject(io, sort!(collect(pairs(x)); by = p -> string(first(p))))
jwrite(io, x::AbstractVector{<:Pair}) = _jobject(io, x)   # ordered object

function write_json(name::AbstractString, data)
    path = joinpath(OUTDIR, name)
    open(path, "w") do io
        jwrite(io, data)
        println(io)
    end
    sz = filesize(path)
    println("  wrote ", path, " (", round(sz / 1024; digits = 1), " KiB)")
    return path
end

# Ordered object helper.
obj(kvs::Pair...) = Pair{String,Any}[string(k) => v for (k, v) in kvs]

# =============================================================================
# Setup and metadata
# =============================================================================

const PARAMS, MQ = load_parameters_and_quark_masses(default_parameters_path())
const GIDIR = pkgdir(GIModel)

function git_commit(dir)
    try
        return strip(read(`git -C $dir rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end
function git_dirty(dir)
    try
        return !isempty(strip(read(`git -C $dir status --porcelain`, String)))
    catch
        return nothing
    end
end

params_dict(p::GIParameters) = obj(
    "potential" => obj("b_GeV2" => p.potential.b, "c_GeV" => p.potential.c),
    "central" => string(nameof(typeof(p.central))),
    "smearing" => obj("sigma0_GeV" => p.smearing.sigma0, "s" => p.smearing.s),
    "factors" => obj(
        "epsilon_c" => p.factors.epsilon_c,
        "epsilon_t" => p.factors.epsilon_t,
        "epsilon_so_vector" => p.factors.epsilon_so_vector,
        "epsilon_so_scalar" => p.factors.epsilon_so_scalar,
        "contact_momentum_sandwich" => p.factors.contact_momentum_sandwich,
        "fine_structure_momentum_sandwich" => p.factors.fine_structure_momentum_sandwich,
        "fine_structure_smeared_kernels" => p.factors.fine_structure_smeared_kernels,
    ),
    "fine_structure_enabled" => p.fine_structure.enabled,
    "annihilation" => obj(
        "p1_A_np" => p.annihilation.p1_A_np, "p1_m_eta" => p.annihilation.p1_m_eta,
        "p2_A_np" => p.annihilation.p2_A_np, "p2_M0" => p.annihilation.p2_M0,
        "s1_A" => p.annihilation.s1_A, "a_3p2" => p.annihilation.a_3p2,
    ),
)

solver_dict(s::FiniteDifferenceSolver) = obj(
    "type" => "FiniteDifferenceSolver", "ngrid" => s.ngrid, "rmax" => s.rmax,
    "kinetic" => s.kinetic, "eigensolver" => s.eigensolver,
    "nlevels_per_channel" => s.nlevels_per_channel,
)
solver_dict(s::OscillatorSolver) = obj(
    "type" => "OscillatorSolver", "nbasis" => s.nbasis, "max_nbasis" => s.max_nbasis,
    "basis_step" => s.basis_step, "energy_tolerance_GeV" => s.energy_tolerance_GeV,
    "beta_grid" => s.beta_grid, "beta_tolerance_GeV" => s.beta_tolerance_GeV,
    "converge" => s.converge, "nlevels_per_channel" => s.nlevels_per_channel,
)
terms_dict(t::SpinTerms) = obj(
    "contact_hyperfine" => t.contact_hyperfine, "fine_structure" => t.fine_structure,
    "same_j_spin_orbit" => t.same_j_spin_orbit, "tensor" => t.tensor,
)

function metadata(file_description; elapsed = nothing, extra...)
    return obj(
        "description" => file_description,
        "generator" => "tools/julia_reference/generate_reference.jl",
        "julia_version" => string(VERSION),
        "gimodel_version" => string(pkgversion(GIModel)),
        "gimodel_commit" => git_commit(GIDIR),
        "gimodel_worktree_dirty" => git_dirty(GIDIR),
        "gimodel_repository" => "https://github.com/mmikhasenko/GIModel.jl",
        "date_utc" => string(Dates.now(Dates.UTC)),
        "blas" => string(LinearAlgebra.BLAS.get_config()),
        "float_format" => "shortest round-trip repr of IEEE-754 double (exact when parsed as float64); non-finite -> null",
        "matrix_layout" => "row-major list of rows",
        "units" => "masses/energies GeV, r GeV^-1, p GeV",
        "parameters_file" => "data/parameters.provisional.toml (default_parameters_path())",
        "parameters" => params_dict(PARAMS),
        "quark_masses_GeV" => obj((k => MQ[k] for k in ("u", "d", "q", "s", "c", "b"))...),
        "generation_seconds" => elapsed,
        extra...,
    )
end

const WAVE_R = collect(range(0.0, 20.0; length = 81))   # h = 0.25 GeV^-1
const TOY = FiniteDifferenceSolver(ngrid = 20, rmax = 10.0)

masses_of(a::Symbol, b::Symbol) = Meson(MQ, a, b).constituent_masses
mpair(m::ConstituentMasses) = [m.m1_GeV, m.m2_GeV]

const MULTIPLETS = [
    ("S", 1, 0), ("S", 3, 1),
    ("P", 1, 1), ("P", 3, 0), ("P", 3, 1), ("P", 3, 2),
    ("D", 1, 2), ("D", 3, 1), ("D", 3, 2), ("D", 3, 3),
]
mlabel(L, m, J) = "^$(m)$(L)_$(J)"

# =============================================================================
# Wave / solution serialization
# =============================================================================

convergence_dict(::Nothing) = nothing
convergence_dict(c::OscillatorConvergence) = obj(
    "status" => c.status, "beta_GeV" => c.beta_GeV, "nbasis" => c.nbasis,
    "energy_delta_GeV" => c.energy_delta_GeV,
    "max_wave_overlap_defect" => c.max_wave_overlap_defect,
    "tolerance_GeV" => c.tolerance_GeV, "refinements" => c.refinements,
)

native_wave(w::MeshWave) = obj("type" => "MeshWave", "h" => w.h, "r_first" => w.r[1], "n" => length(w.r), "u" => w.u)
native_wave(w::OscillatorWave) = obj("type" => "OscillatorWave", "L" => w.L, "beta" => w.beta, "coefficients" => w.coefficients)

function solution_dict(sol::ChannelRadialSolution; native = true)
    return obj(
        "eigenvalues_GeV" => sol.eigenvalues_GeV,
        "convergence" => convergence_dict(sol.convergence),
        "waves_sampled" => [sample_wave(w, WAVE_R).u for w in sol.waves],
        "waves_native" => native ? [native_wave(w) for w in sol.waves] : nothing,
    )
end

# =============================================================================
# a. primitives.json
# =============================================================================

function gen_primitives()
    p = PARAMS
    r_grid = [0.0, 1.0e-9, 9.9e-9, 1.0e-8, 1.01e-8, 5.0e-8, 9.9e-8, 1.0e-7, 1.01e-7, 2.0e-7,
        1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5,
        2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    q_grid = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    pairs_spec = [(:u, :u), (:u, :s), (:s, :s), (:c, :u), (:c, :s), (:c, :c),
        (:b, :u), (:b, :s), (:b, :c), (:b, :b)]
    pair_data = Pair{String,Any}[]
    for (a, b) in pairs_spec
        m = masses_of(a, b)
        m1, m2 = m.m1_GeV, m.m2_GeV
        σ = contact_smearing_sigma(p, m1, m2)
        τ = [1 / sqrt(1 / σ^2 + 1 / γ^2) for γ in GIModel.ALPHA_GAMMAS]
        fsk = [GIModel.fine_structure_radial_kernels(p, m, r) for r in r_grid]
        push!(pair_data, string(a, b) => obj(
            "m1_GeV" => m1, "m2_GeV" => m2,
            "contact_smearing_sigma_GeV" => σ,
            "tau_k_GeV" => τ,
            "G_tilde" => [GIModel.smeared_coulomb_G_closed(p, m1, m2, r) for r in r_grid],
            "S_tilde" => [GIModel.smeared_confinement_S_closed(p, m1, m2, r) for r in r_grid],
            "central_closed_G_plus_S" => GIModel.appendix_a_closed_central_values(p, m1, m2, r_grid),
            "G_tilde_prime" => [GIModel.smeared_coulomb_G_prime_closed(p, m1, m2, r) for r in r_grid],
            "G_tilde_second" => [GIModel.smeared_coulomb_G_second_closed(p, m1, m2, r) for r in r_grid],
            "S_tilde_prime" => [GIModel.smeared_confinement_S_prime_closed(p, m1, m2, r) for r in r_grid],
            "tensor_kernel_smeared" => [GIModel.tensor_kernel_smeared_coulomb(p, m1, m2, r) for r in r_grid],
            "contact_kernel_smeared" => [GIModel.smeared_contact_kernel(p, m, r) for r in r_grid],
            "delta_sigma_3d_at_sigma" => [GIModel.delta_sigma_3d(r, σ) for r in r_grid],
            "fine_structure_radial_kernels" => obj(
                (string(k) => [getproperty(x, k) for x in fsk] for k in
                 (:vector_11, :vector_22, :vector_12, :scalar_11, :scalar_22, :tensor_12))...,
            ),
        ))
    end
    ls_table = []
    tensor_table = []
    for L in 0:4, S in 0:1, J in abs(L - S):(L + S)
        push!(ls_table, obj("L" => L, "S" => S, "J" => J, "LdotS" => LdotS(L, S, J)))
        push!(tensor_table, obj("L" => L, "S" => S, "J" => J,
            "tensor_triplet_LJ" => tensor_triplet_LJ(L, J, S)))
    end
    return obj(
        "metadata" => nothing,
        "constants" => obj(
            "ALPHA_COEFFS" => collect(GIModel.ALPHA_COEFFS),
            "ALPHA_GAMMAS" => collect(GIModel.ALPHA_GAMMAS),
        ),
        "notes" => obj(
            "G_tilde" => "GIModel.smeared_coulomb_G_closed(params,m1,m2,r): |r|<1e-8 uses r->0 limit",
            "S_tilde" => "GIModel.smeared_confinement_S_closed: |r|<1e-8 uses r->0 limit",
            "G_tilde_prime" => "GIModel.smeared_coulomb_G_prime_closed: r clamped to max(r,1e-7)",
            "G_tilde_second" => "GIModel.smeared_coulomb_G_second_closed: r clamped to max(r,1e-7)",
            "S_tilde_prime" => "GIModel.smeared_confinement_S_prime_closed: returns 0 for |r|<1e-7",
            "tensor_kernel_smeared" => "GIModel.tensor_kernel_smeared_coulomb = G'/r - G'' with r clamped to max(r,1e-7)",
            "contact_kernel_smeared" => "GIModel.smeared_contact_kernel = sum_k alpha_k * delta_{tau_k}(r), delta_s(r)=s^3/pi^1.5 exp(-s^2 r^2)",
            "fine_structure_radial_kernels" => "GIModel.fine_structure_radial_kernels(params,masses,r); vector_ii/scalar_ii use the equal-mass pair (m_i,m_i); kernels clamp r to max(r,1e-8) before G'/r, S'/r",
            "running" => "unsmeared running-coupling forms (not on the production path; diagnostics)",
        ),
        "r_grid" => r_grid,
        "q_grid" => q_grid,
        "alpha_s_r" => [alpha_s_r(r) for r in r_grid],
        "alpha_s_q" => [alpha_s_q(q) for q in q_grid],
        "running" => obj(
            "alpha_s_prime_r" => [GIModel.alpha_s_prime_r(r) for r in r_grid],
            "alpha_s_second_r" => [GIModel.alpha_s_second_r(r) for r in r_grid],
            "static_coulomb_G" => [GIModel.static_coulomb_G(r, p) for r in r_grid],
            "static_confinement_S" => [GIModel.static_confinement_S(r, p) for r in r_grid],
            "coulomb_G_running" => [GIModel.coulomb_G_running(r) for r in r_grid],
            "coulomb_G_prime_running" => [GIModel.coulomb_G_prime_running(r) for r in r_grid],
            "coulomb_G_second_running" => [GIModel.coulomb_G_second_running(r) for r in r_grid],
            "tensor_kernel_coulomb_running" => [GIModel.tensor_kernel_coulomb_running(r) for r in r_grid],
        ),
        "pairs" => pair_data,
        "spin_dot" => obj("1" => spin_dot(1), "3" => spin_dot(3)),
        "LdotS" => ls_table,
        "tensor_triplet_LJ" => tensor_table,
        "tensor_triplet_offdiag_sameJ" => [obj("J" => J, "value" => tensor_triplet_offdiag_sameJ(J, 1)) for J in 0:4],
        "gi_spin_dependent_side_exponent" => [obj("epsilon" => e, "value" => GIModel.gi_spin_dependent_side_exponent(e))
                                              for e in (0.0, PARAMS.factors.epsilon_c, PARAMS.factors.epsilon_t,
            PARAMS.factors.epsilon_so_vector, PARAMS.factors.epsilon_so_scalar)],
    )
end

# =============================================================================
# b. ho_matrices.json
# =============================================================================

function gen_ho_matrices()
    p = PARAMS
    cc = masses_of(:c, :c)
    cu = masses_of(:c, :u)
    mc = cc.m1_GeV
    entries = []
    for L in (0, 1, 2), β in (0.5, 1.1), N in (8, 24)
        A_cc = pm -> sqrt(1 + pm^2 / sqrt((pm^2 + mc^2) * (pm^2 + mc^2)))
        push!(entries, obj(
            "L" => L, "beta" => β, "nbasis" => N,
            "r2" => Matrix(ho_r2_matrix(L, β, N)),
            "p2" => Matrix(ho_p2_matrix(L, β, N)),
            "op_r" => Matrix(ho_operator_matrix(L, β, N, r -> r)),
            "op_exp_minus_r2" => Matrix(ho_operator_matrix(L, β, N, r -> exp(-r^2))),
            "op_G_tilde_cc" => Matrix(ho_operator_matrix(L, β, N, r -> GIModel.smeared_coulomb_G_closed(p, mc, mc, r))),
            "op_S_tilde_cc" => Matrix(ho_operator_matrix(L, β, N, r -> GIModel.smeared_confinement_S_closed(p, mc, mc, r))),
            "op_G_tilde_cu" => Matrix(ho_operator_matrix(L, β, N, r -> GIModel.smeared_coulomb_G_closed(p, cu.m1_GeV, cu.m2_GeV, r))),
            "mom_sqrt_p2_plus_mc2" => Matrix(GIModel._ho_momentum_operator_matrix(L, β, N, pm -> sqrt(pm^2 + mc^2))),
            "mom_sqrt_p2_plus_mu2" => Matrix(GIModel._ho_momentum_operator_matrix(L, β, N, pm -> sqrt(pm^2 + cu.m2_GeV^2))),
            "mom_A_cc" => Matrix(GIModel._ho_momentum_operator_matrix(L, β, N, A_cc)),
            "central_matrix_cc" => Matrix(GIModel.oscillator_central_matrix(p, cc, L, β, N)),
            "central_matrix_cu" => Matrix(GIModel.oscillator_central_matrix(p, cu, L, β, N)),
        ))
    end
    # Spin-dependent HO matrices (small basis).
    spin = []
    for (sys, m) in (("cc", cc), ("cu", cu)), β in (0.5, 1.1)
        N = 8
        contact = obj(
            ("L$(L)_mult$(mult)" => Matrix(GIModel.ho_contact_hyperfine_matrix(p, m, L, mult, β, N))
             for (L, mult) in ((0, 1), (0, 3), (1, 1), (1, 3)))...,
        )
        fine = []
        for (L, J) in ((1, 0), (1, 1), (1, 2), (2, 1), (2, 3))
            f = GIModel.ho_fine_structure_matrices(p, m, L, 3, J, β, N)
            push!(fine, obj("L" => L, "multiplicity" => 3, "J" => J,
                "spin_orbit_vector" => Matrix(f.spin_orbit_vector),
                "spin_orbit_thomas" => Matrix(f.spin_orbit_thomas),
                "spin_orbit" => Matrix(f.spin_orbit),
                "tensor" => Matrix(f.tensor),
                "total" => Matrix(f.total)))
        end
        sandwich_contact_L0 = Matrix(GIModel.ho_momentum_sandwich_matrix(
            0, β, N, m, p.factors.epsilon_c, r -> GIModel.smeared_contact_kernel(p, m, r); rtol = 1e-8))
        push!(spin, obj("system" => sys, "masses_GeV" => mpair(m), "beta" => β, "nbasis" => N,
            "contact_hyperfine" => contact,
            "momentum_sandwich_contact_kernel_L0_eps_c" => sandwich_contact_L0,
            "fine_structure" => fine))
    end
    rr = [0.0, 0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0]
    radial_funcs = [obj("n" => n, "L" => L, "beta" => β, "r" => rr,
        "u" => [GIModel.ho_reduced_radial(n, L, β, r) for r in rr])
                    for n in 0:3, L in 0:2, β in (0.5, 1.1)] |> vec
    dvr = [obj("L" => L, "nbasis" => 8, "nq" => 64,
        "sqrt_x" => GIModel.gauss_laguerre_dvr(L, 8, 64)[1]) for L in 0:2]
    return obj(
        "metadata" => nothing,
        "notes" => obj(
            "r2,p2" => "ho_r2_matrix / ho_p2_matrix (exact tridiagonal, stored dense)",
            "op_*" => "ho_operator_matrix(L,beta,N,g) default rtol=1e-10, nq starts at max(64,2N) and doubles until max|cur-prev| <= rtol*max(max|cur|,1) (nq_max 8192); DVR Golub-Welsch on ho_r2_matrix(L,1,nq)",
            "mom_*" => "GIModel._ho_momentum_operator_matrix(L,beta,N,f) = S * ho_operator_matrix(L,1/beta,N,f) * S, S=diag((-1)^n) (continuum momentum function projected, not f of truncated p2)",
            "mom_A_cc" => "f(p)=sqrt(1+p^2/sqrt((p^2+m1^2)(p^2+m2^2))) for cc",
            "central_matrix_*" => "GIModel.oscillator_central_matrix(params,masses,L,beta,N) = K(p) + C' G~ C + S~, C=A[:,1:N], A and G~ on work basis 2N+32",
            "contact_hyperfine" => "GIModel.ho_contact_hyperfine_matrix(params,masses,L,mult,beta,N) (momentum sandwich with epsilon_c, rtol=1e-8)",
            "momentum_sandwich_contact_kernel_L0_eps_c" => "GIModel.ho_momentum_sandwich_matrix(0,beta,N,masses,epsilon_c,smeared_contact_kernel; rtol=1e-8) WITHOUT the 32pi/(9 m1 m2) <S1.S2> strength",
            "fine_structure" => "GIModel.ho_fine_structure_matrices(params,masses,L,3,J,beta,N)",
            "ho_reduced_radial" => "GIModel.ho_reduced_radial(n,L,beta,r) basis function convention",
            "gauss_laguerre_dvr" => "GIModel.gauss_laguerre_dvr(L,nbasis,nq)[1] = sqrt of nodes x_i (beta-independent); Z eigenvector signs are LAPACK-arbitrary and cancel in Z*diag(g)*Z'",
        ),
        "entries" => entries,
        "spin_matrices" => spin,
        "ho_reduced_radial" => radial_funcs,
        "gauss_laguerre_dvr" => dvr,
    )
end

# =============================================================================
# c. fd_matrices.json (toy grid)
# =============================================================================

function gen_fd_matrices()
    p = PARAMS
    r, h = GIModel.radial_grid(TOY.ngrid, TOY.rmax)
    p2 = obj(("L$L" => Matrix(GIModel.p2_operator(1.0, L, r, h)) for L in 0:3)...)
    central = []
    for (sys, a, b) in (("cc", :c, :c), ("cu", :c, :u)), L in (0, 1, 2)
        m = masses_of(a, b)
        H, _ = GIModel.relativistic_hamiltonian(p, m, L; solver = TOY)
        fact = eigen(GIModel.p2_operator(p, m.m1_GeV, L, r, h))
        kin = GIModel.sqrt_kinetic_matrix_from_eigen(fact, m.m1_GeV) +
              GIModel.sqrt_kinetic_matrix_from_eigen(fact, m.m2_GeV)
        pot = GIModel.appendix_a_momentum_sandwich_matrix(p, m, r, fact)
        push!(central, obj("system" => sys, "masses_GeV" => mpair(m), "L" => L,
            "kinetic" => Matrix(kin), "potential" => Matrix(pot), "H0" => Matrix(H),
            "eigenvalues" => eigvals(Symmetric(Matrix(H)))))
    end
    fixed = []
    for (sys, a, b, L, mult, J) in (
        ("cc", :c, :c, "S", 1, 0), ("cc", :c, :c, "S", 3, 1),
        ("cc", :c, :c, "P", 1, 1), ("cc", :c, :c, "P", 3, 0), ("cc", :c, :c, "P", 3, 1), ("cc", :c, :c, "P", 3, 2),
        ("cc", :c, :c, "D", 3, 1),
        ("cu", :c, :u, "S", 1, 0), ("cu", :c, :u, "P", 1, 1), ("cu", :c, :u, "P", 3, 1), ("cu", :c, :u, "P", 3, 2),
    )
        m = masses_of(a, b)
        mats = GIModel._fixed_channel_matrices(TOY, p, m, FineStructureMultiplet(L, mult, J),
            nothing, SpinTerms(), TOY.nlevels_per_channel)
        push!(fixed, obj("system" => sys, "masses_GeV" => mpair(m), "L_label" => L,
            "multiplicity" => mult, "J" => J,
            ("$k" => Matrix(getproperty(mats, k)) for k in
             (:central, :contact, :spin_orbit_vector, :spin_orbit_thomas, :spin_orbit, :tensor, :fine_structure, :total))...,
            "eigenvalues_total" => eigvals(Symmetric(Matrix(mats.total)))))
    end
    # Mixing matrix elements on the toy grid, between fixed-channel ground states.
    cu = masses_of(:c, :u)
    cc = masses_of(:c, :c)
    w(m, L, mult, J, n) = fixed_channel_solution(p, m, FineStructureMultiplet(L, mult, J); solver = TOY).waves[n]
    so = spin_orbit_mixing_components(p, cu, "P", w(cu, "P", 1, 1, 1), w(cu, "P", 3, 1, 1))
    tm = tensor_mixing_components(p, cc, w(cc, "S", 3, 1, 1), w(cc, "D", 3, 1, 1), 1)
    tm_cu = tensor_mixing_components(p, cu, w(cu, "S", 3, 1, 1), w(cu, "D", 3, 1, 1), 1)
    return obj(
        "metadata" => nothing,
        "solver" => solver_dict(TOY),
        "notes" => obj(
            "grid" => "GIModel.radial_grid(ngrid,rmax): h=rmax/(ngrid+1), r_i=i*h, i=1..ngrid (Dirichlet at 0 and rmax)",
            "p2" => "GIModel.p2_operator(m,L,r,h): tridiagonal diag 2/h^2+L(L+1)/r^2, offdiag -1/h^2 (mass-independent)",
            "central" => "relativistic_hamiltonian: kinetic = V diag(sqrt(lam+m1^2)) V' + (m2); potential = A G~ A + S~ with A = V diag(sqrt(1+lam/(E1 E2))) V'; H0 = kinetic + potential; lam = eigen(p2) clipped at 0",
            "fixed" => "GIModel._fixed_channel_matrices(solver,params,masses,multiplet,nothing,SpinTerms(),nlevels); total = central + contact + fine_structure",
            "mixing_elements" => "computed with the toy-grid fixed_channel_solution ground-state waves (n=1), FD MeshWave, outer-lobe-positive phase",
        ),
        "r" => r, "h" => h,
        "p2" => p2,
        "central" => central,
        "fixed" => fixed,
        "mixing_elements" => obj(
            "cu_1P1_3P1_spin_orbit" => obj((string(k) => getproperty(so, k) for k in keys(so))...),
            "cc_3S1_3D1_tensor" => obj((string(k) => getproperty(tm, k) for k in keys(tm))...),
            "cu_3S1_3D1_tensor" => obj((string(k) => getproperty(tm_cu, k) for k in keys(tm_cu))...),
        ),
    )
end

# =============================================================================
# d. channels.json
# =============================================================================

function gen_channels()
    p = PARAMS
    solvers = [("fd_default", FiniteDifferenceSolver()), ("ho_default", OscillatorSolver()), ("fd_toy", TOY)]
    out = Pair{String,Any}[]
    for (sname, solver) in solvers
        systems = Pair{String,Any}[]
        for (sys, a, b) in (("cc", :c, :c), ("cu", :c, :u))
            m = masses_of(a, b)
            t0 = time()
            central = [obj("L" => L, "solution" => solution_dict(channel_solution(p, m, L; solver = solver)))
                       for L in 0:2]
            fixed = []
            for (L, mult, J) in MULTIPLETS
                sol = fixed_channel_solution(p, m, FineStructureMultiplet(L, mult, J); solver = solver)
                push!(fixed, obj("L_label" => L, "multiplicity" => mult, "J" => J,
                    "label" => mlabel(L, mult, J), "solution" => solution_dict(sol)))
            end
            println("    channels $sname $sys: ", round(time() - t0; digits = 1), " s")
            push!(systems, sys => obj("masses_GeV" => mpair(m), "central" => central, "fixed" => fixed))
        end
        push!(out, sname => obj("solver" => solver_dict(solver), "systems" => systems))
    end
    return obj(
        "metadata" => nothing,
        "wave_sample_r" => WAVE_R,
        "notes" => obj(
            "central" => "channel_solution(params,masses,L; solver) with default nlevels = solver.nlevels_per_channel (6)",
            "fixed" => "fixed_channel_solution(params,masses,FineStructureMultiplet(L,mult,J); solver, terms=SpinTerms()) nlevels 6",
            "waves_sampled" => "sample_wave(wave, wave_sample_r).u: MeshWave -> linear interpolation incl. u(0)=0 and u=0 beyond r_last+h; OscillatorWave -> expansion; both renormalized so sum(u^2)*0.25 = 1 on this grid",
            "waves_native" => "MeshWave: u on r_i = r_first + (i-1)h, physical normalization sum(u^2)h=1; OscillatorWave: coefficients (unit norm) of ho_reduced_radial(n,L,beta,r)",
            "phase" => "fix_outer_phase: last sample with |u| > 0.2 max|u| is made positive (HO: evaluated on 2049 points rho in [0, sqrt(4(N-1)+2L+3)+6], r=rho/beta)",
            "convergence" => "HO only: OscillatorConvergence certificate (beta chosen minimizing the highest requested level; nbasis 24:8:80 until two successive refinements change all requested levels by <= 1e-4 GeV)",
        ),
        "solvers" => out,
    )
end

# =============================================================================
# e. spectra
# =============================================================================

const SYSTEMS = [("cc", :c, :c), ("bb", :b, :b), ("cu", :c, :u), ("uc", :u, :c),
    ("cs", :c, :s), ("us", :u, :s), ("qq", :q, :q), ("ss", :s, :s),
    ("bu", :b, :u), ("bs", :b, :s), ("bc", :b, :c)]

basis_dict(b::BasisState) = obj("label" => b.label, "n" => b.n, "L_label" => b.L_label,
    "multiplicity" => b.multiplicity, "J" => b.J,
    "flavors" => isnothing(b.flavors) ? nothing : [string(f) for f in b.flavors])

mixing_dict(mx::StateMixing) = obj(
    "mechanism" => mx.mechanism,
    "block_label" => mx.block_label,
    "partner_labels" => collect(mx.partner_labels),
    "components" => collect(mx.components),
    "offdiag_GeV" => mx.offdiag_GeV,
    "partner_masses_GeV" => collect(mx.partner_masses_GeV),
    "eigenstate" => mx.eigenstate,
    "unmixed_GeV" => mx.unmixed_GeV,
    "block_matrix_GeV" => mx.result.block.matrix,
    "block_vectors" => mx.result.vectors,
)

function state_dict(spec, s::MixedState; with_convergence = false)
    c = s.corrected
    comps = physical_components(spec, s)
    conv = with_convergence ? [obj("label" => x.basis.label,
        "certificate" => convergence_dict(x.certificate)) for x in convergence(spec, s)] : nothing
    return obj(
        "label" => s.label,
        "basis" => basis_dict(s.basis),
        "mass_GeV" => s.mass_GeV,
        "corrected_mass_GeV" => c.mass_GeV,
        "central_GeV" => c.central_GeV,
        "contact_shift_GeV" => c.contact_shift_GeV,
        "spin_orbit_vector_shift_GeV" => c.spin_orbit_vector_shift_GeV,
        "spin_orbit_thomas_shift_GeV" => c.spin_orbit_thomas_shift_GeV,
        "spin_orbit_shift_GeV" => c.spin_orbit_shift_GeV,
        "tensor_shift_GeV" => c.tensor_shift_GeV,
        "fine_structure_shift_GeV" => c.fine_structure_shift_GeV,
        "fine_structure_mass_convention" => s.fine_structure_mass_convention,
        "corrected_fine_structure_mass_convention" => c.fine_structure_mass_convention,
        "mixings" => [mixing_dict(mx) for mx in s.mixings],
        "physical_components" => [obj("label" => x.basis.label, "basis" => basis_dict(x.basis),
            "coefficient" => x.coefficient) for x in comps],
        "precursor_wave_sampled" => sample_wave(radial_wave(spec, c), WAVE_R).u,
        "convergence" => conv,
    )
end

function spectrum_dict(spec; with_convergence = false)
    return [state_dict(spec, s; with_convergence) for s in spec.states]
end

function sector_convergence(spec)
    out = []
    for (key, sol) in spec.computation.channel_cache
        push!(out, obj("L_label" => key.L_label, "multiplicity" => key.multiplicity, "J" => key.J,
            "eigenvalues_GeV" => sol.eigenvalues_GeV, "certificate" => convergence_dict(sol.convergence)))
    end
    sort!(out; by = x -> (x[1][2], x[2][2], x[3][2]))
    return out
end

function gen_spectra(solver; with_convergence = false)
    p = PARAMS
    levels = spectrum_levels(2)
    systems = Pair{String,Any}[]
    for (sys, a, b) in SYSTEMS
        meson = Meson(MQ, a, b)
        t0 = time()
        entry = try
            spec = compute_spectrum(p, meson; levels = levels, solver = solver)
            obj("flavors" => [string(meson.flavor1), string(meson.flavor2)],
                "masses_GeV" => mpair(meson.constituent_masses),
                "is_equal_flavor" => is_equal_flavor(meson),
                "error" => nothing,
                "states" => spectrum_dict(spec; with_convergence),
                "sector_solutions" => with_convergence ? sector_convergence(spec) : nothing,
                "seconds" => time() - t0)
        catch err
            @warn "spectrum failed" sys err
            obj("flavors" => [string(meson.flavor1), string(meson.flavor2)],
                "masses_GeV" => mpair(meson.constituent_masses),
                "error" => sprint(showerror, err), "states" => [], "seconds" => time() - t0)
        end
        println("    spectrum $(nameof(typeof(solver))) $sys: ", round(time() - t0; digits = 1), " s")
        push!(systems, sys => entry)
    end
    return obj(
        "metadata" => nothing,
        "solver" => solver_dict(solver),
        "terms" => terms_dict(SpinTerms()),
        "levels" => [basis_dict(l) for l in levels],
        "wave_sample_r" => WAVE_R,
        "notes" => obj(
            "call" => "compute_spectrum(params, Meson(mq, flavor1, flavor2); levels=spectrum_levels(2), solver) = add_intra_meson_mixing(fixed_spectrum(...))",
            "flavors" => "light quark symbol is :q (u/d average, 0.220 GeV); :u gives the same mass. 'uc' is Meson(mq,:u,:c), i.e. m1=m_u, m2=m_c (swapped order)",
            "state.mass_GeV" => "final (mixed) mass; corrected_mass_GeV is the fixed-sector eigenvalue before mixing; central+contact+fine_structure = corrected_mass",
            "mixings" => "one record per mixing step (StateMixing); components = column `eigenstate` of block_vectors in basis partner_labels; offdiag_GeV = max |offdiag| of the block; partner_masses_GeV = ascending block eigenvalues",
            "physical_components" => "flattened signed composition (physical_components)",
            "precursor_wave_sampled" => "sample_wave(radial_wave(spec, state.corrected), wave_sample_r).u (pre-mixing fixed-sector wave)",
            "convergence" => "HO only: convergence(spec,state) per physical component",
            "sector_solutions" => "HO only: every cached fixed (L,S,J) sector: eigenvalues and OscillatorConvergence (beta, nbasis)",
        ),
        "systems" => systems,
    )
end

# =============================================================================
# f. variations.json
# =============================================================================

function gen_variations()
    p = PARAMS
    solver = FiniteDifferenceSolver()
    levels = spectrum_levels(1)
    P(; kw...) = GIParameters(p; kw...)
    pv = [
        ("baseline", p, nothing, nothing),
        ("b=0.20", P(potential = ConfinementPotential(p.potential; b = 0.20)), nothing, nothing),
        ("c=-0.3", P(potential = ConfinementPotential(p.potential; c = -0.3)), nothing, nothing),
        ("sigma0=1.6", P(smearing = RelativisticSmearing(p.smearing; sigma0 = 1.6)), nothing, nothing),
        ("s=1.7", P(smearing = RelativisticSmearing(p.smearing; s = 1.7)), nothing, nothing),
        ("eps_c=-0.1", P(factors = RelativisticFactors(p.factors; epsilon_c = -0.1)), nothing, nothing),
        ("eps_t=0.05", P(factors = RelativisticFactors(p.factors; epsilon_t = 0.05)), nothing, nothing),
        ("eps_so_v=-0.02", P(factors = RelativisticFactors(p.factors; epsilon_so_vector = -0.02)), nothing, nothing),
        ("eps_so_s=0.08", P(factors = RelativisticFactors(p.factors; epsilon_so_scalar = 0.08)), nothing, nothing),
        ("m_c=1.55", p, Dict(:c => 1.55), nothing),
        ("m_s=0.45", p, Dict(:s => 0.45), nothing),
        ("terms:contact_hyperfine=false", p, nothing, SpinTerms(contact_hyperfine = false)),
        ("terms:fine_structure=false", p, nothing, SpinTerms(fine_structure = false)),
        ("terms:same_j_spin_orbit=false", p, nothing, SpinTerms(same_j_spin_orbit = false)),
        ("terms:tensor=false", p, nothing, SpinTerms(tensor = false)),
    ]
    runs = []
    for (name, params, mover, terms) in pv, (sys, a, b) in (("cc", :c, :c), ("cs", :c, :s))
        mass(f) = !isnothing(mover) && haskey(mover, f) ? mover[f] : Meson(MQ, f, f).constituent_masses.m1_GeV
        meson = Meson(a, b, ConstituentMasses(mass(a), mass(b)))
        t = isnothing(terms) ? SpinTerms() : terms
        spec = compute_spectrum(params, meson; levels = levels, solver = solver, terms = t)
        push!(runs, obj("name" => name, "system" => sys, "levels_spec" => "spectrum_levels(1)",
            "parameters" => params_dict(params), "masses_GeV" => mpair(meson.constituent_masses),
            "terms" => terms_dict(t), "states" => spectrum_dict(spec)))
    end
    levels4 = spectrum_levels(1; L_labels = ("S", "P", "D", "F"))
    meson = Meson(MQ, :c, :c)
    spec = compute_spectrum(p, meson; levels = levels4, solver = solver)
    push!(runs, obj("name" => "L_labels=SPDF", "system" => "cc",
        "levels_spec" => "spectrum_levels(1; L_labels=(\"S\",\"P\",\"D\",\"F\"))",
        "parameters" => params_dict(p), "masses_GeV" => mpair(meson.constituent_masses),
        "terms" => terms_dict(SpinTerms()), "states" => spectrum_dict(spec)))
    return obj(
        "metadata" => nothing,
        "solver" => solver_dict(solver),
        "wave_sample_r" => WAVE_R,
        "notes" => obj(
            "runs" => "one parameter changed at a time relative to the default parameter file; parameters/masses/terms store the full set used",
            "masses" => "m_c/m_s variations use Meson(flavor1, flavor2, ConstituentMasses(...)); the other masses are the defaults",
            "levels" => "spectrum_levels(1) = 1S,1P,1D multiplets unless levels_spec says otherwise",
        ),
        "runs" => runs,
    )
end

# =============================================================================
# Spot checks against the documented GIModel.jl numbers.
# =============================================================================

function gen_spot_checks()
    p = PARAMS
    checks = []
    add(name, documented, computed, tol) = push!(checks, obj("name" => name,
        "documented" => documented, "computed" => computed, "abs_diff" => abs(documented - computed),
        "tolerance" => tol, "pass" => abs(documented - computed) <= tol))
    cc = Meson(MQ, :c, :c)
    spec = compute_spectrum(p, cc; levels = spectrum_levels(2))
    add("cc FD 1^1S_0 mass (4 d.p.)", 2.9667, spectrum_state(spec, "1^1S_0").mass_GeV, 5e-5)
    add("cc FD 1^3S_1 mass (4 d.p.)", 3.0910, spectrum_state(spec, "1^3S_1").mass_GeV, 5e-5)
    add("cc FD 1^3P_0 mass (4 d.p.)", 3.4428, spectrum_state(spec, "1^3P_0").mass_GeV, 5e-5)
    add("cc FD chi_c1 1^3P_1 mass", 3.5081502109538754, spectrum_state(spec, "1^3P_1").mass_GeV, 1e-12)
    add("cc FD chi_c2 1^3P_2 mass", 3.548136774754569, spectrum_state(spec, "1^3P_2").mass_GeV, 1e-12)
    cu = Meson(MQ, :c, :u)
    sp = compute_spectrum(p, cu; levels = spectrum_levels(1; L_labels = ("P",)))
    d1 = spectrum_state(sp, "1^1P_1")
    mx = only(d1.mixings)
    add("cu FD 1P mixed low (4 d.p.)", 2.4556, d1.mass_GeV, 5e-5)
    add("cu FD 1P mixed high (4 d.p.)", 2.4652, spectrum_state(sp, "1^3P_1").mass_GeV, 5e-5)
    add("cu FD 1P offdiag", 0.003742140815267442, mx.offdiag_GeV, 1e-12)
    add("cu FD 1P partner mass low", 2.4556181687426726, mx.partner_masses_GeV[1], 1e-12)
    add("cu FD 1P partner mass high", 2.465176803815588, mx.partner_masses_GeV[2], 1e-12)
    add("cu FD 1P components[1]", 0.9005661482910976, mx.components[1], 1e-10)
    add("cu FD 1P components[2]", -0.43471900413041176, mx.components[2], 1e-10)
    sol = channel_solution(p, cc.constituent_masses, 0; solver = OscillatorSolver())
    for (i, v) in enumerate((3.0645236910656135, 3.666192134948986, 4.090957384190697))
        add("cc HO channel_solution L=0 eigenvalue $i", v, sol.eigenvalues_GeV[i], 1e-10)
    end
    return obj("metadata" => nothing, "checks" => checks)
end

# =============================================================================
# Driver
# =============================================================================

const TARGETS = [
    ("primitives", "primitives.json", "Layer a: running coupling, smearing widths, smeared kernels and angular factors", gen_primitives),
    ("ho_matrices", "ho_matrices.json", "Layer b: harmonic-oscillator basis matrices (r2, p2, DVR position/momentum operators, central and spin matrices)", gen_ho_matrices),
    ("fd_matrices", "fd_matrices.json", "Layer c: finite-difference toy-grid (ngrid=20, rmax=10) operators and full fixed-channel Hamiltonians", gen_fd_matrices),
    ("channels", "channels.json", "Layer d: channel_solution / fixed_channel_solution eigenvalues and waves for cc and cu (FD default, HO default, FD toy)", gen_channels),
    ("spectra_fd", "spectra_fd.json", "Layer e: compute_spectrum with FiniteDifferenceSolver() and spectrum_levels(2)", () -> gen_spectra(FiniteDifferenceSolver())),
    ("spectra_ho", "spectra_ho.json", "Layer e: compute_spectrum with OscillatorSolver() and spectrum_levels(2), with convergence certificates", () -> gen_spectra(OscillatorSolver(); with_convergence = true)),
    ("variations", "variations.json", "Layer f: FD spectra (spectrum_levels(1)) for cc and cs under one-at-a-time parameter/mass/SpinTerms variations", gen_variations),
    ("spot_checks", "spot_checks.json", "Documented GIModel.jl numbers compared with this run", gen_spot_checks),
]

function main(args)
    selected = isempty(args) ? [t[1] for t in TARGETS] : args
    unknown = setdiff(selected, [t[1] for t in TARGETS])
    isempty(unknown) || error("unknown targets: $(unknown)")
    println("GIModel.jl at ", GIDIR, " commit ", git_commit(GIDIR))
    println("Writing to ", OUTDIR)
    total0 = time()
    for (key, file, desc, fn) in TARGETS
        key in selected || continue
        println("[", key, "]")
        t0 = time()
        data = fn()
        elapsed = time() - t0
        data[1] = "metadata" => metadata(desc; elapsed = elapsed)
        write_json(file, data)
        println("  ", round(elapsed; digits = 1), " s")
        if key == "spot_checks"
            for c in data[2][2]
                d = Dict(c)
                println("    ", d["pass"] ? "PASS " : "FAIL ", d["name"], ": documented=", d["documented"],
                    " computed=", d["computed"], " |diff|=", d["abs_diff"])
            end
        end
    end
    println("total ", round(time() - total0; digits = 1), " s")
end

main(ARGS)
