# GIModel.jl reference-data harness

This directory generates the JSON reference files in `tests/gimodel/reference/`.
The Python port (`gimodel`) is tested against these files. The data comes from
the original Julia package
[GIModel.jl](https://github.com/mmikhasenko/GIModel.jl) (branch `main`, v0.4.0).

## Requirements

- Julia ≥ 1.11. The committed reference files were generated with Julia 1.13.1,
  installed user-locally through `juliaup`:
  `curl -fsSL https://install.julialang.org | sh -s -- --yes`.
- A local clone of GIModel.jl:

  ```sh
  git clone https://github.com/mmikhasenko/GIModel.jl /some/path/GIModel.jl
  ```

## Regenerating

```sh
cd tools/julia_reference
GIMODEL_JL_PATH=/some/path/GIModel.jl julia generate_reference.jl            # all files
GIMODEL_JL_PATH=/some/path/GIModel.jl julia generate_reference.jl primitives  # a subset
```

The script activates this directory as its Julia project. When
`GIMODEL_JL_PATH` is set, it runs `Pkg.develop(path=$GIMODEL_JL_PATH)` and then
`Pkg.instantiate()`, so no separate setup step is needed. Leave
`GIMODEL_JL_PATH` unset to reuse the path recorded in the local
`Manifest.toml`. That file is git-ignored because it holds a machine-specific
path.

Available targets, in layer order:

| target        | file                | content |
|---------------|---------------------|---------|
| `primitives`  | `primitives.json`   | α_s(r), α_s(Q), σ(m1,m2), the smeared G̃, S̃, G̃′, G̃″, S̃′, tensor and contact kernels for 10 mass pairs, and the angular factors |
| `ho_matrices` | `ho_matrices.json`  | HO r², p², DVR position operators, momentum operators, central H, and the contact and fine-structure matrices |
| `fd_matrices` | `fd_matrices.json`  | toy FD grid (ngrid=20, rmax=10): p², kinetic, potential, H0, and every term of the fixed-channel H |
| `channels`    | `channels.json`     | `channel_solution` / `fixed_channel_solution` eigenvalues and waves (FD default, HO default, FD toy) for cc̄ and cū |
| `spectra_fd`  | `spectra_fd.json`   | `compute_spectrum(...; levels=spectrum_levels(2))` with the FD solver for 11 systems |
| `spectra_ho`  | `spectra_ho.json`   | the same with `OscillatorSolver()`, plus convergence certificates (β, nbasis) |
| `variations`  | `variations.json`   | FD `spectrum_levels(1)` for cc̄ and cs̄ with one parameter, mass or `SpinTerms` switch changed at a time, plus cc̄ S/P/D/F |
| `spot_checks` | `spot_checks.json`  | numbers documented by GIModel.jl compared with this run |

`GIMODEL_REFERENCE_OUT` overrides the output directory (default `../../tests/gimodel/reference`).

## Format

- Every file has a top-level `metadata` object with the Julia version, the
  GIModel.jl version and commit, a dirty flag, the UTC date, the BLAS
  configuration, the full parameter set and the quark masses.
- Floats are written as Julia's shortest round-trip representation. Python's
  `float()` parses each one back to the identical IEEE-754 double, so no
  precision is lost compared with `%.17g`. Non-finite values are written as
  `null`.
- Matrices are row-major lists of rows.
- Units: GeV for masses and energies, GeV⁻¹ for r.
- Each file has a `notes` object that names the GIModel.jl function behind
  each quantity. Unexported functions are written as `GIModel.<name>`.

The JSON writer is a small built-in writer, not JSON3. This keeps key order
and float formatting deterministic, and avoids another dependency.

## Expected agreement

Eigenvalues depend on LAPACK/BLAS rounding: they agree with the documented
values to ~1e-14 GeV. Eigenvectors are compared only after the package's own
phase fixing (`fix_outer_phase`, `_phase_fix_state_columns!`). Adaptive
pieces (the HO β search, nbasis refinement, and doubling of the quadrature
`nq`) are discrete choices. An implementation that makes the same choices
reproduces the numbers to ~1e-12. One that does not still agrees within the
convergence tolerances (0.1 MeV for HO energies).
