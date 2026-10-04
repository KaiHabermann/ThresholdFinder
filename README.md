# ThresholdFinder

The distribution ships two packages:

- `threshold_finder` (CLIs `threshold-finder`, `qn-options`) finds hadronic thresholds; see below.
- `gimodel` (CLI `gi-spectrum`) computes Godfrey–Isgur quark-model meson spectra; see [Quark-model spectrum](#quark-model-spectrum-gimodel-gi-spectrum).

Finds n-body hadronic thresholds compatible with given J^P quantum numbers. Given a mass range and a target J^P, it scans all combinations of PDG hadrons whose combined mass falls in that range and checks whether they can couple — via some total orbital angular momentum L — to produce the desired quantum numbers. The default is two-body; use `--n-body` (or `n_body=` in the API) to search three-body or higher final states.

## Requirements

- Python >= 3.11
- [`particle`](https://github.com/scikit-hep/particle) >= 0.24
- `numpy` and `scipy` (for the quark model, `gimodel`)
- [`pdg`](https://pdgapi.lbl.gov/doc/) >= 2026.0, the official PDG Python API (for `gi-spectrum --match`)

```bash
pip install thresholds
```

## Usage

### Command line

```
threshold-finder mass_min mass_max [J P] [options]
```

**Positional arguments:**

| Argument   | Description                                      |
|------------|--------------------------------------------------|
| `mass_min` | Lower bound of the threshold search range (MeV)  |
| `mass_max` | Upper bound of the threshold search range (MeV)  |
| `J`        | Target total angular momentum (integer or half-integer, e.g. `1`, `0.5`). Optional if `--particles` is given. |
| `P`        | Target parity: `+1` or `-1`. Optional if `--particles` is given. |

`J` and `P` must either both be given or both be omitted. When omitted, `--particles` is required and the 3 lowest J^P combinations the pair can produce are used automatically.

**Optional arguments:**

| Flag                   | Default  | Description |
|------------------------|----------|-------------|
| `--max-L L`            | auto     | Maximum orbital angular momentum to consider. Without this flag, L is capped automatically at J + J₁ + J₂ + 4 for each pair. |
| `--charge CHARGE`      | `0`      | Required total electric charge of the two-particle system. |
| `--status S [S ...]`   | `0`      | PDG status codes to include: `0` = well-established, `1` = evidence but unconfirmed, `2` = omitted from summary tables. |
| `--unique-pairs`       | off      | Show each particle combination only once (keeping the lowest L), instead of one entry per valid L. |
| `--n-body N`           | `2`      | Number of particles in the final state (must be >= 2). Default is 2 (two-body). Use 3 for three-body, etc. |

**Flavor conservation flags** (all optional, independent):

| Flag                     | Description |
|--------------------------|-------------|
| `--u N`                  | Required net u-quark number of the pair (#u − #ū) |
| `--d N`                  | Required net d-quark number of the pair (#d − #d̄) |
| `--s N`                  | Required net s-quark number of the pair (#s − #s̄) |
| `--c N`                  | Required net charm of the pair (#c − #c̄) |
| `--b N`                  | Required net bottomness of the pair (#b − #b̄) |
| `--particles P1 P2`      | Derive all flavor numbers automatically from two PDG particle names |

Only the flags you provide are enforced. Omit a flag to leave that flavor unconstrained. Pairs involving particles with undefined quark content (e.g. η, ω — mixed states like uū+dd̄) are excluded when any flavor flag is set.

`--particles` does two things:
1. **Derives flavor conservation** automatically — no need to compute net quark numbers by hand. Explicit `--u/--d/...` flags override the derived values.
2. **Checks feasibility** — if J and P are given, the tool first verifies that the specified pair can actually produce that J^P at some L. If not, a warning is printed and no results are shown for that J^P.
3. **Auto-detects J^P** — if J and P are omitted, the 3 lowest J^P combinations the pair can produce (ordered by minimum L) are determined and each is searched separately.

### Examples

Find all 1⁻ channels with threshold between 250 and 300 MeV (the ρ region):

```
$ python3 -m threshold_finder.cli 250 300 1 -1

Thresholds for J^P = 1^-  in [250.0, 300.0] MeV  (max L = ∞)
Found 1 combination(s):
  pi+ + pi-  threshold=279.1 MeV  L=1  J^P=1^-
```

Find 1⁻ channels near 1 GeV with full flavor conservation (all net quark numbers = 0):

```
$ python3 -m threshold_finder.cli 900 1100 1 -1 --u 0 --d 0 --s 0 --c 0 --b 0 --unique-pairs

Thresholds for J^P = 1^-  in [900.0, 1100.0] MeV  (max L = ∞)  flavor: u=+0, d=+0, s=+0, c=+0, b=+0
Found 4 combination(s):
  pi+ + rho(770)-  threshold=914.7 MeV  L=1  J^P=1^-
  pi- + rho(770)+  threshold=914.7 MeV  L=1  J^P=1^-
  K+ + K-  threshold=987.4 MeV  L=1  J^P=1^-
  K0 + K~0  threshold=995.2 MeV  L=1  J^P=1^-
```

Constrain only strangeness (leave u/d free) to find kaonic channels:

```
$ python3 -m threshold_finder.cli 600 700 0 -1 --s -1 --unique-pairs

Thresholds for J^P = 0^-  in [600.0, 700.0] MeV  (max L = ∞)  flavor: s=-1
Found 4 combination(s):
  pi0 + K(L)0  threshold=632.6 MeV  L=0  J^P=0^-
  ...
```

Find open-charm 1⁻ thresholds near ψ(3770) with net zero flavor:

```
$ python3 -m threshold_finder.cli 3700 3900 1 -1 --u 0 --d 0 --s 0 --c 0 --b 0 --unique-pairs

...
  D0 + D~0  threshold=3729.7 MeV  L=1  J^P=1^-
  D+ + D-   threshold=3739.3 MeV  L=1  J^P=1^-
  ...
  D0 + D*(2007)~0  threshold=3871.7 MeV  L=1  J^P=1^-
  ...
```

Omit J and P to auto-detect the 3 lowest J^P combinations D0 + Lambda can produce:

```
$ threshold-finder 2800 3000 --particles 'D0' 'Lambda' --unique-pairs

Flavor conservation derived from 'D0' + 'Lambda':
  u = +0
  d = +1
  s = +1
  c = +1

No J^P given — using the 3 lowest combinations 'D0' + 'Lambda' can produce:
  J^P = 1/2^-  (lowest at L=0)
  J^P = 1/2^+  (lowest at L=1)
  J^P = 3/2^+  (lowest at L=1)

Thresholds for J^P = 1/2^-  in [2800.0, 3000.0] MeV  ...
...
Thresholds for J^P = 1/2^+  in [2800.0, 3000.0] MeV  ...
...
Thresholds for J^P = 3/2^+  in [2800.0, 3000.0] MeV  ...
...
```

If the requested J^P cannot be produced by the given pair at any L, a warning is shown and that J^P is skipped:

```
$ threshold-finder 2800 3000 0.5 +1 --particles 'pi+' 'pi-'

WARNING: 'pi+' + 'pi-' cannot produce J^P = 1/2^+ at any L
```

Derive flavor numbers from reference particles (D0 + Lambda defines the channel):

```
$ threshold-finder 2800 3000 0.5 -1 --particles 'D0' 'Lambda' --unique-pairs

Flavor conservation derived from 'D0' + 'Lambda':
  u = +0
  d = +1
  s = +1
  c = +1

Thresholds for J^P = 1/2^-  in [2800.0, 3000.0] MeV  (max L = ∞)  flavor: u=+0, d=+1, s=+1, c=+1
Found 5 combination(s):
  pi- + Xi(c)(2790)+  threshold=2931.5 MeV  L=1  J^P=1/2^-
  K- + Sigma(c)(2455)+  threshold=2946.3 MeV  L=0  J^P=1/2^-
  ...
```

If the reference threshold is outside the search range, a warning is printed but the search still runs:

```
$ threshold-finder 2500 2800 0.5 -1 --particles 'D0' 'Lambda' --unique-pairs

WARNING: threshold of D0 + Lambda = 2980.5 MeV is above mass_max = 2800.0 MeV
Flavor conservation derived from 'D0' + 'Lambda':
  ...
```

If a particle has ambiguous quark content (mixed state), the tool reports what could be determined and prints a ready-to-edit command with `???` placeholders for the unknown flavors:

```
$ threshold-finder 2800 3000 0.5 -1 --particles 'eta' 'Lambda'

ERROR: Quark content is ambiguous (mixed/superposition state) for:
  eta  (PDG quarks string: 'x(uU+dD)+y(sS)')
Determined from ['Lambda']: d=+1, s=+1, u=+1

Set the remaining flavor flags manually. Example command:
  threshold-finder 2800.0 3000.0 0.5 -1 --u 1 --d 1 --s 1 --c ??? --b ???
```

If a particle name is not found, 5 suggestions are printed as ready-to-run commands:

```
$ threshold-finder 2800 3000 0.5 -1 --particles 'D0' 'Lmabda'

ERROR: Unknown particle 'Lmabda'
Did you mean one of these?
  threshold-finder 2800.0 3000.0 0.5 -1 --particles 'D0' 'Lambda'
  threshold-finder 2800.0 3000.0 0.5 -1 --particles 'D0' 'Lambda~'
  threshold-finder 2800.0 3000.0 0.5 -1 --particles 'D0' 'Lambda(c)+'
  threshold-finder 2800.0 3000.0 0.5 -1 --particles 'D0' 'Lambda(b)0'
  threshold-finder 2800.0 3000.0 0.5 -1 --particles 'D0' 'Lambda(1520)'
```

Find three-body 0⁻ thresholds near 420 MeV (the 3π region):

```
$ threshold-finder 400 450 0 -1 --n-body 3 --max-L 0 --unique-pairs

Thresholds for J^P = 0^-  in [400.0, 450.0] MeV  (max L = 0)  (3-body)
Found 2 combination(s):
  pi- + pi0 + pi+  threshold=414.1 MeV  L=0  J^P=0^-
  pi0 + pi0 + pi0  threshold=404.9 MeV  L=0  J^P=0^-
```

Find 2⁺ channels with threshold 500–700 MeV, restricting to L ≤ 2:

```
$ python3 -m threshold_finder.cli 500 700 2 +1 --max-L 2 --unique-pairs

Thresholds for J^P = 2^+  in [500.0, 700.0] MeV  (max L = 2)
Found 7 combination(s):
  pi0 + K(L)0  threshold=632.6 MeV  L=2  J^P=2^+
  ...
```

### Python API

```python
from threshold_finder import ThresholdFinder, FlavorFilter

finder = ThresholdFinder(
    mass_min=900,
    mass_max=1100,
    J_target=1,
    P_target=-1,
    n_body=2,                                # 2 = two-body (default); set 3 for three-body, etc.
    max_L=None,                              # None = automatic
    total_charge=0.0,
    flavor_filter=FlavorFilter(u=0, d=0, s=0, c=0, b=0),  # all net quark numbers = 0
    status_filter=(0,),                      # established particles only
)
result = finder.run()

print(result)  # formatted summary

for c in result.combinations:
    print(" + ".join(c.particles), "  L =", c.L, "  threshold =", c.threshold, "MeV")
```

Three-body search:

```python
finder = ThresholdFinder(
    mass_min=400,
    mass_max=450,
    J_target=0,
    P_target=-1,
    n_body=3,
    max_L=0,
)
result = finder.run()
for c in result.combinations:
    print(" + ".join(c.particles), "  threshold =", c.threshold, "MeV")
```

Constrain only specific flavors by omitting the rest:

```python
# Only require net charm = 0; u, d, s, b are unconstrained
flavor_filter=FlavorFilter(c=0)

# Only require net strangeness = -1
flavor_filter=FlavorFilter(s=-1)
```

`ThresholdResult` has the fields `J_target`, `P_target`, `mass_min`, `mass_max`, `max_L`, `n_body`, `flavor_filter`, and `combinations` (a list of `CombinationResult`).

Each `CombinationResult` contains:

| Field        | Type              | Description                                    |
|--------------|-------------------|------------------------------------------------|
| `particles`  | `tuple[str, ...]` | PDG names of all particles in the final state  |
| `masses`     | `tuple[float, ...]` | Masses of all particles (MeV)                |
| `charges`    | `tuple[float, ...]` | Charges of all particles                     |
| `spins`      | `tuple[float, ...]` | J values of all particles                    |
| `parities`   | `tuple[int, ...]`   | Parities of all particles                    |
| `threshold`  | `float`           | Sum of all particle masses (MeV)               |
| `L`          | `int`             | Total orbital angular momentum                 |
| `J_total`    | `float`           | Total angular momentum (= J_target)            |
| `P_total`    | `int`             | Total parity (= P_target)                      |

For two-body results the legacy per-particle properties (`particle1`, `particle2`, `mass1`, `mass2`, `charge1`, `charge2`, `J1`, `J2`, `P1`, `P2`, `identical`) are still available as convenience aliases.

## Physics

The tool checks whether a set of n particles (each with Jᵢ^Pᵢ) in a state of total orbital angular momentum L can produce the target J^P.

**Parity:**
```
P_total = P₁ · P₂ · … · Pₙ · (-1)^L
```

**Angular momentum:** J_total must be reachable by sequentially coupling all particle spins J₁ ⊗ J₂ ⊗ … ⊗ Jₙ via the triangle rule, then adding L.

**Identical bosons (two-body only):** For two identical bosons (e.g. π⁰π⁰), the spatial wave function must be symmetric under exchange, which requires L to be even. For n > 2 this constraint is not enforced (exact symmetrisation of n-body wave functions depends on the full Bose/Fermi statistics of all permutations).

**Flavor conservation:** Net quark numbers are computed as #quark − #antiquark for each flavor (u, d, s, c, b). They are additive over all n particles in the final state. Setting a flavor to 0 requires the combination to have no net quark content in that flavor. Particles with mixed or superposition quark content (η, ω, φ, π⁰, …) have undefined quark numbers and are excluded from any result when a flavor constraint is active.

Particle data (masses, J, P, charge, quark content) are read from the PDG via the [`particle`](https://github.com/scikit-hep/particle) package. Only hadrons with known mass, J, and P are considered.

## Quark-model spectrum (`gimodel`, `gi-spectrum`)

A Python port of the core of [GIModel.jl](https://github.com/mmikhasenko/GIModel.jl)
by M. Mikhasenko: the Godfrey–Isgur relativized quark model for meson spectra
(S. Godfrey and N. Isgur, *Phys. Rev. D* **32**, 189 (1985)).

The port covers the spectrum core: the Table II / Appendix A parameters (bundled
TOML), the running coupling, the closed-form smeared potentials, the
finite-difference relativized Hamiltonian, the contact hyperfine, spin-orbit
(vector and Thomas) and tensor operators, fixed-(L,S,J) channel solves,
intra-meson mixing (antisymmetric spin-orbit and tensor), and radial-wave
utilities. It uses the same numerics, conventions and defaults as the Julia
package. Both radial solvers are ported: the default finite-difference grid
(`FiniteDifferenceSolver`) and the paper's harmonic-oscillator basis
(`OscillatorSolver`). Masses agree with GIModel.jl to about 1e-12 GeV on the
grid and about 1e-14 GeV in the oscillator basis.

It does not include flavour annihilation mixing or the transition operators.

### Python API

```python
from gimodel import (Meson, compute_spectrum, load_parameters_and_quark_masses,
                     physical_components, radial_wave, spectrum_levels, spectrum_state,
                     wave_mean_squares)

params, mq = load_parameters_and_quark_masses()          # bundled GI parameter set
spec = compute_spectrum(params, Meson.from_table(mq, "c", "c"), levels=spectrum_levels(2))
print(spec)                                              # table of contributions (GeV)

chi_c1 = spectrum_state(spec, "1^3P_1")
chi_c1.mass_GeV, chi_c1.central_GeV, chi_c1.spin_orbit_shift_GeV, chi_c1.tensor_shift_GeV

[(c.basis.label, c.coefficient) for c in physical_components(spec, "1^3S_1")]  # J/psi
wave_mean_squares(radial_wave(spec, "1^1S_0"), 0)       # <r^2> [GeV^-2], <p^2> [GeV^2]
```

To vary parameters, use `dataclasses.replace`, for example
`replace(params, potential=replace(params.potential, b=0.19))`. To change the
grid, use `FiniteDifferenceSolver(ngrid=900, rmax=24.0)`. To switch terms off,
use `SpinTerms(tensor=False)`.

Units are GeV and GeV⁻¹ throughout.

### Command line: `gi-spectrum`

`gi-spectrum` lists the model's meson states in a mass window. It follows the
conventions of `threshold-finder` from the same repository: masses are in MeV,
`J P` are optional positional filters, and it has the same flavour flags,
`--particles` handling and unknown-name suggestions.

```
gi-spectrum mass_min mass_max [J P] (--quarks Q QBAR | --u/--d/--s/--c/--b N | --particles P [P ...]) [options]
```

**Flavour content.** This is required, and you give it in exactly one way:

| Input | Meaning |
|-------|---------|
| `--quarks Q QBAR` | Quark and antiquark, from `u d q s c b`, e.g. `--quarks c u` (c ū) or `--quarks c c` (charmonium). This is the only way to request hidden flavour. |
| `--u/--d/--s/--c/--b N` | Net quark numbers, as in threshold-finder. They must add up to one quark (+1) and one antiquark (−1) of different flavours, e.g. `--c 1 --s -1`. A net-zero input is rejected with a `--quarks` suggestion. |
| `--particles P [P ...]` | The summed PDG quark content: `D0 pi+` gives c d̄ and `D(s)+` gives c s̄. A single self-conjugate particle uses its own content: `J/psi(1S)` gives c c̄ and `Upsilon(1S)` gives b b̄. Light superpositions (`pi0`, `rho(770)0`) give the light q q̄. Ambiguous content (`eta`, `eta'(958)`, `phi(1020)`, `K(S)0`), net-zero sums such as `D0 D~0`, and anything that isn't one quark plus one antiquark are errors that suggest `--quarks`. Flavour-neutral superpositions in a sum carry no net flavour. As in threshold-finder, explicit `--u/--d/...` flags override the derived values. |

`u` and `d` both map to the model's isospin-averaged light quark `q`. The quark
is flavour 1 and the antiquark flavour 2. This order only affects the sign of
mixing angles. For light hidden flavour (q q̄, s s̄) the output notes that
isoscalar annihilation mixing (η/η′, ω/φ) is not included.

**Options.**

| Flag | Default | Description |
|------|---------|-------------|
| `J P` | – | Filter by J^P. J is an integer (half-integer J is rejected), P is `+1` or `-1`. They must be given together. |
| `--C {+1,-1}` | – | Filter by C-parity. Only valid for self-conjugate systems. |
| `--solver {fd,ho}` | `fd` | Finite-difference or harmonic-oscillator solver. If the solver is unavailable, the CLI prints a clean error. |
| `--nmax N` | `3` | Radial levels per (L, S, J) sector |
| `--max-L L` | `3` | Highest orbital angular momentum (S P D F; at most 4) |
| `--params PATH` | bundled | Parameter TOML file |
| `--details` | off | Mass contributions (central, contact, spin-orbit, tensor, mixing) and mixing components |
| `--json` | off | Machine-readable output |
| `--match` | off | Pair each predicted state with PDG mesons (see [PDG matching](#pdg-matching---match)) |
| `--match-mass-tol MEV` | `50` | Mass tolerance in MeV |
| `--match-width-frac F` | `0.5` | Widen the tolerance to F·Γ_PDG for broad states |
| `--match-status S [S ...]` | `0` | PDG status codes to include, with the same codes and default as threshold-finder's `--status`. See [PDG matching](#pdg-matching---match) for how they map onto the PDG API. |
| `--match-include-uncertain` | off | Also use J/P values the PDG gives in parentheses or as alternatives, and flag those candidates |

P = (−1)^(L+1). C = (−1)^(L+S) is shown only for equal-flavour systems.
Unequal-flavour ¹L_L/³L_L states are labelled by their dominant component and
marked as mixtures. If the highest computed level of a relevant sector is still
below `mass_max`, a warning on stderr suggests a larger `--nmax`. Errors go to
stderr with a non-zero exit status.

#### Examples

```
$ gi-spectrum 2900 3200 --quarks c c
GI spectrum of c c̄  in [2900.0, 3200.0] MeV  (solver = fd, nmax = 3, L = SPDF)
Parameters: .../gimodel/data/parameters.provisional.toml
Found 2 state(s):
  1^1S_0  J^PC=0^-+  mass=2966.7 MeV
  1^3S_1  J^PC=1^--  mass=3091.0 MeV
```

```
$ gi-spectrum 2400 2600 1 +1 --particles 'D(s)+' --details
Reference particles: 'D(s)+'  ->  c s̄
GI spectrum of c s̄ for J^P = 1^+  in [2400.0, 2600.0] MeV  (solver = fd, nmax = 3, L = SPDF)
Parameters: .../gimodel/data/parameters.provisional.toml
Found 2 state(s):
  1^1P_1  J^P=1^+  mass=2547.1 MeV  (mixture: 59% 1^1P_1 + 41% 1^3P_1)
      contributions [MeV]: central=2565.0  contact=-14.7  spin-orbit=+0.0  tensor=+0.0  mixing=-3.2
      components: +0.7708 1^1P_1, -0.6366 1^3P_1, -0.0200 2^3P_1, +0.0166 2^1P_1
  1^3P_1  J^P=1^+  mass=2554.1 MeV  (mixture: 59% 1^3P_1 + 41% 1^1P_1)
      contributions [MeV]: central=2565.5  contact=+5.0  spin-orbit=-30.7  tensor=+11.7  mixing=+2.5
      components: +0.7708 1^3P_1, +0.6366 1^1P_1, -0.0197 2^1P_1, -0.0162 2^3P_1, +0.0015 3^1P_1, +0.0011 3^3P_1
```

```
$ gi-spectrum 3000 3700 --particles 'J/psi(1S)' --C -1
Reference particles: 'J/psi(1S)'  ->  c c̄
GI spectrum of c c̄ with C = -  in [3000.0, 3700.0] MeV  (solver = fd, nmax = 3, L = SPDF)
Parameters: .../gimodel/data/parameters.provisional.toml
Found 3 state(s):
  1^3S_1  J^PC=1^--  mass=3091.0 MeV
  1^1P_1  J^PC=1^+-  mass=3515.1 MeV
  2^3S_1  J^PC=1^--  mass=3678.8 MeV
```

```
$ gi-spectrum 1800 2100 --particles D0 pi+ --max-L 1
Reference particles: 'D0' + 'pi+'  ->  c d̄
GI spectrum of c d̄  in [1800.0, 2100.0] MeV  (solver = fd, nmax = 3, L = SP)
Parameters: .../gimodel/data/parameters.provisional.toml
Note: u and d map to the isospin-averaged light quark: model system c q̄
Found 2 state(s):
  1^1S_0  J^P=0^-  mass=1873.4 MeV
  1^3S_1  J^P=1^-  mass=2038.0 MeV
```

```
$ gi-spectrum 2900 3200 --c 0
ERROR: Net flavour from the flavour flags is zero (hidden flavour), which the --u/--d/--s/--c/--b numbers cannot specify.
Give the quark and antiquark explicitly with --quarks, e.g.:
  gi-spectrum 2900.0 3200.0 --quarks c c
```

`--json` prints the system, solver, parameter file, filters, notes, warnings
and, for each state, its label, n, L, S, J, P, C, mass in MeV, mixing
components and contributions.

#### PDG matching (`--match`)

`--match` pairs every predicted state with existing PDG mesons. The data comes from
the official [PDG Python API](https://pdgapi.lbl.gov/doc/) (`pdg` package). It
ships the full PDG database as SQLite and works offline. Unlike the `particle`
package, it includes states without Monte-Carlo IDs, such as D_s1*(2700),
D_s1*(2860), D_s3*(2860), χc1(3872), ψ(4230) and the X states. The header names
the package version and the edition, e.g.
`PDG data: PDG Python API (pdg 2026.0), 2026 edition`, and JSON gives them
under `match_settings.source`. `threshold-finder` and `--particles` keep using
`particle`. The `--match-*` options require `--match`. Without `--match` the output
is unchanged.

**Data used.** The matcher reads every meson listing from `api.get_particles()`
(listings whose `data_flags` contain `M`) and every charge state in it
(`PdgParticle`). For each state it takes:

- the best mass from `best_summary()`. A mass given as a range, e.g. f₀(500) at
  "400 to 800", uses the midpoint and is flagged. Limits count as unknown.
- the best width, if the listing has a width entry. Upper limits, e.g. D*(2007)⁰ at
  "< 2.1", count as unknown. The API's `width` property returns 0 for listings
  without a width entry, so the matcher doesn't use it.
- the strings `quantum_J`, `quantum_P`, `quantum_C` and `quantum_I`.

Antiparticles (negative MC IDs or charges, and `Dbar…`/`Kbar…`/`Bbar…` names) are
skipped.

**Status.** The PDG API has no per-listing status. It only marks values that appear
in the Summary Tables (`in_summary_table`). `--match-status` uses threshold-finder's
codes as follows:

| Code | threshold-finder (`particle`) | PDG API |
|------|-------------------------------|---------|
| `0` | established | the listing's best mass is in the Summary Tables |
| `1` | evidence, not confirmed | no counterpart: selects nothing |
| `2` | omitted from the summary tables | the listing's best mass appears only in the Listings |

In the 2026 edition every meson listing with a usable mass has status 0, so the
default already includes D_s1*(2700) and ψ(4230). The light-meson "Further States"
(listing M300, e.g. X(1575), ρ(2000)) exist only as text descriptions without mass,
J or P data, so they can't be matched.

**Criterion.** A PDG state is a candidate for a prediction if all of these hold:

1. **Mass.** |m_pred − m_PDG| ≤ max(`--match-mass-tol`, `--match-width-frac` · Γ_PDG).
   The model predicts no widths, so the PDG width only widens the window for broad
   states. If Γ_PDG is unknown, the mass tolerance is used alone. The output shows the
   effective tolerance and whether it came from the width (`tol = 0.5·Γ = 74.1`).
2. **J and P equal.** C must also be equal when both are defined, i.e. the system is
   self-conjugate and the PDG state has C. States whose J or P is `?` are never
   matched, e.g. X(3940), K(1630) and D*(2640). Values in parentheses, and
   alternatives such as f_J(2220)'s `J = 2++ or 4`, count as unknown by default. With
   `--match-include-uncertain` they are used, with one candidate per alternative, and
   flagged `uncertain J^P in PDG`. J and P are never inferred from MC-ID digits,
   because those are a numbering convention, not measured quantum numbers.
3. **Flavour compatible.** The PDG API has no quark content, so the flavour family
   comes from the PDG meson naming scheme:

   | PDG name | Family |
   |----------|--------|
   | `pi rho omega phi eta f a b h` (`eta^'`, `f_2^'`, …) | light unflavoured |
   | `K…` | s q̄ |
   | `D…` / `D_s…` | c q̄ / c s̄ |
   | `B…` / `B_s…` / `B_c…` | b q̄ / b s̄ / b c̄ |
   | `eta_c J/psi psi chi_c h_c` | c c̄ |
   | `eta_b Upsilon chi_b h_b` | b b̄ |
   | `X(…)` | the PDG section it is listed in (the nearest preceding listing), flagged |
   | `T…` (four-quark states, e.g. T_cc̄1(3900)⁺ = Z_c) | never a q q̄ partner |

   This is matched against the system, where u and d are the model's averaged light quark q:

   | System | Matches |
   |--------|---------|
   | open flavour, e.g. c s̄, c ū | the same pair, up to charge conjugation and isospin partners: c q̄ matches D⁰, D⁺, D*⁰ and D*⁺ |
   | c c̄, b b̄ | the c c̄ / b b̄ family, including χc1(3872) and ψ(4230) |
   | q q̄, u ū, d d̄ | light unflavoured I = 1 and I = 0 states |
   | u d̄, d ū | light unflavoured I = 1 only |
   | s s̄ | light unflavoured I = 0 states |

   Light isoscalars (I = 0, or I unknown) fit both q q̄ and s s̄. They are flagged
   **flavour-ambiguous** because isoscalar annihilation mixing isn't modelled.

**Isospin and charge multiplets.** Most charge states share one PDG listing, e.g.
ρ(770)⁰/⁺, D_0*(2300)⁰/⁺ and K*(892)⁰/⁺. Some don't: D⁰/D⁺, D*(2007)⁰/D*(2010)⁺,
K⁺/K⁰ and B⁰/B⁺ are separate listings. These are merged when their MC IDs agree up
to the u/d quark digits (423/413). Members must also share isospin, J and P, which
keeps π⁰ (I = 1) and η (I = 0) apart. The representative is the neutral member, or
the lowest MC ID if there is no neutral member. For a multiplet it is shown with the
charge suffix dropped (`D`, `D^*(2007)`, `rho(770)`), and singletons keep their full
PDG API name (`omega(782)0`, `D_s1^*(2700)+`). The mass is the **average over the
charge states**, matching the model's isospin-averaged light quark, so D is 1867.3 MeV
from D⁰ and D⁺. The width is the average of the known widths. C comes from the
neutral member. `--details` and JSON list the members.

**Assignment.** Every candidate within tolerance is listed for each prediction,
ranked by |Δm| (Δ = m_pred − m_PDG). Each J^P(C) group also gets a one-to-one
assignment from `scipy.optimize.linear_sum_assignment` that minimises the total
|Δm|. Pairs outside tolerance get a large cost and are dropped afterwards, so the
assignment first maximises the number of matched pairs. The assignment only uses
the predictions shown in the window. The table shows the assigned partner
(`→ chi_c1(1P) 3510.7 MeV (Δ = -2.5)`) or `→ no match`. `--details` lists all
candidates with Δm, effective tolerance, assignment and flags.

**Unassigned PDG states.** After the table, the section "PDG states in window
without assigned prediction" lists the compatible states that no prediction took.
These states pass the flavour rules and the J P / `--C` filter, and their mass lies
in [mass_min, mass_max] widened by their own tolerance. This helps to spot missing
levels and exotics. A J^PC that no q q̄ state can have (0⁻⁻, 0⁺⁻, 1⁻⁺, 2⁺⁻, …) is
flagged `exotic J^PC`, e.g. π₁(1600).

**JSON.** Each state gains a `match` object with `assigned` (or `null`) and
`candidates`. Each entry has the name, members, `pdg_listing` (e.g. `M182`), status,
mass, width, J/P/C, isospin, `delta_MeV`, `tolerance_MeV`, `tolerance_source`
(`mass_tol` or `width`), `flags` and `assigned_to`. The top level gains
`match_settings`, which holds the tolerances, status, `include_uncertain` and the
data source, and `unassigned_pdg`. The existing keys are unchanged.

**Data quirks** (2026 edition): φ(2170) lists a charged row with I = 0, which is
merged into the single φ(2170) entry. Some narrow states have only an upper limit on the width
(D_s*, D_s0*(2317)) or no width entry (B*), so they use the mass tolerance alone.

```
$ gi-spectrum 2900 3600 --quarks c c --nmax 2 --max-L 2 --match
GI spectrum of c c̄  in [2900.0, 3600.0] MeV  (solver = fd, nmax = 2, L = SPD)
Parameters: .../gimodel/data/parameters.provisional.toml
PDG match: |m_pred - m_PDG| <= max(50 MeV, 0.5·Γ_PDG), PDG status 0; Δ = m_pred - m_PDG; one-to-one per J^P(C), minimal total |Δ|
PDG data: PDG Python API (pdg 2026.0), 2026 edition; only listings with known mass, J and P are matched (uncertain J/P count as unknown)
Found 6 state(s):
  1^1S_0  J^PC=0^-+  mass=2966.7 MeV  → eta_c(1S) 2984.1 MeV (Δ = -17.4)
  1^3S_1  J^PC=1^--  mass=3091.0 MeV  → J/psi(1S) 3096.9 MeV (Δ = -5.9)
  1^3P_0  J^PC=0^++  mass=3442.8 MeV  → chi_c0(1P) 3415.5 MeV (Δ = +27.3)
  1^3P_1  J^PC=1^++  mass=3508.2 MeV  → chi_c1(1P) 3510.7 MeV (Δ = -2.5)
  1^1P_1  J^PC=1^+-  mass=3515.1 MeV  → h_c(1P) 3525.4 MeV (Δ = -10.3)
  1^3P_2  J^PC=2^++  mass=3548.1 MeV  → chi_c2(1P) 3556.2 MeV (Δ = -8.0)

PDG states in window without assigned prediction ([2900.0, 3600.0] MeV widened by each state's tolerance):
  eta_c(2S)  J^PC=0^-+  mass=3637.8 MeV  Γ=11.6 MeV  tol = 50.0
```

η_c(2S) is listed because its prediction (2^1S_0 at 3625.4 MeV) lies above the
window. With `2900 3700` it is assigned, and so is ψ(2S).

```
$ gi-spectrum 2650 3000 --particles 'D~0' 'K-' --match --match-mass-tol 80
Reference particles: 'D~0' + 'K-'  ->  s c̄
GI spectrum of s c̄  in [2650.0, 3000.0] MeV  (solver = fd, nmax = 3, L = SPDF)
Parameters: .../gimodel/data/parameters.provisional.toml
PDG match: |m_pred - m_PDG| <= max(80 MeV, 0.5·Γ_PDG), PDG status 0; Δ = m_pred - m_PDG; one-to-one per J^P(C), minimal total |Δ|
PDG data: PDG Python API (pdg 2026.0), 2026 edition; only listings with known mass, J and P are matched (uncertain J/P count as unknown)
Found 6 state(s):
  2^1S_0  J^P=0^-  mass=2674.6 MeV                                      → no match
  2^3S_1  J^P=1^-  mass=2734.3 MeV                                      → D_s1^*(2700)+ 2714.0 MeV (Δ = +20.3)
  1^3D_1  J^P=1^-  mass=2898.5 MeV                                      → D_s1^*(2860)+ 2859.0 MeV (Δ = +39.5)
  1^1D_2  J^P=2^-  mass=2899.9 MeV  (mixture: 61% 1^1D_2 + 39% 1^3D_2)  → no match
  1^3D_3  J^P=3^-  mass=2916.1 MeV                                      → D_s3^*(2860)+ 2860.5 MeV (Δ = +55.6)
  1^3D_2  J^P=2^-  mass=2925.4 MeV  (mixture: 61% 1^3D_2 + 39% 1^1D_2)  → no match

PDG states in window without assigned prediction ([2650.0, 3000.0] MeV widened by each state's tolerance):
  D_s0(2590)+  J^P=0^-  mass=2591.0 MeV  Γ=89.0 MeV  tol = 80.0
```

```
$ gi-spectrum 700 850 1 -1 --quarks u u --nmax 1 --max-L 1 --match --details
...
Found 1 state(s):
  1^3S_1  J^PC=1^--  mass=771.2 MeV  → rho(770) 775.2 MeV (Δ = -4.0, tol = 0.5·Γ = 74.1)
      contributions [MeV]: central=667.1  contact=+104.1  spin-orbit=+0.0  tensor=+0.0  mixing=+0.0
      components: +1.0000 1^3S_1
      PDG candidates (by |Δm|):
        rho(770)     mass=775.2 MeV  Γ=148.2 MeV  Δ=-4.0  tol = 0.5·Γ = 74.1  assigned  (members: rho(770)0, rho(770)+)
        omega(782)0  mass=782.7 MeV  Γ=8.7 MeV  Δ=-11.5  tol = 50.0  unassigned  [flavour-ambiguous]

PDG states in window without assigned prediction ([700.0, 850.0] MeV widened by each state's tolerance):
  omega(782)0  J^PC=1^--  mass=782.7 MeV  Γ=8.7 MeV  tol = 50.0  [flavour-ambiguous]
```

With `--quarks u d` (I = 1) ω(782) is not a candidate.

### Tests and reference data

`tests/gimodel/` compares the port layer by layer against JSON reference data generated from GIModel.jl (`tests/gimodel/reference/`). `tools/julia_reference/` holds the Julia generator and explains how to regenerate the data.

```bash
python -m pytest                 # everything
python -m pytest -m "not slow"   # skip the long Julia cross-checks
```
