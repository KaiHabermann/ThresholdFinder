# ThresholdFinder

Finds two-body hadronic thresholds compatible with given J^P quantum numbers. Given a mass range and a target J^P, it scans all pairs of PDG hadrons whose combined mass falls in that range and checks whether they can couple — via some orbital angular momentum L — to produce the desired quantum numbers.

## Requirements

- Python >= 3.11
- [`particle`](https://github.com/scikit-hep/particle) >= 0.24

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
| `--unique-pairs`       | off      | Show each particle pair only once (keeping the lowest L), instead of one entry per valid L. |

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
    max_L=None,                              # None = automatic
    total_charge=0.0,
    flavor_filter=FlavorFilter(u=0, d=0, s=0, c=0, b=0),  # all net quark numbers = 0
    status_filter=(0,),                      # established particles only
)
result = finder.run()

print(result)  # formatted summary

for c in result.combinations:
    print(c.particle1, "+", c.particle2, "  L =", c.L, "  threshold =", c.threshold, "MeV")
```

Constrain only specific flavors by omitting the rest:

```python
# Only require net charm = 0; u, d, s, b are unconstrained
flavor_filter=FlavorFilter(c=0)

# Only require net strangeness = -1
flavor_filter=FlavorFilter(s=-1)
```

`ThresholdResult` has the fields `J_target`, `P_target`, `mass_min`, `mass_max`, `max_L`, `flavor_filter`, and `combinations` (a list of `CombinationResult`).

Each `CombinationResult` contains:

| Field        | Type    | Description                              |
|--------------|---------|------------------------------------------|
| `particle1`  | `str`   | PDG name of the first particle           |
| `particle2`  | `str`   | PDG name of the second particle          |
| `mass1`      | `float` | Mass of particle 1 (MeV)                 |
| `mass2`      | `float` | Mass of particle 2 (MeV)                 |
| `threshold`  | `float` | Combined threshold mass = m₁ + m₂ (MeV) |
| `charge1`    | `float` | Charge of particle 1                     |
| `charge2`    | `float` | Charge of particle 2                     |
| `J1`, `J2`   | `float` | Spins of the two particles               |
| `P1`, `P2`   | `int`   | Parities of the two particles            |
| `L`          | `int`   | Orbital angular momentum                 |
| `J_total`    | `float` | Total angular momentum (= J_target)      |
| `P_total`    | `int`   | Total parity (= P_target)                |
| `identical`  | `bool`  | Whether the two particles are identical  |

## Physics

The tool checks whether a pair (particle 1 with J₁^P₁, particle 2 with J₂^P₂) in a state of orbital angular momentum L can produce the target J^P:

**Parity:**
```
P_total = P₁ · P₂ · (-1)^L
```

**Angular momentum:** J_total must be reachable by coupling J₁ ⊗ J₂ ⊗ L via the triangle rule.

**Identical bosons:** For two identical bosons (e.g. π⁰π⁰), the spatial wave function must be symmetric under exchange, which requires L to be even.

**Flavor conservation:** Net quark numbers are computed as #quark − #antiquark for each flavor (u, d, s, c, b). They are additive: the net quark number of the pair is the sum of the two individual net quark numbers. Setting a flavor to 0 requires the pair to have no net quark content in that flavor (e.g. K⁺K⁻ passes s=0 since K⁺ has s=−1 and K⁻ has s=+1). Particles with mixed or superposition quark content (η, ω, φ, π⁰, …) have undefined quark numbers and are excluded from any result when a flavor constraint is active.

Particle data (masses, J, P, charge, quark content) are read from the PDG via the [`particle`](https://github.com/scikit-hep/particle) package. Only hadrons with known mass, J, and P are considered.
