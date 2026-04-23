"""Command-line interface for ThresholdFinder."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .finder import ThresholdFinder
from .flavor import FlavorFilter, FLAVORS
from .lookup import (
    resolve_flavor_filter_from_particles,
    suggest_particles,
    lowest_jp_combinations,
    pair_can_produce,
)
from .result import _fmt_J


def parse_parity(s: str) -> int:
    s = s.strip()
    if s in ("+1", "+", "1", "plus"):
        return 1
    if s in ("-1", "-", "minus"):
        return -1
    raise argparse.ArgumentTypeError(f"Invalid parity '{s}': use +1 or -1")


def _extra_cli_args(args: argparse.Namespace) -> str:
    parts = []
    if args.max_L is not None:
        parts.append(f"--max-L {args.max_L}")
    if args.charge != 0.0:
        parts.append(f"--charge {args.charge}")
    if args.status != [0]:
        parts.append("--status " + " ".join(str(s) for s in args.status))
    if args.unique_pairs:
        parts.append("--unique-pairs")
    return " ".join(parts)


def _check_not_found(names: list[str], mass_min: float, mass_max: float,
                     J: Optional[float], P: Optional[int], extra: str) -> bool:
    """Check for unknown particle names and print suggestions. Returns True if any not found."""
    from particle import Particle, ParticleNotFound
    not_found = []
    for name in names:
        try:
            Particle.from_name(name)
        except (ParticleNotFound, KeyError):
            suggestions = suggest_particles(name, n=5)
            not_found.append((name, suggestions))

    if not_found:
        p_str = ("+1" if P > 0 else "-1") if P is not None else ""
        jp_part = f"{J} {p_str}" if J is not None and P is not None else ""
        for name, suggestions in not_found:
            other = [n for n in names if n != name]
            print(f"ERROR: Unknown particle '{name}'", file=sys.stderr)
            print("Did you mean one of these?", file=sys.stderr)
            for s in suggestions:
                current = [s if n == name else n for n in names]
                cmd_parts = (
                    ["threshold-finder", str(mass_min), str(mass_max)]
                    + ([jp_part] if jp_part else [])
                    + ["--particles"] + [f"'{p}'" for p in current]
                    + ([extra] if extra else [])
                )
                print(f"  {' '.join(cmd_parts)}", file=sys.stderr)
        return True
    return False


def _run_for_jp(
    J_target: float,
    P_target: int,
    args: argparse.Namespace,
    flavor_filter: FlavorFilter,
    particles: Optional[tuple[str, str]],
) -> None:
    """Run ThresholdFinder for one J^P target and print results."""
    # Check whether the reference pair can produce this J^P at all
    if particles:
        name1, name2 = particles
        if not pair_can_produce(name1, name2, J_target, P_target, args.max_L):
            p_str = "+" if P_target > 0 else "-"
            print(
                f"WARNING: '{name1}' + '{name2}' cannot produce "
                f"J^P = {_fmt_J(J_target)}^{p_str} at any L"
                + (f" <= {args.max_L}" if args.max_L is not None else ""),
                file=sys.stderr,
            )
            return

    finder = ThresholdFinder(
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        J_target=J_target,
        P_target=P_target,
        max_L=args.max_L,
        total_charge=args.charge,
        flavor_filter=flavor_filter,
        status_filter=tuple(args.status),
    )
    result = finder.run()

    if args.unique_pairs:
        seen: dict[tuple[str, str], int] = {}
        filtered = []
        for c in sorted(result.combinations, key=lambda x: (x.threshold, x.L)):
            key = tuple(sorted([c.particle1, c.particle2]))
            if key not in seen:
                seen[key] = c.L
                filtered.append(c)
        result.combinations = filtered

    print(result)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="threshold-finder",
        description=(
            "Find two-body hadronic thresholds compatible with given J^P quantum numbers. "
            "J and P can be omitted when --particles is given; the 3 lowest J^P combinations "
            "the pair can produce are then used automatically."
        ),
    )
    parser.add_argument("mass_min", type=float, help="Lower bound of mass range (MeV)")
    parser.add_argument("mass_max", type=float, help="Upper bound of mass range (MeV)")
    parser.add_argument(
        "J", type=float, nargs="?", default=None,
        help="Target total angular momentum (e.g. 1, 0.5). Optional if --particles is given.",
    )
    parser.add_argument(
        "P", type=parse_parity, nargs="?", default=None,
        help="Target parity: +1 or -1. Optional if --particles is given.",
    )
    parser.add_argument(
        "--max-L", type=int, default=None, metavar="L",
        help="Maximum orbital angular momentum (default: auto)",
    )
    parser.add_argument(
        "--charge", type=float, default=0.0,
        help="Required total charge of the pair (default: 0)",
    )
    parser.add_argument(
        "--status", type=int, nargs="+", default=[0], metavar="S",
        help="PDG status codes to include (0=established, 1=evidence, 2=omitted). Default: 0",
    )
    parser.add_argument(
        "--unique-pairs", action="store_true",
        help="Show each particle pair only once (lowest L per pair)",
    )
    parser.add_argument(
        "--max-mass-particles", type=float, default=None, metavar="M",
        help="Max mass (MeV) for individual particles to consider (default: mass_max)",
    )

    flavor_group = parser.add_argument_group(
        "flavor conservation",
        "Constrain the net quark content of the two-particle system. "
        "Each flag takes an integer (net quark number = #quark - #antiquark). "
        "Only the flags you provide are enforced; omitted flavors are unconstrained. "
        "Use --particles to derive these automatically from two reference particles.",
    )
    for f in FLAVORS:
        flavor_group.add_argument(
            f"--{f}", type=int, default=None, metavar="N",
            help=f"Required net {f}-quark number of the pair",
        )
    flavor_group.add_argument(
        "--particles", nargs=2, metavar=("P1", "P2"),
        help=(
            "Two PDG particle names to derive flavor conservation from. "
            "If J and P are omitted, the 3 lowest J^P combinations this pair "
            "can produce are used automatically."
        ),
    )

    args = parser.parse_args(argv)

    # --- Validate: J and P required unless --particles is given ---
    if args.J is None or args.P is None:
        if not args.particles:
            parser.error("J and P are required unless --particles is given.")
        if args.J is not None or args.P is not None:
            parser.error("Provide either both J and P, or neither (auto-detect via --particles).")

    extra = _extra_cli_args(args)

    # --- Resolve --particles ---
    flavor_filter = FlavorFilter(u=args.u, d=args.d, s=args.s, c=args.c, b=args.b)
    particles: Optional[tuple[str, str]] = None

    if args.particles:
        name1, name2 = args.particles

        if _check_not_found([name1, name2], args.mass_min, args.mass_max,
                            args.J, args.P, extra):
            sys.exit(1)

        try:
            combined, warnings = resolve_flavor_filter_from_particles(
                name1, name2,
                args.mass_min, args.mass_max,
                args.J, args.P,
                extra_cli_args=extra,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        for w in warnings:
            print(w, file=sys.stderr)

        # Explicit flags override derived values
        merged = {**combined}
        for f in FLAVORS:
            explicit = getattr(args, f)
            if explicit is not None:
                merged[f] = explicit
        flavor_filter = FlavorFilter(**{f: merged.get(f) for f in FLAVORS})

        from particle import Particle
        p1_mass = float(Particle.from_name(name1).mass)
        p2_mass = float(Particle.from_name(name2).mass)
        print(f"Reference pair: '{name1}' + '{name2}'  "
              f"threshold = {p1_mass + p2_mass:.1f} MeV  "
              f"({p1_mass:.1f} + {p2_mass:.1f})")
        print(f"Flavor conservation:")
        for f in FLAVORS:
            v = getattr(flavor_filter, f)
            if v is not None:
                print(f"  {f} = {v:+d}")
        print()

        particles = (name1, name2)

    # --- Determine J^P targets ---
    if args.J is not None and args.P is not None:
        jp_targets = [(args.J, args.P)]
    else:
        # Auto-detect 3 lowest J^P from the pair
        combos = lowest_jp_combinations(name1, name2, n=3)
        jp_targets = [(J, P) for J, P, _ in combos]
        print("No J^P given — using the 3 lowest combinations "
              f"'{name1}' + '{name2}' can produce:")
        for J, P, L in combos:
            p_str = "+" if P > 0 else "-"
            print(f"  J^P = {_fmt_J(J)}^{p_str}  (lowest at L={L})")
        print()

    # --- Run ---
    for J_target, P_target in jp_targets:
        _run_for_jp(J_target, P_target, args, flavor_filter, particles)


if __name__ == "__main__":
    main()
