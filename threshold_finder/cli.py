"""Command-line interface for ThresholdFinder."""
import argparse
import sys

from .finder import ThresholdFinder
from .flavor import FlavorFilter, FLAVORS
from .lookup import resolve_flavor_filter_from_particles, suggest_particles


def parse_parity(s: str) -> int:
    s = s.strip()
    if s in ("+1", "+", "1", "plus"):
        return 1
    if s in ("-1", "-", "minus"):
        return -1
    raise argparse.ArgumentTypeError(f"Invalid parity '{s}': use +1 or -1")


def _extra_cli_args(args: argparse.Namespace) -> str:
    """Reconstruct non-flavor CLI flags for the suggested command."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="threshold-finder",
        description="Find two-body hadronic thresholds compatible with given J^P quantum numbers.",
    )
    parser.add_argument("mass_min", type=float, help="Lower bound of mass range (MeV)")
    parser.add_argument("mass_max", type=float, help="Upper bound of mass range (MeV)")
    parser.add_argument("J", type=float, help="Target total angular momentum (e.g. 1, 1.5)")
    parser.add_argument("P", type=parse_parity, help="Target parity: +1 or -1")
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
        help="Two PDG particle names to derive flavor conservation from (e.g. --particles 'D0' 'p')",
    )

    args = parser.parse_args(argv)

    # --- Resolve flavor from --particles if given ---
    flavor_filter = FlavorFilter(
        u=args.u, d=args.d, s=args.s, c=args.c, b=args.b,
    )

    if args.particles:
        name1, name2 = args.particles

        # Check for unknown particles first
        not_found = []
        for name in (name1, name2):
            from particle import Particle, ParticleNotFound
            try:
                Particle.from_name(name)
            except (ParticleNotFound, KeyError):
                suggestions = suggest_particles(name, n=5)
                not_found.append((name, suggestions))

        if not_found:
            extra = _extra_cli_args(args)
            p_str = "+1" if args.P > 0 else "-1"
            for name, suggestions in not_found:
                print(f"ERROR: Unknown particle '{name}'", file=sys.stderr)
                print(f"Did you mean one of these?", file=sys.stderr)
                for s in suggestions:
                    cmd_parts = ["threshold-finder", str(args.mass_min), str(args.mass_max),
                                 str(args.J), p_str, "--particles",
                                 f"'{s}'" if name == name1 else f"'{name1}'",
                                 f"'{s}'" if name == name2 else f"'{name2}'"]
                    if extra:
                        cmd_parts.append(extra)
                    print(f"  {' '.join(cmd_parts)}", file=sys.stderr)
            sys.exit(1)

        try:
            combined, warnings = resolve_flavor_filter_from_particles(
                name1, name2,
                args.mass_min, args.mass_max,
                args.J, args.P,
                extra_cli_args=_extra_cli_args(args),
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        for w in warnings:
            print(w, file=sys.stderr)

        # Merge: explicit --u/--d/etc flags override derived values
        merged = {**combined}
        for f in FLAVORS:
            explicit = getattr(args, f)
            if explicit is not None:
                merged[f] = explicit

        flavor_filter = FlavorFilter(**{f: merged.get(f) for f in FLAVORS})

        # Report what flavor numbers are being used
        print("Flavor conservation derived from "
              f"'{name1}' + '{name2}':")
        for f in FLAVORS:
            v = getattr(flavor_filter, f)
            if v is not None:
                print(f"  {f} = {v:+d}")
        print()

    finder = ThresholdFinder(
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        J_target=args.J,
        P_target=args.P,
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


if __name__ == "__main__":
    main()
