"""Command-line interface for ThresholdFinder."""
import argparse
import sys

from .finder import ThresholdFinder


def parse_parity(s: str) -> int:
    s = s.strip()
    if s in ("+1", "+", "1", "plus"):
        return 1
    if s in ("-1", "-", "minus"):
        return -1
    raise argparse.ArgumentTypeError(f"Invalid parity '{s}': use +1 or -1")


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

    args = parser.parse_args(argv)

    finder = ThresholdFinder(
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        J_target=args.J,
        P_target=args.P,
        max_L=args.max_L,
        total_charge=args.charge,
        status_filter=tuple(args.status),
    )

    result = finder.run()

    if args.unique_pairs:
        # Keep only lowest-L entry per unordered pair
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
