"""Command-line interface ``gi-spectrum``: Godfrey–Isgur meson spectrum in a mass window.

Mirrors the conventions of the ``threshold-finder`` CLI (package ``threshold_finder``,
distribution ``thresholds``), which it reuses for parity parsing, PDG quark-content
parsing, particle lookup and name suggestions. Both packages ship in the same
distribution; threshold_finder and particle are imported only here, and the --match modules
(gimodel.pdg_match, gimodel.pdg_data with the PDG Python API ``pdg``) are loaded only by this
module, never by the gimodel core.

Masses on the command line and in the output are in MeV; the model works in GeV.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import L_LABELS, L_SYMBOLS

PROG = "gi-spectrum"
QUARK_LETTERS = ("u", "d", "q", "s", "c", "b")
MIXTURE_THRESHOLD = 0.99  # dominant |c|^2 below this marks a state as a mixture in the table
COMPONENT_CUTOFF = 1e-3  # smallest |c| listed in --details
MATCH_MASS_TOL_DEFAULT = 50.0  # MeV
MATCH_WIDTH_FRAC_DEFAULT = 0.5
MATCH_STATUS_DEFAULT = [0]


class CLIError(Exception):
    """A user-facing error: printed as ``ERROR: ...`` on stderr, exit status 1."""


def _threshold_finder_modules():
    """Import the threshold_finder helpers (kept out of the gimodel core's import graph)."""
    from threshold_finder import cli as tf_cli
    from threshold_finder import flavor as tf_flavor
    from threshold_finder import lookup as tf_lookup

    return tf_cli, tf_flavor, tf_lookup


# --- argument types ------------------------------------------------------------------------


def parse_meson_J(s: str) -> int:
    """Integer J >= 0; half-integer J is rejected (mesons have integer spin)."""
    try:
        value = Fraction(s.strip())
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"Invalid J '{s}': use a non-negative integer, e.g. 0, 1, 2")
    if value < 0:
        raise argparse.ArgumentTypeError(f"Invalid J '{s}': J must be a non-negative integer")
    if value.denominator != 1:
        raise argparse.ArgumentTypeError(
            f"Invalid J '{s}': mesons (q qbar) have integer J; half-integer J is not possible"
        )
    return int(value)


def _quark_letter(s: str) -> str:
    letter = s.strip().lower()
    if letter not in QUARK_LETTERS:
        raise argparse.ArgumentTypeError(
            f"Invalid quark '{s}': use one of {', '.join(QUARK_LETTERS)} (q = isospin-averaged u/d)"
        )
    return letter


# --- flavour resolution ----------------------------------------------------------------------


def _bar(f: str) -> str:
    return f + "̄"


@dataclass
class System:
    """The q1 q2bar system: physical letters (as given) and the model flavours (u, d -> q)."""

    quark: str
    antiquark: str
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def model_flavors(self) -> tuple[str, str]:
        light = {"u": "q", "d": "q"}
        return light.get(self.quark, self.quark), light.get(self.antiquark, self.antiquark)

    @property
    def self_conjugate(self) -> bool:
        return self.quark == self.antiquark

    @property
    def light_hidden_flavor(self) -> bool:
        return self.self_conjugate and self.model_flavors[0] in ("q", "s")

    @property
    def label(self) -> str:
        return f"{self.quark} {_bar(self.antiquark)}"

    @property
    def model_label(self) -> str:
        f1, f2 = self.model_flavors
        return f"{f1} {_bar(f2)}"


_SQRT = re.compile(r"sqrt\(\d+\)")
_PAIR = re.compile(r"([udscbt])([UDSCBT])")
_COEFF_ONLY = re.compile(r"^[()+\-/*.\d\sxypq]*$")


def _neutral_superposition(quarks: str) -> Optional[set[str]]:
    """Flavours of a superposition of same-flavour q qbar pairs (e.g. ``(uU-dD)/sqrt(2)``).

    Returns the set of flavours involved, or ``None`` if the string is anything else
    (``p(dS)+q(Ds)``, ``Maybe non-qQ``, baryons, ...).
    """
    stripped = _SQRT.sub("", quarks)
    pairs = _PAIR.findall(stripped)
    if not pairs or any(q != a.lower() for q, a in pairs):
        return None
    if not _COEFF_ONLY.match(_PAIR.sub("", stripped)):
        return None
    return {q for q, _ in pairs}


def _shell_quote(name: str) -> str:
    return f"'{name}'"


class _CommandBuilder:
    """Ready-to-run suggestion commands, as threshold-finder prints them."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def __call__(self, flavour_part: str) -> str:
        a = self.args
        parts = [PROG, str(a.mass_min), str(a.mass_max)]
        if a.J is not None and a.P is not None:
            parts += [str(a.J), "+1" if a.P > 0 else "-1"]
        parts.append(flavour_part)
        if a.solver != "fd":
            parts.append(f"--solver {a.solver}")
        if a.nmax != 3:
            parts.append(f"--nmax {a.nmax}")
        if a.max_L != 3:
            parts.append(f"--max-L {a.max_L}")
        return " ".join(parts)


def _system_from_net(net: dict[str, int], flavors: tuple[str, ...], source: str, cmd) -> System:
    nonzero = {f: v for f, v in net.items() if v}
    quarks = [f for f, v in nonzero.items() if v == 1]
    antiquarks = [f for f, v in nonzero.items() if v == -1]
    if not nonzero:
        zero = [f for f in flavors if net.get(f) == 0] or ["c"]
        examples = "\n".join(f"  {cmd(f'--quarks {f} {f}')}" for f in zero)
        raise CLIError(
            f"Net flavour from {source} is zero (hidden flavour), which the --u/--d/--s/--c/--b "
            "numbers cannot specify.\nGive the quark and antiquark explicitly with --quarks, e.g.:\n"
            + examples
        )
    if len(quarks) != 1 or len(antiquarks) != 1 or len(nonzero) != 2:
        shown = ", ".join(f"{f}={v:+d}" for f, v in nonzero.items())
        raise CLIError(
            f"Net flavour from {source} ({shown}) is not one quark (+1) plus one antiquark (-1) "
            "of different flavours, so it is not a q qbar meson.\n"
            f"Give the content explicitly, e.g.:\n  {cmd('--quarks Q QBAR')}"
        )
    return System(quarks[0], antiquarks[0], source)


def _system_from_particles(
    names: list[str], overrides: dict[str, int], flavors: tuple[str, ...], tf_flavor, cmd
) -> System:
    from particle import Particle

    contents: list[tuple[str, str, Optional[dict[str, int]], Optional[set[str]]]] = []
    for name in names:
        p = Particle.from_name(name)
        qs = p.quarks or ""
        clear = tf_flavor.parse_quark_content(qs) if qs else None
        neutral = None if clear is not None else _neutral_superposition(qs)
        contents.append((name, qs, clear, neutral))

    ambiguous = [(n, qs) for n, qs, clear, neutral in contents if clear is None and neutral is None]
    suggestions: list[str] = []
    if len(names) == 1 and not overrides:
        # A single self-conjugate particle defines hidden flavour by its own content.
        name, qs, clear, neutral = contents[0]
        if neutral is not None and neutral <= {"u", "d"}:
            system = System("q", "q", f"'{name}'")
            system.notes.append(
                f"'{name}' has light superposition content {qs}: using the isospin-averaged light q q\u0304"
            )
            return system
        if neutral is not None:  # e.g. eta, omega, phi: x(uU+dD)+y(sS)
            ambiguous = [(name, qs)]
            suggestions = sorted({"q" if f in "ud" else f for f in neutral}, key="qscbt".index)
        elif clear is not None and not any(clear.values()):
            letters = {ch.lower() for ch in qs if ch.isalpha()}
            if len(letters) == 1:
                f = letters.pop()
                return System(f, f, f"'{name}'")

    if ambiguous:
        lines = ["Quark content is ambiguous (mixed/superposition state) for:"]
        lines += [f"  {n}  (PDG quarks string: '{qs}')" for n, qs in ambiguous]
        lines.append("")
        lines.append("Specify the quark and antiquark explicitly. Example command:")
        if suggestions:
            lines += [f"  {cmd(f'--quarks {f} {f}')}" for f in suggestions]
        else:
            lines.append(f"  {cmd('--quarks ??? ???')}")
        raise CLIError("\n".join(lines))

    net: dict[str, int] = {}
    for _, _, clear, _ in contents:
        for f, v in (clear or {}).items():  # neutral superpositions carry no net flavour
            net[f] = net.get(f, 0) + v
    net.update(overrides)
    source = " + ".join(f"'{n}'" for n in names)
    if overrides:
        source += " with " + ", ".join(f"--{f} {v}" for f, v in overrides.items())
    system = _system_from_net(net, flavors, source, cmd)
    if any(neutral is not None for *_, neutral in contents):
        system.notes.append("flavour-neutral superpositions (e.g. pi0, eta) contribute no net flavour")
    return system


def _check_unknown_particles(names: list[str], tf_lookup, cmd) -> None:
    """Print threshold-finder style suggestions for unknown names; exit 1 if any."""
    from particle import Particle, ParticleNotFound

    unknown = []
    for name in names:
        try:
            Particle.from_name(name)
        except (ParticleNotFound, KeyError):
            unknown.append(name)
    if not unknown:
        return
    for name in unknown:
        print(f"ERROR: Unknown particle '{name}'", file=sys.stderr)
        print("Did you mean one of these?", file=sys.stderr)
        for s in tf_lookup.suggest_particles(name, n=5):
            current = [s if n == name else n for n in names]
            print(f"  {cmd('--particles ' + ' '.join(_shell_quote(p) for p in current))}", file=sys.stderr)
    sys.exit(1)


def resolve_system(args: argparse.Namespace, parser: argparse.ArgumentParser, tf_flavor, tf_lookup) -> System:
    flavors = tf_flavor.FLAVORS
    flags = {f: getattr(args, f) for f in flavors if getattr(args, f) is not None}
    cmd = _CommandBuilder(args)

    if args.quarks:
        if flags or args.particles:
            parser.error("--quarks cannot be combined with --particles or the --u/--d/--s/--c/--b flags")
        q, a = args.quarks
        return System(q, a, "--quarks")
    if args.particles:
        _check_unknown_particles(args.particles, tf_lookup, cmd)
        # As in threshold-finder: explicit flavour flags override the derived values.
        return _system_from_particles(args.particles, flags, flavors, tf_flavor, cmd)
    if flags:
        return _system_from_net(flags, flavors, "the flavour flags", cmd)
    parser.error(
        "the flavour content is required: give --quarks Q QBAR, net flavour flags "
        "(e.g. --c 1 --u -1), or --particles P1 [P2 ...]"
    )
    raise AssertionError  # unreachable


# --- spectrum --------------------------------------------------------------------------------


def _make_solver(name: str, nmax: int):
    from .solvers import FiniteDifferenceSolver, OscillatorSolver

    nlevels = max(6, nmax)
    if name == "fd":
        return FiniteDifferenceSolver(nlevels_per_channel=nlevels)
    return OscillatorSolver(nlevels_per_channel=nlevels)


def _fmt_P(P: int) -> str:
    return "+" if P > 0 else "-"


def _state_record(spec, state, self_conjugate: bool) -> dict:
    from .spectrum import physical_components

    L = L_SYMBOLS[state.L]
    S = (state.multiplicity - 1) // 2
    P = (-1) ** (L + 1)
    C = (-1) ** (L + S) if self_conjugate else None
    comps = sorted(
        ((c.basis.label, float(c.coefficient)) for c in physical_components(spec, state)),
        key=lambda x: -abs(x[1]),
    )
    dominant, weight = comps[0][0], comps[0][1] ** 2
    corrected = state.corrected
    contributions = {
        "central": corrected.central_GeV,
        "contact": corrected.contact_shift_GeV,
        "spin_orbit": corrected.spin_orbit_shift_GeV,
        "tensor": corrected.tensor_shift_GeV,
        "mixing": state.mass_GeV - corrected.mass_GeV,
    }
    jpc = f"{state.J}^{_fmt_P(P)}" + (_fmt_P(C) if C is not None else "")
    return {
        "label": dominant,
        "basis_label": state.label,
        "n": state.n,
        "L": L,
        "S": S,
        "J": state.J,
        "P": P,
        "C": C,
        "JPC": jpc,
        "mass_MeV": 1000.0 * state.mass_GeV,
        "mixture": weight < MIXTURE_THRESHOLD,
        "dominant_weight": weight,
        "components": [{"label": l, "coefficient": c} for l, c in comps],
        "contributions_MeV": {k: 1000.0 * v for k, v in contributions.items()},
    }


def _passes(rec: dict, args) -> bool:
    if args.J is not None and (rec["J"] != args.J or rec["P"] != args.P):
        return False
    if args.C is not None and rec["C"] != args.C:
        return False
    return True


def _completeness_warning(records: list[dict], args) -> Optional[str]:
    """Sectors whose highest computed level (n = nmax) still lies below mass_max."""
    top: dict[tuple[int, int, int], dict] = {}
    for rec in records:
        if not _passes(rec, args):
            continue
        key = (rec["L"], rec["S"], rec["J"])
        if key not in top or rec["mass_MeV"] > top[key]["mass_MeV"]:
            top[key] = rec
    short = sorted({key[0] for key, rec in top.items() if rec["mass_MeV"] < args.mass_max})
    if not short:
        return None
    Ls = ", ".join(f"{L_LABELS[L]} (L={L})" for L in short)
    return (
        f"WARNING: the highest computed level (n = {args.nmax}) for {Ls} lies below "
        f"mass_max = {args.mass_max:.1f} MeV; the window may be incomplete. "
        f"Rerun with a larger --nmax (e.g. --nmax {args.nmax + 2})."
    )


# --- output ----------------------------------------------------------------------------------


def _mixture_text(rec: dict) -> str:
    shown = [(x["label"], x["coefficient"] ** 2) for x in rec["components"] if x["coefficient"] ** 2 >= 0.005][:3]
    return "mixture: " + " + ".join(f"{100 * w:.0f}% {l}" for l, w in shown)


def _short_flag(flag: str) -> str:
    return "no PDG quark content" if flag.startswith("quark content missing") else flag


def _flags_text(flags, short: bool = True) -> str:
    if not flags:
        return ""
    shown = [_short_flag(f) for f in flags] if short else list(flags)
    return "  [" + "; ".join(shown) + "]"


def _tol_text(tol: float, source: str, args) -> str:
    if source == "width":
        return f"tol = {args.match_width_frac:g}·Γ = {tol:.1f}"
    return f"tol = {tol:.1f}"


def _assigned_text(cand, args) -> str:
    if cand is None:
        return "→ no match"
    s = cand.state
    text = f"→ {s.name} {s.mass:.1f} MeV (Δ = {cand.delta:+.1f}"
    if cand.tolerance_source == "width":
        text += f", {_tol_text(cand.tolerance, cand.tolerance_source, args)}"
    return text + ")" + _flags_text(cand.flags)


def _fmt_width(width) -> str:
    if width is None:
        return "Γ=?"
    return f"Γ={width:.2g} MeV" if width < 0.05 else f"Γ={width:.1f} MeV"


def _pdg_jpc(state, self_conjugate: bool) -> str:
    C = _fmt_P(state.C) if self_conjugate and state.C is not None else ""
    return f"{state.J}^{_fmt_P(state.P)}{C}"


def _match_details(i: int, matched, owners: dict, records: list[dict], args) -> list[str]:
    cands = matched.candidates[i]
    if not cands:
        return ["      PDG candidates: none within tolerance"]
    lines = ["      PDG candidates (by |Δm|):"]
    w_name = max(len(c.state.name) for c in cands)
    for c in cands:
        s = c.state
        owner = owners.get(s)
        status = ("assigned" if owner == i else
                  f"assigned to {records[owner]['label']}" if owner is not None else "unassigned")
        line = (f"        {s.name.ljust(w_name)}  mass={s.mass:.1f} MeV  {_fmt_width(s.width)}  Δ={c.delta:+.1f}  "
                f"{_tol_text(c.tolerance, c.tolerance_source, args)}  {status}")
        if len(s.members) > 1:
            line += f"  (members: {', '.join(s.members)})"
        lines.append(line + _flags_text(c.flags, short=False))
    return lines


def _unassigned_section(matched, args, self_conjugate: bool) -> list[str]:
    lines = ["", f"PDG states in window without assigned prediction "
                 f"([{args.mass_min:.1f}, {args.mass_max:.1f}] MeV widened by each state's tolerance):"]
    if not matched.unassigned:
        return lines + ["  none"]
    w_name = max(len(u.state.name) for u in matched.unassigned)
    w_jpc = max(len(_pdg_jpc(u.state, self_conjugate)) for u in matched.unassigned)
    jp_name = "J^PC" if self_conjugate else "J^P"
    for u in matched.unassigned:
        s = u.state
        line = (f"  {s.name.ljust(w_name)}  {jp_name}={_pdg_jpc(s, self_conjugate).ljust(w_jpc)}  "
                f"mass={s.mass:.1f} MeV  {_fmt_width(s.width)}  {_tol_text(u.tolerance, u.tolerance_source, args)}")
        if args.details and len(s.members) > 1:
            line += f"  (members: {', '.join(s.members)})"
        lines.append(line + _flags_text(u.flags, short=not args.details))
    return lines


def _format_text(header: list[str], records: list[dict], args, matched=None, self_conjugate: bool = False) -> str:
    lines = list(header)
    lines.append(f"Found {len(records)} state(s):")
    if not records:
        if matched is not None:
            lines += _unassigned_section(matched, args, self_conjugate)
        return "\n".join(lines)
    w_label = max(len(r["label"]) for r in records)
    w_jpc = max(len(r["JPC"]) for r in records)
    w_mass = max(len(f"{r['mass_MeV']:.1f}") for r in records)
    jp_name = "J^PC" if records[0]["C"] is not None else "J^P"

    def row(r):
        line = f"  {r['label'].ljust(w_label)}  {jp_name}={r['JPC'].ljust(w_jpc)}  mass={r['mass_MeV']:{w_mass}.1f} MeV"
        if r["mixture"]:
            line += f"  ({_mixture_text(r)})"
        return line.rstrip()

    if matched is not None:
        w_row = max(len(row(r)) for r in records)
        owners = {c.state: i for i, c in enumerate(matched.assigned) if c is not None}
    for i, r in enumerate(records):
        line = row(r)
        if matched is not None:
            line = f"{line.ljust(w_row)}  {_assigned_text(matched.assigned[i], args)}"
        lines.append(line)
        if args.details:
            c = r["contributions_MeV"]
            lines.append(
                "      contributions [MeV]: "
                f"central={c['central']:.1f}  contact={c['contact']:+.1f}  "
                f"spin-orbit={c['spin_orbit']:+.1f}  tensor={c['tensor']:+.1f}  mixing={c['mixing']:+.1f}"
            )
            comps = [f"{x['coefficient']:+.4f} {x['label']}" for x in r["components"]
                     if abs(x["coefficient"]) >= COMPONENT_CUTOFF]
            lines.append("      components: " + ", ".join(comps))
            if matched is not None:
                lines += _match_details(i, matched, owners, records, args)
    if matched is not None:
        lines += _unassigned_section(matched, args, self_conjugate)
    return "\n".join(lines)


def main(argv=None):
    tf_cli, tf_flavor, tf_lookup = _threshold_finder_modules()

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "List Godfrey–Isgur quark-model meson states (q qbar) in a mass window. "
            "The flavour content is required: --quarks, the net flavour flags, or --particles. "
            "J and P are optional and filter the output."
        ),
    )
    parser.add_argument("mass_min", type=float, help="Lower bound of the mass window (MeV)")
    parser.add_argument("mass_max", type=float, help="Upper bound of the mass window (MeV)")
    parser.add_argument(
        "J", type=parse_meson_J, nargs="?", default=None,
        help="Filter: total angular momentum (integer; mesons have integer J). Optional.",
    )
    parser.add_argument(
        "P", type=tf_cli.parse_parity, nargs="?", default=None,
        help="Filter: parity, +1 or -1. Must be given together with J.",
    )
    parser.add_argument(
        "--C", type=tf_cli.parse_parity, default=None, metavar="{+1,-1}",
        help="Filter: C-parity (self-conjugate systems only)",
    )
    parser.add_argument(
        "--solver", choices=("fd", "ho"), default="fd",
        help="Radial solver: fd = finite differences (default), ho = harmonic-oscillator basis",
    )
    parser.add_argument(
        "--nmax", type=int, default=3, metavar="N",
        help="Radial levels computed per (L, S, J) sector (default: 3)",
    )
    parser.add_argument(
        "--max-L", type=int, default=3, metavar="L",
        help="Maximum orbital angular momentum (default: 3, i.e. S P D F; at most 4)",
    )
    parser.add_argument(
        "--params", type=Path, default=None, metavar="PATH",
        help="Parameter TOML file (default: the parameter set shipped with gimodel)",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Show the mass contributions (central, contact, spin-orbit, tensor, mixing) and mixing components",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    match_group = parser.add_argument_group(
        "PDG matching",
        "Pair each predicted state with compatible PDG mesons: same J^P (and C when both are "
        "defined), compatible flavour, and |m_pred - m_PDG| <= max(MATCH_MASS_TOL, "
        "MATCH_WIDTH_FRAC * Gamma_PDG).",
    )
    match_group.add_argument("--match", action="store_true", help="Match predictions to PDG mesons")
    match_group.add_argument(
        "--match-mass-tol", type=float, default=None, metavar="MEV",
        help=f"Mass tolerance in MeV (default: {MATCH_MASS_TOL_DEFAULT:g})",
    )
    match_group.add_argument(
        "--match-width-frac", type=float, default=None, metavar="F",
        help=f"Widen the window to F * Gamma_PDG for broad states (default: {MATCH_WIDTH_FRAC_DEFAULT:g})",
    )
    match_group.add_argument(
        "--match-status", type=int, nargs="+", default=None, metavar="S",
        help="PDG status codes to include (0=established, 1=evidence, 2=omitted). Default: 0. "
             "With the PDG API data: 0 = mass in the Summary Tables, 2 = mass only in the Listings; "
             "nothing maps to 1",
    )
    match_group.add_argument(
        "--match-include-uncertain", action="store_true", default=None,
        help="Also use PDG J/P values that are parenthesised or given as alternatives "
             "(e.g. J = '2++ or 4'); such candidates are flagged. '?' always counts as unknown",
    )

    flavor_group = parser.add_argument_group(
        "flavour content (required, exactly one way)",
        "u and d both map to the model's isospin-averaged light quark q. "
        "The quark is flavour 1, the antiquark flavour 2.",
    )
    flavor_group.add_argument(
        "--quarks", nargs=2, type=_quark_letter, metavar=("Q", "QBAR"),
        help="Quark and antiquark, e.g. '--quarks c u' (c ubar) or '--quarks c c' (charmonium). "
             "Letters: u d q s c b.",
    )
    for f in tf_flavor.FLAVORS:
        flavor_group.add_argument(
            f"--{f}", type=int, default=None, metavar="N",
            help=f"Net {f}-quark number (#{f} - #{f}bar); the flags must add up to one quark and "
                 "one antiquark of different flavours",
        )
    flavor_group.add_argument(
        "--particles", nargs="+", metavar="P",
        help="PDG particle names whose summed quark content defines the system, e.g. "
             "'--particles D0 pi+' (c dbar) or '--particles J/psi(1S)' (c cbar). "
             "Explicit --u/--d/... flags override the derived values.",
    )

    args = parser.parse_args(argv)

    if (args.J is None) != (args.P is None):
        parser.error("Provide either both J and P, or neither.")
    if args.mass_min > args.mass_max:
        parser.error("mass_min must not exceed mass_max")
    if args.nmax < 1:
        parser.error("--nmax must be >= 1")
    if not 0 <= args.max_L <= max(L_LABELS):
        parser.error(f"--max-L must be between 0 and {max(L_LABELS)}")
    match_opts = [o for o, v in (("--match-mass-tol", args.match_mass_tol),
                                 ("--match-width-frac", args.match_width_frac),
                                 ("--match-status", args.match_status),
                                 ("--match-include-uncertain", args.match_include_uncertain)) if v is not None]
    if match_opts and not args.match:
        parser.error(f"{', '.join(match_opts)} requires --match")
    if args.match_mass_tol is None:
        args.match_mass_tol = MATCH_MASS_TOL_DEFAULT
    if args.match_width_frac is None:
        args.match_width_frac = MATCH_WIDTH_FRAC_DEFAULT
    if args.match_status is None:
        args.match_status = list(MATCH_STATUS_DEFAULT)
    args.match_include_uncertain = bool(args.match_include_uncertain)
    if args.match_mass_tol < 0 or args.match_width_frac < 0:
        parser.error("--match-mass-tol and --match-width-frac must be >= 0")

    try:
        system = resolve_system(args, parser, tf_flavor, tf_lookup)
        if args.C is not None and not system.self_conjugate:
            raise CLIError(
                f"--C applies only to self-conjugate (equal-flavour) systems; {system.label} has no C-parity"
            )
        records, solver, params_path = _compute(args, system)
    except CLIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    notes = list(system.notes)
    if system.model_flavors != (system.quark, system.antiquark):
        notes.append(f"u and d map to the isospin-averaged light quark: model system {system.model_label}")
    if system.light_hidden_flavor:
        notes.append(
            "isoscalar annihilation mixing (eta/eta', omega/phi) is not included; "
            "light hidden-flavour masses are the unmixed q qbar / s sbar values"
        )
    warning = _completeness_warning(records, args)
    shown = sorted(
        (r for r in records if _passes(r, args) and args.mass_min <= r["mass_MeV"] <= args.mass_max),
        key=lambda r: r["mass_MeV"],
    )

    matched = settings = source = pm = None
    if args.match:
        from . import pdg_match as pm

        try:
            settings, source, matched = _run_match(args, system, shown, pm)
        except CLIError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    if warning:
        print(warning, file=sys.stderr)

    if args.json:
        out = {
            "system": {
                "quark": system.quark,
                "antiquark": system.antiquark,
                "label": system.label,
                "model_flavors": list(system.model_flavors),
                "self_conjugate": system.self_conjugate,
                "source": system.source,
            },
            "solver": args.solver,
            "solver_description": str(solver),
            "parameters": str(params_path),
            "nmax": args.nmax,
            "max_L": args.max_L,
            "window_MeV": [args.mass_min, args.mass_max],
            "filter": {"J": args.J, "P": args.P, "C": args.C},
            "notes": notes,
            "warnings": [warning] if warning else [],
            "states": shown if matched is None else _states_with_match(shown, matched),
        }
        if matched is not None:
            out["match_settings"] = {
                "mass_tol_MeV": settings.mass_tol,
                "width_frac": settings.width_frac,
                "status": list(settings.status),
                "criterion": "|m_pred - m_PDG| <= max(mass_tol, width_frac * width_PDG)",
                "include_uncertain": args.match_include_uncertain,
                "source": {"api": "pdg", "package_version": source.package_version,
                           "edition": source.edition, "citation": source.citation},
            }
            out["unassigned_pdg"] = [pm.unassigned_dict(u) for u in matched.unassigned]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.particles:
        overrides = [f"--{f} {getattr(args, f)}" for f in tf_flavor.FLAVORS if getattr(args, f) is not None]
        print(
            f"Reference particles: {' + '.join(_shell_quote(p) for p in args.particles)}  ->  {system.label}"
            + (f"  (overridden by {', '.join(overrides)})" if overrides else "")
        )
    jp = ""
    if args.J is not None:
        jp = f" for J^P{'C' if args.C is not None else ''} = {args.J}^{_fmt_P(args.P)}"
        jp += _fmt_P(args.C) if args.C is not None else ""
    elif args.C is not None:
        jp = f" with C = {_fmt_P(args.C)}"
    Ls = "".join(L_LABELS[L] for L in range(args.max_L + 1))
    header = [
        f"GI spectrum of {system.label}{jp}  in [{args.mass_min:.1f}, {args.mass_max:.1f}] MeV"
        f"  (solver = {args.solver}, nmax = {args.nmax}, L = {Ls})",
        f"Parameters: {params_path}",
    ]
    header += [f"Note: {n}" for n in notes]
    if matched is not None:
        header.append(
            f"PDG match: |m_pred - m_PDG| <= max({settings.mass_tol:g} MeV, {settings.width_frac:g}·Γ_PDG), "
            f"PDG status {' '.join(str(x) for x in settings.status)}; Δ = m_pred - m_PDG; "
            "one-to-one per J^P(C), minimal total |Δ|"
        )
        header.append(f"{source.note}; only listings with known mass, J and P are matched"
                      + ("" if args.match_include_uncertain else " (uncertain J/P count as unknown)"))
    print(_format_text(header, shown, args, matched, system.self_conjugate))


def _run_match(args, system: System, shown: list[dict], pm):
    from . import pdg_data

    settings = pm.MatchSettings(args.match_mass_tol, args.match_width_frac, tuple(args.match_status))
    predictions = [pm.Prediction(r["label"], r["J"], r["P"], r["C"], r["mass_MeV"]) for r in shown]
    try:
        entries = pdg_data.load_entries(settings.status, args.match_include_uncertain)
        source = pdg_data.source_info()
    except ImportError as exc:
        raise CLIError(f"--match needs the PDG Python API ('pip install pdg'): {exc}")
    states = pm.collapse_multiplets(entries)
    flavour = pm.SystemFlavour.from_quarks(system.quark, system.antiquark)

    def jpc_filter(J: int, P: int, C) -> bool:
        return _passes({"J": J, "P": P, "C": C}, args)

    window = (args.mass_min, args.mass_max)
    return settings, source, pm.match(predictions, states, flavour, settings, window, jpc_filter)


def _states_with_match(shown: list[dict], matched) -> list[dict]:
    from .pdg_match import candidate_dict

    owners = {c.state: shown[i]["label"] for i, c in enumerate(matched.assigned) if c is not None}
    out = []
    for i, rec in enumerate(shown):
        assigned = matched.assigned[i]
        out.append({
            **rec,
            "match": {
                "assigned": None if assigned is None else candidate_dict(assigned, rec["label"]),
                "candidates": [candidate_dict(c, owners.get(c.state)) for c in matched.candidates[i]],
            },
        })
    return out


def _compute(args, system: System):
    from .parameters import default_parameters_path, load_parameters_and_quark_masses
    from .quarks import Meson
    from .spectrum import compute_spectrum, spectrum_levels

    params_path = default_parameters_path() if args.params is None else args.params
    try:
        params, mq = load_parameters_and_quark_masses(params_path)
    except FileNotFoundError:
        raise CLIError(f"parameter file not found: {params_path}")
    except Exception as exc:  # TOML syntax or schema errors from the loader
        raise CLIError(f"cannot load parameter file {params_path}: {exc}")

    solver = _make_solver(args.solver, args.nmax)
    meson = Meson.from_table(mq, *system.model_flavors)
    levels = spectrum_levels(args.nmax, tuple(L_LABELS[L] for L in range(args.max_L + 1)))
    try:
        spec = compute_spectrum(params, meson, levels=levels, solver=solver)
    except NotImplementedError as exc:
        detail = f" ({exc})" if str(exc) else ""
        raise CLIError(
            f"the '{args.solver}' solver is not available in this gimodel version{detail}. "
            "Use --solver fd."
        )
    except ValueError as exc:
        raise CLIError(f"{args.solver} solve failed: {exc}")
    records = [_state_record(spec, s, system.self_conjugate) for s in spec.states]
    return records, solver, params_path


if __name__ == "__main__":
    main()
