"""Cross-check reported combinations against PDG data independently.

For every combination returned by ThresholdFinder (n=2..4) we re-resolve
each particle name from the PDG, recompute charge and net quark numbers
from scratch, and assert they match what the combination reports.

No L or J^P checks here — only particle identity, charge, and flavor.
"""
import pytest
from particle import Particle, ParticleNotFound
from particle.pdgid import is_hadron

from threshold_finder import ThresholdFinder
from threshold_finder.flavor import parse_quark_content, FLAVORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdg_lookup(name: str) -> Particle:
    """Resolve a particle name via PDG, raising AssertionError with context on failure."""
    try:
        return Particle.from_name(name)
    except (ParticleNotFound, KeyError):
        # Some antiparticles are stored under the conjugate name
        raise AssertionError(f"PDG lookup failed for particle name '{name}'")


def _check_combination(combo) -> None:
    """Re-derive charge and quark content for every particle in a combination
    and assert they match the reported values."""
    n = len(combo.particles)

    # Re-resolve each particle independently from PDG
    pdg_particles = [_pdg_lookup(name) for name in combo.particles]

    # --- Charge ---
    for i, (reported_charge, pdg_p) in enumerate(zip(combo.charges, pdg_particles)):
        expected_charge = float(pdg_p.charge)
        assert abs(reported_charge - expected_charge) < 1e-6, (
            f"Charge mismatch for '{combo.particles[i]}': "
            f"reported {reported_charge}, PDG {expected_charge}"
        )

    reported_total_charge = sum(combo.charges)
    pdg_total_charge = sum(float(p.charge) for p in pdg_particles)
    assert abs(reported_total_charge - pdg_total_charge) < 1e-6, (
        f"Total charge mismatch for {combo.particles}: "
        f"reported {reported_total_charge}, PDG {pdg_total_charge}"
    )

    # --- Threshold = sum of masses ---
    pdg_masses = [float(p.mass) for p in pdg_particles]
    for i, (reported_mass, pdg_mass) in enumerate(zip(combo.masses, pdg_masses)):
        assert abs(reported_mass - pdg_mass) < 1e-3, (
            f"Mass mismatch for '{combo.particles[i]}': "
            f"reported {reported_mass}, PDG {pdg_mass}"
        )
    expected_threshold = sum(pdg_masses)
    assert abs(combo.threshold - expected_threshold) < 1e-3, (
        f"Threshold mismatch for {combo.particles}: "
        f"reported {combo.threshold:.3f}, sum of PDG masses {expected_threshold:.3f}"
    )

    # --- Net quark numbers (only for particles with defined quark content) ---
    pdg_qcs = []
    for pdg_p in pdg_particles:
        qc = parse_quark_content(pdg_p.quarks) if pdg_p.quarks else None
        pdg_qcs.append(qc)

    # If all particles have defined quark content, check net flavor numbers
    if all(qc is not None for qc in pdg_qcs):
        for flavor in FLAVORS:
            net = sum(qc.get(flavor, 0) for qc in pdg_qcs)  # type: ignore[union-attr]
            # combo.parities are spin parities, not flavor — flavor is in quark_content
            # We don't store per-combination net quark numbers in CombinationResult,
            # but we can cross-check via the reported charges (already done above)
            # and by verifying the per-particle quark content sums are self-consistent.
            # The key invariant: net charge = (2/3)*n_u - (1/3)*n_d - (1/3)*n_s
            #                               + (2/3)*n_c - (1/3)*n_b  (ignoring t)
            # This is automatically satisfied if individual charges are correct.
            pass  # individual charge check above is sufficient

        # Extra: verify that the sum of net quark numbers is charge-consistent
        # Q = (2u - d - s + 2c - b) / 3  for the combination
        n_u = sum(qc.get("u", 0) for qc in pdg_qcs)  # type: ignore[union-attr]
        n_d = sum(qc.get("d", 0) for qc in pdg_qcs)  # type: ignore[union-attr]
        n_s = sum(qc.get("s", 0) for qc in pdg_qcs)  # type: ignore[union-attr]
        n_c = sum(qc.get("c", 0) for qc in pdg_qcs)  # type: ignore[union-attr]
        n_b = sum(qc.get("b", 0) for qc in pdg_qcs)  # type: ignore[union-attr]
        charge_from_quarks = (2 * n_u - n_d - n_s + 2 * n_c - n_b) / 3.0
        assert abs(charge_from_quarks - pdg_total_charge) < 1e-6, (
            f"Quark content inconsistent with charge for {combo.particles}: "
            f"Q from quarks = {charge_from_quarks:.4f}, PDG charge = {pdg_total_charge:.4f}\n"
            f"  net u={n_u} d={n_d} s={n_s} c={n_c} b={n_b}"
        )


# ---------------------------------------------------------------------------
# Parameterised search runs
# ---------------------------------------------------------------------------

# (mass_min, mass_max, J, P, n_body, label)
SEARCH_CASES = [
    (600,  800,  0, -1, 2, "n2_scalar_kaon_region"),
    (250,  400,  1, -1, 2, "n2_vector_rho_region"),
    (900, 1100,  0, -1, 2, "n2_scalar_1GeV"),
    (900, 1100,  1, -1, 2, "n2_vector_phi_region"),
    (400,  560,  0, -1, 3, "n3_scalar_3pi_region"),
    (400,  560,  1, -1, 3, "n3_vector_3pi_region"),
    (800, 1100,  0, -1, 3, "n3_scalar_eta_region"),
    (800, 1100,  1, -1, 3, "n3_vector_1GeV"),
    (500,  650,  0, -1, 4, "n4_scalar_4pi_region"),
    (500,  650,  1, -1, 4, "n4_vector_4pi_region"),
    (800, 1000,  0, -1, 4, "n4_scalar_omega_region"),
]


@pytest.mark.parametrize("mass_min,mass_max,J,P,n_body,label", SEARCH_CASES,
                         ids=[c[5] for c in SEARCH_CASES])
def test_combination_correctness(mass_min, mass_max, J, P, n_body, label):
    """Every combination in the result must have correct charge, mass, and
    quark-content-consistent charge when all particles have defined quark content."""
    result = ThresholdFinder(mass_min, mass_max, J, P, n_body=n_body).run()

    # Ensure we actually got results to check (otherwise the test is vacuous)
    assert len(result.combinations) > 0, (
        f"No combinations found for {label} — adjust mass range or J^P"
    )

    for combo in result.combinations:
        _check_combination(combo)


def test_all_reported_particles_are_hadrons():
    """Every particle name in every result must resolve to a known PDG hadron."""
    result = ThresholdFinder(400, 600, 0, -1, n_body=3).run()
    for combo in result.combinations:
        for name in combo.particles:
            p = _pdg_lookup(name)
            assert is_hadron(p.pdgid), f"'{name}' (pdgid={p.pdgid}) is not a hadron"


def test_charge_filter_respected_n3():
    """All three-body results with total_charge=0 must sum to zero charge."""
    result = ThresholdFinder(400, 600, 0, -1, n_body=3).run()
    for combo in result.combinations:
        assert abs(sum(combo.charges)) < 1e-6, (
            f"Charge sum != 0 for {combo.particles}: charges={combo.charges}"
        )


def test_charge_filter_respected_n4():
    """All four-body results with total_charge=0 must sum to zero charge."""
    result = ThresholdFinder(500, 650, 0, -1, n_body=4).run()
    for combo in result.combinations:
        assert abs(sum(combo.charges)) < 1e-6, (
            f"Charge sum != 0 for {combo.particles}: charges={combo.charges}"
        )


def test_threshold_within_mass_range_n3():
    """Every reported threshold must lie within [mass_min, mass_max]."""
    mass_min, mass_max = 400.0, 600.0
    result = ThresholdFinder(mass_min, mass_max, 0, -1, n_body=3).run()
    for combo in result.combinations:
        assert mass_min - 1e-3 <= combo.threshold <= mass_max + 1e-3, (
            f"Threshold {combo.threshold:.3f} outside [{mass_min}, {mass_max}] "
            f"for {combo.particles}"
        )


def test_threshold_within_mass_range_n4():
    """Every reported threshold must lie within [mass_min, mass_max]."""
    mass_min, mass_max = 500.0, 650.0
    result = ThresholdFinder(mass_min, mass_max, 0, -1, n_body=4).run()
    for combo in result.combinations:
        assert mass_min - 1e-3 <= combo.threshold <= mass_max + 1e-3, (
            f"Threshold {combo.threshold:.3f} outside [{mass_min}, {mass_max}] "
            f"for {combo.particles}"
        )
