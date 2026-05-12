"""Tests for n-body threshold finding."""
import pytest
from threshold_finder import ThresholdFinder
from threshold_finder.qn import j_range_nbody, parity_nbody, can_produce_nbody
from threshold_finder.particles import get_particle_combinations, load_hadrons


# --- qn utilities ---

def test_j_range_nbody_two_scalars_L1():
    # Three scalars coupled: s_total=0, then +L=1 -> J=1
    result = j_range_nbody([0.0, 0.0], 1)
    assert 1.0 in result


def test_j_range_nbody_three_scalars_L0():
    # Three J=0 particles with L=0 -> only J=0 reachable
    result = j_range_nbody([0.0, 0.0, 0.0], 0)
    assert result == {0.0}


def test_j_range_nbody_three_scalars_L1():
    # Three J=0 particles with L=1 -> only J=1 reachable
    result = j_range_nbody([0.0, 0.0, 0.0], 1)
    assert result == {1.0}


def test_j_range_nbody_two_spinhalf_L0():
    # Two J=1/2 particles, L=0: couple to S=0 or S=1, then +L=0 -> {0, 1}
    result = j_range_nbody([0.5, 0.5], 0)
    assert 0.0 in result
    assert 1.0 in result


def test_j_range_nbody_agrees_with_two_body():
    # j_range_nbody for 2 particles should agree with j_range for a sample case
    from threshold_finder.qn import j_range
    for j1, j2, L in [(1.0, 1.0, 1), (0.5, 1.0, 2), (1.5, 0.5, 0)]:
        expected = set(j_range(j1, j2, L))
        got = j_range_nbody([j1, j2], L)
        assert expected == got, f"Mismatch for j1={j1}, j2={j2}, L={L}"


def test_parity_nbody_two_pions_L1():
    # P(-1) * P(-1) * (-1)^1 = -1
    assert parity_nbody([-1, -1], 1) == -1


def test_parity_nbody_three_pions_L0():
    # Three pions (P=-1 each): (-1)^3 * (-1)^0 = -1
    assert parity_nbody([-1, -1, -1], 0) == -1


def test_parity_nbody_three_pions_L1():
    # (-1)^3 * (-1)^1 = +1
    assert parity_nbody([-1, -1, -1], 1) == 1


def test_can_produce_nbody_three_pions_1minus():
    # 3 pions (all J=0, P=-1): to get 1^- need parity=-1 and J=1
    # parity_nbody([-1,-1,-1], L) = (-1)^(3+L); for P=-1 need 3+L odd -> L even
    # with L=2: P=(-1)^5=-1, J_range_nbody([0,0,0],2)={2} -> 1 not reachable
    # with L=1: P=(-1)^4=+1 -> wrong parity
    # with L=3: P=(-1)^6=+1 -> wrong
    # with L=0: P=(-1)^3=-1, J=0 -> not 1^-
    # So 3 pions cannot give 1^-? Let's verify the function agrees.
    assert not can_produce_nbody([0.0, 0.0, 0.0], [-1, -1, -1], 1.0, -1, 0)
    assert not can_produce_nbody([0.0, 0.0, 0.0], [-1, -1, -1], 1.0, -1, 1)
    assert not can_produce_nbody([0.0, 0.0, 0.0], [-1, -1, -1], 1.0, -1, 2)


def test_can_produce_nbody_three_pions_0minus():
    # 3 pions, L=0: P=-1, J=0 -> 0^- reachable
    assert can_produce_nbody([0.0, 0.0, 0.0], [-1, -1, -1], 0.0, -1, 0)


def test_can_produce_nbody_matches_two_body():
    # For n=2, can_produce_nbody should agree with can_produce (non-identical, bosons)
    from threshold_finder.qn import can_produce
    for j1, p1, j2, p2, J, P, L in [
        (0.0, -1, 0.0, -1, 1.0, -1, 1),
        (0.0, -1, 0.0, -1, 0.0, 1, 0),
        (1.0, -1, 1.0, -1, 0.0, 1, 0),
    ]:
        expected = can_produce(j1, p1, j2, p2, J, P, L, identical=False, both_bosons=True)
        got = can_produce_nbody([j1, j2], [p1, p2], J, P, L)
        assert expected == got, f"Mismatch j1={j1} p1={p1} j2={j2} p2={p2} J={J} P={P} L={L}"


# --- get_particle_combinations ---

def test_get_particle_combinations_count():
    hadrons = load_hadrons(max_mass=600.0, status_filter=frozenset({0}))
    pairs = get_particle_combinations(hadrons, n=2, total_charge=0.0)
    triples = get_particle_combinations(hadrons, n=3, total_charge=0.0)
    # There should be more triples than pairs for a non-trivial particle set
    assert len(triples) > len(pairs)


def test_get_particle_combinations_charge_filter():
    hadrons = load_hadrons(max_mass=300.0, status_filter=frozenset({0}))
    combos = get_particle_combinations(hadrons, n=3, total_charge=0.0)
    for combo in combos:
        total_q = sum(p.charge for p in combo)
        assert abs(total_q) < 1e-9, f"Total charge {total_q} != 0"


def test_get_particle_combinations_n2_matches_pairs():
    from threshold_finder.particles import get_particle_pairs
    hadrons = load_hadrons(max_mass=300.0, status_filter=frozenset({0}))
    # Normalize both to frozensets so ordering doesn't matter
    pairs_old = {frozenset([p1.pdgid, p2.pdgid]) for p1, p2, _ in get_particle_pairs(hadrons, total_charge=0.0)}
    combos_new = {frozenset([combo[0].pdgid, combo[1].pdgid]) for combo in get_particle_combinations(hadrons, n=2, total_charge=0.0)}
    assert pairs_old == combos_new


def test_get_particle_combinations_invalid_n():
    hadrons = load_hadrons(max_mass=300.0, status_filter=frozenset({0}))
    with pytest.raises(ValueError):
        get_particle_combinations(hadrons, n=1)


# --- ThresholdFinder n_body ---

def test_nbody_n2_gives_same_as_default():
    # n_body=2 explicitly should match the default
    result_default = ThresholdFinder(250, 300, 1, -1).run()
    result_n2 = ThresholdFinder(250, 300, 1, -1, n_body=2).run()
    keys_default = {(c.particles, c.L) for c in result_default.combinations}
    keys_n2 = {(c.particles, c.L) for c in result_n2.combinations}
    assert keys_default == keys_n2


def test_nbody_3body_0minus_finds_three_pions():
    # 3pi threshold starts at ~3*139.6=419 MeV; search around that
    # 0^- state with three pseudoscalars at L=0 is well-known (e.g. eta -> 3pi)
    result = ThresholdFinder(400, 450, 0, -1, n_body=3, max_L=0).run()
    particle_sets = [tuple(sorted(c.particles)) for c in result.combinations]
    # At least one 3-pion combination should appear
    three_pi = [ps for ps in particle_sets if all("pi" in p for p in ps)]
    assert len(three_pi) > 0, f"No 3-pion combination found in: {particle_sets[:10]}"


def test_nbody_invalid_n_raises():
    with pytest.raises(ValueError, match="n_body must be >= 2"):
        ThresholdFinder(250, 300, 1, -1, n_body=1)


def test_nbody_result_has_correct_n_body_field():
    result = ThresholdFinder(400, 450, 0, -1, n_body=3, max_L=0).run()
    assert result.n_body == 3


def test_nbody_combination_has_n_particles():
    result = ThresholdFinder(400, 450, 0, -1, n_body=3, max_L=0).run()
    for c in result.combinations:
        assert len(c.particles) == 3
        assert len(c.masses) == 3
        assert len(c.charges) == 3
        assert len(c.spins) == 3
        assert len(c.parities) == 3


def test_nbody_threshold_equals_sum_of_masses():
    result = ThresholdFinder(400, 450, 0, -1, n_body=3, max_L=0).run()
    for c in result.combinations:
        assert abs(c.threshold - sum(c.masses)) < 1e-6


# --- backward compatibility properties ---

def test_two_body_backward_compat_properties():
    result = ThresholdFinder(250, 300, 1, -1).run()
    for c in result.combinations:
        assert c.particle1 == c.particles[0]
        assert c.particle2 == c.particles[1]
        assert c.mass1 == c.masses[0]
        assert c.mass2 == c.masses[1]
        assert c.J1 == c.spins[0]
        assert c.J2 == c.spins[1]
        assert c.P1 == c.parities[0]
        assert c.P2 == c.parities[1]
