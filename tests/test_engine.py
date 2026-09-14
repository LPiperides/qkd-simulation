import numpy as np

from qkd_sim import engine
from qkd_sim.engine import transmit
from qkd_sim.primitives import BASES


def random_round(rng, n):
    return rng.integers(0, 2, n), rng.choice(BASES, n), rng.choice(BASES, n)


def test_no_eve_returns_none_and_matches_alice_in_matching_bases(ideal_simulator):
    rng = np.random.default_rng(1)
    bits, alice_bases, bob_bases = random_round(rng, 1000)
    bob, eve = transmit(bits, alice_bases, bob_bases, simulator=ideal_simulator, rng=rng)
    assert eve is None
    same = alice_bases == bob_bases
    assert (bob[same] == bits[same]).all()


def test_eve_matches_alice_in_her_matching_bases(ideal_simulator):
    rng = np.random.default_rng(2)
    bits, alice_bases, bob_bases = random_round(rng, 1000)
    eve_bases = rng.choice(BASES, 1000)
    bob, eve = transmit(
        bits, alice_bases, bob_bases, eve_bases=eve_bases, simulator=ideal_simulator, rng=rng
    )
    assert eve is not None and eve.shape == (1000,)
    same = alice_bases == eve_bases
    assert (eve[same] == bits[same]).all()
    # Bob agrees with Eve whenever he measures in her basis: he receives her re-sent state.
    same_eb = eve_bases == bob_bases
    assert (bob[same_eb] == eve[same_eb]).all()


def test_stages_use_distinct_seeds(monkeypatch, ideal_simulator):
    seeds = []

    def fake_sample(bits, bases, meas, simulator, *, seed=None):
        seeds.append(seed)
        return np.zeros(len(bits), dtype=int)

    monkeypatch.setattr(engine, "sample_outcomes", fake_sample)
    rng = np.random.default_rng(3)
    bits, alice_bases, bob_bases = random_round(rng, 10)
    transmit(
        bits, alice_bases, bob_bases, eve_bases=alice_bases, simulator=ideal_simulator, rng=rng
    )
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]
    assert all(isinstance(s, int) for s in seeds)
