"""BB84 (Bennett & Brassard, 1984).

Alice encodes a random bit in a random basis; Bob measures in a random
basis. After the quantum phase both announce their bases and keep only the
rounds where they agree, which is half of them on average. In a kept round a
noiseless channel gives Bob exactly Alice's bit.

An intercept-and-resend eavesdropper guesses the wrong basis half the time,
and a wrong-basis re-send gives Bob a random result in his (matching) basis.
That produces a 25% error rate on the sifted key, the textbook BB84
signature. Eve herself learns Alice's bit on three quarters of the sifted
rounds (always when her basis matched, half the time when it did not).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from .channel import make_simulator
from .engine import transmit
from .primitives import BASES
from .transcript import build_transcript, masked_bits


def sift(alice_bases: np.ndarray, bob_bases: np.ndarray) -> np.ndarray:
    """Boolean mask of rounds kept by BB84 sifting: the bases agree."""
    return np.asarray(alice_bases) == np.asarray(bob_bases)


def run_bb84(
    num_bits: int,
    *,
    with_eve: bool = False,
    noise: float = 0.0,
    seed: int | None = None,
    simulator: AerSimulator | None = None,
) -> pd.DataFrame:
    """Simulate ``num_bits`` rounds of BB84 and return the transcript.

    Parameters
    ----------
    num_bits:
        Number of qubits Alice sends.
    with_eve:
        Include an intercept-and-resend eavesdropper measuring in random bases.
    noise:
        Depolarizing probability of the channel per hop (0 = ideal).
    seed:
        Seed for all randomness, classical and quantum. Same seed, same transcript.
    simulator:
        Optional pre-built ``AerSimulator``; overrides ``noise`` if given.
    """
    rng = np.random.default_rng(seed)
    simulator = simulator if simulator is not None else make_simulator(noise)

    alice_bits = rng.integers(0, 2, num_bits)
    alice_bases = rng.choice(BASES, num_bits)
    bob_bases = rng.choice(BASES, num_bits)
    eve_bases = rng.choice(BASES, num_bits) if with_eve else None

    bob_outcomes, eve_outcomes = transmit(
        alice_bits, alice_bases, bob_bases, eve_bases=eve_bases, simulator=simulator, rng=rng
    )
    sifted = sift(alice_bases, bob_bases)

    eve_guess = None
    if with_eve:
        # Eve's outcome is her guess in every round; when the bases are
        # revealed she also learns which of her guesses are in the key.
        eve_guess = masked_bits(eve_outcomes, np.ones(num_bits, dtype=bool))

    return build_transcript(
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        bob_outcomes=bob_outcomes,
        sifted=sifted,
        bob_bits=bob_outcomes,
        eve_bases=eve_bases,
        eve_outcomes=eve_outcomes,
        eve_guess=eve_guess,
    )
