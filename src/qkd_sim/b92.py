"""B92 (Bennett, 1992): key distribution with two non-orthogonal states.

Alice encodes bit 0 as |0> and bit 1 as |->. Bob measures in a random basis
and keeps only the rounds where his outcome is *impossible* for one of the
two states, which tells him the other was sent:

    Bob measures '+' and sees |1>  ->  |0> is ruled out  ->  Alice sent |->, bit 1
    Bob measures 'x' and sees |+>  ->  |-> is ruled out  ->  Alice sent |0>, bit 0

Any other outcome is consistent with both states and is discarded. Each
conclusive outcome occurs with probability 1/2 x 1/2, so a quarter of the
rounds survive on an ideal channel, all of them error-free.

The sifting rule depends only on Bob's basis and outcome, never on what
Alice sent; the function signature enforces that. An intercept-and-resend
attacker forwards a wrong-basis eigenstate half the time, which produces
conclusive-but-wrong outcomes and a sifted error rate of 1/3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from .channel import make_simulator
from .engine import transmit
from .primitives import BASES
from .transcript import build_transcript, masked_bits


def sift(bob_bases: np.ndarray, bob_outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unambiguous-discrimination rule of B92.

    Returns ``(mask, bits)``: ``mask`` marks conclusive rounds and ``bits`` is
    the bit Bob infers in each round (only meaningful where ``mask`` holds).
    Deliberately takes no information about Alice.
    """
    bases = np.asarray(bob_bases)
    outcomes = np.asarray(bob_outcomes, dtype=int)
    conclusive_plus = (bases == "+") & (outcomes == 1)  # rules out |0>  -> bit 1
    conclusive_x = (bases == "x") & (outcomes == 0)  # rules out |->  -> bit 0
    mask = conclusive_plus | conclusive_x
    bits = np.where(bases == "+", 1, 0)
    return mask, bits


def run_b92(
    num_bits: int,
    *,
    with_eve: bool = False,
    noise: float = 0.0,
    seed: int | None = None,
    simulator: AerSimulator | None = None,
) -> pd.DataFrame:
    """Simulate ``num_bits`` rounds of B92 and return the transcript.

    Parameters are the same as :func:`qkd_sim.bb84.run_bb84`.
    """
    rng = np.random.default_rng(seed)
    simulator = simulator if simulator is not None else make_simulator(noise)

    alice_bits = rng.integers(0, 2, num_bits)
    # bit 0 -> |0> = ('+', 0); bit 1 -> |-> = ('x', 1). The index of the
    # state inside its basis happens to equal the bit, so the same array
    # drives preparation.
    alice_bases = np.where(alice_bits == 0, "+", "x")
    bob_bases = rng.choice(BASES, num_bits)
    eve_bases = rng.choice(BASES, num_bits) if with_eve else None

    bob_outcomes, eve_outcomes = transmit(
        alice_bits, alice_bases, bob_bases, eve_bases=eve_bases, simulator=simulator, rng=rng
    )
    sifted, bob_bits = sift(bob_bases, bob_outcomes)

    eve_guess = None
    if with_eve:
        eve_conclusive, eve_bits = sift(eve_bases, eve_outcomes)
        eve_guess = masked_bits(eve_bits, eve_conclusive)

    return build_transcript(
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        bob_outcomes=bob_outcomes,
        sifted=sifted,
        bob_bits=bob_bits,
        eve_bases=eve_bases,
        eve_outcomes=eve_outcomes,
        eve_guess=eve_guess,
    )
