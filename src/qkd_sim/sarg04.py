"""SARG04 (Scarani, Acín, Ribordy & Gisin, 2004).

SARG04 uses exactly the four BB84 states but changes the classical
post-processing so that the *basis* carries the bit:

    bit 0  ->  Alice prepares |0> or |1>  (Z basis, '+')
    bit 1  ->  Alice prepares |+> or |->  (X basis, 'x')

After Bob has measured in a random basis, Alice publicly announces an
unordered pair of non-orthogonal states {one Z state, one X state} that
contains the state she sent; the other member is chosen at random. Bob's
outcome is conclusive when it is orthogonal to one member of the pair:
that member cannot have been sent, so the other one was, and its basis is
the bit.

    Bob measured '+' and his outcome != the announced Z state
        -> the Z state is ruled out -> Alice sent the X state -> bit 1
    Bob measured 'x' and his outcome != the announced X state
        -> the X state is ruled out -> Alice sent the Z state -> bit 0

On an ideal channel Bob is conclusive in a quarter of the rounds with no
errors. Under intercept-and-resend the sifted error rate is 1/3.

Getting the bit convention right matters. An earlier version of this
simulator took the bit to be the *index* of the prepared state within its
basis (|0>,|+> -> 0 and |1>,|-> -> 1). With that convention half of Bob's
wrong deductions still land on the correct bit by luck, the eavesdropper's
footprint drops from 1/3 to about 1/6, and the protocol appears more secure
than it is. ``tests/test_sarg04.py`` guards against that regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from .channel import make_simulator
from .engine import transmit
from .primitives import BASES, STATE_LABELS
from .transcript import build_transcript, masked_bits


def key_bits(alice_bases: np.ndarray) -> np.ndarray:
    """SARG04 key bit: 0 for a Z-basis preparation, 1 for an X-basis one."""
    return (np.asarray(alice_bases) == "x").astype(int)


def announce(
    alice_bases: np.ndarray, alice_prep_bits: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Alice's public announcement for every round.

    Returns ``(announced_z, announced_x)``: the index (0 or 1) of the Z-basis
    and X-basis members of the announced pair. The member in Alice's own
    basis is the state she actually sent; the other is a fresh random state.
    """
    bases = np.asarray(alice_bases)
    prep = np.asarray(alice_prep_bits, dtype=int)
    decoys = rng.integers(0, 2, len(bases))
    announced_z = np.where(bases == "+", prep, decoys)
    announced_x = np.where(bases == "x", prep, decoys)
    return announced_z, announced_x


def sift(
    bob_bases: np.ndarray,
    bob_outcomes: np.ndarray,
    announced_z: np.ndarray,
    announced_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bob's discrimination rule given the announced pair.

    Returns ``(mask, bits)`` in the same convention as :func:`qkd_sim.b92.sift`.
    """
    bases = np.asarray(bob_bases)
    outcomes = np.asarray(bob_outcomes, dtype=int)
    rules_out_z = (bases == "+") & (outcomes != np.asarray(announced_z))
    rules_out_x = (bases == "x") & (outcomes != np.asarray(announced_x))
    mask = rules_out_z | rules_out_x
    bits = np.where(bases == "+", 1, 0)
    return mask, bits


def pair_labels(announced_z: np.ndarray, announced_x: np.ndarray) -> list[str]:
    """Human-readable announced pair, e.g. ``'|0> or |+>'``."""
    return [
        f"{STATE_LABELS[('+', int(z))]} or {STATE_LABELS[('x', int(x))]}"
        for z, x in zip(announced_z, announced_x, strict=True)
    ]


def run_sarg04(
    num_bits: int,
    *,
    with_eve: bool = False,
    noise: float = 0.0,
    seed: int | None = None,
    simulator: AerSimulator | None = None,
) -> pd.DataFrame:
    """Simulate ``num_bits`` rounds of SARG04 and return the transcript.

    Parameters are the same as :func:`qkd_sim.bb84.run_bb84`. The transcript
    gains three columns: ``alice_prep_bit`` (index of the prepared state in
    its basis), ``announced_z`` and ``announced_x`` (the announced pair), plus
    the label ``announced_pair``.
    """
    rng = np.random.default_rng(seed)
    simulator = simulator if simulator is not None else make_simulator(noise)

    alice_bases = rng.choice(BASES, num_bits)
    alice_prep_bits = rng.integers(0, 2, num_bits)
    alice_bits = key_bits(alice_bases)
    bob_bases = rng.choice(BASES, num_bits)
    eve_bases = rng.choice(BASES, num_bits) if with_eve else None

    bob_outcomes, eve_outcomes = transmit(
        alice_prep_bits,
        alice_bases,
        bob_bases,
        eve_bases=eve_bases,
        simulator=simulator,
        rng=rng,
    )
    announced_z, announced_x = announce(alice_bases, alice_prep_bits, rng)
    sifted, bob_bits = sift(bob_bases, bob_outcomes, announced_z, announced_x)

    eve_guess = None
    if with_eve:
        # Eve hears the announcement too and applies the same rule to her own outcome.
        eve_conclusive, eve_bits = sift(eve_bases, eve_outcomes, announced_z, announced_x)
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
        extra={
            "alice_prep_bit": alice_prep_bits,
            "announced_z": announced_z,
            "announced_x": announced_x,
            "announced_pair": pair_labels(announced_z, announced_x),
        },
    )
