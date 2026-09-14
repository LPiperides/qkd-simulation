"""Transmission engine: Alice -> (Eve) -> Bob in at most two simulator passes.

Every prepare-and-measure protocol follows the same physical script (E91,
which starts from an entangled pair, has its own two-stage version in
:mod:`qkd_sim.e91`). Alice prepares one of four states per round. Optionally
Eve intercepts each qubit, measures it in a basis of her choosing and re-sends
the state corresponding to her outcome (an *intercept-and-resend* attack). Bob
then measures in his own basis.

Eve's re-sent state depends on her measurement outcome, so her hop has to be
simulated before Bob's can be built. That gives a natural two-stage
structure, each stage being one call to :func:`qkd_sim.primitives.sample_outcomes`:

    stage 1 (only with Eve): measure Alice's states in Eve's bases
    stage 2:                 measure the states reaching Bob in Bob's bases

The two stages must not share a simulator seed. Aer derives per-shot
randomness from ``seed_simulator``; reusing one seed for both hops would make
shot ``i`` of Eve's measurement and shot ``i`` of Bob's measurement consume
correlated random streams and bias the error statistics. Each stage therefore
draws a fresh seed from the caller's numpy ``Generator``, which keeps the whole
run reproducible from a single user-facing seed.
"""

from __future__ import annotations

import numpy as np
from qiskit_aer import AerSimulator

from .primitives import sample_outcomes

_SEED_MAX = 2**31 - 16  # leaves headroom for the +k offset used per circuit group


def draw_seed(rng: np.random.Generator) -> int:
    """Draw a simulator seed from ``rng``."""
    return int(rng.integers(0, _SEED_MAX))


def transmit(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    *,
    eve_bases: np.ndarray | None = None,
    simulator: AerSimulator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Send Alice's states through the channel and return the measured bits.

    Parameters
    ----------
    alice_bits, alice_bases:
        The bit and basis Alice uses to prepare each round's qubit.
    bob_bases:
        The basis Bob measures each round in.
    eve_bases:
        If given, Eve intercepts every round, measures in these bases and
        re-sends her outcome. ``None`` means no eavesdropper.
    simulator:
        The (possibly noisy) ``AerSimulator`` representing the channel. When
        Eve is present the channel noise is applied on both hops.
    rng:
        Source of the per-stage simulator seeds.

    Returns
    -------
    (bob_outcomes, eve_outcomes)
        Raw measurement results, one int per round. ``eve_outcomes`` is
        ``None`` when there is no eavesdropper.
    """
    alice_bits = np.asarray(alice_bits, dtype=int)
    alice_bases = np.asarray(alice_bases)
    bob_bases = np.asarray(bob_bases)

    eve_outcomes: np.ndarray | None = None
    if eve_bases is None:
        source_bits, source_bases = alice_bits, alice_bases
    else:
        eve_bases = np.asarray(eve_bases)
        eve_outcomes = sample_outcomes(
            alice_bits, alice_bases, eve_bases, simulator, seed=draw_seed(rng)
        )
        # Eve re-prepares whatever she measured, in the basis she measured in.
        source_bits, source_bases = eve_outcomes, eve_bases

    bob_outcomes = sample_outcomes(
        source_bits, source_bases, bob_bases, simulator, seed=draw_seed(rng)
    )
    return bob_outcomes, eve_outcomes
