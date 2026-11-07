"""E91 (Ekert, 1991).

The first entanglement-based protocol. A source emits pairs in the Bell state
|Phi+> = (|00> + |11>)/sqrt2 and sends one qubit to Alice and one to Bob. Each
party measures along an angle in the X-Z plane of the Bloch sphere, i.e. the
observable ``cos(theta) Z + sin(theta) X``:

    Alice:  0, pi/4, pi/2          Bob:  pi/4, pi/2, 3pi/4

On |Phi+> the correlator of two such measurements is ``E(a, b) = cos(a - b)``.
After the quantum phase both announce their settings and every round takes
one of three roles:

* ``a == b`` (2 rounds in 9): outcomes agree with certainty. These are the key,
  and Alice's key bit is simply her own outcome.
* ``a in {0, pi/2}`` and ``b in {pi/4, 3pi/4}`` (4 in 9): fed into the CHSH quantity

      S = E(0, pi/4) - E(0, 3pi/4) + E(pi/2, pi/4) + E(pi/2, 3pi/4) = 2 sqrt2,

  the maximal quantum violation of the local-realist bound ``|S| <= 2``.
  Anything that learns about the key disturbs the entanglement and drags S
  towards 2, so S is the security witness that E91 adds on top of the QBER.
* the remaining 3 in 9 are discarded.

The eavesdropper intercepts Bob's qubit, measures it in a random Z or X basis
and re-sends the eigenstate she found, the same intercept-and-resend attack as
in the prepare-and-measure protocols. Her measurement collapses Alice's qubit
as well, so Alice and Bob end up sharing a product state: every correlator is
halved, ``E(a, b) = cos(a - b)/2``, the CHSH value drops to ``sqrt2`` and the
key rounds show a 25% error rate, matching BB84's signature.

The two key settings pi/4 and pi/2 are *not* conjugate bases (their eigenstates
overlap with probability cos^2(pi/8) ~ 0.85), so the Shor-Preskill key rate is
only a heuristic for E91. The device-independent bound of Pironio et al.
(2009), a function of S and the QBER alone, is the rigorous statement and is
reported next to it (:func:`qkd_sim.keyrate.di_key_rate`).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .channel import make_simulator
from .engine import draw_seed
from .primitives import prepare_qubit, require_values, run_grouped
from .transcript import build_transcript, masked_bits

SETTINGS: tuple[str, ...] = ("0", "pi/4", "pi/2", "3pi/4")
"""Every measurement angle used by anyone, as transcript labels."""

ANGLES: dict[str, float] = {
    "0": 0.0,
    "pi/4": np.pi / 4,
    "pi/2": np.pi / 2,
    "3pi/4": 3 * np.pi / 4,
}
"""Bloch-sphere angle from the Z axis towards X for each setting label."""

ALICE_SETTINGS: tuple[str, ...] = ("0", "pi/4", "pi/2")
BOB_SETTINGS: tuple[str, ...] = ("pi/4", "pi/2", "3pi/4")
EVE_SETTINGS: tuple[str, ...] = ("0", "pi/2")
"""Eve measures Z or X, the same choice as the prepare-and-measure attacker."""

EVE_PREP_BASIS: dict[str, str] = {"0": "+", "pi/2": "x"}
"""The :mod:`qkd_sim.primitives` basis Eve re-prepares her outcome in."""

CHSH_TERMS: dict[tuple[str, str], int] = {
    ("0", "pi/4"): +1,
    ("0", "3pi/4"): -1,
    ("pi/2", "pi/4"): +1,
    ("pi/2", "3pi/4"): +1,
}
"""Sign of each correlator in ``S``; the keys are the (Alice, Bob) setting pairs."""

LOCAL_BOUND = 2.0
"""Largest |S| any local-realist model can produce."""

QUANTUM_BOUND = 2 * np.sqrt(2)
"""Tsirelson's bound, reached by |Phi+> with the settings above."""


def state_label(setting: str, bit: int) -> str:
    """Ket label of the eigenstate a party projected onto, e.g. ``'|pi/4->'``."""
    return f"|{setting}{'+' if int(bit) == 0 else '-'}>"


def bell_pair(*, channel: bool = True) -> QuantumCircuit:
    """Two-qubit circuit preparing |Phi+>, with a channel hook on each qubit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    if channel:
        qc.id(0)
        qc.id(1)
    return qc


def measure_along(
    qc: QuantumCircuit, setting: str, *, qubit: int = 0, clbit: int = 0
) -> QuantumCircuit:
    """Return a *copy* of ``qc`` with ``qubit`` measured along ``setting`` into ``clbit``.

    Measuring ``cos(theta) Z + sin(theta) X`` is a rotation ``Ry(-theta)`` followed
    by a Z measurement: the +1 eigenstate maps to |0> (bit 0) and the -1
    eigenstate to |1> (bit 1). For ``pi/2`` this reproduces the X-basis
    measurement of :func:`qkd_sim.primitives.add_measurement`.
    """
    if setting not in ANGLES:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    measured = qc.copy()
    angle = ANGLES[setting]
    if angle:
        measured.ry(-angle, qubit)
    measured.measure(qubit, clbit)
    return measured


def sample_pair_outcomes(
    alice_settings: Sequence[str] | np.ndarray,
    partner_settings: Sequence[str] | np.ndarray,
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure fresh Bell pairs along each round's Alice and partner settings.

    Qubit 0 is measured along ``alice_settings[i]`` and qubit 1 along
    ``partner_settings[i]``. Rounds are grouped by their pair of settings and
    each distinct two-qubit circuit runs once through
    :func:`qkd_sim.primitives.run_grouped`. Returns ``(alice, partner)``
    outcome arrays. Raises ``ValueError`` for a setting outside :data:`SETTINGS`.
    """
    alice_settings = np.asarray(alice_settings)
    partner_settings = np.asarray(partner_settings)
    if len(alice_settings) != len(partner_settings):
        raise ValueError("alice_settings and partner_settings must have the same length")
    require_values(alice_settings, SETTINGS, "alice_settings")
    require_values(partner_settings, SETTINGS, "partner_settings")

    alice = np.empty(len(alice_settings), dtype=int)
    partner = np.empty(len(alice_settings), dtype=int)
    groups = (
        (
            np.flatnonzero((alice_settings == a) & (partner_settings == b)),
            measure_along(measure_along(bell_pair(), a, qubit=0, clbit=0), b, qubit=1, clbit=1),
        )
        for a, b in product(SETTINGS, SETTINGS)
    )
    for idx, memory in run_grouped(groups, simulator, seed=seed):
        # Aer writes bitstrings with clbit 1 first: "<partner><alice>".
        alice[idx] = np.fromiter((int(s[-1]) for s in memory), dtype=int, count=idx.size)
        partner[idx] = np.fromiter((int(s[-2]) for s in memory), dtype=int, count=idx.size)
    return alice, partner


def sample_resent_outcomes(
    eve_outcomes: Sequence[int] | np.ndarray,
    eve_settings: Sequence[str] | np.ndarray,
    bob_settings: Sequence[str] | np.ndarray,
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Bob measures the eigenstate Eve re-sent (her outcome, in her basis) along his setting.

    Raises ``ValueError`` for an outcome other than 0/1, an Eve setting outside
    :data:`EVE_SETTINGS` or a Bob setting outside :data:`SETTINGS`.
    """
    eve_outcomes = np.asarray(eve_outcomes, dtype=int)
    eve_settings = np.asarray(eve_settings)
    bob_settings = np.asarray(bob_settings)
    if not (len(eve_outcomes) == len(eve_settings) == len(bob_settings)):
        raise ValueError("eve_outcomes, eve_settings and bob_settings must have the same length")
    require_values(eve_outcomes, (0, 1), "eve_outcomes")
    require_values(eve_settings, EVE_SETTINGS, "eve_settings")
    require_values(bob_settings, SETTINGS, "bob_settings")

    outcomes = np.empty(len(eve_outcomes), dtype=int)
    groups = (
        (
            np.flatnonzero((eve_outcomes == bit) & (eve_settings == e) & (bob_settings == b)),
            measure_along(prepare_qubit(bit, EVE_PREP_BASIS[e]), b),
        )
        for bit, e, b in product((0, 1), EVE_SETTINGS, SETTINGS)
    )
    for idx, memory in run_grouped(groups, simulator, seed=seed):
        outcomes[idx] = np.fromiter(map(int, memory), dtype=int, count=idx.size)
    return outcomes


def sift(alice_settings: np.ndarray, bob_settings: np.ndarray) -> np.ndarray:
    """Boolean mask of key rounds: Alice and Bob chose the same setting."""
    return np.asarray(alice_settings) == np.asarray(bob_settings)


def round_use(alice_settings: np.ndarray, bob_settings: np.ndarray) -> np.ndarray:
    """Role of each round once settings are announced: ``'key'``, ``'chsh'`` or ``'discard'``."""
    a = np.asarray(alice_settings)
    b = np.asarray(bob_settings)
    is_chsh = np.zeros(len(a), dtype=bool)
    for sa, sb in CHSH_TERMS:
        is_chsh |= (a == sa) & (b == sb)
    return np.select([a == b, is_chsh], ["key", "chsh"], default="discard").astype(str)


def has_chsh(df: pd.DataFrame) -> bool:
    """True if the transcript carries Bell-test rounds (i.e. came from E91)."""
    return "round_use" in df.columns


def correlation(df: pd.DataFrame, alice_setting: str, bob_setting: str) -> float:
    """Empirical ``E(a, b) = P(agree) - P(disagree)`` over the rounds with those settings."""
    rows = df[(df["alice_basis"] == alice_setting) & (df["bob_basis"] == bob_setting)]
    if len(rows) == 0:
        return float("nan")
    agree = rows["alice_bit"].to_numpy() == rows["bob_outcome"].to_numpy()
    return float(2.0 * agree.mean() - 1.0)


def chsh(df: pd.DataFrame) -> float:
    """The CHSH value ``S`` of a transcript; ``nan`` if any of its four setting pairs is absent.

    Equivalent to summing :func:`correlation` over :data:`CHSH_TERMS`, but
    computed from a single pass over the transcript.
    """
    agree = pd.Series(df["alice_bit"].to_numpy() == df["bob_outcome"].to_numpy())
    keys = [df["alice_basis"].to_numpy(), df["bob_basis"].to_numpy()]
    p_agree = agree.groupby(keys).mean()
    total = 0.0
    for pair, sign in CHSH_TERMS.items():
        if pair not in p_agree.index:
            return float("nan")
        total += sign * (2.0 * float(p_agree[pair]) - 1.0)
    return total


def run_e91(
    num_bits: int,
    *,
    with_eve: bool = False,
    noise: float = 0.0,
    seed: int | None = None,
    simulator: AerSimulator | None = None,
) -> pd.DataFrame:
    """Simulate ``num_bits`` entangled pairs of E91 and return the transcript.

    Parameters are the same as :func:`qkd_sim.bb84.run_bb84`; ``num_bits`` counts
    pairs emitted by the source. Basis columns hold the setting labels in
    :data:`SETTINGS` rather than ``'+'``/``'x'``, ``alice_bit`` is Alice's outcome,
    and the transcript gains a ``round_use`` column.
    """
    rng = np.random.default_rng(seed)
    simulator = simulator if simulator is not None else make_simulator(noise)

    alice_settings = rng.choice(ALICE_SETTINGS, num_bits)
    bob_settings = rng.choice(BOB_SETTINGS, num_bits)

    eve_settings = eve_outcomes = eve_guess = None
    if with_eve:
        eve_settings = rng.choice(EVE_SETTINGS, num_bits)
        alice_outcomes, eve_outcomes = sample_pair_outcomes(
            alice_settings, eve_settings, simulator, seed=draw_seed(rng)
        )
        bob_outcomes = sample_resent_outcomes(
            eve_outcomes, eve_settings, bob_settings, simulator, seed=draw_seed(rng)
        )
        eve_guess = masked_bits(eve_outcomes, np.ones(num_bits, dtype=bool))
    else:
        alice_outcomes, bob_outcomes = sample_pair_outcomes(
            alice_settings, bob_settings, simulator, seed=draw_seed(rng)
        )

    sifted = sift(alice_settings, bob_settings)
    return build_transcript(
        alice_bits=alice_outcomes,
        alice_bases=alice_settings,
        bob_bases=bob_settings,
        bob_outcomes=bob_outcomes,
        sifted=sifted,
        bob_bits=bob_outcomes,
        eve_bases=eve_settings,
        eve_outcomes=eve_outcomes,
        eve_guess=eve_guess,
        bases=SETTINGS,
        label=state_label,
        extra={"round_use": round_use(alice_settings, bob_settings)},
    )
