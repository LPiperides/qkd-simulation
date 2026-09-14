"""The transcript: one DataFrame row per transmitted qubit.

Every protocol returns the same core schema so that the metrics in
:mod:`qkd_sim.metrics` are protocol-agnostic:

==============  ========  =======================================================
column          dtype     meaning
==============  ========  =======================================================
round           int       0-based round index
alice_basis     str       '+' or 'x': the basis Alice prepared in
alice_bit       int       the *key* bit Alice intends to share this round
alice_state     str       ket label of the prepared state
eve_basis       str       (with Eve) basis Eve measured in
eve_outcome     int       (with Eve) Eve's raw measurement result
eve_state       str       (with Eve) the state Eve re-sent
eve_guess       Int64     (with Eve) Eve's best guess of alice_bit, NA if none
bob_basis       str       basis Bob measured in
bob_outcome     int       Bob's raw measurement result
bob_state       str       ket label of Bob's outcome in his basis
sifted          bool      True if the round survives public discussion
bob_bit         Int64     Bob's key bit, NA unless sifted
==============  ========  =======================================================

Note that ``alice_bit`` is the key bit, which is not always the same thing as
the index of the prepared state within its basis (SARG04 encodes the bit in
the *basis*). Protocol modules may add extra columns after these.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .primitives import BASES, STATE_LABELS

ALICE_COLUMNS = ["round", "alice_basis", "alice_bit", "alice_state"]
EVE_COLUMNS = ["eve_basis", "eve_outcome", "eve_state", "eve_guess"]
BOB_COLUMNS = ["bob_basis", "bob_outcome", "bob_state", "sifted", "bob_bit"]
CORE_COLUMNS = ALICE_COLUMNS + BOB_COLUMNS


def labels(bases: np.ndarray, bits: np.ndarray) -> list[str]:
    """Vectorised :func:`qkd_sim.primitives.state_label`."""
    return [STATE_LABELS[(b, int(v))] for b, v in zip(bases, bits, strict=True)]


def masked_bits(bits: np.ndarray, mask: np.ndarray) -> pd.Series:
    """Nullable-integer series equal to ``bits`` where ``mask`` holds, NA elsewhere."""
    return pd.Series(np.asarray(bits, dtype=int), dtype="Int64").where(np.asarray(mask, dtype=bool))


def has_eve(df: pd.DataFrame) -> bool:
    """True if the transcript was generated with an eavesdropper."""
    return "eve_guess" in df.columns


def build_transcript(
    *,
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    bob_outcomes: np.ndarray,
    sifted: np.ndarray,
    bob_bits: np.ndarray,
    eve_bases: np.ndarray | None = None,
    eve_outcomes: np.ndarray | None = None,
    eve_guess: pd.Series | None = None,
    extra: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Assemble and validate a transcript DataFrame in the canonical column order.

    ``bob_bits`` is Bob's key bit for every round; it is only exposed (as
    ``bob_bit``) where ``sifted`` is True. ``eve_guess`` should already be a
    nullable series (see :func:`masked_bits`).
    """
    n = len(alice_bits)
    columns: dict[str, object] = {
        "round": np.arange(n),
        "alice_basis": np.asarray(alice_bases),
        "alice_bit": np.asarray(alice_bits, dtype=int),
        "alice_state": labels(alice_bases, alice_bits),
    }
    if eve_bases is not None:
        if eve_outcomes is None or eve_guess is None:
            raise ValueError("eve_bases requires eve_outcomes and eve_guess")
        columns.update(
            {
                "eve_basis": np.asarray(eve_bases),
                "eve_outcome": np.asarray(eve_outcomes, dtype=int),
                "eve_state": labels(eve_bases, eve_outcomes),
                "eve_guess": eve_guess.to_numpy(),
            }
        )
    columns.update(
        {
            "bob_basis": np.asarray(bob_bases),
            "bob_outcome": np.asarray(bob_outcomes, dtype=int),
            "bob_state": labels(bob_bases, bob_outcomes),
            "sifted": np.asarray(sifted, dtype=bool),
            "bob_bit": masked_bits(bob_bits, sifted).to_numpy(),
        }
    )
    if extra:
        columns.update(extra)

    df = pd.DataFrame(columns)
    df["bob_bit"] = df["bob_bit"].astype("Int64")
    if eve_bases is not None:
        df["eve_guess"] = df["eve_guess"].astype("Int64")
    return validate_transcript(df)


def validate_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema invariants and return ``df`` unchanged.

    Raises ``ValueError`` on the first violation. Used by every protocol
    before returning and available to callers loading transcripts from CSV.
    """
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transcript is missing columns {missing}")

    eve_present = [c for c in EVE_COLUMNS if c in df.columns]
    if eve_present and len(eve_present) != len(EVE_COLUMNS):
        raise ValueError(f"transcript has a partial eavesdropper block: {eve_present}")

    for col in ("alice_basis", "bob_basis", *(["eve_basis"] if eve_present else [])):
        if not df[col].isin(BASES).all():
            raise ValueError(f"column {col!r} contains values outside {BASES}")

    for col in ("alice_bit", "bob_outcome", *(["eve_outcome"] if eve_present else [])):
        if not df[col].isin((0, 1)).all():
            raise ValueError(f"column {col!r} must contain only 0 and 1")

    if df["sifted"].dtype != bool:
        raise ValueError("column 'sifted' must be boolean")

    bob_bit_defined = df["bob_bit"].notna().to_numpy()
    if not np.array_equal(bob_bit_defined, df["sifted"].to_numpy()):
        raise ValueError("'bob_bit' must be NA exactly where 'sifted' is False")
    if not df["bob_bit"].dropna().isin((0, 1)).all():
        raise ValueError("'bob_bit' must contain only 0, 1 and NA")

    if eve_present and not df["eve_guess"].dropna().isin((0, 1)).all():
        raise ValueError("'eve_guess' must contain only 0, 1 and NA")

    return df
