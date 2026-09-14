"""Protocol-agnostic metrics computed from a transcript.

The transcript already records which rounds survived sifting and what key
bit each party derived (see :mod:`qkd_sim.transcript`), so every metric here
is a plain reduction over columns:

* :func:`key_retention`      fraction of sent qubits that become sifted key
* :func:`qber`               error rate between Alice's and Bob's sifted bits
* :func:`eve_agreement`      fraction of sifted bits Eve guessed correctly
* :func:`mutual_information` empirical Shannon mutual information in bits
* :func:`info_ab`            I(A:B) of a binary symmetric channel with error Q
* :func:`info_ae`            empirical I(A:E) on the sifted key
* :func:`summarize`          all of the above plus the secret-key-rate bound

Intercept-and-resend on BB84 gives the textbook result I(A:B) = I(A:E): Eve
knows exactly as much about the key as Bob does, and no positive key rate is
possible without privacy amplification beyond that point (Csiszár & Körner,
1978).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .keyrate import h2, secret_key_rate
from .transcript import has_eve


def sifted(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that survived sifting."""
    return df[df["sifted"]]


def key_retention(df: pd.DataFrame) -> float:
    """Fraction of transmitted qubits that end up in the sifted key."""
    if len(df) == 0:
        return float("nan")
    return float(df["sifted"].mean())


def qber(df: pd.DataFrame) -> float:
    """Quantum bit error rate: disagreement between Alice and Bob on the sifted key."""
    kept = sifted(df)
    if len(kept) == 0:
        return float("nan")
    return float((kept["alice_bit"].to_numpy() != kept["bob_bit"].to_numpy(dtype=int)).mean())


def eve_agreement(df: pd.DataFrame) -> float:
    """Fraction of the sifted key Eve guessed correctly (NA guesses count as wrong).

    Returns ``nan`` when the transcript has no eavesdropper.
    """
    if not has_eve(df):
        return float("nan")
    kept = sifted(df)
    if len(kept) == 0:
        return float("nan")
    guesses = kept["eve_guess"]
    correct = guesses.notna() & (guesses.fillna(-1).astype(int) == kept["alice_bit"])
    return float(correct.mean())


def _categories(values) -> pd.Series:
    """Coerce an array-like to string categories with missing values as ``'<NA>'``."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.astype("string").fillna("<NA>").reset_index(drop=True)


def mutual_information(x, y) -> float:
    """Plug-in estimate of I(X;Y) in bits from two aligned categorical arrays.

    Missing values are treated as a category of their own, so an inconclusive
    guess still counts as an observation of "no guess".
    """
    xs = _categories(x)
    ys = _categories(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if len(xs) == 0:
        return float("nan")

    joint = np.array(pd.crosstab(xs, ys).to_numpy(), dtype=float)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / (px @ py)[nonzero])))


def info_ab(q: float) -> float:
    """I(A:B) for a binary symmetric channel with crossover probability ``q``."""
    return float(1.0 - h2(q))


def info_ae(df: pd.DataFrame) -> float:
    """Empirical I(A:E) between Alice's sifted key bits and Eve's guesses; 0 without Eve."""
    if not has_eve(df):
        return 0.0
    kept = sifted(df)
    if len(kept) == 0:
        return float("nan")
    return mutual_information(kept["alice_bit"], kept["eve_guess"])


@dataclass(frozen=True)
class Summary:
    """Headline numbers for one simulation run."""

    protocol: str
    num_bits: int
    with_eve: bool
    retention: float
    qber: float
    eve_agreement: float
    i_ab: float
    i_ae: float
    key_rate: float
    f_ec: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(df: pd.DataFrame, *, protocol: str, f_ec: float = 1.0) -> Summary:
    """Compute every metric for a transcript.

    ``key_rate`` is the asymptotic Shor-Preskill bound per sent qubit
    (rigorous for BB84, heuristic for the other protocols; see
    :mod:`qkd_sim.keyrate`).
    """
    retention = key_retention(df)
    error = qber(df)
    return Summary(
        protocol=protocol,
        num_bits=len(df),
        with_eve=has_eve(df),
        retention=retention,
        qber=error,
        eve_agreement=eve_agreement(df),
        i_ab=info_ab(error) if not np.isnan(error) else float("nan"),
        i_ae=info_ae(df),
        key_rate=secret_key_rate(error, retention, f_ec),
        f_ec=f_ec,
    )
