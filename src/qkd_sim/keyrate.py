"""Asymptotic secret-key-rate estimates.

After sifting, Alice and Bob hold correlated bit strings with error rate
``Q`` (the QBER). Two classical post-processing steps turn them into a secret
key:

* **error correction** leaks at least ``h2(Q)`` bits per sifted bit to an
  eavesdropper listening to the public channel; a practical code leaks
  ``f_ec * h2(Q)`` with an inefficiency ``f_ec >= 1`` (about 1.16 for
  Cascade).
* **privacy amplification** removes the information Eve may hold about the
  raw key, which for BB84 against the optimal collective attack is again
  ``h2(Q)`` bits per sifted bit (Shor & Preskill, 2000).

The asymptotic key rate per *sent* qubit is therefore

    r = retention * max(0, 1 - f_ec * h2(Q) - h2(Q)),

which drops to zero at ``Q ~ 11.0%`` for perfect error correction. That
bound is a theorem for BB84. This package applies the same expression to B92,
SARG04 and E91 for comparison, but their true single-photon thresholds differ
and the numbers should be read as a heuristic, not a proof.

For E91 there is a second, rigorous bound that uses the CHSH value ``S``
instead of assuming anything about the devices (Pironio et al., 2009):

    r = retention * max(0, 1 - f_ec * h2(Q) - h2((1 + sqrt(S^2/4 - 1)) / 2)),

secure against collective attacks. It is zero whenever ``S <= 2``, since a
local-realist correlation could have been produced by a pre-shared classical
key that Eve also holds.
"""

from __future__ import annotations

import numpy as np


def h2(q):
    """Binary entropy in bits, vectorised. ``h2(0) = h2(1) = 0``, ``h2(1/2) = 1``."""
    p = np.asarray(q, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("binary entropy is defined on [0, 1]")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    out = np.where((p == 0) | (p == 1), 0.0, terms)
    return float(out) if out.ndim == 0 else out


def secret_key_rate(qber: float, retention: float = 1.0, f_ec: float = 1.0) -> float:
    """Shor-Preskill asymptotic key rate per sent qubit; never negative."""
    if f_ec < 1.0:
        raise ValueError("error-correction inefficiency f_ec must be >= 1")
    if np.isnan(qber):
        return float("nan")
    return float(retention * max(0.0, 1.0 - (1.0 + f_ec) * h2(qber)))


def di_key_rate(chsh: float, qber: float, retention: float = 1.0, f_ec: float = 1.0) -> float:
    """Device-independent asymptotic key rate per sent pair from the CHSH value and QBER.

    The Pironio et al. (2009) bound against collective attacks; zero when
    ``|S| <= 2``. Sampling noise can push an empirical ``S`` past Tsirelson's
    bound ``2 sqrt2``, where the formula is undefined, so ``S`` is clipped there.
    """
    if f_ec < 1.0:
        raise ValueError("error-correction inefficiency f_ec must be >= 1")
    if np.isnan(chsh) or np.isnan(qber):
        return float("nan")
    s = min(abs(chsh), 2 * np.sqrt(2))
    if s <= 2.0:
        return 0.0
    eve_information = h2((1.0 + np.sqrt(s * s / 4.0 - 1.0)) / 2.0)
    return float(retention * max(0.0, 1.0 - f_ec * h2(qber) - eve_information))


def qber_threshold(f_ec: float = 1.0, tol: float = 1e-6) -> float:
    """QBER above which :func:`secret_key_rate` is zero, found by bisection."""
    if f_ec < 1.0:
        raise ValueError("error-correction inefficiency f_ec must be >= 1")

    def excess(q: float) -> float:
        return 1.0 - (1.0 + f_ec) * h2(q)

    lo, hi = 0.0, 0.5  # excess(0) = 1 > 0, excess(0.5) = -f_ec < 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
