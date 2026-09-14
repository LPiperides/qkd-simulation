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
bound is a theorem for BB84. This package applies the same expression to B92
and SARG04 for comparison, but their true single-photon thresholds differ
and the numbers should be read as a heuristic, not a proof.
"""

from __future__ import annotations

import numpy as np

_LOG2_EPS = 1e-300


def h2(q):
    """Binary entropy in bits, vectorised. ``h2(0) = h2(1) = 0``, ``h2(1/2) = 1``."""
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("binary entropy is defined on [0, 1]")
    p = np.clip(q_arr, _LOG2_EPS, 1 - _LOG2_EPS)
    out = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    out = np.where((q_arr == 0) | (q_arr == 1), 0.0, out)
    return float(out) if out.ndim == 0 else out


def secret_key_rate(qber: float, retention: float = 1.0, f_ec: float = 1.0) -> float:
    """Shor-Preskill asymptotic key rate per sent qubit; never negative."""
    if f_ec < 1.0:
        raise ValueError("error-correction inefficiency f_ec must be >= 1")
    if np.isnan(qber):
        return float("nan")
    return float(retention * max(0.0, 1.0 - (1.0 + f_ec) * h2(qber)))


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
