"""Analytic expectations used as test oracles and figure overlays.

For a depolarizing channel of strength ``p`` (see :mod:`qkd_sim.channel`)
the sifting rate and quantum bit error rate of each protocol are

    BB84:           retention = 1/2,        QBER = p/2
    B92 / SARG04:   retention = (1 + p)/4,  QBER = p/(1 + p)
    E91:            retention = 2/9,        QBER = (1 - V)/2,   CHSH S = 2 sqrt2 V

where ``V = (1 - p)^2`` is the visibility of the entangled pair: each qubit
suffers one depolarizing event on its way from the source, and every traceless
observable on either side loses a factor ``1 - p``.

An intercept-and-resend eavesdropper on an ideal channel produces exactly the
same statistics as a depolarizing channel with ``p = 1/2`` in the three
prepare-and-measure protocols: half the time Eve picks the right basis and
forwards the state untouched, the other half she forwards a conjugate-basis
eigenstate, which looks maximally mixed in Alice's basis. In E91 she measures
Bob's qubit in Z or X, which collapses the pair to a product state and halves
every correlator; the extra hop from Eve to Bob costs another ``1 - p``, so
``V = (1 - p)^3 / 2`` and the closed form survives channel noise. Full
derivations are in ``docs/theory.md``.
"""

from __future__ import annotations

import math

INTERCEPT_RESEND_EQUIVALENT_NOISE = 0.5
"""Depolarizing strength that reproduces intercept-and-resend statistics (prepare-and-measure)."""

E91_RETENTION = 2 / 9
"""Fraction of E91 rounds in which Alice and Bob chose the same setting."""


def _check_noise(noise: float) -> None:
    if not 0.0 <= noise <= 1.0:
        raise ValueError(f"noise must lie in [0, 1], got {noise}")


def e91_visibility(*, with_eve: bool = False, noise: float = 0.0) -> float:
    """Factor multiplying ``cos(a - b)`` in every E91 correlator."""
    _check_noise(noise)
    visibility = (1.0 - noise) ** 2
    if with_eve:
        visibility *= (1.0 - noise) / 2.0
    return visibility


def expected(protocol: str, *, with_eve: bool = False, noise: float = 0.0) -> tuple[float, float]:
    """Return the analytic ``(key_retention, qber)`` for a protocol and setting.

    For the prepare-and-measure protocols, combining an eavesdropper with
    channel noise has no closed form here and raises ``NotImplementedError``.
    """
    _check_noise(noise)
    protocol = protocol.lower()
    if protocol == "e91":
        return E91_RETENTION, (1.0 - e91_visibility(with_eve=with_eve, noise=noise)) / 2.0
    if with_eve and noise > 0:
        raise NotImplementedError("no closed form for Eve plus channel noise")

    p = INTERCEPT_RESEND_EQUIVALENT_NOISE if with_eve else noise
    if protocol == "bb84":
        return 0.5, p / 2
    if protocol in ("b92", "sarg04"):
        return (1 + p) / 4, p / (1 + p)
    raise ValueError(f"unknown protocol {protocol!r}")


def expected_chsh(*, with_eve: bool = False, noise: float = 0.0) -> float:
    """Analytic CHSH value ``S`` of E91: ``2 sqrt2`` on an ideal channel, ``sqrt2`` under attack."""
    return 2.0 * math.sqrt(2.0) * e91_visibility(with_eve=with_eve, noise=noise)
