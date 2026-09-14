"""Analytic expectations used as test oracles and figure overlays.

For a depolarizing channel of strength ``p`` (see :mod:`qkd_sim.channel`)
the sifting rate and quantum bit error rate of each protocol are

    BB84:           retention = 1/2,        QBER = p/2
    B92 / SARG04:   retention = (1 + p)/4,  QBER = p/(1 + p)

An intercept-and-resend eavesdropper on an ideal channel produces exactly the
same statistics as a depolarizing channel with ``p = 1/2``: half the time Eve
picks the right basis and forwards the state untouched, the other half she
forwards a conjugate-basis eigenstate, which looks maximally mixed in Alice's
basis. Full derivations are in ``docs/theory.md``.
"""

from __future__ import annotations

INTERCEPT_RESEND_EQUIVALENT_NOISE = 0.5
"""Depolarizing strength that reproduces intercept-and-resend statistics."""


def expected(protocol: str, *, with_eve: bool = False, noise: float = 0.0) -> tuple[float, float]:
    """Return the analytic ``(key_retention, qber)`` for a protocol and setting.

    Combining an eavesdropper with channel noise has no closed form here and
    raises ``NotImplementedError``.
    """
    if with_eve and noise > 0:
        raise NotImplementedError("no closed form for Eve plus channel noise")
    if not 0.0 <= noise <= 1.0:
        raise ValueError(f"noise must lie in [0, 1], got {noise}")

    p = INTERCEPT_RESEND_EQUIVALENT_NOISE if with_eve else noise
    protocol = protocol.lower()
    if protocol == "bb84":
        return 0.5, p / 2
    if protocol in ("b92", "sarg04"):
        return (1 + p) / 4, p / (1 + p)
    raise ValueError(f"unknown protocol {protocol!r}")
