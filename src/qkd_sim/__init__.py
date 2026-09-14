"""qkd-sim: BB84, B92 and SARG04 quantum key distribution on Qiskit Aer.

Quick use::

    from qkd_sim import simulate, summarize

    transcript = simulate("bb84", 10_000, with_eve=True, seed=42)
    print(summarize(transcript, protocol="bb84"))
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .b92 import run_b92
from .bb84 import run_bb84
from .channel import depolarizing_noise_model, make_simulator
from .keyrate import h2, qber_threshold, secret_key_rate
from .metrics import (
    Summary,
    eve_agreement,
    info_ab,
    info_ae,
    key_retention,
    mutual_information,
    qber,
    summarize,
)
from .sarg04 import run_sarg04
from .theory import expected
from .transcript import validate_transcript

__version__ = "0.1.0"

PROTOCOLS: dict[str, Callable[..., pd.DataFrame]] = {
    "bb84": run_bb84,
    "b92": run_b92,
    "sarg04": run_sarg04,
}
"""Registry of protocol name -> simulation function."""


def simulate(protocol: str, num_bits: int, **kwargs) -> pd.DataFrame:
    """Run ``protocol`` for ``num_bits`` rounds; see the protocol functions for kwargs."""
    try:
        runner = PROTOCOLS[protocol.lower()]
    except KeyError:
        raise ValueError(
            f"unknown protocol {protocol!r}; choose from {sorted(PROTOCOLS)}"
        ) from None
    return runner(num_bits, **kwargs)


__all__ = [
    "PROTOCOLS",
    "Summary",
    "__version__",
    "depolarizing_noise_model",
    "eve_agreement",
    "expected",
    "h2",
    "info_ab",
    "info_ae",
    "key_retention",
    "make_simulator",
    "mutual_information",
    "qber",
    "qber_threshold",
    "run_b92",
    "run_bb84",
    "run_sarg04",
    "secret_key_rate",
    "simulate",
    "summarize",
    "validate_transcript",
]
