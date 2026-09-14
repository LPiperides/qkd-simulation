"""Shared fixtures.

Statistical tests need a few thousand rounds to be meaningful, so simulations
are cached for the whole session and shared between test modules. Every
cached run is seeded, which makes the suite deterministic; tolerances are
nevertheless set at four or more standard deviations so that a change in
Aer's random stream cannot make a test flap.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qkd_sim import simulate
from qkd_sim.channel import make_simulator

N_ROUNDS = 4000
SEED = 12345
NOISY_SEED = 777


@pytest.fixture(scope="session")
def run():
    """Factory returning cached transcripts: ``run(protocol, with_eve=..., noise=...)``."""
    cache: dict[tuple[str, bool, float], pd.DataFrame] = {}

    def _run(protocol: str, *, with_eve: bool = False, noise: float = 0.0) -> pd.DataFrame:
        key = (protocol, with_eve, noise)
        if key not in cache:
            seed = NOISY_SEED if noise else SEED
            cache[key] = simulate(protocol, N_ROUNDS, with_eve=with_eve, noise=noise, seed=seed)
        return cache[key]

    return _run


@pytest.fixture(scope="session")
def ideal_simulator():
    return make_simulator(0.0)
