import pandas as pd
import pytest

from qkd_sim import PROTOCOLS, simulate


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_same_seed_same_transcript(protocol):
    first = simulate(protocol, 500, with_eve=True, noise=0.05, seed=42)
    second = simulate(protocol, 500, with_eve=True, noise=0.05, seed=42)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_different_seed_different_transcript(protocol):
    first = simulate(protocol, 500, with_eve=True, seed=42)
    other = simulate(protocol, 500, with_eve=True, seed=43)
    assert not first.equals(other)


def test_unseeded_runs(run):
    df = simulate("bb84", 50)
    assert len(df) == 50


def test_unknown_protocol():
    with pytest.raises(ValueError):
        simulate("e91", 10)
