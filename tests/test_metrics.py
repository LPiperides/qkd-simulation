import math

import numpy as np
import pandas as pd
import pytest

from qkd_sim import simulate
from qkd_sim.keyrate import secret_key_rate
from qkd_sim.metrics import (
    Summary,
    eve_agreement,
    info_ab,
    info_ae,
    key_retention,
    mutual_information,
    qber,
    summarize,
)


def test_mutual_information_of_identical_balanced_bits():
    x = np.array([0, 1] * 500)
    assert abs(mutual_information(x, x) - 1.0) < 1e-9


def test_mutual_information_of_independent_bits_is_small():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 4000)
    y = rng.integers(0, 2, 4000)
    assert mutual_information(x, y) < 0.01


def test_mutual_information_is_symmetric_and_handles_na():
    x = pd.array([0, 1, 0, 1, 0, 1], dtype="Int64")
    y = pd.array([0, 1, pd.NA, 1, 0, pd.NA], dtype="Int64")
    assert abs(mutual_information(x, y) - mutual_information(y, x)) < 1e-12
    assert mutual_information(x, y) > 0.0
    with pytest.raises(ValueError):
        mutual_information([0, 1], [0])


def test_info_ab_endpoints():
    assert info_ab(0.0) == 1.0
    assert info_ab(0.5) == 0.0


def test_intercept_resend_gives_eve_as_much_as_bob():
    # Mutual information is a nonlinear function of the agreement rate, so it
    # magnifies sampling noise; use a larger run than the shared fixture.
    df = simulate("bb84", 40_000, with_eve=True, seed=2024)
    assert abs(info_ae(df) - 0.189) < 0.02
    assert abs(info_ab(qber(df)) - info_ae(df)) < 0.02


def test_metrics_without_eve(run):
    df = run("bb84")
    assert math.isnan(eve_agreement(df))
    assert info_ae(df) == 0.0


def test_summarize_bundle(run):
    df = run("bb84", with_eve=True)
    s = summarize(df, protocol="bb84", f_ec=1.1)
    assert isinstance(s, Summary)
    assert s.num_bits == len(df) and s.with_eve is True
    assert s.retention == key_retention(df)
    assert s.key_rate == secret_key_rate(qber(df), key_retention(df), 1.1)
    assert set(s.as_dict()) >= {"qber", "retention", "key_rate", "i_ab", "i_ae"}


def test_empty_transcript_metrics():
    df = pd.DataFrame({"sifted": pd.Series([], dtype=bool), "alice_bit": [], "bob_bit": []})
    assert math.isnan(key_retention(df))
    assert math.isnan(qber(df))
