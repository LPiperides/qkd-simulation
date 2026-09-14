import itertools

import numpy as np
import pytest

from qkd_sim.keyrate import di_key_rate, h2, qber_threshold, secret_key_rate


def test_di_key_rate_endpoints():
    tsirelson = 2 * np.sqrt(2)
    assert di_key_rate(tsirelson, 0.0) == 1.0
    assert di_key_rate(2.0, 0.0) == 0.0
    assert di_key_rate(1.4, 0.0) == 0.0
    assert di_key_rate(2.9, 0.0) == 1.0  # empirical S past Tsirelson is clipped
    assert di_key_rate(tsirelson, 0.5) == 0.0
    assert np.isnan(di_key_rate(float("nan"), 0.1))
    assert di_key_rate(2.5, 0.02, retention=0.5) == pytest.approx(0.5 * di_key_rate(2.5, 0.02))
    with pytest.raises(ValueError):
        di_key_rate(2.5, 0.1, f_ec=0.9)


def test_di_key_rate_is_non_decreasing_in_chsh():
    rates = [di_key_rate(s, 0.02) for s in np.linspace(2.0, 2 * np.sqrt(2), 30)]
    assert all(a <= b for a, b in itertools.pairwise(rates))
    assert rates[-1] > rates[0]


def test_binary_entropy_values():
    assert h2(0.0) == 0.0
    assert h2(1.0) == 0.0
    assert h2(0.5) == 1.0
    assert abs(h2(0.11) - 0.4999) < 1e-3
    vec = h2(np.array([0.0, 0.5, 1.0]))
    assert vec.shape == (3,) and vec.tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        h2(1.1)


def test_threshold_values():
    assert abs(qber_threshold(1.0) - 0.1100) < 5e-4
    assert abs(qber_threshold(1.16) - 0.098) < 2e-3
    assert qber_threshold(1.5) < qber_threshold(1.16) < qber_threshold(1.0)
    with pytest.raises(ValueError):
        qber_threshold(0.9)


def test_secret_key_rate_values():
    assert secret_key_rate(0.0, 0.5) == 0.5
    assert secret_key_rate(0.12) == 0.0
    assert secret_key_rate(0.25) == 0.0
    assert 0.0 < secret_key_rate(0.05) < 1.0
    assert np.isnan(secret_key_rate(float("nan")))
    with pytest.raises(ValueError):
        secret_key_rate(0.05, f_ec=0.5)


def test_secret_key_rate_is_non_increasing_in_qber():
    rates = [secret_key_rate(q) for q in np.linspace(0.0, 0.5, 51)]
    assert all(a >= b for a, b in itertools.pairwise(rates))
