import numpy as np
import pytest

from qkd_sim.keyrate import h2, qber_threshold, secret_key_rate


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
    assert all(a >= b for a, b in zip(rates[:-1], rates[1:], strict=True))
