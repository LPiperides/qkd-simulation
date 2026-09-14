import numpy as np
import pytest

from qkd_sim.bb84 import sift
from qkd_sim.metrics import eve_agreement, key_retention, qber
from qkd_sim.theory import expected
from qkd_sim.transcript import EVE_COLUMNS, validate_transcript


def test_sift_rule():
    alice = np.array(["+", "+", "x", "x"])
    bob = np.array(["+", "x", "x", "+"])
    assert sift(alice, bob).tolist() == [True, False, True, False]


def test_transcript_schema(run):
    clean = run("bb84")
    validate_transcript(clean)
    assert not any(c in clean.columns for c in EVE_COLUMNS)

    tapped = run("bb84", with_eve=True)
    validate_transcript(tapped)
    assert all(c in tapped.columns for c in EVE_COLUMNS)
    assert tapped["eve_guess"].notna().all()


def test_retention_is_one_half(run):
    for with_eve in (False, True):
        assert abs(key_retention(run("bb84", with_eve=with_eve)) - 0.5) < 0.04


def test_no_eve_is_error_free(run):
    df = run("bb84")
    assert qber(df) == 0.0
    sifted = df[df["sifted"]]
    assert (sifted["bob_bit"].astype(int) == sifted["alice_bit"]).all()


def test_intercept_resend_signature(run):
    df = run("bb84", with_eve=True)
    assert abs(qber(df) - 0.25) < 0.05
    assert abs(eve_agreement(df) - 0.75) < 0.05


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_qber_tracks_depolarizing_noise(run, p):
    df = run("bb84", noise=p)
    retention, error = expected("bb84", noise=p)
    assert abs(key_retention(df) - retention) < 0.04
    assert abs(qber(df) - error) < 0.04
