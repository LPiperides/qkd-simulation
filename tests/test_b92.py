import inspect

import numpy as np
import pytest

from qkd_sim import b92
from qkd_sim.metrics import eve_agreement, key_retention, qber
from qkd_sim.theory import expected
from qkd_sim.transcript import validate_transcript


def test_sift_rule_on_hand_built_rounds():
    bases = np.array(["+", "+", "x", "x"])
    outcomes = np.array([0, 1, 0, 1])
    mask, bits = b92.sift(bases, outcomes)
    assert mask.tolist() == [False, True, True, False]
    assert bits[mask].tolist() == [1, 0]


def test_sifting_never_reads_alices_bit(run):
    # The rule must be a function of Bob's data alone: the signature has no
    # Alice argument, and shuffling Alice's bits changes nothing.
    params = inspect.signature(b92.sift).parameters
    assert not any(name.startswith("alice") for name in params)

    df = run("b92", with_eve=True)
    mask, bits = b92.sift(df["bob_basis"].to_numpy(), df["bob_outcome"].to_numpy())
    assert np.array_equal(mask, df["sifted"].to_numpy())
    assert np.array_equal(bits[mask], df.loc[df["sifted"], "bob_bit"].to_numpy(dtype=int))


def test_alice_basis_is_a_function_of_her_bit(run):
    df = run("b92")
    validate_transcript(df)
    expected_bases = np.where(df["alice_bit"] == 0, "+", "x")
    assert (df["alice_basis"].to_numpy() == expected_bases).all()
    assert set(df["alice_state"]) == {"|0>", "|->"}


def test_ideal_channel(run):
    df = run("b92")
    assert abs(key_retention(df) - 0.25) < 0.04
    assert qber(df) == 0.0


def test_intercept_resend_signature(run):
    df = run("b92", with_eve=True)
    retention, error = expected("b92", with_eve=True)
    assert abs(key_retention(df) - retention) < 0.04
    assert abs(qber(df) - error) < 0.06
    assert df["eve_guess"].isna().any()  # Eve is inconclusive too
    assert 0.0 < eve_agreement(df) < 1.0


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_depolarizing_noise(run, p):
    df = run("b92", noise=p)
    retention, error = expected("b92", noise=p)
    assert abs(key_retention(df) - retention) < 0.04
    assert abs(qber(df) - error) < 0.06
