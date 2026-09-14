import numpy as np
import pytest

from qkd_sim import sarg04
from qkd_sim.metrics import eve_agreement, key_retention, qber
from qkd_sim.theory import expected
from qkd_sim.transcript import validate_transcript


def test_sift_rule_with_pair_zero_plus():
    # Announced pair {|0>, |+>}: announced_z = 0, announced_x = 0.
    z = np.zeros(4, dtype=int)
    x = np.zeros(4, dtype=int)
    bases = np.array(["x", "+", "+", "x"])
    outcomes = np.array([1, 1, 0, 0])
    mask, bits = sarg04.sift(bases, outcomes, z, x)
    # |-> rules out |+> -> Alice sent |0> -> bit 0; |1> rules out |0> -> |+> -> bit 1.
    assert mask.tolist() == [True, True, False, False]
    assert bits[mask].tolist() == [0, 1]


def test_key_bit_lives_in_the_basis(run):
    df = run("sarg04")
    validate_transcript(df)
    assert (df["alice_bit"].to_numpy() == (df["alice_basis"] == "x").astype(int).to_numpy()).all()


def test_announcement_contains_the_sent_state(run):
    df = run("sarg04")
    z_rounds = df["alice_basis"] == "+"
    assert (df.loc[z_rounds, "announced_z"] == df.loc[z_rounds, "alice_prep_bit"]).all()
    assert (df.loc[~z_rounds, "announced_x"] == df.loc[~z_rounds, "alice_prep_bit"]).all()
    assert df["announced_pair"].str.match(r"\|[01]> or \|[+-]>").all()


def test_ideal_channel(run):
    df = run("sarg04")
    assert abs(key_retention(df) - 0.25) < 0.04
    assert qber(df) == 0.0


def test_bit_lives_in_basis_regression(run):
    # Regression for the original prototype, which took the key bit to be
    # the index of the state inside its basis. That convention hides half of
    # Eve's disturbance: the sifted error rate reads ~1/6 instead of 1/3.
    df = run("sarg04", with_eve=True)
    retention, error = expected("sarg04", with_eve=True)
    assert abs(key_retention(df) - retention) < 0.04
    assert abs(qber(df) - error) < 0.06

    kept = df[df["sifted"]]
    old_bob = np.where(kept["bob_basis"] == "+", kept["announced_x"], kept["announced_z"])
    old_qber = np.mean(old_bob != kept["alice_prep_bit"].to_numpy())
    assert abs(old_qber - 1 / 6) < 0.06
    assert old_qber < qber(df)


def test_eve_is_sometimes_inconclusive(run):
    df = run("sarg04", with_eve=True)
    assert df["eve_guess"].isna().any()
    assert 0.0 < eve_agreement(df) < 1.0


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_depolarizing_noise(run, p):
    df = run("sarg04", noise=p)
    retention, error = expected("sarg04", noise=p)
    assert abs(key_retention(df) - retention) < 0.04
    assert abs(qber(df) - error) < 0.06
