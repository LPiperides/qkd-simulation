import math

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qkd_sim import e91
from qkd_sim.keyrate import di_key_rate
from qkd_sim.metrics import eve_agreement, key_retention, qber, summarize
from qkd_sim.theory import expected, expected_chsh
from qkd_sim.transcript import EVE_COLUMNS, validate_transcript


def test_bell_pair_is_phi_plus():
    amplitudes = Statevector(e91.bell_pair(channel=False)).data
    assert np.allclose(amplitudes, np.array([1, 0, 0, 1]) / np.sqrt(2))
    assert e91.bell_pair().count_ops()["id"] == 2


def test_measure_along_copies_and_rotates():
    qc = e91.bell_pair()
    before = dict(qc.count_ops())
    measured = e91.measure_along(qc, "pi/4", qubit=1, clbit=1)
    assert dict(qc.count_ops()) == before
    assert measured.count_ops()["ry"] == 1 and measured.count_ops()["measure"] == 1
    assert "ry" not in e91.measure_along(qc, "0").count_ops()
    with pytest.raises(ValueError):
        e91.measure_along(qc, "pi")


def test_measure_along_pi_over_2_is_the_x_basis(ideal_simulator):
    # |+> is the +1 eigenstate of X, so measuring along pi/2 must return bit 0 every time.
    plus = QuantumCircuit(1, 1)
    plus.h(0)
    result = ideal_simulator.run(e91.measure_along(plus, "pi/2"), shots=64, memory=True).result()
    assert set(result.get_memory()) == {"0"}


def test_key_settings_are_not_conjugate():
    # Documents why Shor-Preskill is only a heuristic for E91.
    def eigenstate(setting: str) -> np.ndarray:
        theta = e91.ANGLES[setting]
        return np.array([math.cos(theta / 2), math.sin(theta / 2)])

    overlap = abs(eigenstate("pi/4") @ eigenstate("pi/2")) ** 2
    assert overlap == pytest.approx(math.cos(math.pi / 8) ** 2)
    assert overlap > 0.5


def test_round_use_and_sift():
    alice = np.array(["0", "pi/4", "pi/2", "0", "pi/4"])
    bob = np.array(["pi/4", "pi/4", "3pi/4", "pi/2", "3pi/4"])
    assert e91.round_use(alice, bob).tolist() == ["chsh", "key", "chsh", "discard", "discard"]
    assert e91.sift(alice, bob).tolist() == [False, True, False, False, False]


def test_chsh_on_hand_built_rounds():
    # Perfect correlation on the three +1 terms and perfect anticorrelation on
    # the -1 term gives the algebraic maximum S = 4.
    pairs = list(e91.CHSH_TERMS)
    df = pd.DataFrame(
        {
            "alice_basis": [a for a, _ in pairs],
            "bob_basis": [b for _, b in pairs],
            "alice_bit": [0, 0, 0, 0],
            "bob_outcome": [0, 1, 0, 0],
        }
    )
    assert e91.chsh(df) == 4.0
    assert math.isnan(e91.chsh(df.iloc[:3]))
    assert e91.correlation(df, "0", "3pi/4") == -1.0


def test_transcript_schema(run):
    clean = run("e91")
    validate_transcript(clean, bases=e91.SETTINGS)
    assert set(clean["alice_basis"]) == set(e91.ALICE_SETTINGS)
    assert set(clean["bob_basis"]) == set(e91.BOB_SETTINGS)
    assert set(clean["round_use"]) == {"key", "chsh", "discard"}
    assert clean["alice_state"].str.match(r"\|(0|pi/4|pi/2)[+-]>$").all()
    assert (clean["sifted"] == (clean["round_use"] == "key")).all()
    assert not any(c in clean.columns for c in EVE_COLUMNS)

    tapped = run("e91", with_eve=True)
    validate_transcript(tapped, bases=e91.SETTINGS)
    assert set(tapped["eve_basis"]) == set(e91.EVE_SETTINGS)
    assert tapped["eve_guess"].notna().all()


def test_default_alphabet_rejects_e91_settings(run):
    with pytest.raises(ValueError):
        validate_transcript(run("e91"))


def test_ideal_channel_violates_chsh(run):
    df = run("e91")
    retention, _ = expected("e91")
    assert abs(key_retention(df) - retention) < 0.03
    assert qber(df) == 0.0
    s = e91.chsh(df)
    assert abs(s - expected_chsh()) < 0.4
    assert s > e91.LOCAL_BOUND


def test_intercept_resend_signature(run):
    df = run("e91", with_eve=True)
    retention, error = expected("e91", with_eve=True)
    assert error == 0.25
    assert abs(key_retention(df) - retention) < 0.03
    assert abs(qber(df) - error) < 0.06
    s = e91.chsh(df)
    assert abs(s - expected_chsh(with_eve=True)) < 0.4
    assert s < e91.LOCAL_BOUND
    # Eve agrees with Alice on 3/4 of the pi/2 key rounds and cos^2(pi/8) of the pi/4 ones.
    assert abs(eve_agreement(df) - (5 + math.sqrt(2)) / 8) < 0.06


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_depolarizing_noise(run, p):
    df = run("e91", noise=p)
    retention, error = expected("e91", noise=p)
    assert abs(key_retention(df) - retention) < 0.03
    assert abs(qber(df) - error) < 0.06
    assert abs(e91.chsh(df) - expected_chsh(noise=p)) < 0.4


def test_noise_and_eve_have_a_closed_form(run):
    df = run("e91", with_eve=True, noise=0.1)
    _, error = expected("e91", with_eve=True, noise=0.1)
    assert error == pytest.approx((1 - 0.9**3 / 2) / 2)
    assert abs(qber(df) - error) < 0.07
    assert abs(e91.chsh(df) - expected_chsh(with_eve=True, noise=0.1)) < 0.4


def test_summary_carries_chsh_and_di_rate(run):
    s = summarize(run("e91"), protocol="e91")
    assert s.chsh > e91.LOCAL_BOUND
    assert s.di_key_rate > 0
    assert s.di_key_rate == di_key_rate(s.chsh, s.qber, s.retention, 1.0)

    tapped = summarize(run("e91", with_eve=True), protocol="e91")
    assert tapped.di_key_rate == 0.0 and tapped.key_rate == 0.0

    other = summarize(run("bb84"), protocol="bb84")
    assert math.isnan(other.chsh) and math.isnan(other.di_key_rate)
