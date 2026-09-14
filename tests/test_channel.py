import numpy as np
import pytest
from qiskit_aer.noise import NoiseModel

from qkd_sim.channel import depolarizing_noise_model, make_simulator
from qkd_sim.primitives import add_measurement, prepare_qubit, run_batch, sample_outcomes


def test_zero_noise_has_no_model():
    assert depolarizing_noise_model(0.0) is None


def test_noise_model_targets_only_the_channel_gate():
    model = depolarizing_noise_model(0.1)
    assert isinstance(model, NoiseModel)
    assert set(model.noise_instructions) == {"id"}


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_noise_model_rejects_out_of_range(p):
    with pytest.raises(ValueError):
        depolarizing_noise_model(p)
    with pytest.raises(ValueError):
        make_simulator(p)


def test_depolarizing_flip_rate_is_half_p():
    # E(rho) = (1-p) rho + p I/2 flips a basis state with probability p/2.
    simulator = make_simulator(0.4)
    n = 4000
    outcomes = sample_outcomes(
        np.zeros(n, dtype=int), np.full(n, "+"), np.full(n, "+"), simulator, seed=5
    )
    assert abs(outcomes.mean() - 0.2) < 0.03  # sigma ~ 0.006


def test_gateless_state_is_still_noisy():
    # |0> in the '+' basis has no preparation gates; the identity hook is what
    # exposes it to the channel. If Aer ever optimised that gate away this
    # test would fail and the QBER numbers would silently go wrong.
    simulator = make_simulator(0.4)
    circuits = [add_measurement(prepare_qubit(0, "+"), "+")] * 1000
    outcomes = run_batch(circuits, simulator, seed=9)
    assert outcomes.mean() > 0.1


def test_conjugate_basis_is_uniform_regardless_of_noise():
    simulator = make_simulator(0.3)
    n = 4000
    outcomes = sample_outcomes(
        np.zeros(n, dtype=int), np.full(n, "x"), np.full(n, "+"), simulator, seed=2
    )
    assert abs(outcomes.mean() - 0.5) < 0.04
