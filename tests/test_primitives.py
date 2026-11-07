import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qkd_sim.primitives import (
    BASES,
    STATE_LABELS,
    add_measurement,
    prepare_qubit,
    run_batch,
    sample_outcomes,
    state_label,
)

INV_SQRT2 = 1 / np.sqrt(2)
EXPECTED_VECTORS = {
    ("+", 0): [1, 0],
    ("+", 1): [0, 1],
    ("x", 0): [INV_SQRT2, INV_SQRT2],
    ("x", 1): [INV_SQRT2, -INV_SQRT2],
}


def gate_names(qc):
    return [instruction.operation.name for instruction in qc.data]


@pytest.mark.parametrize(
    ("bit", "basis", "gates"),
    [
        (0, "+", ["id"]),
        (1, "+", ["x", "id"]),
        (0, "x", ["h", "id"]),
        (1, "x", ["x", "h", "id"]),
    ],
)
def test_prepare_qubit_gate_sequence(bit, basis, gates):
    assert gate_names(prepare_qubit(bit, basis)) == gates
    assert gate_names(prepare_qubit(bit, basis, channel=False)) == gates[:-1]


@pytest.mark.parametrize(("basis", "bit"), list(EXPECTED_VECTORS))
def test_prepare_qubit_statevector(basis, bit):
    state = Statevector(prepare_qubit(bit, basis, channel=False))
    assert state.equiv(Statevector(EXPECTED_VECTORS[(basis, bit)]))


def test_prepare_qubit_rejects_bad_input():
    with pytest.raises(ValueError):
        prepare_qubit(2, "+")
    with pytest.raises(ValueError):
        prepare_qubit(0, "z")


def test_state_label_matches_table():
    for (basis, bit), label in STATE_LABELS.items():
        assert state_label(basis, bit) == label


def test_add_measurement_does_not_mutate_input():
    qc = prepare_qubit(1, "x")
    before = len(qc.data)
    measured = add_measurement(qc, "x")
    assert measured is not qc
    assert len(qc.data) == before
    assert gate_names(measured) == ["x", "h", "id", "h", "measure"]
    assert gate_names(add_measurement(qc, "+")) == ["x", "h", "id", "measure"]


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("basis", BASES)
def test_matching_basis_measurement_is_deterministic(bit, basis, ideal_simulator):
    circuits = [add_measurement(prepare_qubit(bit, basis), basis)] * 50
    outcomes = run_batch(circuits, ideal_simulator, seed=1)
    assert outcomes.shape == (50,)
    assert (outcomes == bit).all()


def test_mismatched_basis_measurement_is_random_within_a_batch(ideal_simulator):
    circuits = [add_measurement(prepare_qubit(0, "x"), "+")] * 400
    outcomes = run_batch(circuits, ideal_simulator, seed=7)
    assert set(outcomes) == {0, 1}
    assert 0.35 < outcomes.mean() < 0.65


def test_run_batch_unseeded_and_empty(ideal_simulator):
    outcomes = run_batch([add_measurement(prepare_qubit(0, "x"), "+")] * 10, ideal_simulator)
    assert outcomes.shape == (10,)
    assert set(outcomes) <= {0, 1}
    assert run_batch([], ideal_simulator).shape == (0,)


def test_sample_outcomes_matches_physics(ideal_simulator):
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 2000)
    bases = rng.choice(BASES, 2000)
    meas = rng.choice(BASES, 2000)

    outcomes = sample_outcomes(bits, bases, meas, ideal_simulator, seed=3)

    same = bases == meas
    assert (outcomes[same] == bits[same]).all()
    assert 0.44 < outcomes[~same].mean() < 0.56


def test_sample_outcomes_is_reproducible_and_seed_sensitive(ideal_simulator):
    bits = np.zeros(500, dtype=int)
    bases = np.full(500, "x")
    meas = np.full(500, "+")
    first = sample_outcomes(bits, bases, meas, ideal_simulator, seed=11)
    second = sample_outcomes(bits, bases, meas, ideal_simulator, seed=11)
    other = sample_outcomes(bits, bases, meas, ideal_simulator, seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_sample_outcomes_validates_lengths(ideal_simulator):
    with pytest.raises(ValueError):
        sample_outcomes([0, 1], ["+"], ["+", "x"], ideal_simulator)
    assert sample_outcomes([], [], [], ideal_simulator).shape == (0,)


@pytest.mark.parametrize(
    ("bits", "bases", "meas"),
    [
        ([0, 2], ["+", "+"], ["+", "+"]),
        ([0, 0], ["+", "z"], ["+", "+"]),
        ([0, 0], ["+", "+"], ["+", "pi/4"]),
    ],
)
def test_sample_outcomes_rejects_unknown_alphabet(ideal_simulator, bits, bases, meas):
    """An out-of-alphabet round must raise rather than return uninitialised memory."""
    with pytest.raises(ValueError):
        sample_outcomes(bits, bases, meas, ideal_simulator)
