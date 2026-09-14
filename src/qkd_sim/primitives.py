"""Single-qubit circuit primitives shared by every protocol.

All three protocols in this package are prepare-and-measure schemes built
from the same four states and two measurement bases:

    basis '+' (computational / Z):  bit 0 -> |0>,  bit 1 -> |1>
    basis 'x' (diagonal / X):       bit 0 -> |+>,  bit 1 -> |->

This module owns the mapping from (bit, basis) to a Qiskit circuit and the
two ways of executing those circuits on ``AerSimulator``:

* :func:`run_batch` runs an explicit list of circuits, one shot each. It is
  the straightforward reference path and is what the tests exercise directly.
* :func:`sample_outcomes` is the fast path used by the simulation engine. It
  notices that every round of a protocol is one of at most eight distinct
  circuits (four states x two measurement bases) and runs each distinct
  circuit once with as many shots as there are rounds that need it. Cost is
  therefore independent of the number of qubits sent.

Every prepared circuit ends in an identity gate (``CHANNEL_GATE``). That gate
does nothing in the noiseless case; it exists so that a channel noise model
can attach a depolarizing error to *every* transmitted qubit, including the
gate-less |0> state, which would otherwise travel noise-free. See
:mod:`qkd_sim.channel`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

BASES: tuple[str, str] = ("+", "x")
"""Measurement / preparation bases: '+' is the Z basis, 'x' is the X basis."""

STATE_LABELS: dict[tuple[str, int], str] = {
    ("+", 0): "|0>",
    ("+", 1): "|1>",
    ("x", 0): "|+>",
    ("x", 1): "|->",
}
"""Human-readable ket label for each (basis, bit) pair."""

CHANNEL_GATE = "id"
"""Name of the instruction that represents the quantum channel in a circuit."""


def state_label(basis: str, bit: int) -> str:
    """Return the ket label (e.g. ``'|+>'``) for a prepared state."""
    return STATE_LABELS[(basis, int(bit))]


def prepare_qubit(bit: int, basis: str, *, channel: bool = True) -> QuantumCircuit:
    """Build a one-qubit circuit that prepares ``bit`` in ``basis``.

    Parameters
    ----------
    bit:
        0 or 1.
    basis:
        ``'+'`` for the computational basis or ``'x'`` for the diagonal basis.
    channel:
        If true (the default), append an identity gate that acts as the hook
        for channel noise. Pass ``False`` to get the bare preparation, e.g.
        for statevector inspection.
    """
    if bit not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got {bit!r}")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")

    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)
    if basis == "x":
        qc.h(0)
    if channel:
        qc.id(0)
    return qc


def add_measurement(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Return a *copy* of ``qc`` measured in ``basis``.

    The input circuit is never modified, so the same prepared state can be
    measured in several bases (Eve's and Bob's, for instance).
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")

    measured = qc.copy()
    if basis == "x":
        measured.h(0)
    measured.measure(0, 0)
    return measured


def run_batch(
    circuits: Sequence[QuantumCircuit],
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Run each circuit once and return the measured bit for each.

    All circuits go to the simulator in a single job. Aer derives a distinct
    random stream for each circuit in the list from ``seed``, so passing the
    same seed twice reproduces the whole batch. No transpilation is needed:
    ``x``, ``h``, ``id`` and ``measure`` are native Aer instructions.
    """
    circuits = list(circuits)
    if not circuits:
        return np.empty(0, dtype=int)

    options = {} if seed is None else {"seed_simulator": int(seed)}
    result = simulator.run(circuits, shots=1, **options).result()

    outcomes = np.empty(len(circuits), dtype=int)
    for i in range(len(circuits)):
        (bitstring,) = result.get_counts(i)  # shots=1 -> exactly one key
        outcomes[i] = int(bitstring)
    return outcomes


def sample_outcomes(
    bits: Sequence[int] | np.ndarray,
    bases: Sequence[str] | np.ndarray,
    meas_bases: Sequence[str] | np.ndarray,
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Measure state ``(bits[i], bases[i])`` in ``meas_bases[i]`` for every ``i``.

    Rounds are grouped by their (bit, basis, measurement basis) triple. Each of
    the at most eight distinct circuits is executed once with
    ``shots = <number of rounds in the group>`` and per-shot memory enabled,
    and the shot results are scattered back to the rounds that requested
    them. Shots are independent samples of the same circuit, so this is
    statistically identical to running one circuit per round, at a tiny
    fraction of the cost.

    ``seed`` makes the result reproducible: group ``k`` runs with
    ``seed_simulator = seed + k``.
    """
    bits = np.asarray(bits, dtype=int)
    bases = np.asarray(bases)
    meas_bases = np.asarray(meas_bases)
    if not (len(bits) == len(bases) == len(meas_bases)):
        raise ValueError("bits, bases and meas_bases must have the same length")

    outcomes = np.empty(len(bits), dtype=int)
    if len(bits) == 0:
        return outcomes

    for k, (bit, basis, meas_basis) in enumerate(product((0, 1), BASES, BASES)):
        idx = np.flatnonzero((bits == bit) & (bases == basis) & (meas_bases == meas_basis))
        if idx.size == 0:
            continue
        circuit = add_measurement(prepare_qubit(bit, basis), meas_basis)
        options = {} if seed is None else {"seed_simulator": int(seed) + k}
        result = simulator.run(circuit, shots=int(idx.size), memory=True, **options).result()
        memory = result.get_memory()
        outcomes[idx] = np.fromiter(map(int, memory), dtype=int, count=idx.size)

    return outcomes
