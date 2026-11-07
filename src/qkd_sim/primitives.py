"""Single-qubit circuit primitives shared by the prepare-and-measure protocols.

BB84, B92 and SARG04 are prepare-and-measure schemes built from the same
four states and two measurement bases (E91 adds its own two-qubit circuits in
:mod:`qkd_sim.e91` but re-uses these for the state Eve re-sends):

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
  therefore independent of the number of qubits sent. The grouping machinery
  itself lives in :func:`run_grouped` so that :mod:`qkd_sim.e91` can reuse it.

Every prepared circuit ends in an identity gate (``CHANNEL_GATE``). That gate
does nothing in the noiseless case; it exists so that a channel noise model
can attach a depolarizing error to *every* transmitted qubit, including the
gate-less |0> state, which would otherwise travel noise-free. See
:mod:`qkd_sim.channel`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
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

SEED_LIMIT = 2**31
"""Exclusive upper bound on any ``seed_simulator`` handed to Aer (fits an int32)."""


def require_values(values: np.ndarray, allowed: Sequence, name: str) -> None:
    """Raise ``ValueError`` unless every element of ``values`` is one of ``allowed``."""
    bad = ~np.isin(values, list(allowed))
    if bad.any():
        offending = sorted({v.item() if isinstance(v, np.generic) else v for v in values[bad]})
        raise ValueError(f"{name} must be in {tuple(allowed)}, got {offending}")


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


def run_grouped(
    groups: Iterable[tuple[np.ndarray, QuantumCircuit]],
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, list[str]]]:
    """Run each ``(indices, circuit)`` group once with ``shots = len(indices)``.

    This is the engine behind every fast sampler in the package. Rounds that
    share a circuit are grouped; each distinct circuit is executed once with
    per-shot memory enabled and the shots are handed back together with the
    round indices they belong to, so the caller can scatter them into place.
    Empty groups are skipped but still consume their slot in the seed
    sequence, so the result for a given group does not depend on which other
    groups happen to be populated.

    ``seed`` makes the whole sequence reproducible: a ``numpy`` generator
    seeded with it draws an independent ``seed_simulator`` below
    :data:`SEED_LIMIT` for every group, so any number of groups is safe.
    """
    seed_rng = None if seed is None else np.random.default_rng(int(seed))
    for idx, circuit in groups:
        options = {}
        if seed_rng is not None:
            options["seed_simulator"] = int(seed_rng.integers(0, SEED_LIMIT))
        if idx.size == 0:
            continue
        result = simulator.run(circuit, shots=int(idx.size), memory=True, **options).result()
        yield idx, result.get_memory()


def sample_outcomes(
    bits: Sequence[int] | np.ndarray,
    bases: Sequence[str] | np.ndarray,
    meas_bases: Sequence[str] | np.ndarray,
    simulator: AerSimulator,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Measure state ``(bits[i], bases[i])`` in ``meas_bases[i]`` for every ``i``.

    Rounds are grouped by their (bit, basis, measurement basis) triple and each
    of the at most eight distinct circuits runs once through
    :func:`run_grouped`. Shots are independent samples of the same circuit, so
    this is statistically identical to running one circuit per round, at a
    tiny fraction of the cost.

    Raises ``ValueError`` if any bit is not 0/1 or any basis is not in
    :data:`BASES`; every round is therefore guaranteed to be filled.
    """
    bits = np.asarray(bits, dtype=int)
    bases = np.asarray(bases)
    meas_bases = np.asarray(meas_bases)
    if not (len(bits) == len(bases) == len(meas_bases)):
        raise ValueError("bits, bases and meas_bases must have the same length")
    require_values(bits, (0, 1), "bits")
    require_values(bases, BASES, "bases")
    require_values(meas_bases, BASES, "meas_bases")

    outcomes = np.empty(len(bits), dtype=int)
    if len(bits) == 0:
        return outcomes

    groups = (
        (
            np.flatnonzero((bits == bit) & (bases == basis) & (meas_bases == meas_basis)),
            add_measurement(prepare_qubit(bit, basis), meas_basis),
        )
        for bit, basis, meas_basis in product((0, 1), BASES, BASES)
    )
    for idx, memory in run_grouped(groups, simulator, seed=seed):
        outcomes[idx] = np.fromiter(map(int, memory), dtype=int, count=idx.size)
    return outcomes
