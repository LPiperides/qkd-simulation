"""Compare the naive one-job-per-qubit loop with the grouped engine.

The original prototype of this project built a fresh circuit for every
qubit, transpiled it, and submitted it to Aer as its own job. The engine in
``qkd_sim`` instead groups rounds by their (state, measurement basis) triple
and runs at most eight circuits per hop, so the cost no longer grows with the
number of qubits in any meaningful way.

Usage::

    python scripts/benchmark.py [--baseline-qubits 200] [--qubits 200 2000 20000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qkd_sim import simulate
from qkd_sim.primitives import BASES, add_measurement, prepare_qubit


def naive_bb84(num_bits: int, rng: np.random.Generator) -> float:
    """The prototype's execution strategy: transpile + run each qubit separately."""
    simulator = AerSimulator()
    alice_bits = rng.integers(0, 2, num_bits)
    alice_bases = rng.choice(BASES, num_bits)
    bob_bases = rng.choice(BASES, num_bits)
    started = time.perf_counter()
    for bit, basis, meas in zip(alice_bits, alice_bases, bob_bases, strict=True):
        qc = add_measurement(prepare_qubit(int(bit), basis), meas)
        compiled = transpile(qc, simulator)
        simulator.run(compiled, shots=1).result().get_counts()
    return time.perf_counter() - started


def grouped_bb84(num_bits: int, seed: int) -> float:
    started = time.perf_counter()
    simulate("bb84", num_bits, with_eve=True, seed=seed)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--baseline-qubits", type=int, default=200)
    parser.add_argument("--qubits", type=int, nargs="+", default=[200, 2000, 20000])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print(f"{'strategy':<28}{'qubits':>8}{'seconds':>10}{'us/qubit':>12}")
    baseline = naive_bb84(args.baseline_qubits, rng)
    print(
        f"{'per-qubit transpile+run':<28}{args.baseline_qubits:>8}{baseline:>10.2f}"
        f"{1e6 * baseline / args.baseline_qubits:>12.0f}"
    )
    for n in args.qubits:
        grouped = grouped_bb84(n, seed=1)
        print(f"{'grouped engine (with Eve)':<28}{n:>8}{grouped:>10.2f}{1e6 * grouped / n:>12.1f}")


if __name__ == "__main__":
    main()
