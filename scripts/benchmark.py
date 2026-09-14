"""Time the grouped engine as the number of qubits grows.

The engine in ``qkd_sim`` groups rounds by their (state, measurement basis)
triple and runs at most eight circuits per hop (nine two-qubit circuits for
E91), so the cost of a run is dominated by fixed per-circuit overhead and
barely grows with the number of qubits.

Usage::

    python scripts/benchmark.py [--protocol bb84] [--qubits 200 2000 20000]
"""

from __future__ import annotations

import argparse
import time

from qkd_sim import PROTOCOLS, simulate


def timed_run(protocol: str, num_bits: int, *, with_eve: bool, seed: int) -> float:
    started = time.perf_counter()
    simulate(protocol, num_bits, with_eve=with_eve, seed=seed)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", "-p", choices=sorted(PROTOCOLS), default="bb84")
    parser.add_argument("--qubits", type=int, nargs="+", default=[200, 2000, 20000])
    args = parser.parse_args()

    print(f"{'protocol':<10}{'eve':>5}{'qubits':>8}{'seconds':>10}{'us/qubit':>12}")
    for with_eve in (False, True):
        for n in args.qubits:
            seconds = timed_run(args.protocol, n, with_eve=with_eve, seed=1)
            print(
                f"{args.protocol:<10}{'yes' if with_eve else 'no':>5}{n:>8}"
                f"{seconds:>10.2f}{1e6 * seconds / n:>12.1f}"
            )


if __name__ == "__main__":
    main()
