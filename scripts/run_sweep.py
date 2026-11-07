"""Generate ``data/sweep.csv``: the dataset behind every figure in the README.

Two scans are combined:

* a channel-noise scan without an eavesdropper (all protocols, p = 0..0.30),
* the intercept-and-resend cases on an ideal channel and on the same noise
  grid, so I(A:B) and I(A:E) can be compared against noise.

Usage::

    python scripts/run_sweep.py [--num-bits 4000] [--seeds 5]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from qkd_sim.cli import PROTOCOL_CHOICES, parse_noise_spec, sweep_records

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "sweep.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-bits", "-n", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise", type=parse_noise_spec, default="0:0.3:0.02")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    noises = args.noise  # argparse applies parse_noise_spec to the default string as well
    seeds = range(args.seeds)
    started = time.perf_counter()

    clean = sweep_records(PROTOCOL_CHOICES, noises, seeds, num_bits=args.num_bits, with_eve=False)
    tapped = sweep_records(PROTOCOL_CHOICES, noises, seeds, num_bits=args.num_bits, with_eve=True)
    table = pd.concat([clean, tapped], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows to {args.out} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
