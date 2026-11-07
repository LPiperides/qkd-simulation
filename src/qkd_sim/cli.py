"""Command-line interface: ``qkd-simulation run`` and ``qkd-simulation sweep``.

Examples::

    qkd-simulation run --protocol bb84 -n 10000 --eve --seed 42
    qkd-simulation run --protocol sarg04 -n 5000 --noise 0.05 --show 8
    qkd-simulation run --protocol e91 -n 10000 --seed 42 --json
    qkd-simulation sweep --protocol all --noise 0:0.3:0.02 --seeds 5 -n 4000 --out data/sweep.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import PROTOCOLS, simulate
from .keyrate import qber_threshold
from .metrics import Summary, summarize
from .theory import expected, expected_chsh

PROTOCOL_CHOICES = sorted(PROTOCOLS)
PROG = "qkd-simulation"
"""Console-script name, as installed by ``pyproject.toml``."""


def probability(text: str) -> float:
    """argparse type: a float in ``[0, 1]``."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"noise {value} outside [0, 1]")
    return value


def non_negative_int(text: str) -> int:
    """argparse type: an integer ``>= 0``."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def positive_int(text: str) -> int:
    """argparse type: an integer ``>= 1``."""
    value = non_negative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def error_correction_factor(text: str) -> float:
    """argparse type: an error-correction inefficiency ``f_ec >= 1``."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value < 1.0:
        raise argparse.ArgumentTypeError(f"f_ec must be >= 1 (Shannon limit), got {value}")
    return value


def parse_noise_spec(spec: str) -> list[float]:
    """Parse ``'start:stop:step'`` (inclusive, never exceeding ``stop``) or a comma list."""
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("noise range must be start:stop:step")
        start, stop, step = (float(p) for p in parts)
        if step <= 0 or stop < start:
            raise argparse.ArgumentTypeError("noise range needs step > 0 and stop >= start")
        # floor (with a tolerance for binary rounding) so the last point never overshoots stop
        count = math.floor((stop - start) / step + 1e-9) + 1
        values = [round(start + i * step, 10) for i in range(count)]
    else:
        values = [float(p) for p in spec.split(",") if p.strip()]
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(f"noise {value} outside [0, 1]")
    return values


def theory_for(protocol: str, with_eve: bool, noise: float) -> tuple[float, float]:
    """Analytic (retention, qber) or NaNs when no closed form exists."""
    try:
        return expected(protocol, with_eve=with_eve, noise=noise)
    except NotImplementedError:
        return math.nan, math.nan


def theory_chsh_for(protocol: str, with_eve: bool, noise: float) -> float:
    """Analytic CHSH value, or NaN for a protocol without a Bell test."""
    if protocol.lower() != "e91":
        return math.nan
    return expected_chsh(with_eve=with_eve, noise=noise)


def format_summary(s: Summary, *, noise: float, seed: int | None) -> str:
    theory_retention, theory_qber = theory_for(s.protocol, s.with_eve, noise)

    def theory(value: float) -> str:
        return "" if math.isnan(value) else f"   (theory {value:.4f})"

    lines = [
        f"{s.protocol.upper()}  n={s.num_bits}  eve={'yes' if s.with_eve else 'no'}"
        f"  noise={noise:.3f}  seed={'none' if seed is None else seed}",
        f"  key retention    {s.retention:.4f}{theory(theory_retention)}",
        f"  QBER             {s.qber:.4f}{theory(theory_qber)}",
    ]
    if s.with_eve:
        lines.append(f"  Eve agreement    {s.eve_agreement:.4f}")
    lines.append(f"  I(A:B)           {s.i_ab:.4f} bits per sifted bit")
    if s.with_eve:
        lines.append(f"  I(A:E)           {s.i_ae:.4f} bits per sifted bit")
    lines.append(
        f"  secret key rate  {s.key_rate:.4f} per sent qubit"
        f"   (f_ec={s.f_ec:.2f}, threshold QBER {qber_threshold(s.f_ec):.4f})"
    )
    if not math.isnan(s.chsh):
        lines.append(
            f"  CHSH S           {s.chsh:.4f}"
            f"{theory(theory_chsh_for(s.protocol, s.with_eve, noise))}"
            "   (local bound 2, quantum maximum 2.8284)"
        )
        lines.append(
            f"  DI key rate      {s.di_key_rate:.4f} per sent pair"
            "   (device-independent bound from S and QBER)"
        )
    return "\n".join(lines)


def sweep_records(
    protocols: Iterable[str],
    noises: Iterable[float],
    seeds: Iterable[int],
    *,
    num_bits: int,
    with_eve: bool,
    f_ec: float = 1.0,
) -> pd.DataFrame:
    """Run every (protocol, noise, seed) combination and tabulate the summaries."""
    rows = []
    for protocol in protocols:
        for noise in noises:
            theory_retention, theory_qber = theory_for(protocol, with_eve, noise)
            theory_chsh = theory_chsh_for(protocol, with_eve, noise)
            for seed in seeds:
                df = simulate(protocol, num_bits, with_eve=with_eve, noise=noise, seed=seed)
                s = summarize(df, protocol=protocol, f_ec=f_ec)
                rows.append(
                    {
                        "protocol": protocol,
                        "with_eve": with_eve,
                        "noise": noise,
                        "seed": seed,
                        "num_bits": num_bits,
                        "retention": s.retention,
                        "qber": s.qber,
                        "eve_agreement": s.eve_agreement,
                        "i_ab": s.i_ab,
                        "i_ae": s.i_ae,
                        "key_rate": s.key_rate,
                        "f_ec": f_ec,
                        "chsh": s.chsh,
                        "di_key_rate": s.di_key_rate,
                        "theory_retention": theory_retention,
                        "theory_qber": theory_qber,
                        "theory_chsh": theory_chsh,
                    }
                )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Simulate BB84, B92, SARG04 and E91 quantum key distribution on Qiskit Aer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one run and print its metrics")
    run.add_argument("--protocol", "-p", choices=PROTOCOL_CHOICES, required=True)
    run.add_argument(
        "--num-bits",
        "-n",
        type=non_negative_int,
        default=1000,
        help="qubits (or entangled pairs) to send",
    )
    run.add_argument("--eve", action="store_true", help="add an intercept-and-resend Eve")
    run.add_argument(
        "--noise", type=probability, default=0.0, help="depolarizing probability in [0, 1]"
    )
    run.add_argument("--seed", type=int, default=None, help="seed for reproducible runs")
    run.add_argument(
        "--f-ec",
        type=error_correction_factor,
        default=1.0,
        help="error-correction inefficiency (>= 1)",
    )
    run.add_argument(
        "--show", type=non_negative_int, default=0, metavar="K", help="print first K rounds"
    )
    run.add_argument("--csv", metavar="PATH", help="write the full transcript to CSV")
    run.add_argument("--json", action="store_true", help="print the summary as JSON")

    sweep = sub.add_parser("sweep", help="scan channel noise over seeds and write a CSV")
    sweep.add_argument("--protocol", "-p", choices=[*PROTOCOL_CHOICES, "all"], default="all")
    sweep.add_argument("--noise", type=parse_noise_spec, default="0:0.3:0.02")
    sweep.add_argument("--seeds", type=positive_int, default=5, help="number of seeds per point")
    sweep.add_argument("--seed-base", type=int, default=0, help="first seed value")
    sweep.add_argument("--num-bits", "-n", type=non_negative_int, default=4000)
    sweep.add_argument("--eve", action="store_true")
    sweep.add_argument("--f-ec", type=error_correction_factor, default=1.0)
    sweep.add_argument("--out", required=True, metavar="PATH", help="CSV to write")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    df = simulate(
        args.protocol,
        args.num_bits,
        with_eve=args.eve,
        noise=args.noise,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    s = summarize(df, protocol=args.protocol, f_ec=args.f_ec)

    if args.csv:
        df.to_csv(args.csv, index=False)

    if args.json:
        payload = s.as_dict()
        theory_retention, theory_qber = theory_for(s.protocol, s.with_eve, args.noise)
        payload.update(
            noise=args.noise,
            seed=args.seed,
            theory_retention=theory_retention,
            theory_qber=theory_qber,
            theory_chsh=theory_chsh_for(s.protocol, s.with_eve, args.noise),
            seconds=elapsed,
        )
        print(json.dumps(json_ready(payload), indent=2, allow_nan=False))
    else:
        print(format_summary(s, noise=args.noise, seed=args.seed))
        print(f"  simulated in     {elapsed:.2f} s")
        if args.show:
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print()
                print(df.head(args.show).to_string(index=False))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    protocols = PROTOCOL_CHOICES if args.protocol == "all" else [args.protocol]
    noises = args.noise  # argparse applies parse_noise_spec to the default string as well
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    started = time.perf_counter()
    table = sweep_records(
        protocols,
        noises,
        seeds,
        num_bits=args.num_bits,
        with_eve=args.eve,
        f_ec=args.f_ec,
    )
    table.to_csv(args.out, index=False)
    print(
        f"wrote {len(table)} rows ({len(protocols)} protocols x {len(noises)} noise levels"
        f" x {args.seeds} seeds, n={args.num_bits}) to {args.out}"
        f" in {time.perf_counter() - started:.1f} s"
    )
    return 0


def json_ready(payload: dict) -> dict:
    """Copy of ``payload`` with numpy scalars unboxed and NaN replaced by ``None``.

    ``json.dumps`` would otherwise emit the bare token ``NaN``, which is not
    valid JSON and is rejected by ``jq`` and JavaScript parsers.
    """
    out = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        out[key] = value
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "sweep":
        return cmd_sweep(args)
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
