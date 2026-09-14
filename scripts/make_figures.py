"""Render the README figures from ``data/sweep.csv``.

Usage::

    python scripts/make_figures.py [--data data/sweep.csv] [--out figures]

Five PNGs are produced:

* ``qber_comparison.png``          ideal channel vs intercept-and-resend, per protocol
* ``qber_vs_noise.png``            QBER against depolarizing strength, with theory
* ``key_rate_vs_noise.png``        Shor-Preskill key rate against noise, thresholds marked,
                                   plus the device-independent bound for E91
* ``mutual_information_vs_noise.png``  I(A:B) vs I(A:E) under attack, per protocol
* ``chsh_vs_noise.png``            E91's CHSH value against noise, with and without Eve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qkd_sim.e91 import LOCAL_BOUND, QUANTUM_BOUND
from qkd_sim.keyrate import di_key_rate, qber_threshold
from qkd_sim.theory import expected, expected_chsh

ROOT = Path(__file__).resolve().parent.parent

# Palette: four categorical slots (one per protocol), neutral chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_FILL = "#c3c2b7"
SERIES = {"bb84": "#2a78d6", "b92": "#eb6834", "sarg04": "#1baf7a", "e91": "#eda100"}
LABELS = {"bb84": "BB84", "b92": "B92", "sarg04": "SARG04", "e91": "E91"}
PROTOCOLS = ["bb84", "b92", "sarg04", "e91"]
DPI = 150


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 20,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and standard deviation of ``metric`` over seeds, per protocol and noise."""
    grouped = df.groupby(["protocol", "noise"])[metric]
    return grouped.agg(["mean", "std"]).reset_index()


def subtitle(ax, text: str) -> None:
    """Muted one-line note sitting between the title and the plot."""
    ax.text(
        0,
        1.02,
        text,
        transform=ax.transAxes,
        color=INK_MUTED,
        fontsize=8.5,
        va="bottom",
    )


def threshold_line(ax, y: float, text: str, *, align: str = "right") -> None:
    ax.axhline(y, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    x0, x1 = ax.get_xlim()
    ax.text(
        x1 if align == "right" else x0 + 0.01 * (x1 - x0),
        y,
        text,
        color=INK_MUTED,
        fontsize=8.5,
        ha=align,
        va="bottom",
        transform=ax.transData,
    )


def series_plot(ax, table: pd.DataFrame, metric: str, *, label_end: bool = True) -> None:
    """Mean line with markers and a +/- one-std wash for each protocol."""
    stats = aggregate(table, metric)
    for protocol in PROTOCOLS:
        rows = stats[stats["protocol"] == protocol].sort_values("noise")
        if rows.empty:
            continue
        color = SERIES[protocol]
        x = rows["noise"].to_numpy()
        mean = rows["mean"].to_numpy()
        spread = np.nan_to_num(rows["std"].to_numpy())
        ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=LABELS[protocol],
            zorder=3,
        )
        if label_end:
            ax.annotate(
                LABELS[protocol],
                (x[-1], mean[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
                va="center",
            )


def theory_curve(ax, protocol: str, noises: np.ndarray, metric: str) -> None:
    values = [expected(protocol, noise=p)[0 if metric == "retention" else 1] for p in noises]
    ax.plot(noises, values, color=SERIES[protocol], linewidth=1, alpha=0.45, zorder=2)


def noise_where(f, *, lo: float = 0.0, hi: float = 1.0, tol: float = 1e-6) -> float:
    """Smallest ``p`` in ``[lo, hi]`` at which the decreasing function ``f`` reaches zero."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def e91_di_rate(p: float) -> float:
    retention, error = expected("e91", noise=p)
    return di_key_rate(expected_chsh(noise=p), error, retention)


def fig_qber_comparison(df: pd.DataFrame, out: Path) -> None:
    ideal = df[(df["noise"] == 0) & (~df["with_eve"])]
    tapped = df[(df["noise"] == 0) & (df["with_eve"])]
    num_bits = int(df["num_bits"].iloc[0])
    seeds = df["seed"].nunique()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=DPI)
    width = 0.32
    xs = np.arange(len(PROTOCOLS))
    for i, protocol in enumerate(PROTOCOLS):
        for offset, subset, color, name in (
            (-width / 2, ideal, NEUTRAL_FILL, "ideal channel"),
            (width / 2, tapped, SERIES[protocol], "intercept-and-resend Eve"),
        ):
            values = subset.loc[subset["protocol"] == protocol, "qber"]
            ax.bar(
                xs[i] + offset,
                values.mean(),
                width - 0.02,
                color=color,
                yerr=values.std(),
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3},
                label=name if i == 0 else None,
                zorder=3,
            )
        _, theory = expected(protocol, with_eve=True)
        ax.plot(
            [xs[i] + width / 2 - 0.16, xs[i] + width / 2 + 0.16],
            [theory, theory],
            color=INK,
            linewidth=1.2,
            zorder=4,
            label="analytic value" if i == 0 else None,
        )
        ax.annotate(
            f"{tapped.loc[tapped['protocol'] == protocol, 'qber'].mean():.3f}",
            (xs[i] + width / 2, theory),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
            fontsize=9,
        )

    ax.set_xticks(xs, [LABELS[p] for p in PROTOCOLS])
    ax.set_ylabel("QBER on the sifted key")
    ax.set_ylim(0, 0.45)
    threshold_line(ax, qber_threshold(), "11.0% secret-key threshold")
    ax.set_title("An eavesdropper is visible in every protocol")
    subtitle(
        ax,
        f"mean over {seeds} seeds of {num_bits:,} qubits (E91: pairs); error bars = one std;"
        " ideal-channel QBER is zero",
    )
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(out / "qber_comparison.png")
    plt.close(fig)


def fig_qber_vs_noise(df: pd.DataFrame, out: Path) -> None:
    clean = df[~df["with_eve"]]
    noises = np.linspace(0, clean["noise"].max(), 200)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=DPI)
    for protocol in PROTOCOLS:
        theory_curve(ax, protocol, noises, "qber")
    series_plot(ax, clean, "qber", label_end=False)
    ax.set_xlim(0, clean["noise"].max() * 1.05)
    ax.set_ylim(0, 0.28)
    threshold_line(ax, qber_threshold(), "11.0% secret-key threshold", align="left")
    ax.set_xlabel("depolarizing probability p (per hop, no eavesdropper)")
    ax.set_ylabel("QBER on the sifted key")
    ax.set_title("BB84 tolerates almost twice the noise of the other protocols")
    subtitle(
        ax,
        "markers: simulation mean over seeds (band = one std);"
        " thin lines: p/2, p/(1+p) and (1 - (1-p)^2)/2",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "qber_vs_noise.png")
    plt.close(fig)


def fig_key_rate_vs_noise(df: pd.DataFrame, out: Path) -> None:
    clean = df[~df["with_eve"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=DPI)
    series_plot(ax, clean, "key_rate", label_end=False)

    di_stats = aggregate(clean[clean["protocol"] == "e91"], "di_key_rate").sort_values("noise")
    if not di_stats.empty:
        ax.plot(
            di_stats["noise"],
            di_stats["mean"],
            color=SERIES["e91"],
            linestyle=(0, (4, 3)),
            marker="s",
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label="E91, device-independent bound",
            zorder=3,
        )

    ax.set_xlim(0, clean["noise"].max() * 1.05)
    ax.set_ylim(0, 0.55)
    ax.axhline(0, color=AXIS, linewidth=0.8)

    threshold = qber_threshold()
    crossings = {
        "bb84": 2 * threshold,
        "b92": threshold / (1 - threshold),
        "e91": 1 - np.sqrt(1 - 2 * threshold),
    }
    crossings["sarg04"] = crossings["b92"]
    di_crossing = noise_where(e91_di_rate, hi=0.2)
    for protocol, p_star in crossings.items():
        ax.axvline(p_star, color=SERIES[protocol], linewidth=1, alpha=0.45, zorder=1)
    ax.axvline(
        di_crossing, color=SERIES["e91"], linewidth=1, alpha=0.45, linestyle=(0, (4, 3)), zorder=1
    )
    # Crossing labels hang down from the top of each line; "left"/"right" is the
    # side of the line the text sits on, so the two near-coincident lines share.
    for p_star, text, side in (
        (di_crossing, f"E91, device-independent: p = {di_crossing:.3f}", "left"),
        (crossings["e91"], f"E91: p = {crossings['e91']:.3f}", "left"),
        (crossings["b92"], f"B92 / SARG04: p = {crossings['b92']:.3f}", "right"),
        (crossings["bb84"], f"BB84: p = {crossings['bb84']:.3f}", "right"),
    ):
        ax.annotate(
            text,
            (p_star, 0.53),
            xytext=(-3 if side == "left" else 3, 0),
            textcoords="offset points",
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="bottom" if side == "left" else "top",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_xlabel("depolarizing probability p (per hop, no eavesdropper)")
    ax.set_ylabel("secret key rate per sent qubit or pair")
    ax.set_title("Trusting the devices buys E91 almost twice the noise budget")
    subtitle(
        ax,
        "solid: Shor-Preskill r = retention x (1 - 2 h2(Q)), rigorous for BB84 only;"
        " dashed: CHSH-based bound",
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.1))
    fig.tight_layout()
    fig.savefig(out / "key_rate_vs_noise.png")
    plt.close(fig)


def fig_mutual_information(df: pd.DataFrame, out: Path) -> None:
    tapped = df[df["with_eve"]]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=DPI, sharey=True)
    colors = {"i_ab": "#2a78d6", "i_ae": "#eb6834"}
    names = {"i_ab": "I(A:B)  Alice-Bob", "i_ae": "I(A:E)  Alice-Eve"}
    for ax, protocol in zip(axes, PROTOCOLS, strict=True):
        rows = tapped[tapped["protocol"] == protocol]
        for metric in ("i_ab", "i_ae"):
            stats = aggregate(rows, metric).sort_values("noise")
            x = stats["noise"].to_numpy()
            mean = stats["mean"].to_numpy()
            spread = np.nan_to_num(stats["std"].to_numpy())
            ax.fill_between(
                x,
                mean - spread,
                mean + spread,
                color=colors[metric],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=colors[metric],
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=names[metric],
                zorder=3,
            )
        ax.set_title(LABELS[protocol], loc="center", pad=6)
        ax.set_xlabel("depolarizing probability p")
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("bits per sifted bit")
    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Under intercept-and-resend Eve always knows at least as much as Bob",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.91,
        "BB84: I(A:E) = I(A:B) = 0.189 bits at p = 0.  "
        "B92 / SARG04: Eve's conclusive guesses are never wrong (I(A:E) = 0.5)"
        " while Bob's 1/3 error rate leaves I(A:B) = 0.08.\n"
        "E91: measuring Bob's half of the pair in Z or X tells Eve Alice's outcome on 80% of key"
        " rounds (I(A:E) = 0.28) and halves the CHSH value.",
        color=INK_MUTED,
        fontsize=8.5,
        va="top",
        linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out / "mutual_information_vs_noise.png")
    plt.close(fig)


def fig_chsh_vs_noise(df: pd.DataFrame, out: Path) -> None:
    e91 = df[df["protocol"] == "e91"]
    if e91.empty:
        return
    noises = np.linspace(0, e91["noise"].max(), 200)
    color = SERIES["e91"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=DPI)
    ax.plot(noises, [expected_chsh(noise=p) for p in noises], color=color, lw=1, alpha=0.45)
    ax.plot(
        noises,
        [expected_chsh(with_eve=True, noise=p) for p in noises],
        color=color,
        lw=1,
        alpha=0.45,
    )
    for with_eve, marker, style, name in (
        (False, "o", "-", "no eavesdropper"),
        (True, "s", (0, (4, 3)), "intercept-and-resend Eve"),
    ):
        stats = aggregate(e91[e91["with_eve"] == with_eve], "chsh").sort_values("noise")
        x = stats["noise"].to_numpy()
        mean = stats["mean"].to_numpy()
        spread = np.nan_to_num(stats["std"].to_numpy())
        ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=name,
            zorder=3,
        )
        ax.annotate(
            name,
            (x[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
            va="center",
        )

    ax.set_xlim(0, e91["noise"].max() * 1.25)
    ax.set_ylim(0, 3.1)
    threshold_line(ax, LOCAL_BOUND, "local-realist bound S = 2", align="left")
    ax.axhline(QUANTUM_BOUND, color=INK_MUTED, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
    ax.text(
        0.003, QUANTUM_BOUND, "Tsirelson bound 2 sqrt2", color=INK_MUTED, fontsize=8.5, va="bottom"
    )

    bell_crossing = noise_where(lambda p: expected_chsh(noise=p) - LOCAL_BOUND, hi=0.5)
    key_crossing = 1 - np.sqrt(1 - 2 * qber_threshold())
    for p_star, text, y in (
        (key_crossing, f"key vanishes\np = {key_crossing:.3f}", 0.55),
        (bell_crossing, f"violation ends\np = {bell_crossing:.3f}", 1.2),
    ):
        ax.axvline(p_star, color=INK_MUTED, linewidth=1, alpha=0.6, zorder=1)
        ax.annotate(
            text,
            (p_star, y),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    ax.set_xlabel("depolarizing probability p (per hop)")
    ax.set_ylabel("CHSH value S")
    ax.set_title("E91's Bell violation outlives its key, and an eavesdropper erases it")
    subtitle(
        ax,
        "markers: mean over seeds (band = one std); thin lines: 2 sqrt2 (1-p)^2 and,"
        " with Eve, sqrt2 (1-p)^3",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "chsh_vs_noise.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sweep.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    style()
    df = pd.read_csv(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_qber_comparison(df, args.out)
    fig_qber_vs_noise(df, args.out)
    fig_key_rate_vs_noise(df, args.out)
    fig_mutual_information(df, args.out)
    fig_chsh_vs_noise(df, args.out)
    for name in sorted(p.name for p in args.out.glob("*.png")):
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
