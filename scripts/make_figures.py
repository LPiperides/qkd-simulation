"""Render the README figures from ``data/sweep.csv``.

Usage::

    python scripts/make_figures.py [--data data/sweep.csv] [--out figures]

Four PNGs are produced:

* ``qber_comparison.png``          ideal channel vs intercept-and-resend, per protocol
* ``qber_vs_noise.png``            QBER against depolarizing strength, with theory
* ``key_rate_vs_noise.png``        Shor-Preskill key rate against noise, thresholds marked
* ``mutual_information_vs_noise.png``  I(A:B) vs I(A:E) under attack, per protocol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qkd_sim.keyrate import qber_threshold  # noqa: E402
from qkd_sim.theory import expected  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Palette: three categorical slots (one per protocol), neutral chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_FILL = "#c3c2b7"
SERIES = {"bb84": "#2a78d6", "b92": "#eb6834", "sarg04": "#1baf7a"}
LABELS = {"bb84": "BB84", "b92": "B92", "sarg04": "SARG04"}
PROTOCOLS = ["bb84", "b92", "sarg04"]
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
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_MUTED, fontsize=8.5, va="bottom")


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
        f"mean over {seeds} seeds, {num_bits:,} qubits per run; error bars = one std;"
        " ideal-channel QBER is exactly zero",
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
    ax.set_ylim(0, 0.26)
    threshold_line(ax, qber_threshold(), "11.0% secret-key threshold", align="left")
    ax.set_xlabel("depolarizing probability p (per hop, no eavesdropper)")
    ax.set_ylabel("QBER on the sifted key")
    ax.set_title("BB84 tolerates almost twice the channel noise of B92 and SARG04")
    subtitle(
        ax, "markers: simulation mean over seeds (band = one std); thin lines: p/2 and p/(1+p)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "qber_vs_noise.png")
    plt.close(fig)


def fig_key_rate_vs_noise(df: pd.DataFrame, out: Path) -> None:
    clean = df[~df["with_eve"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=DPI)
    series_plot(ax, clean, "key_rate", label_end=False)
    ax.set_xlim(0, clean["noise"].max() * 1.05)
    ax.set_ylim(0, 0.55)
    ax.axhline(0, color=AXIS, linewidth=0.8)

    threshold = qber_threshold()
    crossings = {"bb84": 2 * threshold, "b92": threshold / (1 - threshold)}
    crossings["sarg04"] = crossings["b92"]
    for protocol, p_star in crossings.items():
        ax.axvline(p_star, color=SERIES[protocol], linewidth=1, alpha=0.45, zorder=1)
    ax.annotate(
        f"BB84: key vanishes at p = {crossings['bb84']:.3f}",
        (crossings["bb84"], 0.36),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    ax.annotate(
        f"B92 / SARG04: p = {crossings['b92']:.3f}\n(heuristic bound)",
        (crossings["b92"], 0.36),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    ax.set_xlabel("depolarizing probability p (per hop, no eavesdropper)")
    ax.set_ylabel("secret key rate per sent qubit")
    ax.set_title("Shor-Preskill secret key rate under channel noise")
    subtitle(
        ax,
        "r = retention x max(0, 1 - 2 h2(QBER)); rigorous for BB84, heuristic for B92 and SARG04",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "key_rate_vs_noise.png")
    plt.close(fig)


def fig_mutual_information(df: pd.DataFrame, out: Path) -> None:
    tapped = df[df["with_eve"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=DPI, sharey=True)
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
                x, mean - spread, mean + spread, color=colors[metric], alpha=0.10, linewidth=0
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
        "BB84: I(A:E) = I(A:B) = 0.189 bits at p = 0.\n"
        "B92 / SARG04: Eve's conclusive guesses are never wrong (I(A:E) = 0.5)"
        " while Bob's 1/3 error rate leaves I(A:B) = 0.08.",
        color=INK_MUTED,
        fontsize=8.5,
        va="top",
        linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out / "mutual_information_vs_noise.png")
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
    for name in sorted(p.name for p in args.out.glob("*.png")):
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
