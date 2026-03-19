#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DEFAULT_INPUT = Path(__file__).with_name("数值-随着agent变化.csv")
DEFAULT_PNG = Path(__file__).with_name("混合-随着agent变化_value_panels.png")
DEFAULT_PDF = Path(__file__).with_name("混合-随着agent变化_value_panels.pdf")

DISPLAY_NAMES = {
    "EnhancedTruthFinder-Hybrid": "Enhanced-Hybrid",
    "EnhancedTruthFinder-NonHybrid": "Enhanced-NonHybrid",
    "SenteTruthFinder": "Sente",
    "BasicTruthFinder": "Basic",
    "SenFeedTruthFinder-CompositeTD": "SenFeed-CompositeTD",
}

COLOR_MAP = {
    "Enhanced-Hybrid": "#1f77b4",
    "Enhanced-NonHybrid": "#ff7f0e",
    "Sente": "#2ca02c",
    "Basic": "#d62728",
    "SenFeed-CompositeTD": "#9467bd",
}

TOP_METRICS = [
    ("overall_mean_elapsed_sec", "Avg Elapsed Time (s)", True, "{:.2f}"),
    ("overall_mean_peak_delta_rss_mb", "Avg Peak RSS Delta (MB)", True, "{:.2f}"),
    ("overall_mean_source_score_pct", "Avg Source Score (%)", False, "{:.1f}"),
]
BOTTOM_METRIC = ("overall_mean_composite_score", "Avg Composite Score", False, "{:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot overall value comparison panels from agent-change CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG, help="Output PNG path.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Output PDF path.")
    parser.add_argument("--dpi", type=int, default=240, help="PNG DPI.")
    parser.add_argument(
        "--method-spacing",
        type=float,
        default=0.35,
        help="Vertical spacing between methods inside each subplot; smaller means tighter.",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["method"].map(DISPLAY_NAMES).fillna(df["method"])
    return df


def lollipop_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    title: str,
    lower_is_better: bool,
    fmt: str,
    method_spacing: float,
) -> None:
    plot_df = df[["label", col]].sort_values(col, ascending=lower_is_better).reset_index(drop=True)
    y_positions = np.arange(len(plot_df), dtype=float) * method_spacing
    colors = [COLOR_MAP.get(label, "#4c4c4c") for label in plot_df["label"]]
    values = plot_df[col].astype(float).tolist()
    vmin = 0.0
    vmax = max(values)
    span = max(vmax - vmin, vmax * 0.15, 0.5)

    ax.hlines(y=y_positions, xmin=vmin, xmax=values, color="#d7dde6", linewidth=1.9, zorder=1)
    ax.scatter(values, y_positions, s=110, c=colors, zorder=3, edgecolors="white", linewidths=1.0)

    for y_pos, value in zip(y_positions, values):
        ax.text(value + span * 0.03, y_pos, fmt.format(value), va="center", ha="left", fontsize=9)

    ax.set_yticks(y_positions, plot_df["label"], fontsize=10)
    ax.set_ylim(y_positions[-1] + method_spacing * 0.30, -method_spacing * 0.30)
    ax.set_xlim(vmin, vmax + span * 0.22)
    ax.set_title(title, pad=8)
    ax.set_xlabel("lower is better" if lower_is_better else "higher is better")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_figure(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int, method_spacing: float) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    fig = plt.figure(figsize=(18, 9.2), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    axes.append(fig.add_subplot(gs[1, :]))

    for axis, (col, title, lower_is_better, fmt) in zip(axes[:3], TOP_METRICS):
        lollipop_panel(axis, df, col, title, lower_is_better, fmt, method_spacing)

    lollipop_panel(axes[3], df, *BOTTOM_METRIC, method_spacing)

    fig.suptitle("Five Methods: Overall Value Comparison", fontsize=17, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Top row: elapsed time / memory / source score | Bottom row: composite score",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    plot_figure(df, args.png, args.pdf, args.dpi, args.method_spacing)
    print(f"Saved PNG: {args.png}")
    print(f"Saved PDF: {args.pdf}")


if __name__ == "__main__":
    main()
