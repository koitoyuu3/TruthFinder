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
METHOD_ORDER = list(COLOR_MAP.keys())
BOTTOM_SWEEP_CONFIGS = [
    {
        "title": "Fixed: w_time = 0.1",
        "fixed_key": "time",
        "fixed_weight": 0.1,
        "x_key": "mem",
        "residual_key": "qual",
    },
    {
        "title": "Fixed: w_mem = 0.1",
        "fixed_key": "mem",
        "fixed_weight": 0.1,
        "x_key": "time",
        "residual_key": "qual",
    },
    {
        "title": "Fixed: w_qual = 0.1",
        "fixed_key": "qual",
        "fixed_weight": 0.1,
        "x_key": "time",
        "residual_key": "mem",
    },
]
WEIGHT_LABELS = {
    "time": "w_time",
    "mem": "w_mem",
    "qual": "w_qual",
}
WEIGHT_SWEEP_X = np.linspace(0.0, 0.9, 50)
WEIGHT_SWEEP_TICKS = np.linspace(0.0, 0.9, 5)


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


def _normalize_metric(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_series = numeric_series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid_series.empty:
        return result
    minimum = float(valid_series.min())
    maximum = float(valid_series.max())
    if np.isclose(minimum, maximum, rtol=1e-12, atol=1e-12):
        result.loc[valid_series.index] = 100.0
        return result
    if higher_is_better:
        result.loc[valid_series.index] = 100.0 * (valid_series - minimum) / (maximum - minimum)
    else:
        result.loc[valid_series.index] = 100.0 * (maximum - valid_series) / (maximum - minimum)
    return result


def build_weight_sweep_df(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["time_norm"] = _normalize_metric(plot_df["overall_mean_elapsed_sec"], higher_is_better=False)
    plot_df["mem_norm"] = _normalize_metric(plot_df["overall_mean_peak_delta_rss_mb"], higher_is_better=False)
    plot_df["qual_norm"] = pd.to_numeric(plot_df["overall_mean_source_score_pct"], errors="coerce")
    order_map = {label: idx for idx, label in enumerate(METHOD_ORDER)}
    plot_df["_order"] = plot_df["label"].map(order_map).fillna(len(order_map))
    return plot_df.sort_values(["_order", "label"]).reset_index(drop=True)


def compute_weight_sweep_curve(row: pd.Series, config: dict[str, object]) -> np.ndarray:
    score_map = {
        "time": float(row["time_norm"]),
        "mem": float(row["mem_norm"]),
        "qual": float(row["qual_norm"]),
    }
    fixed_key = str(config["fixed_key"])
    x_key = str(config["x_key"])
    residual_key = str(config["residual_key"])
    fixed_weight = float(config["fixed_weight"])
    return (
        fixed_weight * score_map[fixed_key]
        + WEIGHT_SWEEP_X * score_map[x_key]
        + (0.9 - WEIGHT_SWEEP_X) * score_map[residual_key]
    )


def compute_shared_y_limits(plot_df: pd.DataFrame) -> tuple[float, float]:
    all_curves: list[np.ndarray] = []
    for config in BOTTOM_SWEEP_CONFIGS:
        for _, row in plot_df.iterrows():
            all_curves.append(compute_weight_sweep_curve(row, config))
    values = np.concatenate(all_curves) if all_curves else np.array([0.0, 100.0], dtype=float)
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    span = ymax - ymin
    pad = max(span * 0.12, 4.0)
    return max(0.0, ymin - pad * 0.35), ymax + pad


def add_dual_weight_axis(ax: plt.Axes, x_key: str, residual_key: str) -> None:
    ax.set_xlim(0.0, 0.9)
    ax.set_xticks(WEIGHT_SWEEP_TICKS)
    ax.set_xticklabels([f"{0.9 - tick:.2f}" for tick in WEIGHT_SWEEP_TICKS])
    ax.set_xlabel(f"{WEIGHT_LABELS[residual_key]} = 0.9 - {WEIGHT_LABELS[x_key]}", labelpad=8)

    top_ax = ax.secondary_xaxis("top", functions=(lambda value: value, lambda value: value))
    top_ax.set_xticks(WEIGHT_SWEEP_TICKS)
    top_ax.set_xticklabels([f"{tick:.2f}" for tick in WEIGHT_SWEEP_TICKS])
    top_ax.set_xlabel(WEIGHT_LABELS[x_key], labelpad=8)
    top_ax.tick_params(axis="x", pad=4, labelsize=9)


def add_equal_split_annotation(ax: plt.Axes, split_values: list[float]) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    max_split = max(split_values) if split_values else y_max
    y_pos = min(y_max - y_span * 0.07, max_split + y_span * 0.05)

    ax.annotate(
        "equal split",
        xy=(0.47, y_pos),
        xytext=(0, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
        zorder=6,
    )


def weight_sweep_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    config: dict[str, object],
    *,
    y_limits: tuple[float, float],
    show_ylabel: bool,
    add_legend: bool,
) -> None:
    handles = []
    split_values = []
    for _, row in plot_df.iterrows():
        label = str(row["label"])
        color = COLOR_MAP.get(label, "#4c4c4c")
        curve = compute_weight_sweep_curve(row, config)
        line, = ax.plot(WEIGHT_SWEEP_X, curve, color=color, linewidth=2.2, alpha=0.98, label=label)
        handles.append(line)
        split_values.append(float(np.interp(0.45, WEIGHT_SWEEP_X, curve)))

    ax.axvline(0.45, linestyle="--", linewidth=1.0, color="#8c8c8c", alpha=0.9)
    ax.set_title(str(config["title"]), pad=8)
    add_dual_weight_axis(ax, str(config["x_key"]), str(config["residual_key"]))
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.grid(axis="x", linestyle=":", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(*y_limits)
    add_equal_split_annotation(ax, split_values)
    if show_ylabel:
        ax.set_ylabel("Total Score")
    else:
        ax.tick_params(axis="y", labelleft=False)

    if add_legend:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Method",
            fontsize=9,
            title_fontsize=10,
        )


def plot_figure(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int, method_spacing: float) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    fig = plt.figure(figsize=(18.5, 10.4), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.18])
    top_axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    bottom_axes = []
    for index in range(3):
        shared_axis = bottom_axes[0] if bottom_axes else None
        bottom_axes.append(fig.add_subplot(gs[1, index], sharey=shared_axis))

    for axis, (col, title, lower_is_better, fmt) in zip(top_axes, TOP_METRICS):
        lollipop_panel(axis, df, col, title, lower_is_better, fmt, method_spacing)

    plot_df = build_weight_sweep_df(df)
    y_min, y_max = compute_shared_y_limits(plot_df)
    for index, (axis, config) in enumerate(zip(bottom_axes, BOTTOM_SWEEP_CONFIGS)):
        weight_sweep_panel(
            axis,
            plot_df,
            config,
            y_limits=(y_min, y_max),
            show_ylabel=index == 0,
            add_legend=index == len(bottom_axes) - 1,
        )

    fig.suptitle("Five Methods: Overall Value Comparison", fontsize=17, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Top row: elapsed time / memory / source score | Bottom row: weight sweeps with one metric fixed at 0.1",
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
