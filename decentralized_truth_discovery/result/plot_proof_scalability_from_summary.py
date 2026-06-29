#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/truthfinder_mplconfig")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "outputs"
    / "proof_scalability"
    / "summary.csv"
)
DEFAULT_PNG = Path(__file__).with_name("figures") / "proof_scalability_summary_panels.png"
DEFAULT_PDF = Path(__file__).with_name("figures") / "proof_scalability_summary_panels.pdf"

METHOD_ORDER = (
    "SenFeedTruth",
    "DecentTruth",
    "SenteTruth",
    "BasicTruth",
    "SemanticTruth",
)
DISPLAY_NAMES = {
    "SenFeedTruth": "SenFeed",
    "DecentTruth": "Decent",
    "SenteTruth": "Sente",
    "BasicTruth": "Basic",
    "SemanticTruth": "Semantic",
}
COLOR_MAP = {
    "SenFeedTruth": "#c23b22",
    "DecentTruth": "#1f5aa6",
    "SenteTruth": "#178f5b",
    "BasicTruth": "#d18b00",
    "SemanticTruth": "#5c4b8a",
}
MARKER_MAP = {
    "SenFeedTruth": "o",
    "DecentTruth": "^",
    "SenteTruth": "s",
    "BasicTruth": "D",
    "SemanticTruth": "P",
}
METRIC_SPECS = (
    ("proof_size_bytes", "Bundle Size", "bytes", None, True),
    ("proof_payload_bytes", "Proof Payload Size", "bytes", None, False),
    ("verify_peak_delta_rss_mb_mean", "Verify Peak RSS Delta", "mb", "verify_peak_delta_rss_mb_std", False),
    ("verify_elapsed_sec_mean", "Verify Time", "sec", "verify_elapsed_sec_std", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot proof scalability panels from summary.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--batch-count",
        type=int,
        default=None,
        help="If set, restrict the plot to a single batch_count (single-point lines).",
    )
    parser.add_argument("--dpi", type=int, default=280)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "domain",
        "method",
        "batch_count",
        "proof_size_bytes",
        "proof_payload_bytes",
        "verify_peak_delta_rss_mb_mean",
        "verify_elapsed_sec_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _format_bytes(value: float, _pos: int) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 10_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _format_mb(value: float, _pos: int) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"


def _format_sec(value: float, _pos: int) -> str:
    return f"{value:.2f}"


def _apply_metric_formatter(ax: plt.Axes, formatter_kind: str) -> None:
    if formatter_kind == "bytes":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_bytes))
    elif formatter_kind == "mb":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_mb))
    elif formatter_kind == "sec":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_sec))


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a4a4a")
    ax.spines["bottom"].set_color("#4a4a4a")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", width=0.85, length=3)


def _format_metric_value(value: float, formatter_kind: str) -> str:
    if formatter_kind == "bytes":
        return _format_bytes(value, 0)
    if formatter_kind == "mb":
        return _format_mb(value, 0)
    if formatter_kind == "sec":
        return _format_sec(value, 0)
    return f"{value:.2f}"


def _compute_bar_axis_limits(values: list[float]) -> tuple[float, float, bool]:
    if not values:
        return 0.0, 1.0, False

    numeric_values = [float(value) for value in values if pd.notna(value)]
    if not numeric_values:
        return 0.0, 1.0, False

    minimum = min(numeric_values)
    maximum = max(numeric_values)
    span = maximum - minimum

    if minimum >= 0.0:
        if np.isclose(span, 0.0):
            pad = max(maximum * 0.04, 1.0)
            lower = max(minimum - pad, 0.0)
            upper = maximum + pad
            return lower, upper, lower > 0.0

        relative_span = span / max(abs(maximum), 1.0)
        if minimum > 0.0 and relative_span <= 0.08:
            pad = max(span * 0.45, maximum * 0.005, 1.0)
            lower = max(minimum - pad, 0.0)
            upper = maximum + pad
            return lower, upper, lower > 0.0

        return 0.0, maximum * 1.18 if maximum > 0.0 else 1.0, False

    pad = max(span * 0.15, abs(maximum) * 0.08, 1.0)
    return minimum - pad, maximum + pad, False


def _add_zoom_note(ax: plt.Axes, lower: float) -> None:
    ax.text(
        0.98,
        0.96,
        f"zoomed axis\nstart={lower:.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#666666",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        zorder=6,
    )


def _method_rank(method_name: str) -> int:
    try:
        return METHOD_ORDER.index(method_name)
    except ValueError:
        return len(METHOD_ORDER)


def select_batch(df: pd.DataFrame, batch_count: int) -> pd.DataFrame:
    batch_df = df[df["batch_count"] == batch_count].copy()
    if batch_df.empty:
        available_batches = sorted(int(value) for value in df["batch_count"].dropna().unique().tolist())
        raise ValueError(
            f"batch_count={batch_count} not found in summary input. Available batch counts: {available_batches}"
        )
    batch_df["method_rank"] = batch_df["method"].map(_method_rank)
    batch_df = batch_df.sort_values(["domain", "method_rank", "method"]).drop(columns=["method_rank"])
    return batch_df.reset_index(drop=True)


def _plot_scalability_panel(
    ax: plt.Axes,
    domain_df: pd.DataFrame,
    metric: str,
    std_column: str | None,
    title: str,
    formatter_kind: str,
    row_label: str,
    use_log: bool,
) -> None:
    if domain_df.empty:
        ax.set_visible(False)
        return

    methods_present = sorted(
        domain_df["method"].dropna().unique().tolist(),
        key=_method_rank,
    )

    plotted_any = False
    for method_name in methods_present:
        series = (
            domain_df[domain_df["method"] == method_name]
            .sort_values("batch_count")
        )
        x_values = pd.to_numeric(series["batch_count"], errors="coerce").astype(float).tolist()
        y_values = pd.to_numeric(series[metric], errors="coerce").astype(float).tolist()
        if not x_values or not y_values:
            continue

        std_values = None
        if std_column and std_column in series.columns:
            std_values = pd.to_numeric(series[std_column], errors="coerce").astype(float).tolist()

        color = COLOR_MAP.get(method_name, "#4a4a4a")
        marker = MARKER_MAP.get(method_name, "o")
        label = DISPLAY_NAMES.get(method_name, method_name)

        if std_values is not None and any(not np.isnan(value) and value > 0 for value in std_values):
            ax.errorbar(
                x_values,
                y_values,
                yerr=std_values,
                color=color,
                marker=marker,
                markersize=5.5,
                linewidth=1.5,
                capsize=3,
                elinewidth=0.9,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                markersize=5.5,
                linewidth=1.5,
                label=label,
                zorder=3,
            )
        plotted_any = True

    if not plotted_any:
        ax.set_visible(False)
        return

    ax.set_title(title, pad=8)
    ax.set_xlabel("Batch count")
    if row_label:
        ax.set_ylabel(row_label)
    _apply_metric_formatter(ax, formatter_kind)
    _style_axis(ax)

    unique_batches = sorted(
        {int(value) for value in pd.to_numeric(domain_df["batch_count"], errors="coerce").dropna().tolist()}
    )
    if len(unique_batches) > 1:
        ax.set_xscale("log")
        ax.set_xticks(unique_batches)
        ax.set_xticklabels([str(value) for value in unique_batches])
    else:
        ax.set_xticks(unique_batches)

    if use_log:
        ax.set_yscale("log")


def plot_panels(
    df: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    batch_count: int | None = None,
) -> None:
    apply_style()

    plot_df = df.copy()
    if batch_count is not None:
        plot_df = select_batch(plot_df, batch_count)

    domains = ("numeric", "text")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, left=0.06, right=0.985, bottom=0.12, wspace=0.26, hspace=0.42)

    legend_handles: dict[str, tuple] = {}
    for row_index, domain in enumerate(domains):
        domain_df = plot_df[plot_df["domain"] == domain].copy().reset_index(drop=True)
        row_label = "Numeric" if domain == "numeric" else "Text"

        for col_index, (metric, title, formatter_kind, std_column, use_log) in enumerate(METRIC_SPECS):
            ax = axes[row_index, col_index]
            _plot_scalability_panel(
                ax=ax,
                domain_df=domain_df,
                metric=metric,
                std_column=std_column,
                title=f"{title} ({domain})",
                formatter_kind=formatter_kind,
                row_label=row_label if col_index == 0 else "",
                use_log=use_log,
            )
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_handles.setdefault(label, handle)

    if legend_handles:
        ordered_labels = [
            DISPLAY_NAMES.get(method, method)
            for method in METHOD_ORDER
            if DISPLAY_NAMES.get(method, method) in legend_handles
        ]
        fig.legend(
            handles=[legend_handles[label] for label in ordered_labels],
            labels=ordered_labels,
            loc="lower center",
            ncol=len(ordered_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    if batch_count is None:
        title_text = "Proof Bundle Scalability across Batch Counts"
    else:
        title_text = f"Proof Bundle Summary at Batch = {batch_count}"
    fig.suptitle(title_text, y=0.995, fontsize=14)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    df = load_summary(args.input)
    plot_panels(df, args.png, args.pdf, args.dpi, args.batch_count)
    print(f"Saved PNG: {args.png}")
    print(f"Saved PDF: {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
