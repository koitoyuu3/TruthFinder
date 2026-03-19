#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_PATH = Path(__file__).with_name("数值-恶意节点比例.txt")
OUTPUT_DIR = RESULT_PATH.parent / "figures"
OUTPUT_BASENAME = "num_bad_agent_ratio_topconf_style"
METHOD_ORDER = [
    "EnhancedTruthFinder-Hybrid",
    "BasicTruthFinder",
    "EnhancedTruthFinder-NonHybrid",
    "SenFeedTruthFinder-CompositeTD",
    "SenteTruthFinder",
]
DISPLAY_NAMES = {
    "EnhancedTruthFinder-Hybrid": "EnhancedTruthFinder (Hybrid)",
    "BasicTruthFinder": "BasicTruthFinder",
    "EnhancedTruthFinder-NonHybrid": "EnhancedTruthFinder (w/o Hybrid)",
    "SenFeedTruthFinder-CompositeTD": "SenFeedTruthFinder",
    "SenteTruthFinder": "SenteTruthFinder",
}
METHOD_STYLES = {
    "EnhancedTruthFinder-Hybrid": {"color": "#1B9E77", "marker": "o", "linewidth": 2.8, "zorder": 5},
    "BasicTruthFinder": {"color": "#4C78A8", "marker": "s", "linewidth": 2.1, "zorder": 4},
    "EnhancedTruthFinder-NonHybrid": {"color": "#F58518", "marker": "^", "linewidth": 2.1, "zorder": 4},
    "SenFeedTruthFinder-CompositeTD": {"color": "#B279A2", "marker": "D", "linewidth": 2.1, "zorder": 4},
    "SenteTruthFinder": {"color": "#E45756", "marker": "P", "linewidth": 2.1, "zorder": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw line charts from bad-agent-ratio experiment results.")
    parser.add_argument("--input", type=Path, default=RESULT_PATH, help="Path to the text result file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for exported figures.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG export DPI.")
    parser.add_argument(
        "--visual-y-jitter",
        type=float,
        default=0.0,
        help=(
            "Optional visual-only vertical offset step, expressed as a fraction of each metric range. "
            "Useful for separating overlapping lines without changing the underlying data."
        ),
    )
    return parser.parse_args()


def parse_result_file(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    rows: list[dict[str, object]] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        ratio_match = re.search(r"bad agent\s*=\s*(\d+)%", lines[0])
        if ratio_match is None:
            raise ValueError(f"Unable to parse bad-agent header: {lines[0]!r}")
        bad_agent_ratio = int(ratio_match.group(1))

        for line in lines[2:]:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) != 4:
                raise ValueError(
                    f"Bad-agent ratio {bad_agent_ratio}% row has {len(parts)} fields, expected 4: {line!r}"
                )
            method, clean_rmse, attacked_rmse, growth_rate_pct = parts
            rows.append(
                {
                    "bad_agent_ratio": bad_agent_ratio,
                    "method": method,
                    "clean_rmse": float(clean_rmse),
                    "attacked_rmse": float(attacked_rmse),
                    "growth_rate_pct": float(growth_rate_pct),
                }
            )

    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "bad_agent_ratio"]).reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame) -> None:
    ratios = sorted(df["bad_agent_ratio"].unique().tolist())
    methods = [method for method in METHOD_ORDER if method in set(df["method"].astype(str))]
    missing: list[tuple[str, int]] = []

    for method in methods:
        for ratio in ratios:
            mask = (df["method"].astype(str) == method) & (df["bad_agent_ratio"] == ratio)
            if not mask.any():
                missing.append((method, ratio))

    if missing:
        raise ValueError(f"Missing method/ratio combinations: {missing}")


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.1,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel("Malicious Agent Ratio (%)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def _metric_visual_offsets(df: pd.DataFrame, metric: str, jitter_fraction: float) -> dict[str, float]:
    if jitter_fraction <= 0:
        return {}

    methods = [method for method in METHOD_ORDER if method in set(df["method"].astype(str))]
    if not methods:
        return {}

    metric_values = pd.to_numeric(df[metric], errors="coerce")
    value_range = float(metric_values.max() - metric_values.min())
    if value_range <= 0:
        value_range = max(float(metric_values.abs().max()) * 0.05, 1e-6)

    center = (len(methods) - 1) / 2.0
    step = value_range * float(jitter_fraction)
    return {method: (index - center) * step for index, method in enumerate(methods)}


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, jitter_fraction: float = 0.0) -> None:
    ratios = sorted(df["bad_agent_ratio"].unique().tolist())
    methods = [method for method in METHOD_ORDER if method in set(df["method"].astype(str))]
    visual_offsets = _metric_visual_offsets(df, metric, jitter_fraction)

    for method in methods:
        subset = df[df["method"].astype(str) == method].sort_values("bad_agent_ratio")
        style = METHOD_STYLES[method]
        y_values = subset[metric].astype(float) + visual_offsets.get(method, 0.0)
        ax.plot(
            subset["bad_agent_ratio"],
            y_values,
            label=DISPLAY_NAMES.get(method, method),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=style["zorder"],
        )

    ax.set_xticks(ratios)
    ax.set_xticklabels([str(ratio) for ratio in ratios])


def plot_figure(df: pd.DataFrame, output_dir: Path, dpi: int, visual_y_jitter: float = 0.0) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.79, wspace=0.24)

    plot_metric(axes[0], df, "attacked_rmse", jitter_fraction=visual_y_jitter)
    style_axis(axes[0], "Attack Stability", "Attacked RMSE")

    plot_metric(axes[1], df, "growth_rate_pct", jitter_fraction=visual_y_jitter)
    style_axis(axes[1], "Relative Degradation", "Growth Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.55,
        borderpad=0.45,
    )
    legend.get_frame().set_edgecolor("#CFCFCF")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.96)
    legend.get_frame().set_facecolor("white")

    if visual_y_jitter > 0:
        fig.text(
            0.985,
            0.02,
            f"Visual y-offset applied ({visual_y_jitter:.1%} of metric range per step); data values unchanged.",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#606060",
        )

    png_path = output_dir / f"{OUTPUT_BASENAME}.png"
    pdf_path = output_dir / f"{OUTPUT_BASENAME}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def print_analysis(df: pd.DataFrame, input_path: Path) -> None:
    attacked_table = df.pivot(index="method", columns="bad_agent_ratio", values="attacked_rmse")
    growth_table = df.pivot(index="method", columns="bad_agent_ratio", values="growth_rate_pct")
    clean_table = df.groupby("method", as_index=True, observed=False)["clean_rmse"].first().to_frame(name="clean_rmse")

    attacked_table = attacked_table.reindex([method for method in METHOD_ORDER if method in attacked_table.index])
    growth_table = growth_table.reindex([method for method in METHOD_ORDER if method in growth_table.index])
    clean_table = clean_table.reindex([method for method in METHOD_ORDER if method in clean_table.index])

    print(f"Input file: {input_path}")
    print("Clean RMSE baseline:")
    print(clean_table.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nAttacked RMSE by malicious-agent ratio:")
    print(attacked_table.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nGrowth rate by malicious-agent ratio:")
    print(growth_table.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    df = parse_result_file(args.input)
    validate_data(df)
    png_path, pdf_path = plot_figure(df, args.output_dir, args.dpi, visual_y_jitter=args.visual_y_jitter)
    print_analysis(df, args.input)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
