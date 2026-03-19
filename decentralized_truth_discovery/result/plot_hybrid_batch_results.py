#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_PATH = Path(__file__).with_name("混合-随着batch变化.txt")
OUTPUT_DIR = RESULT_PATH.parent / "figures"
OUTPUT_BASENAME = "hybrid_batch_topconf_style"
SELECTED_BATCHES = [10, 50, 90, 130]
METHOD_ORDER = [
    "EnhancedTruthFinder-Hybrid",
    "EnhancedTruthFinder-NonHybrid",
    "SenteTruthFinder",
    "BasicTruthFinder",
    "SenFeedTruthFinder-CompositeTD",
]
DISPLAY_NAMES = {
    "EnhancedTruthFinder-Hybrid": "EnhancedTruthFinder\n(Hybrid)",
    "EnhancedTruthFinder-NonHybrid": "EnhancedTruthFinder\n(w/o Hybrid)",
    "SenteTruthFinder": "SenteTruthFinder",
    "BasicTruthFinder": "BasicTruthFinder",
    "SenFeedTruthFinder-CompositeTD": "SenFeedTruthFinder",
}
BATCH_STYLES = {
    10: {"color": "#8BD17C", "hatch": "||", "label": "batch = 10"},
    50: {"color": "#66C2A5", "hatch": "\\", "label": "batch = 50"},
    90: {"color": "#B9BDC7", "hatch": "//", "label": "batch = 90"},
    130: {"color": "#8E95A5", "hatch": "xx", "label": "batch = 130"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw grouped bar charts from hybrid batch results without legend overlap."
    )
    parser.add_argument("--input", type=Path, default=RESULT_PATH, help="Path to the text result file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for exported figures.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG export DPI.")
    return parser.parse_args()


def parse_result_file(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    rows: list[dict[str, object]] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        batch_match = re.search(r"batch\s*=\s*(\d+)", lines[0])
        if batch_match is None:
            raise ValueError(f"Unable to parse batch header: {lines[0]!r}")
        batch = int(batch_match.group(1))
        methods = re.split(r"\s{2,}", lines[1].strip())

        for line in lines[3:]:
            parts = re.split(r"\s{2,}", line.strip())
            node = parts[0]
            values = parts[1:]
            if len(values) != len(methods):
                raise ValueError(
                    f"Batch {batch} row {node!r} has {len(values)} values, expected {len(methods)}."
                )
            for method, value in zip(methods, values):
                rows.append(
                    {
                        "batch": batch,
                        "node": node,
                        "method": method,
                        "score": float(value),
                    }
                )

    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return pd.DataFrame(rows)


def summarize_selected_means(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["batch", "method"], as_index=False).agg(mean_score=("score", "mean"))
    summary = summary[summary["batch"].isin(SELECTED_BATCHES)].copy()
    methods = [method for method in METHOD_ORDER if method in summary["method"].unique()]
    summary["method"] = pd.Categorical(summary["method"], categories=methods, ordered=True)
    summary = summary.sort_values(["method", "batch"])

    expected_rows = len(methods) * len(SELECTED_BATCHES)
    if len(summary) != expected_rows:
        missing = []
        for method in methods:
            for batch in SELECTED_BATCHES:
                mask = (summary["method"] == method) & (summary["batch"] == batch)
                if not mask.any():
                    missing.append((method, batch))
        raise ValueError(f"Missing selected batch data: {missing}")
    return summary


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 13,
            "axes.linewidth": 1.1,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def method_sequence(summary: pd.DataFrame) -> list[str]:
    present = list(summary["method"].astype(str).drop_duplicates())
    preferred = [method for method in METHOD_ORDER if method in present]
    return preferred + [method for method in present if method not in preferred]


def compute_y_axis_upper(summary: pd.DataFrame) -> float:
    max_score = float(summary["mean_score"].max())
    padded_upper = np.ceil((max_score + 8.0) / 2.0) * 2.0
    return float(max(92.0, min(110.0, padded_upper)))


def plot_figure(summary: pd.DataFrame, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    methods = method_sequence(summary)
    x_positions = np.arange(len(methods))
    bar_width = 0.18
    y_upper = compute_y_axis_upper(summary)

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.79)

    for index, batch in enumerate(SELECTED_BATCHES):
        style = BATCH_STYLES[batch]
        subset = summary[summary["batch"] == batch].set_index("method").reindex(methods)
        positions = x_positions + (index - 1.5) * bar_width
        ax.bar(
            positions,
            subset["mean_score"],
            width=bar_width,
            color=style["color"],
            edgecolor="#1A1A1A",
            linewidth=1.05,
            hatch=style["hatch"],
            label=style["label"],
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods])
    ax.set_xlabel("Methods")
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, y_upper)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="both", width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(SELECTED_BATCHES),
        fontsize=9,
        frameon=True,
        borderpad=0.38,
        labelspacing=0.35,
        handlelength=1.9,
        handletextpad=0.55,
        columnspacing=0.95,
    )
    legend.get_frame().set_edgecolor("#CFCFCF")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.96)
    legend.get_frame().set_facecolor("white")

    png_path = output_dir / f"{OUTPUT_BASENAME}.png"
    pdf_path = output_dir / f"{OUTPUT_BASENAME}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def print_analysis(summary: pd.DataFrame, input_path: Path) -> None:
    pivot = summary.pivot(index="method", columns="batch", values="mean_score")
    pivot = pivot.reindex([method for method in METHOD_ORDER if method in pivot.index])
    print(f"Input file: {input_path}")
    print("Selected batch means:")
    print(pivot.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    df = parse_result_file(args.input)
    summary = summarize_selected_means(df)
    png_path, pdf_path = plot_figure(summary, args.output_dir, args.dpi)
    print_analysis(summary, args.input)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
