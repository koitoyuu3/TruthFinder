#!/usr/bin/env python3
"""Regenerate mixed_text.png in each of the 4 affected output dirs from the
updated text_data_accuracy.csv. Uses the same style as
exp1_text_50agent_0_80.py:_plot_text."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 0.90
ROOT = Path(
    "/Users/huabowen/TruthFinder/decentralized_truth_discovery/examples/outputs/exp1_text"
)
DIRS = [
    "1.2-agent-50-news-300-50%-0.9-numonly",
    "1.2-llama-news-300-50%-0.9-numonly",
    "1.4-agent-50-weather-300-50%-0.9-numonly",
    "1.4-llama-weather-300-50%-0.9-numonly",
]


def replot(out_dir: Path) -> None:
    csv_path = out_dir / "text_data_accuracy.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    x_vals = [float(r["malicious_ratio_pct"]) for r in rows]
    results = {
        m: [float(r[m]) for r in rows] for m in ("SenteTruth", "BasicTruth", "SemanticTruth")
    }

    plt.rcParams["font.family"] = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "Songti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "Songti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"SenteTruth": "#2ecc71", "BasicTruth": "#e67e22", "SemanticTruth": "#9b59b6"}
    markers = {"SenteTruth": "s", "BasicTruth": "D", "SemanticTruth": "*"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ("SenteTruth", "BasicTruth", "SemanticTruth"):
        ax.plot(
            x_vals,
            results[m],
            color=colors[m],
            marker=markers[m],
            label=m,
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xlabel("恶意节点比例 (%)")
    ax.set_ylabel(f"Data Accuracy (hybrid score ≥ {THRESHOLD})")
    ax.set_title("E1-Text: 恶意节点比例 — 文本数据准确率（混合场景）")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "mixed_text.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def main() -> None:
    for name in DIRS:
        replot(ROOT / name)


if __name__ == "__main__":
    main()
