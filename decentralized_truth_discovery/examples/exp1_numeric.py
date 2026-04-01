from __future__ import annotations

"""
实验1-数值: 改变恶意节点比例（逐 batch RMSE）
================================================
自变量: 恶意节点数 k ∈ {0,1,2,3,4} → 比例 {0,10,20,30,40}%
数据文件:
  collude（默认）→ num_demo.json / num_demo_1~4bad_collude.json
                   所有恶意节点使用同一 profile（协同攻击）
  rotate         → num_demo.json / num_demo_1~6bad_agent.json
固定: 使用全部 batch，全部 10 个 agent

评估框架:
  preds_b = 方法在 demo_Kbad_collude.json 的第 b 个 batch 上的输出
  gt_b    = demo.json（0 恶意）第 b 个 batch 中的真实值
  指标    = RMSE(preds_b, gt_b)
          若 metric-scale=relative（默认），先按每个币种的 |gt_b,o|
          归一化误差，再聚合为 batch 级 RMSE

输出:
  - numeric_per_batch_rmse.png
  - numeric_per_batch_rmse.csv

运行: python exp1_numeric.py [--attack-mode collude|rotate] [--metric-scale raw|relative]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Arial Unicode MS"]
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

from exp_utils import (
    COLORS,
    MARKERS,
    N_AGENTS,
    NUM_METHODS,
    build_gt_numeric,
    eval_numeric_per_batch,
    get_numeric_demo_files,
    load_batches,
    to_numeric_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp1_numeric"
TITLE = "E1-Numeric: 恶意节点比例"


def plot_numeric_per_batch(
    results_df: pd.DataFrame,
    bad_counts: list[int],
    out_dir: Path,
    metric_scale: str,
) -> None:
    """绘制每个 bad_count 一个子图的逐 batch RMSE 曲线。"""
    n_plots = len(bad_counts)
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    is_relative = metric_scale == "relative"
    metric_label = "归一化单 batch RMSE" if is_relative else "单 batch RMSE"

    finite_rmse = results_df["rmse"].dropna()
    y_max = float(finite_rmse.max()) if not finite_rmse.empty else None

    for idx, bad_k in enumerate(bad_counts):
        ax = axes[idx]
        subset = results_df[results_df["bad_count"] == bad_k]
        for method in NUM_METHODS:
            method_df = subset[subset["method"] == method].sort_values("batch_idx")
            if method_df.empty:
                continue
            ax.plot(
                method_df["batch_idx"],
                method_df["rmse"],
                color=COLORS[method],
                marker=MARKERS[method],
                label=method,
                linewidth=1.6,
                markersize=4,
            )
        ratio = bad_k / N_AGENTS * 100
        ax.set_title(f"bad_count={bad_k} ({ratio:.0f}%)")
        ax.grid(True, alpha=0.3)
        if y_max is not None:
            ax.set_ylim(0.0, y_max * 1.05 if y_max > 0 else 1.0)

    for ax in axes[len(bad_counts):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(NUM_METHODS), frameon=False)

    fig.suptitle(f"{TITLE} — 每个恶意节点比例下的{metric_label}", y=0.98)
    fig.text(0.5, 0.03, "batch 序号", ha="center")
    fig.text(0.02, 0.5, metric_label, va="center", rotation="vertical")
    fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.93))

    path = out_dir / "numeric_per_batch_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1-数值: 改变恶意节点比例（逐 batch RMSE）")
    parser.add_argument(
        "--attack-mode",
        choices=["collude", "rotate"],
        default="collude",
        help="collude: 协同攻击（默认）; rotate: 4-profile 轮换攻击",
    )
    parser.add_argument(
        "--metric-scale",
        choices=["raw", "relative"],
        default="relative",
        help="raw: 原始绝对误差; relative: 按每个币种的 |ground truth| 归一化后再算 RMSE（默认）",
    )
    args = parser.parse_args()

    num_files = get_numeric_demo_files(mode=args.attack_mode)
    bad_counts = sorted(num_files)
    out_dir = BASE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize_metric = args.metric_scale == "relative"

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(f"dataset = numeric, attack_mode={args.attack_mode}, metric_scale={args.metric_scale}")

    clean_num_batches = load_batches(num_files[0])
    gt_num = build_gt_numeric(clean_num_batches)

    rows = []
    for bad_k in bad_counts:
        ratio = bad_k / N_AGENTS * 100
        print(f"\n  bad_count={bad_k}  ({ratio:.0f}% 恶意)")
        num_batches = load_batches(num_files[bad_k])
        noisy_num_df = to_numeric_df(num_batches)
        per_batch_results = eval_numeric_per_batch(noisy_num_df, gt_num, normalize=normalize_metric)

        for method in NUM_METHODS:
            for batch_idx, rmse in sorted(per_batch_results[method].items()):
                rows.append({
                    "bad_count": bad_k,
                    "method": method,
                    "batch_idx": batch_idx,
                    "rmse": rmse,
                    "metric_scale": args.metric_scale,
                })

    results_df = pd.DataFrame(rows, columns=["bad_count", "method", "batch_idx", "rmse", "metric_scale"])
    csv_path = out_dir / "numeric_per_batch_rmse.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {csv_path}")

    plot_numeric_per_batch(results_df, bad_counts, out_dir, metric_scale=args.metric_scale)
    print(f"\n[E1-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
