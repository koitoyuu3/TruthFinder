from __future__ import annotations

"""
实验1-数值: 改变恶意节点比例（逐 batch RMSE）
================================================
自变量: 恶意节点数 k ∈ {0,1,2,3,4} → 比例 {0,10,20,30,40}%
数据文件:
  collude（默认）→ num_demo.json / num_demo_1~4bad_collude.json
                   恶意节点围绕固定错误中心协同，恶意内部/ honest 侧都带
                   固定的 agent 级确定性扰动，整体环境对时间静态
  rotate         → num_demo.json / num_demo_1~4bad_agent.json
固定: 使用全部 batch，全部 10 个 agent

评估框架:
  preds_b = 方法在 num_demo_Kbad_*.json 的第 b 个 batch 上的输出
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


def plot_bar_by_ratio(
    results_df: pd.DataFrame,
    bad_counts: list[int],
    out_dir: Path,
    metric_scale: str,
) -> None:
    """绘制分组条形图：横轴为恶意节点占比，纵轴为平均 RMSE，每组含五个方法。"""
    import numpy as np

    is_relative = metric_scale == "relative"
    metric_label = "归一化 RMSE" if is_relative else "RMSE"

    # 聚合：对每个 (bad_count, method) 求 batch 均值
    mean_df = (
        results_df.groupby(["bad_count", "method"])["rmse"]
        .mean()
        .reset_index()
        .rename(columns={"rmse": "mean_rmse"})
    )

    ratios = [f"{int(k / N_AGENTS * 100)}%" for k in bad_counts]
    n_groups = len(bad_counts)
    n_methods = len(NUM_METHODS)
    bar_width = 0.13
    group_gap = 0.18  # 组间额外间距
    group_width = n_methods * bar_width + group_gap
    x = np.arange(n_groups) * group_width

    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(max(16, n_groups * 2.2), 6))

    y_max = mean_df["mean_rmse"].max()
    y_pad = y_max * 0.012  # 标注与条顶的间距

    # 收集所有标注位置用于碰撞检测 (x_center, y_bottom, y_top)
    placed: list[tuple[float, float, float]] = []
    label_h = y_max * 0.032  # 估算横向文字占据的纵向高度

    # 绘制每个方法的条形
    for i, method in enumerate(NUM_METHODS):
        method_data = mean_df[mean_df["method"] == method]
        vals = []
        for bad_k in bad_counts:
            row = method_data[method_data["bad_count"] == bad_k]
            vals.append(float(row["mean_rmse"].iloc[0]) if not row.empty else 0.0)

        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.90,
            color=COLORS[method],
            label=method,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        # 在每根条上方标注数值（横向，跳过 0，避免重叠）
        for bar, val in zip(bars, vals):
            if val <= 1e-9:
                continue
            x_c = bar.get_x() + bar.get_width() / 2
            # 从条顶 + 间距开始，若与已放置标注重叠则上移
            y_try = bar.get_height() + y_pad
            for (_, py_bot, py_top) in placed:
                if py_bot < y_try + label_h and y_try < py_top:
                    y_try = py_top + y_pad * 0.5
            placed.append((x_c, y_try, y_try + label_h))
            ax.text(
                x_c,
                y_try,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#333333",
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=11)
    ax.set_xlabel("恶意节点占比 (bad_count / N_agents)", fontsize=12, labelpad=8)
    ax.set_ylabel(metric_label, fontsize=12, labelpad=8)
    ax.set_title(f"E1-Numeric: 不同恶意节点占比下各方法的平均 {metric_label}", fontsize=13, pad=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, y_max * 1.40 if y_max > 0 else 1.0)

    legend = ax.legend(
        title="方法",
        title_fontsize=10,
        fontsize=9.5,
        loc="upper left",
        framealpha=0.85,
        edgecolor="#cccccc",
        ncol=1,
    )

    fig.tight_layout()

    path = out_dir / "numeric_per_batch_rmse.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
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
    bad_counts = [k for k in sorted(num_files) if k <= 4]
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

    plot_bar_by_ratio(results_df, bad_counts, out_dir, metric_scale=args.metric_scale)
    print(f"\n[E1-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
