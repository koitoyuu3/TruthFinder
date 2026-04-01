"""
实验2: 随着 Batch 增加的前缀累计收敛过程
================================================
自变量: batch 序号 b = 1, 2, ..., N
场景设定: 固定 40% 恶意节点（最严苛场景）
目的: 观察随着数据积累，各方法在前缀数据上的累计精度变化

评估框架:
  对每个前缀 b = 1..N，重新构造并运行前缀数据集：
    数值场景 → RMSE(preds_1..b, gt_1..b), MAE(...)
    混合场景 → DataAccuracy(X_clean_1..b, Y_noisy_1..b)
  展示前缀累计曲线（随着 prefix 扩大，累计指标如何变化）

运行: python3 exp2_batch_variation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── 路径 & 共享工具 ────────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp_utils import (
    NUM_METHODS, TEXT_METHODS,
    COLORS,
    get_numeric_demo_files,
    get_text_demo_files,
    load_batches,
    to_numeric_df, to_text_df,
    build_gt_numeric,
    eval_numeric, eval_text,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp2_batch_variation"

BAD_COUNT = 4
TITLE     = "E2: 收敛过程（前缀累计）"
X_LABEL   = "Batch 序号"

# ── 绘图 ─────────────────────────────────────────────────────────────────────

def plot_convergence(
    batch_indices: List[int],
    results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for m in NUM_METHODS:
        kw = dict(color=COLORS[m], label=m, linewidth=1.5, alpha=0.85)
        ax1.plot(batch_indices, results[m]["rmse"], **kw)
        ax2.plot(batch_indices, results[m]["mae"],  **kw)

    for ax, ylabel, suffix in [
        (ax1, "RMSE（前缀累计 1..b）", "RMSE 收敛曲线（数值场景）"),
        (ax2, "MAE（前缀累计 1..b）",  "MAE 收敛曲线（数值场景）"),
    ]:
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{TITLE} — {suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "numeric_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 已保存 {path}")


def plot_text_convergence(
    batch_indices: List[int],
    results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in TEXT_METHODS:
        ax.plot(batch_indices, results[m]["data_accuracy"],
                color=COLORS[m], label=m, linewidth=1.5, alpha=0.85)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Data Accuracy（前缀累计）")
    ax.set_title(f"{TITLE} — 文本 DataAccuracy 收敛曲线")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "text_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 已保存 {path}")


def save_results(
    batch_indices: List[int],
    num_results: Dict,
    txt_results: Dict,
    out_dir: Path,
) -> None:
    rows = []
    for i, bi in enumerate(batch_indices):
        for m in NUM_METHODS:
            rows.append({
                "batch_index": bi,
                "method": m,
                "scene": "numeric",
                "rmse": num_results[m]["rmse"][i] if i < len(num_results[m]["rmse"]) else float("nan"),
                "mae":  num_results[m]["mae"][i]  if i < len(num_results[m]["mae"])  else float("nan"),
                "data_accuracy": float("nan"),
            })
        for m in TEXT_METHODS:
            rows.append({
                "batch_index": bi,
                "method": m,
                "scene": "mixed_text",
                "rmse": float("nan"),
                "mae":  float("nan"),
                "data_accuracy": txt_results[m]["data_accuracy"][i] if i < len(txt_results[m]["data_accuracy"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="实验2: 随着 Batch 增加的前缀累计收敛过程")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json，文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="恶意节点数量。",
    )
    args = parser.parse_args()

    num_files = get_numeric_demo_files()
    text_files = get_text_demo_files()
    if args.bad_count not in num_files:
        available = ", ".join(str(k) for k in sorted(num_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for numeric files. "
            f"Available bad_count values: {available}"
        )
    if args.bad_count not in text_files:
        available = ", ".join(str(k) for k in sorted(text_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for text files. "
            f"Available bad_count values: {available}"
        )
    out_dir = BASE_OUT_DIR / "demo_agent_text_collude" / f"bad_{args.bad_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(f"datasets = numeric:num_demo_*bad_agent.json, text:demo_*bad_collude.json, bad_count = {args.bad_count}")

    clean_num_batches = load_batches(num_files[0])
    mal_num_batches   = load_batches(num_files[args.bad_count])
    clean_txt_batches = load_batches(text_files[0])
    mal_txt_batches   = load_batches(text_files[args.bad_count])

    n_batches = min(
        len(clean_num_batches),
        len(mal_num_batches),
        len(clean_txt_batches),
        len(mal_txt_batches),
    )
    batch_indices = list(range(1, n_batches + 1))

    num_results = {m: {"rmse": [], "mae": []} for m in NUM_METHODS}
    txt_results = {m: {"data_accuracy": []} for m in TEXT_METHODS}

    for k in batch_indices:
        print(f"\n  prefix batches 1..{k}")

        gt_num = build_gt_numeric(clean_num_batches, max_batches=k)
        noisy_num_df = to_numeric_df(mal_num_batches, max_batches=k)
        clean_txt_df = to_text_df(clean_txt_batches, max_batches=k)
        noisy_txt_df = to_text_df(mal_txt_batches, max_batches=k)

        # ── 数值场景：在前缀 1..k 上重跑并累计评估 ───────────────────────────
        res_num = eval_numeric(noisy_num_df, gt_num, last_pct=1.0)
        for m in NUM_METHODS:
            num_results[m]["rmse"].append(res_num[m]["rmse"])
            num_results[m]["mae"].append(res_num[m]["mae"])

        # ── 文本场景：在前缀 1..k 上重跑并累计评估 ───────────────────────────
        if not clean_txt_df.empty and not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(res_txt[m]["data_accuracy"])
        else:
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(float("nan"))

    # ── 输出 ──────────────────────────────────────────────────────────────────
    plot_convergence(batch_indices, num_results, out_dir)
    plot_text_convergence(batch_indices, txt_results, out_dir)
    save_results(batch_indices, num_results, txt_results, out_dir)

    print(f"\n[E2] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
