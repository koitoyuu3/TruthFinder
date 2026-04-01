"""
实验2-数值: 随着 Batch 增加的前缀累计收敛过程
================================================
自变量: batch 序号 b = 1, 2, ..., N
场景设定: 固定 40% 恶意节点（最严苛场景）
目的: 观察随着数据积累，各方法在前缀数据上的累计精度变化

评估框架:
  对每个前缀 b = 1..N，重新构造并运行前缀数据集：
    数值场景 → RMSE(preds_1..b, gt_1..b), MAE(...)
  展示前缀累计曲线（随着 prefix 扩大，累计指标如何变化）

运行: python3 exp2_numeric.py
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp_utils import (
    COLORS,
    NUM_METHODS,
    build_gt_numeric,
    eval_numeric,
    get_numeric_demo_files,
    load_batches,
    to_numeric_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp2_numeric"

BAD_COUNT = 4
TITLE = "E2-Numeric: 收敛过程（前缀累计）"
X_LABEL = "Batch 序号"


def plot_convergence(
    batch_indices: List[int],
    results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in NUM_METHODS:
        kw = dict(color=COLORS[method], label=method, linewidth=1.5, alpha=0.85)
        ax1.plot(batch_indices, results[method]["rmse"], **kw)
        ax2.plot(batch_indices, results[method]["mae"], **kw)

    for ax, ylabel, suffix in [
        (ax1, "RMSE（前缀累计 1..b）", "RMSE 收敛曲线（数值场景）"),
        (ax2, "MAE（前缀累计 1..b）", "MAE 收敛曲线（数值场景）"),
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


def save_results(
    batch_indices: List[int],
    num_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, batch_index in enumerate(batch_indices):
        for method in NUM_METHODS:
            rows.append({
                "batch_index": batch_index,
                "method": method,
                "rmse": num_results[method]["rmse"][i] if i < len(num_results[method]["rmse"]) else float("nan"),
                "mae": num_results[method]["mae"][i] if i < len(num_results[method]["mae"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验2-数值: 随着 Batch 增加的前缀累计收敛过程")
    parser.add_argument(
        "--attack-profile",
        default="agent",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="恶意节点数量。",
    )
    args = parser.parse_args()

    num_files = get_numeric_demo_files()
    if args.bad_count not in num_files:
        available = ", ".join(str(k) for k in sorted(num_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for numeric files. "
            f"Available bad_count values: {available}"
        )
    out_dir = BASE_OUT_DIR / f"bad_{args.bad_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(f"dataset = numeric:num_demo_*bad_agent.json, bad_count = {args.bad_count}")

    clean_num_batches = load_batches(num_files[0])
    mal_num_batches = load_batches(num_files[args.bad_count])

    n_batches = min(len(clean_num_batches), len(mal_num_batches))
    batch_indices = list(range(1, n_batches + 1))

    num_results = {method: {"rmse": [], "mae": []} for method in NUM_METHODS}

    for k in batch_indices:
        print(f"\n  prefix batches 1..{k}")
        gt_num = build_gt_numeric(clean_num_batches, max_batches=k)
        noisy_num_df = to_numeric_df(mal_num_batches, max_batches=k)

        res_num = eval_numeric(noisy_num_df, gt_num, last_pct=1.0)
        for method in NUM_METHODS:
            num_results[method]["rmse"].append(res_num[method]["rmse"])
            num_results[method]["mae"].append(res_num[method]["mae"])

    plot_convergence(batch_indices, num_results, out_dir)
    save_results(batch_indices, num_results, out_dir)

    print(f"\n[E2-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
