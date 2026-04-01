"""
实验3-数值: 随着恶意节点注入时间变化
================================================
自变量: 注入起始 batch t ∈ {1, 2, ..., N}
场景设定:
  batch_index < t  → 使用干净数据（num_demo.json）
  batch_index >= t → 使用恶意数据（num_demo_4bad_agent.json，40% 恶意节点）

评估框架（对齐论文 Section VI）:
  preds = 方法在注入合成数据集上的输出
  gt    = num_demo.json（全程干净）中的真实值
  数值场景 → RMSE(preds_all, gt_all), MAE(...)

运行: python exp3_numeric.py
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from exp_utils import (
    NUM_METHODS,
    build_gt_numeric,
    build_injection_df,
    eval_numeric,
    get_numeric_demo_files,
    load_batches,
    plot_numeric,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_numeric"

BAD_COUNT = 4
TITLE = "E3-Numeric: 恶意节点注入时间"
X_LABEL = "恶意注入起始 Batch"


def save_results(
    injection_points: List[int],
    num_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, injection_start in enumerate(injection_points):
        for method in NUM_METHODS:
            rows.append({
                "injection_start_batch": injection_start,
                "method": method,
                "rmse": num_results[method]["rmse"][i] if i < len(num_results[method]["rmse"]) else float("nan"),
                "mae": num_results[method]["mae"][i] if i < len(num_results[method]["mae"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3-数值: 随着恶意节点注入时间变化")
    parser.add_argument(
        "--attack-profile",
        default="agent",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="注入后使用的恶意节点数量。",
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
    injection_points = list(range(1, n_batches + 1))

    gt_num = build_gt_numeric(clean_num_batches)
    num_results = {method: {"rmse": [], "mae": []} for method in NUM_METHODS}

    for injection_start in injection_points:
        print(
            f"\n  注入起始 batch = {injection_start}  "
            f"（前 {injection_start - 1} 个干净，后 {n_batches - injection_start + 1} 个恶意）"
        )
        noisy_num_df = build_injection_df(
            clean_num_batches,
            mal_num_batches,
            inj_start=injection_start,
            n_total=n_batches,
            mode="numeric",
        )
        res_num = eval_numeric(noisy_num_df, gt_num, last_pct=1.0)
        for method in NUM_METHODS:
            num_results[method]["rmse"].append(res_num[method]["rmse"])
            num_results[method]["mae"].append(res_num[method]["mae"])

    plot_numeric(
        injection_points,
        num_results,
        X_LABEL,
        TITLE,
        out_dir,
        metric_scope_label="all batches",
    )
    save_results(injection_points, num_results, out_dir)

    print(f"\n[E3-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
