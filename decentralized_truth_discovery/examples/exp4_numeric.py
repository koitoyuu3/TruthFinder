"""
实验4-数值: 随着 Agent 数量变化
================================================
自变量: 总 agent 数 N ∈ {3, 4, 5, 6, 7, 8, 9, 10}
场景设定:
  固定 1 个恶意节点（agent-1），总数 N 变化
  → 实际恶意比例随 N 从 33% 降至 10%
  → 节点越多，诚实节点比例越高，方法 RMSE 应逐渐降低

数据过滤: 只保留 agent-1..agent-N 的数据

评估框架（对齐论文 Section VI）:
  preds = 方法在 num_demo_1bad_agent.json（agent subset, 1 恶意）上的输出
  gt    = num_demo.json 中的真实值（价格与 agent subset 无关）
  数值场景 → RMSE(preds_last10%, gt_last10%), MAE(...)

运行: python3 exp4_numeric.py
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from exp_utils import (
    N_AGENTS,
    NUM_METHODS,
    build_gt_numeric,
    eval_numeric,
    get_numeric_demo_files,
    load_batches,
    plot_numeric,
    to_numeric_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp4_numeric"

BAD_COUNT = 1
TITLE = "E4-Numeric: Agent 数量变化"
X_LABEL = "Agent 总数"
MIN_AGENTS = 3


def save_results(
    agent_range: List[int],
    num_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, agent_count in enumerate(agent_range):
        for method in NUM_METHODS:
            rows.append({
                "agent_count": agent_count,
                "method": method,
                "rmse": num_results[method]["rmse"][i] if i < len(num_results[method]["rmse"]) else float("nan"),
                "mae": num_results[method]["mae"][i] if i < len(num_results[method]["mae"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验4-数值: 随着 Agent 数量变化")
    parser.add_argument(
        "--attack-profile",
        default="agent",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="固定恶意节点数量。",
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
    gt_num = build_gt_numeric(clean_num_batches)

    all_agents = [f"agent-{i}" for i in range(1, N_AGENTS + 1)]
    agent_range = list(range(MIN_AGENTS, N_AGENTS + 1))

    num_results = {method: {"rmse": [], "mae": []} for method in NUM_METHODS}

    for n_agents in agent_range:
        subset = all_agents[:n_agents]
        ratio = 1.0 / n_agents * 100
        print(f"\n  n_agents={n_agents}  恶意比例≈{ratio:.1f}%  subset={subset}")

        noisy_num_df = to_numeric_df(mal_num_batches, agent_subset=subset)
        res_num = eval_numeric(noisy_num_df, gt_num)
        for method in NUM_METHODS:
            num_results[method]["rmse"].append(res_num[method]["rmse"])
            num_results[method]["mae"].append(res_num[method]["mae"])

    plot_numeric(agent_range, num_results, X_LABEL, TITLE, out_dir)
    save_results(agent_range, num_results, out_dir)

    print(f"\n[E4-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
