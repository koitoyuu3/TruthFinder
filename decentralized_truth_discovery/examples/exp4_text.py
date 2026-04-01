"""
实验4-文本: 随着 Agent 数量变化
================================================
自变量: 总 agent 数 N ∈ {3, 4, 5, 6, 7, 8, 9, 10}
场景设定:
  固定 1 个恶意节点（agent-1），总数 N 变化
  → 实际恶意比例随 N 从 33% 降至 10%

数据过滤: 只保留 agent-1..agent-N 的数据

评估框架（对齐论文 Section VI）:
  X = 方法在 demo.json（agent subset）上的输出（ground truth 代理）
  Y = 方法在 demo_1bad_collude.json（agent subset）上的输出（预测结果）
  混合场景 → DataAccuracy(X_clean, Y_noisy)

运行: python3 exp4_text.py
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from exp_utils import (
    N_AGENTS,
    TEXT_METHODS,
    eval_text,
    get_text_demo_files,
    load_batches,
    plot_text,
    to_text_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp4_text"

BAD_COUNT = 1
TITLE = "E4-Text: Agent 数量变化"
X_LABEL = "Agent 总数"
MIN_AGENTS = 3


def save_results(
    agent_range: List[int],
    txt_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, agent_count in enumerate(agent_range):
        for method in TEXT_METHODS:
            rows.append({
                "agent_count": agent_count,
                "method": method,
                "data_accuracy": txt_results[method]["data_accuracy"][i] if i < len(txt_results[method]["data_accuracy"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验4-文本: 随着 Agent 数量变化")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="固定恶意节点数量。",
    )
    args = parser.parse_args()

    text_files = get_text_demo_files()
    if args.bad_count not in text_files:
        available = ", ".join(str(k) for k in sorted(text_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for text files. "
            f"Available bad_count values: {available}"
        )
    out_dir = BASE_OUT_DIR / f"bad_{args.bad_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(f"dataset = text:demo_*bad_collude.json, bad_count = {args.bad_count}")

    clean_txt_batches = load_batches(text_files[0])
    mal_txt_batches = load_batches(text_files[args.bad_count])

    all_agents = [f"agent-{i}" for i in range(1, N_AGENTS + 1)]
    agent_range = list(range(MIN_AGENTS, N_AGENTS + 1))

    txt_results = {method: {"data_accuracy": []} for method in TEXT_METHODS}

    for n_agents in agent_range:
        subset = all_agents[:n_agents]
        ratio = 1.0 / n_agents * 100
        print(f"\n  n_agents={n_agents}  恶意比例≈{ratio:.1f}%  subset={subset}")

        clean_txt_df = to_text_df(clean_txt_batches, agent_subset=subset)
        noisy_txt_df = to_text_df(mal_txt_batches, agent_subset=subset)

        if not clean_txt_df.empty and not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(res_txt[method]["data_accuracy"])
        else:
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(float("nan"))

    plot_text(agent_range, txt_results, X_LABEL, TITLE, out_dir)
    save_results(agent_range, txt_results, out_dir)

    print(f"\n[E4-Text] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
