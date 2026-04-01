"""
实验2-文本: 随着 Batch 增加的前缀累计收敛过程
================================================
自变量: batch 序号 b = 1, 2, ..., N
场景设定: 固定 40% 恶意节点（最严苛场景）
目的: 观察随着数据积累，各方法在前缀数据上的累计精度变化

评估框架:
  对每个前缀 b = 1..N，重新构造并运行前缀数据集：
    混合场景 → DataAccuracy(X_clean_1..b, Y_noisy_1..b)
  展示前缀累计曲线（随着 prefix 扩大，累计指标如何变化）

运行: python3 exp2_text.py
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
    TEXT_METHODS,
    eval_text,
    get_text_demo_files,
    load_batches,
    to_text_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp2_text"

BAD_COUNT = 4
TITLE = "E2-Text: 收敛过程（前缀累计）"
X_LABEL = "Batch 序号"


def plot_text_convergence(
    batch_indices: List[int],
    results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in TEXT_METHODS:
        ax.plot(
            batch_indices,
            results[method]["data_accuracy"],
            color=COLORS[method],
            label=method,
            linewidth=1.5,
            alpha=0.85,
        )
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
    txt_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, batch_index in enumerate(batch_indices):
        for method in TEXT_METHODS:
            rows.append({
                "batch_index": batch_index,
                "method": method,
                "data_accuracy": txt_results[method]["data_accuracy"][i] if i < len(txt_results[method]["data_accuracy"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验2-文本: 随着 Batch 增加的前缀累计收敛过程")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="恶意节点数量。",
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

    n_batches = min(len(clean_txt_batches), len(mal_txt_batches))
    batch_indices = list(range(1, n_batches + 1))

    txt_results = {method: {"data_accuracy": []} for method in TEXT_METHODS}

    for k in batch_indices:
        print(f"\n  prefix batches 1..{k}")
        clean_txt_df = to_text_df(clean_txt_batches, max_batches=k)
        noisy_txt_df = to_text_df(mal_txt_batches, max_batches=k)

        if not clean_txt_df.empty and not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(res_txt[method]["data_accuracy"])
        else:
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(float("nan"))

    plot_text_convergence(batch_indices, txt_results, out_dir)
    save_results(batch_indices, txt_results, out_dir)

    print(f"\n[E2-Text] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
