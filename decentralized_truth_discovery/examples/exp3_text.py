"""
实验3-文本: 随着恶意节点注入时间变化
================================================
自变量: 注入起始 batch t ∈ {1, 2, ..., N}
场景设定:
  batch_index < t  → 使用干净数据（demo.json）
  batch_index >= t → 使用恶意数据（demo_4bad_collude.json，40% 恶意节点）

评估框架（对齐论文 Section VI）:
  X = 方法在干净数据上的输出（ground truth 代理）
  Y = 方法在注入合成数据集上的输出（预测结果）
  混合场景 → DataAccuracy(X_clean, Y_noisy)

运行: python exp3_text.py
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from exp_utils import (
    TEXT_METHODS,
    build_injection_df,
    eval_text,
    get_text_demo_files,
    load_batches,
    plot_text,
    to_text_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_text"

BAD_COUNT = 4
TITLE = "E3-Text: 恶意节点注入时间"
X_LABEL = "恶意注入起始 Batch"


def save_results(
    injection_points: List[int],
    txt_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    rows = []
    for i, injection_start in enumerate(injection_points):
        for method in TEXT_METHODS:
            rows.append({
                "injection_start_batch": injection_start,
                "method": method,
                "data_accuracy": txt_results[method]["data_accuracy"][i] if i < len(txt_results[method]["data_accuracy"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3-文本: 随着恶意节点注入时间变化")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="注入后使用的恶意节点数量。",
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
    injection_points = list(range(1, n_batches + 1))

    clean_txt_df = to_text_df(clean_txt_batches)
    txt_results = {method: {"data_accuracy": []} for method in TEXT_METHODS}

    for injection_start in injection_points:
        print(
            f"\n  注入起始 batch = {injection_start}  "
            f"（前 {injection_start - 1} 个干净，后 {n_batches - injection_start + 1} 个恶意）"
        )
        noisy_txt_df = build_injection_df(
            clean_txt_batches,
            mal_txt_batches,
            inj_start=injection_start,
            n_total=n_batches,
            mode="text",
        )
        if not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(res_txt[method]["data_accuracy"])
        else:
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(float("nan"))

    plot_text(injection_points, txt_results, X_LABEL, TITLE, out_dir)
    save_results(injection_points, txt_results, out_dir)

    print(f"\n[E3-Text] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
