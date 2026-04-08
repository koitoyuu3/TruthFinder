"""
实验1-文本: 改变恶意节点比例
================================================
自变量: 恶意节点数 k ∈ {0,1,2,3,4} → 比例 {0,10,20,30,40}%
数据文件:
  文本   → demo.json / demo_1~4bad_collude.json
固定: 使用全部 batch，全部 10 个 agent

评估框架（对齐论文 Section VI）:
  X = 方法在 demo.json（0 恶意）上的输出（ground truth 代理）
  Y = 方法在 demo_Kbad_collude.json 上的输出（预测结果）
  混合场景 → DataAccuracy(X_clean, Y_noisy)

运行: python exp1_text.py
"""
import argparse
from pathlib import Path

from exp_utils import (
    N_AGENTS,
    TEXT_METHODS,
    get_text_demo_files,
    get_shared_bad_counts,
    load_batches,
    to_text_df,
    eval_text,
    plot_text,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp1_text"

TITLE = "E1-Text: 恶意节点比例"
X_LABEL = "恶意节点比例 (%)"


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1-文本: 改变恶意节点比例")
    parser.parse_args()

    text_files = get_text_demo_files()
    bad_counts = get_shared_bad_counts()
    out_dir = BASE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print("dataset = text:demo_*bad_collude.json")

    clean_txt_batches = load_batches(text_files[0])
    clean_txt_df = to_text_df(clean_txt_batches)

    x_vals = [k / N_AGENTS * 100 for k in bad_counts]
    txt_results = {m: {"data_accuracy": []} for m in TEXT_METHODS}

    for bad_k, ratio in zip(bad_counts, x_vals):
        print(f"\n  bad_count={bad_k}  ({ratio:.0f}% 恶意)")
        txt_batches = load_batches(text_files[bad_k])
        noisy_txt_df = to_text_df(txt_batches)

        res_txt = eval_text(clean_txt_df, noisy_txt_df)
        for method in TEXT_METHODS:
            txt_results[method]["data_accuracy"].append(res_txt[method]["data_accuracy"])

    plot_text(x_vals, txt_results, X_LABEL, TITLE, out_dir)
    print(f"\n[E1-Text] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
