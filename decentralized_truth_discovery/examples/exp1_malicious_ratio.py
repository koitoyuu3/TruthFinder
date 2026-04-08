"""
实验1: 改变恶意节点比例
================================================
自变量: 恶意节点数 k ∈ {0,1,2,3,4} → 比例 {0,10,20,30,40}%
数据文件:
  数值   → num_demo.json / num_demo_1~4bad_agent.json
  文本   → demo.json / demo_1~4bad_collude.json
固定: 使用全部 batch，全部 10 个 agent

评估框架（对齐论文 Section VI）:
  preds = 方法在 demo_Kbad.json 上的真值发现结果
  gt    = demo.json（0 恶意）中的真实值
  数值场景 → RMSE(preds_last10%, gt_last10%), MAE(...)
  混合场景 → SemanticDist(X_clean, Y_noisy)

运行: python exp1_malicious_ratio.py
"""
import argparse
from pathlib import Path

from exp_utils import (
    N_AGENTS,
    NUM_METHODS, TEXT_METHODS,
    get_numeric_demo_files,
    get_text_demo_files,
    get_shared_bad_counts,
    load_batches,
    to_numeric_df, to_text_df,
    build_gt_numeric,
    eval_numeric, eval_text,
    plot_numeric, plot_text, save_csv,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp1_malicious_ratio"

TITLE   = "E1: 恶意节点比例"
X_LABEL = "恶意节点比例 (%)"


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1: 改变恶意节点比例")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json，文本=demo_*bad_collude.json。",
    )
    args = parser.parse_args()

    num_files = get_numeric_demo_files()
    text_files = get_text_demo_files()
    bad_counts = get_shared_bad_counts()
    out_dir = BASE_OUT_DIR / "demo_agent_text_collude"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print("datasets = numeric:num_demo_*bad_agent.json, text:demo_*bad_collude.json")

    # 干净数据：用于构建 ground truth 和文本基准 X
    clean_num_batches = load_batches(num_files[0])
    clean_txt_batches = load_batches(text_files[0])
    gt_num        = build_gt_numeric(clean_num_batches)   # 数值真值
    clean_txt_df  = to_text_df(clean_txt_batches)

    x_vals     = [k / N_AGENTS * 100 for k in bad_counts]

    num_results = {m: {"rmse": [], "mae": []}    for m in NUM_METHODS}
    txt_results = {m: {"data_accuracy": []}           for m in TEXT_METHODS}

    for bad_k, ratio in zip(bad_counts, x_vals):
        print(f"\n  bad_count={bad_k}  ({ratio:.0f}% 恶意)")
        num_batches  = load_batches(num_files[bad_k])
        txt_batches  = load_batches(text_files[bad_k])
        noisy_num_df = to_numeric_df(num_batches)
        noisy_txt_df = to_text_df(txt_batches)

        # ── 数值场景 ──────────────────────────────────────────────
        res_num = eval_numeric(noisy_num_df, gt_num)
        for m in NUM_METHODS:
            num_results[m]["rmse"].append(res_num[m]["rmse"])
            num_results[m]["mae"].append(res_num[m]["mae"])

        # ── 混合场景（文本） ───────────────────────────────────────
        res_txt = eval_text(clean_txt_df, noisy_txt_df)
        for m in TEXT_METHODS:
            txt_results[m]["data_accuracy"].append(res_txt[m]["data_accuracy"])

    plot_numeric(x_vals, num_results, X_LABEL, TITLE, out_dir)
    plot_text(x_vals, txt_results, X_LABEL, TITLE, out_dir)
    save_csv("malicious_ratio_pct", x_vals, num_results, txt_results, out_dir)

    print(f"\n[E1] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
