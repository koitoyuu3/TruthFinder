"""
实验3: 随着恶意节点注入时间变化
================================================
自变量: 注入起始 batch t ∈ {1, 2, ..., N}
场景设定:
  batch_index < t  → 使用干净数据（demo.json）
  batch_index >= t → 使用恶意数据（demo_4bad.json，40% 恶意节点）

直觉:
  t=1  → 全程恶意（最差）
  t=N  → 最后一刻才注入（方法已积累足够权重，抵御能力最强）
  → 随 t 增大，各方法全程 RMSE 应逐渐降低

评估框架（对齐论文 Section VI）:
  preds = 方法在注入合成数据集上的输出
  gt    = demo.json（全程干净）中的真实值
  数值场景 → RMSE(preds_all, gt_all), MAE(...)
  混合场景 → DataAccuracy(X_clean, Y_noisy)

运行: python exp3_injection_time.py
"""
import argparse
from pathlib import Path

from exp_utils import (
    NUM_METHODS, TEXT_METHODS,
    get_numeric_demo_files,
    get_text_demo_files,
    load_batches,
    to_numeric_df, to_text_df,
    build_gt_numeric,
    eval_numeric, eval_text,
    plot_numeric, plot_text, save_csv,
    build_injection_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_injection_time"

BAD_COUNT = 4        # 注入后的恶意比例为 40%
TITLE     = "E3: 恶意节点注入时间"
X_LABEL   = "恶意注入起始 Batch"


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3: 随着恶意节点注入时间变化")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json，文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="注入后使用的恶意节点数量。",
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

    n_batches        = min(
        len(clean_num_batches),
        len(mal_num_batches),
        len(clean_txt_batches),
        len(mal_txt_batches),
    )
    injection_points = list(range(1, n_batches + 1))

    # ground truth（干净数据，全程固定）
    gt_num       = build_gt_numeric(clean_num_batches)
    clean_txt_df = to_text_df(clean_txt_batches)

    num_results = {m: {"rmse": [], "mae": []} for m in NUM_METHODS}
    txt_results = {m: {"data_accuracy": []}        for m in TEXT_METHODS}

    for t in injection_points:
        print(f"\n  注入起始 batch = {t}  "
              f"（前 {t-1} 个干净，后 {n_batches-t+1} 个恶意）")

        # ── 数值场景 ──────────────────────────────────────────────
        noisy_num_df = build_injection_df(clean_num_batches, mal_num_batches,
                                          inj_start=t, n_total=n_batches,
                                          mode="numeric")
        res_num = eval_numeric(noisy_num_df, gt_num, last_pct=1.0)
        for m in NUM_METHODS:
            num_results[m]["rmse"].append(res_num[m]["rmse"])
            num_results[m]["mae"].append(res_num[m]["mae"])

        # ── 混合场景（文本） ───────────────────────────────────────
        noisy_txt_df = build_injection_df(clean_txt_batches, mal_txt_batches,
                                          inj_start=t, n_total=n_batches,
                                          mode="text")
        if not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(res_txt[m]["data_accuracy"])
        else:
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(float("nan"))

    plot_numeric(
        injection_points,
        num_results,
        X_LABEL,
        TITLE,
        out_dir,
        metric_scope_label="all batches",
    )
    plot_text(injection_points, txt_results, X_LABEL, TITLE, out_dir)
    save_csv("injection_start_batch", injection_points,
             num_results, txt_results, out_dir)

    print(f"\n[E3] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
