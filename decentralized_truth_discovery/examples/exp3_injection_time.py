"""
实验3: 随着恶意节点注入时间变化
================================================
自变量: 注入时间 ∈ {早, 中, 晚}
场景设定:
  batch_index < t  → 使用干净数据
  batch_index >= t → 使用 4 bad 恶意数据（40% 恶意节点）

直觉:
  早  → 全程更早暴露于恶意数据（最差）
  晚  → 最后一刻才注入（方法已积累足够权重，抵御能力最强）
  → 从早到晚，各方法全程 Relative RMSE 通常应逐渐降低

评估框架（对齐论文 Section VI）:
  preds = 方法在注入合成数据集上的输出
  gt    = demo.json（全程干净）中的真实值
  数值场景 → per-batch Relative RMSE(preds_b, gt_b)
  混合场景 → per-batch DataAccuracy(X_b, Y_b)

运行: python exp3_injection_time.py
"""
import argparse
from pathlib import Path
from typing import List, Tuple

from exp_utils import (
    NUM_METHODS, TEXT_METHODS,
    get_numeric_demo_files,
    get_text_demo_files,
    load_batches,
    to_numeric_df, to_text_df,
    build_gt_numeric,
    eval_numeric_per_batch, eval_text_per_batch,
    plot_stage_metric_panels,
    build_injection_df,
    get_injection_stages,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_injection_time"

BAD_COUNT = 4        # 注入后的恶意比例为 40%
TITLE     = "E3: 恶意节点注入时间"
X_LABEL   = "恶意注入时间"


def save_results(
    stages: List[Tuple[str, int]],
    num_results_by_stage,
    txt_results_by_stage,
    out_dir: Path,
) -> None:
    rows = []
    for stage_label, injection_start in stages:
        for method in NUM_METHODS:
            series = num_results_by_stage.get(stage_label, {}).get(method, {})
            for batch_idx, relative_rmse in sorted(series.items()):
                rows.append({
                    "injection_stage": stage_label,
                    "injection_start_batch": injection_start,
                    "batch_idx": batch_idx,
                    "method": method,
                    "scene": "numeric",
                    "relative_rmse": relative_rmse,
                    "data_accuracy": float("nan"),
                })
        for method in TEXT_METHODS:
            series = txt_results_by_stage.get(stage_label, {}).get(method, {})
            for batch_idx, data_accuracy in sorted(series.items()):
                rows.append({
                    "injection_stage": stage_label,
                    "injection_start_batch": injection_start,
                    "batch_idx": batch_idx,
                    "method": method,
                    "scene": "mixed_text",
                    "relative_rmse": float("nan"),
                    "data_accuracy": data_accuracy,
                })
    path = out_dir / "results.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3: 随着恶意节点注入时间变化")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_collude.json，文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="注入后使用的恶意节点数量。",
    )
    args = parser.parse_args()

    num_files = get_numeric_demo_files(mode="collude")
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
    print(f"datasets = numeric:num_demo_*bad_collude.json, text:demo_*bad_collude.json, bad_count = {args.bad_count}")

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
    stages = get_injection_stages(n_batches)
    stage_labels = [label for label, _ in stages]

    # ground truth（干净数据，全程固定）
    gt_num       = build_gt_numeric(clean_num_batches)
    clean_txt_df = to_text_df(clean_txt_batches)

    num_results_by_stage = {}
    txt_results_by_stage = {}

    for stage_label, injection_start in stages:
        print(
            f"\n  注入时间 = {stage_label}  "
            f"(start batch = {injection_start}; 前 {injection_start - 1} 个干净，后 {n_batches - injection_start + 1} 个恶意)"
        )

        # ── 数值场景 ──────────────────────────────────────────────
        noisy_num_df = build_injection_df(clean_num_batches, mal_num_batches,
                                          inj_start=injection_start, n_total=n_batches,
                                          mode="numeric")
        num_results_by_stage[stage_label] = eval_numeric_per_batch(noisy_num_df, gt_num, normalize=True)

        # ── 混合场景（文本） ───────────────────────────────────────
        noisy_txt_df = build_injection_df(clean_txt_batches, mal_txt_batches,
                                          inj_start=injection_start, n_total=n_batches,
                                          mode="text")
        if not noisy_txt_df.empty:
            txt_results_by_stage[stage_label] = eval_text_per_batch(clean_txt_df, noisy_txt_df)
        else:
            txt_results_by_stage[stage_label] = {
                method: {} for method in TEXT_METHODS
            }

    plot_stage_metric_panels(
        stages,
        num_results_by_stage,
        NUM_METHODS,
        ylabel="Relative RMSE",
        title_prefix=TITLE,
        out_path=out_dir / "numeric.png",
    )
    plot_stage_metric_panels(
        stages,
        txt_results_by_stage,
        TEXT_METHODS,
        ylabel="Data Accuracy",
        title_prefix=f"{TITLE} — 文本",
        out_path=out_dir / "mixed_text.png",
        y_lim=(0.0, 1.05),
    )
    save_results(stages, num_results_by_stage, txt_results_by_stage, out_dir)

    print(f"\n[E3] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
