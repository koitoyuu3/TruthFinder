"""
实验3-文本: 分阶段恶意源激活
================================================
场景设定:
  batch 1-49   → 使用干净数据（0 bad）
  batch 50-99  → 使用 2 bad 数据（20% 恶意源）
  batch 100-150 → 使用 4 bad 数据（40% 恶意源）

评估框架:
  X = 方法在干净数据上的输出（ground truth 代理）
  Y = 方法在分阶段攻击数据集上的输出（预测结果）
  主图指标 = per-batch Mean Hybrid Score(X_b, Y_b)
  附图指标 = per-batch DataAccuracy(X_b, Y_b)

运行: python exp3_text.py
"""
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from exp_utils import (
    N_AGENTS,
    TEXT_METHODS,
    build_phase_switched_df,
    eval_text_metrics_per_batch,
    get_text_demo_files,
    load_batches,
    plot_temporal_metric,
    to_text_df,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_text"

PHASE1_START = 50
PHASE2_START = 100
PHASE1_BAD_COUNT = 2
PHASE2_BAD_COUNT = 4
TITLE = "E3-Text: Progressive Byzantine Source Mutiny"


def save_results(
    accuracy_results: Dict[str, Dict[int, float]],
    hybrid_score_results: Dict[str, Dict[int, float]],
    out_dir: Path,
    attack_profile: str,
    phase1_start: int,
    phase2_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
) -> None:
    rows = []
    for method in TEXT_METHODS:
        acc_series = accuracy_results.get(method, {})
        score_series = hybrid_score_results.get(method, {})
        all_batches = sorted(set(acc_series) | set(score_series))
        for batch_idx in all_batches:
            rows.append({
                "batch_idx": batch_idx,
                "method": method,
                "data_accuracy": acc_series.get(batch_idx, float("nan")),
                "mean_hybrid_score": score_series.get(batch_idx, float("nan")),
                "attack_profile": attack_profile,
                "phase1_start_batch": phase1_start,
                "phase2_start_batch": phase2_start,
                "phase1_bad_count": phase1_bad_count,
                "phase2_bad_count": phase2_bad_count,
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def _validate_phase_schedule(phase1_start: int, phase2_start: int, n_batches: int) -> None:
    if phase1_start < 1:
        raise ValueError(f"phase1_start must be >= 1, got {phase1_start}")
    if phase2_start <= phase1_start:
        raise ValueError(
            f"phase2_start must be > phase1_start, got phase1_start={phase1_start}, phase2_start={phase2_start}"
        )
    if phase2_start > n_batches:
        raise ValueError(
            f"phase2_start={phase2_start} exceeds available batches ({n_batches})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3-文本: 分阶段恶意源激活")
    parser.add_argument(
        "--attack-profile",
        default="agent",
        choices=["agent", "collude"],
        help="恶意数据模式。agent=动态偏离；collude=静态串谋。",
    )
    parser.add_argument(
        "--phase1-start",
        type=int,
        default=PHASE1_START,
        help="第一阶段恶意源激活 batch。",
    )
    parser.add_argument(
        "--phase2-start",
        type=int,
        default=PHASE2_START,
        help="第二阶段恶意源激活 batch。",
    )
    parser.add_argument(
        "--phase1-bad-count",
        type=int,
        default=PHASE1_BAD_COUNT,
        help="第一阶段恶意源数量。",
    )
    parser.add_argument(
        "--phase2-bad-count",
        type=int,
        default=PHASE2_BAD_COUNT,
        help="第二阶段恶意源数量。",
    )
    args = parser.parse_args()

    text_files = get_text_demo_files(mode=args.attack_profile)
    for bad_count in [0, args.phase1_bad_count, args.phase2_bad_count]:
        if bad_count not in text_files:
            available = ", ".join(str(k) for k in sorted(text_files))
            raise ValueError(
                f"bad_count={bad_count} is unavailable for text files under attack_profile={args.attack_profile}. "
                f"Available bad_count values: {available}"
            )

    out_dir = (
        BASE_OUT_DIR
        / args.attack_profile
        / f"{args.phase1_bad_count}bad_at_{args.phase1_start}__{args.phase2_bad_count}bad_at_{args.phase2_start}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "dataset = "
        f"text:demo_*bad_{args.attack_profile}.json, "
        f"schedule = 0 bad -> {args.phase1_bad_count} bad -> {args.phase2_bad_count} bad"
    )

    clean_txt_batches = load_batches(text_files[0])
    phase1_txt_batches = load_batches(text_files[args.phase1_bad_count])
    phase2_txt_batches = load_batches(text_files[args.phase2_bad_count])

    n_batches = min(len(clean_txt_batches), len(phase1_txt_batches), len(phase2_txt_batches))
    _validate_phase_schedule(args.phase1_start, args.phase2_start, n_batches)

    clean_txt_df = to_text_df(clean_txt_batches)
    phased_txt_df = build_phase_switched_df(
        [
            (1, clean_txt_batches),
            (args.phase1_start, phase1_txt_batches),
            (args.phase2_start, phase2_txt_batches),
        ],
        n_total=n_batches,
        mode="text",
    )
    accuracy_results, hybrid_score_results = eval_text_metrics_per_batch(clean_txt_df, phased_txt_df)

    event_lines = [
        (
            args.phase1_start,
            f"{args.phase1_bad_count} bad ({args.phase1_bad_count / N_AGENTS:.0%})",
        ),
        (
            args.phase2_start,
            f"{args.phase2_bad_count} bad ({args.phase2_bad_count / N_AGENTS:.0%})",
        ),
    ]
    plot_temporal_metric(
        hybrid_score_results,
        TEXT_METHODS,
        ylabel="Mean Hybrid Score",
        title=TITLE,
        out_path=out_dir / "mixed_text.png",
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    plot_temporal_metric(
        accuracy_results,
        TEXT_METHODS,
        ylabel="Data Accuracy",
        title=f"{TITLE} — Thresholded Accuracy",
        out_path=out_dir / "mixed_text_accuracy.png",
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    save_results(
        accuracy_results,
        hybrid_score_results,
        out_dir,
        attack_profile=args.attack_profile,
        phase1_start=args.phase1_start,
        phase2_start=args.phase2_start,
        phase1_bad_count=args.phase1_bad_count,
        phase2_bad_count=args.phase2_bad_count,
    )

    print(f"\n[E3-Text] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
