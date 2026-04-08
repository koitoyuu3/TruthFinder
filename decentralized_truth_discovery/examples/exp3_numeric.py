"""
实验3-数值: 分阶段恶意源激活
================================================
场景设定:
  batch 1-49   → 使用干净数据（0 bad）
  batch 50-99  → 使用 2 bad 数据（20% 恶意源）
  batch 100-150 → 使用 4 bad 数据（40% 恶意源）

评估框架（对齐 DecentTruth / SenFEED 的 temporal attack setting）:
  preds = 方法在分阶段攻击数据集上的输出
  gt    = num_demo.json（全程干净）中的真实值
  指标  = per-batch Relative RMSE(preds_b, gt_b)

默认攻击模式: collude

运行: python exp3_numeric.py
"""
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from exp_utils import (
    N_AGENTS,
    NUM_METHODS,
    build_gt_numeric,
    build_phase_switched_df,
    eval_numeric_per_batch,
    get_numeric_demo_files,
    load_batches,
    plot_temporal_metric,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp3_numeric"

PHASE1_START = 50
PHASE2_START = 100
PHASE1_BAD_COUNT = 2
PHASE2_BAD_COUNT = 4
TITLE = "E3-Numeric: Progressive Byzantine Source Mutiny"


def save_results(
    per_batch_results: Dict[str, Dict[int, float]],
    out_dir: Path,
    attack_profile: str,
    phase1_start: int,
    phase2_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
) -> None:
    rows = []
    for method in NUM_METHODS:
        series = per_batch_results.get(method, {})
        for batch_idx, relative_rmse in sorted(series.items()):
            rows.append({
                "batch_idx": batch_idx,
                "method": method,
                "relative_rmse": relative_rmse,
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
    parser = argparse.ArgumentParser(description="实验3-数值: 分阶段恶意源激活")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        choices=["agent", "collude"],
        help="恶意数据模式。collude=更有辨识度的同向串谋（默认）；agent=动态偏离。",
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

    num_files = get_numeric_demo_files(mode=args.attack_profile)
    for bad_count in [0, args.phase1_bad_count, args.phase2_bad_count]:
        if bad_count not in num_files:
            available = ", ".join(str(k) for k in sorted(num_files))
            raise ValueError(
                f"bad_count={bad_count} is unavailable for numeric files under attack_profile={args.attack_profile}. "
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
        f"numeric:num_demo_*bad_{args.attack_profile}.json, "
        f"schedule = 0 bad -> {args.phase1_bad_count} bad -> {args.phase2_bad_count} bad"
    )

    clean_num_batches = load_batches(num_files[0])
    phase1_num_batches = load_batches(num_files[args.phase1_bad_count])
    phase2_num_batches = load_batches(num_files[args.phase2_bad_count])

    n_batches = min(len(clean_num_batches), len(phase1_num_batches), len(phase2_num_batches))
    _validate_phase_schedule(args.phase1_start, args.phase2_start, n_batches)

    gt_num = build_gt_numeric(clean_num_batches)
    phased_num_df = build_phase_switched_df(
        [
            (1, clean_num_batches),
            (args.phase1_start, phase1_num_batches),
            (args.phase2_start, phase2_num_batches),
        ],
        n_total=n_batches,
        mode="numeric",
    )
    per_batch_results = eval_numeric_per_batch(phased_num_df, gt_num, normalize=True)

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
        per_batch_results,
        NUM_METHODS,
        ylabel="Relative RMSE",
        title=TITLE,
        out_path=out_dir / "numeric.png",
        event_lines=event_lines,
    )
    save_results(
        per_batch_results,
        out_dir,
        attack_profile=args.attack_profile,
        phase1_start=args.phase1_start,
        phase2_start=args.phase2_start,
        phase1_bad_count=args.phase1_bad_count,
        phase2_bad_count=args.phase2_bad_count,
    )

    print(f"\n[E3-Numeric] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
