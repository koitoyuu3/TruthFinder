from __future__ import annotations

"""
50-agent E3-Text experiment with staged malicious-source activation.

Default schedule:
  batch 1-49    -> clean data
  batch 50-99   -> 10 bad agents (20%)
  batch 100-199 -> 20 bad agents (40%)
  batch 200-300 -> 30 bad agents (60%)

Usage:
    PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/truthfinder_mpl_exp3_text_50 \
    python3 decentralized_truth_discovery/examples/exp3_text_50agent.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.examples.exp_utils import (
    TEXT_METHODS,
    build_phase_switched_df,
    eval_text_metrics_per_batch,
    load_batches,
    plot_temporal_metric,
    to_text_df,
)

DEFAULT_DATA_DIR = ROOT / "decentralized_truth_discovery" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "decentralized_truth_discovery"
    / "examples"
    / "outputs"
    / "exp3_text"
    / "data_agent-50_news-300"
    / "10bad_at_50__20bad_at_100__30bad_at_200"
)
TITLE = "E3-Text: Progressive Byzantine Source Mutiny"
N_AGENTS_50 = 50


def save_results(
    accuracy_results: Dict[str, Dict[int, float]],
    hybrid_score_results: Dict[str, Dict[int, float]],
    out_dir: Path,
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
) -> Path:
    rows = []
    for method in TEXT_METHODS:
        acc_series = accuracy_results.get(method, {})
        score_series = hybrid_score_results.get(method, {})
        all_batches = sorted(set(acc_series) | set(score_series))
        for batch_idx in all_batches:
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "method": method,
                    "data_accuracy": acc_series.get(batch_idx, float("nan")),
                    "mean_hybrid_score": score_series.get(batch_idx, float("nan")),
                    "dataset": "data_agent-50_news-300",
                    "attack_profile": "collude",
                    "phase1_start_batch": phase1_start,
                    "phase2_start_batch": phase2_start,
                    "phase3_start_batch": phase3_start,
                    "phase1_bad_count": phase1_bad_count,
                    "phase2_bad_count": phase2_bad_count,
                    "phase3_bad_count": phase3_bad_count,
                }
            )
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _validate_inputs(
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
    n_batches: int,
) -> None:
    if phase1_start < 1:
        raise ValueError(f"phase1_start must be >= 1, got {phase1_start}")
    if phase2_start <= phase1_start:
        raise ValueError(
            f"phase2_start must be > phase1_start, got phase1_start={phase1_start}, phase2_start={phase2_start}"
        )
    if phase3_start <= phase2_start:
        raise ValueError(
            f"phase3_start must be > phase2_start, got phase2_start={phase2_start}, phase3_start={phase3_start}"
        )
    if phase3_start > n_batches:
        raise ValueError(f"phase3_start={phase3_start} exceeds available batches ({n_batches})")
    if phase1_bad_count < 0 or phase2_bad_count < 0:
        raise ValueError("bad_count must be non-negative")
    if phase3_bad_count < 0:
        raise ValueError("phase3_bad_count must be non-negative")
    if phase2_bad_count < phase1_bad_count:
        raise ValueError("phase2_bad_count must be >= phase1_bad_count")
    if phase3_bad_count < phase2_bad_count:
        raise ValueError("phase3_bad_count must be >= phase2_bad_count")
    if phase3_bad_count >= N_AGENTS_50:
        raise ValueError(f"phase3_bad_count must be < {N_AGENTS_50}")


def build_file_map(
    data_dir: Path,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
) -> Dict[int, Path]:
    files = {
        0: data_dir / "demo.json",
        phase1_bad_count: data_dir / f"demo_{phase1_bad_count}bad_collude.json",
        phase2_bad_count: data_dir / f"demo_{phase2_bad_count}bad_collude.json",
        phase3_bad_count: data_dir / f"demo_{phase3_bad_count}bad_collude.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing text experiment files:\n" + "\n".join(missing))
    return files


def run_experiment(
    data_dir: Path,
    out_dir: Path,
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
) -> None:
    files = build_file_map(data_dir, phase1_bad_count, phase2_bad_count, phase3_bad_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_txt_batches = load_batches(files[0])
    phase1_txt_batches = load_batches(files[phase1_bad_count])
    phase2_txt_batches = load_batches(files[phase2_bad_count])
    phase3_txt_batches = load_batches(files[phase3_bad_count])

    n_batches = min(
        len(clean_txt_batches),
        len(phase1_txt_batches),
        len(phase2_txt_batches),
        len(phase3_txt_batches),
    )
    _validate_inputs(
        phase1_start,
        phase2_start,
        phase3_start,
        phase1_bad_count,
        phase2_bad_count,
        phase3_bad_count,
        n_batches,
    )

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "dataset = text:data_agent-50_news-300/demo_*bad_collude.json, "
        f"schedule = 0 bad -> {phase1_bad_count} bad -> {phase2_bad_count} bad -> {phase3_bad_count} bad"
    )

    clean_txt_df = to_text_df(clean_txt_batches)
    phased_txt_df = build_phase_switched_df(
        [
            (1, clean_txt_batches),
            (phase1_start, phase1_txt_batches),
            (phase2_start, phase2_txt_batches),
            (phase3_start, phase3_txt_batches),
        ],
        n_total=n_batches,
        mode="text",
    )
    accuracy_results, hybrid_score_results = eval_text_metrics_per_batch(clean_txt_df, phased_txt_df)

    event_lines = [
        (phase1_start, f"{phase1_bad_count} bad ({phase1_bad_count / N_AGENTS_50:.0%})"),
        (phase2_start, f"{phase2_bad_count} bad ({phase2_bad_count / N_AGENTS_50:.0%})"),
        (phase3_start, f"{phase3_bad_count} bad ({phase3_bad_count / N_AGENTS_50:.0%})"),
    ]
    mixed_path = out_dir / "mixed_text.png"
    accuracy_path = out_dir / "mixed_text_accuracy.png"
    plot_temporal_metric(
        hybrid_score_results,
        TEXT_METHODS,
        ylabel="Mean Hybrid Score",
        title=TITLE,
        out_path=mixed_path,
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    plot_temporal_metric(
        accuracy_results,
        TEXT_METHODS,
        ylabel="Data Accuracy",
        title=f"{TITLE} — Thresholded Accuracy",
        out_path=accuracy_path,
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    results_path = save_results(
        accuracy_results,
        hybrid_score_results,
        out_dir,
        phase1_start=phase1_start,
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        phase1_bad_count=phase1_bad_count,
        phase2_bad_count=phase2_bad_count,
        phase3_bad_count=phase3_bad_count,
    )

    print(f"saved {mixed_path}")
    print(f"saved {accuracy_path}")
    print(f"saved {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent text exp3 with staged malicious activation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase1-start", type=int, default=50)
    parser.add_argument("--phase2-start", type=int, default=100)
    parser.add_argument("--phase3-start", type=int, default=200)
    parser.add_argument("--phase1-bad-count", type=int, default=10)
    parser.add_argument("--phase2-bad-count", type=int, default=20)
    parser.add_argument("--phase3-bad-count", type=int, default=30)
    args = parser.parse_args()
    run_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        phase1_start=args.phase1_start,
        phase2_start=args.phase2_start,
        phase3_start=args.phase3_start,
        phase1_bad_count=args.phase1_bad_count,
        phase2_bad_count=args.phase2_bad_count,
        phase3_bad_count=args.phase3_bad_count,
    )


if __name__ == "__main__":
    main()
