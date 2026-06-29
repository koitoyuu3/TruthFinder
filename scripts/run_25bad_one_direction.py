#!/usr/bin/env python3
"""
For each of the 4 sub-experiments (1.2/1.4 × {agent-50, llama}), recompute the
50% (25-bad / numonly) DataAcc using the freshly regenerated single-direction
data file, and overwrite ONLY the 50% row in the existing
``text_data_accuracy.csv`` (keeping the 0/10/20/30/40 rows untouched).

Run AFTER ``scripts/regen_25bad_one_direction.py``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path("/Users/huabowen/TruthFinder")
PKG = ROOT / "decentralized_truth_discovery"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.core import SemanticTruthFinder
from decentralized_truth_discovery.examples import exp_utils
from decentralized_truth_discovery.examples.exp_utils import (
    load_batches,
    metric_data_accuracy,
    to_text_df,
)


THRESHOLD = 0.90
BAD_K = 25
BAD_RATIO_PCT = 50.0  # 25 / 50 * 100


def run_semantic_single(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    if df.empty:
        return {}
    model = SemanticTruthFinder(
        enable_zk_proof=False, use_sbert=True, local_files_only=True
    )
    preds: Dict[Tuple[int, str], str] = {}
    empty_numeric_df = pd.DataFrame(
        columns=["website", "object", "fact", "batch_index"]
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        _, out_txt = model.process_batch(empty_numeric_df.copy(), grp)
        for _, row in (
            out_txt[["batch_index", "object", "global_truth"]]
            .drop_duplicates()
            .iterrows()
        ):
            preds[(int(row["batch_index"]), str(row["object"]))] = str(
                row["global_truth"]
            )
    return preds


JOBS = [
    # (label, data_dir, out_dir)
    (
        "1.2-agent-50-news",
        PKG / "data_agent-50_news-300",
        PKG / "examples/outputs/exp1_text/1.2-agent-50-news-300-50%-0.9-numonly",
    ),
    (
        "1.2-llama-news",
        PKG / "data-llama-agent-50_news-300",
        PKG / "examples/outputs/exp1_text/1.2-llama-news-300-50%-0.9-numonly",
    ),
    (
        "1.4-agent-50-weather",
        PKG / "data_agent-50_weather-300",
        PKG / "examples/outputs/exp1_text/1.4-agent-50-weather-300-50%-0.9-numonly",
    ),
    (
        "1.4-llama-weather",
        PKG / "data-llama-agent-50_weather-300",
        PKG / "examples/outputs/exp1_text/1.4-llama-weather-300-50%-0.9-numonly",
    ),
]


def update_50_row(out_dir: Path, sente: float, basic: float, semantic: float) -> None:
    csv_path = out_dir / "text_data_accuracy.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    updated = False
    for row in rows:
        if abs(float(row["malicious_ratio_pct"]) - BAD_RATIO_PCT) < 1e-6:
            row["SenteTruth"] = repr(sente) if False else str(sente)
            row["BasicTruth"] = str(basic)
            row["SemanticTruth"] = str(semantic)
            updated = True
            break
    if not updated:
        rows.append(
            {
                "malicious_ratio_pct": BAD_RATIO_PCT,
                "SenteTruth": str(sente),
                "BasicTruth": str(basic),
                "SemanticTruth": str(semantic),
            }
        )
    fieldnames = ["malicious_ratio_pct", "SenteTruth", "BasicTruth", "SemanticTruth"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"updated 50% row in {csv_path}")


def run_one(label: str, data_dir: Path, out_dir: Path) -> None:
    print(f"\n=== {label} ===")
    clean_path = data_dir / "demo_div.json"
    noisy_path = data_dir / "demo_25bad_numonly.json"
    for p in (clean_path, noisy_path):
        if not p.exists():
            raise FileNotFoundError(p)

    clean_df = to_text_df(load_batches(clean_path))
    noisy_df = to_text_df(load_batches(noisy_path))

    print("clean: SenteTruth")
    clean_sente = exp_utils._run_sente_txt(clean_df)
    print("clean: BasicTruth")
    clean_basic = exp_utils._run_basic_txt(clean_df)
    print("clean: SemanticTruth")
    clean_semantic = run_semantic_single(clean_df)

    print(f"{BAD_K}bad ({BAD_RATIO_PCT:.0f}%): SenteTruth")
    noisy_sente = exp_utils._run_sente_txt(noisy_df)
    print(f"{BAD_K}bad ({BAD_RATIO_PCT:.0f}%): BasicTruth")
    noisy_basic = exp_utils._run_basic_txt(noisy_df)
    print(f"{BAD_K}bad ({BAD_RATIO_PCT:.0f}%): SemanticTruth")
    noisy_semantic = run_semantic_single(noisy_df)

    sente = metric_data_accuracy(clean_sente, noisy_sente, threshold=THRESHOLD)
    basic = metric_data_accuracy(clean_basic, noisy_basic, threshold=THRESHOLD)
    semantic = metric_data_accuracy(clean_semantic, noisy_semantic, threshold=THRESHOLD)
    print(
        f"  result: SenteTruth={sente:.6f}  BasicTruth={basic:.6f}  "
        f"SemanticTruth={semantic:.6f}"
    )

    update_50_row(out_dir, sente, basic, semantic)


def main() -> None:
    for label, data_dir, out_dir in JOBS:
        run_one(label, data_dir, out_dir)


if __name__ == "__main__":
    main()
