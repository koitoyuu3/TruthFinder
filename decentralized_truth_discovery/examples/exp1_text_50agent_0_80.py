"""
50-agent E1-Text experiment over 0-80% malicious ratios.

Usage:
    python3 decentralized_truth_discovery/examples/exp1_text_50agent_0_80.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.core import SemanticTruthFinder
from decentralized_truth_discovery.examples import exp_utils
from decentralized_truth_discovery.examples.exp_utils import (
    load_batches,
    metric_data_accuracy,
    plot_text,
    to_text_df,
)

DEFAULT_DATA_DIR = ROOT / "decentralized_truth_discovery" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "decentralized_truth_discovery"
    / "examples"
    / "outputs"
    / "exp1_text"
    / "data_agent-50_news-300_0_80"
)
BAD_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40]


def build_file_map(data_dir: Path, bad_counts: Iterable[int]) -> Dict[int, Path]:
    files = {0: data_dir / "demo.json"}
    for bad_k in bad_counts:
        files[bad_k] = data_dir / f"demo_{bad_k}bad_collude.json"
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment files:\n" + "\n".join(missing))
    return files


def run_semantic_single(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    if df.empty:
        return {}
    model = SemanticTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
    preds: Dict[Tuple[int, str], str] = {}
    empty_numeric_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        _, out_txt = model.process_batch(empty_numeric_df.copy(), grp)
        for _, row in out_txt[["batch_index", "object", "global_truth"]].drop_duplicates().iterrows():
            preds[(int(row["batch_index"]), str(row["object"]))] = str(row["global_truth"])
    return preds


def run_experiment(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    files = build_file_map(data_dir, BAD_COUNTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_df = to_text_df(load_batches(files[0]))
    noisy_dfs = {bad_k: to_text_df(load_batches(path)) for bad_k, path in files.items() if bad_k != 0}

    print("clean: SenteTruth")
    clean_sente = exp_utils._run_sente_txt(clean_df)
    print("clean: BasicTruth")
    clean_basic = exp_utils._run_basic_txt(clean_df)
    print("clean: SemanticTruth")
    clean_semantic = run_semantic_single(clean_df)

    results = {m: {"data_accuracy": []} for m in ["SenteTruth", "BasicTruth", "SemanticTruth"]}
    x_vals = [0.0]
    results["SenteTruth"]["data_accuracy"].append(metric_data_accuracy(clean_sente, clean_sente))
    results["BasicTruth"]["data_accuracy"].append(metric_data_accuracy(clean_basic, clean_basic))
    results["SemanticTruth"]["data_accuracy"].append(metric_data_accuracy(clean_semantic, clean_semantic))

    for bad_k in BAD_COUNTS:
        ratio = bad_k / 50 * 100
        noisy_df = noisy_dfs[bad_k]
        print(f"{bad_k} bad ({ratio:.0f}%): SenteTruth")
        noisy_sente = exp_utils._run_sente_txt(noisy_df)
        print(f"{bad_k} bad ({ratio:.0f}%): BasicTruth")
        noisy_basic = exp_utils._run_basic_txt(noisy_df)
        print(f"{bad_k} bad ({ratio:.0f}%): SemanticTruth")
        noisy_semantic = run_semantic_single(noisy_df)

        x_vals.append(ratio)
        results["SenteTruth"]["data_accuracy"].append(metric_data_accuracy(clean_sente, noisy_sente))
        results["BasicTruth"]["data_accuracy"].append(metric_data_accuracy(clean_basic, noisy_basic))
        results["SemanticTruth"]["data_accuracy"].append(metric_data_accuracy(clean_semantic, noisy_semantic))

    plot_text(x_vals, results, "恶意节点比例 (%)", "E1-Text: 恶意节点比例", out_dir)

    rows = []
    for idx, ratio in enumerate(x_vals):
        rows.append(
            {
                "malicious_ratio_pct": ratio,
                "SenteTruth": results["SenteTruth"]["data_accuracy"][idx],
                "BasicTruth": results["BasicTruth"]["data_accuracy"][idx],
                "SemanticTruth": results["SemanticTruth"]["data_accuracy"][idx],
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "text_data_accuracy.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"saved {csv_path}")
    print(f"saved {out_dir / 'mixed_text.png'}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent text exp1 over 0-80% malicious ratios.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
