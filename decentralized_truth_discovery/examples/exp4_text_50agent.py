from __future__ import annotations

"""
50-agent E4-Text experiment with fixed 5 malicious agents.

Agent counts: 15, 20, 25, 30, 35, 40, 45, 50
Malicious agents are fixed to agent-1..agent-5 within each subset.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.examples.exp_utils import (
    TEXT_METHODS,
    eval_text,
    load_batches,
    plot_text,
    to_text_df,
)

DEFAULT_DATA_DIR = ROOT / "decentralized_truth_discovery" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "decentralized_truth_discovery"
    / "examples"
    / "outputs"
    / "exp4_text"
    / "data_agent-50_news-300"
    / "bad_5"
)
BAD_COUNT = 5
AGENT_RANGE = [15, 20, 25, 30, 35, 40, 45, 50]
TITLE = "E4-Text: Agent 数量变化"
X_LABEL = "Agent 总数"
N_AGENTS_50 = 50


def save_results(
    agent_range: List[int],
    txt_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> Path:
    rows = []
    for i, agent_count in enumerate(agent_range):
        for method in TEXT_METHODS:
            rows.append(
                {
                    "agent_count": agent_count,
                    "method": method,
                    "data_accuracy": txt_results[method]["data_accuracy"][i]
                    if i < len(txt_results[method]["data_accuracy"])
                    else float("nan"),
                    "dataset": "data_agent-50_news-300",
                    "bad_count": BAD_COUNT,
                }
            )
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_experiment(data_dir: Path, out_dir: Path) -> None:
    clean_path = data_dir / "demo.json"
    noisy_path = data_dir / f"demo_{BAD_COUNT}bad_collude.json"
    for path in [clean_path, noisy_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir.mkdir(parents=True, exist_ok=True)

    clean_txt_batches = load_batches(clean_path)
    mal_txt_batches = load_batches(noisy_path)

    all_agents = [f"agent-{i}" for i in range(1, N_AGENTS_50 + 1)]
    txt_results = {method: {"data_accuracy": []} for method in TEXT_METHODS}

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "dataset = text:data_agent-50_news-300/demo_*bad_collude.json, "
        f"bad_count = {BAD_COUNT}"
    )

    for n_agents in AGENT_RANGE:
        subset = all_agents[:n_agents]
        ratio = BAD_COUNT / n_agents * 100
        print(f"\n  n_agents={n_agents}  恶意比例≈{ratio:.1f}%")

        clean_txt_df = to_text_df(clean_txt_batches, agent_subset=subset)
        noisy_txt_df = to_text_df(mal_txt_batches, agent_subset=subset)

        if not clean_txt_df.empty and not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(res_txt[method]["data_accuracy"])
        else:
            for method in TEXT_METHODS:
                txt_results[method]["data_accuracy"].append(float("nan"))

    plot_text(AGENT_RANGE, txt_results, X_LABEL, TITLE, out_dir)
    results_path = save_results(AGENT_RANGE, txt_results, out_dir)
    print(f"saved {out_dir / 'mixed_text.png'}")
    print(f"saved {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent text exp4 with fixed 5 malicious agents.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
