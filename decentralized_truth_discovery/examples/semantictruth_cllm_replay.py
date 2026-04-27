#!/usr/bin/env python3
"""
Use CLLM-style sequential oracle replay to test SemanticTruth on the CLLM dataset.

This script follows the CLLM prototype idea:
  - one target model at a time
  - 10 oracle nodes per question
  - process questions sequentially
  - preserve cross-question source reliability

Optional attack injection:
  - none
  - random
  - incorrect
  - substitution
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

from exp_utils import metric_data_accuracy
from decentralized_truth_discovery.core import SemanticTruthFinder


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ROOT = ROOT / "decentralized_truth_discovery" / "data_CLLM"
EXTERNAL_DATA_ROOT = Path("/Users/huabowen/CLLM/datas")
MODEL_FILES = ("gpt", "google", "chatglm", "llama", "tencent")


def default_data_root() -> Path:
    if EXTERNAL_DATA_ROOT.exists():
        return EXTERNAL_DATA_ROOT
    return LOCAL_DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SemanticTruth in CLLM-style sequential mode")
    parser.add_argument("--data-root", default=str(default_data_root()), help="CLLM dataset root")
    parser.add_argument("--dataset", default="BASE", help="BASE / MIX / PRO / PRO-single / BASE-fix")
    parser.add_argument("--model", default="gpt", choices=MODEL_FILES, help="target model")
    parser.add_argument(
        "--attack",
        default="none",
        choices=("none", "random", "incorrect", "substitution"),
        help="attack type",
    )
    parser.add_argument(
        "--substitute-model",
        default="google",
        choices=MODEL_FILES,
        help="used when attack=substitution",
    )
    parser.add_argument("--bad-count", type=int, default=4, help="number of malicious nodes")
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all questions")
    parser.add_argument(
        "--compare-clean",
        action="store_true",
        help="when attack!=none, also replay clean data and report data_accuracy",
    )
    parser.add_argument("--output-csv", default="", help="optional csv output path")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def resolve_clean_file(data_root: Path, dataset: str, model: str) -> Path:
    path = data_root / dataset / f"{model}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing clean dataset file: {path}")
    return path


def resolve_attack_file(data_root: Path, dataset: str, attack: str) -> Path:
    candidates = [dataset, dataset.replace("-", "_"), dataset.replace("_", "-")]
    for candidate in candidates:
        path = data_root / attack / f"{candidate}.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing attack dataset file under {data_root}/{attack} for {dataset}")


def normalize_question(text: str) -> str:
    return " ".join(str(text).split())


def validate_alignment(
    clean_records: Sequence[Dict],
    attack_records: Sequence[Dict],
    clean_path: Path,
    attack_path: Path,
) -> None:
    if len(clean_records) != len(attack_records):
        raise ValueError(
            f"Question count mismatch: {clean_path} has {len(clean_records)}, "
            f"{attack_path} has {len(attack_records)}"
        )
    for idx, (clean_item, attack_item) in enumerate(zip(clean_records, attack_records)):
        if normalize_question(clean_item["question"]) != normalize_question(attack_item["question"]):
            raise ValueError(f"Question mismatch at index {idx}: {clean_path.name} vs {attack_path.name}")


def choose_attack_answer(
    attack_answers: Sequence[str],
    node_idx: int,
    malicious_start: int,
    node_count: int,
) -> str:
    if len(attack_answers) >= node_count:
        return str(attack_answers[node_idx])
    attack_idx = node_idx - malicious_start
    if attack_idx < 0 or attack_idx >= len(attack_answers):
        raise ValueError(
            f"Attack answer index out of range: node_idx={node_idx}, "
            f"malicious_start={malicious_start}, attack_len={len(attack_answers)}"
        )
    return str(attack_answers[attack_idx])


def build_replay_records(
    clean_records: Sequence[Dict],
    attack_records: Optional[Sequence[Dict]],
    bad_count: int,
    *,
    max_questions: int = 0,
) -> List[Dict]:
    if not clean_records:
        return []

    records = list(clean_records[:max_questions] if max_questions else clean_records)
    if attack_records is None:
        return records

    attack_used = list(attack_records[: len(records)])
    node_count = len(records[0]["answers"])
    if bad_count <= 0 or bad_count >= node_count:
        raise ValueError(f"bad_count must be in [1, {node_count - 1}]")

    malicious_start = node_count - bad_count
    replay_records: List[Dict] = []
    for clean_item, attack_item in zip(records, attack_used):
        clean_answers = list(map(str, clean_item["answers"]))
        attack_answers = list(map(str, attack_item["answers"]))
        if len(attack_answers) < bad_count:
            raise ValueError(
                f"Attack pool too small for question: need >= {bad_count}, got {len(attack_answers)}"
            )
        merged_answers: List[str] = []
        for node_idx in range(node_count):
            if node_idx < malicious_start:
                merged_answers.append(clean_answers[node_idx])
            else:
                merged_answers.append(
                    choose_attack_answer(
                        attack_answers=attack_answers,
                        node_idx=node_idx,
                        malicious_start=malicious_start,
                        node_count=node_count,
                    )
                )
        replay_records.append(
            {
                "question": clean_item["question"],
                "answers": merged_answers,
            }
        )
    return replay_records


def record_to_text_df(batch_index: int, question: str, answers: Sequence[str]) -> pd.DataFrame:
    object_id = f"Q{batch_index:03d}"
    rows = [
        {
            "website": f"Node_{node_idx + 1}",
            "object": object_id,
            "fact": str(answer),
            "batch_index": batch_index,
        }
        for node_idx, answer in enumerate(answers)
    ]
    return pd.DataFrame(rows)


def replay_semantictruth(records: Sequence[Dict]) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], str], Dict[str, Dict]]:
    model = SemanticTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
    empty_numeric_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

    rows: List[Dict] = []
    preds: Dict[Tuple[int, str], str] = {}
    for batch_index, item in enumerate(records, start=1):
        text_df = record_to_text_df(batch_index, item["question"], item["answers"])
        _, out_txt = model.process_batch(empty_numeric_df.copy(), text_df)
        object_id = f"Q{batch_index:03d}"
        truth = str(out_txt["global_truth"].dropna().iloc[0]) if not out_txt.empty else ""
        preds[(batch_index, object_id)] = truth
        reliabilities = model.get_reliability()
        rows.append(
            {
                "batch_index": batch_index,
                "object": object_id,
                "question": item["question"],
                "global_truth": truth,
                "reliability_snapshot": json.dumps(reliabilities, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows), preds, model.get_reliability()


def maybe_compare_clean(
    clean_records: Sequence[Dict],
    replay_records: Sequence[Dict],
    enabled: bool,
) -> Optional[float]:
    if not enabled:
        return None
    clean_df, clean_preds, _ = replay_semantictruth(clean_records)
    noisy_df, noisy_preds, _ = replay_semantictruth(replay_records)
    _ = clean_df, noisy_df
    return metric_data_accuracy(clean_preds, noisy_preds)


def output_path(args: argparse.Namespace) -> Path:
    if args.output_csv:
        return Path(args.output_csv)
    out_dir = Path(__file__).parent / "outputs" / "semantictruth_cllm_replay"
    suffix = args.attack
    if args.attack == "substitution":
        suffix = f"{suffix}_{args.model}_vs_{args.substitute_model}"
    else:
        suffix = f"{suffix}_{args.model}"
    return out_dir / f"{args.dataset}_{suffix}_bad{args.bad_count}.csv"


def print_reliability_summary(reliabilities: Dict[str, Dict]) -> None:
    if not reliabilities:
        print("No reliability state.")
        return
    ordered = sorted(
        (
            (source, float(state["reliability"]))
            for source, state in reliabilities.items()
        ),
        key=lambda item: item[0],
    )
    print("\nFinal Source Reliability")
    for source, value in ordered:
        print(f"{source}: {value:.4f}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    clean_path = resolve_clean_file(data_root, args.dataset, args.model)
    clean_records = load_records(clean_path)

    attack_records: Optional[List[Dict]] = None
    if args.attack == "random":
        attack_path = resolve_attack_file(data_root, args.dataset, "random")
        attack_records = load_records(attack_path)
        validate_alignment(clean_records, attack_records, clean_path, attack_path)
    elif args.attack == "incorrect":
        attack_path = resolve_attack_file(data_root, args.dataset, "incorrect")
        attack_records = load_records(attack_path)
        validate_alignment(clean_records, attack_records, clean_path, attack_path)
    elif args.attack == "substitution":
        substitute_path = resolve_clean_file(data_root, args.dataset, args.substitute_model)
        attack_records = load_records(substitute_path)
        validate_alignment(clean_records, attack_records, clean_path, substitute_path)

    clean_used = list(clean_records[: args.max_questions] if args.max_questions else clean_records)
    replay_records = build_replay_records(
        clean_records=clean_records,
        attack_records=attack_records,
        bad_count=args.bad_count,
        max_questions=args.max_questions,
    )

    result_df, preds, final_reliability = replay_semantictruth(replay_records)
    out_path = output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print(
        f"dataset={args.dataset} model={args.model} attack={args.attack} "
        f"questions={len(replay_records)}"
    )
    if args.attack == "substitution":
        print(f"substitute_model={args.substitute_model}")

    if args.attack != "none" and args.compare_clean:
        data_acc = maybe_compare_clean(clean_used, replay_records, enabled=True)
        print(f"semantictruth_data_accuracy_vs_clean={data_acc:.4f}")

    print_reliability_summary(final_reliability)
    print(f"\nSaved replay CSV to {out_path}")

    if not result_df.empty:
        preview = result_df.head(3)[["batch_index", "question", "global_truth"]]
        print("\nPreview")
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
