#!/usr/bin/env python3
"""
data_CLLM 文本方法对比实验
===============================================

对比方法:
  - SenteTruth
  - BasicTruth
  - SemanticTruth

支持攻击场景:
  - random
  - incorrect
  - substitution

输入数据:
  decentralized_truth_discovery/data_CLLM/

输出:
  CSV 结果文件 + 控制台透视表

示例:
  python decentralized_truth_discovery/examples/compare_cllm_text_methods.py \
    --dataset BASE --attack random --target-model all --bad-count 4

  python decentralized_truth_discovery/examples/compare_cllm_text_methods.py \
    --dataset BASE --attack incorrect --target-model all --bad-count 4

  python decentralized_truth_discovery/examples/compare_cllm_text_methods.py \
    --dataset BASE --attack substitution --target-model gpt --substitute-model all --bad-count 4
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

from exp_utils import eval_text


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "decentralized_truth_discovery" / "data_CLLM"
OUT_ROOT = Path(__file__).parent / "outputs" / "compare_cllm_text_methods"

MODEL_FILES = ("gpt", "google", "chatglm", "llama", "tencent")
MODEL_LABELS = {
    "gpt": "ChatGPT",
    "google": "Gemini",
    "chatglm": "ChatGLM",
    "llama": "Llama",
    "tencent": "Hunyuan",
}
METHOD_ORDER = ("SenteTruth", "BasicTruth", "SemanticTruth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 data_CLLM 的文本方法对比实验")
    parser.add_argument(
        "--dataset",
        default="BASE",
        help="数据集目录名，例如 BASE / MIX / PRO / PRO-single / BASE-fix",
    )
    parser.add_argument(
        "--attack",
        required=True,
        choices=("random", "incorrect", "substitution"),
        help="攻击类型",
    )
    parser.add_argument(
        "--target-model",
        default="all",
        help="目标模型文件名，例如 gpt / google / chatglm / llama / tencent / all",
    )
    parser.add_argument(
        "--substitute-model",
        default="all",
        help="替代模型文件名，仅 substitution 使用；可取 gpt/google/chatglm/llama/tencent/all",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=4,
        help="恶意节点数，默认 4，对应 40%%",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.85,
        help="eval_text 使用的 data accuracy 阈值",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="可选，自定义输出 CSV 路径",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def normalize_question(text: str) -> str:
    return " ".join(str(text).split())


def resolve_model_list(name: str) -> List[str]:
    if name == "all":
        return list(MODEL_FILES)
    if name not in MODEL_FILES:
        supported = ", ".join(MODEL_FILES)
        raise ValueError(f"Unsupported model: {name}. Expected one of: {supported}, all")
    return [name]


def resolve_clean_file(dataset: str, model: str) -> Path:
    path = DATA_ROOT / dataset / f"{model}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path


def resolve_attack_file(dataset: str, attack: str) -> Path:
    candidates = [
        dataset,
        dataset.replace("-", "_"),
        dataset.replace("_", "-"),
    ]
    for candidate in candidates:
        path = DATA_ROOT / attack / f"{candidate}.json"
        if path.exists():
            return path
    if attack == "incorrect" and dataset == "PRO":
        raise FileNotFoundError(
            "incorrect/PRO.json 不存在。当前仓库只有 incorrect/PRO_single.json，"
            "这意味着 incorrect 场景请改用 --dataset PRO-single。"
        )
    raise FileNotFoundError(f"Missing attack file for dataset={dataset}, attack={attack}")


def validate_records_alignment(
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
        clean_q = normalize_question(clean_item.get("question", ""))
        attack_q = normalize_question(attack_item.get("question", ""))
        if clean_q != attack_q:
            raise ValueError(
                f"Question mismatch at index {idx}: {clean_path.name} != {attack_path.name}"
            )


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


def build_clean_noisy_frames(
    clean_records: Sequence[Dict],
    attack_records: Sequence[Dict],
    bad_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not clean_records:
        empty = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
        return empty, empty

    node_count = len(clean_records[0]["answers"])
    if bad_count <= 0 or bad_count >= node_count:
        raise ValueError(f"bad_count must be in [1, {node_count - 1}]")

    clean_rows: List[Dict] = []
    noisy_rows: List[Dict] = []
    malicious_start = node_count - bad_count

    for batch_index, (clean_item, attack_item) in enumerate(zip(clean_records, attack_records), start=1):
        clean_answers = list(map(str, clean_item["answers"]))
        attack_answers = list(map(str, attack_item["answers"]))
        if len(clean_answers) != node_count:
            raise ValueError(
                f"Inconsistent node count at question {batch_index}: "
                f"expected {node_count}, got {len(clean_answers)}"
            )
        if len(attack_answers) < bad_count:
            raise ValueError(
                f"Attack pool too small at question {batch_index}: "
                f"need >= {bad_count}, got {len(attack_answers)}"
            )

        object_id = f"Q{batch_index:03d}"
        for node_idx in range(node_count):
            website = f"Node_{node_idx + 1}"
            clean_fact = clean_answers[node_idx]
            noisy_fact = clean_fact
            if node_idx >= malicious_start:
                noisy_fact = choose_attack_answer(
                    attack_answers=attack_answers,
                    node_idx=node_idx,
                    malicious_start=malicious_start,
                    node_count=node_count,
                )

            clean_rows.append(
                {
                    "website": website,
                    "object": object_id,
                    "fact": clean_fact,
                    "batch_index": batch_index,
                }
            )
            noisy_rows.append(
                {
                    "website": website,
                    "object": object_id,
                    "fact": noisy_fact,
                    "batch_index": batch_index,
                }
            )

    clean_df = pd.DataFrame(clean_rows)
    noisy_df = pd.DataFrame(noisy_rows)
    return clean_df, noisy_df


def run_single_experiment(
    *,
    dataset: str,
    attack: str,
    target_model: str,
    substitute_model: Optional[str],
    bad_count: int,
    accuracy_threshold: float,
) -> List[Dict]:
    clean_path = resolve_clean_file(dataset, target_model)
    clean_records = load_records(clean_path)

    if attack == "substitution":
        if not substitute_model:
            raise ValueError("substitute_model is required for substitution attack")
        attack_path = resolve_clean_file(dataset, substitute_model)
    else:
        attack_path = resolve_attack_file(dataset, attack)

    attack_records = load_records(attack_path)
    validate_records_alignment(clean_records, attack_records, clean_path, attack_path)

    clean_df, noisy_df = build_clean_noisy_frames(clean_records, attack_records, bad_count)
    result = eval_text(clean_df, noisy_df, accuracy_threshold=accuracy_threshold)

    rows: List[Dict] = []
    for method in METHOD_ORDER:
        rows.append(
            {
                "dataset": dataset,
                "attack": attack,
                "target_model": target_model,
                "target_label": MODEL_LABELS[target_model],
                "substitute_model": substitute_model or "",
                "substitute_label": MODEL_LABELS.get(substitute_model or "", ""),
                "bad_count": bad_count,
                "bad_ratio": bad_count / clean_df["website"].nunique(),
                "question_count": clean_df["object"].nunique(),
                "method": method,
                "data_accuracy": result[method]["data_accuracy"],
            }
        )
    return rows


def build_run_plan(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    targets = resolve_model_list(args.target_model)
    if args.attack != "substitution":
        return [(target, None) for target in targets]

    substitutes = resolve_model_list(args.substitute_model)
    plan: List[Tuple[str, Optional[str]]] = []
    for target in targets:
        for sub in substitutes:
            if sub == target:
                continue
            plan.append((target, sub))
    if not plan:
        raise ValueError("No valid substitution pairs generated")
    return plan


def default_output_csv(args: argparse.Namespace) -> Path:
    suffix = args.attack
    if args.attack == "substitution":
        suffix = f"{suffix}_{args.target_model}_vs_{args.substitute_model}"
    else:
        suffix = f"{suffix}_{args.target_model}"
    filename = f"{args.dataset}_{suffix}_bad{args.bad_count}.csv"
    return OUT_ROOT / filename


def print_summary(df: pd.DataFrame, attack: str) -> None:
    if df.empty:
        print("No results.")
        return
    if attack == "substitution":
        pivot_key = "substitute_label"
    else:
        pivot_key = "target_label"
    pivot = (
        df.pivot_table(
            index="method",
            columns=pivot_key,
            values="data_accuracy",
            aggfunc="first",
        )
        .reindex(METHOD_ORDER)
        .round(4)
    )
    print("\n=== Data Accuracy Pivot ===")
    print(pivot.to_string())


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for target_model, substitute_model in build_run_plan(args):
        print(
            f"\n[run] dataset={args.dataset} attack={args.attack} "
            f"target={target_model}"
            + (f" substitute={substitute_model}" if substitute_model else "")
        )
        rows.extend(
            run_single_experiment(
                dataset=args.dataset,
                attack=args.attack,
                target_model=target_model,
                substitute_model=substitute_model,
                bad_count=args.bad_count,
                accuracy_threshold=args.accuracy_threshold,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No results generated")

    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv(args)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print_summary(df, args.attack)
    print(f"\nSaved CSV to {output_csv}")


if __name__ == "__main__":
    main()
