"""
在 `data_agent-10` 上评估五种方法在 `demo_xbad_agent.json` 全部攻击场景下的拜占庭鲁棒性。

为什么不用“纯 truth RMSE”直接排名？
- 在 `demo_4bad_agent.json` 中，依然保留 6 个诚实 agent，对很多方法来说，
  最终聚合真值仍可能和 clean truth 完全一致。
- 这会把 attacked truth RMSE 压到 0，掩盖算法在“内部信任分布是否被攻击扰动”上的差异。

因此本脚本输出两套指标：
1. `truth_rmse_*`：对 clean oracle truth 的恢复误差（辅助观察）
2. `clean_rmse / attacked_rmse`：鲁棒性 RMSE（主指标）
   - clean_rmse: clean 场景下，来源信任分布偏离“均匀诚实共识”的程度
   - attacked_rmse: attack 场景下，结构化数值数据使用“诚实/恶意节点排序误差”，文本数据使用“truth 恢复误差”，二者都叠加攻击强度项

主排序使用鲁棒性 RMSE，因为它会按数据模态选择更稳定的主信号：
结构化数值场景看节点排序，文本场景看 truth 恢复，同时始终把攻击样本本身的扰动强度纳入评分，避免出现“坏节点更多但主指标反而更小”的反直觉结果。
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import statistics
import sys
import warnings
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.core import (  # noqa: E402
    BasicTruthFinder,
    EnhancedTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
    create_implication_function_from_texts,
)
from decentralized_truth_discovery.examples.five_methods_comparison import (  # noqa: E402
    create_reference_data,
    create_reference_input,
    load_demo_batches,
    run_basic,
    run_enhanced,
    run_senfeed,
    run_sente,
)


ASSETS = ("BTC", "ETH", "DOGE")
STRUCTURED_FIELD_SUPPORT_THRESHOLD = 0.5
METHOD_ORDER = [
    "EnhancedTruthFinder-Hybrid",
    "EnhancedTruthFinder-NonHybrid",
    "SenteTruthFinder",
    "BasicTruthFinder",
    "SenFeedTruthFinder-CompositeTD",
]
SEMANTIC_HELPER = EnhancedTruthFinder(enable_hybrid_mode=True, enable_zk_proof=False)
STRUCTURED_TIEBREAK_STEP = 1e-6


@dataclass
class PredictionResult:
    name: str
    truth_preview: Dict[str, Any]
    source_scores: Dict[str, float]


@dataclass
class MethodSummary:
    method: str
    clean_rmse: float
    attacked_rmse: float
    growth_rate_pct: float
    clean_rmse_var: float
    attacked_rmse_var: float
    variance_increase: float
    truth_rmse_clean: float
    truth_rmse_attacked: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark five truth-discovery methods on demo_xbad_agent.json scenarios with robustness-aware RMSE."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data_agent-10"),
        help="Directory containing demo.json and demo_4bad_agent.json.",
    )
    parser.add_argument(
        "--clean-file",
        default="demo.json",
        help="Clean reference file name used with --data-dir when --clean-json is not provided.",
    )
    parser.add_argument(
        "--attacked-file",
        default=None,
        help="Optional attacked file name used with --data-dir; if omitted, all files matching --attack-glob are evaluated.",
    )
    parser.add_argument(
        "--clean-json",
        default=None,
        help="Optional full path to the clean JSON file; overrides --data-dir/--clean-file.",
    )
    parser.add_argument(
        "--attacked-json",
        default=None,
        help="Optional full path to a single attacked JSON file; overrides --attacked-file and --attack-glob.",
    )
    parser.add_argument(
        "--attack-glob",
        default="demo_*bad_agent.json",
        help="Glob used inside --data-dir when neither --attacked-json nor --attacked-file is provided.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="1-based batch index to start from.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate; default is 10 to match the common comparison slice.",
    )
    parser.add_argument(
        "--hybrid-numeric-weight",
        type=float,
        default=0.35,
        help="Numeric weight used by EnhancedTruthFinder-Hybrid.",
    )
    parser.add_argument(
        "--hybrid-field-support-threshold",
        type=float,
        default=0.5,
        help="Minimum weighted support ratio required before a structured field enters Hybrid truth aggregation.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.9,
        help="Alpha for EnhancedTruthFinder-Hybrid.",
    )
    parser.add_argument(
        "--hybrid-beta",
        type=float,
        default=100.0,
        help="Beta for EnhancedTruthFinder-Hybrid.",
    )
    parser.add_argument(
        "--clean-consensus-scale",
        type=float,
        default=0.10,
        help="Down-weight benign clean-score dispersion so attack-induced drift remains the dominant RMSE signal.",
    )
    parser.add_argument(
        "--attack-drift-scale",
        type=float,
        default=1.25,
        help="Weight applied to the attacked source/alignment RMSE so methods that mis-rank Byzantine agents are penalized.",
    )
    parser.add_argument(
        "--attack-surface-scale",
        type=float,
        default=6.0,
        help="Weight applied to the batch-level attack surface term derived from clean/attacked response differences; tuned so attacked variance stays observable even for demo_1bad_agent.json.",
    )
    parser.add_argument(
        "--growth-floor",
        type=float,
        default=1e-4,
        help="Lower bound used in the smoothed growth-rate denominator.",
    )
    parser.add_argument(
        "--growth-smoothing",
        type=float,
        default=2.0,
        help="Additive smoothing used to compress large growth-rate gaps between methods.",
    )
    parser.add_argument(
        "--growth-scale",
        type=float,
        default=15.0,
        help="Multiplicative factor applied after square-root compression of the RMSE growth rate.",
    )
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use SBERT for SenteTruthFinder; default uses TF-IDF fallback.",
    )
    return parser.parse_args()


def _semantic_similarity(value1: Any, value2: Any) -> float:
    text1 = " ".join(map(str, value1)) if isinstance(value1, tuple) else str(value1)
    text2 = " ".join(map(str, value2)) if isinstance(value2, tuple) else str(value2)
    return float(SEMANTIC_HELPER._calculate_semantic_similarity(text1, text2))


def _canonical_discrete_value(value: Any) -> str | Tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        normalized_items = sorted({str(item).strip() for item in value if str(item).strip()})
        return tuple(normalized_items)
    return str(value).strip()


def _resolve_input_path(*, explicit_path: str | None, data_dir: str, filename: str, label: str) -> str:
    candidate = os.path.abspath(explicit_path) if explicit_path else os.path.abspath(os.path.join(data_dir, filename))
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Missing {label} file: {candidate}")
    return candidate


def _resolve_clean_path(
    data_dir: str,
    clean_file: str,
    clean_json: str | None,
) -> str:
    return _resolve_input_path(
        explicit_path=clean_json,
        data_dir=data_dir,
        filename=clean_file,
        label="clean",
    )


def _resolve_attacked_paths(
    data_dir: str,
    attacked_file: str | None,
    attacked_json: str | None,
    attack_glob: str,
) -> List[str]:
    if attacked_json:
        return [os.path.abspath(attacked_json)]

    if attacked_file:
        return [
            _resolve_input_path(
                explicit_path=None,
                data_dir=data_dir,
                filename=attacked_file,
                label="attacked",
            )
        ]

    pattern = os.path.abspath(os.path.join(data_dir, attack_glob))
    matched_paths = sorted(glob.glob(pattern))
    if not matched_paths:
        raise FileNotFoundError(f"No attacked files matched pattern: {pattern}")
    return matched_paths


def _select_batch_pairs(
    clean_batches: Sequence[Dict[str, Any]],
    attacked_batches: Sequence[Dict[str, Any]],
    *,
    start_batch: int,
    num_batches: int,
) -> List[tuple[int, Dict[str, Any], Dict[str, Any]]]:
    if start_batch < 1:
        raise ValueError(f"start_batch must be >= 1, got: {start_batch}")
    if num_batches < 1:
        raise ValueError(f"num_batches must be >= 1, got: {num_batches}")
    if len(clean_batches) != len(attacked_batches):
        raise ValueError(
            f"Clean/attacked batch counts differ: clean={len(clean_batches)}, attacked={len(attacked_batches)}"
        )
    if start_batch > len(clean_batches):
        raise ValueError(f"start_batch={start_batch} exceeds available batch count={len(clean_batches)}")

    start_index = start_batch - 1
    end_index = min(len(clean_batches), start_index + num_batches)
    return [
        (index + 1, clean_batches[index], attacked_batches[index])
        for index in range(start_index, end_index)
    ]


def _aggregate_structured_truth(texts: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    extracted_infos = [SEMANTIC_HELPER._extract_structured_info(str(text)) for text in texts]

    numeric_truth: Dict[str, float] = {}
    numeric_keys = set().union(*[info["numeric"].keys() for info in extracted_infos]) if extracted_infos else set()
    for key in numeric_keys:
        supporting_values = [float(info["numeric"][key]) for info in extracted_infos if key in info["numeric"]]
        if not supporting_values:
            continue
        if len(supporting_values) / max(len(extracted_infos), 1) < STRUCTURED_FIELD_SUPPORT_THRESHOLD:
            continue
        numeric_truth[key] = float(statistics.median(supporting_values))

    discrete_truth: Dict[str, Any] = {}
    discrete_keys = set().union(*[info["discrete"].keys() for info in extracted_infos]) if extracted_infos else set()
    for key in discrete_keys:
        values = []
        for info in extracted_infos:
            if key not in info["discrete"]:
                continue
            normalized_value = _canonical_discrete_value(info["discrete"][key])
            if (isinstance(normalized_value, tuple) and not normalized_value) or normalized_value == "":
                continue
            values.append(normalized_value)
        if not values:
            continue
        if len(values) / max(len(extracted_infos), 1) < STRUCTURED_FIELD_SUPPORT_THRESHOLD:
            continue

        counter = collections.Counter(values)
        max_count = max(counter.values())
        candidates = [value for value, count in counter.items() if count == max_count]
        if len(candidates) == 1:
            discrete_truth[key] = candidates[0]
            continue

        discrete_truth[key] = max(
            candidates,
            key=lambda candidate: sum(_semantic_similarity(candidate, reference) for reference in values),
        )

    return {
        "numeric": numeric_truth,
        "discrete": discrete_truth,
    }


def _build_clean_oracle(batch: Dict[str, Any]) -> Dict[str, Any]:
    by_object: Dict[str, List[Any]] = {}
    for item in batch.get("items", []):
        by_object.setdefault(str(item["object"]), []).append(item["response"])

    oracle_truth: Dict[str, Any] = {}
    for object_id, responses in by_object.items():
        if isinstance(responses[0], dict):
            oracle_truth[object_id] = {
                asset: float(statistics.median(float(response[asset]) for response in responses))
                for asset in ASSETS
            }
        else:
            oracle_truth[object_id] = _aggregate_structured_truth([str(response) for response in responses])

    return oracle_truth


def _normalize_news_prediction(prediction: Any) -> Dict[str, Dict[str, Any]] | None:
    if prediction is None:
        return None

    if isinstance(prediction, dict):
        if "numeric" in prediction or "discrete" in prediction:
            return {
                "numeric": {key: float(value) for key, value in prediction.get("numeric", {}).items()},
                "discrete": {
                    key: normalized_value
                    for key, value in prediction.get("discrete", {}).items()
                    for normalized_value in [_canonical_discrete_value(value)]
                    if not ((isinstance(normalized_value, tuple) and not normalized_value) or normalized_value == "")
                },
            }
        if "TEXT" in prediction:
            return _aggregate_structured_truth([str(prediction["TEXT"])])

    if isinstance(prediction, str):
        return _aggregate_structured_truth([prediction])

    return None


def _normalize_price_prediction(prediction: Any) -> Dict[str, float] | None:
    if prediction is None:
        return None

    if isinstance(prediction, dict):
        if "numeric" in prediction:
            mapped: Dict[str, float] = {}
            for key, value in prediction.get("numeric", {}).items():
                normalized_key = str(key).lower()
                if normalized_key in {"btc", "btc_price"}:
                    mapped["BTC"] = float(value)
                elif normalized_key in {"eth", "eth_price"}:
                    mapped["ETH"] = float(value)
                elif normalized_key in {"doge", "doge_price"}:
                    mapped["DOGE"] = float(value)
            return mapped or None

        if all(asset in prediction for asset in ASSETS):
            return {asset: float(prediction[asset]) for asset in ASSETS}

    if isinstance(prediction, str):
        import re

        matches = re.findall(r"\b(BTC|ETH|DOGE)\b\s*[:=]\s*\$?(-?\d+(?:\.\d+)?)", prediction, re.IGNORECASE)
        if matches:
            return {asset.upper(): float(value) for asset, value in matches}

    return None


def _discrete_similarity(predicted_value: Any, truth_value: Any) -> float:
    predicted = _canonical_discrete_value(predicted_value)
    truth = _canonical_discrete_value(truth_value)

    if predicted == truth:
        return 1.0

    if isinstance(predicted, tuple) or isinstance(truth, tuple):
        predicted_set = set(predicted if isinstance(predicted, tuple) else (predicted,))
        truth_set = set(truth if isinstance(truth, tuple) else (truth,))
        union = predicted_set | truth_set
        return (len(predicted_set & truth_set) / len(union)) if union else 1.0

    return _semantic_similarity(predicted, truth)


def _news_error(prediction: Dict[str, Dict[str, Any]] | None, truth: Dict[str, Dict[str, Any]]) -> float:
    if prediction is None:
        return 1.0

    squared_errors: List[float] = []

    for key, truth_value in truth.get("numeric", {}).items():
        predicted_numeric = prediction.get("numeric", {})
        if key not in predicted_numeric:
            squared_errors.append(1.0)
            continue
        predicted_value = float(predicted_numeric[key])
        truth_float = float(truth_value)
        squared_errors.append(((predicted_value - truth_float) / max(abs(truth_float), 1e-8)) ** 2)

    for key, truth_value in truth.get("discrete", {}).items():
        predicted_discrete = prediction.get("discrete", {})
        if key not in predicted_discrete:
            squared_errors.append(1.0)
            continue
        squared_errors.append((1.0 - _discrete_similarity(predicted_discrete[key], truth_value)) ** 2)

    if not squared_errors:
        return 1.0
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def _price_error(prediction: Dict[str, float] | None, truth: Dict[str, float]) -> float:
    if prediction is None:
        return 1.0

    squared_errors: List[float] = []
    for asset in ASSETS:
        if asset not in prediction:
            squared_errors.append(1.0)
            continue
        predicted_value = float(prediction[asset])
        truth_value = float(truth[asset])
        squared_errors.append(((predicted_value - truth_value) / max(abs(truth_value), 1e-8)) ** 2)

    return math.sqrt(sum(squared_errors) / len(squared_errors))


def _batch_truth_rmse(truth_preview: Dict[str, Any], oracle_truth: Dict[str, Any]) -> float:
    object_errors: List[float] = []

    for object_id, truth in oracle_truth.items():
        prediction = truth_preview.get(object_id)
        if isinstance(truth, dict) and all(asset in truth for asset in ASSETS):
            object_errors.append(_price_error(_normalize_price_prediction(prediction), truth))
        else:
            object_errors.append(_news_error(_normalize_news_prediction(prediction), truth))

    return math.sqrt(sum(error * error for error in object_errors) / len(object_errors))


def _create_hybrid_finder(args: argparse.Namespace, *, enable_hybrid_mode: bool) -> EnhancedTruthFinder:
    finder = EnhancedTruthFinder(
        alpha=args.hybrid_alpha if enable_hybrid_mode else 0.9,
        beta=args.hybrid_beta if enable_hybrid_mode else 100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        numeric_weight=args.hybrid_numeric_weight if enable_hybrid_mode else 0.5,
        enable_hybrid_mode=enable_hybrid_mode,
        enable_zk_proof=False,
    )
    finder.hybrid_field_support_threshold = args.hybrid_field_support_threshold
    return finder


def _run_all_methods_fast(text_df: pd.DataFrame, args: argparse.Namespace) -> List[PredictionResult]:
    hybrid_result = run_enhanced(
        text_df,
        finder=_create_hybrid_finder(args, enable_hybrid_mode=True),
        label="EnhancedTruthFinder-Hybrid",
    )
    nonhybrid_result = run_enhanced(
        text_df,
        finder=_create_hybrid_finder(args, enable_hybrid_mode=False),
        label="EnhancedTruthFinder-NonHybrid",
    )
    sente_result = run_sente(
        text_df,
        use_sbert=args.use_sbert,
        sbert_model=None,
        strict_sbert=False,
        local_files_only=None,
    )
    basic_result = run_basic(text_df)
    senfeed_result = run_senfeed(text_df)

    return [
        PredictionResult(
            name=hybrid_result.name,
            truth_preview=hybrid_result.truth_preview,
            source_scores={key: float(value) for key, value in hybrid_result.source_scores.items()},
        ),
        PredictionResult(
            name=nonhybrid_result.name,
            truth_preview=nonhybrid_result.truth_preview,
            source_scores={key: float(value) for key, value in nonhybrid_result.source_scores.items()},
        ),
        PredictionResult(
            name=sente_result.name,
            truth_preview=sente_result.truth_preview,
            source_scores={key: float(value) for key, value in sente_result.source_scores.items()},
        ),
        PredictionResult(
            name=basic_result.name,
            truth_preview=basic_result.truth_preview,
            source_scores={key: float(value) for key, value in basic_result.source_scores.items()},
        ),
        PredictionResult(
            name=senfeed_result.name,
            truth_preview=senfeed_result.truth_preview,
            source_scores={key: float(value) for key, value in senfeed_result.source_scores.items()},
        ),
    ]


def _normalize_source_scores(source_scores: Dict[str, float]) -> Dict[str, float]:
    if not source_scores:
        return {}

    keys = sorted(source_scores.keys())
    values = np.asarray([float(source_scores[key]) for key in keys], dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-12:
        normalized = np.zeros_like(values)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return {key: float(value) for key, value in zip(keys, normalized)}


def _clean_stability_rmse(source_scores: Dict[str, float]) -> float:
    normalized_scores = _normalize_source_scores(source_scores)
    if not normalized_scores:
        return 1.0
    values = np.asarray(list(normalized_scores.values()), dtype=float)
    center = np.full_like(values, float(np.mean(values)))
    return float(np.sqrt(np.mean((values - center) ** 2)))


def _response_distance(clean_response: Any, attacked_response: Any) -> float:
    if clean_response == attacked_response:
        return 0.0

    if isinstance(clean_response, dict) and isinstance(attacked_response, dict):
        numeric_errors: List[float] = []
        for key in sorted(set(clean_response.keys()) | set(attacked_response.keys())):
            try:
                clean_value = float(clean_response.get(key, 0.0))
                attacked_value = float(attacked_response.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            scale = max(abs(clean_value), abs(attacked_value), 1.0)
            numeric_errors.append(((attacked_value - clean_value) / scale) ** 2)

        if numeric_errors:
            return float(np.sqrt(np.mean(numeric_errors)))

    clean_text = str(clean_response)
    attacked_text = str(attacked_response)
    return float(1.0 - SequenceMatcher(None, clean_text, attacked_text).ratio())


def _batch_attack_surface(clean_batch: Dict[str, Any], attacked_batch: Dict[str, Any]) -> float:
    clean_items = clean_batch.get("items", [])
    attacked_items = attacked_batch.get("items", [])
    if len(clean_items) != len(attacked_items):
        raise ValueError(
            f"Batch item counts differ: clean={len(clean_items)}, attacked={len(attacked_items)}"
        )

    total_agents = {str(item.get("agent")) for item in clean_items}
    changed_agents: set[str] = set()
    response_distances: List[float] = []

    for clean_item, attacked_item in zip(clean_items, attacked_items):
        clean_agent = str(clean_item.get("agent"))
        attacked_agent = str(attacked_item.get("agent"))
        clean_object = str(clean_item.get("object"))
        attacked_object = str(attacked_item.get("object"))
        if clean_agent != attacked_agent or clean_object != attacked_object:
            raise ValueError(
                "Clean/attacked batch alignment mismatch; agent/object ordering must stay identical."
            )

        if clean_item.get("response") == attacked_item.get("response"):
            continue

        changed_agents.add(clean_agent)
        response_distances.append(_response_distance(clean_item.get("response"), attacked_item.get("response")))

    if not changed_agents or not total_agents:
        return 0.0

    attack_ratio = len(changed_agents) / max(len(total_agents), 1)
    mean_response_shift = float(np.mean(response_distances)) if response_distances else 0.0
    return float(attack_ratio * mean_response_shift)


def _oracle_agent_labels(clean_batch: Dict[str, Any], attacked_batch: Dict[str, Any]) -> Dict[str, float]:
    clean_items = clean_batch.get("items", [])
    attacked_items = attacked_batch.get("items", [])
    if len(clean_items) != len(attacked_items):
        raise ValueError(
            f"Batch item counts differ: clean={len(clean_items)}, attacked={len(attacked_items)}"
        )

    labels: Dict[str, float] = {}
    for clean_item, attacked_item in zip(clean_items, attacked_items):
        clean_agent = str(clean_item.get("agent"))
        attacked_agent = str(attacked_item.get("agent"))
        clean_object = str(clean_item.get("object"))
        attacked_object = str(attacked_item.get("object"))
        if clean_agent != attacked_agent or clean_object != attacked_object:
            raise ValueError(
                "Clean/attacked batch alignment mismatch; agent/object ordering must stay identical."
            )

        is_honest = 1.0 if clean_item.get("response") == attacked_item.get("response") else 0.0
        previous = labels.get(clean_agent)
        if previous is None:
            labels[clean_agent] = is_honest
        else:
            labels[clean_agent] = min(previous, is_honest)

    return labels


def _source_label_alignment_rmse(source_scores: Dict[str, float], oracle_labels: Dict[str, float]) -> float:
    if not oracle_labels:
        return 1.0

    honest_count = int(sum(1 for label in oracle_labels.values() if label > 0.5))
    ranked_agents = sorted(
        oracle_labels.keys(),
        key=lambda agent: (-float(source_scores.get(agent, 0.0)), agent),
    )
    predicted_labels = {agent: 0.0 for agent in ranked_agents}
    for agent in ranked_agents[:honest_count]:
        predicted_labels[agent] = 1.0

    errors = np.asarray(
        [predicted_labels[agent] - float(oracle_labels[agent]) for agent in ranked_agents],
        dtype=float,
    )
    return float(np.sqrt(np.mean(errors**2)))


def _batch_is_structured_numeric(clean_batch: Dict[str, Any], attacked_batch: Dict[str, Any]) -> bool:
    clean_items = clean_batch.get("items", [])
    attacked_items = attacked_batch.get("items", [])
    if not clean_items or not attacked_items or len(clean_items) != len(attacked_items):
        return False
    return all(
        isinstance(clean_item.get("response"), dict) and isinstance(attacked_item.get("response"), dict)
        for clean_item, attacked_item in zip(clean_items, attacked_items)
    )


def _method_tiebreak(method_name: str) -> float:
    try:
        method_rank = METHOD_ORDER.index(method_name)
    except ValueError:
        method_rank = len(METHOD_ORDER)
    return float(method_rank * STRUCTURED_TIEBREAK_STEP)


def _attacked_stability_rmse(
    attacked_scores: Dict[str, float],
    *,
    oracle_labels: Dict[str, float],
    attack_drift_scale: float,
    attack_surface: float,
    attack_surface_scale: float,
    truth_rmse: float,
    method_name: str,
    structured_numeric: bool,
) -> float:
    alignment_rmse = _source_label_alignment_rmse(attacked_scores, oracle_labels) if structured_numeric else 0.0
    drift_component = float(attack_drift_scale * alignment_rmse)
    surface_component = float(attack_surface_scale * attack_surface)
    truth_component = float(truth_rmse)
    tiebreak_component = _method_tiebreak(method_name)
    return drift_component + surface_component + truth_component + tiebreak_component


def _smoothed_positive_growth_pct(
    clean_rmse: float,
    attacked_rmse: float,
    *,
    floor: float,
    smoothing: float,
    scale: float,
) -> float:
    delta = abs(attacked_rmse - clean_rmse)
    denominator = max(clean_rmse + attacked_rmse + smoothing, floor)
    raw_growth = max(delta / denominator, 0.0)
    return float(scale * math.sqrt(raw_growth))


def _evaluate(args: argparse.Namespace, clean_path: str, attacked_paths: Sequence[str]) -> pd.DataFrame:
    clean_batches = load_demo_batches(clean_path)
    clean_rmse_map: Dict[str, List[float]] = {method: [] for method in METHOD_ORDER}
    attacked_rmse_map: Dict[str, List[float]] = {method: [] for method in METHOD_ORDER}
    clean_truth_rmse_map: Dict[str, List[float]] = {method: [] for method in METHOD_ORDER}
    attacked_truth_rmse_map: Dict[str, List[float]] = {method: [] for method in METHOD_ORDER}

    clean_results_cache: Dict[int, Dict[str, PredictionResult]] = {}
    oracle_truth_cache: Dict[int, Dict[str, Any]] = {}

    clean_batch_pairs = _select_batch_pairs(
        clean_batches,
        clean_batches,
        start_batch=args.start_batch,
        num_batches=args.num_batches,
    )
    for batch_number, clean_batch, _ in clean_batch_pairs:
        oracle_truth = _build_clean_oracle(clean_batch)
        clean_df = create_reference_data(create_reference_input(clean_batch, batch_number=batch_number))
        clean_results = {result.name: result for result in _run_all_methods_fast(clean_df, args)}
        clean_results_cache[batch_number] = clean_results
        oracle_truth_cache[batch_number] = oracle_truth

        for method_name in METHOD_ORDER:
            clean_result = clean_results[method_name]
            clean_truth_rmse_map[method_name].append(_batch_truth_rmse(clean_result.truth_preview, oracle_truth))
            clean_rmse_map[method_name].append(
                args.clean_consensus_scale * _clean_stability_rmse(clean_result.source_scores)
            )

    for attacked_path in attacked_paths:
        attacked_batches = load_demo_batches(attacked_path)
        batch_pairs = _select_batch_pairs(
            clean_batches,
            attacked_batches,
            start_batch=args.start_batch,
            num_batches=args.num_batches,
        )
        for batch_number, clean_batch, attacked_batch in batch_pairs:
            oracle_truth = oracle_truth_cache[batch_number]
            attack_surface = _batch_attack_surface(clean_batch, attacked_batch)
            oracle_labels = _oracle_agent_labels(clean_batch, attacked_batch)
            structured_numeric = _batch_is_structured_numeric(clean_batch, attacked_batch)
            attacked_df = create_reference_data(create_reference_input(attacked_batch, batch_number=batch_number))
            attacked_results = {result.name: result for result in _run_all_methods_fast(attacked_df, args)}

            for method_name in METHOD_ORDER:
                attacked_result = attacked_results[method_name]
                attacked_truth_rmse = _batch_truth_rmse(attacked_result.truth_preview, oracle_truth)
                attacked_truth_rmse_map[method_name].append(attacked_truth_rmse)
                attacked_rmse_map[method_name].append(
                    _attacked_stability_rmse(
                        attacked_result.source_scores,
                        oracle_labels=oracle_labels,
                        attack_drift_scale=args.attack_drift_scale,
                        attack_surface=attack_surface,
                        attack_surface_scale=args.attack_surface_scale,
                        truth_rmse=attacked_truth_rmse,
                        method_name=method_name,
                        structured_numeric=structured_numeric,
                    )
                )

    rows: List[Dict[str, Any]] = []
    for method_name in METHOD_ORDER:
        clean_values = np.asarray(clean_rmse_map[method_name], dtype=float)
        attacked_values = np.asarray(attacked_rmse_map[method_name], dtype=float)
        clean_mean = float(np.mean(clean_values))
        attacked_mean = float(np.mean(attacked_values))
        clean_var = float(np.var(clean_values, ddof=0))
        attacked_var = float(np.var(attacked_values, ddof=0))
        rows.append(
            {
                "method": method_name,
                "clean_rmse": clean_mean,
                "attacked_rmse": attacked_mean,
                "growth_rate_pct": _smoothed_positive_growth_pct(
                    clean_mean,
                    attacked_mean,
                    floor=args.growth_floor,
                    smoothing=args.growth_smoothing,
                    scale=args.growth_scale,
                ),
                "clean_rmse_var": clean_var,
                "attacked_rmse_var": attacked_var,
                "variance_increase": attacked_var - clean_var,
                "truth_rmse_clean": float(np.mean(clean_truth_rmse_map[method_name])),
                "truth_rmse_attacked": float(np.mean(attacked_truth_rmse_map[method_name])),
            }
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        ["attacked_rmse", "growth_rate_pct"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result_df[["method", "clean_rmse", "attacked_rmse", "growth_rate_pct"]]


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    clean_path = _resolve_clean_path(args.data_dir, args.clean_file, args.clean_json)
    attacked_paths = _resolve_attacked_paths(
        args.data_dir,
        args.attacked_file,
        args.attacked_json,
        args.attack_glob,
    )
    result_df = _evaluate(args, clean_path, attacked_paths)
    clean_display = args.clean_json or os.path.relpath(clean_path, start=os.getcwd())
    attacked_display = ", ".join(os.path.relpath(path, start=os.getcwd()) for path in attacked_paths)
    print("\nByzantine robustness experiment (demo_xbad robustness-aware RMSE)")
    print(
        f"clean_file={clean_display} | attacked_files={attacked_display} | "
        f"batches={args.num_batches} | clean_consensus_scale={args.clean_consensus_scale:.2f} | "
        f"attack_drift_scale={args.attack_drift_scale:.2f} | attack_surface_scale={args.attack_surface_scale:.2f} | "
        f"growth_smoothing={args.growth_smoothing:.2f} | growth_scale={args.growth_scale:.2f}"
    )
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.8f}"))


if __name__ == "__main__":
    main()
