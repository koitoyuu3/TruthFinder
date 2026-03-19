from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.examples.proof_benchmark_utils import (  # noqa: E402
    HYBRID_METHOD_NAME,
    METHOD_NAMES,
    build_saved_bundle_artifacts,
    bundle_group_payload_bytes,
    create_reference_input,
    create_reference_data,
    default_experiment_id,
    ensure_directory,
    format_float,
    load_batches_from_path,
    mean_source_score_pct,
    normalize_series,
    verify_bundle_payload,
    write_json,
)


DEFAULT_INPUT_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_agent-10", "demo.json")
)
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))


@dataclass(frozen=True)
class WorkloadSpec:
    workload_index: int
    batch_start: int
    batch_end: int
    batch_group_size: int


@dataclass(frozen=True)
class RunContext:
    workload_index: int
    agent_count: int
    subset_index: int
    total_subsets: int
    selected_agents: List[str]
    batch_start: int
    batch_end: int
    item_count: int
    website_count: int
    object_count: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected integer >= 1, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected integer >= 0, got {value!r}")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected float >= 0, got {value!r}")
    return parsed


def _normalize_weights(*weights: float) -> List[float]:
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        raise ValueError("At least one weight must be > 0.")
    return [float(weight) / total for weight in weights]


def _agent_sort_key(agent_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(agent_name))
    if match:
        return int(match.group(1)), str(agent_name)
    return math.inf, str(agent_name)


def _parse_agent_counts(raw_value: str, max_agents: int) -> List[int]:
    if max_agents < 1:
        raise ValueError("max_agents must be >= 1.")

    if raw_value.strip().lower() == "auto":
        if max_agents <= 5:
            counts = list(range(2, max_agents + 1)) if max_agents >= 2 else [1]
        else:
            counts = [count for count in [4, 6, 8, 10] if count <= max_agents]
            if max_agents not in counts:
                counts.append(max_agents)
        return sorted(set(counts))

    counts = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        counts.append(_positive_int(token))
    if not counts:
        raise ValueError("agent_counts cannot be empty.")

    unique_counts = sorted(set(counts))
    invalid_counts = [count for count in unique_counts if count > max_agents]
    if invalid_counts:
        raise ValueError(
            f"Requested agent counts {invalid_counts} exceed available max agent count {max_agents}."
        )
    return unique_counts


def _resolve_num_workloads(raw_value: int, total_batches: int, start_batch: int, batch_group_size: int) -> int:
    if raw_value > 0:
        return raw_value
    available_full_workloads = (total_batches - start_batch + 1) // batch_group_size
    if available_full_workloads < 1:
        raise ValueError(
            f"No full workloads available for total_batches={total_batches}, start_batch={start_batch}, "
            f"batch_group_size={batch_group_size}."
        )
    return available_full_workloads


def _build_workload_specs(
    total_batches: int,
    *,
    start_batch: int,
    batch_group_size: int,
    num_workloads: int,
) -> List[WorkloadSpec]:
    specs: List[WorkloadSpec] = []
    current_start = start_batch
    for workload_index in range(1, num_workloads + 1):
        current_end = current_start + batch_group_size - 1
        if current_end > total_batches:
            break
        specs.append(
            WorkloadSpec(
                workload_index=workload_index,
                batch_start=current_start,
                batch_end=current_end,
                batch_group_size=batch_group_size,
            )
        )
        current_start = current_end + 1

    if not specs:
        raise ValueError(
            f"No full workloads available for start_batch={start_batch}, batch_group_size={batch_group_size}, "
            f"num_workloads={num_workloads}, total_batches={total_batches}."
        )
    return specs


def _collect_workload_items(all_batches: Sequence[Dict[str, Any]], workload: WorkloadSpec) -> List[Dict[str, Any]]:
    workload_items: List[Dict[str, Any]] = []
    for batch_number in range(workload.batch_start, workload.batch_end + 1):
        batch = all_batches[batch_number - 1]
        workload_items.extend(create_reference_input(batch, batch_number=batch_number))
    return workload_items


def _extract_agents(reference_items: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(item["agent"]).strip() for item in reference_items}, key=_agent_sort_key)


def _select_agent_subsets(
    agents: Sequence[str],
    *,
    agent_count: int,
    subset_samples: int,
    base_seed: int,
    workload_index: int,
    strategy: str,
) -> List[List[str]]:
    sorted_agents = sorted([str(agent) for agent in agents], key=_agent_sort_key)
    if agent_count > len(sorted_agents):
        raise ValueError(
            f"Requested agent_count={agent_count} but workload only has {len(sorted_agents)} agents."
        )

    if strategy == "prefix":
        return [sorted_agents[:agent_count]]

    if subset_samples < 1:
        raise ValueError(f"subset_samples must be >= 1, got: {subset_samples}")

    max_combinations = math.comb(len(sorted_agents), agent_count)
    target = min(subset_samples, max_combinations)
    rng = random.Random(base_seed + 1009 * workload_index + 101 * agent_count)
    seen = set()
    subsets: List[List[str]] = []
    attempts = 0
    max_attempts = max(target * 50, 100)
    while len(subsets) < target and attempts < max_attempts:
        sampled = tuple(sorted(rng.sample(sorted_agents, agent_count), key=_agent_sort_key))
        if sampled not in seen:
            seen.add(sampled)
            subsets.append(list(sampled))
        attempts += 1

    if len(subsets) < target:
        raise RuntimeError(
            f"Unable to sample {target} unique subsets for agent_count={agent_count} from {len(sorted_agents)} agents."
        )
    return subsets


def _filter_reference_items_by_agents(
    reference_items: Sequence[Dict[str, Any]],
    selected_agents: Sequence[str],
) -> List[Dict[str, Any]]:
    selected = {str(agent) for agent in selected_agents}
    return [item for item in reference_items if str(item["agent"]).strip() in selected]


def _run_method_by_name(
    method_name: str,
    text_df: pd.DataFrame,
    *,
    use_sbert: bool,
    sbert_model: str | None,
    strict_sbert: bool,
    local_files_only: bool,
):
    from decentralized_truth_discovery.examples.proof_benchmark_utils import run_all_methods_for_text_df

    method_map = {
        result.name: result
        for result in run_all_methods_for_text_df(
            text_df,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            strict_sbert=strict_sbert,
            local_files_only=local_files_only,
        )
    }
    return method_map[method_name]


def _compute_scores(
    raw_results_df: pd.DataFrame,
    *,
    size_weight: float,
    verify_time_weight: float,
    verify_memory_weight: float,
    quality_weight: float,
) -> pd.DataFrame:
    df = raw_results_df.copy()
    df["proof_size_rank"] = np.nan
    df["verify_time_rank"] = np.nan
    df["verify_memory_rank"] = np.nan
    df["quality_rank"] = np.nan
    df["proof_size_score"] = np.nan
    df["verify_time_score"] = np.nan
    df["verify_memory_score"] = np.nan
    df["quality_score"] = np.nan
    df["composite_score"] = np.nan
    df["composite_rank"] = np.nan

    group_columns = ["workload_index", "agent_count", "subset_index"]
    for _, group_df in df.groupby(group_columns, sort=False):
        valid_index = group_df.index[group_df["success"].fillna(False)]
        if len(valid_index) == 0:
            continue
        valid_df = df.loc[valid_index]

        proof_size_score = normalize_series(valid_df["proof_bundle_bytes"], higher_is_better=False)
        verify_time_score = normalize_series(valid_df["verify_total_elapsed_sec"], higher_is_better=False)
        verify_memory_score = normalize_series(valid_df["verify_peak_rss_mb"], higher_is_better=False)
        quality_score = normalize_series(valid_df["mean_source_score_pct"], higher_is_better=True)

        composite_score = (
            size_weight * proof_size_score.fillna(0.0)
            + verify_time_weight * verify_time_score.fillna(0.0)
            + verify_memory_weight * verify_memory_score.fillna(0.0)
            + quality_weight * quality_score.fillna(0.0)
        )

        df.loc[valid_index, "proof_size_score"] = proof_size_score
        df.loc[valid_index, "verify_time_score"] = verify_time_score
        df.loc[valid_index, "verify_memory_score"] = verify_memory_score
        df.loc[valid_index, "quality_score"] = quality_score
        df.loc[valid_index, "composite_score"] = composite_score

        df.loc[valid_index, "proof_size_rank"] = valid_df["proof_bundle_bytes"].rank(method="average", ascending=True)
        df.loc[valid_index, "verify_time_rank"] = valid_df["verify_total_elapsed_sec"].rank(method="average", ascending=True)
        df.loc[valid_index, "verify_memory_rank"] = valid_df["verify_peak_rss_mb"].rank(method="average", ascending=True)
        df.loc[valid_index, "quality_rank"] = valid_df["mean_source_score_pct"].rank(method="average", ascending=False)
        df.loc[valid_index, "composite_rank"] = composite_score.rank(method="average", ascending=False)

    return df


def _summarize_by_method_agent(scored_results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        scored_results_df.groupby(["method", "agent_count"], sort=False)
        .agg(
            run_count=("method", "size"),
            success_count=("success", lambda series: int(pd.Series(series).fillna(False).sum())),
            mean_method_elapsed_sec=("method_elapsed_sec", "mean"),
            mean_proof_bundle_bytes=("proof_bundle_bytes", "mean"),
            mean_proof_payload_bytes=("proof_payload_bytes_total", "mean"),
            mean_verify_total_elapsed_sec=("verify_total_elapsed_sec", "mean"),
            mean_verify_peak_rss_mb=("verify_peak_rss_mb", "mean"),
            mean_source_score_pct=("mean_source_score_pct", "mean"),
            mean_proof_size_rank=("proof_size_rank", "mean"),
            mean_verify_time_rank=("verify_time_rank", "mean"),
            mean_verify_memory_rank=("verify_memory_rank", "mean"),
            mean_quality_rank=("quality_rank", "mean"),
            mean_composite_rank=("composite_rank", "mean"),
            mean_composite_score=("composite_score", "mean"),
            mean_group_count=("proof_group_count", "mean"),
        )
        .reset_index()
    )
    summary_df["success_rate"] = summary_df["success_count"] / summary_df["run_count"]
    return summary_df.sort_values(["agent_count", "mean_composite_rank", "method"]).reset_index(drop=True)


def _fit_scaling_exponent(summary_df: pd.DataFrame, value_column: str) -> Dict[str, float]:
    exponents: Dict[str, float] = {}
    for method_name, method_df in summary_df.groupby("method", sort=False):
        valid_df = method_df[["agent_count", value_column]].dropna()
        valid_df = valid_df[valid_df[value_column] > 0]
        if valid_df.shape[0] < 2:
            exponents[method_name] = float("nan")
            continue
        x = np.log(valid_df["agent_count"].astype(float).to_numpy())
        y = np.log(valid_df[value_column].astype(float).to_numpy())
        slope, _ = np.polyfit(x, y, deg=1)
        exponents[method_name] = float(slope)
    return exponents


def _summarize_overall(scored_results_df: pd.DataFrame, method_agent_summary_df: pd.DataFrame) -> pd.DataFrame:
    overall_df = (
        scored_results_df.groupby("method", sort=False)
        .agg(
            run_count=("method", "size"),
            success_count=("success", lambda series: int(pd.Series(series).fillna(False).sum())),
            overall_mean_method_elapsed_sec=("method_elapsed_sec", "mean"),
            overall_mean_proof_bundle_bytes=("proof_bundle_bytes", "mean"),
            overall_mean_proof_payload_bytes=("proof_payload_bytes_total", "mean"),
            overall_mean_verify_total_elapsed_sec=("verify_total_elapsed_sec", "mean"),
            overall_mean_verify_peak_rss_mb=("verify_peak_rss_mb", "mean"),
            overall_mean_source_score_pct=("mean_source_score_pct", "mean"),
            overall_mean_proof_size_rank=("proof_size_rank", "mean"),
            overall_mean_verify_time_rank=("verify_time_rank", "mean"),
            overall_mean_verify_memory_rank=("verify_memory_rank", "mean"),
            overall_mean_quality_rank=("quality_rank", "mean"),
            overall_mean_composite_rank=("composite_rank", "mean"),
            overall_mean_composite_score=("composite_score", "mean"),
        )
        .reset_index()
    )
    overall_df["success_rate"] = overall_df["success_count"] / overall_df["run_count"]
    overall_df["verify_time_scaling_exponent"] = overall_df["method"].map(
        _fit_scaling_exponent(method_agent_summary_df, "mean_verify_total_elapsed_sec")
    )
    overall_df["proof_size_scaling_exponent"] = overall_df["method"].map(
        _fit_scaling_exponent(method_agent_summary_df, "mean_proof_bundle_bytes")
    )
    overall_df["verify_memory_scaling_exponent"] = overall_df["method"].map(
        _fit_scaling_exponent(method_agent_summary_df, "mean_verify_peak_rss_mb")
    )
    return overall_df.sort_values(["overall_mean_composite_rank", "overall_mean_proof_bundle_bytes", "method"]).reset_index(drop=True)


def _evaluate_hybrid_requirement(overall_summary_df: pd.DataFrame, *, top_k: int) -> Dict[str, Any]:
    if top_k <= 0:
        return {
            "enabled": False,
            "passed": True,
            "message": "Hybrid ranking constraint disabled.",
        }
    hybrid_row = overall_summary_df[overall_summary_df["method"] == HYBRID_METHOD_NAME]
    if hybrid_row.empty:
        return {
            "enabled": True,
            "passed": False,
            "message": f"Hybrid method {HYBRID_METHOD_NAME} was not found in overall summary.",
        }

    hybrid_rank = float(hybrid_row.iloc[0]["overall_mean_composite_rank"])
    passed = hybrid_rank <= float(top_k)
    message = (
        f"Hybrid average composite rank={hybrid_rank:.3f}, required <= {top_k}."
        if passed
        else f"Hybrid average composite rank={hybrid_rank:.3f}, exceeds required <= {top_k}."
    )
    return {
        "enabled": True,
        "passed": passed,
        "hybrid_average_composite_rank": hybrid_rank,
        "required_top_k": int(top_k),
        "message": message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark proof scalability for five truth-discovery methods."
    )
    parser.add_argument(
        "--input-json",
        default=DEFAULT_INPUT_JSON_PATH,
        help="Full path to the input JSON file.",
    )
    parser.add_argument(
        "--start-batch",
        type=_positive_int,
        default=1,
        help="1-based batch index to start from.",
    )
    parser.add_argument(
        "--batch-group-size",
        type=_positive_int,
        default=1,
        help="Number of consecutive batches aggregated into one workload.",
    )
    parser.add_argument(
        "--num-workloads",
        type=_non_negative_int,
        default=3,
        help="How many workloads to evaluate. Set 0 to auto-use all full workloads.",
    )
    parser.add_argument(
        "--agent-counts",
        default="4,6,8,10",
        help="Comma-separated agent counts or 'auto'.",
    )
    parser.add_argument(
        "--subset-samples",
        type=_positive_int,
        default=2,
        help="How many agent subsets to sample for each agent count when strategy=random.",
    )
    parser.add_argument(
        "--subset-strategy",
        choices=["random", "prefix"],
        default="random",
        help="Subset selection strategy across agents.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260316,
        help="Base random seed for subset sampling.",
    )
    parser.add_argument(
        "--size-weight",
        type=_non_negative_float,
        default=0.35,
        help="Composite weight for proof bundle size.",
    )
    parser.add_argument(
        "--verify-time-weight",
        type=_non_negative_float,
        default=0.25,
        help="Composite weight for proof verification time.",
    )
    parser.add_argument(
        "--verify-memory-weight",
        type=_non_negative_float,
        default=0.15,
        help="Composite weight for proof verification peak RSS.",
    )
    parser.add_argument(
        "--quality-weight",
        type=_non_negative_float,
        default=0.25,
        help="Composite weight for mixed-modality quality proxy (mean source score).",
    )
    parser.add_argument(
        "--hybrid-top-k",
        type=_non_negative_int,
        default=1,
        help="Require Hybrid average composite rank to be <= this value. Set 0 to disable.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=_non_negative_float,
        default=0.02,
        help="RSS polling interval while verifying proof groups.",
    )
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use SBERT for SenteTruthFinder; default uses TF-IDF fallback.",
    )
    parser.add_argument(
        "--sbert-model",
        default=os.environ.get("SENTE_SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        help="Optional SBERT model name.",
    )
    parser.add_argument(
        "--strict-sbert",
        action="store_true",
        help="Fail instead of falling back when SBERT cannot be loaded.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load SBERT only from local cache.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment identifier.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_json_path = os.path.abspath(args.input_json)
    all_batches = load_batches_from_path(input_json_path)
    total_batches = len(all_batches)
    num_workloads = _resolve_num_workloads(args.num_workloads, total_batches, args.start_batch, args.batch_group_size)
    workload_specs = _build_workload_specs(
        total_batches,
        start_batch=args.start_batch,
        batch_group_size=args.batch_group_size,
        num_workloads=num_workloads,
    )

    first_workload_items = _collect_workload_items(all_batches, workload_specs[0])
    max_agents = len(_extract_agents(first_workload_items))
    agent_counts = _parse_agent_counts(args.agent_counts, max_agents)
    methods = METHOD_NAMES if args.methods.strip().lower() == "all" else [token.strip() for token in args.methods.split(",") if token.strip()]
    unsupported_methods = [method for method in methods if method not in METHOD_NAMES]
    if unsupported_methods:
        raise ValueError(f"Unsupported methods: {unsupported_methods}")

    size_weight, verify_time_weight, verify_memory_weight, quality_weight = _normalize_weights(
        args.size_weight,
        args.verify_time_weight,
        args.verify_memory_weight,
        args.quality_weight,
    )

    experiment_id = args.experiment_id or default_experiment_id("proof_scalability", input_json_path)
    output_dir = ensure_directory(os.path.join(args.output_root, experiment_id))
    artifacts_root = ensure_directory(os.path.join(output_dir, "proof_artifacts"))

    print("=" * 108)
    print("Proof scalability benchmark for five truth-discovery methods")
    print("=" * 108)
    print(f"Input JSON: {input_json_path}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output dir: {output_dir}")
    print(f"Workloads: {len(workload_specs)} | batch_group_size={args.batch_group_size} | start_batch={args.start_batch}")
    print(f"Agent counts: {agent_counts}")
    print(f"Methods: {methods}")
    print(
        f"Composite weights: size={size_weight:.2f}, verify_time={verify_time_weight:.2f}, "
        f"verify_memory={verify_memory_weight:.2f}, quality={quality_weight:.2f}"
    )

    rows: List[Dict[str, Any]] = []
    for workload in workload_specs:
        workload_items = _collect_workload_items(all_batches, workload)
        agents = _extract_agents(workload_items)
        for agent_count in agent_counts:
            subsets = _select_agent_subsets(
                agents,
                agent_count=agent_count,
                subset_samples=args.subset_samples,
                base_seed=args.seed,
                workload_index=workload.workload_index,
                strategy=args.subset_strategy,
            )
            total_subsets = len(subsets)
            for subset_offset, selected_agents in enumerate(subsets, start=1):
                filtered_items = _filter_reference_items_by_agents(workload_items, selected_agents)
                text_df = create_reference_data(filtered_items)
                websites = text_df["website"].astype(str).drop_duplicates().tolist() if not text_df.empty else list(selected_agents)
                run_context = RunContext(
                    workload_index=workload.workload_index,
                    agent_count=agent_count,
                    subset_index=subset_offset,
                    total_subsets=total_subsets,
                    selected_agents=list(selected_agents),
                    batch_start=workload.batch_start,
                    batch_end=workload.batch_end,
                    item_count=int(text_df.shape[0]),
                    website_count=int(text_df["website"].nunique()) if "website" in text_df.columns else len(selected_agents),
                    object_count=int(text_df["object"].nunique()) if "object" in text_df.columns else 0,
                )

                for method_name in methods:
                    row: Dict[str, Any] = {
                        **asdict(run_context),
                        "method": method_name,
                        "success": False,
                    }
                    try:
                        method_result = _run_method_by_name(
                            method_name,
                            text_df,
                            use_sbert=args.use_sbert,
                            sbert_model=args.sbert_model,
                            strict_sbert=args.strict_sbert,
                            local_files_only=args.local_files_only,
                        )
                        run_output_dir = ensure_directory(
                            os.path.join(
                                artifacts_root,
                                f"workload_{workload.workload_index:03d}",
                                f"agents_{agent_count:02d}",
                                f"subset_{subset_offset:02d}",
                            )
                        )
                        bundle_artifacts = build_saved_bundle_artifacts(
                            method_result,
                            experiment_id=(
                                f"{experiment_id}_w{workload.workload_index:03d}_"
                                f"a{agent_count:02d}_s{subset_offset:02d}"
                            ),
                            output_dir=run_output_dir,
                        )
                        verify_metric = verify_bundle_payload(
                            bundle_artifacts.bundle_payload,
                            poll_interval_sec=args.poll_interval_sec,
                        )
                        mean_score_pct, scored_website_count = mean_source_score_pct(method_result, websites)
                        row.update(
                            {
                                "success": True,
                                "method_elapsed_sec": float(method_result.elapsed_sec),
                                "proof_bundle_bytes": int(os.path.getsize(bundle_artifacts.bundle_path)),
                                "proof_payload_bytes_total": bundle_group_payload_bytes(bundle_artifacts.bundle_payload),
                                "proof_group_count": int(len(bundle_artifacts.bundle_payload.get("proof_groups", []))),
                                "verify_total_elapsed_sec": float(verify_metric.total_elapsed_sec),
                                "verify_peak_rss_mb": float(verify_metric.max_peak_rss_mb),
                                "verified_group_count": int(verify_metric.verified_group_count),
                                "success_group_count": int(verify_metric.success_group_count),
                                "all_groups_verified": bool(verify_metric.all_verified),
                                "mean_source_score_pct": mean_score_pct,
                                "scored_website_count": int(scored_website_count),
                                "bundle_path": bundle_artifacts.bundle_path,
                                "manifest_path": bundle_artifacts.manifest_path,
                                "chainmaker_manifest_path": bundle_artifacts.chainmaker_manifest_path,
                                "group_verify_metrics": [asdict(metric) for metric in verify_metric.group_metrics],
                            }
                        )
                    except Exception as exc:
                        row.update(
                            {
                                "success": False,
                                "error": str(exc),
                                "method_elapsed_sec": float("nan"),
                                "proof_bundle_bytes": float("nan"),
                                "proof_payload_bytes_total": float("nan"),
                                "proof_group_count": float("nan"),
                                "verify_total_elapsed_sec": float("nan"),
                                "verify_peak_rss_mb": float("nan"),
                                "verified_group_count": 0,
                                "success_group_count": 0,
                                "all_groups_verified": False,
                                "mean_source_score_pct": float("nan"),
                                "scored_website_count": 0,
                                "group_verify_metrics": [],
                            }
                        )
                    rows.append(row)
                    print(
                        f"[workload {workload.workload_index}] agents={agent_count} subset={subset_offset}/{total_subsets} "
                        f"method={method_name} status={'OK' if row['success'] else 'FAIL'} "
                        f"bundle={format_float(row.get('proof_bundle_bytes'), 2)}B "
                        f"verify={format_float(row.get('verify_total_elapsed_sec'))}s "
                        f"rss={format_float(row.get('verify_peak_rss_mb'))}MB"
                    )

    raw_results_df = pd.DataFrame(rows)
    scored_results_df = _compute_scores(
        raw_results_df,
        size_weight=size_weight,
        verify_time_weight=verify_time_weight,
        verify_memory_weight=verify_memory_weight,
        quality_weight=quality_weight,
    )
    method_agent_summary_df = _summarize_by_method_agent(scored_results_df)
    overall_summary_df = _summarize_overall(scored_results_df, method_agent_summary_df)
    hybrid_requirement = _evaluate_hybrid_requirement(overall_summary_df, top_k=args.hybrid_top_k)

    raw_results_path = os.path.join(output_dir, "proof_scalability_runs.csv")
    scored_results_path = os.path.join(output_dir, "proof_scalability_scored_runs.csv")
    method_agent_summary_path = os.path.join(output_dir, "proof_scalability_summary_by_method_agent.csv")
    overall_summary_path = os.path.join(output_dir, "proof_scalability_overall_summary.csv")
    raw_results_df.drop(columns=["group_verify_metrics"], errors="ignore").to_csv(raw_results_path, index=False)
    scored_results_df.drop(columns=["group_verify_metrics"], errors="ignore").to_csv(scored_results_path, index=False)
    method_agent_summary_df.to_csv(method_agent_summary_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "input_json": input_json_path,
        "output_dir": output_dir,
        "workloads": [asdict(spec) for spec in workload_specs],
        "agent_counts": agent_counts,
        "methods": methods,
        "weights": {
            "proof_size": size_weight,
            "verify_time": verify_time_weight,
            "verify_memory": verify_memory_weight,
            "quality": quality_weight,
        },
        "hybrid_requirement": hybrid_requirement,
        "paths": {
            "raw_results_csv": raw_results_path,
            "scored_results_csv": scored_results_path,
            "method_agent_summary_csv": method_agent_summary_path,
            "overall_summary_csv": overall_summary_path,
        },
    }
    write_json(os.path.join(output_dir, "proof_scalability_metadata.json"), metadata)

    print("\nOverall summary:")
    display_columns = [
        "method",
        "overall_mean_composite_rank",
        "overall_mean_composite_score",
        "overall_mean_proof_bundle_bytes",
        "overall_mean_verify_total_elapsed_sec",
        "overall_mean_verify_peak_rss_mb",
        "overall_mean_source_score_pct",
        "proof_size_scaling_exponent",
        "verify_time_scaling_exponent",
        "verify_memory_scaling_exponent",
    ]
    print(overall_summary_df[display_columns].round(4).to_string(index=False))
    print(f"\nHybrid requirement: {hybrid_requirement['message']}")
    print(f"raw runs: {raw_results_path}")
    print(f"scored runs: {scored_results_path}")
    print(f"summary by method+agent: {method_agent_summary_path}")
    print(f"overall summary: {overall_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
