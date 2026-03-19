from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.examples.five_methods_comparison import (
    _create_enhanced_finder,
    _native_score_to_percentage,
    create_reference_data,
    create_reference_input,
    load_demo_batches,
    run_basic,
    run_enhanced,
    run_senfeed,
    run_sente,
)


DEFAULT_INPUT_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_agent-10", "demo.json")
)
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
HYBRID_METHOD_NAME = "EnhancedTruthFinder-Hybrid"
METHOD_NAMES = [
    HYBRID_METHOD_NAME,
    "EnhancedTruthFinder-NonHybrid",
    "SenteTruthFinder",
    "BasicTruthFinder",
    "SenFeedTruthFinder-CompositeTD",
]


@dataclass(frozen=True)
class WorkloadSpec:
    workload_index: int
    batch_start: int
    batch_end: int
    batch_group_size: int


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


def _bounded_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"Expected float in [0, 1], got {value!r}")
    return parsed


def _sanitize_path_stem(file_path: str) -> str:
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", file_stem).strip("._-")
    return sanitized or "input"


def _agent_sort_key(agent_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(agent_name))
    if match:
        return int(match.group(1)), str(agent_name)
    return math.inf, str(agent_name)


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _normalize_weights(time_weight: float, memory_weight: float, quality_weight: float) -> tuple[float, float, float]:
    total = float(time_weight) + float(memory_weight) + float(quality_weight)
    if total <= 0.0:
        raise ValueError("At least one ranking weight must be > 0.")
    return time_weight / total, memory_weight / total, quality_weight / total


def _parse_agent_counts(raw_value: str, max_agents: int) -> List[int]:
    if max_agents < 1:
        raise ValueError("max_agents must be >= 1.")

    if raw_value.strip().lower() == "auto":
        if max_agents <= 5:
            counts = list(range(2, max_agents + 1)) if max_agents >= 2 else [1]
        else:
            candidates = [2, 4, 6, 8, 10]
            counts = [count for count in candidates if count <= max_agents]
            if not counts:
                counts = [max_agents]
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


def _build_workload_specs(
    total_batches: int,
    *,
    start_batch: int,
    batch_group_size: int,
    num_workloads: int,
) -> List[WorkloadSpec]:
    if start_batch < 1:
        raise ValueError(f"start_batch must be >= 1, got: {start_batch}")
    if batch_group_size < 1:
        raise ValueError(f"batch_group_size must be >= 1, got: {batch_group_size}")
    if num_workloads < 1:
        raise ValueError(f"num_workloads must be >= 1, got: {num_workloads}")
    if total_batches < 1:
        raise ValueError(f"total_batches must be >= 1, got: {total_batches}")

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


def _collect_workload_items(all_batches: List[Dict[str, Any]], workload: WorkloadSpec) -> List[Dict[str, Any]]:
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
    selected_agent_set = {str(agent) for agent in selected_agents}
    return [item for item in reference_items if str(item["agent"]).strip() in selected_agent_set]


def _current_rss_mb(pid: int | None = None) -> float:
    if pid is None or int(pid) == os.getpid():
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return max(float(rss) / (1024.0 * 1024.0), 0.0)
            return max(float(rss) / 1024.0, 0.0)
        except Exception:
            pass

    target_pid = os.getpid() if pid is None else int(pid)
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(target_pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not output:
            return float("nan")
        return max(float(output) / 1024.0, 0.0)
    except Exception:
        return float("nan")


class _PeakRssSampler:
    def __init__(self, interval_sec: float):
        self.interval_sec = max(float(interval_sec), 0.01)
        self.baseline_rss_mb = float("nan")
        self.peak_rss_mb = float("nan")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.baseline_rss_mb = _current_rss_mb()
        self.peak_rss_mb = self.baseline_rss_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            current_rss_mb = _current_rss_mb()
            if not pd.isna(current_rss_mb):
                if pd.isna(self.peak_rss_mb):
                    self.peak_rss_mb = current_rss_mb
                else:
                    self.peak_rss_mb = max(self.peak_rss_mb, current_rss_mb)
            time.sleep(self.interval_sec)

    def stop(self) -> tuple[float, float, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2.0, 0.1))
        final_rss_mb = _current_rss_mb()
        if not pd.isna(final_rss_mb):
            if pd.isna(self.peak_rss_mb):
                self.peak_rss_mb = final_rss_mb
            else:
                self.peak_rss_mb = max(self.peak_rss_mb, final_rss_mb)
        peak_delta_rss_mb = float("nan")
        if not pd.isna(self.baseline_rss_mb) and not pd.isna(self.peak_rss_mb):
            peak_delta_rss_mb = max(float(self.peak_rss_mb) - float(self.baseline_rss_mb), 0.0)
        return self.baseline_rss_mb, self.peak_rss_mb, peak_delta_rss_mb


def _run_method_by_name(
    method_name: str,
    text_df: pd.DataFrame,
    *,
    use_sbert: bool,
    sbert_model: str | None,
    strict_sbert: bool,
    local_files_only: bool,
):
    if method_name == HYBRID_METHOD_NAME:
        return run_enhanced(
            text_df,
            finder=_create_enhanced_finder(enable_hybrid_mode=True),
            label=HYBRID_METHOD_NAME,
        )
    if method_name == "EnhancedTruthFinder-NonHybrid":
        return run_enhanced(
            text_df,
            finder=_create_enhanced_finder(enable_hybrid_mode=False),
            label="EnhancedTruthFinder-NonHybrid",
        )
    if method_name == "SenteTruthFinder":
        return run_sente(
            text_df,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            strict_sbert=strict_sbert,
            local_files_only=local_files_only,
        )
    if method_name == "BasicTruthFinder":
        return run_basic(text_df)
    if method_name == "SenFeedTruthFinder-CompositeTD":
        return run_senfeed(text_df)
    raise ValueError(f"Unsupported method: {method_name}")


def _mean_percentage_score(method_result, websites: Sequence[str]) -> tuple[float, int]:
    if not method_result.source_scores:
        return float("nan"), 0

    raw_scores = pd.Series(method_result.source_scores, dtype=float)
    raw_scores = raw_scores.reindex(list(websites))
    percentage_scores = _native_score_to_percentage(raw_scores, method_result.name)
    valid_scores = percentage_scores.dropna()
    if valid_scores.empty:
        return float("nan"), 0
    return float(valid_scores.mean()), int(valid_scores.shape[0])


def _normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_series = numeric_series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid_series.empty:
        return result
    minimum = float(valid_series.min())
    maximum = float(valid_series.max())
    if math.isclose(minimum, maximum, rel_tol=1e-12, abs_tol=1e-12):
        result.loc[valid_series.index] = 100.0
        return result
    if higher_is_better:
        result.loc[valid_series.index] = 100.0 * (valid_series - minimum) / (maximum - minimum)
    else:
        result.loc[valid_series.index] = 100.0 * (maximum - valid_series) / (maximum - minimum)
    return result


def _compute_group_scores(
    raw_results_df: pd.DataFrame,
    *,
    time_weight: float,
    memory_weight: float,
    quality_weight: float,
) -> pd.DataFrame:
    df = raw_results_df.copy()
    df["latency_rank"] = np.nan
    df["memory_rank"] = np.nan
    df["quality_rank"] = np.nan
    df["latency_score"] = np.nan
    df["memory_score"] = np.nan
    df["quality_score"] = np.nan
    df["composite_score"] = np.nan
    df["composite_rank"] = np.nan

    group_columns = ["workload_index", "agent_count", "subset_index"]
    for _, group_df in df.groupby(group_columns, sort=False):
        valid_index = group_df.index[group_df["success"].fillna(False)]
        if len(valid_index) == 0:
            continue
        valid_df = df.loc[valid_index]

        latency_score = _normalize_series(valid_df["elapsed_sec"], higher_is_better=False)
        memory_score = _normalize_series(valid_df["peak_delta_rss_mb"], higher_is_better=False)
        quality_score = _normalize_series(valid_df["mean_source_score_pct"], higher_is_better=True)

        composite_score = (
            time_weight * latency_score.fillna(0.0)
            + memory_weight * memory_score.fillna(0.0)
            + quality_weight * quality_score.fillna(0.0)
        )

        df.loc[valid_index, "latency_score"] = latency_score
        df.loc[valid_index, "memory_score"] = memory_score
        df.loc[valid_index, "quality_score"] = quality_score
        df.loc[valid_index, "composite_score"] = composite_score

        df.loc[valid_index, "latency_rank"] = valid_df["elapsed_sec"].rank(method="average", ascending=True)
        df.loc[valid_index, "memory_rank"] = valid_df["peak_delta_rss_mb"].rank(method="average", ascending=True)
        df.loc[valid_index, "quality_rank"] = valid_df["mean_source_score_pct"].rank(method="average", ascending=False)
        df.loc[valid_index, "composite_rank"] = composite_score.rank(method="average", ascending=False)

    return df


def _fit_scaling_exponent(summary_df: pd.DataFrame, value_column: str) -> Dict[str, float]:
    exponents: Dict[str, float] = {}
    for method_name, method_df in summary_df.groupby("method", sort=False):
        valid_df = method_df[["agent_count", value_column]].dropna()
        valid_df = valid_df[valid_df[value_column] > 0]
        if valid_df.shape[0] < 2:
            exponents[method_name] = float("nan")
            continue
        slope = np.polyfit(np.log(valid_df["agent_count"].astype(float)), np.log(valid_df[value_column].astype(float)), 1)[0]
        exponents[method_name] = float(slope)
    return exponents


def _summarize_by_method_agent(scored_results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        scored_results_df.groupby(["method", "agent_count"], sort=False)
        .agg(
            run_count=("method", "size"),
            success_count=("success", lambda series: int(pd.Series(series).fillna(False).sum())),
            mean_elapsed_sec=("elapsed_sec", "mean"),
            std_elapsed_sec=("elapsed_sec", "std"),
            mean_peak_rss_mb=("peak_rss_mb", "mean"),
            mean_peak_delta_rss_mb=("peak_delta_rss_mb", "mean"),
            std_peak_delta_rss_mb=("peak_delta_rss_mb", "std"),
            mean_source_score_pct=("mean_source_score_pct", "mean"),
            std_source_score_pct=("mean_source_score_pct", "std"),
            mean_latency_rank=("latency_rank", "mean"),
            mean_memory_rank=("memory_rank", "mean"),
            mean_quality_rank=("quality_rank", "mean"),
            mean_composite_rank=("composite_rank", "mean"),
            mean_composite_score=("composite_score", "mean"),
        )
        .reset_index()
    )

    summary_df["success_rate"] = summary_df["success_count"] / summary_df["run_count"]
    baseline_agent_count = int(summary_df["agent_count"].min())
    baseline_df = summary_df[summary_df["agent_count"] == baseline_agent_count].set_index("method")
    summary_df["runtime_scale_vs_min_agents"] = summary_df.apply(
        lambda row: row["mean_elapsed_sec"] / baseline_df.loc[row["method"], "mean_elapsed_sec"]
        if row["method"] in baseline_df.index and baseline_df.loc[row["method"], "mean_elapsed_sec"] > 0
        else float("nan"),
        axis=1,
    )
    summary_df["memory_scale_vs_min_agents"] = summary_df.apply(
        lambda row: row["mean_peak_delta_rss_mb"] / baseline_df.loc[row["method"], "mean_peak_delta_rss_mb"]
        if row["method"] in baseline_df.index and baseline_df.loc[row["method"], "mean_peak_delta_rss_mb"] > 0
        else float("nan"),
        axis=1,
    )

    runtime_exponents = _fit_scaling_exponent(summary_df, "mean_elapsed_sec")
    memory_exponents = _fit_scaling_exponent(summary_df, "mean_peak_delta_rss_mb")
    summary_df["runtime_scaling_exponent"] = summary_df["method"].map(runtime_exponents)
    summary_df["memory_scaling_exponent"] = summary_df["method"].map(memory_exponents)
    return summary_df


def _summarize_overall(scored_results_df: pd.DataFrame, method_agent_summary_df: pd.DataFrame) -> pd.DataFrame:
    overall_df = (
        scored_results_df.groupby("method", sort=False)
        .agg(
            run_count=("method", "size"),
            success_count=("success", lambda series: int(pd.Series(series).fillna(False).sum())),
            overall_mean_elapsed_sec=("elapsed_sec", "mean"),
            overall_mean_peak_delta_rss_mb=("peak_delta_rss_mb", "mean"),
            overall_mean_source_score_pct=("mean_source_score_pct", "mean"),
            overall_mean_latency_rank=("latency_rank", "mean"),
            overall_mean_memory_rank=("memory_rank", "mean"),
            overall_mean_quality_rank=("quality_rank", "mean"),
            overall_mean_composite_rank=("composite_rank", "mean"),
            overall_mean_composite_score=("composite_score", "mean"),
        )
        .reset_index()
    )
    overall_df["success_rate"] = overall_df["success_count"] / overall_df["run_count"]

    method_runtime_exponents = (
        method_agent_summary_df[["method", "runtime_scaling_exponent"]]
        .drop_duplicates(subset=["method"])
        .set_index("method")["runtime_scaling_exponent"]
        .to_dict()
    )
    method_memory_exponents = (
        method_agent_summary_df[["method", "memory_scaling_exponent"]]
        .drop_duplicates(subset=["method"])
        .set_index("method")["memory_scaling_exponent"]
        .to_dict()
    )
    overall_df["runtime_scaling_exponent"] = overall_df["method"].map(method_runtime_exponents)
    overall_df["memory_scaling_exponent"] = overall_df["method"].map(method_memory_exponents)
    return overall_df.sort_values(["overall_mean_composite_rank", "overall_mean_elapsed_sec", "method"]).reset_index(drop=True)


def _evaluate_hybrid_requirement(
    overall_summary_df: pd.DataFrame,
    *,
    hybrid_method_name: str,
    top_k: int,
) -> Dict[str, Any]:
    if top_k <= 0:
        return {
            "enabled": False,
            "passed": True,
            "message": "Hybrid ranking constraint disabled.",
        }

    hybrid_row = overall_summary_df[overall_summary_df["method"] == hybrid_method_name]
    if hybrid_row.empty:
        return {
            "enabled": True,
            "passed": False,
            "message": f"Hybrid method {hybrid_method_name} was not found in overall summary.",
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
        "hybrid_method_name": hybrid_method_name,
        "hybrid_average_composite_rank": hybrid_rank,
        "required_top_k": int(top_k),
        "message": message,
    }


def _write_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _invoke_worker(
    *,
    script_path: str,
    input_json_path: str,
    workload: WorkloadSpec,
    method_name: str,
    selected_agents: Sequence[str],
    subset_index: int,
    memory_sample_interval: float,
    use_sbert: bool,
    sbert_model: str | None,
    strict_sbert: bool,
    local_files_only: bool,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        script_path,
        "--worker-mode",
        "--input-json",
        input_json_path,
        "--worker-method",
        method_name,
        "--worker-batch-start",
        str(workload.batch_start),
        "--worker-batch-group-size",
        str(workload.batch_group_size),
        "--worker-selected-agents",
        ",".join(selected_agents),
        "--worker-subset-index",
        str(subset_index),
        "--memory-sample-interval",
        str(memory_sample_interval),
    ]
    if use_sbert:
        command.append("--use-sbert")
    if sbert_model:
        command.extend(["--sbert-model", sbert_model])
    if strict_sbert:
        command.append("--strict-sbert")
    if local_files_only:
        command.append("--local-files-only")

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and stdout_lines:
        try:
            return json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            pass

    return {
        "method": method_name,
        "success": False,
        "error": (
            f"Worker failed with code {completed.returncode}. "
            f"stdout_tail={stdout_lines[-1] if stdout_lines else ''!r}; stderr_tail={completed.stderr.strip()[-500:]!r}"
        ),
        "elapsed_sec": float("nan"),
        "wall_elapsed_sec": float("nan"),
        "baseline_rss_mb": float("nan"),
        "peak_rss_mb": float("nan"),
        "peak_delta_rss_mb": float("nan"),
        "mean_source_score_pct": float("nan"),
        "scored_website_count": 0,
    }


def _print_progress(
    *,
    workload_index: int,
    agent_count: int,
    subset_index: int,
    total_subsets: int,
    method_name: str,
    result_row: Dict[str, Any],
):
    status = "OK" if result_row.get("success") else "FAIL"
    print(
        f"[workload {workload_index}] agents={agent_count} subset={subset_index}/{total_subsets} "
        f"method={method_name} status={status} time={_format_float(result_row.get('elapsed_sec'))}s "
        f"memΔ={_format_float(result_row.get('peak_delta_rss_mb'))}MB"
    )


def _run_worker_mode(args: argparse.Namespace) -> int:
    all_batches = load_demo_batches(os.path.abspath(args.input_json))
    workload = WorkloadSpec(
        workload_index=1,
        batch_start=args.worker_batch_start,
        batch_end=args.worker_batch_start + args.worker_batch_group_size - 1,
        batch_group_size=args.worker_batch_group_size,
    )
    selected_agents = [token.strip() for token in args.worker_selected_agents.split(",") if token.strip()]
    reference_items = _collect_workload_items(all_batches, workload)
    filtered_items = _filter_reference_items_by_agents(reference_items, selected_agents)
    text_df = create_reference_data(filtered_items)
    websites = text_df["website"].astype(str).drop_duplicates().tolist() if not text_df.empty else selected_agents

    sampler = _PeakRssSampler(args.memory_sample_interval)
    wall_start = time.perf_counter()
    sampler.start()
    try:
        method_result = _run_method_by_name(
            args.worker_method,
            text_df,
            use_sbert=args.use_sbert,
            sbert_model=args.sbert_model,
            strict_sbert=args.strict_sbert,
            local_files_only=args.local_files_only,
        )
        success = True
        error_message = ""
        mean_source_score_pct, scored_website_count = _mean_percentage_score(method_result, websites)
        method_elapsed_sec = float(method_result.elapsed_sec)
    except Exception:
        success = False
        error_message = traceback.format_exc(limit=20)
        mean_source_score_pct = float("nan")
        scored_website_count = 0
        method_elapsed_sec = float("nan")
    baseline_rss_mb, peak_rss_mb, peak_delta_rss_mb = sampler.stop()
    wall_elapsed_sec = time.perf_counter() - wall_start

    payload = {
        "method": args.worker_method,
        "success": success,
        "error": error_message,
        "elapsed_sec": method_elapsed_sec,
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "baseline_rss_mb": float(baseline_rss_mb) if not pd.isna(baseline_rss_mb) else float("nan"),
        "peak_rss_mb": float(peak_rss_mb) if not pd.isna(peak_rss_mb) else float("nan"),
        "peak_delta_rss_mb": float(peak_delta_rss_mb) if not pd.isna(peak_delta_rss_mb) else float("nan"),
        "mean_source_score_pct": mean_source_score_pct,
        "scored_website_count": int(scored_website_count),
        "website_count": int(text_df["website"].nunique()) if "website" in text_df.columns else len(selected_agents),
        "item_count": int(text_df.shape[0]),
        "object_count": int(text_df["object"].nunique()) if "object" in text_df.columns else 0,
        "selected_agents": selected_agents,
        "subset_index": int(args.worker_subset_index),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if success else 1


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


def _run_benchmark_mode(args: argparse.Namespace) -> int:
    input_json_path = os.path.abspath(args.input_json)
    all_batches = load_demo_batches(input_json_path)
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

    time_weight, memory_weight, quality_weight = _normalize_weights(
        args.time_weight,
        args.memory_weight,
        args.quality_weight,
    )

    experiment_id = args.experiment_id or (
        f"agent_scalability_{_sanitize_path_stem(input_json_path)}_{int(time.time())}"
    )
    output_dir = os.path.abspath(os.path.join(args.output_root, experiment_id))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 108)
    print("Agent scalability benchmark for five truth discovery methods")
    print("=" * 108)
    print(f"Input JSON: {input_json_path}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output dir: {output_dir}")
    print(f"Workloads: {len(workload_specs)} | batch_group_size={args.batch_group_size} | start_batch={args.start_batch}")
    print(f"Agent counts: {agent_counts}")
    print(f"Methods: {methods}")
    print(
        f"Composite weights: time={time_weight:.2f}, memory={memory_weight:.2f}, quality={quality_weight:.2f}"
    )
    print(
        f"Hybrid rank constraint: top_k={args.hybrid_top_k} using average composite rank "
        f"(quality-adjusted scalability score)."
    )
    print(
        f"SenteTruthFinder config: use_sbert={args.use_sbert}, sbert_model={args.sbert_model}, "
        f"strict_sbert={args.strict_sbert}, local_files_only={args.local_files_only}"
    )

    raw_rows: List[Dict[str, Any]] = []
    script_path = os.path.abspath(__file__)

    for workload in workload_specs:
        workload_items = _collect_workload_items(all_batches, workload)
        workload_agents = _extract_agents(workload_items)
        for agent_count in agent_counts:
            if agent_count > len(workload_agents):
                continue
            subsets = _select_agent_subsets(
                workload_agents,
                agent_count=agent_count,
                subset_samples=args.subset_samples,
                base_seed=args.subset_seed,
                workload_index=workload.workload_index,
                strategy=args.selection_strategy,
            )
            for subset_offset, selected_agents in enumerate(subsets, start=1):
                filtered_items = _filter_reference_items_by_agents(workload_items, selected_agents)
                object_count = len({str(item["object"]).strip() for item in filtered_items})
                for method_name in methods:
                    result_row = _invoke_worker(
                        script_path=script_path,
                        input_json_path=input_json_path,
                        workload=workload,
                        method_name=method_name,
                        selected_agents=selected_agents,
                        subset_index=subset_offset,
                        memory_sample_interval=args.memory_sample_interval,
                        use_sbert=args.use_sbert,
                        sbert_model=args.sbert_model,
                        strict_sbert=args.strict_sbert,
                        local_files_only=args.local_files_only,
                    )
                    raw_row = {
                        "experiment_id": experiment_id,
                        "input_json": input_json_path,
                        "workload_index": workload.workload_index,
                        "batch_start": workload.batch_start,
                        "batch_end": workload.batch_end,
                        "batch_group_size": workload.batch_group_size,
                        "selection_strategy": args.selection_strategy,
                        "subset_seed": args.subset_seed,
                        "subset_index": subset_offset,
                        "total_subsets_for_agent_count": len(subsets),
                        "agent_count": agent_count,
                        "available_agent_count": len(workload_agents),
                        "selected_agents": "|".join(selected_agents),
                        "selected_agent_count": len(selected_agents),
                        "input_item_count": len(filtered_items),
                        "input_object_count": object_count,
                        **result_row,
                    }
                    raw_rows.append(raw_row)
                    _print_progress(
                        workload_index=workload.workload_index,
                        agent_count=agent_count,
                        subset_index=subset_offset,
                        total_subsets=len(subsets),
                        method_name=method_name,
                        result_row=result_row,
                    )

    raw_results_df = pd.DataFrame(raw_rows)
    scored_results_df = _compute_group_scores(
        raw_results_df,
        time_weight=time_weight,
        memory_weight=memory_weight,
        quality_weight=quality_weight,
    )
    method_agent_summary_df = _summarize_by_method_agent(scored_results_df)
    overall_summary_df = _summarize_overall(scored_results_df, method_agent_summary_df)
    hybrid_check = _evaluate_hybrid_requirement(
        overall_summary_df,
        hybrid_method_name=HYBRID_METHOD_NAME,
        top_k=args.hybrid_top_k,
    )

    raw_results_path = os.path.join(output_dir, "raw_runs.csv")
    scored_results_path = os.path.join(output_dir, "scored_runs.csv")
    method_agent_summary_path = os.path.join(output_dir, "summary_by_method_agent_count.csv")
    overall_summary_path = os.path.join(output_dir, "数值-随着agent变化.csv")
    benchmark_summary_path = os.path.join(output_dir, "benchmark_summary.json")

    raw_results_df.to_csv(raw_results_path, index=False)
    scored_results_df.to_csv(scored_results_path, index=False)
    method_agent_summary_df.to_csv(method_agent_summary_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)
    _write_json(
        benchmark_summary_path,
        {
            "experiment_id": experiment_id,
            "config": {
                "input_json": input_json_path,
                "start_batch": args.start_batch,
                "batch_group_size": args.batch_group_size,
                "num_workloads": len(workload_specs),
                "agent_counts": agent_counts,
                "subset_samples": args.subset_samples,
                "subset_seed": args.subset_seed,
                "selection_strategy": args.selection_strategy,
                "methods": methods,
                "weights": {
                    "time": time_weight,
                    "memory": memory_weight,
                    "quality": quality_weight,
                },
                "hybrid_top_k": args.hybrid_top_k,
                "use_sbert": args.use_sbert,
                "sbert_model": args.sbert_model,
                "strict_sbert": args.strict_sbert,
                "local_files_only": args.local_files_only,
            },
            "workloads": [asdict(workload) for workload in workload_specs],
            "hybrid_check": hybrid_check,
            "artifacts": {
                "raw_runs_csv": raw_results_path,
                "scored_runs_csv": scored_results_path,
                "summary_by_method_agent_count_csv": method_agent_summary_path,
                "summary_overall_csv": overall_summary_path,
            },
        },
    )

    print("\nOverall summary:")
    summary_columns = [
        "method",
        "overall_mean_composite_rank",
        "overall_mean_composite_score",
        "overall_mean_elapsed_sec",
        "overall_mean_peak_delta_rss_mb",
        "overall_mean_source_score_pct",
        "runtime_scaling_exponent",
        "memory_scaling_exponent",
    ]
    print(overall_summary_df[summary_columns].round(4).to_string(index=False))
    print(f"\nHybrid check: {hybrid_check['message']}")
    print(f"raw runs: {raw_results_path}")
    print(f"scored runs: {scored_results_path}")
    print(f"method-agent summary: {method_agent_summary_path}")
    print(f"overall summary: {overall_summary_path}")
    print(f"benchmark summary: {benchmark_summary_path}")

    return 0 if hybrid_check.get("passed", False) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark five truth discovery methods across different agent counts."
    )
    parser.add_argument(
        "--input-json",
        default=DEFAULT_INPUT_JSON_PATH,
        help="Input JSON path. Default uses decentralized_truth_discovery/data_agent-10/demo.json.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store benchmark outputs.",
    )
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Optional experiment ID. Defaults to agent_scalability_<input>_<timestamp>.",
    )
    parser.add_argument(
        "--start-batch",
        type=_positive_int,
        default=1,
        help="Start batch number, 1-based.",
    )
    parser.add_argument(
        "--batch-group-size",
        type=_positive_int,
        default=10,
        help="Number of consecutive batches merged into one workload.",
    )
    parser.add_argument(
        "--num-workloads",
        type=_non_negative_int,
        default=5,
        help="Number of workloads to benchmark. Use 0 to consume all full workloads from start_batch.",
    )
    parser.add_argument(
        "--agent-counts",
        default="auto",
        help="Comma-separated agent counts like 2,4,6,8,10 or 'auto'.",
    )
    parser.add_argument(
        "--subset-samples",
        type=_positive_int,
        default=3,
        help="How many agent subsets to sample per workload and agent count.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=20260315,
        help="Random seed for agent subset sampling.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["random", "prefix"],
        default="random",
        help="How to choose the active agents for each agent count.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names or 'all'.",
    )
    parser.add_argument(
        "--time-weight",
        type=_bounded_float,
        default=0.45,
        help="Composite ranking weight for latency.",
    )
    parser.add_argument(
        "--memory-weight",
        type=_bounded_float,
        default=0.35,
        help="Composite ranking weight for memory delta.",
    )
    parser.add_argument(
        "--quality-weight",
        type=_bounded_float,
        default=0.20,
        help="Composite ranking weight for quality proxy (mean normalized source score).",
    )
    parser.add_argument(
        "--hybrid-top-k",
        type=_non_negative_int,
        default=3,
        help="Require Hybrid average composite rank to be <= this value. Set 0 to disable.",
    )
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=0.02,
        help="RSS sampling interval in seconds used by the worker process.",
    )
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Enable SBERT in SenteTruthFinder. Disabled by default for more stable scalability benchmarking.",
    )
    parser.add_argument(
        "--sbert-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SBERT model name for SenteTruthFinder.",
    )
    parser.add_argument(
        "--strict-sbert",
        action="store_true",
        help="Enable strict SBERT loading behavior.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Restrict SBERT loading to local files only.",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-method",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-batch-start",
        type=_positive_int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-batch-group-size",
        type=_positive_int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-selected-agents",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-subset-index",
        type=_positive_int,
        default=1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_mode:
        return _run_worker_mode(args)
    return _run_benchmark_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
