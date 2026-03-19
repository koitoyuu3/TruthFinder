from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from decentralized_truth_discovery.core import (
    build_experiment_proof_bundle,
    export_chainmaker_group_payloads,
)
from decentralized_truth_discovery.core.chainmaker_evm import (
    build_chainmaker_submit_params,
    build_chainmaker_verify_params,
)
from decentralized_truth_discovery.core.zk_proof import digest_object
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


HYBRID_METHOD_NAME = "EnhancedTruthFinder-Hybrid"
METHOD_NAMES = [
    HYBRID_METHOD_NAME,
    "EnhancedTruthFinder-NonHybrid",
    "SenteTruthFinder",
    "BasicTruthFinder",
    "SenFeedTruthFinder-CompositeTD",
]
SNARKJS_MODULE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "core",
        "circuits",
        "node_modules",
        "snarkjs",
    )
)


@dataclass(frozen=True)
class SavedBundleArtifacts:
    method: str
    method_slug: str
    bundle_payload: Dict[str, Any]
    bundle_path: str
    manifest_path: str
    chainmaker_manifest_path: str
    chainmaker_manifest: Dict[str, Any]


@dataclass(frozen=True)
class VerifyGroupMetric:
    group_index: int
    proof_id: str
    success: bool
    elapsed_sec: float
    peak_rss_mb: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VerifyBundleMetric:
    verified_group_count: int
    success_group_count: int
    all_verified: bool
    total_elapsed_sec: float
    max_peak_rss_mb: float
    group_metrics: List[VerifyGroupMetric]


def _sanitize_path_stem(file_path: str) -> str:
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", file_stem).strip("._-")
    return sanitized or "input"


def sanitize_filename_part(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    return normalized or "unknown"


def ensure_directory(path: str) -> str:
    absolute = os.path.abspath(path)
    os.makedirs(absolute, exist_ok=True)
    return absolute


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def write_json(path: str, payload: Any) -> str:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "w", encoding="utf-8") as file_obj:
        json.dump(json_safe(payload), file_obj, ensure_ascii=False, indent=2)
    return absolute


def json_size_bytes(payload: Any) -> int:
    return len(json.dumps(json_safe(payload), ensure_ascii=False, indent=2).encode("utf-8"))


def load_batches_from_path(json_path: str) -> List[Dict[str, Any]]:
    return load_demo_batches(os.path.abspath(json_path))


def select_batch(all_batches: Sequence[Dict[str, Any]], batch_number: int) -> Dict[str, Any]:
    if batch_number < 1 or batch_number > len(all_batches):
        raise ValueError(
            f"batch_number must be within [1, {len(all_batches)}], got {batch_number}."
        )
    return all_batches[batch_number - 1]


def create_text_df_from_batch(batch: Dict[str, Any], *, batch_number: int) -> pd.DataFrame:
    reference_items = create_reference_input(batch, batch_number=batch_number)
    return create_reference_data(reference_items)


def run_all_methods_for_text_df(
    text_df: pd.DataFrame,
    *,
    use_sbert: bool,
    sbert_model: Optional[str],
    strict_sbert: bool,
    local_files_only: bool,
):
    return [
        run_enhanced(
            text_df,
            finder=_create_enhanced_finder(enable_hybrid_mode=True),
            label=HYBRID_METHOD_NAME,
        ),
        run_enhanced(
            text_df,
            finder=_create_enhanced_finder(enable_hybrid_mode=False),
            label="EnhancedTruthFinder-NonHybrid",
        ),
        run_sente(
            text_df,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            strict_sbert=strict_sbert,
            local_files_only=local_files_only,
        ),
        run_basic(text_df),
        run_senfeed(text_df),
    ]


def build_saved_bundle_artifacts(
    method_result,
    *,
    experiment_id: str,
    output_dir: str,
    save_chainmaker: bool = True,
) -> SavedBundleArtifacts:
    output_root = ensure_directory(output_dir)
    method_slug = sanitize_filename_part(method_result.name)
    method_experiment_id = f"{experiment_id}:{method_slug}"
    bundle = build_experiment_proof_bundle(
        method_result.result_df,
        experiment_id=method_experiment_id,
        method=method_result.name,
        include_rows=True,
        include_group_rows=False,
    )
    bundle_payload = bundle.to_dict()

    bundle_path = write_json(os.path.join(output_root, f"{method_slug}.proof_bundle.json"), bundle_payload)
    if save_chainmaker:
        try:
            chainmaker_manifest = export_chainmaker_group_payloads(
                bundle_payload,
                output_dir=output_root,
                bundle_name=method_slug,
            )
            chainmaker_manifest = dict(chainmaker_manifest, exported=True, error="")
        except Exception as exc:
            chainmaker_manifest = {
                "bundle_name": method_slug,
                "groups": [],
                "manifest_path": "",
                "exported": False,
                "error": str(exc),
            }
    else:
        chainmaker_manifest = {
            "bundle_name": method_slug,
            "groups": [],
            "manifest_path": "",
            "exported": False,
            "error": "chainmaker export disabled",
        }

    manifest = {
        "experiment_id": experiment_id,
        "method": method_result.name,
        "method_experiment_id": method_experiment_id,
        "row_count": bundle.row_count,
        "proof_group_count": len(bundle.proof_groups),
        "bundle_path": bundle_path,
        "chainmaker": chainmaker_manifest,
    }
    manifest_path = write_json(os.path.join(output_root, f"{method_slug}.proof_manifest.json"), manifest)

    return SavedBundleArtifacts(
        method=method_result.name,
        method_slug=method_slug,
        bundle_payload=bundle_payload,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        chainmaker_manifest_path=str(chainmaker_manifest.get("manifest_path", "")),
        chainmaker_manifest=dict(chainmaker_manifest),
    )


def mean_source_score_pct(method_result, websites: Sequence[str]) -> tuple[float, int]:
    if not method_result.source_scores:
        return float("nan"), 0

    raw_scores = pd.Series(method_result.source_scores, dtype=float)
    raw_scores = raw_scores.reindex(list(websites))
    percentage_scores = _native_score_to_percentage(raw_scores, method_result.name)
    valid_scores = percentage_scores.dropna()
    if valid_scores.empty:
        return float("nan"), 0
    return float(valid_scores.mean()), int(valid_scores.shape[0])


def _run_and_measure_peak_rss(command: Sequence[str]) -> Dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_sec = time.perf_counter() - started
    return {
        "returncode": int(process.returncode),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "elapsed_sec": float(elapsed_sec),
        "peak_rss_mb": float("nan"),
    }


def _verify_group_with_node_wrapper(
    verification_key_path: str,
    public_path: str,
    proof_path: str,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="truthfinder-proof-verify-js-") as temp_dir:
        script_path = os.path.join(temp_dir, "verify_proof.js")
        script_text = f"""
const fs = require('fs');
const snarkjs = require({json.dumps(SNARKJS_MODULE)});

(async () => {{
  const verificationKey = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const publicSignals = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const proof = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
  let peakRss = process.memoryUsage().rss;
  const started = process.hrtime.bigint();
  const sampler = setInterval(() => {{
    const rss = process.memoryUsage().rss;
    if (rss > peakRss) peakRss = rss;
  }}, 5);
  try {{
    const ok = await snarkjs.groth16.verify(verificationKey, publicSignals, proof);
    clearInterval(sampler);
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
    const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
    console.log(JSON.stringify({{ ok, elapsed_ms: elapsedMs, peak_rss_mb: peakRss / (1024 * 1024) }}));
    process.exit(ok ? 0 : 2);
  }} catch (error) {{
    clearInterval(sampler);
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
    const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
    console.error(error && error.stack ? error.stack : String(error));
    console.log(JSON.stringify({{ ok: false, elapsed_ms: elapsedMs, peak_rss_mb: peakRss / (1024 * 1024) }}));
    process.exit(1);
  }}
}})();
"""
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(script_text)
        result = _run_and_measure_peak_rss(["node", script_path, verification_key_path, public_path, proof_path])
    stdout_text = str(result.get("stdout") or "").strip()
    parsed = None
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(parsed, Mapping):
        result["elapsed_sec"] = float(parsed.get("elapsed_ms", 0.0)) / 1000.0
        result["peak_rss_mb"] = float(parsed.get("peak_rss_mb", float("nan")))
    return result


def verify_bundle_payload(
    bundle_payload: Mapping[str, Any],
    *,
    poll_interval_sec: float = 0.02,
) -> VerifyBundleMetric:
    group_metrics: List[VerifyGroupMetric] = []
    total_elapsed_sec = 0.0
    max_peak_rss_mb = float("nan")

    proof_groups = bundle_payload.get("proof_groups")
    if not isinstance(proof_groups, list):
        raise ValueError("bundle_payload.proof_groups must be a list")
    if not os.path.exists(SNARKJS_MODULE):
        raise FileNotFoundError(f"snarkjs module not found: {SNARKJS_MODULE}")

    for group_index, group in enumerate(proof_groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"proof_groups[{group_index}] must be an object")

        scheme = str(group.get("proof_scheme") or "")
        if scheme != "groth16_truth_aggregation_v1":
            continue

        proof_json = group.get("proof_json")
        public_json = group.get("public_json")
        verification_key_path = group.get("verification_key_path")
        if not isinstance(proof_json, Mapping):
            raise ValueError(f"proof_groups[{group_index}] missing proof_json")
        if not isinstance(public_json, list):
            raise ValueError(f"proof_groups[{group_index}] missing public_json")
        if not verification_key_path:
            raise ValueError(f"proof_groups[{group_index}] missing verification_key_path")

        with tempfile.TemporaryDirectory(prefix="truthfinder-proof-verify-") as temp_dir:
            proof_path = write_json(os.path.join(temp_dir, "proof.json"), proof_json)
            public_path = write_json(os.path.join(temp_dir, "public.json"), public_json)
            result = _verify_group_with_node_wrapper(
                str(verification_key_path),
                public_path,
                proof_path,
            )

        total_elapsed_sec += float(result["elapsed_sec"])
        peak_rss_mb = float(result["peak_rss_mb"])
        if not pd.isna(peak_rss_mb):
            max_peak_rss_mb = peak_rss_mb if pd.isna(max_peak_rss_mb) else max(max_peak_rss_mb, peak_rss_mb)

        group_metrics.append(
            VerifyGroupMetric(
                group_index=group_index,
                proof_id=str(group.get("proof_id") or f"group-{group_index}"),
                success=int(result["returncode"]) == 0,
                elapsed_sec=float(result["elapsed_sec"]),
                peak_rss_mb=peak_rss_mb,
                stdout=str(result["stdout"]),
                stderr=str(result["stderr"]),
            )
        )

    success_group_count = sum(1 for metric in group_metrics if metric.success)
    return VerifyBundleMetric(
        verified_group_count=len(group_metrics),
        success_group_count=success_group_count,
        all_verified=(len(group_metrics) > 0 and success_group_count == len(group_metrics)),
        total_elapsed_sec=float(total_elapsed_sec),
        max_peak_rss_mb=float(max_peak_rss_mb) if not pd.isna(max_peak_rss_mb) else float("nan"),
        group_metrics=group_metrics,
    )


def bundle_group_payload_bytes(bundle_payload: Mapping[str, Any]) -> int:
    total = 0
    for group in bundle_payload.get("proof_groups", []):
        total += json_size_bytes(group.get("proof_json"))
        total += json_size_bytes(group.get("public_json"))
    return int(total)


def parse_attack_ratio_from_path(path: str, *, total_agents: int = 10) -> float:
    name = os.path.basename(path)
    match_bad = re.search(r"(?:^|_)(\d+)bad_agent", name)
    if match_bad:
        return min(max(int(match_bad.group(1)) / float(total_agents), 0.0), 1.0)
    match_pct = re.search(r"(\d+)%wrong_data", name)
    if match_pct:
        return min(max(int(match_pct.group(1)) / 100.0, 0.0), 1.0)
    return 0.0


def to_bytes32_hex(value: Any) -> str:
    text = str(value).lower().removeprefix("0x")
    if len(text) != 64:
        text = digest_object(value)
    return f"0x{text}"


def filter_group_rows(rows: Sequence[Mapping[str, Any]], group_id: Mapping[str, Any]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if all(row.get(key) == value for key, value in group_id.items()):
            filtered.append(dict(row))
    return filtered


def integrity_check_bundle_payload(bundle_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = bundle_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("bundle_payload.rows must be a list")

    issues: List[str] = []
    expected_top_result_digest = digest_object(rows)
    actual_top_result_digest = str(bundle_payload.get("result_digest"))
    if expected_top_result_digest != actual_top_result_digest:
        issues.append("top_level_result_digest_mismatch")

    group_results: List[Dict[str, Any]] = []
    for group_index, group in enumerate(bundle_payload.get("proof_groups", [])):
        if not isinstance(group, Mapping):
            raise ValueError(f"proof_groups[{group_index}] must be an object")
        group_id = group.get("group_id") if isinstance(group.get("group_id"), Mapping) else {}
        group_rows = filter_group_rows(rows, group_id)
        expected_group_digest = digest_object(group_id)
        group_issues: List[str] = []
        if str(group.get("group_digest")) != expected_group_digest:
            group_issues.append("group_digest_mismatch")

        contract_submission = group.get("contract_submission")
        if isinstance(contract_submission, Mapping):
            if str(contract_submission.get("groupId")) != to_bytes32_hex(expected_group_digest):
                group_issues.append("contract_group_id_mismatch")
            if str(contract_submission.get("proofId")) != to_bytes32_hex(group.get("proof_id")):
                group_issues.append("contract_proof_id_mismatch")
            if str(contract_submission.get("statementDigestSha256")) != to_bytes32_hex(
                group.get("statement_digest_sha256")
            ):
                group_issues.append("contract_statement_digest_mismatch")

        group_results.append(
            {
                "group_index": int(group_index),
                "proof_id": str(group.get("proof_id") or f"group-{group_index}"),
                "row_count": int(len(group_rows)),
                "passed": len(group_issues) == 0,
                "issues": group_issues,
            }
        )
        issues.extend(f"group_{group_index}:{issue}" for issue in group_issues)

    return {
        "passed": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
        "group_results": group_results,
    }


def compare_verify_params(
    contract_submission: Mapping[str, Any],
    candidate_payload: Sequence[Any],
) -> bool:
    expected = build_chainmaker_verify_params(contract_submission)
    return json_safe(expected) == json_safe(candidate_payload)


def compare_submit_params(
    contract_submission: Mapping[str, Any],
    candidate_payload: Sequence[Any],
    *,
    result_uri: str = "",
    proof_uri: str = "",
) -> bool:
    expected = build_chainmaker_submit_params(
        contract_submission,
        result_uri=result_uri,
        proof_uri=proof_uri,
    )
    return json_safe(expected) == json_safe(candidate_payload)


def extract_score_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    candidate_columns = [
        "source_reliability",
        "trustworthiness",
        "source_weight",
        "credibility",
        "error_metric",
        "consistency_metric",
    ]
    available = []
    row_keys = set().union(*(row.keys() for row in rows)) if rows else set()
    for column in candidate_columns:
        if column in row_keys:
            available.append(column)
    return available


def mutate_first_numeric_field(payload: MutableMapping[str, Any], field_path: Sequence[Any], delta: float) -> None:
    target: Any = payload
    for key in field_path[:-1]:
        target = target[key]
    last_key = field_path[-1]
    current = target[last_key]
    if isinstance(current, str):
        try:
            numeric = int(current)
            target[last_key] = str(numeric + int(delta))
            return
        except ValueError:
            try:
                numeric = float(current)
                target[last_key] = str(numeric + float(delta))
                return
            except ValueError as exc:
                raise ValueError(f"Field path {field_path} is not numeric") from exc
    if isinstance(current, (int, float)):
        target[last_key] = current + delta
        return
    raise ValueError(f"Field path {field_path} is not numeric")


def normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
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


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def default_experiment_id(prefix: str, input_path: str) -> str:
    return f"{prefix}_{_sanitize_path_stem(input_path)}_{int(time.time())}"


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"

