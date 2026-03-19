from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Mapping

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.examples.proof_benchmark_utils import (  # noqa: E402
    METHOD_NAMES,
    build_saved_bundle_artifacts,
    compare_verify_params,
    create_text_df_from_batch,
    default_experiment_id,
    ensure_directory,
    extract_score_columns,
    format_float,
    integrity_check_bundle_payload,
    load_batches_from_path,
    mutate_first_numeric_field,
    verify_bundle_payload,
    write_json,
)


DEFAULT_INPUT_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_agent-10", "demo.json")
)
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))



def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected integer >= 1, got {value!r}")
    return parsed



def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected float >= 0, got {value!r}")
    return parsed



def _json_clone(payload: Any) -> Any:
    return copy.deepcopy(payload)



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



def _load_first_verify_params(chainmaker_manifest_path: str | None) -> list[dict[str, str]] | None:
    if not chainmaker_manifest_path:
        return None
    manifest = json.load(open(chainmaker_manifest_path, "r", encoding="utf-8"))
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    verify_path = groups[0].get("verify_params_path")
    if not verify_path:
        return None
    return json.load(open(verify_path, "r", encoding="utf-8"))



def _tamper_first_score_row(bundle_payload: Dict[str, Any], delta: float) -> tuple[Dict[str, Any], str]:
    rows = bundle_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("bundle_payload.rows must be a non-empty list")
    score_columns = extract_score_columns(rows)
    if not score_columns:
        raise ValueError("No score-like columns found in bundle rows")
    tampered = _json_clone(bundle_payload)
    target_column = score_columns[0]
    mutate_first_numeric_field(tampered, ["rows", 0, target_column], delta)
    return tampered, f"rows[0].{target_column}"



def _tamper_first_public_signal(bundle_payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    tampered = _json_clone(bundle_payload)
    mutate_first_numeric_field(tampered, ["proof_groups", 0, "public_json", 1], 1)
    return tampered, "proof_groups[0].public_json[1]"



def _tamper_first_proof_coordinate(bundle_payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    tampered = _json_clone(bundle_payload)
    mutate_first_numeric_field(tampered, ["proof_groups", 0, "proof_json", "pi_a", 0], 1)
    return tampered, "proof_groups[0].proof_json.pi_a[0]"



def _tamper_verify_params(verify_params: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
    tampered = _json_clone(verify_params)
    first_item = tampered[0]
    key = next(iter(first_item.keys()))
    first_item[key] = str(int(first_item[key]) + 1)
    return tampered, f"verify_params[0].{key}"



def _evaluate_case(
    *,
    method_name: str,
    case_name: str,
    tamper_target: str,
    bundle_payload: Dict[str, Any],
    baseline_contract_submission: Mapping[str, Any],
    verify_params_payload: List[Dict[str, str]] | None,
    poll_interval_sec: float,
) -> Dict[str, Any]:
    integrity = integrity_check_bundle_payload(bundle_payload)
    verify_metric = verify_bundle_payload(bundle_payload, poll_interval_sec=poll_interval_sec)
    verify_params_applicable = verify_params_payload is not None
    verify_params_match = (
        compare_verify_params(baseline_contract_submission, verify_params_payload)
        if verify_params_applicable
        else None
    )
    detected = (
        (not integrity["passed"])
        or (not verify_metric.all_verified)
        or (verify_params_applicable and not bool(verify_params_match))
    )
    return {
        "method": method_name,
        "tamper_case": case_name,
        "tamper_target": tamper_target,
        "integrity_passed": bool(integrity["passed"]),
        "integrity_issue_count": int(integrity["issue_count"]),
        "integrity_issues": json.dumps(integrity["issues"], ensure_ascii=False),
        "verify_passed": bool(verify_metric.all_verified),
        "verified_group_count": int(verify_metric.verified_group_count),
        "success_group_count": int(verify_metric.success_group_count),
        "verify_total_elapsed_sec": float(verify_metric.total_elapsed_sec),
        "verify_peak_rss_mb": float(verify_metric.max_peak_rss_mb),
        "verify_params_applicable": bool(verify_params_applicable),
        "verify_params_match": (float(verify_params_match) if verify_params_match is not None else float("nan")),
        "detected": bool(detected),
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark proof tamper resilience and show which guardrail catches which mutation."
    )
    parser.add_argument(
        "--input-json",
        default=DEFAULT_INPUT_JSON_PATH,
        help="Full path to the input JSON file.",
    )
    parser.add_argument(
        "--batch-number",
        type=_positive_int,
        default=1,
        help="1-based batch index to benchmark.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names or 'all'.",
    )
    parser.add_argument(
        "--score-delta",
        type=_non_negative_float,
        default=0.05,
        help="Delta added to the first score-like row field when simulating score tampering.",
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
    methods = METHOD_NAMES if args.methods.strip().lower() == "all" else [token.strip() for token in args.methods.split(",") if token.strip()]
    unsupported_methods = [method for method in methods if method not in METHOD_NAMES]
    if unsupported_methods:
        raise ValueError(f"Unsupported methods: {unsupported_methods}")

    batches = load_batches_from_path(input_json_path)
    if args.batch_number > len(batches):
        raise ValueError(
            f"batch_number={args.batch_number} exceeds available batch count={len(batches)}"
        )
    text_df = create_text_df_from_batch(batches[args.batch_number - 1], batch_number=args.batch_number)

    experiment_id = args.experiment_id or default_experiment_id("proof_tamper_resilience", input_json_path)
    output_dir = ensure_directory(os.path.join(args.output_root, experiment_id))
    baseline_root = ensure_directory(os.path.join(output_dir, "baseline"))
    tampered_root = ensure_directory(os.path.join(output_dir, "tampered"))

    print("=" * 108)
    print("Proof tamper resilience benchmark")
    print("=" * 108)
    print(f"Input JSON: {input_json_path}")
    print(f"Batch number: {args.batch_number}")
    print(f"Methods: {methods}")
    print(f"Output dir: {output_dir}")

    rows: List[Dict[str, Any]] = []
    for method_name in methods:
        method_result = _run_method_by_name(
            method_name,
            text_df,
            use_sbert=args.use_sbert,
            sbert_model=args.sbert_model,
            strict_sbert=args.strict_sbert,
            local_files_only=args.local_files_only,
        )
        method_output_dir = ensure_directory(os.path.join(baseline_root, method_name))
        bundle_artifacts = build_saved_bundle_artifacts(
            method_result,
            experiment_id=f"{experiment_id}_batch_{args.batch_number:03d}",
            output_dir=method_output_dir,
        )
        baseline_bundle = _json_clone(bundle_artifacts.bundle_payload)
        baseline_verify_params = _load_first_verify_params(bundle_artifacts.chainmaker_manifest_path)
        baseline_contract_submission = baseline_bundle["proof_groups"][0]["contract_submission"]

        baseline_case = _evaluate_case(
            method_name=method_name,
            case_name="baseline",
            tamper_target="none",
            bundle_payload=baseline_bundle,
            baseline_contract_submission=baseline_contract_submission,
            verify_params_payload=baseline_verify_params,
            poll_interval_sec=args.poll_interval_sec,
        )
        rows.append(baseline_case)

        case_builders = [
            ("score_row_tamper", _tamper_first_score_row, baseline_verify_params),
            ("public_signal_tamper", _tamper_first_public_signal, baseline_verify_params),
            ("proof_json_tamper", _tamper_first_proof_coordinate, baseline_verify_params),
        ]
        for case_name, tamper_builder, verify_params_payload in case_builders:
            tampered_bundle, tamper_target = tamper_builder(baseline_bundle, args.score_delta) if case_name == "score_row_tamper" else tamper_builder(baseline_bundle)
            tampered_path = write_json(
                os.path.join(tampered_root, method_name, f"{case_name}.proof_bundle.json"),
                tampered_bundle,
            )
            row = _evaluate_case(
                method_name=method_name,
                case_name=case_name,
                tamper_target=tamper_target,
                bundle_payload=tampered_bundle,
                baseline_contract_submission=baseline_contract_submission,
                verify_params_payload=verify_params_payload,
                poll_interval_sec=args.poll_interval_sec,
            )
            row["tampered_bundle_path"] = tampered_path
            rows.append(row)

        if baseline_verify_params is not None:
            tampered_verify_params, tamper_target = _tamper_verify_params(baseline_verify_params)
            tampered_verify_params_path = write_json(
                os.path.join(tampered_root, method_name, "verify_params_tamper.verify_params.json"),
                tampered_verify_params,
            )
            row = _evaluate_case(
                method_name=method_name,
                case_name="verify_params_tamper",
                tamper_target=tamper_target,
                bundle_payload=baseline_bundle,
                baseline_contract_submission=baseline_contract_submission,
                verify_params_payload=tampered_verify_params,
                poll_interval_sec=args.poll_interval_sec,
            )
            row["tampered_verify_params_path"] = tampered_verify_params_path
            rows.append(row)

        print(
            f"[{method_name}] baseline_detected={baseline_case['detected']} "
            f"baseline_verify={format_float(baseline_case['verify_total_elapsed_sec'])}s"
        )

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df.groupby(["method", "tamper_case"], sort=False)
        .agg(
            run_count=("tamper_case", "size"),
            detected_rate=("detected", "mean"),
            integrity_pass_rate=("integrity_passed", "mean"),
            verify_pass_rate=("verify_passed", "mean"),
            verify_params_applicable_rate=("verify_params_applicable", "mean"),
            verify_params_match_rate=("verify_params_match", "mean"),
            mean_verify_total_elapsed_sec=("verify_total_elapsed_sec", "mean"),
            mean_verify_peak_rss_mb=("verify_peak_rss_mb", "mean"),
        )
        .reset_index()
        .sort_values(["method", "tamper_case"])
        .reset_index(drop=True)
    )

    results_path = os.path.join(output_dir, "proof_tamper_resilience_results.csv")
    summary_path = os.path.join(output_dir, "proof_tamper_resilience_summary.csv")
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "input_json": input_json_path,
        "batch_number": int(args.batch_number),
        "methods": methods,
        "score_delta": float(args.score_delta),
        "paths": {
            "results_csv": results_path,
            "summary_csv": summary_path,
        },
    }
    write_json(os.path.join(output_dir, "proof_tamper_resilience_metadata.json"), metadata)

    print("\nSummary:")
    print(summary_df.round(4).to_string(index=False))
    print(f"\nresults: {results_path}")
    print(f"summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
