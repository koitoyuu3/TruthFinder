from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.examples.proof_benchmark_utils import (  # noqa: E402
    METHOD_NAMES,
    build_saved_bundle_artifacts,
    bundle_group_payload_bytes,
    create_text_df_from_batch,
    default_experiment_id,
    ensure_directory,
    format_float,
    load_batches_from_path,
    mean_source_score_pct,
    parse_attack_ratio_from_path,
    verify_bundle_payload,
    write_json,
)


DEFAULT_CLEAN_JSON_PATH = os.path.abspath(
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



def _resolve_attacked_paths(args: argparse.Namespace, clean_json_path: str) -> List[str]:
    if args.attacked_jsons:
        attacked = [os.path.abspath(token.strip()) for token in args.attacked_jsons.split(",") if token.strip()]
        if not attacked:
            raise ValueError("--attacked-jsons was provided but no valid path was parsed.")
        return attacked

    attack_dir = os.path.abspath(args.attack_dir) if args.attack_dir else os.path.dirname(clean_json_path)
    pattern = os.path.join(attack_dir, args.attacked_glob)
    import glob

    attacked = sorted(glob.glob(pattern))
    if not attacked:
        raise FileNotFoundError(f"No attacked JSON matched pattern: {pattern}")
    return attacked



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



def _extract_statement_map(result_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "proof" not in result_df.columns or "proof_id" not in result_df.columns:
        return {}

    statement_map: Dict[str, Dict[str, Any]] = {}
    for _, row in result_df.drop_duplicates(subset=["proof_id"]).iterrows():
        proof = row.get("proof") if isinstance(row.get("proof"), Mapping) else {}
        statement = proof.get("statement") if isinstance(proof.get("statement"), Mapping) else {}
        group_id = statement.get("group_id") if isinstance(statement.get("group_id"), Mapping) else {}
        group_key = json.dumps(group_id, sort_keys=True, ensure_ascii=False)
        statement_map[group_key] = {
            "proof_id": proof.get("proof_id"),
            "statement_digest_sha256": proof.get("statement_digest_sha256"),
            "input_digest": statement.get("input_digest"),
            "truth_digest": statement.get("truth_digest"),
            "source_score_digest": statement.get("source_score_digest"),
        }
    return statement_map



def _change_ratio(
    baseline_map: Mapping[str, Mapping[str, Any]],
    current_map: Mapping[str, Mapping[str, Any]],
    field: str,
) -> float:
    keys = sorted(set(baseline_map) | set(current_map))
    if not keys:
        return float("nan")
    changed = 0
    for key in keys:
        baseline_value = baseline_map.get(key, {}).get(field)
        current_value = current_map.get(key, {}).get(field)
        if baseline_value != current_value:
            changed += 1
    return changed / float(len(keys))



def _pct_delta(current: float, baseline: float) -> float:
    if pd.isna(current) or pd.isna(baseline) or abs(float(baseline)) < 1e-12:
        return float("nan")
    return (float(current) - float(baseline)) / float(baseline) * 100.0



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how malicious-ratio scenarios affect proof structure and proof content drift."
    )
    parser.add_argument(
        "--clean-json",
        default=DEFAULT_CLEAN_JSON_PATH,
        help="Full path to the clean baseline JSON.",
    )
    parser.add_argument(
        "--attack-dir",
        default=None,
        help="Directory to search attacked JSON files in when --attacked-jsons is not provided.",
    )
    parser.add_argument(
        "--attacked-glob",
        default="demo_*bad_agent.json",
        help="Glob pattern under --attack-dir for attacked scenarios.",
    )
    parser.add_argument(
        "--attacked-jsons",
        default=None,
        help="Comma-separated full paths to attacked JSON files; overrides --attack-dir/--attacked-glob.",
    )
    parser.add_argument(
        "--start-batch",
        type=_positive_int,
        default=1,
        help="1-based batch index to start from.",
    )
    parser.add_argument(
        "--num-batches",
        type=_positive_int,
        default=5,
        help="How many aligned batches to compare per scenario.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names or 'all'.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=_non_negative_float,
        default=0.02,
        help="RSS polling interval while verifying proof groups.",
    )
    parser.add_argument(
        "--total-agents",
        type=_positive_int,
        default=10,
        help="Used to convert '*bad_agent' filenames into attack ratios.",
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
    clean_json_path = os.path.abspath(args.clean_json)
    attacked_paths = _resolve_attacked_paths(args, clean_json_path)
    methods = METHOD_NAMES if args.methods.strip().lower() == "all" else [token.strip() for token in args.methods.split(",") if token.strip()]
    unsupported_methods = [method for method in methods if method not in METHOD_NAMES]
    if unsupported_methods:
        raise ValueError(f"Unsupported methods: {unsupported_methods}")

    clean_batches = load_batches_from_path(clean_json_path)
    selected_batch_numbers = list(range(args.start_batch, args.start_batch + args.num_batches))
    if selected_batch_numbers[-1] > len(clean_batches):
        raise ValueError(
            f"Requested clean batch range [{selected_batch_numbers[0]}, {selected_batch_numbers[-1]}] exceeds available batches {len(clean_batches)}."
        )

    experiment_id = args.experiment_id or default_experiment_id("proof_malicious_ratio", clean_json_path)
    output_dir = ensure_directory(os.path.join(args.output_root, experiment_id))
    artifacts_root = ensure_directory(os.path.join(output_dir, "proof_artifacts"))

    print("=" * 108)
    print("Proof malicious-ratio benchmark")
    print("=" * 108)
    print(f"Clean JSON: {clean_json_path}")
    print(f"Attacked JSONs: {attacked_paths}")
    print(f"Batch range: {selected_batch_numbers[0]} -> {selected_batch_numbers[-1]}")
    print(f"Methods: {methods}")
    print(f"Output dir: {output_dir}")

    scenario_rows: List[Dict[str, Any]] = []
    scenario_registry = [("clean", clean_json_path, 0.0)] + [
        (os.path.splitext(os.path.basename(path))[0], path, parse_attack_ratio_from_path(path, total_agents=args.total_agents))
        for path in attacked_paths
    ]

    for scenario_label, scenario_path, attack_ratio in scenario_registry:
        scenario_batches = load_batches_from_path(scenario_path)
        if selected_batch_numbers[-1] > len(scenario_batches):
            raise ValueError(
                f"Scenario {scenario_label} does not have enough batches for requested range. "
                f"Need {selected_batch_numbers[-1]}, got {len(scenario_batches)}."
            )

        for batch_number in selected_batch_numbers:
            text_df = create_text_df_from_batch(scenario_batches[batch_number - 1], batch_number=batch_number)
            websites = text_df["website"].astype(str).drop_duplicates().tolist()
            scenario_batch_output = ensure_directory(
                os.path.join(output_dir, "proof_artifacts", scenario_label, f"batch_{batch_number:03d}")
            )

            for method_name in methods:
                row: Dict[str, Any] = {
                    "scenario": scenario_label,
                    "scenario_path": scenario_path,
                    "attack_ratio": attack_ratio,
                    "batch_number": int(batch_number),
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
                    bundle_artifacts = build_saved_bundle_artifacts(
                        method_result,
                        experiment_id=f"{experiment_id}_{scenario_label}_batch_{batch_number:03d}",
                        output_dir=scenario_batch_output,
                    )
                    verify_metric = verify_bundle_payload(
                        bundle_artifacts.bundle_payload,
                        poll_interval_sec=args.poll_interval_sec,
                    )
                    mean_score_pct, scored_website_count = mean_source_score_pct(method_result, websites)
                    statement_map = _extract_statement_map(method_result.result_df)
                    row.update(
                        {
                            "success": True,
                            "website_count": int(len(websites)),
                            "object_count": int(text_df["object"].nunique()),
                            "method_elapsed_sec": float(method_result.elapsed_sec),
                            "proof_bundle_bytes": int(os.path.getsize(bundle_artifacts.bundle_path)),
                            "proof_payload_bytes_total": bundle_group_payload_bytes(bundle_artifacts.bundle_payload),
                            "proof_group_count": int(len(bundle_artifacts.bundle_payload.get("proof_groups", []))),
                            "verify_total_elapsed_sec": float(verify_metric.total_elapsed_sec),
                            "verify_peak_rss_mb": float(verify_metric.max_peak_rss_mb),
                            "all_groups_verified": bool(verify_metric.all_verified),
                            "mean_source_score_pct": mean_score_pct,
                            "scored_website_count": int(scored_website_count),
                            "result_digest": str(bundle_artifacts.bundle_payload.get("result_digest")),
                            "statement_digest_map_json": json.dumps(statement_map, ensure_ascii=False, sort_keys=True),
                            "bundle_path": bundle_artifacts.bundle_path,
                            "manifest_path": bundle_artifacts.manifest_path,
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "success": False,
                            "error": str(exc),
                            "website_count": int(len(websites)),
                            "object_count": int(text_df["object"].nunique()),
                            "method_elapsed_sec": float("nan"),
                            "proof_bundle_bytes": float("nan"),
                            "proof_payload_bytes_total": float("nan"),
                            "proof_group_count": float("nan"),
                            "verify_total_elapsed_sec": float("nan"),
                            "verify_peak_rss_mb": float("nan"),
                            "all_groups_verified": False,
                            "mean_source_score_pct": float("nan"),
                            "scored_website_count": 0,
                            "result_digest": "",
                            "statement_digest_map_json": "{}",
                        }
                    )

                scenario_rows.append(row)
                print(
                    f"[{scenario_label}] batch={batch_number} method={method_name} status={'OK' if row['success'] else 'FAIL'} "
                    f"bundle={format_float(row.get('proof_bundle_bytes'), 2)}B "
                    f"verify={format_float(row.get('verify_total_elapsed_sec'))}s"
                )

    raw_df = pd.DataFrame(scenario_rows)
    clean_df = raw_df[raw_df["scenario"] == "clean"].copy()
    attacked_df = raw_df[raw_df["scenario"] != "clean"].copy()
    drift_rows: List[Dict[str, Any]] = []

    for _, attacked_row in attacked_df.iterrows():
        matching_clean = clean_df[
            (clean_df["batch_number"] == attacked_row["batch_number"])
            & (clean_df["method"] == attacked_row["method"])
        ]
        if matching_clean.empty:
            continue
        baseline = matching_clean.iloc[0]
        baseline_map = json.loads(str(baseline.get("statement_digest_map_json") or "{}"))
        current_map = json.loads(str(attacked_row.get("statement_digest_map_json") or "{}"))

        drift_rows.append(
            {
                "scenario": attacked_row["scenario"],
                "scenario_path": attacked_row["scenario_path"],
                "attack_ratio": float(attacked_row["attack_ratio"]),
                "batch_number": int(attacked_row["batch_number"]),
                "method": attacked_row["method"],
                "proof_bundle_bytes_delta_pct": _pct_delta(attacked_row["proof_bundle_bytes"], baseline["proof_bundle_bytes"]),
                "proof_payload_bytes_delta_pct": _pct_delta(attacked_row["proof_payload_bytes_total"], baseline["proof_payload_bytes_total"]),
                "verify_total_elapsed_sec_delta_pct": _pct_delta(attacked_row["verify_total_elapsed_sec"], baseline["verify_total_elapsed_sec"]),
                "verify_peak_rss_mb_delta_pct": _pct_delta(attacked_row["verify_peak_rss_mb"], baseline["verify_peak_rss_mb"]),
                "mean_source_score_pct_delta": float(attacked_row["mean_source_score_pct"] - baseline["mean_source_score_pct"])
                if pd.notna(attacked_row["mean_source_score_pct"]) and pd.notna(baseline["mean_source_score_pct"])
                else float("nan"),
                "proof_group_count_delta": float(attacked_row["proof_group_count"] - baseline["proof_group_count"])
                if pd.notna(attacked_row["proof_group_count"]) and pd.notna(baseline["proof_group_count"])
                else float("nan"),
                "result_digest_changed": attacked_row["result_digest"] != baseline["result_digest"],
                "statement_digest_change_ratio": _change_ratio(baseline_map, current_map, "statement_digest_sha256"),
                "truth_digest_change_ratio": _change_ratio(baseline_map, current_map, "truth_digest"),
                "source_score_digest_change_ratio": _change_ratio(baseline_map, current_map, "source_score_digest"),
                "input_digest_change_ratio": _change_ratio(baseline_map, current_map, "input_digest"),
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    summary_df = (
        drift_df.groupby(["scenario", "attack_ratio", "method"], sort=False)
        .agg(
            run_count=("method", "size"),
            mean_proof_bundle_bytes_delta_pct=("proof_bundle_bytes_delta_pct", "mean"),
            mean_proof_payload_bytes_delta_pct=("proof_payload_bytes_delta_pct", "mean"),
            mean_verify_total_elapsed_sec_delta_pct=("verify_total_elapsed_sec_delta_pct", "mean"),
            mean_verify_peak_rss_mb_delta_pct=("verify_peak_rss_mb_delta_pct", "mean"),
            mean_source_score_pct_delta=("mean_source_score_pct_delta", "mean"),
            mean_proof_group_count_delta=("proof_group_count_delta", "mean"),
            result_digest_changed_rate=("result_digest_changed", "mean"),
            mean_statement_digest_change_ratio=("statement_digest_change_ratio", "mean"),
            mean_truth_digest_change_ratio=("truth_digest_change_ratio", "mean"),
            mean_source_score_digest_change_ratio=("source_score_digest_change_ratio", "mean"),
            mean_input_digest_change_ratio=("input_digest_change_ratio", "mean"),
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(["attack_ratio", "mean_statement_digest_change_ratio", "method"]).reset_index(drop=True)

    raw_results_path = os.path.join(output_dir, "proof_malicious_ratio_runs.csv")
    drift_path = os.path.join(output_dir, "proof_malicious_ratio_drift.csv")
    summary_path = os.path.join(output_dir, "proof_malicious_ratio_summary.csv")
    raw_df.to_csv(raw_results_path, index=False)
    drift_df.to_csv(drift_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "clean_json": clean_json_path,
        "attacked_jsons": attacked_paths,
        "batch_range": [selected_batch_numbers[0], selected_batch_numbers[-1]],
        "methods": methods,
        "paths": {
            "raw_results_csv": raw_results_path,
            "drift_csv": drift_path,
            "summary_csv": summary_path,
        },
    }
    write_json(os.path.join(output_dir, "proof_malicious_ratio_metadata.json"), metadata)

    print("\nSummary:")
    display_columns = [
        "scenario",
        "attack_ratio",
        "method",
        "mean_proof_bundle_bytes_delta_pct",
        "mean_verify_total_elapsed_sec_delta_pct",
        "mean_verify_peak_rss_mb_delta_pct",
        "mean_source_score_pct_delta",
        "mean_statement_digest_change_ratio",
        "mean_truth_digest_change_ratio",
        "mean_source_score_digest_change_ratio",
    ]
    print(summary_df[display_columns].round(4).to_string(index=False))
    print(f"\nraw runs: {raw_results_path}")
    print(f"drift rows: {drift_path}")
    print(f"summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
