"""
Numeric-only dynamic attack-injection benchmark for five truth-discovery methods.

This experiment reuses the same round-level source-shock construction as
`dynamic_attack_injection_comparison.py`, but runs on the pure numeric
`num_demo*.json` streams under `data_agent-10`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.examples.agent10_numeric_hybrid_supplement import (  # noqa: E402
    compute_numeric_rmse,
    load_numeric_stream,
)
from decentralized_truth_discovery.examples.dynamic_attack_injection_comparison import (  # noqa: E402
    METHOD_ORDER,
    _attack_round_map,
    _blend_agent10_batches,
    build_runners,
    compute_attack_metrics,
    compute_source_shock_tables,
)
from decentralized_truth_discovery.examples.five_methods_comparison import (  # noqa: E402
    create_reference_data,
)


@dataclass(frozen=True)
class NumericAttackConfig:
    data_agent_dir: str
    start_batch: int
    num_rounds: int
    num_seeds: int
    bad_agent_count: int
    replay_window: int
    max_iterations: int
    convergence_threshold: float
    hybrid_numeric_weight: float
    hybrid_field_support_threshold: float
    attack_scale: float
    recovery_tolerance: float
    recovery_consecutive_rounds: int
    use_sbert: bool
    output_dir: str
    skip_plots: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run numeric-only dynamic attack comparison across five truth-discovery methods."
    )
    parser.add_argument(
        "--data-agent-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data_agent-10"),
        help="Directory containing num_demo.json and num_demo_xbad_agent.json.",
    )
    parser.add_argument("--start-batch", type=int, default=1, help="1-based starting batch index.")
    parser.add_argument("--num-rounds", type=int, default=50, help="Number of batches in each scenario window.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of sliding windows to evaluate; each seed shifts the start batch by one.",
    )
    parser.add_argument(
        "--bad-agent-count",
        type=int,
        default=4,
        help="Use num_demo_<k>bad_agent.json, where k must be in [1, 4].",
    )
    parser.add_argument(
        "--replay-window",
        type=int,
        default=3,
        help="Sliding replay window used by replay-based baselines.",
    )
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum iterations per round.")
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-6,
        help="Shared convergence threshold.",
    )
    parser.add_argument(
        "--hybrid-numeric-weight",
        type=float,
        default=0.5,
        help="Numeric weight used by the Hybrid finder in numeric-only mode.",
    )
    parser.add_argument(
        "--hybrid-field-support-threshold",
        type=float,
        default=0.5,
        help="Minimum weighted support ratio required before a field enters Hybrid aggregation.",
    )
    parser.add_argument(
        "--attack-scale",
        type=float,
        default=4.0,
        help="Amplify malicious deltas from num_demo_<k>bad_agent.json after the injection round.",
    )
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.01,
        help="Allowed clean-vs-attacked RMSE gap to declare recovery.",
    )
    parser.add_argument(
        "--recovery-consecutive-rounds",
        type=int,
        default=2,
        help="Consecutive recovered rounds required to stop the recovery timer.",
    )
    parser.add_argument("--use-sbert", action="store_true", help="Use SBERT in SenteTruthFinder.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs", "numeric_attack_injection"),
        help="Directory for CSV / JSON outputs.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Reserved flag for interface compatibility.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NumericAttackConfig:
    if args.bad_agent_count < 1 or args.bad_agent_count > 4:
        raise ValueError(f"bad-agent-count must be in [1, 4], got: {args.bad_agent_count}")
    return NumericAttackConfig(
        data_agent_dir=os.path.abspath(args.data_agent_dir),
        start_batch=args.start_batch,
        num_rounds=args.num_rounds,
        num_seeds=args.num_seeds,
        bad_agent_count=args.bad_agent_count,
        replay_window=args.replay_window,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        hybrid_numeric_weight=args.hybrid_numeric_weight,
        hybrid_field_support_threshold=args.hybrid_field_support_threshold,
        attack_scale=args.attack_scale,
        recovery_tolerance=args.recovery_tolerance,
        recovery_consecutive_rounds=args.recovery_consecutive_rounds,
        use_sbert=args.use_sbert,
        output_dir=os.path.abspath(args.output_dir),
        skip_plots=args.skip_plots,
    )


def evaluate_numeric_stream(
    batches: Sequence[Dict[str, Any]],
    truth_maps: Sequence[Dict[str, Any]],
    config: NumericAttackConfig,
    *,
    seed: int,
    scenario: str,
    track: str,
    malicious_agents: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runners = build_runners(config)
    malicious_agent_set = set(malicious_agents)
    curve_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []

    for round_index, batch in enumerate(batches, start=1):
        text_df = create_reference_data(batch["items"])
        truth_map = truth_maps[round_index - 1]
        for runner in runners:
            method_result = runner.step(text_df, round_index - 1)
            rmse_value = compute_numeric_rmse(method_result.truth_preview, truth_map)
            curve_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "track": track,
                    "round": round_index,
                    "method": method_result.name,
                    "rmse": rmse_value,
                    "elapsed_sec": method_result.elapsed_sec,
                }
            )
            for website, score in method_result.source_scores.items():
                source_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "track": track,
                        "round": round_index,
                        "method": method_result.name,
                        "website": website,
                        "score": float(score),
                        "is_malicious": int(website in malicious_agent_set),
                    }
                )

    return pd.DataFrame(curve_rows), pd.DataFrame(source_rows)


def aggregate_metrics(
    attacked_curves: pd.DataFrame,
    clean_curves: pd.DataFrame,
    attack_round_map: Dict[str, int],
    config: NumericAttackConfig,
) -> pd.DataFrame:
    clean_lookup = clean_curves.pivot_table(index=["seed", "round"], columns="method", values="rmse")
    attacked_lookup = attacked_curves.pivot_table(index=["seed", "scenario", "round"], columns="method", values="rmse")
    rows: List[Dict[str, Any]] = []

    for seed in sorted(clean_curves["seed"].unique()):
        clean_seed = clean_lookup.loc[seed]
        for scenario, attack_round in attack_round_map.items():
            attacked_seed = attacked_lookup.loc[(seed, scenario)]
            for method in METHOD_ORDER:
                metrics = compute_attack_metrics(
                    attacked_seed[method].tolist(),
                    clean_seed[method].tolist(),
                    attack_round,
                    recovery_tolerance=config.recovery_tolerance,
                    consecutive_rounds=config.recovery_consecutive_rounds,
                )
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method": method,
                        "attack_round": attack_round,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        metrics_df.groupby(["scenario", "method"], sort=False)
        .agg(
            dynamic_score_mean=("dynamic_score", "mean"),
            dynamic_score_std=("dynamic_score", "std"),
            max_deviation_mean=("max_deviation", "mean"),
            post_attack_auc_mean=("post_attack_auc", "mean"),
            post_attack_rmse_auc_mean=("post_attack_rmse_auc", "mean"),
            rmse_spike_height_mean=("rmse_spike_height", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            final_gap_mean=("final_gap", "mean"),
            attacked_final_rmse_mean=("attacked_final_rmse", "mean"),
            clean_final_rmse_mean=("clean_final_rmse", "mean"),
        )
        .reset_index()
    )
    summary_df["dynamic_score_std"] = summary_df["dynamic_score_std"].fillna(0.0)
    summary_df = summary_df.sort_values(["scenario", "dynamic_score_mean", "post_attack_auc_mean", "recovery_time_mean"])
    summary_df["rank"] = summary_df.groupby("scenario").cumcount() + 1
    return summary_df


def curve_summary(curves_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curves_df.groupby(["scenario", "track", "round", "method"], sort=False)["rmse"]
        .agg([("rmse_mean", "mean"), ("rmse_std", "std")])
        .reset_index()
        .fillna({"rmse_std": 0.0})
    )


def source_summary(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return source_df
    return (
        source_df.groupby(["scenario", "track", "round", "method", "website", "is_malicious"], sort=False)["score"]
        .agg([("score_mean", "mean"), ("score_std", "std")])
        .reset_index()
        .fillna({"score_std": 0.0})
    )


def run_experiment(
    config: NumericAttackConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    attack_round_map = _attack_round_map(config.num_rounds)
    clean_curve_frames: List[pd.DataFrame] = []
    clean_source_frames: List[pd.DataFrame] = []
    attacked_curve_frames: List[pd.DataFrame] = []
    attacked_source_frames: List[pd.DataFrame] = []
    seed_manifest: List[Dict[str, Any]] = []

    for seed_offset in range(config.num_seeds):
        seed = seed_offset + 1
        clean_batches, attacked_reference_batches, truth_maps, malicious_agents, metadata = load_numeric_stream(
            config,
            window_offset=seed_offset,
        )
        seed_manifest.append({"seed": seed, "malicious_agents": malicious_agents, **metadata})

        clean_curves, clean_sources = evaluate_numeric_stream(
            clean_batches,
            truth_maps,
            config,
            seed=seed,
            scenario="clean",
            track="clean",
            malicious_agents=malicious_agents,
        )
        clean_curve_frames.append(clean_curves)
        clean_source_frames.append(clean_sources)

        for scenario, attack_round in attack_round_map.items():
            attacked_batches = _blend_agent10_batches(
                clean_batches,
                attacked_reference_batches,
                malicious_agents,
                attack_round,
                attack_scale=config.attack_scale,
            )
            attacked_curves, attacked_sources = evaluate_numeric_stream(
                attacked_batches,
                truth_maps,
                config,
                seed=seed,
                scenario=scenario,
                track="attacked",
                malicious_agents=malicious_agents,
            )
            attacked_curve_frames.append(attacked_curves)
            attacked_source_frames.append(attacked_sources)

    clean_curves_df = pd.concat(clean_curve_frames, ignore_index=True)
    clean_sources_df = pd.concat(clean_source_frames, ignore_index=True)
    attacked_curves_df = pd.concat(attacked_curve_frames, ignore_index=True)
    attacked_sources_df = pd.concat(attacked_source_frames, ignore_index=True)

    metrics_df = aggregate_metrics(attacked_curves_df, clean_curves_df, attack_round_map, config)
    summary_df = summarize_metrics(metrics_df)
    sources_df = pd.concat([clean_sources_df, attacked_sources_df], ignore_index=True)
    source_shock_seed_df, source_shock_round_df, source_shock_summary_df = compute_source_shock_tables(
        sources_df,
        attack_round_map,
        consecutive_rounds=config.recovery_consecutive_rounds,
    )
    if not source_shock_summary_df.empty:
        summary_df = summary_df.merge(source_shock_summary_df, on=["scenario", "method"], how="left")

    manifest = {
        "attack_rounds": attack_round_map,
        "seed_manifest": seed_manifest,
        "config": config.__dict__,
    }
    curves_df = pd.concat([clean_curves_df, attacked_curves_df], ignore_index=True)
    return curves_df, sources_df, metrics_df, summary_df, source_shock_seed_df, source_shock_round_df, manifest


def print_summary(summary_df: pd.DataFrame, attack_round_map: Dict[str, int]) -> None:
    print("\nFive-method numeric dynamic attack summary")
    print(
        f"Attack rounds: early={attack_round_map['early']}, mid={attack_round_map['mid']}, late={attack_round_map['late']}"
    )
    for scenario in ["early", "mid", "late"]:
        scenario_df = summary_df[summary_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue
        print(f"\nScenario: {scenario}")
        print(
            scenario_df[
                [
                    "rank",
                    "method",
                    "rmse_spike_height_mean",
                    "max_deviation_mean",
                    "post_attack_rmse_auc_mean",
                    "score_shock_peak_mean",
                    "score_shock_auc_mean",
                    "score_shock_recovery_time_mean",
                    "recovery_time_mean",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    curves_df, sources_df, metrics_df, summary_df, source_shock_seed_df, source_shock_round_df, manifest = run_experiment(config)
    curves_summary_df = curve_summary(curves_df)
    sources_summary_df = source_summary(sources_df)
    attack_round_map = manifest["attack_rounds"]

    curves_df.to_csv(os.path.join(config.output_dir, "numeric_attack_curves.csv"), index=False)
    curves_summary_df.to_csv(os.path.join(config.output_dir, "numeric_attack_curve_summary.csv"), index=False)
    sources_df.to_csv(os.path.join(config.output_dir, "numeric_attack_source_scores.csv"), index=False)
    sources_summary_df.to_csv(os.path.join(config.output_dir, "numeric_attack_source_summary.csv"), index=False)
    source_shock_seed_df.to_csv(os.path.join(config.output_dir, "numeric_attack_source_shock_seed_round.csv"), index=False)
    source_shock_round_df.to_csv(os.path.join(config.output_dir, "numeric_attack_source_shock_round.csv"), index=False)
    source_shock_round_df.to_csv(os.path.join(config.output_dir, "数值-随着攻击时间变化.csv"), index=False)
    metrics_df.to_csv(os.path.join(config.output_dir, "numeric_attack_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(config.output_dir, "numeric_attack_summary.csv"), index=False)
    with open(os.path.join(config.output_dir, "numeric_attack_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print_summary(summary_df, attack_round_map)
    print(f"\nOutputs written to: {config.output_dir}")


if __name__ == "__main__":
    main()
