"""
Hybrid-only numeric supplement experiment on `data_agent-10/num_demo*.json`.

This script focuses on the pure numeric stream and evaluates only
`EnhancedTruthFinder-Hybrid` under dynamic attack injection. It produces a
fine-grained result bundle including:
- round-level RMSE curves
- asset-level prediction errors for BTC / ETH / DOGE
- per-round source reliability details
- malicious-vs-honest suppression summaries
- hybrid-only plots for timing and source-weight evolution
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.core import EnhancedTruthFinder  # noqa: E402
from decentralized_truth_discovery.examples.byzantine_rmse_comparison import (  # noqa: E402
    _build_clean_oracle,
    _normalize_price_prediction,
    _price_error,
)
from decentralized_truth_discovery.examples.dynamic_attack_injection_comparison import (  # noqa: E402
    ASSETS,
    _attack_round_map,
    _blend_agent10_batches,
    _infer_changed_agents,
    _select_batch_window,
    compute_attack_metrics,
    compute_source_shock_tables,
)
from decentralized_truth_discovery.examples.five_methods_comparison import (  # noqa: E402
    _build_asset_truth_preview,
    _build_hybrid_coverage_scores,
    _run_quietly,
    _sort_score_dict,
    create_reference_data,
    load_demo_batches,
)


@dataclass(frozen=True)
class NumericHybridConfig:
    data_agent_dir: str
    start_batch: int
    num_rounds: int
    num_seeds: int
    bad_agent_count: int
    max_iterations: int
    convergence_threshold: float
    hybrid_numeric_weight: float
    hybrid_field_support_threshold: float
    attack_scale: float
    recovery_tolerance: float
    recovery_consecutive_rounds: int
    output_dir: str
    skip_plots: bool


class HybridNumericRunner:
    method_name = "EnhancedTruthFinder-Hybrid"

    def __init__(self, config: NumericHybridConfig):
        self.finder = EnhancedTruthFinder(
            alpha=0.9,
            beta=100.0,
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations,
            initial_reliability=0.5,
            numeric_weight=config.hybrid_numeric_weight,
            enable_hybrid_mode=True,
            enable_zk_proof=False,
        )
        self.finder.hybrid_field_support_threshold = config.hybrid_field_support_threshold

    def step(self, text_df: pd.DataFrame, epoch: int) -> Dict[str, Any]:
        start = time.perf_counter()
        result_df = _run_quietly(lambda: self.finder.process_batch(text_df.copy(), epoch=epoch))
        elapsed = time.perf_counter() - start

        details_raw = self.finder.get_source_reliability()
        coverage_scores = _build_hybrid_coverage_scores(self.finder, result_df)
        source_details: Dict[str, Dict[str, float]] = {}
        source_scores: Dict[str, float] = {}
        for website, metrics in details_raw.items():
            reliability = float(metrics.get("r", np.nan))
            error_metric = float(metrics.get("e", np.nan))
            consistency_metric = float(metrics.get("k", np.nan))
            coverage = float(coverage_scores.get(website, 0.0))
            calibrated_score = min(1.0, reliability + (0.12 * coverage))
            source_details[str(website)] = {
                "r": reliability,
                "e": error_metric,
                "k": consistency_metric,
                "coverage": coverage,
                "score": calibrated_score,
            }
            if pd.notna(calibrated_score):
                source_scores[str(website)] = calibrated_score

        return {
            "result_df": result_df,
            "truth_preview": _build_asset_truth_preview(result_df),
            "source_details": source_details,
            "source_scores": _sort_score_dict(source_scores),
            "elapsed_sec": elapsed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a detailed Hybrid-only numeric supplement experiment on num_demo*.json."
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
        default=5,
        help="Number of sliding windows to evaluate; each seed shifts the start batch by one.",
    )
    parser.add_argument(
        "--bad-agent-count",
        type=int,
        default=4,
        help="Use num_demo_<k>bad_agent.json, where k must be in [1, 4].",
    )
    parser.add_argument("--max-iterations", type=int, default=80, help="Maximum Hybrid iterations per round.")
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-6,
        help="Hybrid convergence threshold.",
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
        help="Amplify malicious deltas from num_demo_<k>bad_agent.json after the injection round; 1.0 replays the original file.",
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
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs", "agent10_numeric_hybrid_supplement"),
        help="Directory for CSV / JSON / PNG outputs.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NumericHybridConfig:
    if args.bad_agent_count < 1 or args.bad_agent_count > 4:
        raise ValueError(f"bad-agent-count must be in [1, 4], got: {args.bad_agent_count}")
    return NumericHybridConfig(
        data_agent_dir=os.path.abspath(args.data_agent_dir),
        start_batch=args.start_batch,
        num_rounds=args.num_rounds,
        num_seeds=args.num_seeds,
        bad_agent_count=args.bad_agent_count,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        hybrid_numeric_weight=args.hybrid_numeric_weight,
        hybrid_field_support_threshold=args.hybrid_field_support_threshold,
        attack_scale=args.attack_scale,
        recovery_tolerance=args.recovery_tolerance,
        recovery_consecutive_rounds=args.recovery_consecutive_rounds,
        output_dir=os.path.abspath(args.output_dir),
        skip_plots=args.skip_plots,
    )


def load_numeric_stream(
    config: NumericHybridConfig,
    *,
    window_offset: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    clean_path = os.path.join(config.data_agent_dir, "num_demo.json")
    attacked_path = os.path.join(config.data_agent_dir, f"num_demo_{config.bad_agent_count}bad_agent.json")

    clean_batches_all = load_demo_batches(clean_path)
    attacked_batches_all = load_demo_batches(attacked_path)
    if len(clean_batches_all) != len(attacked_batches_all):
        raise ValueError(
            f"Clean/attacked batch counts differ: clean={len(clean_batches_all)}, attacked={len(attacked_batches_all)}"
        )

    window_start_batch = config.start_batch + window_offset
    clean_batches = _select_batch_window(
        clean_batches_all,
        start_batch=window_start_batch,
        num_rounds=config.num_rounds,
    )
    attacked_reference_batches = _select_batch_window(
        attacked_batches_all,
        start_batch=window_start_batch,
        num_rounds=config.num_rounds,
    )
    truth_maps = [_build_clean_oracle(batch) for batch in clean_batches]
    malicious_agents = _infer_changed_agents(clean_batches[0], attacked_reference_batches[0])
    if not malicious_agents:
        raise ValueError(f"Failed to infer malicious agents from {attacked_path}")

    metadata = {
        "clean_path": os.path.abspath(clean_path),
        "attacked_path": os.path.abspath(attacked_path),
        "window_start_batch": window_start_batch,
        "window_end_batch": window_start_batch + config.num_rounds - 1,
        "bad_agent_count": config.bad_agent_count,
    }
    return clean_batches, attacked_reference_batches, truth_maps, malicious_agents, metadata


def compute_numeric_rmse(predictions: Dict[str, Any], truth_map: Dict[str, Any]) -> float:
    object_errors: List[float] = []
    for object_id, truth in truth_map.items():
        prediction = predictions.get(object_id)
        object_errors.append(_price_error(_normalize_price_prediction(prediction), truth))
    if not object_errors:
        return 0.0
    return float(math.sqrt(np.mean(np.square(object_errors))))


def extract_asset_rows(
    *,
    predictions: Dict[str, Any],
    truth_map: Dict[str, Any],
    seed: int,
    scenario: str,
    track: str,
    round_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for object_id, truth in truth_map.items():
        predicted_map = _normalize_price_prediction(predictions.get(object_id)) or {}
        for asset in ASSETS:
            oracle_value = float(truth[asset])
            estimated_value = predicted_map.get(asset)
            if estimated_value is None:
                abs_pct_error = 1.0
                signed_pct_error = np.nan
            else:
                signed_pct_error = float((float(estimated_value) - oracle_value) / max(abs(oracle_value), 1e-12))
                abs_pct_error = abs(signed_pct_error)
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "track": track,
                    "round": round_index,
                    "object": object_id,
                    "asset": asset,
                    "oracle_value": oracle_value,
                    "estimated_value": float(estimated_value) if estimated_value is not None else np.nan,
                    "abs_pct_error": float(abs_pct_error),
                    "signed_pct_error": float(signed_pct_error) if not pd.isna(signed_pct_error) else np.nan,
                }
            )
    return rows


def extract_input_surface_rows(
    *,
    batch: Dict[str, Any],
    truth_map: Dict[str, Any],
    seed: int,
    scenario: str,
    track: str,
    round_index: int,
    malicious_agents: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    malicious_agent_set = set(malicious_agents)
    for item in batch.get("items", []):
        object_id = str(item["object"])
        if object_id not in truth_map:
            continue
        response = item.get("response")
        if not isinstance(response, dict):
            continue
        truth = truth_map[object_id]
        for asset in ASSETS:
            observed = float(response[asset])
            oracle = float(truth[asset])
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "track": track,
                    "round": round_index,
                    "website": str(item["agent"]),
                    "is_malicious": int(str(item["agent"]) in malicious_agent_set),
                    "asset": asset,
                    "input_abs_pct_deviation": abs((observed - oracle) / max(abs(oracle), 1e-12)),
                }
            )
    return rows


def evaluate_hybrid_stream(
    batches: Sequence[Dict[str, Any]],
    truth_maps: Sequence[Dict[str, Any]],
    config: NumericHybridConfig,
    *,
    seed: int,
    scenario: str,
    track: str,
    malicious_agents: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runner = HybridNumericRunner(config)
    curve_rows: List[Dict[str, Any]] = []
    asset_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    input_rows: List[Dict[str, Any]] = []
    malicious_agent_set = set(malicious_agents)

    for round_index, batch in enumerate(batches, start=1):
        text_df = create_reference_data(batch["items"])
        truth_map = truth_maps[round_index - 1]
        result = runner.step(text_df, round_index - 1)
        rmse_value = compute_numeric_rmse(result["truth_preview"], truth_map)
        input_rows.extend(
            extract_input_surface_rows(
                batch=batch,
                truth_map=truth_map,
                seed=seed,
                scenario=scenario,
                track=track,
                round_index=round_index,
                malicious_agents=malicious_agents,
            )
        )
        curve_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "track": track,
                "round": round_index,
                "method": HybridNumericRunner.method_name,
                "rmse": rmse_value,
                "elapsed_sec": float(result["elapsed_sec"]),
            }
        )
        asset_rows.extend(
            extract_asset_rows(
                predictions=result["truth_preview"],
                truth_map=truth_map,
                seed=seed,
                scenario=scenario,
                track=track,
                round_index=round_index,
            )
        )
        for website, detail in result["source_details"].items():
            source_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "track": track,
                    "round": round_index,
                    "website": website,
                    "method": HybridNumericRunner.method_name,
                    "is_malicious": int(website in malicious_agent_set),
                    "score": float(detail.get("score", np.nan)),
                    "r": float(detail.get("r", np.nan)),
                    "e": float(detail.get("e", np.nan)),
                    "k": float(detail.get("k", np.nan)),
                    "coverage": float(detail.get("coverage", np.nan)),
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(asset_rows), pd.DataFrame(source_rows), pd.DataFrame(input_rows)


def aggregate_metrics(
    attacked_curves: pd.DataFrame,
    clean_curves: pd.DataFrame,
    attack_round_map: Dict[str, int],
    config: NumericHybridConfig,
) -> pd.DataFrame:
    clean_lookup = clean_curves.pivot_table(index=["seed", "round"], columns="method", values="rmse")
    attacked_lookup = attacked_curves.pivot_table(index=["seed", "scenario", "round"], columns="method", values="rmse")
    rows: List[Dict[str, Any]] = []
    for seed in sorted(clean_curves["seed"].unique()):
        clean_seed = clean_lookup.loc[seed][HybridNumericRunner.method_name].tolist()
        for scenario, attack_round in attack_round_map.items():
            attacked_seed = attacked_lookup.loc[(seed, scenario)][HybridNumericRunner.method_name].tolist()
            metrics = compute_attack_metrics(
                attacked_seed,
                clean_seed,
                attack_round,
                recovery_tolerance=config.recovery_tolerance,
                consecutive_rounds=config.recovery_consecutive_rounds,
            )
            rows.append({
                "seed": seed,
                "scenario": scenario,
                "method": HybridNumericRunner.method_name,
                "attack_round": attack_round,
                **metrics,
            })
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
    summary_df["rank"] = 1
    return summary_df


def summarize_assets(asset_df: pd.DataFrame, attack_round_map: Dict[str, int]) -> pd.DataFrame:
    attacked_df = asset_df[asset_df["track"] == "attacked"].copy()
    rows: List[Dict[str, Any]] = []
    for scenario, attack_round in attack_round_map.items():
        scenario_df = attacked_df[attacked_df["scenario"] == scenario]
        post_df = scenario_df[scenario_df["round"] >= attack_round]
        if post_df.empty:
            continue
        for asset in ASSETS:
            asset_post = post_df[post_df["asset"] == asset]
            if asset_post.empty:
                continue
            final_round = int(asset_post["round"].max())
            final_df = asset_post[asset_post["round"] == final_round]
            rows.append(
                {
                    "scenario": scenario,
                    "asset": asset,
                    "post_attack_mean_abs_pct_error": float(asset_post["abs_pct_error"].mean()),
                    "post_attack_max_abs_pct_error": float(asset_post["abs_pct_error"].max()),
                    "final_round_mean_abs_pct_error": float(final_df["abs_pct_error"].mean()),
                    "final_round_mean_signed_pct_error": float(final_df["signed_pct_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_sources(source_df: pd.DataFrame, attack_round_map: Dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_summary = (
        source_df.groupby(["scenario", "track", "round", "is_malicious"], sort=False)
        .agg(
            score_mean=("score", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            r_mean=("r", "mean"),
            e_mean=("e", "mean"),
            k_mean=("k", "mean"),
            coverage_mean=("coverage", "mean"),
        )
        .reset_index()
    )

    attacked_df = source_summary[source_summary["track"] == "attacked"].copy()
    rows: List[Dict[str, Any]] = []
    for scenario, attack_round in attack_round_map.items():
        scenario_df = attacked_df[attacked_df["scenario"] == scenario]
        if scenario_df.empty:
            continue
        final_round = int(scenario_df["round"].max())
        final_df = scenario_df[scenario_df["round"] == final_round]
        malicious_mean = float(final_df[final_df["is_malicious"] == 1]["score_mean"].mean())
        honest_mean = float(final_df[final_df["is_malicious"] == 0]["score_mean"].mean())
        attack_df = scenario_df[scenario_df["round"] == attack_round]
        rows.append(
            {
                "scenario": scenario,
                "attack_round": attack_round,
                "final_round": final_round,
                "attack_round_malicious_score": float(attack_df[attack_df["is_malicious"] == 1]["score_mean"].mean()),
                "attack_round_honest_score": float(attack_df[attack_df["is_malicious"] == 0]["score_mean"].mean()),
                "final_malicious_score": malicious_mean,
                "final_honest_score": honest_mean,
                "final_suppression_gap": float(honest_mean - malicious_mean),
            }
        )
    return source_summary, pd.DataFrame(rows)


def prepare_matplotlib(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".mplconfig"))
    import matplotlib.pyplot as plt

    return plt


def plot_timing(curve_df: pd.DataFrame, attack_round_map: Dict[str, int], output_dir: str):
    plt = prepare_matplotlib(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clean_df = curve_df[curve_df["scenario"] == "clean"]
    ax.plot(clean_df["round"], clean_df["rmse_mean"], linestyle="--", color="grey", linewidth=1.8, label="Clean baseline")
    color_map = {"early": "#d62728", "mid": "#ff7f0e", "late": "#2ca02c"}
    for scenario, attack_round in attack_round_map.items():
        scenario_df = curve_df[curve_df["scenario"] == scenario]
        ax.plot(
            scenario_df["round"],
            scenario_df["rmse_mean"],
            linewidth=2.0,
            color=color_map[scenario],
            label=f"{scenario.capitalize()} (T_attack={attack_round})",
        )
        ax.fill_between(
            scenario_df["round"],
            scenario_df["rmse_mean"] - scenario_df["rmse_std"],
            scenario_df["rmse_mean"] + scenario_df["rmse_std"],
            alpha=0.12,
            color=color_map[scenario],
        )
        ax.axvline(attack_round, linestyle=":", color=color_map[scenario], alpha=0.8)
    ax.set_title("Hybrid-only numeric dynamic injection")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numeric_hybrid_timing.png"), dpi=200)
    plt.close(fig)


def plot_weight_shock_timing(shock_round_df: pd.DataFrame, attack_round_map: Dict[str, int], output_dir: str):
    if shock_round_df.empty:
        return
    plt = prepare_matplotlib(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clean_df = shock_round_df[(shock_round_df["scenario"] == "clean") & (shock_round_df["track"] == "clean")].copy()
    if not clean_df.empty:
        ax.plot(
            clean_df["round"],
            clean_df["attack_induced_shock_mean"],
            linestyle="--",
            linewidth=1.6,
            color="grey",
            label="Clean baseline (attack-induced shock=0)",
        )
    color_map = {"early": "#d62728", "mid": "#ff7f0e", "late": "#2ca02c"}
    for scenario, attack_round in attack_round_map.items():
        scenario_df = shock_round_df[(shock_round_df["scenario"] == scenario) & (shock_round_df["track"] == "attacked")].copy()
        if scenario_df.empty:
            continue
        color = color_map[scenario]
        ax.plot(
            scenario_df["round"],
            scenario_df["attack_induced_shock_mean"],
            linewidth=2.0,
            color=color,
            label=f"{scenario.capitalize()} shock",
        )
        ax.fill_between(
            scenario_df["round"],
            scenario_df["attack_induced_shock_mean"] - scenario_df["attack_induced_shock_std"],
            scenario_df["attack_induced_shock_mean"] + scenario_df["attack_induced_shock_std"],
            alpha=0.12,
            color=color,
        )
        ax.axvline(attack_round, linestyle=":", color=color, alpha=0.8)
    ax.set_title("Hybrid numeric attack-induced weight shock")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("Attack-induced mean |Δ weight|")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numeric_hybrid_weight_shock_timing.png"), dpi=200)
    plt.close(fig)


def plot_mid_weight_heatmap(source_df: pd.DataFrame, output_dir: str):
    plt = prepare_matplotlib(output_dir)
    scenario_df = source_df[(source_df["scenario"] == "mid") & (source_df["track"] == "attacked")].copy()
    if scenario_df.empty:
        return
    ordered_websites = (
        scenario_df[["website", "is_malicious"]]
        .drop_duplicates()
        .sort_values(["is_malicious", "website"], ascending=[False, True])["website"]
        .tolist()
    )
    heatmap_df = scenario_df.pivot_table(index="website", columns="round", values="score", aggfunc="mean").reindex(ordered_websites)
    heatmap_df = heatmap_df.sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.42 * len(ordered_websites))))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Hybrid numeric source-weight evolution (mid attack)")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("Agent")
    ax.set_yticks(range(len(ordered_websites)))
    ax.set_yticklabels(ordered_websites)
    step = max(1, len(heatmap_df.columns) // 10)
    ax.set_xticks(range(0, len(heatmap_df.columns), step))
    ax.set_xticklabels(list(heatmap_df.columns[::step]))
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Hybrid score")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numeric_hybrid_weight_heatmap_mid.png"), dpi=200)
    plt.close(fig)


def summarize_input_surface(input_df: pd.DataFrame, attack_round_map: Dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_summary_df = (
        input_df.groupby(["scenario", "track", "round", "is_malicious", "asset"], sort=False)["input_abs_pct_deviation"]
        .mean()
        .reset_index(name="mean_input_abs_pct_deviation")
    )
    attacked_df = input_summary_df[input_summary_df["track"] == "attacked"].copy()
    rows: List[Dict[str, Any]] = []
    for scenario, attack_round in attack_round_map.items():
        scenario_df = attacked_df[(attacked_df["scenario"] == scenario) & (attacked_df["round"] >= attack_round)]
        for asset in ASSETS:
            asset_df = scenario_df[scenario_df["asset"] == asset]
            if asset_df.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "asset": asset,
                    "malicious_post_attack_input_deviation": float(asset_df[asset_df["is_malicious"] == 1]["mean_input_abs_pct_deviation"].mean()),
                    "honest_post_attack_input_deviation": float(asset_df[asset_df["is_malicious"] == 0]["mean_input_abs_pct_deviation"].mean()),
                }
            )
    return input_summary_df, pd.DataFrame(rows)


def run_experiment(config: NumericHybridConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    attack_round_map = _attack_round_map(config.num_rounds)
    clean_curve_frames: List[pd.DataFrame] = []
    clean_asset_frames: List[pd.DataFrame] = []
    clean_source_frames: List[pd.DataFrame] = []
    clean_input_frames: List[pd.DataFrame] = []
    attacked_curve_frames: List[pd.DataFrame] = []
    attacked_asset_frames: List[pd.DataFrame] = []
    attacked_source_frames: List[pd.DataFrame] = []
    attacked_input_frames: List[pd.DataFrame] = []
    seed_manifest: List[Dict[str, Any]] = []

    for seed_offset in range(config.num_seeds):
        seed = seed_offset + 1
        clean_batches, attacked_reference_batches, truth_maps, malicious_agents, metadata = load_numeric_stream(
            config,
            window_offset=seed_offset,
        )
        seed_manifest.append({"seed": seed, "malicious_agents": malicious_agents, **metadata})

        clean_curves, clean_assets, clean_sources, clean_inputs = evaluate_hybrid_stream(
            clean_batches,
            truth_maps,
            config,
            seed=seed,
            scenario="clean",
            track="clean",
            malicious_agents=malicious_agents,
        )
        clean_curve_frames.append(clean_curves)
        clean_asset_frames.append(clean_assets)
        clean_source_frames.append(clean_sources)
        clean_input_frames.append(clean_inputs)

        for scenario, attack_round in attack_round_map.items():
            attacked_batches = _blend_agent10_batches(
                clean_batches,
                attacked_reference_batches,
                malicious_agents,
                attack_round,
                attack_scale=config.attack_scale,
            )
            attacked_curves, attacked_assets, attacked_sources, attacked_inputs = evaluate_hybrid_stream(
                attacked_batches,
                truth_maps,
                config,
                seed=seed,
                scenario=scenario,
                track="attacked",
                malicious_agents=malicious_agents,
            )
            attacked_curve_frames.append(attacked_curves)
            attacked_asset_frames.append(attacked_assets)
            attacked_source_frames.append(attacked_sources)
            attacked_input_frames.append(attacked_inputs)

    clean_curves_df = pd.concat(clean_curve_frames, ignore_index=True)
    attacked_curves_df = pd.concat(attacked_curve_frames, ignore_index=True)
    metrics_df = aggregate_metrics(attacked_curves_df, clean_curves_df, attack_round_map, config)
    summary_df = summarize_metrics(metrics_df)

    asset_df = pd.concat(clean_asset_frames + attacked_asset_frames, ignore_index=True)
    source_df = pd.concat(clean_source_frames + attacked_source_frames, ignore_index=True)
    source_shock_seed_df, source_shock_round_df, source_shock_summary_df = compute_source_shock_tables(
        source_df,
        attack_round_map,
        consecutive_rounds=config.recovery_consecutive_rounds,
        default_method=HybridNumericRunner.method_name,
    )
    if not source_shock_summary_df.empty:
        summary_df = summary_df.merge(source_shock_summary_df, on=["scenario", "method"], how="left")
    asset_summary_df = summarize_assets(asset_df, attack_round_map)
    source_summary_df, suppression_df = summarize_sources(source_df, attack_round_map)
    input_df = pd.concat(clean_input_frames + attacked_input_frames, ignore_index=True)
    input_summary_df, input_attack_summary_df = summarize_input_surface(input_df, attack_round_map)

    manifest = {
        "attack_rounds": attack_round_map,
        "seed_manifest": seed_manifest,
        "config": config.__dict__,
    }
    curves_df = pd.concat([clean_curves_df, attacked_curves_df], ignore_index=True)
    return curves_df, metrics_df, summary_df, asset_df, asset_summary_df, source_df, source_summary_df, source_shock_seed_df, source_shock_round_df, suppression_df, input_df, input_summary_df, input_attack_summary_df, manifest


def curve_summary(curves_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curves_df.groupby(["scenario", "round"], sort=False)["rmse"]
        .agg([("rmse_mean", "mean"), ("rmse_std", "std")])
        .reset_index()
        .fillna({"rmse_std": 0.0})
    )


def build_attack_indicator_table(
    source_shock_round_df: pd.DataFrame,
    attack_round_map: Dict[str, int],
) -> pd.DataFrame:
    indicator_df = source_shock_round_df[source_shock_round_df["track"] == "attacked"].copy()
    if indicator_df.empty:
        return indicator_df
    indicator_df = indicator_df[[
        "scenario",
        "round",
        "attack_induced_shock_mean",
        "attack_induced_shock_std",
    ]].copy()
    indicator_df["attack_round"] = indicator_df["scenario"].map(attack_round_map).astype(int)
    indicator_df["relative_round"] = indicator_df["round"] - indicator_df["attack_round"]
    indicator_df["phase"] = np.where(indicator_df["round"] < indicator_df["attack_round"], "pre_attack", "post_attack")
    return indicator_df[[
        "scenario",
        "attack_round",
        "round",
        "relative_round",
        "phase",
        "attack_induced_shock_mean",
        "attack_induced_shock_std",
    ]].sort_values(["scenario", "round"]).reset_index(drop=True)


def print_report(attack_indicator_df: pd.DataFrame, attack_round_map: Dict[str, int]):
    print("\nHybrid-only numeric attack indicator")
    print(
        f"Attack rounds: early={attack_round_map['early']}, mid={attack_round_map['mid']}, late={attack_round_map['late']}"
    )
    if attack_indicator_df.empty:
        print("\nNo attack indicator rows generated.")
        return

    for scenario in ["early", "mid", "late"]:
        scenario_df = attack_indicator_df[attack_indicator_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue
        print(f"\nScenario: {scenario}")
        print(
            scenario_df[[
                "round",
                "relative_round",
                "phase",
                "attack_induced_shock_mean",
                "attack_induced_shock_std",
            ]]
            .round(6)
            .to_string(index=False)
        )

def main():
    args = parse_args()
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    curves_df, metrics_df, summary_df, asset_df, asset_summary_df, source_df, source_summary_df, source_shock_seed_df, source_shock_round_df, suppression_df, input_df, input_summary_df, input_attack_summary_df, manifest = run_experiment(config)
    attack_round_map = manifest["attack_rounds"]
    curve_summary_df = curve_summary(curves_df)
    attack_indicator_df = build_attack_indicator_table(source_shock_round_df, attack_round_map)

    curves_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_curves.csv"), index=False)
    curve_summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_curve_summary.csv"), index=False)
    metrics_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_summary.csv"), index=False)
    asset_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_asset_errors.csv"), index=False)
    asset_summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_asset_summary.csv"), index=False)
    source_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_source_details.csv"), index=False)
    source_summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_source_summary.csv"), index=False)
    source_shock_seed_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_source_shock_seed_round.csv"), index=False)
    source_shock_round_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_source_shock_summary.csv"), index=False)
    attack_indicator_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_attack_indicator_rounds.csv"), index=False)
    suppression_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_suppression_summary.csv"), index=False)
    input_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_input_surface.csv"), index=False)
    input_summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_input_surface_summary.csv"), index=False)
    input_attack_summary_df.to_csv(os.path.join(config.output_dir, "numeric_hybrid_input_attack_summary.csv"), index=False)
    with open(os.path.join(config.output_dir, "numeric_hybrid_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if not config.skip_plots:
        plot_timing(curve_summary_df, attack_round_map, config.output_dir)
        plot_weight_shock_timing(source_shock_round_df, attack_round_map, config.output_dir)
        plot_mid_weight_heatmap(source_df, config.output_dir)

    print_report(attack_indicator_df, attack_round_map)
    print(f"\nOutputs written to: {config.output_dir}")


if __name__ == "__main__":
    main()
