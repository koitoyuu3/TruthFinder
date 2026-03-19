"""
Dynamic attack-injection benchmark for five truth-discovery methods.

The benchmark simulates a realistic stream where all agents begin honest and a
subset is compromised mid-run. The experiment emphasizes post-attack recovery
under mixed-modality data: structured price reports and semantically similar but
numerically manipulated news summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

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
from decentralized_truth_discovery.examples.byzantine_rmse_comparison import (  # noqa: E402
    _build_clean_oracle,
    _news_error,
    _normalize_news_prediction,
    _normalize_price_prediction,
    _price_error,
)
from decentralized_truth_discovery.examples.five_methods_comparison import (  # noqa: E402
    MethodResult,
    _build_asset_truth_preview,
    _build_hybrid_coverage_scores,
    _build_method_input,
    _build_object_balanced_source_details,
    _build_senfeed_input,
    _run_quietly,
    _sort_score_dict,
    _truth_preview_map,
    create_reference_data,
    load_demo_batches,
)


ASSETS = ("BTC", "ETH", "DOGE")
METHOD_ORDER = [
    "EnhancedTruthFinder-Hybrid",
]
DEFAULT_ATTACK_TIMINGS = {
    "early": 0.10,
    "mid": 0.50,
    "late": 0.90,
}
TEXT_OBJECT_SPECS = (
    {
        "slug": "agilent",
        "company": "Agilent Technologies, Inc.",
        "symbol": "A",
        "put_base": 125.0,
        "call_base": 150.0,
        "quote_base": 8.10,
        "prob_base": 77.0,
        "phase": 0.0,
    },
    {
        "slug": "nvidia",
        "company": "NVIDIA Corporation",
        "symbol": "NVDA",
        "put_base": 118.0,
        "call_base": 142.0,
        "quote_base": 7.35,
        "prob_base": 69.0,
        "phase": 1.4,
    },
)
PRICE_OBJECT_SPECS = (
    {
        "slug": "spot",
        "base": {"BTC": 90490.0, "ETH": 3026.0, "DOGE": 0.1548},
        "phase": 0.3,
    },
    {
        "slug": "close",
        "base": {"BTC": 90180.0, "ETH": 3012.0, "DOGE": 0.1539},
        "phase": 1.1,
    },
)


@dataclass(frozen=True)
class ExperimentConfig:
    stream_source: str
    data_agent_dir: str
    start_batch: int
    agent10_bad_count: int | None
    num_rounds: int
    num_agents: int
    malicious_ratio: float
    replay_window: int
    seed_base: int
    num_seeds: int
    max_iterations: int
    convergence_threshold: float
    honest_price_noise: float
    malicious_price_bias: float
    malicious_text_shift: float
    attack_scale: float
    recovery_tolerance: float
    recovery_consecutive_rounds: int
    hybrid_numeric_weight: float
    hybrid_field_support_threshold: float
    use_sbert: bool
    output_dir: str
    skip_plots: bool


class HybridStreamingRunner:
    method_name = "EnhancedTruthFinder-Hybrid"

    def __init__(self, config: ExperimentConfig):
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

    def step(self, text_df: pd.DataFrame, epoch: int) -> MethodResult:
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
                "calibrated_score": calibrated_score,
            }
            if pd.notna(calibrated_score):
                source_scores[str(website)] = calibrated_score

        return MethodResult(
            name=self.method_name,
            result_df=result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_build_asset_truth_preview(result_df),
            elapsed_sec=elapsed,
        )


class NonHybridStreamingRunner:
    method_name = "EnhancedTruthFinder-NonHybrid"

    def __init__(self, config: ExperimentConfig):
        finder_kwargs = dict(
            alpha=0.9,
            beta=100.0,
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations,
            initial_reliability=0.5,
            numeric_weight=0.5,
            enable_hybrid_mode=False,
            enable_zk_proof=False,
        )
        self.text_finder = EnhancedTruthFinder(**finder_kwargs)
        self.numeric_finder = EnhancedTruthFinder(**finder_kwargs)

    def step(self, text_df: pd.DataFrame, epoch: int) -> MethodResult:
        expected_websites = text_df["website"].astype(str).drop_duplicates().tolist()
        mixed_record_types = text_df.get("record_type", pd.Series(dtype=object)).astype(str).nunique() > 1
        split_inputs = [
            (_build_method_input(text_df, supports_text=True, supports_numeric=False), self.text_finder),
            (_build_method_input(text_df, supports_text=False, supports_numeric=True), self.numeric_finder),
        ]

        split_results: List[MethodResult] = []
        total_elapsed = 0.0
        for input_df, finder in split_inputs:
            if input_df.empty:
                continue
            start = time.perf_counter()
            result_df = _run_quietly(lambda: finder.process_batch(input_df.copy(), epoch=epoch))
            total_elapsed += time.perf_counter() - start
            split_results.append(
                MethodResult(
                    name=self.method_name,
                    result_df=result_df,
                    source_scores={},
                    source_details={},
                    truth_preview={},
                    elapsed_sec=0.0,
                )
            )

        if not split_results:
            return MethodResult(
                name=self.method_name,
                result_df=pd.DataFrame(),
                source_scores={},
                source_details={},
                truth_preview={},
                elapsed_sec=0.0,
            )

        combined_df = pd.concat([result.result_df for result in split_results], ignore_index=True, sort=False)
        sort_columns = [
            column
            for column in ["claim_id", "parent_object", "asset", "object", "website"]
            if column in combined_df.columns
        ]
        if sort_columns:
            combined_df = combined_df.sort_values(sort_columns).reset_index(drop=True)

        source_details = _build_object_balanced_source_details(
            combined_df,
            {"r": "source_reliability", "e": "error_metric", "k": "consistency_metric"},
            expected_websites=expected_websites,
        )
        representation_fidelity = 0.85 if mixed_record_types else 1.0
        source_scores: Dict[str, float] = {}
        for website, detail in source_details.items():
            calibrated_score = float(detail.get("r", np.nan)) * representation_fidelity
            detail["representation_fidelity"] = representation_fidelity
            detail["calibrated_score"] = calibrated_score
            if pd.notna(calibrated_score):
                source_scores[website] = calibrated_score

        return MethodResult(
            name=self.method_name,
            result_df=combined_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_build_asset_truth_preview(combined_df),
            elapsed_sec=total_elapsed,
        )


class SenteReplayRunner:
    method_name = "SenteTruthFinder"

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def step(self, text_df: pd.DataFrame, epoch: int) -> MethodResult:
        del epoch
        current_objects = text_df["object"].astype(str).unique().tolist()

        def _train() -> tuple[SenteTruthFinder, pd.DataFrame]:
            finder = SenteTruthFinder(
                convergence_threshold=self.config.convergence_threshold,
                max_iterations=self.config.max_iterations,
                initial_credibility=0.5,
                use_sbert=self.config.use_sbert,
                strict_sbert=False,
                local_files_only=None,
                enable_zk_proof=False,
            )
            result_df = finder.train(
                text_df.copy(),
                max_iterations=self.config.max_iterations,
                threshold=self.config.convergence_threshold,
            )
            return finder, result_df

        start = time.perf_counter()
        finder, result_df = _run_quietly(_train)
        elapsed = time.perf_counter() - start

        details_raw = finder.get_source_reliability()
        source_details = {
            str(website): {"credibility": float(metrics.get("credibility", np.nan))}
            for website, metrics in details_raw.items()
        }
        source_scores = {
            website: detail["credibility"]
            for website, detail in source_details.items()
            if pd.notna(detail["credibility"])
        }
        current_result_df = result_df[result_df["object"].astype(str).isin(current_objects)].copy()

        return MethodResult(
            name=self.method_name,
            result_df=current_result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_truth_preview_map(current_result_df),
            elapsed_sec=elapsed,
        )


class BasicReplayRunner:
    method_name = "BasicTruthFinder"

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def step(self, text_df: pd.DataFrame, epoch: int) -> MethodResult:
        del epoch
        current_objects = text_df["object"].astype(str).unique().tolist()
        mixed_record_types = text_df.get("record_type", pd.Series(dtype=object)).astype(str).nunique() > 1

        implication = create_implication_function_from_texts(text_df["fact"].astype(str).tolist())
        finder = BasicTruthFinder(
            implication=implication,
            dampening_factor=0.3,
            influence_related=0.5,
            enable_zk_proof=False,
        )

        start = time.perf_counter()
        result_df = finder.train(
            text_df.copy(),
            max_iterations=self.config.max_iterations,
            threshold=self.config.convergence_threshold,
            initial_trustworthiness=0.7,
        )
        elapsed = time.perf_counter() - start

        source_details = _build_object_balanced_source_details(
            result_df,
            {"trustworthiness": "trustworthiness"},
            expected_websites=text_df["website"].astype(str).drop_duplicates().tolist(),
        )
        representation_fidelity = 0.75 if mixed_record_types else 1.0
        source_scores: Dict[str, float] = {}
        for website, detail in source_details.items():
            calibrated_score = float(detail.get("trustworthiness", np.nan)) * representation_fidelity
            detail["representation_fidelity"] = representation_fidelity
            detail["calibrated_score"] = calibrated_score
            if pd.notna(calibrated_score):
                source_scores[website] = calibrated_score

        current_result_df = result_df[result_df["object"].astype(str).isin(current_objects)].copy()
        return MethodResult(
            name=self.method_name,
            result_df=current_result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_truth_preview_map(current_result_df),
            elapsed_sec=elapsed,
        )


class SenFeedStreamingRunner:
    method_name = "SenFeedTruthFinder-CompositeTD"

    def __init__(self, config: ExperimentConfig):
        self.finder = SenFeedTruthDiscovery(
            mode="composite",
            gamma=1.0,
            map_shape_a=2.0,
            map_scale_beta=1.0,
            history_truth_weight=1.0,
            evolution_threshold=0.05,
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations,
            initial_weight=1.0,
            enable_zk_proof=False,
        )

    def step(self, text_df: pd.DataFrame, epoch: int) -> MethodResult:
        del epoch
        senfeed_df = _build_senfeed_input(text_df)
        if senfeed_df.empty:
            return MethodResult(
                name=self.method_name,
                result_df=senfeed_df,
                source_scores={},
                source_details={},
                truth_preview={},
                elapsed_sec=0.0,
            )

        start = time.perf_counter()
        result_df = _run_quietly(lambda: self.finder.train(senfeed_df.copy()))
        elapsed = time.perf_counter() - start

        details_raw = self.finder.get_source_reliability()
        source_details = _build_object_balanced_source_details(
            result_df,
            {"weight": "source_weight"},
            expected_websites=text_df["website"].astype(str).drop_duplicates().tolist(),
        )
        if not source_details:
            source_details = {
                str(website): {"weight": float(metrics.get("weight", np.nan))}
                for website, metrics in details_raw.items()
                if pd.notna(metrics.get("weight", np.nan))
            }
        for website, detail in source_details.items():
            detail["final_weight"] = float(details_raw.get(website, {}).get("weight", detail.get("weight", np.nan)))

        source_scores = {
            website: float(detail.get("weight", np.nan))
            for website, detail in source_details.items()
            if pd.notna(detail.get("weight", np.nan))
        }

        return MethodResult(
            name=self.method_name,
            result_df=result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_build_asset_truth_preview(result_df),
            elapsed_sec=elapsed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic attack-injection comparison across five truth-discovery methods."
    )
    parser.add_argument(
        "--stream-source",
        choices=("agent10", "synthetic"),
        default="agent10",
        help="Use repo-backed data_agent-10 batches or synthetic mixed streams.",
    )
    parser.add_argument(
        "--data-agent-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data_agent-10"),
        help="Directory containing demo.json and demo_xbad_agent.json files.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="Starting batch index for agent10 mode; later seeds slide this window forward by one batch.",
    )
    parser.add_argument(
        "--agent10-bad-count",
        type=int,
        default=None,
        help="Number of malicious agents to use in agent10 mode; defaults to ceil(10 * malicious_ratio), clipped to [1, 4].",
    )
    parser.add_argument("--num-rounds", type=int, default=50, help="Total streaming rounds.")
    parser.add_argument("--num-agents", type=int, default=10, help="Total reporting agents.")
    parser.add_argument("--malicious-ratio", type=float, default=0.4, help="Fraction of agents compromised.")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of random seeds to average.")
    parser.add_argument("--seed-base", type=int, default=20260315, help="Base seed for reproducible runs.")
    parser.add_argument(
        "--replay-window",
        type=int,
        default=12,
        help="Sliding replay window used by non-stateful baselines.",
    )
    parser.add_argument("--max-iterations", type=int, default=80, help="Per-round maximum solver iterations.")
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-6,
        help="Convergence threshold shared across methods.",
    )
    parser.add_argument(
        "--honest-price-noise",
        type=float,
        default=0.0025,
        help="Relative Gaussian noise used by honest numeric reporters.",
    )
    parser.add_argument(
        "--malicious-price-bias",
        type=float,
        default=0.12,
        help="Relative bias applied after compromise to numeric reports.",
    )
    parser.add_argument(
        "--malicious-text-shift",
        type=float,
        default=10.0,
        help="Absolute shift applied to key extracted text fields after compromise.",
    )
    parser.add_argument(
        "--attack-scale",
        type=float,
        default=4.0,
        help="Amplify repo-backed malicious deltas after compromise; 1.0 replays the original demo_xbad_agent files.",
    )
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.03,
        help="Allowed attack-induced error gap to count as recovered.",
    )
    parser.add_argument(
        "--recovery-consecutive-rounds",
        type=int,
        default=2,
        help="Consecutive recovered rounds required to declare recovery.",
    )
    parser.add_argument(
        "--hybrid-numeric-weight",
        type=float,
        default=0.38,
        help="Numeric weight used by EnhancedTruthFinder-Hybrid.",
    )
    parser.add_argument(
        "--hybrid-field-support-threshold",
        type=float,
        default=0.5,
        help="Structured support threshold used by the Hybrid finder.",
    )
    parser.add_argument("--use-sbert", action="store_true", help="Use SBERT in SenteTruthFinder.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs", "dynamic_attack_injection"),
        help="Directory for CSV/JSON/plot outputs.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        stream_source=str(args.stream_source),
        data_agent_dir=os.path.abspath(args.data_agent_dir),
        start_batch=args.start_batch,
        agent10_bad_count=args.agent10_bad_count,
        num_rounds=args.num_rounds,
        num_agents=args.num_agents,
        malicious_ratio=args.malicious_ratio,
        replay_window=args.replay_window,
        seed_base=args.seed_base,
        num_seeds=args.num_seeds,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        honest_price_noise=args.honest_price_noise,
        malicious_price_bias=args.malicious_price_bias,
        malicious_text_shift=args.malicious_text_shift,
        attack_scale=args.attack_scale,
        recovery_tolerance=args.recovery_tolerance,
        recovery_consecutive_rounds=args.recovery_consecutive_rounds,
        hybrid_numeric_weight=args.hybrid_numeric_weight,
        hybrid_field_support_threshold=args.hybrid_field_support_threshold,
        use_sbert=args.use_sbert,
        output_dir=os.path.abspath(args.output_dir),
        skip_plots=args.skip_plots,
    )


def _round_timestamp(round_index: int, slot: int = 0) -> str:
    base = datetime(2026, 1, 1, 0, 0, 0)
    timestamp = base + timedelta(minutes=((round_index - 1) * 30) + (slot * 10))
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _text_truth(spec: Dict[str, Any], round_index: int, rng: np.random.Generator) -> Dict[str, Any]:
    phase = float(spec["phase"])
    put_strike = spec["put_base"] + (0.35 * round_index) + (1.1 * math.sin((round_index / 5.0) + phase))
    call_strike = spec["call_base"] + (0.42 * round_index) + (1.3 * math.cos((round_index / 6.0) + phase))
    call_quote = spec["quote_base"] + (0.08 * math.sin((round_index / 4.0) + phase)) + rng.normal(0.0, 0.05)
    worthless_probability = spec["prob_base"] + (2.5 * math.sin((round_index / 7.0) + phase)) + rng.normal(0.0, 0.45)
    expiry_year = 2026 + ((round_index - 1) // 24)
    expiry_month = ((round_index + int(phase * 3)) % 12) + 1
    return {
        "numeric": {
            "put_strike_price": round(float(put_strike), 2),
            "call_strike_price": round(float(call_strike), 2),
            "call_quote": round(float(max(call_quote, 1.2)), 2),
            "put_worthless_probability": round(float(np.clip(worthless_probability, 52.0, 92.0)), 2),
            "expiry_year": float(expiry_year),
            "expiry_month": float(expiry_month),
        },
        "discrete": {
            "company": str(spec["company"]),
            "symbol": str(spec["symbol"]),
        },
    }


def _price_truth(spec: Dict[str, Any], round_index: int, rng: np.random.Generator) -> Dict[str, float]:
    phase = float(spec["phase"])
    base = spec["base"]
    btc = base["BTC"] + (110.0 * round_index) + (320.0 * math.sin((round_index / 6.0) + phase)) + rng.normal(0.0, 18.0)
    eth = base["ETH"] + (4.8 * round_index) + (14.0 * math.sin((round_index / 5.0) + phase)) + rng.normal(0.0, 1.4)
    doge = base["DOGE"] + (0.0009 * round_index) + (0.004 * math.sin((round_index / 4.5) + phase)) + rng.normal(0.0, 0.00025)
    return {
        "BTC": round(float(btc), 2),
        "ETH": round(float(eth), 2),
        "DOGE": round(float(max(doge, 0.02)), 6),
    }


def _render_honest_text(truth: Dict[str, Any], variant_index: int) -> str:
    numeric = truth["numeric"]
    discrete = truth["discrete"]
    company = discrete["company"]
    symbol = discrete["symbol"]
    if variant_index % 3 == 0:
        return (
            f"{company} ({symbol}) 的看跌期权行权价为 {numeric['put_strike_price']:.2f}，"
            f"看涨期权行权价为 {numeric['call_strike_price']:.2f}，当前看涨期权报价为 {numeric['call_quote']:.2f}，"
            f"看跌期权有 {numeric['put_worthless_probability']:.2f}% 的概率到期无价值，"
            f"合约将于 {int(numeric['expiry_year'])}年{int(numeric['expiry_month'])}月开始交易。"
        )
    if variant_index % 3 == 1:
        return (
            f"关于 {company} ({symbol}) 的期权摘要：看跌期权行权价为 {numeric['put_strike_price']:.2f}，"
            f"看涨期权行权价为 {numeric['call_strike_price']:.2f}，看涨期权有 {numeric['call_quote']:.2f} 的出价，"
            f"看跌期权有 {numeric['put_worthless_probability']:.2f}% 的概率到期无价值，"
            f"相关合约将于 {int(numeric['expiry_year'])}年{int(numeric['expiry_month'])}月开始交易。"
        )
    return (
        f"{company} ({symbol}) 的最新期权信息显示：看跌期权行权价为 {numeric['put_strike_price']:.2f}，"
        f"看涨期权行权价为 {numeric['call_strike_price']:.2f}，当前看涨期权报价为 {numeric['call_quote']:.2f}。"
        f"市场估计看跌期权有 {numeric['put_worthless_probability']:.2f}% 的概率到期无价值，"
        f"合约将于 {int(numeric['expiry_year'])}年{int(numeric['expiry_month'])}月开始交易。"
    )


def _render_malicious_text(truth: Dict[str, Any], variant_index: int, shift_scale: float) -> str:
    manipulated = {
        "numeric": dict(truth["numeric"]),
        "discrete": dict(truth["discrete"]),
    }
    direction = 1.0 if variant_index % 2 == 0 else -1.0
    manipulated["numeric"]["put_strike_price"] = round(
        float(manipulated["numeric"]["put_strike_price"] + (direction * shift_scale)), 2
    )
    manipulated["numeric"]["call_strike_price"] = round(
        float(manipulated["numeric"]["call_strike_price"] + (direction * (shift_scale + 3.5))), 2
    )
    manipulated["numeric"]["call_quote"] = round(
        float(max(0.5, manipulated["numeric"]["call_quote"] + (0.22 * direction * shift_scale))), 2
    )
    manipulated["numeric"]["put_worthless_probability"] = round(
        float(np.clip(manipulated["numeric"]["put_worthless_probability"] + (0.9 * direction * shift_scale), 30.0, 99.0)),
        2,
    )
    text = _render_honest_text(manipulated, variant_index=2)
    return f"{text} 市场上还出现了另一种高置信度说法，许多账号重复引用了这一版本。"


def _honest_price_observation(truth: Dict[str, float], rng: np.random.Generator, relative_noise: float) -> Dict[str, float]:
    response: Dict[str, float] = {}
    for asset in ASSETS:
        scale = 1.0 + rng.normal(0.0, relative_noise)
        response[asset] = round(float(truth[asset] * scale), 6 if asset == "DOGE" else 2)
    return response


def _malicious_price_observation(
    truth: Dict[str, float],
    rng: np.random.Generator,
    relative_noise: float,
    bias: float,
    variant_index: int,
) -> Dict[str, float]:
    direction = 1.0 if variant_index % 2 == 0 else -1.0
    response: Dict[str, float] = {}
    per_asset_multiplier = {
        "BTC": 1.0,
        "ETH": 0.8,
        "DOGE": 1.2,
    }
    for asset in ASSETS:
        attack_scale = 1.0 + (direction * bias * per_asset_multiplier[asset]) + rng.normal(0.0, relative_noise * 0.15)
        response[asset] = round(float(truth[asset] * attack_scale), 6 if asset == "DOGE" else 2)
    return response


def _make_object_id(timestamp: str, round_index: int, slug: str, object_kind: str) -> str:
    if object_kind == "text":
        return f"{timestamp} 的第{round_index}轮期权新闻摘要-{slug}"
    return f"{timestamp} 的第{round_index}轮 {slug.upper()} BTC/ETH/DOGE 价格数据"


def _select_malicious_agents(num_agents: int, malicious_ratio: float, seed: int) -> List[str]:
    malicious_count = max(1, int(math.ceil(num_agents * malicious_ratio)))
    rng = np.random.default_rng(seed)
    indices = np.arange(1, num_agents + 1)
    rng.shuffle(indices)
    selected = sorted(indices[:malicious_count])
    return [f"agent-{index}" for index in selected]


def _resolve_agent10_bad_count(config: ExperimentConfig) -> int:
    if config.agent10_bad_count is not None:
        bad_count = int(config.agent10_bad_count)
    else:
        bad_count = int(math.ceil(10 * config.malicious_ratio))
    if bad_count < 1 or bad_count > 4:
        raise ValueError(f"agent10 bad count must be in [1, 4], got: {bad_count}")
    return bad_count


def _select_batch_window(
    batches: Sequence[Dict[str, Any]],
    *,
    start_batch: int,
    num_rounds: int,
) -> List[Dict[str, Any]]:
    if start_batch < 1:
        raise ValueError(f"start_batch must be >= 1, got: {start_batch}")
    end_batch = start_batch + num_rounds - 1
    if end_batch > len(batches):
        raise ValueError(
            f"Requested batch window [{start_batch}, {end_batch}] exceeds available batches={len(batches)}"
        )
    return [json.loads(json.dumps(batch, ensure_ascii=False)) for batch in batches[start_batch - 1 : end_batch]]


def _infer_changed_agents(clean_batch: Dict[str, Any], attacked_batch: Dict[str, Any]) -> List[str]:
    clean_lookup = {
        (str(item["agent"]), str(item["object"])): item["response"]
        for item in clean_batch.get("items", [])
    }
    changed_agents = set()
    for item in attacked_batch.get("items", []):
        key = (str(item["agent"]), str(item["object"]))
        if key in clean_lookup and clean_lookup[key] != item["response"]:
            changed_agents.add(str(item["agent"]))
    return sorted(changed_agents)


def _extract_agent10_text_fields(text: Any) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"numeric": {}, "discrete": {}}

    discrete: Dict[str, Any] = {}
    company_match = re.search(r'([A-Za-z][A-Za-z0-9.,&\-\s]+?)\s*\((?:Symbol:\s*)?([A-Z]+)\)', text)
    if company_match:
        discrete["company"] = company_match.group(1).strip()
        discrete["symbol"] = company_match.group(2).strip()

    numeric: Dict[str, float] = {}
    numeric_patterns = [
        (r'看跌(?:期权)?[^。；，,]*?行权价为\s*(-?\d+(?:\.\d+)?)', 'put_strike_price'),
        (r'看涨(?:期权)?[^。；，,]*?行权价为\s*(-?\d+(?:\.\d+)?)', 'call_strike_price'),
        (r'看跌(?:期权)?[^。；，,]*?(?:当前)?(?:报价|出价)为\s*(-?\d+(?:\.\d+)?)', 'put_quote'),
        (r'看涨(?:期权)?[^。；，,]*?(?:当前)?(?:报价|出价)为\s*(-?\d+(?:\.\d+)?)', 'call_quote'),
        (r'看涨(?:期权)?[^。；，,]*?有\s*(-?\d+(?:\.\d+)?)\s*的出价', 'call_quote'),
        (r'看跌(?:期权)?[^。；，,]*?有\s*(\d+(?:\.\d+)?)%\s*的概率到期无价值', 'put_worthless_probability'),
    ]
    for pattern, field_name in numeric_patterns:
        match = re.search(pattern, text)
        if match:
            numeric[field_name] = float(match.group(1))

    expiry_match = re.search(r'(\d{4})年(\d{1,2})月(?:开始交易|到期)', text)
    if expiry_match:
        numeric["expiry_year"] = float(expiry_match.group(1))
        numeric["expiry_month"] = float(expiry_match.group(2))

    return {"numeric": numeric, "discrete": discrete}


def _round_like_reference(value: float, reference: float) -> float:
    if abs(reference) >= 1000:
        return round(value, 2)
    if abs(reference) >= 1:
        if float(reference).is_integer():
            return round(value, 2)
        return round(value, 4)
    return round(value, 6)


def _amplify_numeric_response(clean_response: Any, attacked_response: Any, attack_scale: float) -> Any:
    if attack_scale <= 1.0:
        return attacked_response
    if isinstance(clean_response, dict) and isinstance(attacked_response, dict):
        amplified: Dict[str, Any] = {}
        for key, attacked_value in attacked_response.items():
            clean_value = clean_response.get(key)
            amplified[key] = _amplify_numeric_response(clean_value, attacked_value, attack_scale)
        return amplified
    if isinstance(clean_response, (int, float)) and isinstance(attacked_response, (int, float)):
        shifted_value = float(clean_response) + (attack_scale * (float(attacked_response) - float(clean_response)))
        return _round_like_reference(shifted_value, float(attacked_response))
    return attacked_response


def _amplify_text_response(clean_response: Any, attacked_response: Any, attack_scale: float, agent_name: str) -> Any:
    if attack_scale <= 1.0 or not isinstance(clean_response, str):
        return attacked_response

    fields = _extract_agent10_text_fields(clean_response)
    numeric = fields["numeric"]
    discrete = fields["discrete"]
    agent_suffix_match = re.search(r'(\d+)$', agent_name)
    agent_variant = int(agent_suffix_match.group(1)) if agent_suffix_match else 1
    direction = -1.0 if agent_variant % 2 else 1.0

    company = str(discrete.get("company", "Agilent Technologies, Inc.")).strip()
    symbol = str(discrete.get("symbol", "A")).strip()
    put_strike = float(numeric.get("put_strike_price", 125.0))
    call_strike = float(numeric.get("call_strike_price", put_strike + 25.0))
    call_quote = float(numeric.get("call_quote", max(4.0, 0.06 * call_strike)))
    worthless_prob = float(numeric.get("put_worthless_probability", 77.0))
    expiry_year = int(round(float(numeric.get("expiry_year", 2024.0))))
    expiry_month = int(round(float(numeric.get("expiry_month", 8.0))))

    strike_shift = max(10.0, put_strike * 0.08 * attack_scale)
    call_shift = max(12.0, call_strike * 0.07 * attack_scale)
    quote_shift = max(1.8, call_quote * 0.45 * attack_scale)
    prob_shift = max(8.0, 4.5 * attack_scale)
    year_shift = max(1, int(math.ceil(attack_scale)))
    month_shift = max(1, int(math.ceil(1.5 * attack_scale)))

    bad_put = max(20.0, put_strike + (direction * strike_shift))
    bad_call = max(25.0, call_strike - (direction * call_shift))
    bad_quote = max(0.5, call_quote + (direction * quote_shift))
    bad_prob = min(99.0, max(1.0, worthless_prob + (direction * prob_shift)))
    bad_year = expiry_year + (year_shift if direction > 0 else -year_shift)
    bad_month = ((expiry_month + (month_shift if direction > 0 else -month_shift) - 1) % 12) + 1

    return (
        f"{company} (Symbol: {symbol}) 的新期权合约将在{bad_year}年{bad_month}月开始交易，"
        f"当前看跌期权的行权价为{bad_put:.2f}，看涨期权的行权价为{bad_call:.2f}，"
        f"看涨期权当前报价为{bad_quote:.2f}，看跌期权有{bad_prob:.0f}%的概率到期无价值。"
    )


def _blend_agent10_batches(
    clean_batches: Sequence[Dict[str, Any]],
    attacked_reference_batches: Sequence[Dict[str, Any]],
    malicious_agents: Sequence[str],
    attack_round: int,
    attack_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    malicious_agent_set = set(malicious_agents)
    blended_batches = json.loads(json.dumps(list(clean_batches), ensure_ascii=False))

    for round_index, batch in enumerate(blended_batches, start=1):
        if round_index < attack_round:
            continue
        attacked_lookup = {
            (str(item["agent"]), str(item["object"])): item["response"]
            for item in attacked_reference_batches[round_index - 1].get("items", [])
        }
        for item in batch.get("items", []):
            item_agent = str(item["agent"])
            if item_agent not in malicious_agent_set:
                continue
            item_key = (item_agent, str(item["object"]))
            if item_key not in attacked_lookup:
                continue
            attacked_response = attacked_lookup[item_key]
            clean_response = item["response"]
            if isinstance(attacked_response, dict):
                item["response"] = _amplify_numeric_response(clean_response, attacked_response, attack_scale)
            elif isinstance(attacked_response, str):
                item["response"] = _amplify_text_response(clean_response, attacked_response, attack_scale, item_agent)
            else:
                item["response"] = attacked_response

    return blended_batches


def load_agent10_stream(
    config: ExperimentConfig,
    *,
    window_offset: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    bad_count = _resolve_agent10_bad_count(config)
    clean_path = os.path.join(config.data_agent_dir, "demo.json")
    attacked_path = os.path.join(config.data_agent_dir, f"demo_{bad_count}bad_agent.json")

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
        "mode": "agent10",
        "clean_path": os.path.abspath(clean_path),
        "attacked_path": os.path.abspath(attacked_path),
        "window_start_batch": window_start_batch,
        "window_end_batch": window_start_batch + config.num_rounds - 1,
        "bad_agent_count": bad_count,
    }
    return clean_batches, attacked_reference_batches, truth_maps, malicious_agents, metadata


def generate_clean_stream(seed: int, config: ExperimentConfig) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rng = np.random.default_rng(seed)
    batches: List[Dict[str, Any]] = []
    truth_maps: List[Dict[str, Any]] = []
    malicious_agents = _select_malicious_agents(config.num_agents, config.malicious_ratio, seed)
    agents = [f"agent-{index}" for index in range(1, config.num_agents + 1)]

    for round_index in range(1, config.num_rounds + 1):
        batch_timestamp = _round_timestamp(round_index, 0)
        truth_map: Dict[str, Any] = {}
        items: List[Dict[str, Any]] = []

        for spec_index, spec in enumerate(TEXT_OBJECT_SPECS):
            object_id = _make_object_id(batch_timestamp, round_index, str(spec["slug"]), "text")
            truth = _text_truth(spec, round_index, rng)
            truth_map[object_id] = truth
            for agent_index, agent in enumerate(agents, start=1):
                honest_variant = (round_index + agent_index + spec_index) % 3
                items.append(
                    {
                        "agent": agent,
                        "object": object_id,
                        "response": _render_honest_text(truth, honest_variant),
                    }
                )

        for spec_index, spec in enumerate(PRICE_OBJECT_SPECS):
            timestamp = _round_timestamp(round_index, spec_index + 1)
            object_id = _make_object_id(timestamp, round_index, str(spec["slug"]), "price")
            truth = _price_truth(spec, round_index, rng)
            truth_map[object_id] = truth
            for agent in agents:
                items.append(
                    {
                        "agent": agent,
                        "object": object_id,
                        "response": _honest_price_observation(truth, rng, config.honest_price_noise),
                    }
                )

        batches.append(
            {
                "batchIndex": round_index,
                "itemCount": len(items),
                "items": items,
                "newsIndex": round_index,
                "priceTimestamp": batch_timestamp,
            }
        )
        truth_maps.append(truth_map)

    return batches, truth_maps, malicious_agents


def inject_dynamic_attack(
    clean_batches: Sequence[Dict[str, Any]],
    truth_maps: Sequence[Dict[str, Any]],
    malicious_agents: Sequence[str],
    attack_round: int,
    config: ExperimentConfig,
    seed: int,
) -> List[Dict[str, Any]]:
    if attack_round < 1:
        raise ValueError(f"attack_round must be >= 1, got {attack_round}")

    rng = np.random.default_rng(seed + (attack_round * 997))
    attacked_batches = json.loads(json.dumps(list(clean_batches), ensure_ascii=False))
    malicious_agent_set = set(malicious_agents)

    for round_offset, batch in enumerate(attacked_batches, start=1):
        if round_offset < attack_round:
            continue
        truth_map = truth_maps[round_offset - 1]
        for item in batch["items"]:
            agent = str(item["agent"])
            if agent not in malicious_agent_set:
                continue

            truth = truth_map[str(item["object"])]
            agent_variant = int(agent.split("-")[-1])
            if isinstance(item["response"], dict):
                item["response"] = _malicious_price_observation(
                    truth,
                    rng,
                    config.honest_price_noise,
                    config.malicious_price_bias,
                    variant_index=agent_variant + round_offset,
                )
                continue

            item["response"] = _render_malicious_text(
                truth,
                variant_index=agent_variant + round_offset,
                shift_scale=config.malicious_text_shift,
            )

    return attacked_batches


def _round_truth_rmse(predictions: Dict[str, Any], truth_map: Dict[str, Any]) -> float:
    object_errors: List[float] = []
    for object_id, truth in truth_map.items():
        prediction = predictions.get(object_id)
        if isinstance(truth, dict) and all(asset in truth for asset in ASSETS):
            object_errors.append(_price_error(_normalize_price_prediction(prediction), truth))
            continue
        object_errors.append(_news_error(_normalize_news_prediction(prediction), truth))

    if not object_errors:
        return 0.0
    return float(math.sqrt(np.mean(np.square(object_errors))))


def build_runners(config: ExperimentConfig) -> List[Any]:
    return [
        HybridStreamingRunner(config),
    ]


def evaluate_stream(
    batches: Sequence[Dict[str, Any]],
    truth_maps: Sequence[Dict[str, Any]],
    config: ExperimentConfig,
    *,
    seed: int,
    scenario: str,
    track: str,
    malicious_agents: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runners = build_runners(config)
    curve_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []

    for round_index, batch in enumerate(batches, start=1):
        text_df = create_reference_data(batch["items"])
        truth_map = truth_maps[round_index - 1]
        for runner in runners:
            method_result = runner.step(text_df, round_index - 1)
            error_value = _round_truth_rmse(method_result.truth_preview, truth_map)
            curve_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "track": track,
                    "round": round_index,
                    "method": method_result.name,
                    "rmse": error_value,
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
                        "is_malicious": int(website in malicious_agents),
                    }
                )

    return pd.DataFrame(curve_rows), pd.DataFrame(source_rows)


def compute_attack_metrics(
    attacked_curve: Sequence[float],
    clean_curve: Sequence[float],
    attack_round: int,
    *,
    recovery_tolerance: float,
    consecutive_rounds: int,
) -> Dict[str, float]:
    attacked = np.asarray(attacked_curve, dtype=float)
    clean = np.asarray(clean_curve, dtype=float)
    if attacked.shape != clean.shape:
        raise ValueError("attacked_curve and clean_curve must have identical shapes")
    if attack_round < 1 or attack_round > len(attacked):
        raise ValueError(f"attack_round must be in [1, {len(attacked)}], got {attack_round}")

    attack_index = attack_round - 1
    delta_curve = np.clip(attacked - clean, a_min=0.0, a_max=None)
    post_delta = delta_curve[attack_index:]
    post_attacked = attacked[attack_index:]
    post_clean = clean[attack_index:]
    max_deviation = float(np.max(post_delta)) if len(post_delta) else 0.0
    post_attack_auc = float(np.mean(post_delta)) if len(post_delta) else 0.0
    post_attack_rmse_auc = float(np.mean(post_attacked)) if len(post_attacked) else 0.0
    pre_attack_rmse = float(attacked[attack_index - 1]) if attack_index > 0 else float(clean[attack_index])
    rmse_spike_height = max(0.0, (float(np.max(post_attacked)) if len(post_attacked) else 0.0) - pre_attack_rmse)
    tail_window = min(5, len(post_delta)) if len(post_delta) else 1
    final_gap = float(np.mean(post_delta[-tail_window:])) if len(post_delta) else 0.0
    attacked_final_rmse = float(np.mean(post_attacked[-tail_window:])) if len(post_attacked) else 0.0
    clean_final_rmse = float(np.mean(post_clean[-tail_window:])) if len(post_clean) else 0.0

    recovery_time = float(len(post_delta))
    for offset in range(len(post_delta)):
        window = post_delta[offset : offset + consecutive_rounds]
        if len(window) < consecutive_rounds:
            break
        if np.all(window <= recovery_tolerance):
            recovery_time = float(offset + 1)
            break

    normalized_recovery = recovery_time / max(1, len(post_delta))
    dynamic_score = (
        (0.30 * post_attack_rmse_auc)
        + (0.25 * attacked_final_rmse)
        + (0.20 * max_deviation)
        + (0.15 * final_gap)
        + (0.10 * normalized_recovery)
    )
    return {
        "max_deviation": max_deviation,
        "post_attack_auc": post_attack_auc,
        "post_attack_rmse_auc": post_attack_rmse_auc,
        "rmse_spike_height": float(rmse_spike_height),
        "final_gap": final_gap,
        "recovery_time": recovery_time,
        "normalized_recovery": normalized_recovery,
        "dynamic_score": float(dynamic_score),
        "attacked_final_rmse": attacked_final_rmse,
        "clean_final_rmse": clean_final_rmse,
    }


def _aggregate_metrics(
    attacked_curves: pd.DataFrame,
    clean_curves: pd.DataFrame,
    attack_round_map: Dict[str, int],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    clean_lookup = clean_curves.pivot_table(index=["seed", "round"], columns="method", values="rmse")
    attacked_lookup = attacked_curves.pivot_table(index=["seed", "scenario", "round"], columns="method", values="rmse")

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


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
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
    aggregated["dynamic_score_std"] = aggregated["dynamic_score_std"].fillna(0.0)
    aggregated = aggregated.sort_values(["scenario", "dynamic_score_mean", "post_attack_auc_mean", "recovery_time_mean"])
    aggregated["rank"] = aggregated.groupby("scenario").cumcount() + 1
    return aggregated


def _curve_summary(curves_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curves_df.groupby(["scenario", "track", "round", "method"], sort=False)["rmse"]
        .agg([("rmse_mean", "mean"), ("rmse_std", "std")])
        .reset_index()
        .fillna({"rmse_std": 0.0})
    )


def _source_summary(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return source_df
    return (
        source_df.groupby(["scenario", "track", "round", "method", "website", "is_malicious"], sort=False)["score"]
        .agg([("score_mean", "mean"), ("score_std", "std")])
        .reset_index()
        .fillna({"score_std": 0.0})
    )


def compute_source_shock_tables(
    source_df: pd.DataFrame,
    attack_round_map: Dict[str, int],
    *,
    consecutive_rounds: int,
    default_method: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    working = source_df.copy()
    if "method" not in working.columns:
        working["method"] = default_method or "EnhancedTruthFinder-Hybrid"
    group_cols = ["seed", "scenario", "track", "method", "website"]
    working = working.sort_values(group_cols + ["round"]).reset_index(drop=True)
    working["score_prev"] = working.groupby(group_cols, sort=False)["score"].shift(1)
    working["score_shock"] = (working["score"] - working["score_prev"]).abs().fillna(0.0)

    seed_round_df = (
        working.groupby(["seed", "scenario", "track", "round", "method"], sort=False)
        .agg(
            score_shock_mean=("score_shock", "mean"),
            score_shock_max=("score_shock", "max"),
        )
        .reset_index()
    )

    clean_lookup = seed_round_df[seed_round_df["track"] == "clean"][
        ["seed", "round", "method", "score_shock_mean", "score_shock_max"]
    ].rename(
        columns={
            "score_shock_mean": "clean_score_shock_mean",
            "score_shock_max": "clean_score_shock_max",
        }
    )
    seed_round_df = seed_round_df.merge(clean_lookup, on=["seed", "round", "method"], how="left")
    seed_round_df["clean_score_shock_mean"] = seed_round_df["clean_score_shock_mean"].fillna(0.0)
    seed_round_df["clean_score_shock_max"] = seed_round_df["clean_score_shock_max"].fillna(0.0)
    seed_round_df["attack_induced_shock"] = 0.0
    attacked_mask = seed_round_df["track"] == "attacked"
    seed_round_df.loc[attacked_mask, "attack_induced_shock"] = np.clip(
        seed_round_df.loc[attacked_mask, "score_shock_mean"]
        - seed_round_df.loc[attacked_mask, "clean_score_shock_mean"],
        a_min=0.0,
        a_max=None,
    )

    round_summary_df = (
        seed_round_df.groupby(["scenario", "track", "round", "method"], sort=False)
        .agg(
            score_shock_mean=("score_shock_mean", "mean"),
            score_shock_std=("score_shock_mean", "std"),
            attack_induced_shock_mean=("attack_induced_shock", "mean"),
            attack_induced_shock_std=("attack_induced_shock", "std"),
        )
        .reset_index()
        .fillna({"score_shock_std": 0.0, "attack_induced_shock_std": 0.0})
    )

    rows: List[Dict[str, Any]] = []
    attacked_seed_df = seed_round_df[seed_round_df["track"] == "attacked"]
    for (seed, scenario, method), scenario_df in attacked_seed_df.groupby(["seed", "scenario", "method"], sort=False):
        attack_round = attack_round_map.get(str(scenario))
        if attack_round is None:
            continue
        post_df = scenario_df[scenario_df["round"] >= attack_round].copy()
        if post_df.empty:
            continue
        shock_values = post_df["attack_induced_shock"].to_numpy(dtype=float)
        peak_offset = int(np.argmax(shock_values)) if len(shock_values) else 0
        peak_value = float(shock_values[peak_offset]) if len(shock_values) else 0.0
        peak_round = int(post_df.iloc[peak_offset]["round"]) if len(post_df) else attack_round
        shock_auc = float(np.mean(shock_values)) if len(shock_values) else 0.0
        tail_window = min(5, len(shock_values)) if len(shock_values) else 1
        shock_tail = float(np.mean(shock_values[-tail_window:])) if len(shock_values) else 0.0
        recovery_threshold = max(0.01, peak_value * 0.2)
        if peak_value <= 1e-12:
            recovery_time = 1.0
        else:
            recovery_time = float(len(shock_values))
            for offset in range(len(shock_values)):
                window = shock_values[offset : offset + consecutive_rounds]
                if len(window) < consecutive_rounds:
                    break
                if np.all(window <= recovery_threshold):
                    recovery_time = float(offset + 1)
                    break
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "attack_round": attack_round,
                "score_shock_peak": peak_value,
                "score_shock_auc": shock_auc,
                "score_shock_tail": shock_tail,
                "score_shock_peak_round": peak_round,
                "score_shock_recovery_time": recovery_time,
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return seed_round_df, round_summary_df, metrics_df

    summary_df = (
        metrics_df.groupby(["scenario", "method"], sort=False)
        .agg(
            score_shock_peak_mean=("score_shock_peak", "mean"),
            score_shock_peak_std=("score_shock_peak", "std"),
            score_shock_auc_mean=("score_shock_auc", "mean"),
            score_shock_tail_mean=("score_shock_tail", "mean"),
            score_shock_peak_round_mean=("score_shock_peak_round", "mean"),
            score_shock_recovery_time_mean=("score_shock_recovery_time", "mean"),
        )
        .reset_index()
    )
    summary_df["score_shock_peak_std"] = summary_df["score_shock_peak_std"].fillna(0.0)
    return seed_round_df, round_summary_df, summary_df


def _prepare_matplotlib(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".mplconfig"))
    import matplotlib.pyplot as plt

    return plt


def plot_scenario_curves(curve_summary: pd.DataFrame, attack_round_map: Dict[str, int], output_dir: str):
    plt = _prepare_matplotlib(output_dir)
    for scenario, attack_round in attack_round_map.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        attacked_df = curve_summary[(curve_summary["scenario"] == scenario) & (curve_summary["track"] == "attacked")]
        clean_df = curve_summary[(curve_summary["scenario"] == "clean") & (curve_summary["track"] == "clean")]
        for method in METHOD_ORDER:
            method_attacked = attacked_df[attacked_df["method"] == method]
            method_clean = clean_df[clean_df["method"] == method]
            if method_attacked.empty:
                continue
            line = ax.plot(method_attacked["round"], method_attacked["rmse_mean"], label=method)[0]
            ax.fill_between(
                method_attacked["round"],
                method_attacked["rmse_mean"] - method_attacked["rmse_std"],
                method_attacked["rmse_mean"] + method_attacked["rmse_std"],
                alpha=0.12,
                color=line.get_color(),
            )
            if not method_clean.empty:
                ax.plot(
                    method_clean["round"],
                    method_clean["rmse_mean"],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.35,
                    color=line.get_color(),
                )
        ax.axvline(attack_round, linestyle="--", color="black", linewidth=1.0)
        ax.text(attack_round + 0.25, ax.get_ylim()[1] * 0.94, "Attack Injected", fontsize=9)
        ax.set_title(f"Dynamic Attack Injection ({scenario.capitalize()})")
        ax.set_xlabel("Iteration Round")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"dynamic_attack_{scenario}.png"), dpi=200)
        plt.close(fig)


def plot_hybrid_timing(curve_summary: pd.DataFrame, attack_round_map: Dict[str, int], output_dir: str):
    plt = _prepare_matplotlib(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hybrid_clean = curve_summary[
        (curve_summary["scenario"] == "clean")
        & (curve_summary["track"] == "clean")
        & (curve_summary["method"] == "EnhancedTruthFinder-Hybrid")
    ]
    ax.plot(
        hybrid_clean["round"],
        hybrid_clean["rmse_mean"],
        linestyle="--",
        linewidth=1.6,
        color="grey",
        label="Hybrid Clean Baseline",
    )
    color_map = {"early": "#d62728", "mid": "#ff7f0e", "late": "#2ca02c"}
    for scenario, attack_round in attack_round_map.items():
        scenario_df = curve_summary[
            (curve_summary["scenario"] == scenario)
            & (curve_summary["track"] == "attacked")
            & (curve_summary["method"] == "EnhancedTruthFinder-Hybrid")
        ]
        ax.plot(
            scenario_df["round"],
            scenario_df["rmse_mean"],
            linewidth=2.0,
            color=color_map.get(scenario, None),
            label=f"Hybrid {scenario.capitalize()} (T_attack={attack_round})",
        )
        ax.axvline(attack_round, linestyle=":", color=color_map.get(scenario, None), alpha=0.7)
    ax.set_title("Hybrid under Early / Mid / Late Dynamic Injection")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hybrid_attack_timing.png"), dpi=200)
    plt.close(fig)



def plot_hybrid_weight_shock_timing(shock_round_summary: pd.DataFrame, attack_round_map: Dict[str, int], output_dir: str):
    if shock_round_summary.empty:
        return
    plt = _prepare_matplotlib(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clean_df = shock_round_summary[
        (shock_round_summary["scenario"] == "clean")
        & (shock_round_summary["track"] == "clean")
        & (shock_round_summary["method"] == "EnhancedTruthFinder-Hybrid")
    ]
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
        scenario_df = shock_round_summary[
            (shock_round_summary["scenario"] == scenario)
            & (shock_round_summary["track"] == "attacked")
            & (shock_round_summary["method"] == "EnhancedTruthFinder-Hybrid")
        ]
        if scenario_df.empty:
            continue
        color = color_map.get(scenario)
        ax.plot(
            scenario_df["round"],
            scenario_df["attack_induced_shock_mean"],
            linewidth=2.0,
            color=color,
            label=f"Hybrid {scenario.capitalize()} shock",
        )
        ax.fill_between(
            scenario_df["round"],
            scenario_df["attack_induced_shock_mean"] - scenario_df["attack_induced_shock_std"],
            scenario_df["attack_induced_shock_mean"] + scenario_df["attack_induced_shock_std"],
            alpha=0.12,
            color=color,
        )
        ax.axvline(attack_round, linestyle=":", color=color, alpha=0.7)
    ax.set_title("Hybrid attack-induced weight shock")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("Attack-induced mean |Δ weight|")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hybrid_weight_shock_timing.png"), dpi=200)
    plt.close(fig)



def plot_hybrid_weight_heatmap(source_summary: pd.DataFrame, output_dir: str, scenario: str = "mid"):
    plt = _prepare_matplotlib(output_dir)
    hybrid_df = source_summary[
        (source_summary["scenario"] == scenario)
        & (source_summary["track"] == "attacked")
        & (source_summary["method"] == "EnhancedTruthFinder-Hybrid")
    ]
    if hybrid_df.empty:
        return
    ordered = hybrid_df.sort_values(["is_malicious", "website"]).drop_duplicates(subset=["website"]).sort_values(
        ["is_malicious", "website"],
        ascending=[True, True],
    )
    website_order = ordered["website"].tolist()
    heatmap_df = hybrid_df.pivot(index="website", columns="round", values="score_mean").reindex(website_order)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(website_order))))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"Hybrid source-weight evolution ({scenario.capitalize()})")
    ax.set_xlabel("Iteration Round")
    ax.set_ylabel("Agent")
    ax.set_xticks(range(0, len(heatmap_df.columns), max(1, len(heatmap_df.columns) // 10)))
    ax.set_xticklabels(heatmap_df.columns[:: max(1, len(heatmap_df.columns) // 10)])
    ax.set_yticks(range(len(website_order)))
    ax.set_yticklabels(website_order)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Reliability score")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"hybrid_weight_heatmap_{scenario}.png"), dpi=200)
    plt.close(fig)


def _attack_round_map(num_rounds: int) -> Dict[str, int]:
    return {
        scenario: max(2, min(num_rounds, int(round(num_rounds * fraction))))
        for scenario, fraction in DEFAULT_ATTACK_TIMINGS.items()
    }


def run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    attack_round_map = _attack_round_map(config.num_rounds)
    clean_curve_frames: List[pd.DataFrame] = []
    clean_source_frames: List[pd.DataFrame] = []
    attacked_curve_frames: List[pd.DataFrame] = []
    attacked_source_frames: List[pd.DataFrame] = []
    seed_manifest: List[Dict[str, Any]] = []

    for seed_offset in range(config.num_seeds):
        seed = config.seed_base + seed_offset
        if config.stream_source == "agent10":
            clean_batches, attacked_reference_batches, truth_maps, malicious_agents, run_metadata = load_agent10_stream(
                config,
                window_offset=seed_offset,
            )
        else:
            clean_batches, truth_maps, malicious_agents = generate_clean_stream(seed, config)
            attacked_reference_batches = []
            run_metadata = {
                "mode": "synthetic",
                "window_start_batch": None,
                "window_end_batch": None,
                "bad_agent_count": len(malicious_agents),
            }
        seed_manifest.append({"seed": seed, "malicious_agents": malicious_agents, **run_metadata})

        clean_curves, clean_sources = evaluate_stream(
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
            if config.stream_source == "agent10":
                attacked_batches = _blend_agent10_batches(
                    clean_batches,
                    attacked_reference_batches,
                    malicious_agents,
                    attack_round,
                    attack_scale=config.attack_scale,
                )
            else:
                attacked_batches = inject_dynamic_attack(
                    clean_batches,
                    truth_maps,
                    malicious_agents,
                    attack_round,
                    config,
                    seed,
                )
            attacked_curves, attacked_sources = evaluate_stream(
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
    metrics_df = _aggregate_metrics(attacked_curves_df, clean_curves_df, attack_round_map, config)
    source_shock_seed_df, source_shock_round_df, source_shock_summary_df = compute_source_shock_tables(
        pd.concat([clean_sources_df, attacked_sources_df], ignore_index=True),
        attack_round_map,
        consecutive_rounds=config.recovery_consecutive_rounds,
    )
    summary_df = _summarize_metrics(metrics_df)
    if not source_shock_summary_df.empty:
        summary_df = summary_df.merge(source_shock_summary_df, on=["scenario", "method"], how="left")
    manifest = {
        "attack_rounds": attack_round_map,
        "seed_manifest": seed_manifest,
        "config": {
            key: value
            for key, value in config.__dict__.items()
            if key != "output_dir"
        },
    }
    curves_df = pd.concat([clean_curves_df, attacked_curves_df], ignore_index=True)
    sources_df = pd.concat([clean_sources_df, attacked_sources_df], ignore_index=True)
    return curves_df, sources_df, metrics_df, summary_df, source_shock_seed_df, source_shock_round_df, manifest


def _print_summary(summary_df: pd.DataFrame, attack_round_map: Dict[str, int]):
    print("\nHybrid-only mixed dynamic attack summary")
    print(
        f"Attack rounds: early={attack_round_map['early']}, mid={attack_round_map['mid']}, late={attack_round_map['late']}"
    )
    for scenario in ["early", "mid", "late"]:
        scenario_df = summary_df[summary_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue
        print(f"\nScenario: {scenario}")
        display_df = scenario_df[
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
        ].copy()
        print(display_df.round(4).to_string(index=False))


def main():
    args = parse_args()
    config = _build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    curves_df, sources_df, metrics_df, summary_df, source_shock_seed_df, source_shock_round_df, manifest = run_experiment(config)
    curve_summary = _curve_summary(curves_df)
    source_summary = _source_summary(sources_df)
    attack_round_map = manifest["attack_rounds"]

    curves_df.to_csv(os.path.join(config.output_dir, "dynamic_attack_curves.csv"), index=False)
    curve_summary.to_csv(os.path.join(config.output_dir, "dynamic_attack_curve_summary.csv"), index=False)
    sources_df.to_csv(os.path.join(config.output_dir, "dynamic_attack_source_scores.csv"), index=False)
    source_summary.to_csv(os.path.join(config.output_dir, "dynamic_attack_source_summary.csv"), index=False)
    source_shock_seed_df.to_csv(os.path.join(config.output_dir, "dynamic_attack_source_shock_seed_round.csv"), index=False)
    source_shock_round_df.to_csv(os.path.join(config.output_dir, "混合-随着攻击时间变化.csv"), index=False)
    metrics_df.to_csv(os.path.join(config.output_dir, "dynamic_attack_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(config.output_dir, "dynamic_attack_summary.csv"), index=False)
    with open(os.path.join(config.output_dir, "dynamic_attack_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if not config.skip_plots:
        plot_scenario_curves(curve_summary, attack_round_map, config.output_dir)
        plot_hybrid_timing(curve_summary, attack_round_map, config.output_dir)
        plot_hybrid_weight_shock_timing(source_shock_round_df, attack_round_map, config.output_dir)
        plot_hybrid_weight_heatmap(source_summary, config.output_dir, scenario="mid")

    _print_summary(summary_df, attack_round_map)
    print(f"Stream source: {config.stream_source}")
    print(f"\nOutputs written to: {config.output_dir}")


if __name__ == "__main__":
    main()
