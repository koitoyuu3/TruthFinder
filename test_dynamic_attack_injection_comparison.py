import unittest

import pandas as pd

from decentralized_truth_discovery.examples.agent10_numeric_hybrid_supplement import (
    NumericHybridConfig,
    compute_numeric_rmse,
    load_numeric_stream,
)
from decentralized_truth_discovery.examples.dynamic_attack_injection_comparison import (
    METHOD_ORDER,
    ExperimentConfig,
    _blend_agent10_batches,
    _round_truth_rmse,
    build_runners,
    compute_attack_metrics,
    compute_source_shock_tables,
    generate_clean_stream,
    inject_dynamic_attack,
    load_agent10_stream,
)


class TestDynamicAttackInjectionComparison(unittest.TestCase):
    @staticmethod
    def _config() -> ExperimentConfig:
        return ExperimentConfig(
            stream_source="synthetic",
            data_agent_dir="decentralized_truth_discovery/data_agent-10",
            start_batch=1,
            agent10_bad_count=None,
            num_rounds=6,
            num_agents=4,
            malicious_ratio=0.5,
            replay_window=3,
            seed_base=1234,
            num_seeds=1,
            max_iterations=10,
            convergence_threshold=1e-6,
            honest_price_noise=0.001,
            malicious_price_bias=0.08,
            malicious_text_shift=6.0,
            attack_scale=4.0,
            recovery_tolerance=0.03,
            recovery_consecutive_rounds=2,
            hybrid_numeric_weight=0.38,
            hybrid_field_support_threshold=0.5,
            use_sbert=False,
            output_dir="/tmp/dynamic_attack_test",
            skip_plots=True,
        )

    @staticmethod
    def _numeric_config() -> NumericHybridConfig:
        return NumericHybridConfig(
            data_agent_dir="decentralized_truth_discovery/data_agent-10",
            start_batch=1,
            num_rounds=4,
            num_seeds=1,
            bad_agent_count=3,
            max_iterations=10,
            convergence_threshold=1e-6,
            hybrid_numeric_weight=0.5,
            hybrid_field_support_threshold=0.5,
            attack_scale=4.0,
            recovery_tolerance=0.01,
            recovery_consecutive_rounds=2,
            output_dir="/tmp/numeric_hybrid_test",
            skip_plots=True,
        )

    def test_mixed_dynamic_script_builds_all_five_methods(self):
        runners = build_runners(self._config())
        self.assertEqual(
            METHOD_ORDER,
            [
                "EnhancedTruthFinder-Hybrid",
                "EnhancedTruthFinder-NonHybrid",
                "SenteTruthFinder",
                "BasicTruthFinder",
                "SenFeedTruthFinder-CompositeTD",
            ],
        )
        self.assertEqual([runner.method_name for runner in runners], METHOD_ORDER)

    def test_load_agent10_stream_infers_malicious_agents_from_repo_files(self):
        config = ExperimentConfig(
            stream_source="agent10",
            data_agent_dir="decentralized_truth_discovery/data_agent-10",
            start_batch=1,
            agent10_bad_count=2,
            num_rounds=4,
            num_agents=10,
            malicious_ratio=0.2,
            replay_window=3,
            seed_base=1234,
            num_seeds=1,
            max_iterations=10,
            convergence_threshold=1e-6,
            honest_price_noise=0.001,
            malicious_price_bias=0.08,
            malicious_text_shift=6.0,
            attack_scale=4.0,
            recovery_tolerance=0.03,
            recovery_consecutive_rounds=2,
            hybrid_numeric_weight=0.38,
            hybrid_field_support_threshold=0.5,
            use_sbert=False,
            output_dir="/tmp/dynamic_attack_test",
            skip_plots=True,
        )
        clean_batches, attacked_batches, truth_maps, malicious_agents, metadata = load_agent10_stream(
            config,
            window_offset=0,
        )

        self.assertEqual(len(clean_batches), 4)
        self.assertEqual(len(attacked_batches), 4)
        self.assertEqual(len(truth_maps), 4)
        self.assertEqual(malicious_agents, ["agent-1", "agent-2"])
        self.assertEqual(metadata["bad_agent_count"], 2)

    def test_load_numeric_stream_infers_numeric_bad_agents(self):
        clean_batches, attacked_batches, truth_maps, malicious_agents, metadata = load_numeric_stream(
            self._numeric_config(),
            window_offset=0,
        )

        self.assertEqual(len(clean_batches), 4)
        self.assertEqual(len(attacked_batches), 4)
        self.assertEqual(len(truth_maps), 4)
        self.assertEqual(malicious_agents, ["agent-1", "agent-2", "agent-3"])
        self.assertEqual(metadata["bad_agent_count"], 3)
        self.assertTrue(all(len(batch["items"]) == 10 for batch in clean_batches))

    def test_attack_injection_only_changes_compromised_agents_after_attack_round(self):
        config = self._config()
        clean_batches, truth_maps, malicious_agents = generate_clean_stream(1234, config)
        attacked_batches = inject_dynamic_attack(
            clean_batches,
            truth_maps,
            malicious_agents,
            attack_round=4,
            config=config,
            seed=1234,
        )

        clean_before = clean_batches[2]["items"]
        attacked_before = attacked_batches[2]["items"]
        self.assertEqual(clean_before, attacked_before)

        clean_after = clean_batches[3]["items"]
        attacked_after = attacked_batches[3]["items"]
        changed_agents = {
            attacked_item["agent"]
            for clean_item, attacked_item in zip(clean_after, attacked_after)
            if clean_item["response"] != attacked_item["response"]
        }
        self.assertTrue(changed_agents)
        self.assertTrue(changed_agents.issubset(set(malicious_agents)))

    def test_agent10_blend_attack_scale_amplifies_numeric_and_text_payloads(self):
        config = ExperimentConfig(
            stream_source="agent10",
            data_agent_dir="decentralized_truth_discovery/data_agent-10",
            start_batch=1,
            agent10_bad_count=4,
            num_rounds=4,
            num_agents=10,
            malicious_ratio=0.4,
            replay_window=3,
            seed_base=1234,
            num_seeds=1,
            max_iterations=10,
            convergence_threshold=1e-6,
            honest_price_noise=0.001,
            malicious_price_bias=0.08,
            malicious_text_shift=6.0,
            attack_scale=4.0,
            recovery_tolerance=0.03,
            recovery_consecutive_rounds=2,
            hybrid_numeric_weight=0.38,
            hybrid_field_support_threshold=0.5,
            use_sbert=False,
            output_dir="/tmp/dynamic_attack_test",
            skip_plots=True,
        )
        clean_batches, attacked_batches, _truth_maps, malicious_agents, _metadata = load_agent10_stream(
            config,
            window_offset=0,
        )
        blended_batches = _blend_agent10_batches(
            clean_batches,
            attacked_batches,
            malicious_agents,
            attack_round=1,
            attack_scale=4.0,
        )

        clean_price = next(
            item["response"]
            for item in clean_batches[0]["items"]
            if item["agent"] == "agent-1" and isinstance(item["response"], dict)
        )
        attacked_price = next(
            item["response"]
            for item in attacked_batches[0]["items"]
            if item["agent"] == "agent-1" and isinstance(item["response"], dict)
        )
        blended_price = next(
            item["response"]
            for item in blended_batches[0]["items"]
            if item["agent"] == "agent-1" and isinstance(item["response"], dict)
        )
        self.assertGreater(
            abs(blended_price["BTC"] - clean_price["BTC"]),
            abs(attacked_price["BTC"] - clean_price["BTC"]),
        )

        blended_text = next(
            item["response"]
            for item in blended_batches[0]["items"]
            if item["agent"] == "agent-1" and item["object"] == "第1条新闻简短摘要"
        )
        self.assertIn("看跌期权的行权价为", blended_text)
        self.assertIn("看涨期权当前报价为", blended_text)

    def test_round_truth_rmse_is_zero_for_perfect_prediction(self):
        truth_map = {
            "obj-news": {
                "numeric": {
                    "put_strike_price": 125.0,
                    "call_strike_price": 150.0,
                    "call_quote": 8.1,
                    "put_worthless_probability": 77.0,
                    "expiry_year": 2026.0,
                    "expiry_month": 3.0,
                },
                "discrete": {"company": "Agilent Technologies, Inc.", "symbol": "A"},
            },
            "obj-price": {"BTC": 90490.0, "ETH": 3026.0, "DOGE": 0.1548},
        }
        prediction = {
            "obj-news": {
                "numeric": dict(truth_map["obj-news"]["numeric"]),
                "discrete": dict(truth_map["obj-news"]["discrete"]),
            },
            "obj-price": dict(truth_map["obj-price"]),
        }
        self.assertAlmostEqual(_round_truth_rmse(prediction, truth_map), 0.0, places=9)

    def test_compute_numeric_rmse_is_zero_for_perfect_price_prediction(self):
        truth_map = {
            "obj-price": {"BTC": 90490.0, "ETH": 3026.0, "DOGE": 0.1548},
        }
        prediction = {
            "obj-price": {"BTC": 90490.0, "ETH": 3026.0, "DOGE": 0.1548},
        }
        self.assertAlmostEqual(compute_numeric_rmse(prediction, truth_map), 0.0, places=9)

    def test_compute_attack_metrics_detects_recovery(self):
        clean_curve = [0.08, 0.07, 0.06, 0.06, 0.05, 0.05]
        attacked_curve = [0.08, 0.07, 0.06, 0.19, 0.07, 0.05]
        metrics = compute_attack_metrics(
            attacked_curve,
            clean_curve,
            attack_round=4,
            recovery_tolerance=0.03,
            consecutive_rounds=2,
        )
        self.assertGreater(metrics["max_deviation"], 0.1)
        self.assertEqual(metrics["recovery_time"], 2.0)
        self.assertGreater(metrics["dynamic_score"], 0.0)
        self.assertGreater(metrics["rmse_spike_height"], 0.1)

    def test_compute_source_shock_tables_captures_spike_then_recovery(self):
        source_df = pd.DataFrame(
            [
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 1, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.60},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 1, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.58},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 2, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.61},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 2, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.59},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 3, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.60},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 3, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.58},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 4, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.60},
                {"seed": 1, "scenario": "clean", "track": "clean", "round": 4, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.58},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 1, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.60},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 1, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.58},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 2, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.61},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 2, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.59},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 3, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.20},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 3, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.90},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 4, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-1", "score": 0.56},
                {"seed": 1, "scenario": "mid", "track": "attacked", "round": 4, "method": "EnhancedTruthFinder-Hybrid", "website": "agent-2", "score": 0.61},
            ]
        )
        shock_seed_df, shock_round_df, shock_summary_df = compute_source_shock_tables(
            source_df,
            {"mid": 3},
            consecutive_rounds=1,
        )
        self.assertIn("attack_induced_shock", shock_seed_df.columns)
        mid_summary = shock_summary_df.iloc[0]
        self.assertGreater(mid_summary["score_shock_peak_mean"], 0.2)
        self.assertEqual(mid_summary["score_shock_peak_round_mean"], 3.0)
        self.assertEqual(mid_summary["score_shock_recovery_time_mean"], 2.0)
        mid_round = shock_round_df[(shock_round_df["scenario"] == "mid") & (shock_round_df["round"] == 3)]
        self.assertGreater(float(mid_round.iloc[0]["attack_induced_shock_mean"]), 0.2)


if __name__ == "__main__":
    unittest.main()
