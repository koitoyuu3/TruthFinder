import unittest

from decentralized_truth_discovery.examples.dynamic_attack_injection_comparison import METHOD_ORDER
from decentralized_truth_discovery.examples.numeric_attack_injection_comparison import (
    NumericAttackConfig,
    run_experiment,
)


class TestNumericAttackInjectionComparison(unittest.TestCase):
    @staticmethod
    def _config() -> NumericAttackConfig:
        return NumericAttackConfig(
            data_agent_dir="decentralized_truth_discovery/data_agent-10",
            start_batch=1,
            num_rounds=2,
            num_seeds=1,
            bad_agent_count=1,
            replay_window=2,
            max_iterations=2,
            convergence_threshold=1e-6,
            hybrid_numeric_weight=0.5,
            hybrid_field_support_threshold=0.5,
            attack_scale=1.0,
            recovery_tolerance=0.01,
            recovery_consecutive_rounds=1,
            use_sbert=False,
            output_dir="/tmp/numeric_attack_injection_test",
            skip_plots=True,
        )

    def test_run_experiment_emits_all_five_methods(self):
        curves_df, sources_df, metrics_df, summary_df, _shock_seed_df, shock_round_df, manifest = run_experiment(
            self._config()
        )

        self.assertEqual(sorted(curves_df["method"].unique().tolist()), sorted(METHOD_ORDER))
        self.assertEqual(sorted(sources_df["method"].unique().tolist()), sorted(METHOD_ORDER))
        self.assertEqual(sorted(metrics_df["method"].unique().tolist()), sorted(METHOD_ORDER))
        self.assertEqual(sorted(summary_df["method"].unique().tolist()), sorted(METHOD_ORDER))
        self.assertEqual(sorted(shock_round_df["method"].unique().tolist()), sorted(METHOD_ORDER))
        self.assertIn("attack_rounds", manifest)


if __name__ == "__main__":
    unittest.main()
