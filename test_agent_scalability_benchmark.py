import unittest

import pandas as pd

from decentralized_truth_discovery.examples.agent_scalability_benchmark import (
    HYBRID_METHOD_NAME,
    _build_workload_specs,
    _compute_group_scores,
    _evaluate_hybrid_requirement,
    _select_agent_subsets,
)


class TestAgentScalabilityBenchmark(unittest.TestCase):
    def test_build_workload_specs_uses_non_overlapping_windows(self):
        specs = _build_workload_specs(
            total_batches=15,
            start_batch=2,
            batch_group_size=4,
            num_workloads=3,
        )

        self.assertEqual(len(specs), 3)
        self.assertEqual((specs[0].batch_start, specs[0].batch_end), (2, 5))
        self.assertEqual((specs[1].batch_start, specs[1].batch_end), (6, 9))
        self.assertEqual((specs[2].batch_start, specs[2].batch_end), (10, 13))

    def test_select_agent_subsets_returns_unique_random_subsets(self):
        subsets = _select_agent_subsets(
            [f"agent-{index}" for index in range(1, 7)],
            agent_count=3,
            subset_samples=4,
            base_seed=123,
            workload_index=1,
            strategy="random",
        )

        self.assertEqual(len(subsets), 4)
        self.assertEqual(len({tuple(subset) for subset in subsets}), 4)
        self.assertTrue(all(len(subset) == 3 for subset in subsets))

    def test_hybrid_requirement_passes_when_hybrid_rank_is_middle_upper(self):
        raw_results_df = pd.DataFrame(
            [
                {
                    "workload_index": 1,
                    "agent_count": 4,
                    "subset_index": 1,
                    "method": HYBRID_METHOD_NAME,
                    "success": True,
                    "elapsed_sec": 1.20,
                    "peak_delta_rss_mb": 18.0,
                    "mean_source_score_pct": 88.0,
                },
                {
                    "workload_index": 1,
                    "agent_count": 4,
                    "subset_index": 1,
                    "method": "EnhancedTruthFinder-NonHybrid",
                    "success": True,
                    "elapsed_sec": 1.60,
                    "peak_delta_rss_mb": 20.0,
                    "mean_source_score_pct": 82.0,
                },
                {
                    "workload_index": 1,
                    "agent_count": 4,
                    "subset_index": 1,
                    "method": "SenteTruthFinder",
                    "success": True,
                    "elapsed_sec": 2.10,
                    "peak_delta_rss_mb": 35.0,
                    "mean_source_score_pct": 90.0,
                },
                {
                    "workload_index": 1,
                    "agent_count": 4,
                    "subset_index": 1,
                    "method": "BasicTruthFinder",
                    "success": True,
                    "elapsed_sec": 0.90,
                    "peak_delta_rss_mb": 15.0,
                    "mean_source_score_pct": 60.0,
                },
                {
                    "workload_index": 1,
                    "agent_count": 4,
                    "subset_index": 1,
                    "method": "SenFeedTruthFinder-CompositeTD",
                    "success": True,
                    "elapsed_sec": 1.40,
                    "peak_delta_rss_mb": 19.0,
                    "mean_source_score_pct": 70.0,
                },
            ]
        )

        scored_df = _compute_group_scores(
            raw_results_df,
            time_weight=0.45,
            memory_weight=0.35,
            quality_weight=0.20,
        )
        overall_summary_df = (
            scored_df.groupby("method", sort=False)
            .agg(overall_mean_composite_rank=("composite_rank", "mean"))
            .reset_index()
        )

        result = _evaluate_hybrid_requirement(
            overall_summary_df,
            hybrid_method_name=HYBRID_METHOD_NAME,
            top_k=3,
        )

        self.assertTrue(result["passed"])
        self.assertLessEqual(result["hybrid_average_composite_rank"], 3.0)


if __name__ == "__main__":
    unittest.main()
