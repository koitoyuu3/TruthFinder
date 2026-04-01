import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import erfc

from decentralized_truth_discovery.examples.exp_utils import get_numeric_demo_files
from decentralized_truth_discovery.core import (
    BasicTruthFinder,
    EnhancedTruthFinder,
    SemanticTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
)
from decentralized_truth_discovery.core.hybrid_text_alignment import (
    DEFAULT_NUMERIC_MATCH_TOL,
    DEFAULT_SEMANTIC_NUMERIC_ALPHA,
)


class TestSenteTruthPaperAlignment(unittest.TestCase):
    def test_similarity_matrix_keeps_raw_cosine_values(self):
        finder = SenteTruthFinder(use_sbert=False, enable_zk_proof=False)
        embeddings = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])

        similarity_matrix = finder._calculate_semantic_similarity_matrix(embeddings)

        self.assertAlmostEqual(similarity_matrix[0, 1], -1.0)
        self.assertAlmostEqual(similarity_matrix[0, 2], 0.0)
        self.assertAlmostEqual(similarity_matrix[1, 2], 0.0)

    def test_credibility_update_matches_paper_example(self):
        finder = SenteTruthFinder(use_sbert=False, enable_zk_proof=False)
        finder.node_credibility = {
            'n1': 0.7,
            'n2': 0.5,
            'n3': 0.3,
        }
        df = pd.DataFrame([
            ['n1', 'D1', 'q1'],
            ['n2', 'D2', 'q1'],
            ['n3', 'D3', 'q1'],
        ], columns=['website', 'fact', 'object'])
        similarity_matrix = np.array([
            [1.0, 0.64, 0.35],
            [0.64, 1.0, 0.30],
            [0.35, 0.30, 1.0],
        ])

        truth = finder._aggregate_truth(df, np.zeros((3, 1)), similarity_matrix)
        finder._update_credibility(df, np.zeros((3, 1)), similarity_matrix, truth)

        self.assertEqual(truth['q1'], 'D1')
        self.assertAlmostEqual(finder.node_credibility['n1'], 0.765, delta=0.002)
        self.assertAlmostEqual(finder.node_credibility['n2'], 0.518, delta=0.002)
        self.assertAlmostEqual(finder.node_credibility['n3'], 0.215, delta=0.002)

    def test_process_batch_carries_credibility_across_batches(self):
        class StubSenteTruthFinder(SenteTruthFinder):
            def __init__(self):
                super().__init__(use_sbert=False, enable_zk_proof=False)
                self.embedding_map = {
                    "H1A": np.array([1.0, 0.0, 0.0]),
                    "H2A": np.array([1.0, 0.0, 0.0]),
                    "M1A": np.array([0.0, 1.0, 0.0]),
                    "M2A": np.array([0.0, 0.1, np.sqrt(0.99)]),
                    "H1B": np.array([1.0, 0.0, 0.0]),
                    "H2B": np.array([1.0, 0.0, 0.0]),
                    "M1B": np.array([0.0, 1.0, 0.0]),
                    "M2B": np.array([0.0, 1.0, 0.0]),
                }

            def _encode_texts(self, texts):
                return np.asarray([self.embedding_map[str(text)] for text in texts], dtype=float)

        batch_1 = pd.DataFrame(
            [
                ["n1", "H1A", "q1", 1],
                ["n2", "H2A", "q1", 1],
                ["n3", "M1A", "q1", 1],
                ["n4", "M2A", "q1", 1],
            ],
            columns=["website", "fact", "object", "batch_index"],
        )
        batch_2 = pd.DataFrame(
            [
                ["n3", "M1B", "q2", 2],
                ["n4", "M2B", "q2", 2],
                ["n1", "H1B", "q2", 2],
                ["n2", "H2B", "q2", 2],
            ],
            columns=["website", "fact", "object", "batch_index"],
        )

        persistent = StubSenteTruthFinder()
        persistent.process_batch(batch_1)
        self.assertGreater(persistent.node_credibility["n1"], persistent.node_credibility["n3"])

        out_persistent = persistent.process_batch(batch_2)
        fresh = StubSenteTruthFinder()
        out_fresh = fresh.process_batch(batch_2)

        self.assertEqual(persistent.last_iteration_count, 1)
        self.assertEqual(str(out_fresh["global_truth"].iloc[0]), "M1B")
        self.assertEqual(str(out_persistent["global_truth"].iloc[0]), "H1B")

    def test_local_files_only_uses_cached_snapshot_path(self):
        with patch(
            "decentralized_truth_discovery.core.sente_truth_finder.snapshot_download",
            return_value="/tmp/sente-sbert-cache",
        ) as mock_snapshot, patch(
            "decentralized_truth_discovery.core.sente_truth_finder.SentenceTransformer"
        ) as mock_encoder:
            SenteTruthFinder(use_sbert=True, local_files_only=True, enable_zk_proof=False)

        mock_snapshot.assert_called_once_with(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        self.assertEqual(mock_encoder.call_args.args[0], "/tmp/sente-sbert-cache")
        self.assertTrue(mock_encoder.call_args.kwargs["local_files_only"])


class TestExperimentDatasetRouting(unittest.TestCase):
    def test_numeric_experiments_use_dedicated_numeric_datasets(self):
        numeric_files = get_numeric_demo_files()

        self.assertEqual(sorted(numeric_files), list(range(7)))
        self.assertEqual(numeric_files[0].name, "num_demo.json")
        self.assertEqual(numeric_files[6].name, "num_demo_6bad_agent.json")


class TestBasicTruthFinderOnlineHistory(unittest.TestCase):
    def test_process_batch_carries_trustworthiness_across_batches(self):
        finder = BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False)
        batch_1 = pd.DataFrame(
            [
                ["n1", "HONEST", "q1", 1],
                ["n2", "HONEST", "q1", 1],
                ["n3", "MALICIOUS", "q1", 1],
            ],
            columns=["website", "fact", "object", "batch_index"],
        )
        batch_2 = pd.DataFrame(
            [
                ["n3", "M2", "q2", 2],
                ["n1", "H2", "q2", 2],
            ],
            columns=["website", "fact", "object", "batch_index"],
        )

        finder.process_batch(batch_1, implication_texts=["HONEST", "HONEST", "MALICIOUS"])
        self.assertGreater(finder.website_trustworthiness["n1"], finder.website_trustworthiness["n3"])

        out_persistent = finder.process_batch(
            batch_2,
            implication_texts=["HONEST", "HONEST", "MALICIOUS", "M2", "H2"],
        )

        fresh = BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False)
        out_fresh = fresh.process_batch(batch_2, implication_texts=["M2", "H2"])

        self.assertEqual(str(out_fresh["global_truth"].iloc[0]), "M2")
        self.assertEqual(str(out_persistent["global_truth"].iloc[0]), "H2")

    def test_reset_history_clears_persistent_trustworthiness(self):
        finder = BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False)
        batch = pd.DataFrame(
            [
                ["n1", "FACT_A", "q1", 1],
                ["n2", "FACT_A", "q1", 1],
                ["n3", "FACT_B", "q1", 1],
            ],
            columns=["website", "fact", "object", "batch_index"],
        )

        finder.process_batch(batch, implication_texts=["FACT_A", "FACT_A", "FACT_B"])
        self.assertTrue(finder.get_source_reliability())

        finder.reset_history()

        self.assertEqual(finder.get_source_reliability(), {})


class TestSenFeedPaperAlignment(unittest.TestCase):
    def test_iterative_mode_ignores_history_truth_virtual_source(self):
        finder = SenFeedTruthDiscovery(mode='iterative', enable_zk_proof=False)
        finder.source_weights = {'s1': 1.0, 's2': 1.0}
        finder.previous_truth = {'o1': 1000.0}
        df = pd.DataFrame([
            ['s1', 'o1', 10.0, 0],
            ['s2', 'o1', 12.0, 0],
        ], columns=['website', 'object', 'fact', 'timestamp'])

        truth_map = finder._run_iterative_truth_discovery(df)

        self.assertAlmostEqual(truth_map['o1'], 11.0, places=6)

    def test_composite_uses_epsilon_without_source_count_scaling(self):
        class StubComposite(SenFeedTruthDiscovery):
            def _run_iterative_truth_discovery(self, df_t, include_history_truth=False):
                for source in df_t['website'].unique():
                    self.source_weights[str(source)] = 1.0
                self.last_iteration_count = 2
                return {str(df_t['object'].iloc[0]): 10.0}

            def _run_incremental_map_update(self, df_t):
                for source in df_t['website'].unique():
                    self.source_weights[str(source)] = 1.03
                self.last_iteration_count = 1
                return (
                    {str(df_t['object'].iloc[0]): 11.0},
                    {str(source): 0.0 for source in df_t['website'].unique()},
                    {str(source): 1 for source in df_t['website'].unique()},
                )

        finder = StubComposite(mode='composite', evolution_threshold=0.05, enable_zk_proof=False)
        df = pd.DataFrame([
            ['s1', 'o1', 10.0, 0],
            ['s2', 'o1', 10.0, 0],
            ['s1', 'o1', 11.0, 1],
            ['s2', 'o1', 11.0, 1],
        ], columns=['website', 'object', 'fact', 'timestamp'])

        result = finder.train(df)
        second_timestamp = result[result['timestamp'] == 1]

        self.assertTrue((second_timestamp['td_mode'] == 'incremental').all())
        self.assertTrue(np.allclose(second_timestamp['weight_evolution'].to_numpy(), 0.03))

    def test_incremental_map_accumulates_observation_count_across_steps(self):
        finder = SenFeedTruthDiscovery(
            mode='incremental',
            history_truth_weight=0.0,
            enable_zk_proof=False,
        )
        step_1 = pd.DataFrame([
            ['s1', 'o1', 10.0, 0],
        ], columns=['website', 'object', 'fact', 'timestamp'])
        step_2 = pd.DataFrame([
            ['s1', 'o1', 10.0, 1],
        ], columns=['website', 'object', 'fact', 'timestamp'])

        finder._initialize_sources(pd.concat([step_1, step_2], ignore_index=True))

        truth_1, _, _ = finder._run_incremental_map_update(step_1)
        weight_after_step_1 = finder.source_weights['s1']
        finder.previous_truth = truth_1.copy()
        truth_2, _, _ = finder._run_incremental_map_update(step_2)

        self.assertAlmostEqual(truth_1['o1'], 10.0, places=6)
        self.assertAlmostEqual(truth_2['o1'], 10.0, places=6)
        self.assertAlmostEqual(weight_after_step_1, 1.5, places=6)
        self.assertAlmostEqual(finder.source_weights['s1'], 2.0, places=6)
        self.assertEqual(finder.cumulative_obs_count['s1'], 2)

        finder.reset_history()

        self.assertEqual(finder.cumulative_obs_count, {})

    def test_composite_fallback_rolls_back_incremental_observation_count(self):
        finder = SenFeedTruthDiscovery(
            mode='composite',
            evolution_threshold=0.05,
            history_truth_weight=0.0,
            enable_zk_proof=False,
        )
        df = pd.DataFrame([
            ['s1', 'o1', 10.0, 0],
            ['s1', 'o1', 10.0, 1],
        ], columns=['website', 'object', 'fact', 'timestamp'])

        result = finder.train(df)
        second_timestamp = result[result['timestamp'] == 1]

        self.assertTrue((second_timestamp['td_mode'] == 'iterative').all())
        self.assertEqual(finder.cumulative_obs_count['s1'], 0)


class TestSemanticTruthFinder(unittest.TestCase):
    def test_local_files_only_uses_cached_snapshot_path(self):
        with patch(
            "decentralized_truth_discovery.core.semantic_truth_finder.snapshot_download",
            return_value="/tmp/semantic-sbert-cache",
        ) as mock_snapshot, patch(
            "decentralized_truth_discovery.core.semantic_truth_finder.SentenceTransformer"
        ) as mock_encoder:
            SemanticTruthFinder(use_sbert=True, local_files_only=True, enable_zk_proof=False)

        mock_snapshot.assert_called_once_with(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        self.assertEqual(mock_encoder.call_args.args[0], "/tmp/semantic-sbert-cache")
        self.assertTrue(mock_encoder.call_args.kwargs["local_files_only"])

    def test_numeric_pipeline_is_more_robust_than_simple_mean(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
        numeric_df = pd.DataFrame(
            [
                ["h1", "btc", 100.0, 1],
                ["h2", "btc", 101.0, 1],
                ["h3", "btc", 99.0, 1],
                ["h4", "btc", 100.5, 1],
                ["h5", "btc", 99.5, 1],
                ["h6", "btc", 100.2, 1],
                ["m1", "btc", 160.0, 1],
                ["m2", "btc", 170.0, 1],
                ["m3", "btc", 180.0, 1],
                ["m4", "btc", 190.0, 1],
            ],
            columns=["website", "object", "fact", "batch_index"],
        )
        empty_text = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

        out_num, _ = finder.process_batch(numeric_df, empty_text)
        truth = float(out_num["global_truth"].iloc[0])
        simple_mean = float(numeric_df["fact"].mean())

        self.assertLess(abs(truth - 100.0), abs(simple_mean - 100.0))

    def test_text_pipeline_prefers_centroid_supported_claim(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
        text_df = pd.DataFrame(
            [
                ["h1", "news", "Agilent put strike is 125 and call strike is 150.", 1],
                ["h2", "news", "Agilent options show put 125 and call 150.", 1],
                ["h3", "news", "The option chain reports a 125 put and 150 call for Agilent.", 1],
                ["m1", "news", "Agilent put strike is 1000 and call strike is 1.", 1],
                ["m2", "news", "Agilent put strike is 1000 and call strike is 1.", 1],
            ],
            columns=["website", "object", "fact", "batch_index"],
        )
        empty_num = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

        _, out_txt = finder.process_batch(empty_num, text_df)
        truth = str(out_txt["global_truth"].iloc[0])

        self.assertIn("125", truth)
        self.assertIn("150", truth)

    def test_numeric_similarity_matches_data_accuracy_tolerance(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)

        clean_numbers = [2024.0, 8.0, 125.0, 77.0]
        attacked_numbers = [2024.0, 8.0, 110.0, 57.0]

        similarity = finder._numeric_similarity(clean_numbers, attacked_numbers)

        self.assertAlmostEqual(finder.numeric_tol, DEFAULT_NUMERIC_MATCH_TOL)
        self.assertAlmostEqual(similarity, 0.5)

    def test_affinity_uses_equal_weight_additive_fusion(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
        peer_sims = np.array([
            [1.0, 0.9],
            [0.9, 1.0],
        ])

        affinity = finder._semantic_affinity_matrix(
            peer_sims=peer_sims,
            report_numeric_values=[[125.0, 77.0], [125.0, 57.0]],
        )

        self.assertAlmostEqual(finder.semantic_numeric_alpha, DEFAULT_SEMANTIC_NUMERIC_ALPHA)
        self.assertAlmostEqual(affinity[0, 1], 0.7)
        self.assertAlmostEqual(affinity[1, 0], 0.7)

    def test_text_pipeline_downweights_collusive_near_duplicates_in_generic_text(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
        text_df = pd.DataFrame(
            [
                ["h1", "story", "The package arrived on Tuesday and contained 12 blue adapters for the lab kit.", 1],
                ["h2", "story", "The lab kit package came on Tuesday with 12 blue adapters inside.", 1],
                ["h3", "story", "Tuesday delivery brought the lab kit together with 12 blue adapters.", 1],
                ["h4", "story", "The shipment for the lab kit showed up on Tuesday and included 12 blue adapters.", 1],
                ["m1", "story", "The package arrived on Friday and contained 18 red adapters for the lab kit.", 1],
                ["m2", "story", "The package arrived on Friday and contained 18 red adapters for the lab kit.", 1],
                ["m3", "story", "Friday delivery brought the lab kit together with 18 red adapters.", 1],
            ],
            columns=["website", "object", "fact", "batch_index"],
        )
        empty_num = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

        _, out_txt = finder.process_batch(empty_num, text_df)
        truth = str(out_txt["global_truth"].iloc[0]).lower()

        self.assertIn("tuesday", truth)
        self.assertIn("12", truth)
        self.assertIn("blue", truth)
        self.assertNotIn("friday", truth)
        self.assertNotIn("18", truth)

    def test_reliability_update_penalizes_persistent_numeric_outliers(self):
        finder = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
        empty_text = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

        for batch_index in range(1, 5):
            numeric_df = pd.DataFrame(
                [
                    ["h1", "btc", 100.0 + 0.1 * batch_index, batch_index],
                    ["h2", "btc", 100.2 + 0.1 * batch_index, batch_index],
                    ["h3", "btc", 99.8 + 0.1 * batch_index, batch_index],
                    ["m1", "btc", 170.0, batch_index],
                    ["m2", "btc", 185.0, batch_index],
                ],
                columns=["website", "object", "fact", "batch_index"],
            )
            finder.process_batch(numeric_df, empty_text)

        reliabilities = finder.get_reliability()
        honest_avg = np.mean([reliabilities["h1"]["reliability"], reliabilities["h2"]["reliability"], reliabilities["h3"]["reliability"]])
        malicious_avg = np.mean([reliabilities["m1"]["reliability"], reliabilities["m2"]["reliability"]])

        self.assertGreater(honest_avg, malicious_avg)
        self.assertGreater(honest_avg, 0.6)
        self.assertLess(malicious_avg, 0.5)


class TestEnhancedTruthFinderDecentTruthFixes(unittest.TestCase):
    def test_source_reliability_update_uses_raw_mse_and_k_over_e(self):
        finder = EnhancedTruthFinder(alpha=0.9, beta=1.0, enable_zk_proof=False)
        finder.source_reliability_history = {
            's1': {'r': 0.5, 'e': 1.0, 'k': 0.8},
        }
        df = pd.DataFrame([
            ['s1', 3.0, 'o1'],
        ], columns=['website', 'fact', 'object'])

        finder._update_source_reliability(df, {'o1': 1.0})
        metrics = finder.get_source_reliability('s1')

        expected_e = 4.0
        expected_k = 0.9 * 0.8 + 0.1 * erfc(abs(expected_e - 1.0))
        expected_r = expected_k / (expected_e + 1e-10)

        self.assertAlmostEqual(metrics['e'], expected_e, places=9)
        self.assertAlmostEqual(metrics['k'], expected_k, places=9)
        self.assertAlmostEqual(metrics['r'], expected_r, places=9)

    def test_numeric_global_truth_uses_coordinate_descent(self):
        finder = EnhancedTruthFinder(enable_zk_proof=False, global_td_max_iter=50, global_td_tol=1e-9)
        df = pd.DataFrame([
            ['a', 10.0, 'o1'],
            ['a', 20.0, 'o2'],
            ['b', 10.1, 'o1'],
            ['b', 19.9, 'o2'],
            ['c', 9.9, 'o1'],
            ['c', 20.2, 'o2'],
            ['d', 100.0, 'o1'],
            ['d', 200.0, 'o2'],
        ], columns=['website', 'fact', 'object'])

        truth = finder._run_global_td(df)

        self.assertAlmostEqual(float(truth['o1']), 10.0, delta=1.0)
        self.assertAlmostEqual(float(truth['o2']), 20.0, delta=1.0)


if __name__ == '__main__':
    unittest.main()
