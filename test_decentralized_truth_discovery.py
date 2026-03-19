import unittest

import numpy as np
import pandas as pd

from decentralized_truth_discovery.core import (
    EnhancedTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
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


class TestEnhancedTruthFinderHybrid(unittest.TestCase):
    def test_hybrid_truth_resists_outlier_and_uses_named_option_fields(self):
        finder = EnhancedTruthFinder(enable_hybrid_mode=True, enable_zk_proof=False, max_iterations=3)
        df = pd.DataFrame([
            ['a', 'Agilent Technologies (Symbol: A) opened with a $125 put priced at $4.50 and a $150 call priced at $8.10.', 'opt'],
            ['b', 'Agilent Technologies (Symbol: A) opened with a $126 put priced at $4.55 and a $149 call priced at $8.05.', 'opt'],
            ['c', 'Agilent Technologies (Symbol: A) opened with a $1000 put priced at $99.99 and a $1 call priced at $0.01.', 'opt'],
        ], columns=['website', 'fact', 'object'])

        result = finder.train(df, max_iterations=3)
        truth = result['global_truth'].iloc[0]

        self.assertIn('put_strike_price', truth['numeric'])
        self.assertIn('put_quote', truth['numeric'])
        self.assertIn('call_strike_price', truth['numeric'])
        self.assertIn('call_quote', truth['numeric'])
        self.assertAlmostEqual(truth['numeric']['put_strike_price'], 125.5, delta=0.6)
        self.assertAlmostEqual(truth['numeric']['put_quote'], 4.525, delta=0.1)
        self.assertAlmostEqual(truth['numeric']['call_strike_price'], 149.5, delta=0.6)
        self.assertAlmostEqual(truth['numeric']['call_quote'], 8.075, delta=0.1)

        source_scores = finder.get_source_reliability()
        self.assertGreater(source_scores['a']['r'], source_scores['c']['r'])
        self.assertGreater(source_scores['b']['r'], source_scores['c']['r'])


if __name__ == '__main__':
    unittest.main()
