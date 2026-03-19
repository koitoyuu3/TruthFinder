import unittest

import numpy as np
import pandas as pd

from decentralized_truth_discovery.examples.five_methods_comparison import (
    _build_object_balanced_source_details,
    _build_senfeed_input,
    _create_enhanced_finder,
    _native_score_to_percentage,
    create_reference_data,
    run_basic,
    run_enhanced,
    run_senfeed,
)


class TestFiveMethodsComparison(unittest.TestCase):
    @staticmethod
    def _mixed_reference_df() -> pd.DataFrame:
        return create_reference_data(
            [
                {
                    "agent": "agent-1",
                    "object": "第1条新闻简短摘要",
                    "response": (
                        "Agilent Technologies, Inc. (A) 的新期权合约将于2024年8月开始交易，"
                        "当前看跌期权行权价为125.00，看涨期权行权价为150.00，当前报价为8.10。"
                    ),
                },
                {
                    "agent": "agent-1",
                    "object": "2025-11-27 00:00:00 的 BTC/ETH/DOGE 价格数据",
                    "response": {"BTC": 90490.22, "ETH": 3026.33, "DOGE": 0.1548},
                },
                {
                    "agent": "agent-2",
                    "object": "第1条新闻简短摘要",
                    "response": "Agilent Technologies, Inc. (A) 的看涨期权行权价为150.00，当前报价为8.10。",
                },
                {
                    "agent": "agent-2",
                    "object": "2025-11-27 00:00:00 的 BTC/ETH/DOGE 价格数据",
                    "response": {"BTC": 90490.22, "ETH": 3026.33, "DOGE": 0.1548},
                },
            ]
        )

    def test_object_balanced_source_details_does_not_zero_fill_missing_parent_objects(self):
        result_df = pd.DataFrame(
            [
                {"website": "agent-1", "parent_object": "news", "trustworthiness": 0.8},
                {"website": "agent-2", "parent_object": "news", "trustworthiness": 0.4},
                {"website": "agent-2", "parent_object": "price", "trustworthiness": 0.6},
            ]
        )

        details = _build_object_balanced_source_details(
            result_df,
            {"trustworthiness": "trustworthiness"},
            expected_parent_objects=["news", "price"],
            expected_websites=["agent-1", "agent-2", "agent-3"],
        )

        self.assertAlmostEqual(details["agent-1"]["trustworthiness"], 0.8)
        self.assertAlmostEqual(details["agent-2"]["trustworthiness"], 0.5)
        self.assertNotIn("agent-3", details)

    def test_run_basic_supports_text_and_numeric(self):
        text_df = self._mixed_reference_df()

        result = run_basic(text_df)

        self.assertEqual(result.result_df["object"].nunique(), 2)
        self.assertEqual(set(result.result_df["object"].astype(str)), set(text_df["object"].astype(str)))
        self.assertTrue(all(np.isfinite(value) for value in result.source_scores.values()))
        self.assertTrue(all("representation_fidelity" in detail for detail in result.source_details.values()))
        self.assertTrue(all(detail["representation_fidelity"] == 0.75 for detail in result.source_details.values()))
        self.assertTrue(all(detail["calibrated_score"] < detail["trustworthiness"] for detail in result.source_details.values()))

    def test_run_enhanced_nonhybrid_supports_text_and_numeric_via_split_runs(self):
        text_df = self._mixed_reference_df()

        result = run_enhanced(
            text_df,
            finder=_create_enhanced_finder(enable_hybrid_mode=False),
            label="EnhancedTruthFinder-NonHybrid",
        )
        assets = set(result.result_df["asset"].astype(str))

        self.assertIn("TEXT", assets)
        self.assertIn("BTC", assets)
        self.assertEqual(result.result_df["parent_object"].nunique(), 2)
        self.assertTrue(all(np.isfinite(value) for value in result.source_scores.values()))

    def test_senfeed_text_proxy_not_forced_to_zero(self):
        text_df = self._mixed_reference_df()
        senfeed_df = _build_senfeed_input(text_df[text_df["record_type"] == "text"])
        text_rows = senfeed_df[senfeed_df["asset"] == "TEXT"]

        self.assertFalse(text_rows.empty)
        self.assertFalse(text_rows["fact"].isna().any())
        self.assertTrue((text_rows["fact"] != 0.0).all())

        result = run_senfeed(text_df)
        result_text_rows = result.result_df[result.result_df["asset"].astype(str) == "TEXT"]
        self.assertFalse(result_text_rows.empty)
        self.assertTrue((result_text_rows["source_weight"] > 0.0).all())

    def test_senfeed_percentage_no_longer_maps_weight_one_to_fifty(self):
        percentage = _native_score_to_percentage(
            pd.Series([1.5, 1.0, 0.5, np.nan]),
            "SenFeedTruthFinder-CompositeTD",
        )

        self.assertAlmostEqual(float(percentage.iloc[0]), float((1.0 - np.exp(-1.5)) * 100.0), places=6)
        self.assertAlmostEqual(float(percentage.iloc[1]), float((1.0 - np.exp(-1.0)) * 100.0), places=6)
        self.assertAlmostEqual(float(percentage.iloc[2]), float((1.0 - np.exp(-0.5)) * 100.0), places=6)
        self.assertNotAlmostEqual(float(percentage.iloc[1]), 50.0, places=4)
        self.assertNotAlmostEqual(float(percentage.iloc[2]), 33.3333, places=3)
        self.assertTrue(pd.isna(percentage.iloc[3]))


if __name__ == "__main__":
    unittest.main()
