import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from decentralized_truth_discovery.examples.proof_scalability_benchmark import (
    DEFAULT_BATCH_COUNTS,
    _build_summary_df,
)


def _load_plot_module():
    module_path = (
        Path(__file__).resolve().parent
        / "decentralized_truth_discovery"
        / "result"
        / "plot_proof_scalability_from_summary.py"
    )
    spec = importlib.util.spec_from_file_location("plot_proof_scalability_from_summary", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestProofScalabilityOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_module = _load_plot_module()

    def test_default_batch_counts_sweep_for_scalability(self):
        self.assertEqual(DEFAULT_BATCH_COUNTS, (5, 25, 50, 100, 150, 300))

    def test_build_summary_df_preserves_proof_bundle_path(self):
        raw_df = pd.DataFrame(
            [
                {
                    "domain": "numeric",
                    "method": "BasicTruth",
                    "batch_count": 150,
                    "bundle_path": "/tmp/bundles_batch_150/BasicTruth.numeric.proof_bundle.json",
                    "row_count": 4500,
                    "proof_group_count": 1,
                    "proof_scheme": "schnorr_nizk_v1",
                    "proof_backend": "schnorr",
                    "proof_size_bytes": 2063365,
                    "proof_payload_bytes": 2546,
                    "repeat_index": 1,
                    "verified": True,
                    "result_digest_ok": True,
                    "experiment_digest_ok": True,
                    "verify_elapsed_sec": 0.50,
                    "peak_rss_mb": 360.0,
                    "peak_delta_rss_mb": 12.0,
                },
                {
                    "domain": "numeric",
                    "method": "BasicTruth",
                    "batch_count": 150,
                    "bundle_path": "/tmp/bundles_batch_150/BasicTruth.numeric.proof_bundle.json",
                    "row_count": 4500,
                    "proof_group_count": 1,
                    "proof_scheme": "schnorr_nizk_v1",
                    "proof_backend": "schnorr",
                    "proof_size_bytes": 2063365,
                    "proof_payload_bytes": 2546,
                    "repeat_index": 2,
                    "verified": True,
                    "result_digest_ok": True,
                    "experiment_digest_ok": True,
                    "verify_elapsed_sec": 0.70,
                    "peak_rss_mb": 364.0,
                    "peak_delta_rss_mb": 14.0,
                },
            ]
        )

        summary_df = _build_summary_df(raw_df)

        self.assertEqual(len(summary_df), 1)
        self.assertEqual(
            summary_df.loc[0, "proof_bundle_path"],
            "/tmp/bundles_batch_150/BasicTruth.numeric.proof_bundle.json",
        )
        self.assertEqual(int(summary_df.loc[0, "batch_count"]), 150)
        self.assertAlmostEqual(float(summary_df.loc[0, "verify_elapsed_sec_mean"]), 0.60)
        self.assertAlmostEqual(float(summary_df.loc[0, "verify_peak_delta_rss_mb_mean"]), 13.0)

    def test_select_batch_returns_only_requested_batch(self):
        df = pd.DataFrame(
            [
                {
                    "domain": "numeric",
                    "method": "SemanticTruth",
                    "batch_count": 100,
                    "proof_size_bytes": 1.0,
                    "proof_payload_bytes": 1.0,
                    "verify_peak_delta_rss_mb_mean": 1.0,
                    "verify_elapsed_sec_mean": 1.0,
                },
                {
                    "domain": "numeric",
                    "method": "SemanticTruth",
                    "batch_count": 150,
                    "proof_size_bytes": 2.0,
                    "proof_payload_bytes": 2.0,
                    "verify_peak_delta_rss_mb_mean": 2.0,
                    "verify_elapsed_sec_mean": 2.0,
                },
                {
                    "domain": "numeric",
                    "method": "BasicTruth",
                    "batch_count": 150,
                    "proof_size_bytes": 3.0,
                    "proof_payload_bytes": 3.0,
                    "verify_peak_delta_rss_mb_mean": 3.0,
                    "verify_elapsed_sec_mean": 3.0,
                },
            ]
        )

        selected_df = self.plot_module.select_batch(df, 150)

        self.assertTrue((selected_df["batch_count"] == 150).all())
        self.assertEqual(selected_df["method"].tolist(), ["BasicTruth", "SemanticTruth"])

    def test_format_bytes_keeps_small_thousands_precise(self):
        self.assertEqual(self.plot_module._format_bytes(2553, 0), "2553")
        self.assertEqual(self.plot_module._format_bytes(1786358, 0), "1.8M")

    def test_compute_bar_axis_limits_zooms_low_variance_values(self):
        lower, upper, zoomed = self.plot_module._compute_bar_axis_limits([2541.0, 2550.0, 2555.0])

        self.assertTrue(zoomed)
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, 2555.0)


if __name__ == "__main__":
    unittest.main()
