#!/usr/bin/env python3
"""Regenerate ONLY the four affected figures in docs/figs_v2/ after updating
the 50% (one-direction) row in the exp1_text 1.2 / 1.4 CSVs.

This re-uses ``fig_exp1_text`` from ``docs/_make_figs_v2.py`` so styling stays
identical, but skips the unrelated jobs (exp3/exp4/exp7) and the global PDF
bundle to keep the run quick.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path("/Users/huabowen/TruthFinder")
MODULE_PATH = ROOT / "docs" / "_make_figs_v2.py"

spec = importlib.util.spec_from_file_location("_make_figs_v2", MODULE_PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules["_make_figs_v2"] = m
spec.loader.exec_module(m)

CSV_ROOT = m.ROOT  # examples/outputs

JOBS = [
    (
        "exp1_text_1_2_agent",
        CSV_ROOT / "exp1_text/1.2-agent-50-news-300-50%-0.9-numonly/text_data_accuracy.csv",
    ),
    (
        "exp1_text_1_2_llama",
        CSV_ROOT / "exp1_text/1.2-llama-news-300-50%-0.9-numonly/text_data_accuracy.csv",
    ),
    (
        "exp1_text_1_4_agent",
        CSV_ROOT / "exp1_text/1.4-agent-50-weather-300-50%-0.9-numonly/text_data_accuracy.csv",
    ),
    (
        "exp1_text_1_4_llama",
        CSV_ROOT / "exp1_text/1.4-llama-weather-300-50%-0.9-numonly/text_data_accuracy.csv",
    ),
]


def main() -> None:
    for name, csv_path in JOBS:
        png = m.fig_exp1_text(csv_path, name)
        print(f"  -> {png}")


if __name__ == "__main__":
    main()
