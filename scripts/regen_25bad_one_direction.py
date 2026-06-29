#!/usr/bin/env python3
"""
One-shot regeneration of demo_25bad_numonly.json with a SINGLE attack direction
(group_sizes=[25]) for the four 50-agent text-experiment data dirs:

  - data_agent-50_news-300
  - data_agent-50_weather-300
  - data-llama-agent-50_news-300
  - data-llama-agent-50_weather-300

The default 25-bad spec is [15, 10] (two attack directions). This script keeps
the original spec untouched in the regen scripts and only overrides for the
50%/numonly case used by the 1.2 / 1.4 sub-experiments.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/Users/huabowen/TruthFinder/decentralized_truth_discovery")
NEWS_REF = ROOT / "data_agent-10" / "regen_bad_text.py"
WEATHER_REGEN = ROOT / "data_agent-50_weather-300" / "regen_bad_text.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def regen_news_25bad_numonly(data_dir: Path, ref_module) -> Path:
    """Use the data_agent-10 reference module (same as data_agent-50_news-300
    delegates to) to generate a numeric-only 25-bad file with one direction."""
    div_path = data_dir / "demo_div.json"
    if not div_path.exists():
        raise FileNotFoundError(f"missing {div_path}")
    diverse = _load_json(div_path)

    canonical_clean_map = ref_module.build_clean_text_map(diverse)
    clean_agent_map = ref_module.build_clean_agent_text_map(diverse)
    bad_agents = [f"agent-{i}" for i in range(1, 26)]

    out_path = data_dir / "demo_25bad_numonly.json"
    ref_module.regenerate_collude_numeric_only_file(
        out_file_path=str(out_path),
        demo_data=diverse,
        canonical_clean_map=canonical_clean_map,
        clean_agent_map=clean_agent_map,
        bad_agents=bad_agents,
        group_sizes=[25],
        direction_names=ref_module.collude_direction_names(1),
    )
    print(f"[news] wrote {out_path} group_sizes=[25] direction=1")
    return out_path


def regen_weather_25bad_numonly(data_dir: Path, weather_module) -> Path:
    div_path = data_dir / "demo_div.json"
    if not div_path.exists():
        raise FileNotFoundError(f"missing {div_path}")
    diverse = _load_json(div_path)

    canonical_clean_map = weather_module.build_clean_text_map(diverse)
    bad_agents = [f"agent-{i}" for i in range(1, 26)]

    out_path = data_dir / "demo_25bad_numonly.json"
    weather_module.regenerate_collude_numeric_only_file(
        out_file_path=out_path,
        demo_data=diverse,
        canonical_clean_map=canonical_clean_map,
        bad_agents=bad_agents,
        group_sizes=[25],
    )
    print(f"[weather] wrote {out_path} group_sizes=[25] direction=1")
    return out_path


def main() -> None:
    news_ref = _load_module("news_ref", NEWS_REF)
    weather_mod = _load_module("weather_regen", WEATHER_REGEN)

    news_dirs = [
        ROOT / "data_agent-50_news-300",
        ROOT / "data-llama-agent-50_news-300",
    ]
    weather_dirs = [
        ROOT / "data_agent-50_weather-300",
        ROOT / "data-llama-agent-50_weather-300",
    ]

    for d in news_dirs:
        regen_news_25bad_numonly(d, news_ref)
    for d in weather_dirs:
        regen_weather_25bad_numonly(d, weather_mod)


if __name__ == "__main__":
    main()
