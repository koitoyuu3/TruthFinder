#!/usr/bin/env python3
"""
Generate diversified clean demo.json and collusive text variants for 50 agents.

This script reuses the text-noise and collusion mutation logic from
decentralized_truth_discovery/data_agent-10/regen_bad_text.py and only adds
custom malicious-group sizing required by the 50-agent dataset.
Each malicious group is treated as one explicit attack direction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
REFERENCE_SCRIPT = DATA_DIR.parent / "data_agent-10" / "regen_bad_text.py"
DEMO_PATH = DATA_DIR / "demo.json"

BAD_GROUP_SPECS = {
    5: [5],
    10: [10],
    15: [15],
    20: [20],
    25: [15, 10],
    30: [15, 15],
    35: [12, 12, 11],
    40: [8, 8, 8, 8, 8],
}


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("regen_bad_text_ref", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REF = _load_reference_module()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def regenerate_collude_file(
    out_file_path: Path,
    demo_data,
    canonical_clean_map: dict,
    clean_agent_map: dict,
    bad_agents: list[str],
    group_sizes: list[int],
):
    direction_names = REF.collude_direction_names(len(group_sizes))
    data, _ = REF.regenerate_collude_file(
        out_file_path=str(out_file_path),
        demo_data=demo_data,
        canonical_clean_map=canonical_clean_map,
        clean_agent_map=clean_agent_map,
        bad_agents=bad_agents,
        group_sizes=group_sizes,
        direction_names=direction_names,
    )
    return data


def main() -> None:
    demo_data = _load_json(DEMO_PATH)
    diverse_demo_data, _ = REF.diversify_clean_demo_data(demo_data)
    _save_json(DEMO_PATH, diverse_demo_data)

    canonical_clean_map = REF.build_clean_text_map(diverse_demo_data)
    clean_agent_map = REF.build_clean_agent_text_map(diverse_demo_data)

    for bad_count, group_sizes in BAD_GROUP_SPECS.items():
        bad_agents = [f"agent-{idx}" for idx in range(1, bad_count + 1)]
        out_path = DATA_DIR / f"demo_{bad_count}bad_collude.json"
        regenerate_collude_file(
            out_file_path=out_path,
            demo_data=diverse_demo_data,
            canonical_clean_map=canonical_clean_map,
            clean_agent_map=clean_agent_map,
            bad_agents=bad_agents,
            group_sizes=group_sizes,
        )
        print(
            f"generated {out_path.name} with groups={group_sizes} "
            f"directions={REF.collude_direction_names(len(group_sizes))}"
        )


if __name__ == "__main__":
    main()
