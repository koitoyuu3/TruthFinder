#!/usr/bin/env python3
"""
Regenerate numeric malicious-agent datasets from num_demo.json.

Two attack modes:
  rotate  (default): four-profile cycle per (batch, agent). Profile index =
          (t + j - 2) % 4, so bad agents disagree with each other.
  collude: all bad agents always use ATTACK_PROFILES[0] for every batch,
           forming a coordinated wrong-value cluster.
"""

from __future__ import annotations

import argparse
import copy
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
CLEAN_FILE = DATA_DIR / "num_demo.json"

ATTACK_PROFILES = (
    {"BTC": Decimal("0.985"), "ETH": Decimal("0.982"), "DOGE": Decimal("1.026")},
    {"BTC": Decimal("1.012"), "ETH": Decimal("1.015"), "DOGE": Decimal("0.977")},
    {"BTC": Decimal("0.979"), "ETH": Decimal("0.976"), "DOGE": Decimal("1.020")},
    {"BTC": Decimal("1.018"), "ETH": Decimal("1.021"), "DOGE": Decimal("0.971")},
)

COLLUDE_PROFILE = ATTACK_PROFILES[0]


def _decimal_places(value: Decimal) -> int:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return max(0, -value.as_tuple().exponent)


def _quantize_like(value: Decimal, reference: Decimal) -> Decimal:
    places = _decimal_places(reference)
    step = Decimal("1").scaleb(-places)
    return value.quantize(step, rounding=ROUND_HALF_UP)


def _to_plain_number(value: Decimal):
    places = _decimal_places(value)
    if places == 0:
        return int(value)
    return float(value)


def _rotate_profile(batch_index: int, agent_name: str) -> dict[str, Decimal]:
    agent_idx = int(agent_name.split("-")[-1])
    profile_idx = (batch_index + agent_idx - 2) % len(ATTACK_PROFILES)
    return ATTACK_PROFILES[profile_idx]


def _mutate_numeric_response(response: dict[str, Decimal], profile: dict[str, Decimal]) -> dict[str, Decimal]:
    attacked: dict[str, Decimal] = {}
    for coin, value in response.items():
        attacked_value = _quantize_like(value * profile[coin], value)
        attacked[coin] = attacked_value
    return attacked


def _materialize_json_numbers(obj):
    if isinstance(obj, Decimal):
        return _to_plain_number(obj)
    if isinstance(obj, list):
        return [_materialize_json_numbers(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _materialize_json_numbers(v) for k, v in obj.items()}
    return obj


def regenerate_numeric_dataset(clean_data: dict, bad_count: int, mode: str = "rotate") -> dict:
    data = copy.deepcopy(clean_data)
    bad_agents = {f"agent-{idx}" for idx in range(1, bad_count + 1)}
    for batch in data["job"]["batches"]:
        batch_index = int(batch["batchIndex"])
        for item in batch["items"]:
            if item["agent"] not in bad_agents or not isinstance(item["response"], dict):
                continue
            if mode == "collude":
                profile = COLLUDE_PROFILE
            else:
                profile = _rotate_profile(batch_index, item["agent"])
            item["response"] = _mutate_numeric_response(item["response"], profile)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate num_demo_*bad_*.json datasets.")
    parser.add_argument(
        "--bad-counts",
        default="1,2,3,4",
        help="Comma-separated malicious agent counts to generate.",
    )
    parser.add_argument(
        "--mode",
        choices=["rotate", "collude"],
        default="collude",
        help="Attack mode: rotate (4-profile cycle) or collude (all same profile).",
    )
    args = parser.parse_args()

    bad_counts = [int(part.strip()) for part in args.bad_counts.split(",") if part.strip()]
    clean_data = json.loads(CLEAN_FILE.read_text(encoding="utf-8"), parse_float=Decimal)

    suffix = "collude" if args.mode == "collude" else "agent"
    for bad_count in bad_counts:
        if bad_count < 1:
            raise ValueError(f"bad_count must be >= 1, got {bad_count}")
        out_path = DATA_DIR / f"num_demo_{bad_count}bad_{suffix}.json"
        data = regenerate_numeric_dataset(clean_data, bad_count, mode=args.mode)
        out_path.write_text(
            json.dumps(_materialize_json_numbers(data), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        print(f"generated {out_path}")


if __name__ == "__main__":
    main()
