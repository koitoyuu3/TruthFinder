#!/usr/bin/env python3
"""
Regenerate numeric malicious-agent datasets from num_demo.json.

Two attack modes:
  rotate  (default): four-profile cycle per (batch, agent). Profile index =
          (t + j - 2) % 4, so bad agents disagree with each other.
  collude: bad agents form a coordinated wrong-value cluster with a fixed
           center across all batches, while malicious members keep small
           agent-specific jitter so the cluster is aligned but not identical.

In collude mode, honest agents are also perturbed with small deterministic
agent-specific sensor noise so that honest reports are no longer perfectly
identical, but the environment stays stationary over time.
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

COLLUDE_CENTER_PROFILE = ATTACK_PROFILES[0]

COLLUDE_JITTER_PROFILES = (
    {"BTC": Decimal("0.9994"), "ETH": Decimal("1.0008"), "DOGE": Decimal("0.9991")},
    {"BTC": Decimal("1.0007"), "ETH": Decimal("0.9993"), "DOGE": Decimal("1.0011")},
    {"BTC": Decimal("0.9998"), "ETH": Decimal("1.0004"), "DOGE": Decimal("1.0006")},
    {"BTC": Decimal("1.0005"), "ETH": Decimal("0.9997"), "DOGE": Decimal("0.9995")},
    {"BTC": Decimal("0.9991"), "ETH": Decimal("1.0009"), "DOGE": Decimal("1.0002")},
    {"BTC": Decimal("1.0009"), "ETH": Decimal("0.9992"), "DOGE": Decimal("0.9989")},
)

HONEST_NOISE_PROFILES = (
    {"BTC": Decimal("0.9980"), "ETH": Decimal("0.9986"), "DOGE": Decimal("1.0025")},
    {"BTC": Decimal("1.0018"), "ETH": Decimal("1.0014"), "DOGE": Decimal("0.9978")},
    {"BTC": Decimal("0.9992"), "ETH": Decimal("1.0021"), "DOGE": Decimal("1.0016")},
    {"BTC": Decimal("1.0024"), "ETH": Decimal("0.9989"), "DOGE": Decimal("0.9984")},
    {"BTC": Decimal("0.9977"), "ETH": Decimal("1.0010"), "DOGE": Decimal("1.0029")},
    {"BTC": Decimal("1.0020"), "ETH": Decimal("0.9979"), "DOGE": Decimal("0.9975")},
)


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


def _collude_jitter_profile(agent_name: str) -> dict[str, Decimal]:
    agent_idx = int(agent_name.split("-")[-1])
    profile_idx = (agent_idx - 1) % len(COLLUDE_JITTER_PROFILES)
    return COLLUDE_JITTER_PROFILES[profile_idx]


def _collude_center_profile(agent_name: str, bad_count: int) -> dict[str, Decimal]:
    """
    Static collude center selection.

    All malicious agents share the same wrong-value center for every batch.
    """
    del agent_name, bad_count
    return COLLUDE_CENTER_PROFILE


def _honest_noise_profile(agent_name: str) -> dict[str, Decimal]:
    agent_idx = int(agent_name.split("-")[-1])
    profile_idx = (agent_idx - 1) % len(HONEST_NOISE_PROFILES)
    return HONEST_NOISE_PROFILES[profile_idx]


def _compose_profiles(left: dict[str, Decimal], right: dict[str, Decimal]) -> dict[str, Decimal]:
    return {
        coin: left[coin] * right[coin]
        for coin in left
    }


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
        for item in batch["items"]:
            if not isinstance(item["response"], dict):
                continue
            if item["agent"] in bad_agents:
                if mode == "collude":
                    profile = _compose_profiles(
                        _collude_center_profile(item["agent"], bad_count),
                        _collude_jitter_profile(item["agent"]),
                    )
                else:
                    batch_index = int(batch["batchIndex"])
                    profile = _rotate_profile(batch_index, item["agent"])
                item["response"] = _mutate_numeric_response(item["response"], profile)
            elif mode == "collude":
                profile = _honest_noise_profile(item["agent"])
                item["response"] = _mutate_numeric_response(item["response"], profile)
            else:
                continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate num_demo_*bad_*.json datasets.")
    parser.add_argument(
        "--bad-counts",
        default="1,2,3,4,5,6",
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
