"""
统一示例：使用同一份 `agent/object/response` 结构化输入运行五种真值发现方法并对比结果。

输入数据约定为一个列表，列表中的每一项均为：
- `agent`: 节点/来源标识
- `object`: 待求真的对象描述
- `response`: 节点对该对象的响应；可以是文本，也可以是结构化数值字典

示例会将结构化输入统一归一化为算法使用的 DataFrame：
- `website`: 来源节点
- `object`: 对象标识（直接使用输入的 `object`）
- `fact`: 供文本型方法使用的规范化字符串
- `response_value`: 原始响应（保留原始文本或字典）
- `btc_price/eth_price/doge_price`: 对结构化币价对象的显式数值拆解
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from decentralized_truth_discovery.core import (
    BasicTruthFinder,
    EnhancedTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
    build_experiment_proof_bundle,
    create_implication_function_from_texts,
    export_chainmaker_group_payloads,
)


DEFAULT_INPUT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
LEGACY_DEFAULT_DEMO_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "demo.json")
)


@dataclass
class MethodResult:
    name: str
    result_df: pd.DataFrame
    source_scores: Dict[str, float]
    source_details: Dict[str, Dict[str, float]]
    truth_preview: Dict[str, Any]
    elapsed_sec: float


@dataclass
class ExperimentRunResult:
    batch_number: int
    batch_metadata: Dict[str, Any]
    methods: List[MethodResult]
    raw_score_matrix: pd.DataFrame
    percentage_matrix: pd.DataFrame
    proof_manifest: Dict[str, Any]


_CRYPTO_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_CRYPTO_ASSET_VALUE_RE = re.compile(
    r"\b(?P<symbol>BTC|ETH|DOGE)\b\s*[:=]\s*\$?(?P<price>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CRYPTO_ASSET_ORDER = ("BTC", "ETH", "DOGE")


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw_value!r}") from exc

    if parsed < 1:
        raise ValueError(f"Environment variable {name} must be >= 1, got: {parsed}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按配置的 batch 输入运行五种真值发现方法，并输出多次实验平均可靠性分数表。"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TRUTHFINDER_DATA_DIR", DEFAULT_INPUT_DATA_DIR),
        help="输入数据目录，默认读取 decentralized_truth_discovery/data",
    )
    parser.add_argument(
        "--input-file",
        default=os.environ.get("TRUTHFINDER_INPUT_FILE", "demo.json"),
        help="输入文件名，默认 demo.json；与 --data-dir 组合使用。",
    )
    parser.add_argument(
        "--input-json",
        default=os.environ.get("TRUTHFINDER_INPUT_JSON"),
        help="输入 JSON 完整路径；提供后优先级高于 --data-dir/--input-file。",
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=_env_positive_int("TRUTHFINDER_NUM_EXPERIMENTS", 1),
        help="实验次数；第 n 次实验读取第 n 个 batch（受 --start-batch 影响时为顺延）。",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=_env_positive_int("TRUTHFINDER_START_BATCH", 1),
        help="起始 batch 编号，按 1 开始计数。",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="打印每次实验的详细方法输出；默认仅在单次实验时打印详细信息。",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.environ.get("TRUTHFINDER_EXPERIMENT_ID"),
        help="实验 ID；默认根据输入文件名自动生成。",
    )
    parser.add_argument(
        "--proof-output-dir",
        default=os.environ.get("TRUTHFINDER_PROOF_OUTPUT_DIR"),
        help="证明输出目录；默认写入 examples/outputs/<experiment-id>",
    )
    return parser.parse_args()


def resolve_input_json_path(args: argparse.Namespace) -> str:
    if args.input_json:
        return os.path.abspath(args.input_json)

    candidate_path = os.path.abspath(os.path.join(args.data_dir, args.input_file))
    if os.path.exists(candidate_path):
        return candidate_path

    if args.input_file == "demo.json" and os.path.exists(LEGACY_DEFAULT_DEMO_JSON_PATH):
        return LEGACY_DEFAULT_DEMO_JSON_PATH

    return candidate_path


def _sanitize_path_stem(file_path: str) -> str:
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", file_stem).strip("._-")
    return sanitized or "input"


def resolve_experiment_id(args: argparse.Namespace, input_json_path: str) -> str:
    if args.experiment_id:
        return args.experiment_id
    return f"five_methods_comparison_{_sanitize_path_stem(input_json_path)}_{int(time.time())}"


def resolve_proof_output_dir(args: argparse.Namespace, experiment_id: str) -> str:
    if args.proof_output_dir:
        return os.path.abspath(args.proof_output_dir)

    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "outputs", experiment_id)
    )


def load_demo_batches(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON root must be an object.")

    job_payload = payload.get("job")
    if not isinstance(job_payload, dict):
        raise ValueError("Input JSON must contain object field 'job'.")

    batches = job_payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("Input JSON must contain a non-empty 'job.batches' list.")

    return batches


def select_batches(
    batches: List[Dict[str, Any]],
    *,
    start_batch: int,
    num_experiments: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    if start_batch < 1:
        raise ValueError(f"start_batch must be >= 1, got: {start_batch}")
    if num_experiments < 1:
        raise ValueError(f"num_experiments must be >= 1, got: {num_experiments}")

    end_batch = start_batch + num_experiments - 1
    if start_batch > len(batches):
        raise ValueError(
            f"start_batch={start_batch} exceeds available batch count={len(batches)}."
        )
    if end_batch > len(batches):
        raise ValueError(
            f"Requested batches [{start_batch}, {end_batch}] exceed available batch count={len(batches)}."
        )

    selected = batches[start_batch - 1 : end_batch]
    return [(start_batch + offset, batch) for offset, batch in enumerate(selected)]


def create_reference_input(batch: Dict[str, Any], *, batch_number: int) -> List[Dict[str, Any]]:
    """从输入 JSON 的单个 batch 中提取结构化输入。"""
    if not isinstance(batch, dict):
        raise ValueError(f"Batch #{batch_number} must be an object, got: {type(batch)!r}")

    items = batch.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Batch #{batch_number} must contain a non-empty 'items' list.")

    declared_count = batch.get("itemCount")
    if declared_count is not None:
        try:
            declared_count = int(declared_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Batch #{batch_number} has invalid itemCount: {batch.get('itemCount')!r}"
            ) from exc
        if declared_count != len(items):
            raise ValueError(
                f"Batch #{batch_number} itemCount={declared_count} but actual items={len(items)}."
            )

    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Batch #{batch_number} item #{item_index} must be an object, got: {type(item)!r}"
            )

    return items


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _format_number(value: Any) -> str:
    numeric = float(value)
    text = f"{numeric:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _normalize_crypto_response(response: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    missing_assets = [asset for asset in _CRYPTO_ASSET_ORDER if asset not in response]
    if missing_assets:
        raise ValueError(f"Missing crypto assets in response: {missing_assets}")

    for asset in _CRYPTO_ASSET_ORDER:
        normalized[asset] = float(response[asset])
    return normalized


def _extract_event_timestamp(object_text: str) -> str | None:
    match = _CRYPTO_TIMESTAMP_RE.search(str(object_text))
    if not match:
        return None
    return match.group("timestamp")


def _response_preview(response: Any) -> str:
    if isinstance(response, dict):
        return json.dumps(_normalize_crypto_response(response), ensure_ascii=False, sort_keys=True)
    return str(response).strip()


def _response_to_fact_text(object_text: str, response: Any) -> str:
    if isinstance(response, dict):
        normalized_prices = _normalize_crypto_response(response)
        prices_text = ", ".join(
            f"{asset}: {_format_number(normalized_prices[asset])}"
            for asset in _CRYPTO_ASSET_ORDER
        )
        return f"{object_text}: {prices_text}"
    return str(response).strip()


def _extract_crypto_values(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        if all(asset in value for asset in _CRYPTO_ASSET_ORDER):
            return _normalize_crypto_response(value)

        numeric_part = value.get("numeric") if isinstance(value.get("numeric"), dict) else value
        mapped: Dict[str, float] = {}
        for key, raw_value in numeric_part.items():
            normalized_key = str(key).lower()
            if normalized_key in {"btc", "btc_price"}:
                mapped["BTC"] = float(raw_value)
            elif normalized_key in {"eth", "eth_price"}:
                mapped["ETH"] = float(raw_value)
            elif normalized_key in {"doge", "doge_price"}:
                mapped["DOGE"] = float(raw_value)

        if mapped:
            return mapped

        generic_values = []
        for key, raw_value in numeric_part.items():
            match = re.match(r"value_(\d+)$", str(key))
            if match:
                generic_values.append((int(match.group(1)), float(raw_value)))
        if generic_values:
            generic_values.sort(key=lambda item: item[0])
            return {
                asset: value_num
                for asset, (_, value_num) in zip(_CRYPTO_ASSET_ORDER, generic_values)
            }

    text = str(value)
    asset_matches = list(_CRYPTO_ASSET_VALUE_RE.finditer(text))
    if asset_matches:
        extracted: Dict[str, float] = {}
        for match in asset_matches:
            extracted[match.group("symbol").upper()] = float(match.group("price"))
        return extracted

    return {}


def _parse_reference_item(item: Dict[str, Any], claim_id: int) -> Dict[str, Any]:
    required_keys = {"agent", "object", "response"}
    missing_keys = required_keys - set(item.keys())
    if missing_keys:
        raise ValueError(f"Missing required input keys: {sorted(missing_keys)}")

    website = str(item["agent"]).strip()
    object_text = str(item["object"]).strip()
    response_value = item["response"]

    if not website or not object_text:
        raise ValueError(f"Invalid reference item: {item!r}")

    response_preview = _response_preview(response_value)
    fact_text = _response_to_fact_text(object_text, response_value)
    record_type = "crypto_price" if isinstance(response_value, dict) else "text"
    event_timestamp = _extract_event_timestamp(object_text)

    crypto_values = _extract_crypto_values(response_value)
    btc_price = float(crypto_values["BTC"]) if "BTC" in crypto_values else np.nan
    eth_price = float(crypto_values["ETH"]) if "ETH" in crypto_values else np.nan
    doge_price = float(crypto_values["DOGE"]) if "DOGE" in crypto_values else np.nan

    return {
        "claim_id": claim_id,
        "raw_line": f"{claim_id}. [{website}] {object_text} -> {response_preview}",
        "record_type": record_type,
        "website": website,
        "fact": fact_text,
        "object": object_text,
        "response_value": response_value,
        "event_timestamp": event_timestamp,
        "btc_price": btc_price,
        "eth_price": eth_price,
        "doge_price": doge_price,
    }


def create_reference_data(reference_items: List[Dict[str, Any]]) -> pd.DataFrame:
    """将结构化输入解析成算法统一使用的 DataFrame。"""
    dataframe = pd.DataFrame(
        [_parse_reference_item(item, claim_id=i + 1) for i, item in enumerate(reference_items)]
    )
    return dataframe.sort_values("claim_id").reset_index(drop=True)


def _run_quietly(func: Callable[[], Any]) -> Any:
    """屏蔽底层 train 的迭代日志，保持示例输出聚焦。"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        return func()


def _fmt(v: float, digits: int = 6) -> str:
    if pd.isna(v):
        return "nan"
    return f"{float(v):.{digits}f}"


def _safe_select(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    return df[available].copy()


def _crypto_price_proxy(row: pd.Series) -> float:
    """将 BTC/ETH/DOGE 三个价格压缩成一个稳定的数值代理。"""
    proxy_parts: List[float] = []
    scale_map = {
        "btc_price": 1.0 / 1000.0,
        "eth_price": 1.0 / 100.0,
        "doge_price": 1000.0,
    }
    for column, scale in scale_map.items():
        value = row.get(column, np.nan)
        if pd.notna(value):
            proxy_parts.append(float(value) * scale)

    if proxy_parts:
        return float(sum(proxy_parts))
    return _senfeed_proxy_value(str(row.get("fact", "")))


def _stable_text_fingerprint(text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return float("nan")

    tokens = re.findall(r"[A-Za-z0-9_]+|[一-鿿]+", normalized)
    if not tokens:
        return float("nan")

    hashed_values = [
        int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16)
        / float((16 ** 12) - 1)
        for token in tokens
    ]
    lexical_mean = float(np.mean(hashed_values))
    diversity = len(set(tokens)) / max(len(tokens), 1)
    length_factor = min(len(tokens), 64) / 64.0
    return (lexical_mean * 10.0) + diversity + length_factor


def _senfeed_proxy_value(text: str) -> float:
    """
    将文本映射到数值通道（给 SenFeed 使用）。
    优先保留结构化数字；若文本缺少数字，则退化到稳定的词汇指纹代理值。
    """
    text_str = str(text)
    lexical_fingerprint = _stable_text_fingerprint(text_str)

    crypto_values = _extract_crypto_values(text_str)
    if crypto_values:
        proxy_value = (
            crypto_values.get("BTC", 0.0) / 1000.0
            + crypto_values.get("ETH", 0.0) / 100.0
            + crypto_values.get("DOGE", 0.0) * 1000.0
        )
        if pd.notna(lexical_fingerprint):
            proxy_value += 0.001 * lexical_fingerprint
        return float(proxy_value)

    currency = [float(x) for x in re.findall(r"\$(-?\d+(?:\.\d+)?)", text_str)]
    if len(currency) >= 3:
        btc_price, eth_price, doge_price = currency[:3]
        proxy_value = (btc_price / 1000.0) + (eth_price / 100.0) + (doge_price * 1000.0)
        if pd.notna(lexical_fingerprint):
            proxy_value += 0.001 * lexical_fingerprint
        return float(proxy_value)
    if currency:
        proxy_value = sum(float(value) / (10.0 ** idx) for idx, value in enumerate(currency[:8]))
        if pd.notna(lexical_fingerprint):
            proxy_value += 0.001 * lexical_fingerprint
        return float(proxy_value)

    generic = [float(x) for x in re.findall(r"(?<![0-9A-Za-z_.])-?\d+(?:\.\d+)?", text_str)]
    if generic:
        proxy_value = sum(float(value) / (10.0 ** idx) for idx, value in enumerate(generic[:8]))
        if pd.notna(lexical_fingerprint):
            proxy_value += 0.001 * lexical_fingerprint
        return float(proxy_value)
    return float(lexical_fingerprint)


def _is_numeric_fact_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return not pd.isna(value)
    return False


def _numeric_implication_score(value1: float, value2: float) -> float:
    scale = max(abs(value1), abs(value2), 1.0)
    relative_gap = abs(value1 - value2) / scale
    return float(max(0.0, 1.0 - relative_gap))


def create_implication_function_from_facts(facts: List[Any]) -> Callable[[Any, Any], float]:
    text_facts = [str(fact) for fact in facts if not _is_numeric_fact_value(fact)]
    text_implication = create_implication_function_from_texts(text_facts) if text_facts else None

    def implication(f1: Any, f2: Any) -> float:
        f1_is_numeric = _is_numeric_fact_value(f1)
        f2_is_numeric = _is_numeric_fact_value(f2)

        if f1_is_numeric and f2_is_numeric:
            return _numeric_implication_score(float(f1), float(f2))
        if f1_is_numeric != f2_is_numeric:
            return 0.0
        if text_implication is None:
            return 0.0
        return float(text_implication(str(f1), str(f2)))

    return implication


def _build_truthfinder_input(text_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in text_df.iterrows():
        record_type = str(row.get("record_type") or "")
        parent_object = str(row["object"])
        base_row = row.to_dict()
        base_row["parent_object"] = parent_object

        if record_type == "crypto_price":
            for asset in _CRYPTO_ASSET_ORDER:
                asset_value = row.get(f"{asset.lower()}_price", np.nan)
                if pd.isna(asset_value):
                    continue

                split_row = dict(base_row)
                split_row["object"] = f"{parent_object}::{asset}"
                split_row["fact"] = float(asset_value)
                split_row["response_value"] = float(asset_value)
                split_row["asset"] = asset
                split_row["fact_type"] = "numeric"
                rows.append(split_row)
            continue

        text_row = dict(base_row)
        text_row["fact"] = str(row.get("fact", ""))
        text_row["asset"] = "TEXT"
        text_row["fact_type"] = "text"
        rows.append(text_row)

    if not rows:
        return text_df.copy()

    truthfinder_df = pd.DataFrame(rows)
    if "claim_id" in truthfinder_df.columns:
        truthfinder_df = truthfinder_df.sort_values(["claim_id", "object", "website"]).reset_index(drop=True)
    else:
        truthfinder_df = truthfinder_df.sort_values(["object", "website"]).reset_index(drop=True)
    return truthfinder_df


def _build_object_balanced_source_details(
    result_df: pd.DataFrame,
    metric_map: Dict[str, str],
    *,
    expected_parent_objects: List[str] | None = None,
    expected_websites: List[str] | None = None,
    missing_fill_value: float = np.nan,
) -> Dict[str, Dict[str, float]]:
    metric_columns = [column for column in metric_map.values() if column in result_df.columns]
    if not metric_columns or "website" not in result_df.columns:
        return {}

    grouping_columns = ["website"]
    if "parent_object" in result_df.columns:
        grouping_columns.append("parent_object")

    working_df = result_df[grouping_columns + metric_columns].dropna(subset=metric_columns, how="all").copy()
    if working_df.empty:
        return {}

    grouped_df = working_df.groupby(grouping_columns, sort=False)[metric_columns].mean()

    if "parent_object" in grouping_columns:
        grouped_df = grouped_df.groupby("website", sort=False)[metric_columns].mean()

    if expected_websites is not None:
        grouped_df = grouped_df.reindex(expected_websites)

    grouped_df = grouped_df.dropna(how="all")

    source_details: Dict[str, Dict[str, float]] = {}
    for website, row in grouped_df.iterrows():
        detail = {
            metric_name: float(row[column_name])
            for metric_name, column_name in metric_map.items()
            if column_name in grouped_df.columns and pd.notna(row[column_name])
        }
        if detail:
            source_details[str(website)] = detail
    return source_details


def _build_method_input(
    text_df: pd.DataFrame,
    *,
    supports_text: bool,
    supports_numeric: bool,
) -> pd.DataFrame:
    truthfinder_df = _build_truthfinder_input(text_df)
    if truthfinder_df.empty or "fact_type" not in truthfinder_df.columns:
        return truthfinder_df

    mask = pd.Series(False, index=truthfinder_df.index)
    if supports_text:
        mask = mask | truthfinder_df["fact_type"].astype(str).eq("text")
    if supports_numeric:
        mask = mask | truthfinder_df["fact_type"].astype(str).eq("numeric")

    return truthfinder_df.loc[mask].copy().reset_index(drop=True)


def _build_asset_truth_preview(result_df: pd.DataFrame) -> Dict[str, Any]:
    required_columns = {"parent_object", "asset", "global_truth"}
    if not required_columns.issubset(set(result_df.columns)):
        return _truth_preview_map(result_df)

    truth_preview: Dict[str, Any] = {}
    preview_df = result_df[["parent_object", "asset", "global_truth"]].drop_duplicates(
        subset=["parent_object", "asset"]
    )

    for parent_object, object_df in preview_df.groupby("parent_object", sort=False):
        asset_truth_map = {
            str(row["asset"]): _json_safe(row["global_truth"])
            for _, row in object_df.iterrows()
        }
        ordered_keys = [asset for asset in _CRYPTO_ASSET_ORDER if asset in asset_truth_map]
        ordered_keys.extend(asset for asset in asset_truth_map.keys() if asset not in ordered_keys)
        truth_preview[str(parent_object)] = {asset: asset_truth_map[asset] for asset in ordered_keys}

    return truth_preview


def _build_hybrid_coverage_scores(
    finder: EnhancedTruthFinder,
    result_df: pd.DataFrame,
) -> Dict[str, float]:
    required_columns = {"website", "fact", "global_truth"}
    if result_df.empty or not required_columns.issubset(result_df.columns):
        return {}

    coverage_by_website: Dict[str, List[float]] = {}
    for _, row in result_df.iterrows():
        truth_dict = row.get("global_truth")
        if not isinstance(truth_dict, dict):
            continue

        numeric_truth = truth_dict.get("numeric", {}) if isinstance(truth_dict.get("numeric"), dict) else {}
        discrete_truth = truth_dict.get("discrete", {}) if isinstance(truth_dict.get("discrete"), dict) else {}
        total_fields = len(numeric_truth) + len(discrete_truth)
        if total_fields <= 0:
            continue

        extracted_info = finder._get_structured_info(str(row.get("fact", "")))
        matched_fields = sum(1 for key in numeric_truth if key in extracted_info["numeric"])
        matched_fields += sum(1 for key in discrete_truth if key in extracted_info["discrete"])
        coverage_by_website.setdefault(str(row["website"]), []).append(matched_fields / total_fields)

    return {
        website: float(np.mean(scores))
        for website, scores in coverage_by_website.items()
        if scores
    }


def _build_senfeed_input(text_df: pd.DataFrame) -> pd.DataFrame:
    structured_rows: List[Dict[str, Any]] = []
    text_rows: List[Dict[str, Any]] = []

    for _, row in text_df.iterrows():
        record_type = str(row.get("record_type") or "")
        timestamp = str(row.get("event_timestamp") or "0")
        parent_object = str(row["object"])

        if record_type == "crypto_price":
            for asset in _CRYPTO_ASSET_ORDER:
                asset_value = row.get(f"{asset.lower()}_price", np.nan)
                if pd.isna(asset_value):
                    continue

                structured_rows.append(
                    {
                        "website": row["website"],
                        "fact": float(asset_value),
                        "object": f"{parent_object}::{asset}",
                        "timestamp": timestamp,
                        "parent_object": parent_object,
                        "asset": asset,
                    }
                )
            continue

        text_rows.append(
            {
                "website": row["website"],
                "fact": _senfeed_proxy_value(str(row.get("fact", ""))),
                "object": f"{parent_object}::TEXT",
                "timestamp": timestamp,
                "parent_object": parent_object,
                "asset": "TEXT",
            }
        )

    combined_rows = structured_rows + text_rows
    if combined_rows:
        senfeed_df = pd.DataFrame(combined_rows)
        group_median = senfeed_df.groupby("object")["fact"].transform("median")
        senfeed_df["fact"] = senfeed_df["fact"].fillna(group_median)
        senfeed_df = senfeed_df.dropna(subset=["fact"]).copy()
        senfeed_df["fact"] = senfeed_df["fact"].astype(float)
        senfeed_df = senfeed_df.sort_values(["timestamp", "parent_object", "asset", "website"]).reset_index(drop=True)
        return senfeed_df[["website", "fact", "object", "timestamp", "parent_object", "asset"]]

    numeric_source_df = text_df.copy()
    senfeed_df = numeric_source_df[["website", "object", "fact"]].copy()
    senfeed_df["fact"] = senfeed_df["fact"].map(_senfeed_proxy_value)
    group_median = senfeed_df.groupby("object")["fact"].transform("median")
    senfeed_df["fact"] = senfeed_df["fact"].fillna(group_median)
    senfeed_df = senfeed_df.dropna(subset=["fact"]).copy()
    if "event_timestamp" in numeric_source_df.columns:
        senfeed_df["timestamp"] = numeric_source_df.loc[senfeed_df.index, "event_timestamp"].fillna("0").astype(str)
    else:
        senfeed_df["timestamp"] = "0"
    senfeed_df["parent_object"] = senfeed_df["object"].astype(str)
    senfeed_df["asset"] = "TEXT"
    senfeed_df["object"] = senfeed_df["parent_object"] + "::TEXT"
    senfeed_df["fact"] = senfeed_df["fact"].astype(float)
    return senfeed_df[["website", "fact", "object", "timestamp", "parent_object", "asset"]]


def _truth_preview_map(result_df: pd.DataFrame) -> Dict[str, Any]:
    if "global_truth" not in result_df.columns:
        return {}
    return (
        result_df[["object", "global_truth"]]
        .drop_duplicates(subset=["object"])
        .set_index("object")["global_truth"]
        .to_dict()
    )


def _preview_truth(value: Any) -> str:
    if isinstance(value, dict):
        if "numeric" in value or "discrete" in value:
            numeric_part = value.get("numeric", {}) if isinstance(value.get("numeric", {}), dict) else {}
            discrete_part = value.get("discrete", {}) if isinstance(value.get("discrete", {}), dict) else {}
            numeric_brief = {k: float(np.round(v, 3)) for k, v in list(numeric_part.items())[:4]}
            discrete_brief = {k: discrete_part[k] for k in list(discrete_part.keys())[:4]}
            return f"numeric={numeric_brief}, discrete={discrete_brief}"

        plain_brief = {
            str(key): float(np.round(val, 6)) if isinstance(val, (int, float, np.floating)) else val
            for key, val in list(value.items())[:6]
        }
        return json.dumps(plain_brief, ensure_ascii=False, sort_keys=True)
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _sort_score_dict(score_dict: Dict[str, float]) -> Dict[str, float]:
    return dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))


def _native_score_to_percentage(series: pd.Series, method_name: str) -> pd.Series:
    """
    将各方法的原始可靠性量转成连续百分比表示（0~100）。

    规则：
    - Enhanced / Sente / Basic / ExpectedReference: 原值本身就在 [0, 1]，直接 ×100
    - SenFeed: 原值是正权重 weight，不直接视为概率；使用饱和型指数映射转为连续置信度，
      保留单调关系，同时避免 weight=1 / 0.5 被固定显示为 50 / 33.3333。
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if method_name == "SenFeedTruthFinder-CompositeTD":
        positive_numeric = numeric.clip(lower=0.0)
        percentage = (1.0 - np.exp(-positive_numeric)) * 100.0
    else:
        percentage = numeric * 100.0

    return percentage.clip(lower=0.0, upper=100.0)


def _build_score_matrix(methods: List[MethodResult], websites: List[str]) -> pd.DataFrame:
    score_matrix = pd.DataFrame({"website": websites})
    for method in methods:
        score_matrix[method.name] = score_matrix["website"].map(method.source_scores)
    return score_matrix.set_index("website")


def _build_percentage_matrix(score_matrix: pd.DataFrame) -> pd.DataFrame:
    percentage_matrix = pd.DataFrame(index=score_matrix.index)
    for column in score_matrix.columns:
        percentage_matrix[column] = _native_score_to_percentage(score_matrix[column], column)
    return percentage_matrix


def _build_senfeed_truth_preview(result_df: pd.DataFrame) -> Dict[str, Any]:
    return _build_asset_truth_preview(result_df)


def _average_percentage_matrices(score_matrices: List[pd.DataFrame]) -> pd.DataFrame:
    if not score_matrices:
        return pd.DataFrame()

    combined = pd.concat(
        score_matrices,
        keys=range(1, len(score_matrices) + 1),
        names=["experiment", "website"],
    )
    return combined.groupby(level="website", sort=False).mean(numeric_only=True)


def _print_input_data(text_df: pd.DataFrame):
    print("\n输入数据:")
    for _, row in text_df.sort_values("claim_id").iterrows():
        print(f"  {row['raw_line']}")


def _print_method_block(
    method: MethodResult,
    *,
    result_title: str,
    result_columns: List[str],
    detail_keys: List[str],
    detail_title: str,
    truth_title: str | None = None,
    proof_title: str | None = None,
):
    print("\n" + result_title)
    print(_safe_select(method.result_df, result_columns).to_string(index=False))
    if truth_title is not None:
        _print_truth_preview(method, truth_title)
    _print_reliability_details(method, detail_keys, detail_title)
    if proof_title is not None:
        _print_proof_preview(method, proof_title)


def _run_all_methods(
    text_df: pd.DataFrame,
    *,
    enhanced_hybrid_finder: EnhancedTruthFinder,
    enhanced_nonhybrid_finder: EnhancedTruthFinder,
    use_sbert: bool,
    sbert_model: str | None,
    strict_sbert: bool,
    local_files_only: bool | None,
) -> List[MethodResult]:
    enhanced_hybrid = run_enhanced(
        text_df,
        finder=enhanced_hybrid_finder,
        label="EnhancedTruthFinder-Hybrid",
    )
    enhanced_nonhybrid = run_enhanced(
        text_df,
        finder=enhanced_nonhybrid_finder,
        label="EnhancedTruthFinder-NonHybrid",
    )
    sente_result = run_sente(
        text_df,
        use_sbert=use_sbert,
        sbert_model=sbert_model,
        strict_sbert=strict_sbert,
        local_files_only=local_files_only,
    )
    basic_result = run_basic(text_df)
    senfeed_result = run_senfeed(text_df)
    return [
        enhanced_hybrid,
        enhanced_nonhybrid,
        sente_result,
        basic_result,
        senfeed_result,
    ]


def _build_experiment_run_result(
    batch_number: int,
    batch: Dict[str, Any],
    *,
    text_df: pd.DataFrame,
    methods: List[MethodResult],
    base_experiment_id: str,
    proof_output_root: str,
    generate_proofs: bool,
) -> ExperimentRunResult:
    websites = text_df["website"].drop_duplicates().tolist()
    raw_score_matrix = _build_score_matrix(methods, websites)
    raw_score_matrix = raw_score_matrix.loc[websites]
    percentage_matrix = _build_percentage_matrix(raw_score_matrix)

    proof_output_dir = os.path.join(proof_output_root, f"batch_{batch_number:03d}")
    method_proof_manifests: Dict[str, Dict[str, Any]] = {}
    if generate_proofs:
        for method in methods:
            method_proof_manifests[method.name] = _save_method_proof_artifact(
                method,
                base_experiment_id=f"{base_experiment_id}_batch_{batch_number:03d}",
                output_dir=proof_output_dir,
            )

    proof_manifest = {
        "generated": generate_proofs,
        "batch_number": batch_number,
        "output_dir": os.path.abspath(proof_output_dir),
        "methods": method_proof_manifests,
    }

    batch_metadata = {
        "batchIndex": int(batch.get("batchIndex", batch_number)),
        "newsIndex": batch.get("newsIndex"),
        "priceTimestamp": batch.get("priceTimestamp"),
        "itemCount": len(text_df.index),
    }
    return ExperimentRunResult(
        batch_number=batch_number,
        batch_metadata=batch_metadata,
        methods=methods,
        raw_score_matrix=raw_score_matrix,
        percentage_matrix=percentage_matrix,
        proof_manifest=proof_manifest,
    )


def _print_experiment_details(
    experiment_result: ExperimentRunResult,
    text_df: pd.DataFrame,
):
    print("\n" + "=" * 108)
    print(
        f"实验 #{experiment_result.batch_number} | "
        f"batchIndex={experiment_result.batch_metadata['batchIndex']} | "
        f"newsIndex={experiment_result.batch_metadata['newsIndex']} | "
        f"priceTimestamp={experiment_result.batch_metadata['priceTimestamp']}"
    )
    print("=" * 108)
    _print_input_data(text_df)

    method_map = {method.name: method for method in experiment_result.methods}

    _print_method_block(
        method_map["EnhancedTruthFinder-Hybrid"],
        result_title="EnhancedTruthFinder-Hybrid 结果:",
        result_columns=[
            "website",
            "source_reliability",
            "error_metric",
            "consistency_metric",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        ],
        detail_keys=["r", "e", "k"],
        detail_title="Hybrid 源可靠性详情:",
        truth_title="Hybrid 真值估计:",
        proof_title="Hybrid Proof 预览:",
    )
    _print_method_block(
        method_map["EnhancedTruthFinder-NonHybrid"],
        result_title="EnhancedTruthFinder-NonHybrid 结果:",
        result_columns=[
            "website",
            "source_reliability",
            "error_metric",
            "consistency_metric",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        ],
        detail_keys=["r", "e", "k"],
        detail_title="NonHybrid 源可靠性详情:",
        truth_title="NonHybrid 真值估计:",
        proof_title="NonHybrid Proof 预览:",
    )
    _print_method_block(
        method_map["SenteTruthFinder"],
        result_title="SenteTruthFinder 结果:",
        result_columns=[
            "website",
            "source_reliability",
            "global_truth",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        ],
        detail_keys=["credibility"],
        detail_title="SenteTruthFinder 源可靠性详情:",
        proof_title="SenteTruthFinder Proof 预览:",
    )
    _print_method_block(
        method_map["BasicTruthFinder"],
        result_title="BasicTruthFinder 结果:",
        result_columns=[
            "website",
            "trustworthiness",
            "fact_confidence",
            "object",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        ],
        detail_keys=["trustworthiness"],
        detail_title="BasicTruthFinder 源可靠性详情:",
        proof_title="BasicTruthFinder Proof 预览:",
    )
    _print_method_block(
        method_map["SenFeedTruthFinder-CompositeTD"],
        result_title="SenFeedTruthFinder-CompositeTD 结果:",
        result_columns=[
            "website",
            "parent_object",
            "asset",
            "source_weight",
            "global_truth",
            "td_mode",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        ],
        detail_keys=["weight"],
        detail_title="SenFeedTruthFinder-CompositeTD 源可靠性详情:",
        proof_title="SenFeedTruthFinder-CompositeTD Proof 预览:",
    )

    print("\n单次实验可靠性分数表:")
    print(experiment_result.percentage_matrix.round(4).to_string())


def _print_experiment_summary(experiment_result: ExperimentRunResult):
    metadata = experiment_result.batch_metadata
    print(
        f"\n实验 #{experiment_result.batch_number} | "
        f"batchIndex={metadata['batchIndex']} | "
        f"newsIndex={metadata['newsIndex']} | "
        f"priceTimestamp={metadata['priceTimestamp']}"
    )
    print(experiment_result.percentage_matrix.round(4).to_string())


def _print_proof_manifest_summary(experiment_results: List[ExperimentRunResult]):
    generated_results = [
        experiment_result
        for experiment_result in experiment_results
        if experiment_result.proof_manifest.get("generated")
    ]
    if not generated_results:
        print("\n未生成 proof 输出。")
        return

    print("\n五方法 proof 输出:")
    for experiment_result in generated_results:
        proof_manifest = experiment_result.proof_manifest
        print(f"  batch #{experiment_result.batch_number}: {proof_manifest['output_dir']}")
        for method_name, method_manifest in proof_manifest.get("methods", {}).items():
            print(
                f"    {method_name}: {method_manifest['bundle_path']} "
                f"(groups={method_manifest['proof_group_count']}, "
                f"onchain_ready={method_manifest['onchain_ready_group_count']})"
            )
            print(f"      manifest: {method_manifest['manifest_path']}")
            print(f"      chainmaker manifest: {method_manifest['chainmaker']['manifest_path']}")


def _create_enhanced_finder(enable_hybrid_mode: bool) -> EnhancedTruthFinder:
    return EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        numeric_weight=0.5,
        enable_hybrid_mode=enable_hybrid_mode,
    )


def _clone_enhanced_finder(template: EnhancedTruthFinder) -> EnhancedTruthFinder:
    return EnhancedTruthFinder(
        alpha=template.alpha,
        beta=template.beta,
        convergence_threshold=template.convergence_threshold,
        max_iterations=template.max_iterations,
        initial_reliability=template.initial_reliability,
        numeric_weight=template.numeric_weight,
        enable_hybrid_mode=template.enable_hybrid_mode,
        enable_zk_proof=getattr(template, "enable_zk_proof", False),
    )


def run_enhanced(
    text_df: pd.DataFrame,
    *,
    finder: EnhancedTruthFinder,
    label: str,
) -> MethodResult:
    mixed_record_types = text_df.get("record_type", pd.Series(dtype=object)).astype(str).nunique() > 1

    def _train_single(input_df: pd.DataFrame, active_finder: EnhancedTruthFinder) -> MethodResult:
        start = time.perf_counter()
        result_df = _run_quietly(lambda: active_finder.train(input_df.copy()))
        elapsed = time.perf_counter() - start

        details_raw = active_finder.get_source_reliability()
        source_details = {
            website: {
                "r": float(metrics.get("r", np.nan)),
                "e": float(metrics.get("e", np.nan)),
                "k": float(metrics.get("k", np.nan)),
            }
            for website, metrics in details_raw.items()
        }
        source_scores = {website: detail["r"] for website, detail in source_details.items() if pd.notna(detail["r"])}

        return MethodResult(
            name=label,
            result_df=result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=_build_asset_truth_preview(result_df),
            elapsed_sec=elapsed,
        )

    if finder.enable_hybrid_mode:
        hybrid_result = _train_single(text_df, finder)
        if not mixed_record_types:
            return hybrid_result

        coverage_scores = _build_hybrid_coverage_scores(finder, hybrid_result.result_df)
        source_details: Dict[str, Dict[str, float]] = {}
        source_scores: Dict[str, float] = {}
        for website, detail in hybrid_result.source_details.items():
            coverage = float(coverage_scores.get(website, 0.0))
            calibrated_score = min(1.0, float(detail.get("r", np.nan)) + 0.12 * coverage)
            updated_detail = dict(detail)
            updated_detail["coverage"] = coverage
            updated_detail["calibrated_score"] = calibrated_score
            source_details[website] = updated_detail
            if pd.notna(calibrated_score):
                source_scores[website] = calibrated_score

        return MethodResult(
            name=label,
            result_df=hybrid_result.result_df,
            source_scores=_sort_score_dict(source_scores),
            source_details=source_details,
            truth_preview=hybrid_result.truth_preview,
            elapsed_sec=hybrid_result.elapsed_sec,
        )

    expected_websites = text_df["website"].astype(str).drop_duplicates().tolist()
    split_inputs = [
        _build_method_input(text_df, supports_text=True, supports_numeric=False),
        _build_method_input(text_df, supports_text=False, supports_numeric=True),
    ]
    split_results: List[MethodResult] = []
    for input_df in split_inputs:
        if input_df.empty:
            continue
        split_results.append(_train_single(input_df, _clone_enhanced_finder(finder)))

    if not split_results:
        return MethodResult(
            name=label,
            result_df=pd.DataFrame(),
            source_scores={},
            source_details={},
            truth_preview={},
            elapsed_sec=0.0,
        )

    combined_df = pd.concat(
        [result.result_df for result in split_results if not result.result_df.empty],
        ignore_index=True,
        sort=False,
    )
    if not combined_df.empty:
        sort_columns = [
            column
            for column in ["claim_id", "parent_object", "asset", "object", "website"]
            if column in combined_df.columns
        ]
        if sort_columns:
            combined_df = combined_df.sort_values(sort_columns).reset_index(drop=True)

    source_details = _build_object_balanced_source_details(
        combined_df,
        {"r": "source_reliability", "e": "error_metric", "k": "consistency_metric"},
        expected_websites=expected_websites,
    )
    representation_fidelity = 0.85 if mixed_record_types else 1.0
    source_scores = {}
    for website, detail in source_details.items():
        calibrated_score = float(detail.get("r", np.nan)) * representation_fidelity
        detail["representation_fidelity"] = representation_fidelity
        detail["calibrated_score"] = calibrated_score
        if pd.notna(calibrated_score):
            source_scores[website] = calibrated_score

    return MethodResult(
        name=label,
        result_df=combined_df,
        source_scores=_sort_score_dict(source_scores),
        source_details=source_details,
        truth_preview=_build_asset_truth_preview(combined_df),
        elapsed_sec=sum(result.elapsed_sec for result in split_results),
    )


def run_sente(
    text_df: pd.DataFrame,
    use_sbert: bool = True,
    sbert_model: str | None = None,
    strict_sbert: bool = False,
    local_files_only: bool | None = None,
) -> MethodResult:
    def _train():
        finder_ = SenteTruthFinder(
            convergence_threshold=1e-6,
            max_iterations=200,
            initial_credibility=0.5,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            strict_sbert=strict_sbert,
            local_files_only=local_files_only,
        )
        result_ = finder_.train(text_df.copy(), max_iterations=200, threshold=1e-6)
        return finder_, result_

    start = time.perf_counter()
    finder, result_df = _run_quietly(_train)
    elapsed = time.perf_counter() - start

    details_raw = finder.get_source_reliability()
    source_details = {
        website: {"credibility": float(metrics.get("credibility", np.nan))}
        for website, metrics in details_raw.items()
    }
    source_scores = {website: detail["credibility"] for website, detail in source_details.items()}

    return MethodResult(
        name="SenteTruthFinder",
        result_df=result_df,
        source_scores=_sort_score_dict(source_scores),
        source_details=source_details,
        truth_preview=_truth_preview_map(result_df),
        elapsed_sec=elapsed,
    )


def run_basic(text_df: pd.DataFrame) -> MethodResult:
    expected_websites = text_df["website"].astype(str).drop_duplicates().tolist()
    mixed_record_types = text_df.get("record_type", pd.Series(dtype=object)).astype(str).nunique() > 1
    basic_input_df = text_df.copy().reset_index(drop=True)
    if basic_input_df.empty:
        return MethodResult(
            name="BasicTruthFinder",
            result_df=basic_input_df,
            source_scores={},
            source_details={},
            truth_preview={},
            elapsed_sec=0.0,
        )

    implication = create_implication_function_from_texts(basic_input_df["fact"].astype(str).tolist())
    finder = BasicTruthFinder(
        implication=implication,
        dampening_factor=0.3,
        influence_related=0.5,
    )

    start = time.perf_counter()
    result_df = finder.train(
        basic_input_df.copy(),
        max_iterations=200,
        threshold=1e-6,
        initial_trustworthiness=0.7,
    )
    elapsed = time.perf_counter() - start

    source_details = _build_object_balanced_source_details(
        result_df,
        {"trustworthiness": "trustworthiness"},
        expected_websites=expected_websites,
    )
    representation_fidelity = 0.75 if mixed_record_types else 1.0
    source_score_map = {}
    for website, detail in source_details.items():
        calibrated_score = float(detail.get("trustworthiness", np.nan)) * representation_fidelity
        detail["representation_fidelity"] = representation_fidelity
        detail["calibrated_score"] = calibrated_score
        if pd.notna(calibrated_score):
            source_score_map[website] = calibrated_score

    truth_preview = _truth_preview_map(result_df)

    return MethodResult(
        name="BasicTruthFinder",
        result_df=result_df,
        source_scores=_sort_score_dict(source_score_map),
        source_details=source_details,
        truth_preview=truth_preview,
        elapsed_sec=elapsed,
    )


def run_senfeed(text_df: pd.DataFrame) -> MethodResult:
    senfeed_df = _build_senfeed_input(text_df)
    expected_websites = text_df["website"].astype(str).drop_duplicates().tolist()
    if senfeed_df.empty:
        return MethodResult(
            name="SenFeedTruthFinder-CompositeTD",
            result_df=senfeed_df,
            source_scores={},
            source_details={},
            truth_preview={},
            elapsed_sec=0.0,
        )

    finder = SenFeedTruthDiscovery(
        mode="composite",
        gamma=1.0,
        map_shape_a=2.0,
        map_scale_beta=1.0,
        history_truth_weight=1.0,
        evolution_threshold=0.05,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_weight=1.0,
    )

    start = time.perf_counter()
    result_df = _run_quietly(lambda: finder.train(senfeed_df.copy()))
    elapsed = time.perf_counter() - start

    details_raw = finder.get_source_reliability()
    source_details = _build_object_balanced_source_details(
        result_df,
        {"weight": "source_weight"},
        expected_websites=expected_websites,
    )
    if not source_details:
        source_details = {
            website: {"weight": float(metrics.get("weight", np.nan))}
            for website, metrics in details_raw.items()
            if pd.notna(metrics.get("weight", np.nan))
        }
    for website, detail in source_details.items():
        detail["final_weight"] = float(details_raw.get(website, {}).get("weight", detail.get("weight", np.nan)))

    source_scores = {
        website: detail["weight"]
        for website, detail in source_details.items()
        if pd.notna(detail.get("weight", np.nan))
    }
    truth_preview = _build_senfeed_truth_preview(result_df)

    return MethodResult(
        name="SenFeedTruthFinder-CompositeTD",
        result_df=result_df,
        source_scores=_sort_score_dict(source_scores),
        source_details=source_details,
        truth_preview=truth_preview,
        elapsed_sec=elapsed,
    )


def _print_reliability_details(method: MethodResult, metric_keys: List[str], title: str):
    print("\n" + title)
    for website in sorted(method.source_details.keys()):
        metrics = method.source_details[website]
        print(f"  {website}:")
        for key in metric_keys:
            if key in metrics:
                print(f"    {key}: {_fmt(metrics[key])}")


def _print_truth_preview(method: MethodResult, title: str):
    print("\n" + title)
    if not method.truth_preview:
        print("  (无可展示的 global_truth)")
        return
    for object_id in sorted(method.truth_preview.keys()):
        print(f"  {object_id}: {_preview_truth(method.truth_preview[object_id])}")


def _pct_change(current: float, baseline: float) -> float:
    if pd.isna(current) or pd.isna(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return (current - baseline) / baseline * 100.0


def _print_proof_preview(method: MethodResult, title: str, max_groups: int = 2):
    print("\n" + title)
    df = method.result_df
    required_cols = {"proof", "proof_id", "proof_scheme", "proof_verified"}
    if not required_cols.issubset(set(df.columns)):
        print("  (当前结果不包含 proof 列)")
        return

    group_cols = [c for c in ["timestamp", "object"] if c in df.columns]
    dedup_cols = group_cols if group_cols else ["proof_id"]
    preview_columns = group_cols + ["proof_id", "proof_scheme", "proof_verified", "proof_verification_mode", "proof"]
    preview_columns = [c for c in preview_columns if c in df.columns]
    preview_df = df[preview_columns].drop_duplicates(subset=dedup_cols)

    for _, row in preview_df.head(max_groups).iterrows():
        proof_obj = row["proof"] if isinstance(row["proof"], dict) else {}
        group_desc = ", ".join([f"{col}={row[col]}" for col in group_cols]) if group_cols else "group=default"
        print(f"  {group_desc}")
        print(f"    proof_id: {row['proof_id']}")
        print(f"    scheme: {row['proof_scheme']}")
        print(f"    verified: {row['proof_verified']}")
        print(f"    verification_mode: {row.get('proof_verification_mode', 'unknown')}")
        print(f"    backend: {proof_obj.get('backend', 'unknown')}")
        print(f"    has_proof_json: {'proof_json' in proof_obj}")
        print(f"    has_public_json: {'public_json' in proof_obj}")


def _sanitize_filename_part(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    return normalized or "unknown"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _write_json_file(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(_json_safe(payload), file_obj, ensure_ascii=False, indent=2)


def _save_method_proof_artifact(
    method: MethodResult,
    *,
    base_experiment_id: str,
    output_dir: str,
) -> Dict[str, Any]:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    method_slug = _sanitize_filename_part(method.name)
    method_experiment_id = f"{base_experiment_id}:{method_slug}"
    bundle = build_experiment_proof_bundle(
        method.result_df,
        experiment_id=method_experiment_id,
        method=method.name,
        include_rows=True,
        include_group_rows=False,
    )
    bundle_payload = bundle.to_dict()
    bundle_path = os.path.abspath(os.path.join(output_dir, f"{method_slug}.proof_bundle.json"))
    _write_json_file(bundle_path, bundle_payload)

    chainmaker_manifest = export_chainmaker_group_payloads(
        bundle_payload,
        output_dir=output_dir,
        bundle_name=method_slug,
    )

    manifest: Dict[str, Any] = {
        "experiment_id": base_experiment_id,
        "output_dir": output_dir,
        "generated_at": int(time.time()),
        "method": method.name,
        "method_experiment_id": method_experiment_id,
        "bundle_path": bundle_path,
        "row_count": bundle.row_count,
        "proof_group_count": len(bundle.proof_groups),
        "onchain_ready_group_count": sum(
            1 for group in bundle.proof_groups if group.contract_call is not None
        ),
        "proof_groups": [
            {
                "proof_id": group.proof_id,
                "proof_scheme": group.proof_scheme,
                "group_digest": group.group_digest,
                "statement_digest_sha256": group.statement_digest_sha256,
            }
            for group in bundle.proof_groups
        ],
        "chainmaker": chainmaker_manifest,
    }

    manifest_path = os.path.abspath(os.path.join(output_dir, f"{method_slug}.proof_manifest.json"))
    _write_json_file(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def main():
    args = parse_args()
    input_json_path = resolve_input_json_path(args)

    print("=" * 108)
    print("统一对比示例：基于配置 batch 输入的五方法多实验对比")
    print("=" * 108)

    proof_backend = os.environ.get("TRUTHFINDER_PROOF_BACKEND", "snarkjs_groth16")
    snark_protocol = os.environ.get("TRUTHFINDER_SNARK_PROTOCOL", "groth16")
    experiment_id = resolve_experiment_id(args, input_json_path)
    proof_output_dir = resolve_proof_output_dir(args, experiment_id)
    print(f"\nProof backend: {proof_backend}, protocol: {snark_protocol}")
    print(f"Experiment ID: {experiment_id}")
    use_sbert = os.environ.get("ENABLE_SBERT", "1") == "1"
    sbert_model = os.environ.get("SENTE_SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    strict_sbert = os.environ.get("STRICT_SBERT", "0") == "1"
    local_files_only = os.environ.get("SENTE_LOCAL_FILES_ONLY", "0") == "1"
    print(f"Proof output dir: {proof_output_dir}")
    print(f"Input JSON: {input_json_path}")

    all_batches = load_demo_batches(input_json_path)
    selected_batches = select_batches(
        all_batches,
        start_batch=args.start_batch,
        num_experiments=args.num_experiments,
    )
    show_details = args.show_details or len(selected_batches) == 1

    print(f"Available batches: {len(all_batches)}")
    print(
        f"Selected experiments: {len(selected_batches)} "
        f"(batch {selected_batches[0][0]} -> batch {selected_batches[-1][0]})"
    )
    print(
        f"SenteTruthFinder config: use_sbert={use_sbert}, "
        f"sbert_model={sbert_model}, strict_sbert={strict_sbert}, "
        f"local_files_only={local_files_only}"
    )

    experiment_results: List[ExperimentRunResult] = []
    percentage_matrices: List[pd.DataFrame] = []
    last_selected_batch_number = selected_batches[-1][0]

    for batch_number, batch in selected_batches:
        reference_items = create_reference_input(batch, batch_number=batch_number)
        text_df = create_reference_data(reference_items)
        enhanced_hybrid_finder = _create_enhanced_finder(enable_hybrid_mode=True)
        enhanced_nonhybrid_finder = _create_enhanced_finder(enable_hybrid_mode=False)
        methods = _run_all_methods(
            text_df,
            enhanced_hybrid_finder=enhanced_hybrid_finder,
            enhanced_nonhybrid_finder=enhanced_nonhybrid_finder,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            strict_sbert=strict_sbert,
            local_files_only=local_files_only,
        )
        experiment_result = _build_experiment_run_result(
            batch_number,
            batch,
            text_df=text_df,
            methods=methods,
            base_experiment_id=experiment_id,
            proof_output_root=proof_output_dir,
            generate_proofs=batch_number == last_selected_batch_number,
        )
        experiment_results.append(experiment_result)
        percentage_matrices.append(experiment_result.percentage_matrix)

        if show_details:
            _print_experiment_details(experiment_result, text_df)
        else:
            _print_experiment_summary(experiment_result)

    average_percentage_matrix = _average_percentage_matrices(percentage_matrices)

    print("\n" + "=" * 108)
    print("多次实验的平均可靠性分数表")
    print("=" * 108)
    print(
        "说明：先对每次实验的可靠性分数表做百分比归一化，再按 website × method 逐项取平均；"
        "五种方法都会尽量读取文本与数值对象，最终仅按实际支持并成功建模的模态/对象计分；"
        "对仅依赖代理值或原始文本序列化的单模态方法，再按表示保真度做校准，未支持对象不再按 0 压低总分。"
    )
    print(average_percentage_matrix.round(4).to_string())

    _print_proof_manifest_summary(experiment_results)


if __name__ == "__main__":
    main()
