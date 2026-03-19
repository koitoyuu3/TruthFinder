#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional

from qwen_agent.agents import Assistant

# TODO: 调用工具获取数据，然后是langexact获取文本

DEFAULT_INPUT = Path(__file__).resolve().with_name("step2_price_news_summaries.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("step3_news_scores.csv")

AGENT_NAMES = ["NewsOracleAgent-1", "NewsOracleAgent-2", "NewsOracleAgent-3"]
LLM_MODEL = "qwen2.5:1.5b"
LLM_SERVER = "http://127.0.0.1:11434/v1"
LLM_API_KEY = "ollama"
SCORE_RANGE = 5.0


@dataclass(frozen=True)
class NewsRow:
    ticker: str
    date: str
    close_price: Decimal
    summary: str
    titles: str
    bodies: str


@dataclass(frozen=True)
class AgentScore:
    sentiment: str
    reasoning: List[str]

class NewsScoringAgent:
    def __init__(self, agent_name: str) -> None:
        system = (
            f"你是{agent_name}，负责根据股票新闻摘要给出打分。\n"
            "- 只能返回一个JSON字符串，不能包含其他文本。\n"
            '- JSON格式：{"sentiment": "积极|消极|中性", "reasoning": ["理由1: ...", "理由2: ..."]}\n'
            "- 统一情绪标准：\n"
            "  * 积极：摘要明确出现业绩增长、盈利改善、市场份额提升、监管利好等强烈利好信号。\n"
            "  * 消极：出现业绩下滑、重大风险/诉讼、市场份额流失、监管不利等明显利空。\n"
            "  * 中性：仅描述事实、利好利空混杂或信息不足以判断方向。\n"
            "- 若信息不足，sentiment必须为中性并在reasoning中说明原因。\n"
            "- reasoning 必须以列表形式返回1条证据，格式为“理由: 摘要中的具体事实或数字”，只能用中文。\n"
            "- 严格遵守上述格式。"
        )
        self.name = agent_name
        self.assistant = Assistant(
            llm={"model": LLM_MODEL, "model_server": LLM_SERVER, "api_key": LLM_API_KEY},
            name=agent_name,
            description=f"{agent_name} news scoring agent",
            system_message=system,
        )

    def score(self, row: NewsRow) -> Optional[AgentScore]:
        user_prompt = (
            f"Ticker: {row.ticker}\n"
            f"Date: {row.date}\n"
            f"Close Price: {row.close_price}\n"
            f"Summary: {row.summary}\n"
            "请按照系统要求给出JSON。"
        )
        response = self._run([{"role": "user", "content": user_prompt}])
        payload = self._extract_payload(response)
        if payload is None:
            return None
        return self._build_score(payload)

    def _run(self, messages: List[dict], max_retries: int = 2) -> Optional[List[dict]]:
        last_response: Optional[List[dict]] = None
        for _ in range(max_retries):
            for chunk in self.assistant.run(messages):
                last_response = chunk
            if self._contains_assistant(last_response):
                return last_response
            messages.append(
                {
                    "role": "user",
                    "content": "请务必只返回符合要求的JSON。",
                }
            )
        return last_response

    @staticmethod
    def _contains_assistant(messages: Optional[List[dict]]) -> bool:
        if not messages:
            return False
        return any(msg.get("role") == "assistant" for msg in messages)

    def _extract_payload(self, response: Optional[List[dict]]) -> Optional[dict]:
        if not response:
            return None
        for msg in response:
            if msg.get("role") != "assistant":
                continue
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            candidate = self._extract_json_str(text)
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _extract_json_str(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1]

    def _build_score(self, payload: dict) -> Optional[AgentScore]:
        sentiment_raw = str(payload.get("sentiment", "")).strip()
        mapping = {"积极": "positive", "消极": "negative", "中性": "neutral"}
        sentiment = mapping.get(sentiment_raw, "")
        if not sentiment:
            return None
        reasoning_raw = payload.get("reasoning", [])
        if isinstance(reasoning_raw, str):
            reasoning_list = [reasoning_raw.strip()] if reasoning_raw.strip() else []
        elif isinstance(reasoning_raw, list):
            reasoning_list = [str(item).strip() for item in reasoning_raw if str(item).strip()]
        else:
            reasoning_list = []
        if not reasoning_list:
            reasoning_list = ["理由缺失：未提供证据。"]
        return AgentScore(sentiment=sentiment, reasoning=reasoning_list)


class MockBlockchain:
    def __init__(self) -> None:
        self._txs: List[dict] = []

    def publish(self, agent_name: str, row: NewsRow, score: AgentScore, currency: str) -> dict:
        payload = {
            "agent": agent_name,
            "ticker": row.ticker,
            "date": row.date,
            "price": str(row.close_price),
            "currency": currency,
            "sentiment": score.sentiment,
            "reasoning": score.reasoning,
            "timestamp": int(time.time()),
        }
        payload["tx_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._txs.append(payload)
        return payload

    def aggregated_price(self) -> Optional[Decimal]:
        prices = [Decimal(tx["price"]) for tx in self._txs]
        if not prices:
            return None
        avg = sum(prices) / Decimal(len(prices))
        return avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def dump(self) -> List[dict]:
        return list(self._txs)


def load_rows(path: Path, limit: Optional[int]) -> List[NewsRow]:
    rows: List[NewsRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            ticker = (raw.get("ticker") or "").strip().upper()
            summary = (raw.get("news_summary") or "").strip()
            if not ticker or not summary:
                continue
            close_value = _safe_decimal(raw.get("close"))
            if close_value is None:
                continue
            rows.append(
                NewsRow(
                    ticker=ticker,
                    date=(raw.get("date") or "").strip(),
                    close_price=close_value,
                    summary=summary,
                    titles=(raw.get("news_titles") or "").strip(),
                    bodies=(raw.get("news_bodies") or "").strip(),
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def _safe_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三个 LLM agent 同时为所有摘要打分。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="最多处理的行数（默认全部）。")
    parser.add_argument("--currency", type=str, default="USD")
    parser.add_argument("--show-lengths", action="store_true")
    return parser.parse_args()


def _normalize_for_similarity(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
    return " ".join(tokens)


def aggregate_reasons(rows: List[dict], similarity_threshold: float = 0.6) -> List[dict]:
    clusters: List[dict] = []
    for row in rows:
        reasoning_entries = (row.get("reasoning") or "").split("||")
        for text in reasoning_entries:
            key = text.strip()
            if not key:
                continue
            normalized = _normalize_for_similarity(key)
            if not normalized:
                continue
            best_cluster = None
            best_score = 0.0
            for cluster in clusters:
                score = SequenceMatcher(None, cluster["norm"], normalized).ratio()
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
            if best_cluster is not None and best_score >= similarity_threshold:
                best_cluster["count"] += 1
            else:
                clusters.append({"norm": normalized, "count": 1, "example": key})

    return [{"label": cluster["example"], "count": cluster["count"]} for cluster in clusters]


def per_row_reason_similarity(rows: List[dict]) -> Dict[tuple, float]:
    grouped: Dict[tuple, List[str]] = {}
    for row in rows:
        key = (row.get("ticker"), row.get("date"))
        text = (row.get("reasoning") or "").strip()
        if not text:
            continue
        grouped.setdefault(key, []).append(text)

    scores: Dict[tuple, float] = {}
    for key, reasons in grouped.items():
        if len(reasons) < 2:
            continue
        total = 0.0
        comparisons = 0
        for i in range(len(reasons)):
            for j in range(i + 1, len(reasons)):
                total += SequenceMatcher(None, reasons[i], reasons[j]).ratio()
                comparisons += 1
        if comparisons:
            scores[key] = total / comparisons
    return scores


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} 不存在，请先完成前两步。")

    rows = load_rows(args.input, args.limit)
    if not rows:
        print("无可用摘要。", file=sys.stderr)
        return

    agents = [NewsScoringAgent(name) for name in AGENT_NAMES]
    ledger = MockBlockchain()
    output_rows: List[dict] = []

    for row in rows:
        if args.show_lengths:
            orig_len = len((row.titles + " " + row.bodies).split())
            summary_len = len(row.summary.split())
            print(f"{row.ticker} {row.date}: {orig_len} -> {summary_len}")
        for agent in agents:
            score = agent.score(row)
            if score is None:
                print(f"{agent.name} {row.ticker} {row.date} 未获得有效JSON，跳过。", file=sys.stderr)
                continue
            tx = ledger.publish(agent.name, row, score, args.currency)
            output_rows.append(
                {
                    "agent": agent.name,
                    "ticker": row.ticker,
                    "date": row.date,
                    "close": str(row.close_price),
                    "news_summary": row.summary,
                    "sentiment": score.sentiment,
                    "reasoning": " || ".join(score.reasoning),
                    "tx_hash": tx["tx_hash"],
                }
            )

    aggregate = ledger.aggregated_price()
    if aggregate is not None:
        print(f"链上均价: {aggregate} {args.currency}")
    print("\n共识理由：")
    consensus = aggregate_reasons(output_rows)
    for cluster in consensus:
        print(f"{cluster['label']}: {cluster['count']} 次")
    similarity = per_row_reason_similarity(output_rows)
    if similarity:
        print("\n逐条语义一致性（平均相似度）：")
        for key, score in similarity.items():
            ticker, date = key
            print(f"{ticker} {date}: {score:.2f}")
    print("\n原始交易记录：")
    for tx in ledger.dump():
        print(tx)


if __name__ == "__main__":
    main()
