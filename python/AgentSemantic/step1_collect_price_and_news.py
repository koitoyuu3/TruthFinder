#!/usr/bin/env python3
"""
Step 1 of the mini pipeline:
Find the first three tickers that appear in nasdaq_external_data.csv and also
exist inside full_history, then collect at least MIN_TRADING_DAYS_WITH_NEWS trading
days that contain news for each ticker. Every trading day that has one or more
news articles is paired with its price row and written to a CSV for downstream
processing.
"""

from __future__ import annotations

import csv
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = REPO_ROOT / "full_history"
NEWS_FILE = REPO_ROOT / "nasdaq_external_data.csv"
OUTPUT_FILE = Path(__file__).resolve().with_name("step1_recent_price_and_news.csv")

NUM_TICKERS = 3
MIN_TRADING_DAYS_WITH_NEWS = 50

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class PriceRow:
    ticker: str
    trade_date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    adj_close: float
    volume: int


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    published: datetime
    title: str
    url: str
    body: str


def parse_news_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.endswith("UTC"):
        raw = raw[:-3].strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def clean_text(value: str) -> str:
    if not value:
        return ""
    # Collapse whitespace so the CSV stays compact.
    return " ".join(value.split())


def load_price_data(ticker: str) -> tuple[List[date], Dict[date, PriceRow]]:
    target_path = PRICE_DIR / f"{ticker}.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} is missing.")

    rows: List[PriceRow] = []
    with target_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            normalized = {
                (key or "").strip().lower().replace(" ", "_"): (value or "").strip()
                for key, value in raw_row.items()
            }
            if not normalized.get("date"):
                continue
            trade_date = datetime.strptime(normalized["date"], "%Y-%m-%d").date()

            def as_float(key: str) -> float:
                try:
                    return float(normalized[key])
                except (KeyError, ValueError):
                    return 0.0

            def as_int(key: str) -> int:
                try:
                    return int(float(normalized[key]))
                except (KeyError, ValueError):
                    return 0

            rows.append(
                PriceRow(
                    ticker=ticker,
                    trade_date=trade_date,
                    open_price=as_float("open"),
                    high_price=as_float("high"),
                    low_price=as_float("low"),
                    close_price=as_float("close"),
                    adj_close=as_float("adj_close"),
                    volume=as_int("volume"),
                )
            )

    rows.sort(key=lambda item: item.trade_date)
    if not rows:
        raise RuntimeError(f"{target_path} did not contain any price rows.")
    trade_dates = [row.trade_date for row in rows]
    lookup = {row.trade_date: row for row in rows}
    return trade_dates, lookup


def map_to_trade_date(trade_dates: Sequence[date], news_day: date) -> date | None:
    if not trade_dates:
        return None
    idx = bisect_right(trade_dates, news_day) - 1
    if idx >= 0:
        return trade_dates[idx]
    return None


def collect_news_with_prices(
    ticker_limit: int, min_days: int
) -> tuple[List[str], Dict[str, Dict[date, List[NewsItem]]], Dict[str, Dict[date, PriceRow]]]:
    price_dates: Dict[str, Sequence[date]] = {}
    price_lookup: Dict[str, Dict[date, PriceRow]] = {}
    news_by_date: Dict[str, Dict[date, List[NewsItem]]] = {}
    completed: List[str] = []

    with NEWS_FILE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if len(completed) >= ticker_limit:
                break

            symbol = (row.get("Stock_symbol") or "").strip().upper()
            if not symbol:
                continue

            if symbol not in price_dates:
                price_file = PRICE_DIR / f"{symbol}.csv"
                if not price_file.exists():
                    continue
                dates, lookup = load_price_data(symbol)
                price_dates[symbol] = dates
                price_lookup[symbol] = lookup
                news_by_date[symbol] = defaultdict(list)

            timestamp = parse_news_timestamp(row.get("Date", ""))
            if timestamp is None:
                continue

            trade_date = map_to_trade_date(price_dates[symbol], timestamp.date())
            if trade_date is None:
                continue

            news_by_date[symbol][trade_date].append(
                NewsItem(
                    ticker=symbol,
                    published=timestamp,
                    title=(row.get("Article_title") or "").strip(),
                    url=(row.get("Url") or "").strip(),
                    body=clean_text(row.get("Article") or ""),
                )
            )

            if symbol not in completed and len(news_by_date[symbol]) >= min_days:
                completed.append(symbol)

    if len(completed) < ticker_limit:
        raise RuntimeError(
            f"Only found {len(completed)} tickers with >= {min_days} news-bearing trading days. "
            "Consider lowering MIN_TRADING_DAYS_WITH_NEWS or checking the dataset."
        )

    ordered_news = {ticker: news_by_date[ticker] for ticker in completed}
    ordered_prices = {ticker: price_lookup[ticker] for ticker in completed}
    return completed, ordered_news, ordered_prices


def limit_rows_per_ticker(
    tickers: Sequence[str], news_by_date: Dict[str, Dict[date, List[NewsItem]]]
) -> Dict[str, Dict[date, List[NewsItem]]]:
    """
    Use the final ticker in `tickers` (e.g. AADR) as the reference count and keep
    only that many latest trading days for every ticker so each symbol contributes
    an equal number of rows.
    """

    reference_ticker = tickers[-1]
    reference_count = len(news_by_date[reference_ticker])
    if reference_count == 0:
        raise RuntimeError(f"{reference_ticker} does not have any news rows to use as reference.")

    limited: Dict[str, Dict[date, List[NewsItem]]] = {}
    for ticker in tickers:
        per_day = news_by_date[ticker]
        dates = sorted(per_day.keys(), reverse=True)[:reference_count]
        limited[ticker] = {day: per_day[day] for day in dates}
    return limited


def format_join(values: Iterable[str]) -> str:
    cleaned = [value for value in (v.strip() for v in values) if value]
    return " || ".join(cleaned)


def build_rows() -> List[Dict[str, str]]:
    tickers, news_by_date, price_lookup = collect_news_with_prices(
        NUM_TICKERS, MIN_TRADING_DAYS_WITH_NEWS
    )
    news_by_date = limit_rows_per_ticker(tickers, news_by_date)
    output_rows: List[Dict[str, str]] = []

    for ticker in tickers:
        for trade_date in sorted(news_by_date[ticker].keys(), reverse=True):
            price = price_lookup[ticker].get(trade_date)
            if price is None:
                continue
            articles = news_by_date[ticker][trade_date]
            output_rows.append(
                {
                    "ticker": ticker,
                    "date": trade_date.isoformat(),
                    "open": f"{price.open_price:.6f}",
                    "high": f"{price.high_price:.6f}",
                    "low": f"{price.low_price:.6f}",
                    "close": f"{price.close_price:.6f}",
                    "adj_close": f"{price.adj_close:.6f}",
                    "volume": str(price.volume),
                    "news_count": str(len(articles)),
                    "news_datetimes": format_join(
                        art.published.isoformat() for art in articles
                    ),
                    "news_titles": format_join(art.title for art in articles),
                    "news_urls": format_join(art.url for art in articles),
                    "news_bodies": format_join(art.body for art in articles),
                }
            )
    return output_rows


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    fieldnames = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "news_count",
        "news_datetimes",
        "news_titles",
        "news_urls",
        "news_bodies",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
