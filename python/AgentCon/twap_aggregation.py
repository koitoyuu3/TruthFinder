"""Compute TWAP (time-weighted average price) for local oracle feeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_ROOT = DATA_DIR if DATA_DIR.exists() else BASE_DIR
COIN_DIRS = {
    'BTC': DATA_ROOT / 'price_BTC_USD',
    'ETH': DATA_ROOT / 'price_ETH_USD',
    'DOGE': DATA_ROOT / 'price_DOGE_USD',
}
PRICE_FIELDS = ('price', 'last', 'close', 'value')
TIME_FIELDS = ('time', 'timestamp', 'ts')


class MockBlockchain:
    """Simple ledger storing oracle submissions and computing on-chain averages."""

    def __init__(self):
        self._txs: List[dict] = []

    def publish_price(self, agent_name: str, price: Decimal, currency: str = 'USD') -> dict:
        payload = {
            'agent': agent_name,
            'price': str(price),
            'currency': currency,
            'timestamp': int(time.time()),
        }
        serialized = json.dumps(payload, sort_keys=True).encode('utf-8')
        payload['tx_hash'] = hashlib.sha256(serialized).hexdigest()
        self._txs.append(payload)
        return payload

    def average_price(self) -> Optional[Decimal]:
        if not self._txs:
            return None
        values = [Decimal(tx['price']) for tx in self._txs]
        avg = sum(values) / Decimal(len(values))
        return Decimal(avg).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def dump(self) -> List[dict]:
        return list(self._txs)


def _parse_decimal(value) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if hasattr(value, 'item') and callable(value.item):
        return _parse_decimal(value.item())
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    return None


def _parse_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            cleaned = value.replace('Z', '+00:00') if value.endswith('Z') else value
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, ValueError):
            return None
    if hasattr(value, 'item') and callable(value.item):
        return _parse_datetime(value.item())
    if isinstance(value, Sequence) and value:
        for candidate in value:
            ts = _parse_datetime(candidate)
            if ts:
                return ts
    return None


def load_price_series(path: Path) -> List[Tuple[datetime, Decimal]]:
    try:
        with path.open('rb') as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f'Failed to load {path}: {exc}')
        return []
    if not isinstance(data, list):
        print(f'Unexpected data type in {path}: {type(data)}')
        return []

    series: List[Tuple[datetime, Decimal]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ts_raw = next((entry.get(key) for key in TIME_FIELDS if key in entry), None)
        price_raw = next((entry.get(key) for key in PRICE_FIELDS if key in entry), None)
        ts = _parse_datetime(ts_raw)
        price = _parse_decimal(price_raw)
        if ts and price is not None:
            series.append((ts, price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)))
    return series


def compute_twap(series: List[Tuple[datetime, Decimal]],
                 start_time: Optional[datetime],
                 end_time: Optional[datetime]) -> Optional[Decimal]:
    if not series:
        return None
    points = sorted(series, key=lambda x: x[0])
    window_start = start_time or points[0][0]
    window_end = end_time or points[-1][0]
    window_start = max(window_start, points[0][0])
    window_end = min(window_end, points[-1][0])
    if window_end <= window_start:
        return None

    idx = 0
    while idx < len(points) and points[idx][0] <= window_start:
        idx += 1

    current_price = points[0][1]
    for ts, price in points:
        if ts <= window_start:
            current_price = price
        else:
            break
    current_time = window_start
    numerator = Decimal('0')
    total_seconds = Decimal('0')

    while current_time < window_end:
        next_ts = window_end
        if idx < len(points):
            next_ts = min(points[idx][0], window_end)
        duration = (next_ts - current_time).total_seconds()
        if duration > 0:
            numerator += current_price * Decimal(duration)
            total_seconds += Decimal(duration)
        if idx >= len(points) or points[idx][0] >= window_end:
            break
        current_price = points[idx][1]
        current_time = points[idx][0]
        idx += 1

    if current_time < window_end:
        duration = (window_end - current_time).total_seconds()
        if duration > 0:
            numerator += current_price * Decimal(duration)
            total_seconds += Decimal(duration)

    if total_seconds == 0:
        return None
    return (numerator / total_seconds).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


@dataclass
class OracleSource:
    """Represents a historical price feed sourced from disk."""

    name: str
    file_path: Path

    def fetch_twap(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Optional[Decimal]:
        series = load_price_series(self.file_path)
        twap = compute_twap(series, start_time, end_time)
        if twap is not None:
            print(f'{self.name} TWAP: {twap} USD/oz')
        else:
            print(f'{self.name} has no TWAP in the requested range.')
        return twap


class OracleAggregator:
    """Collects TWAPs from local historical sources and aggregates them."""

    def __init__(self,
                 sources: Iterable[OracleSource],
                 blockchain: MockBlockchain,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None):
        self.sources = list(sources)
        self.blockchain = blockchain
        self.start_time = start_time
        self.end_time = end_time

    def collect_quotes(self) -> List[dict]:
        quotes = []
        for source in self.sources:
            price = source.fetch_twap(self.start_time, self.end_time)
            if price is not None:
                quotes.append({'agent': source.name, 'price': price})
        return quotes

    @staticmethod
    def remove_outliers(quotes: List[dict], threshold_pct: Decimal = Decimal('0.8')) -> List[dict]:
        if not quotes:
            return []
        prices = [entry['price'] for entry in quotes]
        median_price = statistics.median(prices)
        filtered: List[dict] = []
        for entry in quotes:
            deviation = abs(entry['price'] - Decimal(median_price)) / Decimal(median_price) * 100
            if deviation <= threshold_pct:
                filtered.append(entry)
        return filtered

    def aggregate_price(self) -> Optional[Decimal]:
        raw_quotes = self.collect_quotes()
        filtered_quotes = self.remove_outliers(raw_quotes)
        if not filtered_quotes:
            print('No valid quotes available.')
            return None
        for entry in filtered_quotes:
            tx = self.blockchain.publish_price(entry['agent'], entry['price'])
            print(f"Published on-chain: {tx['tx_hash']} -> {tx['price']} USD/oz")
        final_price = self.blockchain.average_price()
        return final_price


def _expand_path_argument(arg: str) -> Tuple[Path, int]:
    path_str, repeat = arg, 1
    if '@' in arg:
        path_str, _, repeat_str = arg.rpartition('@')
        try:
            repeat = int(repeat_str)
        except ValueError:
            raise ValueError(f'Invalid repeat count in {arg}')
        if repeat <= 0:
            raise ValueError(f'Repeat count must be positive in {arg}')
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist.')
    return path, repeat


def discover_sources(coin: Optional[str], paths: Optional[List[str]]) -> List[OracleSource]:
    sources: List[OracleSource] = []
    if paths:
        name_counts = {}
        for raw in paths:
            path, repeat = _expand_path_argument(raw)
            base = path.stem or 'source'
            count_so_far = name_counts.get(base, 0)
            for i in range(repeat):
                if count_so_far == 0 and repeat == 1 and base not in name_counts:
                    name = base
                else:
                    name = f'{base}#{count_so_far + i + 1}'
                sources.append(OracleSource(name=name, file_path=path))
            name_counts[base] = count_so_far + repeat
    elif coin:
        coin = coin.upper()
        if coin not in COIN_DIRS:
            raise ValueError(f'Unsupported coin {coin}, choose from {", ".join(COIN_DIRS)}')
        data_dir = COIN_DIRS[coin]
        if not data_dir.exists():
            raise FileNotFoundError(f'Data directory {data_dir} does not exist.')
        for path in sorted(data_dir.glob('*.pk')):
            sources.append(OracleSource(name=path.stem, file_path=path))
        if not sources:
            raise FileNotFoundError(f'No .pk files found under {data_dir}')
    else:
        raise ValueError('Either coin or explicit data paths must be provided.')
    return sources


def simulate_publish_cycle(coin: Optional[str],
                           data_paths: Optional[List[str]],
                           start_time: Optional[str],
                           end_time: Optional[str],
                           currency: str = 'USD'):
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else None
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None
    sources = discover_sources(coin, data_paths)
    blockchain = MockBlockchain()
    aggregator = OracleAggregator(sources, blockchain, start_dt, end_dt)
    final_price = aggregator.aggregate_price()
    if final_price is None:
        return
    print(f'On-chain TWAP: {final_price} {currency}/oz')
    print('Ledger snapshot:')
    for tx in blockchain.dump():
        print(tx)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compute TWAP from local historical price data.')
    parser.add_argument('--coin', type=str, default=None, help='Coin symbol (BTC/ETH/DOGE). Ignored if --data.path is provided.')
    parser.add_argument('--data.path', dest='data_paths', nargs='+', help='Explicit pickle files to aggregate. Supports path@N repetition syntax.')
    parser.add_argument('--data.start_time', dest='start_time', type=str, help='ISO timestamp filter start, e.g. 2025-11-27T00:00')
    parser.add_argument('--data.end_time', dest='end_time', type=str, help='ISO timestamp filter end, e.g. 2025-11-27T00:59')
    parser.add_argument('--data.currency', dest='currency', type=str, default='USD', help='Display currency label.')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    simulate_publish_cycle(args.coin, args.data_paths, args.start_time, args.end_time, args.currency)
