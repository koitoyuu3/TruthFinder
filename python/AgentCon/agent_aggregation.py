"""Blockchain oracle assistant orchestrated via Qwen Agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Optional, Tuple

from qwen_agent.agents import Assistant
import statistics

import pickle
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

from qwen_agent.tools.base import BaseTool, register_tool

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_ROOT = DATA_DIR if DATA_DIR.exists() else BASE_DIR
COIN_DIRS = {
    'BTC': DATA_ROOT / 'price_BTC_USD',
    'ETH': DATA_ROOT / 'price_ETH_USD',
    'DOGE': DATA_ROOT / 'price_DOGE_USD',
}

@register_tool('coin_price')
class CoinPriceTool(BaseTool):
    """Tool for querying coin prices from local historical datasets."""

    description = '查询稳定币价格'
    parameters = [
        {
            'name': 'currency',
            'type': 'string',
            'description': '想要获取报价的货币（如USD、CNY），为空则默认USD',
            'required': False
        },
        {
            'name': 'source',
            'type': 'string',
            'description': '可选的报价来源或代理名称，用于模拟不同节点的反馈',
            'required': False
        },
        {
            'name': 'timestamp',
            'type': 'string',
            'description': '可选的ISO时间戳（如2025-11-27T00:59），为空则取数据中的最新时间点',
            'required': False
        },
    ]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        data_path = self.cfg.get('data_path')
        if not data_path:
            raise ValueError('CoinPriceTool requires `data_path` in cfg.')
        self.data_path = Path(data_path).expanduser()
        if not self.data_path.exists():
            raise FileNotFoundError(f'{self.data_path} does not exist for CoinPriceTool.')
        self.currency = self.cfg.get('default_currency', 'USD')
        self._series_cache: Optional[List[Tuple[datetime, Decimal]]] = None

    def _parse_decimal(self, value) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if hasattr(value, 'item') and callable(value.item):
            return self._parse_decimal(value.item())
        if isinstance(value, str):
            try:
                return Decimal(value)
            except InvalidOperation:
                return None
        return None

    def _parse_datetime(self, value) -> Optional[datetime]:
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
            return self._parse_datetime(value.item())
        if isinstance(value, Sequence):
            for candidate in value:
                ts = self._parse_datetime(candidate)
                if ts:
                    return ts
        if isinstance(value, dict):
            for key in ('time', 'timestamp', 'ts'):
                if key in value:
                    ts = self._parse_datetime(value[key])
                    if ts:
                        return ts
        return None

    def _load_series(self) -> List[Tuple[datetime, Decimal]]:
        if self._series_cache is not None:
            return self._series_cache
        try:
            with self.data_path.open('rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            raise RuntimeError(f'Failed to load {self.data_path}: {exc}') from exc
        if not isinstance(data, list):
            raise ValueError(f'Unexpected data type in {self.data_path}: {type(data)}')
        series: List[Tuple[datetime, Decimal]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            ts_raw = entry.get('time')
            price_raw = entry.get('price')
            ts = self._parse_datetime(ts_raw)
            price = self._parse_decimal(price_raw)
            if ts and price is not None:
                series.append((ts, price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)))
        if not series:
            raise ValueError(f'No valid (time, price) pairs found in {self.data_path}')
        series.sort(key=lambda x: x[0])
        self._series_cache = series
        return series

    def _select_price(self, target_time: Optional[datetime]) -> Tuple[datetime, Decimal]:
        points = self._load_series()
        if target_time is None:
            return points[-1]
        candidate: Optional[Tuple[datetime, Decimal]] = None
        for ts, price in points:
            if ts > target_time:
                break
            candidate = (ts, price)
        if candidate is None:
            raise ValueError(f'No price available at or before {target_time.isoformat()} in {self.data_path}')
        return candidate

    def _compute_twap(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Tuple[datetime, Decimal]:
        points = self._load_series()
        if not points:
            raise ValueError(f'No data found in {self.data_path}')
        window_start = start_time or points[0][0]
        window_end = end_time or points[-1][0]
        window_start = max(window_start, points[0][0])
        window_end = min(window_end, points[-1][0])
        if window_end <= window_start:
            raise ValueError('Invalid time window for TWAP.')

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
            raise ValueError('TWAP duration is zero.')
        twap = (numerator / total_seconds).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return window_end, twap

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        currency = params.get('currency', self.currency)
        source = params.get('source', 'default')
        timestamp_raw = params.get('timestamp')
        mode = params.get('mode', 'latest')
        start_raw = params.get('start_time')
        end_raw = params.get('end_time')
        target_time = self._parse_datetime(timestamp_raw) if timestamp_raw else None
        start_time = self._parse_datetime(start_raw) if start_raw else None
        end_time = self._parse_datetime(end_raw) if end_raw else target_time

        if mode == 'twap':
            ts, price = self._compute_twap(start_time, end_time)
        else:
            ts, price = self._select_price(end_time or target_time)

        return {
            'source': source,
            'currency': currency,
            'price': str(price),
            'data_source': str(self.data_path),
            'timestamp': ts.isoformat(),
        }


class MockBlockchain:
    """A minimal ledger to store oracle submissions and compute aggregated price."""

    def __init__(self, aggregation: str = 'mean', truncation_ratio: float = 0.2):
        self._txs: List[dict] = []
        self.aggregation = aggregation
        self.truncation_ratio = truncation_ratio

    def publish_price(self, agent_name: str, price: Decimal, currency: str) -> dict:
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

    def _aggregate_median(self, values: List[Decimal]) -> Decimal:
        return Decimal(statistics.median(values)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _aggregate_truncated_mean(self, values: List[Decimal]) -> Decimal:
        sorted_vals = sorted(values)
        cut = int(len(sorted_vals) * self.truncation_ratio)
        trimmed = sorted_vals[cut:len(sorted_vals) - cut] or sorted_vals
        avg = sum(trimmed) / Decimal(len(trimmed))
        return Decimal(avg).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def aggregated_price(self) -> Optional[Decimal]:
        if not self._txs:
            return None
        values = [Decimal(tx['price']) for tx in self._txs]
        if self.aggregation == 'median':
            return self._aggregate_median(values)
        if self.aggregation == 'truncated_mean':
            return self._aggregate_truncated_mean(values)
        avg = sum(values) / Decimal(len(values))
        return Decimal(avg).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def dump(self) -> List[dict]:
        return list(self._txs)


def _parse_source_spec(spec: str) -> Tuple[str, str, int]:
    """Parse strings like `Agent=data/price_BTC...pk@30` into name, path, repeat."""
    if '=' in spec:
        name_hint, _, path_spec = spec.partition('=')
    else:
        name_hint, path_spec = '', spec
    repeat = 1
    if '@' in path_spec:
        path_str, _, repeat_str = path_spec.rpartition('@')
        path_spec = path_str
        try:
            repeat = int(repeat_str)
        except ValueError as exc:
            raise ValueError(f'Invalid repeat count in {spec}') from exc
        if repeat <= 0:
            raise ValueError(f'Repeat count must be positive in {spec}')
    return name_hint.strip(), path_spec.strip(), repeat


def _discover_coin_sources(coin: str) -> List[Tuple[str, Path]]:
    coin = coin.upper()
    if coin not in COIN_DIRS:
        raise ValueError(f'Unsupported coin {coin}, choose from {", ".join(COIN_DIRS)}')
    folder = COIN_DIRS[coin]
    if not folder.exists():
        raise FileNotFoundError(f'Data directory {folder} does not exist.')
    sources: List[Tuple[str, Path]] = []
    for path in sorted(folder.glob('*.pk')):
        sources.append((f'{coin}-{path.stem}', path))
    if not sources:
        raise FileNotFoundError(f'No .pk files found under {folder}')
    return sources


def build_agent_configs(data_paths: List[str]) -> List[dict]:
    configs: List[dict] = []
    for spec in data_paths:
        name_hint, path_str, repeat = _parse_source_spec(spec)
        data_path = Path(path_str).expanduser()
        if not data_path.exists():
            raise FileNotFoundError(f'{data_path} does not exist for agent config.')
        for idx in range(repeat):
            base_name = name_hint or f'Agent-{len(configs) + 1}'
            suffix = f'#{idx + 1}' if repeat > 1 else ''
            name = base_name + suffix if suffix else base_name
            configs.append({'name': name, 'data_path': str(data_path)})
    return configs


def init_agent_service(agent_name: str,
                       data_path: str,
                       currency: str = 'USD',
                       timestamp_hint: Optional[str] = None,
                       mode: str = 'latest',
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> Assistant:
    llm_cfg = {
        'model': 'qwen2.5:1.5b',
        'model_server': 'http://127.0.0.1:11434/v1',
        'api_key': 'ollama',
    }
    model_name = llm_cfg['model']
    timestamp_line = f'- 如果需要，请把JSON中的timestamp设置为"{timestamp_hint}"。\n' if timestamp_hint else ''
    mode_line = f'- JSON中必须包含 "mode": "{mode}" 字段。\n'
    time_line = ''
    if start_time:
        time_line += f'- JSON中必须包含 "start_time": "{start_time}"。\n'
    if end_time:
        time_line += f'- JSON中必须包含 "end_time": "{end_time}"。\n'
    system = (
        f'你是{agent_name}，一名区块链预言机节点。\n'
        f'- 你收到任何请求时都必须调用一次且仅调用一次工具`coin_price`。\n'
        f'- 调用时必须传入JSON，至少包含 {{"source": "{agent_name}", "currency": "{currency}"}}。\n'
        f'- 你的数据源文件为 {data_path}，不要自行伪造价格。\n'
        f'{timestamp_line}'
        f'{mode_line}'
        f'{time_line}'
        '- 不允许调用不存在的工具名称，也不允许猜测或转换单位。\n'
        '- 工具返回的字段中包含 price/currency/data_source。请用固定格式输出：\n'
        f'  我是{agent_name}（模型{model_name}），价格为<price><currency>/oz，数据来源：<data_source>。\n'
        '- 若工具报错，就直接把错误内容写出来并说明来自工具。\n'
        '严格遵循以上要求，除该格式外不要添加额外解释。'
    )
    tool_instance = CoinPriceTool(cfg={
        'data_path': data_path,
        'default_currency': currency,
    })
    bot = Assistant(
        llm=llm_cfg,
        name=agent_name,
        description=f'{agent_name} 资产报价节点',
        system_message=system,
        function_list=[tool_instance],
    )
    bot._coin_price_tool = tool_instance
    bot._coin_currency = currency
    bot._model_name = model_name
    bot._timestamp_hint = timestamp_hint
    bot._mode = mode
    bot._start_time = start_time
    bot._end_time = end_time
    return bot


def _contains_function_call(messages: Optional[List[dict]]) -> bool:
    return any(isinstance(msg, dict) and 'function_call' in msg for msg in (messages or []))


def _extract_price_payload(response: Optional[List[dict]]) -> Optional[dict]:
    if not response:
        return None
    for msg in response:
        if msg.get('role') == 'function' and msg.get('name') == 'coin_price':
            try:
                payload = json.loads(msg.get('content', '{}'))
            except json.JSONDecodeError:
                continue
            if 'price' in payload and 'currency' in payload:
                return payload
    return None


def _direct_tool_payload(agent: Assistant,
                         mode: str,
                         currency: str,
                         start_time: Optional[str],
                         end_time: Optional[str]) -> Optional[dict]:
    tool = getattr(agent, '_coin_price_tool', None)
    if tool is None:
        return None
    args = {
        'source': agent.name,
        'currency': currency,
        'mode': mode,
    }
    if start_time:
        args['start_time'] = start_time
    if end_time:
        args['end_time'] = end_time
    timestamp_hint = getattr(agent, '_timestamp_hint', None)
    if timestamp_hint:
        args['timestamp'] = timestamp_hint
    result = tool.call(args)
    return result


def run_agent(agent: Assistant, query: str, max_retries: int = 2) -> Optional[List[dict]]:
    messages = [{'role': 'user', 'content': query}]
    last_response: Optional[List[dict]] = None
    for _ in range(max_retries):
        last_response = None
        for response in agent.run(messages):
            last_response = response
        if _contains_function_call(last_response):
            return last_response
        messages.append({'role': 'user', 'content': '你的上一轮没有调用gold_price工具，请严格按要求调用。'})
    return None


def test(query: str,
         agent_configs: List[dict],
         currency: str = 'USD',
         timestamp_hint: Optional[str] = None,
         aggregation: str = 'mean',
         truncation_ratio: float = 0.2,
         mode: str = 'latest',
         start_time: Optional[str] = None,
         end_time: Optional[str] = None):
    ledger = MockBlockchain(aggregation=aggregation, truncation_ratio=truncation_ratio)
    if mode == 'twap':
        window_desc = f'{start_time or "数据起点"} 至 {end_time or "数据终点"}'
        query = f'{query}（请按TWAP模式处理时间窗口：{window_desc}）'
    for cfg in agent_configs:
        agent = init_agent_service(
            cfg['name'],
            cfg['data_path'],
            currency=currency,
            timestamp_hint=timestamp_hint,
            mode=mode,
            start_time=start_time,
            end_time=end_time,
        )
        response = run_agent(agent, query)
        print(f"\n{cfg['name']} response:")
        if response:
            print(response)
        else:
            print('No response.')

        payload = _extract_price_payload(response)
        direct_payload = _direct_tool_payload(agent, mode, currency, start_time, end_time)
        final_payload = direct_payload or payload
        if final_payload:
            tx = ledger.publish_price(cfg['name'], Decimal(final_payload['price']), final_payload['currency'])
            print('On-chain tx:' if direct_payload is None else 'On-chain tx (direct tool override):', tx)
        else:
            print('No price payload found; nothing published on-chain.')

    agg_price = ledger.aggregated_price()
    if agg_price is not None:
        label = {'median': 'median', 'truncated_mean': 'truncated mean', 'mean': 'arithmetic mean'}.get(aggregation, aggregation)
        print(f'\nOn-chain {label}: {agg_price} {currency}/oz')
    else:
        print('\nOn-chain aggregation result unavailable.')

    print('\nFinal on-chain ledger snapshot:')
    for tx in ledger.dump():
        print(tx)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run multi-agent coin price oracle.')
    parser.add_argument('--data.path', dest='data_paths', nargs='+', required=True, help='One or more pickle files for agents. Supports name=path@N syntax for repetition.')
    parser.add_argument('--data.currency', dest='currency', type=str, default='USD', help='Currency label.')
    parser.add_argument('--data.start_time', dest='start_time', type=str, help='ISO start time (for reference/logging).')
    parser.add_argument('--data.end_time', dest='end_time', type=str, help='ISO time the agents should sample (defaults to latest if omitted).')
    parser.add_argument('--aggregation', choices=['mean', 'median', 'truncated_mean'], default='mean', help='Aggregation method for on-chain result.')
    parser.add_argument('--truncation-ratio', dest='truncation_ratio', type=float, default=0.1, help='Trim ratio for truncated mean.')
    parser.add_argument('--mode', choices=['latest', 'twap'], default='latest', help='Price mode: latest sample or TWAP over [start, end].')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    configs = build_agent_configs(args.data_paths)
    timestamp_hint = args.end_time or args.start_time
    test(
        '请查询币价并返回报价',
        configs,
        currency=args.currency,
        timestamp_hint=timestamp_hint,
        aggregation=args.aggregation,
        truncation_ratio=args.truncation_ratio,
        mode=args.mode,
        start_time=args.start_time,
        end_time=args.end_time,
    )
