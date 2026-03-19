import os
import argparse
import pickle
import glob
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def expand_data_paths(paths: List[str]) -> List[str]:
    """支持 path@N 写法，快速复制同一个数据源。"""
    expanded = []
    for spec in paths:
        if '@' in spec:
            base, repeat = spec.rsplit('@', 1)
            try:
                count = int(repeat)
            except ValueError:
                expanded.append(spec)
                continue
            expanded.extend([base] * count)
        else:
            expanded.append(spec)
    return expanded


def resolve_price_file(path: str) -> str:
    """定位数据文件，路径相对于上一级目录。"""
    abs_path = os.path.join(REPO_ROOT, path)
    if os.path.isfile(abs_path):
        return abs_path
    if os.path.isdir(abs_path):
        pk_files = sorted(glob.glob(os.path.join(abs_path, '*.pk')))
        if not pk_files:
            raise FileNotFoundError(f'目录 {path} 下没有找到 .pk 文件')
        return pk_files[0]
    raise FileNotFoundError(f'无法找到数据文件或目录: {path}')


def load_price_series(
    file_path: str,
    max_rows: int = None,
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
) -> List[Dict[str, Any]]:
    """读取 pickled 价格序列，可按时间窗口与行数做裁剪。"""
    with open(file_path, 'rb') as f:
        raw = pickle.load(f)

    series = []
    for idx, row in enumerate(raw):
        if max_rows is not None and idx >= max_rows:
            break
        time_val = row['time'].astype('datetime64[s]')
        if time_start is not None and time_val < time_start:
            continue
        if time_end is not None and time_val > time_end:
            continue
        price_val = float(row['price']) if not isinstance(row['price'], float) else row['price']
        series.append({'time': time_val, 'price': price_val})
    return series


def summarize_series(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not series:
        return {'count': 0}
    prices = np.array([row['price'] for row in series], dtype=float)
    times = [row['time'] for row in series]
    return {
        'count': len(series),
        'time_start': str(times[0]),
        'time_end': str(times[-1]),
        'price_min': float(np.min(prices)),
        'price_max': float(np.max(prices)),
        'price_mean': float(np.mean(prices)),
        'price_median': float(np.median(prices)),
    }


def preview_series(series: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return series[:max(0, limit)]


@dataclass
class DataNode:
    node_id: str
    raw_path: str
    resolved_file: str
    series: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        return summarize_series(self.series)

    def preview(self, limit: int) -> List[Dict[str, Any]]:
        return preview_series(self.series, limit)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple data loader for Async ACon prototype')
    parser.add_argument(
        '--data-path',
        type=str,
        nargs='+',
        default=[
            'data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@2',
            'data/price_BTC_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
        ],
        help='数据文件或目录，可用 path@N 快速复制同一数据源',
    )
    parser.add_argument('--max-rows', type=int, default=None, help='读取每个数据源的最大行数')
    parser.add_argument('--preview-rows', type=int, default=3, help='打印前几行用于快速检查')
    parser.add_argument('--start-time', type=str, default=None, help='起始时间（ISO8601），只保留大于等于该时间的数据')
    parser.add_argument('--end-time', type=str, default=None, help='结束时间（ISO8601），只保留小于等于该时间的数据')
    return parser.parse_args()


def load_nodes(args) -> List[DataNode]:
    paths = expand_data_paths(args.data_path)
    time_start = np.datetime64(args.start_time) if args.start_time else None
    time_end = np.datetime64(args.end_time) if args.end_time else None
    nodes: List[DataNode] = []
    for i, raw_path in enumerate(paths):
        resolved = resolve_price_file(raw_path)
        series = load_price_series(
            resolved,
            max_rows=args.max_rows,
            time_start=time_start,
            time_end=time_end,
        )
        node_id = f'node_{i:02d}'
        nodes.append(DataNode(node_id=node_id, raw_path=raw_path, resolved_file=resolved, series=series))
    return nodes


def print_report(nodes: List[DataNode], preview_rows: int):
    print(f'[data] 注册节点数量: {len(nodes)}')
    for node in nodes:
        stats = node.summary()
        print(f'  - {node.node_id}')
        print(f'    path: {node.raw_path}')
        print(f'    resolved: {node.resolved_file}')
        if stats['count'] == 0:
            print('    no data loaded')
            continue
        print(
            f"    rows={stats['count']}, "
            f"time=[{stats['time_start']} ~ {stats['time_end']}], "
            f"price_min={stats['price_min']:.4f}, "
            f"price_max={stats['price_max']:.4f}, "
            f"price_mean={stats['price_mean']:.4f}"
        )
        preview = node.preview(preview_rows)
        for idx, row in enumerate(preview):
            print(f"      preview[{idx}]: time={row['time']}, price={row['price']:.4f}")
    print()


def main():
    args = parse_args()
    nodes = load_nodes(args)
    print_report(nodes, args.preview_rows)


if __name__ == '__main__':
    main()
