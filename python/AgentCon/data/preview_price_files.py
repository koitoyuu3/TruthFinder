import os
import sys
import pickle
from pathlib import Path
from typing import Optional


def preview_entries(folder: Path, head_limit: int = 100, tail_limit: int = 20):
    pk_files = sorted(folder.glob('*.pk'))
    if not pk_files:
        print(f'目录 {folder} 中没有 .pk 文件')
        return

    for pk_path in pk_files:
        print(f'\n=== {pk_path.name} ===')
        try:
            with pk_path.open('rb') as fh:
                data = pickle.load(fh)
        except Exception as exc:
            print(f'读取失败: {exc}')
            continue

        if not isinstance(data, list):
            print(f'文件内容不是列表，实际类型: {type(data)}')
            continue

        head = data[:head_limit]
        if not head:
            print('文件为空')
            continue

        for i, entry in enumerate(head, 1):
            time_val = entry.get('time')
            price_val = entry.get('price')
            print(f'{i:02d}. time={time_val}, price={price_val}')

        if tail_limit > 0:
            print('-- 尾部 --')
            tail = data[-tail_limit:]
            total = len(data)
            for idx, entry in enumerate(tail, total - len(tail) + 1):
                time_val = entry.get('time')
                price_val = entry.get('price')
                print(f'{idx:02d}. time={time_val}, price={price_val}')


def preview_full_file(pk_path: Path):
    print(f'\n=== 全量输出 {pk_path} ===')
    try:
        with pk_path.open('rb') as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f'读取失败: {exc}')
        return
    if not isinstance(data, list):
        print(f'文件内容不是列表，实际类型: {type(data)}')
        return
    for idx, entry in enumerate(data, 1):
        time_val = entry.get('time') if isinstance(entry, dict) else entry
        price_val = entry.get('price') if isinstance(entry, dict) else ''
        print(f'{idx:05d}. time={time_val}, price={price_val}')


if __name__ == '__main__':
    # folder = Path(__file__).resolve().parent / 'price_ETH_USD'
    folder = Path(__file__).resolve().parent / 'price_BTC_USD'
    # folder = Path(__file__).resolve().parent / 'price_DOGE_USD'
    if len(sys.argv) > 2 and sys.argv[1] == '--all':
        pk_file = folder / sys.argv[2]
        if not pk_file.exists():
            print(f'文件不存在: {pk_file}')
        else:
            preview_full_file(pk_file)
    else:
        preview_entries(folder)
