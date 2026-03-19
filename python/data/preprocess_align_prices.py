import pickle
from pathlib import Path

# PRICE_DIR = Path(__file__).resolve().parent / 'price_ETH_USD'
# REF_FILE = 'cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk'
# TARGET_FILES = [
#     'coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
#     'kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
# ]

# PRICE_DIR = Path(__file__).resolve().parent / 'price_BTC_USD'
# REF_FILE = 'cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk'
# TARGET_FILES = [
#     'coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
#     'kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
# ]

PRICE_DIR = Path(__file__).resolve().parent / 'price_DOGE_USD'
REF_FILE = 'cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk'
TARGET_FILES = [
    'coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
    'kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
]

def load_pk(path: Path):
    with path.open('rb') as fh:
        data = pickle.load(fh)
    data.sort(key=lambda x: x['time'])
    return data


def align_to_reference(ref_data, target_data):
    if not target_data:
        raise ValueError('目标文件数据为空，无法扩充')

    target_data.sort(key=lambda x: x['time'])
    ref_data.sort(key=lambda x: x['time'])

    idx = 0
    last_price = None
    ref_start = ref_data[0]['time']

    # 如果目标数据在参考开始前已有记录，先用它们初始化 last_price
    while idx < len(target_data) and target_data[idx]['time'] < ref_start:
        last_price = target_data[idx]['price']
        idx += 1

    if last_price is None:
        last_price = target_data[0]['price']

    aligned = []
    for ref_entry in ref_data:
        ref_time = ref_entry['time']
        while idx < len(target_data) and target_data[idx]['time'] <= ref_time:
            last_price = target_data[idx]['price']
            idx += 1
        aligned.append({'time': ref_time, 'price': last_price})
    return aligned


def main():
    ref_path = PRICE_DIR / REF_FILE
    if not ref_path.exists():
        raise FileNotFoundError(f'参考文件不存在: {ref_path}')
    ref_data = load_pk(ref_path)

    for target_name in TARGET_FILES:
        target_path = PRICE_DIR / target_name
        if not target_path.exists():
            print(f'跳过，找不到文件: {target_path}')
            continue
        target_data = load_pk(target_path)
        aligned_data = align_to_reference(ref_data, target_data)
        with target_path.open('wb') as fh:
            pickle.dump(aligned_data, fh)
        print(f'已扩充: {target_name}, 共 {len(aligned_data)} 条')


if __name__ == '__main__':
    main()
