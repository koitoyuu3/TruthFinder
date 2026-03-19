import os, sys
import base64
import hashlib
import hmac
import json
import pandas as pd
#sfrom datetime import datetime, timedelta
import pickle
import time
import numpy as np
from urllib.parse import urlencode

COINGECKO_SYMBOL_MAP = None
COINGECKO_OVERRIDE = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'USDT': 'tether',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'DOGE': 'dogecoin',
}

KUCOIN_INTERVALS = {
    60: '1min',
    180: '3min',
    300: '5min',
    900: '15min',
    1800: '30min',
    3600: '1hour',
    7200: '2hour',
    14400: '4hour',
    21600: '6hour',
    43200: '12hour',
    86400: '1day',
    604800: '1week',
}
def sanity_check(data, time_start, time_end, time_step_sec):
    if time_start is None or time_end is None:
        return data
    t_start = data[0]['time']
    # assert(data[0]['time'] == time_start)
    # for i, d in enumerate(data):
    #     assert d['time'] == t_start + np.timedelta64(time_step_sec, 's')*i, f'time_obs ({d["time"]}) != time_exp ({t_start + np.timedelta64(time_step_sec, "s")*i})'
    # assert(data[-1]['time'] == time_end)
    for i in range(len(data)-1):
        assert data[i]['time'] != data[i+1]['time'], f"should hold {data[i]['time']} != {data[i+1]['time']}"
    
    return data


def request_batch(name, time_start, time_end, time_step_sec, request_func, postprocess_func):
    
    print(f'[{name}] start time =', time_start, ', end time =', time_end)
    data_pk = []

    t_end = time_start
    f_end = False
    while not f_end:
        t_start = t_end
        t_end = t_start

        for _ in range(99): # do not request too much at once
            if t_end + np.timedelta64(time_step_sec, 's') > time_end:
                f_end = True
                break
            t_end += np.timedelta64(time_step_sec, 's')

        print(f'[{name}, request] start time = {t_start}, end time = {t_end}')
        data_req_i = request_func(t_start, t_end, time_step_sec)
        data_pk_i = postprocess_func(data_req_i)
        data_pk += data_pk_i
        t_end += np.timedelta64(time_step_sec, 's')

        time.sleep(0.05)

    return sanity_check(data_pk, time_start, time_end, time_step_sec)
def _load_coingecko_symbol_map():
    global COINGECKO_SYMBOL_MAP
    if COINGECKO_SYMBOL_MAP is not None:
        return COINGECKO_SYMBOL_MAP
    import requests
    resp = requests.get('https://api.coingecko.com/api/v3/coins/list?include_platform=false', timeout=10)
    if resp.status_code != 200:
        raise ValueError(f'获取Coingecko代币列表失败: {resp.status_code} {resp.text}')
    data = resp.json()
    COINGECKO_SYMBOL_MAP = {}
    for item in data:
        symbol = item.get('symbol')
        coin_id = item.get('id')
        if symbol and coin_id and symbol.upper() not in COINGECKO_SYMBOL_MAP:
            COINGECKO_SYMBOL_MAP[symbol.upper()] = coin_id
    for sym, coin_id in COINGECKO_OVERRIDE.items():
        COINGECKO_SYMBOL_MAP[sym] = coin_id

    if not COINGECKO_SYMBOL_MAP:
        raise ValueError('Coingecko代币列表为空')
    return COINGECKO_SYMBOL_MAP

def get_from_coingecko(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('Coingecko 抓取需要提供 time_start, time_end, time_step_sec')

    now_utc = np.datetime64(pd.Timestamp.utcnow(), 's')
    earliest_allowed = now_utc - np.timedelta64(365, 'D')
    if time_end < earliest_allowed:
        raise ValueError('Coingecko公共API仅支持最近365天的数据，请缩短时间范围')
    if time_start < earliest_allowed:
        print(f'[coingecko] 请求起始时间早于允许范围，自动调整为 {earliest_allowed}')
        time_start = earliest_allowed

    symbol_map = _load_coingecko_symbol_map()
    coin_id = symbol_map.get(pair0.upper())
    if coin_id is None:
        raise ValueError(f'在Coingecko代币列表中找不到 {pair0}')
    vs_currency = pair1.lower()

    def to_unix(ts):
        return int(pd.Timestamp(ts).timestamp())

    chunk_days = 90
    chunk_delta = np.timedelta64(chunk_days, 'D')
    data_pk = []
    cur_start = time_start

    while cur_start < time_end:
        cur_end = min(time_end, cur_start + chunk_delta)
        params = {
            'vs_currency': vs_currency,
            'from': to_unix(cur_start),
            'to': to_unix(cur_end)
        }
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range'
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f'Coingecko接口返回错误: {resp.status_code} {resp.text}')
        json_data = resp.json()
        prices = json_data.get('prices', [])
        if not prices:
            raise ValueError(f'Coingecko未返回价格数据: {json_data}')

        df = pd.DataFrame(prices, columns=['time', 'price'])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        freq = f'{int(time_step_sec)}S'
        df_resampled = df.resample(freq).ffill().dropna()
        ts_start = pd.Timestamp(cur_start, tz='UTC')
        ts_end = pd.Timestamp(cur_end, tz='UTC')
        df_resampled = df_resampled[(df_resampled.index >= ts_start) &
                                    (df_resampled.index <= ts_end)]

        for t, price in df_resampled['price'].items():
            if t.to_numpy() < np.datetime64(time_start) or t.to_numpy() > np.datetime64(time_end):
                continue
            data_pk.append({'time': np.datetime64(t.to_datetime64()),
                            'price': float(price)})

        cur_start = cur_end
        time.sleep(0.2)

    if not data_pk:
        raise ValueError('Coingecko未生成任何数据，请检查参数')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

def _extract_av_close(entry, market):
    fields = [
        f'4b. close ({market.upper()})',
        f'4a. close ({market.upper()})',
        '4. close',
        '4a. close (USD)',
        '4b. close (USD)',
    ]
    for f in fields:
        val = entry.get(f)
        if val is None:
            continue
        try:
            return float(val)
        except ValueError:
            continue
    return None

def get_from_kucoin(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('KuCoin 抓取需要提供 time_start, time_end, time_step_sec')

    step = int(time_step_sec)
    interval = KUCOIN_INTERVALS.get(step)
    if interval is None:
        supported = ', '.join(str(k) for k in sorted(KUCOIN_INTERVALS.keys()))
        raise ValueError(f'KuCoin 只支持以下秒级间隔: {supported}')

    quote = 'USDT' if pair1.upper() == 'USD' else pair1.upper()
    symbol = f'{pair0.upper()}-{quote}'
    limit = 1500
    start_ts = int(pd.Timestamp(time_start).timestamp())
    end_ts = int(pd.Timestamp(time_end).timestamp())
    cur_end = end_ts
    data_pk = []

    while cur_end >= start_ts:
        params = {
            'symbol': symbol,
            'type': interval,
            'endAt': cur_end,
            'startAt': start_ts
        }
        resp = requests.get('https://api.kucoin.com/api/v1/market/candles', params=params, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f'KuCoin 接口返回错误: {resp.status_code} {resp.text}')
        payload = resp.json()
        if payload.get('code') != '200000':
            raise ValueError(f'KuCoin 返回错误: {payload}')
        candles = payload.get('data', [])
        if not candles:
            break

        for candle in candles:
            if len(candle) < 7:
                continue
            ts = np.datetime64(int(candle[0]), 's')
            if ts < time_start or ts > time_end:
                continue
            try:
                close_price = float(candle[2])
            except (TypeError, ValueError):
                continue
            data_pk.append({'time': ts, 'price': close_price})

        earliest = int(candles[-1][0])
        if earliest <= start_ts:
            break
        cur_end = earliest - step
        time.sleep(0.2)

    if not data_pk:
        raise ValueError('KuCoin 未返回所需时间范围的数据')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

def get_from_cryptocompare(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('CryptoCompare 抓取需要提供 time_start, time_end, time_step_sec')
    if time_step_sec <= 0:
        raise ValueError('time_step_sec 必须为正')

    fsym = pair0.upper()
    tsym = pair1.upper()
    headers = {}
    api_key = os.environ.get('CRYPTOCOMPARE_API_KEY')
    if api_key:
        headers['authorization'] = f'Apikey {api_key}'

    def to_unix(ts):
        return int(pd.Timestamp(ts).timestamp())

    start_ts = to_unix(time_start)
    end_ts = to_unix(time_end)
    step = int(time_step_sec)

    endpoint_map = {
        60: ('https://min-api.cryptocompare.com/data/v2/histominute', 60),
        3600: ('https://min-api.cryptocompare.com/data/v2/histohour', 3600),
        86400: ('https://min-api.cryptocompare.com/data/v2/histoday', 86400),
    }

    data_pk = []

    if step in endpoint_map:
        url, granularity = endpoint_map[step]
        limit = 2000
        cur_end = end_ts

        while cur_end >= start_ts:
            max_points = max(1, (cur_end - start_ts) // granularity)
            count = min(limit - 1, max_points)
            params = {
                'fsym': fsym,
                'tsym': tsym,
                'toTs': cur_end,
                'limit': count
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            payload = resp.json()
            if resp.status_code != 200 or payload.get('Response') != 'Success':
                raise ValueError(f'CryptoCompare接口返回错误: {payload}')
            candles = payload.get('Data', {}).get('Data', [])
            if not candles:
                break
            for candle in candles:
                t = np.datetime64(candle['time'], 's')
                if t < time_start or t > time_end:
                    continue
                data_pk.append({'time': t, 'price': float(candle['close'])})
            earliest = np.datetime64(candles[0]['time'], 's')
            if earliest <= time_start:
                break
            cur_end = int((earliest - np.timedelta64(granularity, 's')).astype('int'))
            time.sleep(0.2)
    else:
        # fallback to pricehistorical for arbitrary steps
        url = 'https://min-api.cryptocompare.com/data/pricehistorical'
        cur_ts = start_ts
        while cur_ts <= end_ts:
            params = {'fsym': fsym, 'tsyms': tsym, 'ts': cur_ts}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            payload = resp.json()
            if resp.status_code != 200 or fsym not in payload or tsym not in payload.get(fsym, {}):
                raise ValueError(f'CryptoCompare pricehistorical接口返回错误: {payload}')
            price = payload[fsym][tsym]
            data_pk.append({'time': np.datetime64(cur_ts, 's'), 'price': float(price)})
            cur_ts += step
            time.sleep(0.1)

    if not data_pk:
        raise ValueError('CryptoCompare未返回任何数据，请检查参数或API限制')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

    
if __name__ == '__main__':
    # market = 'kucoin'
    market = 'coingecko'
    # market = 'cryptocompare'
    pair0 = 'BTC' # ETH/BTC/DOGE
    pair1 = 'USD'
    root = f'price_{pair0}_{pair1}'
    now_utc = pd.Timestamp('2025-12-03T00:00:00Z')
    time_end = np.datetime64('2025-12-02T23:59')
    time_start = np.datetime64('2025-11-27T00:00')
    time_step_sec = 60

    os.makedirs(root, exist_ok=True)
    
    if market == 'coingecko':
        data = get_from_coingecko(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'coingecko_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'cryptocompare':
        data = get_from_cryptocompare(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'cryptocompare_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'kucoin':
        data = get_from_kucoin(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'kucoin_{time_start}_{time_end}.pk'), 'wb'))
