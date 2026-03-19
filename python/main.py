import os, sys
import argparse
import copy
import warnings
import pickle
import time

import data
import utils
import acon2

import numpy as np

# def parse_args():
#     ## init a parser
#     parser = argparse.ArgumentParser(description='online learning')
#
#     ## meta args
#     parser.add_argument('--exp_name', type=str, required=True)
#     parser.add_argument('--output_root', type=str, default='output')
#     parser.add_argument('--cpu', action='store_true')
#
#     ## data args
#     parser.add_argument('--data.name', type=str, default='PriceDataset')
#     parser.add_argument('--data.path', type=str, nargs='+', default=[
#         'data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_BTC_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_BTC_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_BTC_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_DOGE_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
#         'data/price_DOGE_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
#
#     ])
#     parser.add_argument('--data.start_time', type=str, default='2025-11-27T00:00')
#     parser.add_argument('--data.end_time', type=str, default='2025-12-02T23:59')
#     parser.add_argument('--data.time_step_sec', type=int, default=60) #60
#     parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)
#
#     ## model args
#     parser.add_argument('--model_base.name', type=str, nargs='+', default=['KF1D', 'KF1D', 'KF1D', 'KF1D'])
#     parser.add_argument('--model_base.score_min', type=float, nargs='+', default=[0.0, 0.0, 0.0])
#     parser.add_argument('--model_base.score_max', type=float, nargs='+', default=[1.0, 1.0, 1.0])
#     parser.add_argument('--model_base.lr', type=float, nargs='+', default=[1e-3, 1e-3, 1e-3])
#     parser.add_argument('--model_base.state_noise_init', type=float, nargs='+', default=[0.1, 0.1, 0.1])
#     parser.add_argument('--model_base.obs_noise_init', type=float, nargs='+', default=[0.1, 0.1, 0.1])
#
#     parser.add_argument('--model_ps.name', type=str, nargs='+', default=['SpecialMVP', 'SpecialMVP', 'SpecialMVP'])
#     parser.add_argument('--model_ps.n_bins', type=int, nargs='+', default=[100, 100, 100])
#
#     parser.add_argument('--model_ps.eta', type=float, default=5)
#     parser.add_argument('--model_ps.alpha', type=float, nargs='+', default=[0.01, 0.01, 0.01])
#     parser.add_argument('--model_ps.beta', type=int, default=1)
#     parser.add_argument('--model_ps.nonconsensus_param', type=float, default=0)
#
#     ## training algorithm args
#     parser.add_argument('--train.method', type=str, default='skip')
#
#     args = parser.parse_args()
#     args = utils.to_tree_namespace(args)
#     args.exp_name = f'{args.exp_name}_K_{len(args.data.path)}_beta_{args.model_ps.beta}'
#     args = utils.propagate_args(args, 'exp_name')
#     args = utils.propagate_args(args, 'output_root')
#
#     ## set loggers
#     os.makedirs(os.path.join(args.output_root, args.exp_name), exist_ok=True)
#     sys.stdout = utils.Logger(os.path.join(args.output_root, args.exp_name, 'out'))
#
#     ## print args
#     utils.print_args(args)
#
#     return args


def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='online learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--cpu', action='store_true')

    ## data args
    parser.add_argument('--data.name', type=str, default='PriceDataset')
    parser.add_argument('--data.path', type=str, nargs='+', default=[
        'data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_BTC_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_BTC_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_BTC_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_DOGE_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk',
        'data/price_DOGE_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk',

    ])
    parser.add_argument('--data.start_time', type=str, default='2025-11-27T00:00')
    parser.add_argument('--data.end_time', type=str, default='2025-12-02T23:59')
    parser.add_argument('--data.time_step_sec', type=int, default=60) #60
    parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)

    ## model args
    parser.add_argument('--model_base.name', type=str, nargs='+', default=['KF1D'])
    parser.add_argument('--model_base.score_min', type=float, nargs='+', default=[0.0])
    parser.add_argument('--model_base.score_max', type=float, nargs='+', default=[1.0])
    parser.add_argument('--model_base.lr', type=float, nargs='+', default=[1e-3])
    parser.add_argument('--model_base.state_noise_init', type=float, nargs='+', default=[0.1])
    parser.add_argument('--model_base.obs_noise_init', type=float, nargs='+', default=[0.1])

    parser.add_argument('--model_ps.name', type=str, nargs='+', default=['SpecialMVP'])
    parser.add_argument('--model_ps.n_bins', type=int, nargs='+', default=[100])

    parser.add_argument('--model_ps.eta', type=float, default=5)
    parser.add_argument('--model_ps.alpha', type=float, nargs='+', default=[0.01])
    parser.add_argument('--model_ps.beta', type=int, default=49)
    parser.add_argument('--model_ps.nonconsensus_param', type=float, default=0)

    ## training algorithm args
    parser.add_argument('--train.method', type=str, default='skip')

    args = parser.parse_args()
    args = utils.to_tree_namespace(args)

    def expand_data_paths(paths):
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

    def ensure_length(lst, target_len, name):
        if len(lst) == 1:
            return lst * target_len
        if len(lst) != target_len:
            raise ValueError(f'{name} 长度应为 1 或 {target_len}，当前为 {len(lst)}')
        return lst

    args.data.path = expand_data_paths(args.data.path)
    K = len(args.data.path)
    args.model_base.name = ensure_length(args.model_base.name, K, 'model_base.name')
    args.model_base.score_min = ensure_length(args.model_base.score_min, K, 'model_base.score_min')
    args.model_base.score_max = ensure_length(args.model_base.score_max, K, 'model_base.score_max')
    args.model_base.lr = ensure_length(args.model_base.lr, K, 'model_base.lr')
    args.model_base.state_noise_init = ensure_length(args.model_base.state_noise_init, K, 'model_base.state_noise_init')
    args.model_base.obs_noise_init = ensure_length(args.model_base.obs_noise_init, K, 'model_base.obs_noise_init')

    args.model_ps.name = ensure_length(args.model_ps.name, K, 'model_ps.name')
    args.model_ps.n_bins = ensure_length(args.model_ps.n_bins, K, 'model_ps.n_bins')
    args.model_ps.alpha = ensure_length(args.model_ps.alpha, K, 'model_ps.alpha')
    args.exp_name = f'{args.exp_name}_K_{len(args.data.path)}_beta_{args.model_ps.beta}'
    args = utils.propagate_args(args, 'exp_name')
    args = utils.propagate_args(args, 'output_root')

    ## set loggers
    os.makedirs(os.path.join(args.output_root, args.exp_name), exist_ok=True)
    sys.stdout = utils.Logger(os.path.join(args.output_root, args.exp_name, 'out'))

    ## print args
    utils.print_args(args)

    return args



def split_args(args):
    args_split = []
    for i in range(len(args.name)):
        args_new = copy.deepcopy(args)
        for d in args.__dict__:
            if type(getattr(args, d)) == list:
                setattr(args_new, d, getattr(args, d)[i])
        args_split.append(args_new)
    return args_split
    

class Clock:
    def __init__(self, time_start, time_end, delta_sec):
        self.time_start = time_start
        self.time_end = time_end
        self.delta_sec = delta_sec


    def __iter__(self):
        self.time = self.time_start
        return self

    def __next__(self):
        if self.time.astype('datetime64[s]') > self.time_end.astype('datetime64[s]'):
            raise StopIteration
        time = self.time
        self.time += self.delta_sec
        return time
        

def run(args):
    time_start = np.datetime64(args.data.start_time)
    time_end = np.datetime64(args.data.end_time)
    time_delta = np.timedelta64(args.data.time_step_sec, 's')

    ## load a dataset
    ds = getattr(data, args.data.name)(args.data.path)

    data_ids = list(ds.seq.keys())

    ## load a base model
    model_base = {k: getattr(acon2, v)(model_base_args) for k, v, model_base_args in zip(data_ids, args.model_base.name, split_args(args.model_base))}
    
    ## load a prediction set
    model_ps_src = {k: getattr(acon2, model_name)(model_args, model_base[k]) for k, model_name, model_args in zip(data_ids, args.model_ps.name, split_args(args.model_ps))}

    model_ps = acon2.ACon2(args.model_ps, model_ps_src)

    ## prediction
    results = []
    outputs_fn = os.path.join(args.output_root, args.exp_name, 'out.pk')
    os.makedirs(os.path.dirname(outputs_fn), exist_ok=True)

    for i, time in enumerate(Clock(time_start, time_end, time_delta)):
        # read observations
        try:
            obs = ds.read(time)
        except StopIteration:
            break

        if all([obs[k] is None for k in obs.keys()]):
            continue
        # update
        if not model_ps.initialized:
            model_ps.init_or_update(obs)
        else:
            model_ps.init_or_update(obs)

            print(f"[time = {time}] median(obs) = {np.median([obs[k] for k in obs.keys() if obs[k] is not None]):.4f}, "\
                  f"interval = [{model_ps.ps[0]:.4f}, {model_ps.ps[1]:.4f}], length = {model_ps.ps[1] - model_ps.ps[0]:.4f}, "\
                  f"error = {model_ps.n_err / model_ps.n_obs:.4f}")
            results.append({'time': time, 'prediction_summary': model_ps.summary(), 'observation': obs})

    # save
    pickle.dump({'results': results, 'args': args}, open(outputs_fn, 'wb'))

            
if __name__ == '__main__':
    start_ts = time.perf_counter()
    args = parse_args()
    run(args)
    elapsed_ms = (time.perf_counter() - start_ts) * 1000
    print(f'[profiling] total runtime = {elapsed_ms:.1f} ms')
