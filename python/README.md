conda create -n acon python=3.9 -y

conda activate acon

pip install -r requirements.txt
pip install pandas
pip install python-binance
pip install cbpro

通过coingecko/cryptocompare/kucoin免费抓取过去7天的BTC/ETH/DOGE数据：python data/read_offchain_price.py 

查看前100条和后20条数据内容：python data/preview_price_files.py 

数据对齐，保证sample数量相同：python data/preprocess_align_prices.py

# 正常的
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

[MVP] threshold = 0.5000, score = 0.5000, size = 0.0004, interval = [3024.6786, 3024.6789], obs = 3024.6787, error_cur = 0.0, error = 0.8621
[MVP] threshold = 0.4400, score = 0.4788, size = 1.8972, interval = [3035.1439, 3037.0411], obs = 3035.5400, error_cur = 0.0, error = 0.9655
[MVP] threshold = 0.4400, score = 0.4797, size = 1.9009, interval = [3034.9512, 3036.8521], obs = 3035.3600, error_cur = 0.0, error = 0.9655
[time = 2025-11-27T00:59:00] median(obs) = 3035.3600, interval = [3035.1439, 3036.8521], length = 1.7082, error = 0.9828

# 一个恶意的
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

[MVP] threshold = 0.5000, score = 0.5000, size = 0.0004, interval = [0.1546, 0.1549], obs = 0.1548, error_cur = 0.0, error = 0.8621
[MVP] threshold = 0.4400, score = 0.4788, size = 1.8972, interval = [3035.1439, 3037.0411], obs = 3035.5400, error_cur = 0.0, error = 0.9655
[MVP] threshold = 0.4400, score = 0.4797, size = 1.9009, interval = [3034.9512, 3036.8521], obs = 3035.3600, error_cur = 0.0, error = 0.9655
[time = 2025-11-27T00:59:00] median(obs) = 3035.3600, interval = [3035.1439, 3036.8521], length = 1.7082, error = 0.9828

beta=30
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@30 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@30 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@40 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

       [profiling] total runtime = 7854.7 ms

beta=49
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@40 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@30 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@30 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

       [profiling] total runtime = 7762.4 ms


beta=4
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@4 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@3 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@3 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

beta=499
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@499 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@300 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@301 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59

beta=1
python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:59


python main.py \
       --exp_name demo_run \
       --data.path \
         data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
         data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
         data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
       --data.start_time 2025-11-27T00:00 \
       --data.end_time 2025-11-27T00:04
