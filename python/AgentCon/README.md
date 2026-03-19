Macbook Air

Download ollama from https://ollama.com/download

ollama serve

ollama run qwen:0.5b（这个有点笨笨的）
ollama run qwen2.5:1.5b（用这个）

在llm_cfg中改model
llm_cfg = {
        'model': 'qwen2.5:1.5b',
        'model_server': 'http://127.0.0.1:11434/v1',
        'api_key': 'ollama',
    }

查看前100条和后20条加密货币数据内容：python data/preview_price_files.py

python statistic_aggregation.py \
      --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --aggregation truncated_mean \
      --truncation-ratio 0.2

Published on-chain: 93969d4fc4e3261f8653eb53899330c3678bb259fc21595ee562b87807ce5058 -> 3024.68 USD/oz
Published on-chain: 4d4c45557a65915c37476bce0b7bc841af52c9cb12c7451409864a87d74c6216 -> 3035.54 USD/oz
Published on-chain: 3a9872f41b8776c3f65ede138cd7447f866d51317a6c1b9fa89b5da1bddb2e94 -> 3035.36 USD/oz
On-chain truncated mean (ratio=0.2): 3031.86 USD/oz

python statistic_aggregation.py \
      --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --aggregation median

Published on-chain: a30f36513338fabfc54007ad0709a74ce79cd90cd57c3ba84f602653c50a2b27 -> 3024.68 USD/oz
Published on-chain: 1a3b31b1284800dafd57b8a520ed02a99d8496b3070fef9b0f233690eead7e74 -> 3035.54 USD/oz
Published on-chain: d88d79c04777a95e0c9bf286732dfdfffb2a2cdc0118d37ebbbc24f6c805fee6 -> 3035.36 USD/oz
On-chain median: 3035.36 USD/oz

python statistic_aggregation.py \
      --data.path \
      data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_DOGE_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --aggregation median

python twap_aggregation.py \
      --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 

Published on-chain: bc172c46d408290229050f77d7484f86dc26a620e9f8901e9814066bdc96bad9 -> 3024.68 USD/oz
Published on-chain: f05c57f21f4a7790c88a5f6f8989c3eff817a47a94a3264bd156fd0a9812260e -> 3022.37 USD/oz
Published on-chain: 30d2b01791371548c60b579ca632c11b0e92d9b0c4a4b31cee0c02e482f8c6ad -> 3022.47 USD/oz
On-chain TWAP: 3023.17 USD/oz

python twap_aggregation.py \
      --data.path \
      data/price_DOGE_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59

各 agent 在该窗口计算 TWAP，然后取链上中位数。默认不加 --mode 时就是“取end_time（或最新）那条样本”
python agent_aggregation.py --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --mode twap \
      --aggregation median    

On-chain median: 3022.47 USD/oz

Final on-chain ledger snapshot:
{'agent': 'Agent-1', 'price': '3024.68', 'currency': 'USD', 'timestamp': 1766046444, 'tx_hash': 'c1f01b9e3aba56f7fffbd6106e4244575331564169f210c203c8ef6016aabce2'}
{'agent': 'Agent-2', 'price': '3022.37', 'currency': 'USD', 'timestamp': 1766046461, 'tx_hash': '9e5a93a2dd76222fd1df3a3d59e2047a29bec5146d9e8222be5b7cd19f06dd6a'}
{'agent': 'Agent-3', 'price': '3022.47', 'currency': 'USD', 'timestamp': 1766046470, 'tx_hash': '62b184d818c90a625cf24b74a1522fd61f93ff5b4a93003450bde34cbd35e206'}

python agent_aggregation.py --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --mode latest \
      --aggregation median    

On-chain median: 3035.36 USD/oz

Final on-chain ledger snapshot:
{'agent': 'Agent-1', 'price': '3024.68', 'currency': 'USD', 'timestamp': 1766047651, 'tx_hash': 'eae659967d00857e6badfc167effcd51c66c179ef9bbee6696c0f14eb013f1a2'}
{'agent': 'Agent-2', 'price': '3035.54', 'currency': 'USD', 'timestamp': 1766047657, 'tx_hash': '16455d986cb90d49b31ae079a3d63c756414f1231006a0aedb49ae1fb88fc0ba'}
{'agent': 'Agent-3', 'price': '3035.36', 'currency': 'USD', 'timestamp': 1766047664, 'tx_hash': 'e6f226307577bc13afaec54ef151e188ea4b8b98979bd65a4c03e5ef6326a1b8'}

python agent_aggregation.py --data.path \
      data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1 \
      --data.start_time 2025-11-27T00:00 \
      --data.end_time 2025-11-27T00:59 \
      --mode latest \
      --aggregation truncated_mean \
      --truncation-ratio 0.2

On-chain truncated mean: 3031.86 USD/oz

Final on-chain ledger snapshot:
{'agent': 'Agent-1', 'price': '3024.68', 'currency': 'USD', 'timestamp': 1766047866, 'tx_hash': '2317245e5162d02d38e224a1051d0db8a269b01f42e8b07f40d256361e68c2be'}
{'agent': 'Agent-2', 'price': '3035.54', 'currency': 'USD', 'timestamp': 1766047875, 'tx_hash': 'd024d2c7aef94907221a06431150672b66da3252741abfaf282f67fce2db3f17'}
{'agent': 'Agent-3', 'price': '3035.36', 'currency': 'USD', 'timestamp': 1766047896, 'tx_hash': '1838f8ba49e9f0df8099b179c664047e098c6f04d0bfaa9dc8a2e121ee516eb4'}