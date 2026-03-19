# Python Files Overview

本文件统一说明 `core` 与 `examples` 目录下主要 `.py` 文件。

## 通用数据格式

TruthFinder 系列实现通常使用以下字段：
- `website`: 数据源/节点标识
- `fact`: 源提供的事实值（文本、离散值或数值）
- `object`: 被讨论对象标识

## `core` 目录（定义文件统一为 `*_truth_finder.py`）

### `core/basic_truth_finder.py`
- 原始 TruthFinder 思路的基础实现。
- 核心接口：
  - `BasicTruthFinder.train(...)`
  - `create_implication_function_from_texts(...)`

### `core/enhanced_truth_finder.py`
- 增强版 TruthFinder（DECENTRUTH 风格更新机制）。
- 同时覆盖两种方法：
  - 非混合模式（`enable_hybrid_mode=False`）
  - 混合模式（`enable_hybrid_mode=True`）
- 核心接口：
  - `EnhancedTruthFinder.train(...)`
  - `EnhancedTruthFinder.process_batch(...)`

### `core/sente_truth_finder.py`
- SenteTruth 语义型真值发现（文本场景）。
- 核心接口：
  - `SenteTruthFinder.train(...)`
  - `SenteTruthFinder.get_source_reliability(...)`

### `core/senfeed_truth_finder.py`
- SenFEED 真值发现实现（CompositeTD / iterative / incremental）。
- 核心接口：
  - `SenFeedTruthDiscovery.train(...)`
  - `SenFeedTruthDiscovery.get_source_reliability(...)`

### `core/__init__.py`
- `core` 包入口，统一导出上述类与函数。

## `examples` 目录（统一示例方法）

### `examples/five_methods_comparison.py`
- 统一示例：
  1. 定义一批数据
  2. 运行五种方法
  3. 输出统一对比结果（误差、来源评分、耗时）

### 其他示例文件
- `examples/enhanced_example.py`
- `examples/comparison_test.py`
- `examples/hybrid_data_example.py`
- `examples/evaluation_comparison.py`
- `examples/senfeed_truth_discovery_comparison.py`

以上文件均作为统一示例入口，执行逻辑与 `five_methods_comparison.py` 保持一致。

常用参数：
- `--data-dir`：输入数据目录，默认 `decentralized_truth_discovery/data`
- `--input-file`：输入文件名，默认 `demo.json`
- `--input-json`：输入文件完整路径，优先级高于前两个参数
- `--experiment-id`：实验名；不传时会按输入文件名自动生成
- `--proof-output-dir`：证明输出目录；不传时默认写入 `examples/outputs/<experiment-id>`

跑前 10 个 batch：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/five_methods_comparison.py --num-experiments 10`

从第 21 个 batch 开始跑 5 次：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/five_methods_comparison.py --start-batch 21 --num-experiments 5`

运行 10% 恶意数据版本：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/five_methods_comparison.py --input-file 'demo_50batch_agent-1_10%wrong_data.json' --num-experiments 10`

使用完整路径运行指定文件：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/five_methods_comparison.py --input-json decentralized_truth_discovery/data/demo_50batch_agent-1_20%wrong_data.json --num-experiments 10`

恶意数据
python3 decentralized_truth_discovery/examples/byzantine_rmse_comparison.py --attacked-file demo_4bad_agent.json --num-batches 10
python3 decentralized_truth_discovery/examples/byzantine_rmse_comparison.py --clean-file num_demo.json --attacked-file num_demo_1bad_agent.json --num-batches 10

混合数据动态注入攻击（只跑 `EnhancedTruthFinder-Hybrid`）：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/dynamic_attack_injection_comparison.py --num-rounds 50 --num-seeds 5 --attack-scale 4.0`

上述命令默认使用 `data_agent-10/demo.json` 与 `demo_xbad_agent.json` 做真实流式注入；其中 `demo.json` 已同时包含文本与数值价格项。

纯数值补充实验（只跑 `EnhancedTruthFinder-Hybrid`，使用 `num_demo*.json`）：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/agent10_numeric_hybrid_supplement.py --num-rounds 50 --num-seeds 5 --bad-agent-count 4 --attack-scale 4.0`

快速烟测：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/dynamic_attack_injection_comparison.py --stream-source agent10 --start-batch 1 --num-rounds 12 --num-seeds 1 --agent10-bad-count 4 --attack-scale 4.0 --skip-plots`

输出文件默认写入：
- `examples/outputs/dynamic_attack_injection/dynamic_attack_summary.csv`
- `examples/outputs/dynamic_attack_injection/dynamic_attack_metrics.csv`
- `examples/outputs/dynamic_attack_injection/dynamic_attack_curve_summary.csv`
- `examples/outputs/dynamic_attack_injection/dynamic_attack_source_shock_summary.csv`
- `examples/outputs/dynamic_attack_injection/hybrid_attack_timing.png`
- `examples/outputs/dynamic_attack_injection/hybrid_weight_shock_timing.png`

数值补充实验也会额外输出：
- `examples/outputs/agent10_numeric_hybrid_supplement/numeric_hybrid_source_shock_summary.csv`
- `examples/outputs/agent10_numeric_hybrid_supplement/numeric_hybrid_weight_shock_timing.png`

Proof 实验脚本：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/proof_scalability_benchmark.py --start-batch 1 --batch-group-size 1 --num-workloads 3 --agent-counts 4,6,8,10 --subset-samples 2`

恶意比例对 proof 影响：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/proof_malicious_ratio_benchmark.py --clean-json decentralized_truth_discovery/data_agent-10/demo.json --attacked-glob 'demo_*bad_agent.json' --num-batches 5`

如果想切到单节点错误比例数据：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/proof_malicious_ratio_benchmark.py --clean-json decentralized_truth_discovery/data/demo.json --attack-dir decentralized_truth_discovery/data --attacked-glob 'demo_50batch_agent-1_*wrong_data.json' --num-batches 5 --total-agents 1`

proof 篡改韧性实验：
`ENABLE_SBERT=0 PYTHONDONTWRITEBYTECODE=1 python3 decentralized_truth_discovery/examples/proof_tamper_resilience_benchmark.py --batch-number 1`

新增输出目录示例：
- `examples/outputs/proof_scalability_*/proof_scalability_runs.csv`
- `examples/outputs/proof_scalability_*/proof_scalability_overall_summary.csv`
- `examples/outputs/proof_malicious_ratio_*/proof_malicious_ratio_summary.csv`
- `examples/outputs/proof_tamper_resilience_*/proof_tamper_resilience_results.csv`
