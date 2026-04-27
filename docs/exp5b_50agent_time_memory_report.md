# Exp5b 50-Agent Distributed Time/Memory Experiment

## 1. 实验目的

Exp5b 的目标是评估五类 truth discovery 方法在 50 个 agent 场景下的可扩展性。本次实验只使用两个可扩展性指标：

- 时间开销：方法计算 + 真实通信开销。
- 内存开销：coordinator 进程 + 50 个 worker 进程的总进程内存占用。

本实验不使用 source score、quality rank、composite rank 作为 exp5b 结论指标。source score 是方法内部对 source/agent 可靠性的估计，适合作为质量代理指标，但不适合作为本实验的可扩展性指标。

## 2. 实验环境

- 远端机器：`10.3.242.100`
- 远端用户：`gao`
- 远端工程目录：`/home/gao/TruthFinder_exp5b`
- 数据目录：`/home/gao/data/data_agent-50_news-300`
- Python：`/home/gao/anaconda3/bin/python`
- 文本编码模型：`sentence-transformers/all-MiniLM-L6-v2`
- 文本运行方式：强制 CPU SBERT，运行时设置 `CUDA_VISIBLE_DEVICES=""`，避免 GPU 显存占用和 OOM 干扰。
- 通信方式：ZeroMQ TCP，`msgpack` 序列化。

本地拉回的数值结果目录：

```text
decentralized_truth_discovery/examples/outputs/remote_10_3_242_100_exp5b_50agent_time_memory/
```

本地拉回的新闻摘要文本结果目录：

```text
decentralized_truth_discovery/examples/outputs/remote_10_3_242_100_exp5b_50agent_news_summary_text/
```

主要结果文件：

```text
remote_10_3_242_100_exp5b_50agent_time_memory/exp5b_numeric_dist_50agent_time_memory/summary_time_memory.csv
remote_10_3_242_100_exp5b_50agent_time_memory/exp5b_numeric_dist_50agent_time_memory/raw_runs.csv
remote_10_3_242_100_exp5b_50agent_news_summary_text/summary_time_memory.csv
remote_10_3_242_100_exp5b_50agent_news_summary_text/raw_runs.csv
remote_10_3_242_100_exp5b_50agent_time_memory/logs/exp5b_50agent_time_memory.log
remote_10_3_242_100_exp5b_50agent_news_summary_text/logs/exp5b_50agent_news_summary_text_time_memory.log
```

## 3. 分布式实验设计

### 3.1 分布式模型

本实验是单机多进程分布式模拟，不是单进程 in-process benchmark。

具体实现方式：

- coordinator 在主进程中运行。
- 每个 agent 对应一个独立 worker 进程。
- 50-agent 实验中，每次方法运行会启动 50 个 `agent_worker_entry.py` worker 进程。
- worker 通过 ZeroMQ DEALER socket 连接 coordinator。
- coordinator 通过 ROUTER socket 对 worker 执行 send、gather、broadcast。
- 通信走真实 TCP socket：数值实验端口 `tcp://127.0.0.1:5560`，文本实验端口 `tcp://127.0.0.1:5588`。

结果文件验证：

```text
mode = dist-tcp
transport = zmq-tcp
serialization = msgpack
agent_count = 50
selected_agent_count = 50
fleet_proc_count = 50
```

因此本实验不是本地函数调用模拟，而是包含真实跨进程通信和序列化开销的分布式运行。

### 3.2 为什么每个方法单独冷启动

早期连续运行多个方法时，进程 RSS 高水位会受到方法顺序影响。例如前一个方法加载了大型库或模型后，后一个方法的内存增量会被压低。

本次实验改为每个方法单独启动一次完整 distributed run：

- 每个方法拥有独立 coordinator 运行窗口。
- 每个方法重新启动 50 个 worker 进程。
- 每个方法单独输出 `raw_runs.csv` 和 `summary_time_memory.csv`。
- 最后只合并 time/memory 结果。

这样可以避免方法顺序污染内存指标，同时保留真实通信开销。

## 4. 实验方法

### 4.1 数值场景

数值场景运行五种方法：

- `SenFeedTruth`
- `DecentTruth`
- `SenteTruth`
- `BasicTruth`
- `SemanticTruth`

配置：

```text
agent_count = 50
num_workloads = 3
batch_group_size = 10
subset_samples = 2
```

由于数据集中正好是 50 个 agent，当 `agent_count=50` 时只有一个完整 agent 子集，因此实际每个 workload 只产生一个 subset。数值实验总运行记录数为：

```text
5 methods * 3 workloads * 1 subset = 15 runs
```

所有 15 条记录均成功。

### 4.2 文本场景

文本场景运行三种方法：

- `BasicTruth`
- `SemanticTruth`
- `SenteTruth`

配置：

```text
agent_count = 50
num_workloads = 1
batch_group_size = 5
subset_samples = 1
use_sbert = true
input_text_json = /home/gao/data/data_agent-50_news-300/demo.json
```

文本输入明确使用新闻摘要记录：`demo.json` 中 `response` 为字符串的条目，例如 object 为 `第1条新闻简短摘要` 的新闻摘要。价格记录是 `response` 为 dict 的条目，文本 loader 会过滤掉这些 dict 记录。

文本实验总运行记录数为：

```text
3 methods * 1 workload * 1 subset = 3 runs
```

每条文本 run 的输入规模为：

```text
input_item_count = 250
input_object_count = 5
```

其中 `paired_numeric_input_item_count=750` 只表示同一批 batch 中配套存在的价格记录数量，不是文本方法的实际输入。

所有 3 条记录均成功。

日志检查结果：

- 无 `Falling back`
- 无 `CUDA out`
- 无 `Failed to load SBERT`
- 无 `TimeoutError`
- 无 `status=FAIL`

## 5. 指标定义

### 5.1 时间指标

本实验主要使用以下时间指标：

| 指标 | 含义 |
|---|---|
| `mean_elapsed_with_comm_sec` | 方法计算时间 + 通信时间。该值是本文主时间指标。 |
| `mean_wall_time_sec` | 完整运行墙钟时间，包含更多运行时冷启动、首次 import、worker 启动后首次库初始化等成本。 |
| `mean_network_sec` | 分布式 send/gather/broadcast 等通信时间。 |

代码中 distributed 模式的 `elapsed_sec` 定义为：

```text
elapsed_sec = method_compute_sec + network_sec
```

因此 `mean_elapsed_with_comm_sec` 可以反映方法本身和通信协议共同造成的运行时间。

### 5.2 内存指标

本实验主要使用以下内存指标：

| 指标 | 含义 |
|---|---|
| `mean_total_peak_rss_mb` | coordinator peak RSS + 50 个 worker peak RSS 之和。该值是本文主内存指标。 |
| `mean_total_peak_delta_rss_mb` | coordinator 和 worker 在采样窗口中的 peak delta 之和。 |
| `mean_coord_peak_rss_mb` | coordinator 进程 peak RSS。 |
| `mean_fleet_peak_rss_mb_sum` | 50 个 worker 进程 peak RSS 之和。 |
| `mean_fleet_peak_delta_rss_mb_sum` | 50 个 worker 在方法运行窗口内的 RSS 增量之和。 |

`total peak RSS` 更适合描述 50-agent 分布式部署的总体内存 footprint。`delta` 可作为增量参考，但它依赖采样起点；如果 worker 在采样前已经预加载 SBERT，模型常驻内存会主要体现在 peak RSS，而不一定体现在 delta。

## 6. 协议形状与理论预期

不同方法的通信轮数不同，这是数值场景时间差异的主要来源之一。

| 方法 | 协议轮次设计 | 理论含义 |
|---|---|---|
| `BasicTruth` | gather + 5 次 extra round trip + reliability broadcast | TruthFinder 的 trust-confidence 互相强化需要多轮迭代，通信轮数最高。 |
| `DecentTruth` | gather + reliability broadcast | 本地估计到全局聚合再广播可靠性，单轮协议。 |
| `SenFeedTruth` | gather + 2 次 extra round trip + reliability broadcast | 增量 MAP 本身较轻，但 shock/fallback 路径按平均额外轮次计入。 |
| `SenteTruth` | gather + reliability broadcast | 单轮文本/数值 vote-style 协议。 |
| `SemanticTruth` | gather + reliability broadcast | 双通道/软投票协议，单轮通信。 |

实际 RPC 统计与协议设计一致：

- `BasicTruth`：`rpc_count=350`
- `SenFeedTruth`：`rpc_count=200`
- `DecentTruth/SenteTruth/SemanticTruth`：`rpc_count=100`

## 7. 数值实验结果

主结果表如下，均为 50-agent distributed TCP 运行。

| 方法 | runs | success | wall time(s) | elapsed+comm(s) | network(s) | total peak RSS(MB) | total peak delta(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SenFeedTruth` | 3 | 3 | 1.8025 | 0.2786 | 0.0298 | 1806.8 | 232.4 |
| `DecentTruth` | 3 | 3 | 1.8530 | 0.3026 | 0.0213 | 1805.2 | 232.1 |
| `SenteTruth` | 3 | 3 | 1.8314 | 0.3150 | 0.0259 | 1806.1 | 232.2 |
| `BasicTruth` | 3 | 3 | 2.7897 | 1.2501 | 0.0492 | 1805.9 | 232.1 |
| `SemanticTruth` | 3 | 3 | 1.6650 | 0.1297 | 0.0199 | 1805.3 | 232.1 |

### 7.1 数值结果解析

数值场景中，五种方法的 total peak RSS 都在约 `1.8GB`。原因是该实验架构下，主要内存来自：

- 50 个 Python worker 进程的基础常驻内存。
- coordinator 进程加载数据、算法代码和运行时库的内存。

数值场景没有 SBERT 模型复制，因此不同方法之间的内存差异很小。

时间方面：

- `SemanticTruth` 的 `elapsed+comm` 最低，为 `0.1297s`。在数值场景中它不使用 SBERT，仅执行轻量数值一致性/软投票逻辑，且通信轮数为单轮。
- `SenFeedTruth`、`DecentTruth`、`SenteTruth` 时间接近，均在 `0.28s-0.32s` 左右。它们都属于轻量数值聚合或单轮 vote-style 协议。
- `BasicTruth` 时间最高，为 `1.2501s`。这是因为它保留 TruthFinder 的 trust-confidence 多轮更新思想，实验协议中有 5 次额外 round trip，RPC 计数达到 `350`，高于其他方法。

因此数值场景的主要结论是：

```text
内存主要由 50 个 worker 基线决定；时间主要由通信轮数和 coordinator 聚合复杂度决定。
```

## 8. 文本实验结果

主结果表如下，均为 50-agent distributed TCP + SBERT 运行。

| 方法 | runs | success | wall time(s) | elapsed+comm(s) | network(s) | total peak RSS(MB) | total peak delta(MB) | fleet peak RSS sum(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BasicTruth` | 1 | 1 | 8.7821 | 4.3965 | 0.0386 | 1808.4 | 658.1 | 951.9 |
| `SemanticTruth` | 1 | 1 | 7.6675 | 3.0371 | 1.7405 | 46792.4 | 2597.8 | 45684.1 |
| `SenteTruth` | 1 | 1 | 7.9166 | 2.8730 | 1.8959 | 46779.8 | 2587.3 | 45690.7 |

### 8.1 文本结果解析

文本场景的主要差异来自是否使用 SBERT，以及文本聚合逻辑的复杂度。

#### BasicTruth

`BasicTruth` 的文本路径使用 TF-IDF implication，不要求每个 worker 预加载 SBERT。因此：

- worker fleet peak RSS 约 `951MB`，接近数值场景 worker 基线。
- total peak RSS 约 `1.8GB`，显著低于 SemanticTruth/SenteTruth。
- network time 只有 `0.0386s`，因为 worker 不执行 SBERT 编码，通信负担较低。

但 `BasicTruth` 的 `elapsed+comm` 最高，为 `4.3965s`。原因是：

- BasicTruth 要按 batch prefix 构建 TF-IDF implication。
- TruthFinder 的 trust-confidence 更新逻辑比简单向量投票更重。
- 协议保留 5 次额外 round trip，RPC 计数为 `350`。

因此它表现为低内存、高时间。

#### SemanticTruth

`SemanticTruth` 使用 SBERT，并且文本模式下 50 个 worker 会预加载 SBERT，以反映真实分布式部署中模型复制带来的内存 footprint。因此：

- fleet peak RSS sum 达到约 `45684MB`。
- total peak RSS 达到约 `46792MB`。
- network time 为 `1.7405s`，明显高于 BasicTruth，因为 worker 端还要执行文本编码相关操作，并通过通信协议完成 gather/broadcast。

时间上，`SemanticTruth` 的 `elapsed+comm` 为 `3.0371s`，低于 BasicTruth，高于 SenteTruth。原因是它不仅做语义 embedding，还包含双通道/软投票、source reliability、semantic affinity 等逻辑。

#### SenteTruth

`SenteTruth` 同样使用 SBERT，因此内存量级与 SemanticTruth 接近：

- fleet peak RSS sum 约 `45691MB`
- total peak RSS 约 `46780MB`

但它的 `elapsed+comm` 最低，为 `2.8730s`。原因是 SenteTruth 的文本聚合逻辑更直接：

- SBERT 编码文本。
- 计算 cosine similarity。
- 基于节点信誉进行加权选择。

相对 SemanticTruth，它少了更复杂的双通道融合和 affinity 更新逻辑，因此时间更低。

文本场景的主要结论是：

```text
SBERT 方法承担模型复制带来的高内存；BasicTruth 不加载 SBERT，内存低但 TF-IDF implication 与多轮 TruthFinder 更新使时间最高。
```

## 9. 为什么文本 SBERT 方法的内存远高于 BasicTruth

在 50-agent distributed 设置下，SemanticTruth 和 SenteTruth 的 worker 进程会各自预加载 SBERT。也就是说，模型不是在一个进程里共享一次，而是被 50 个 worker 进程分别持有。

这导致：

```text
50 workers * SBERT/torch runtime footprint => 约 45GB fleet peak RSS
```

BasicTruth 文本路径不预加载 SBERT，所以它的 worker fleet peak RSS 仍接近普通 Python worker 基线，约 `951MB`。

这正是分布式部署中模型型方法和非模型型方法在内存可扩展性上的核心差异。

## 10. 为什么数值场景 worker delta 为 0

数值场景中 worker 的主要工作是：

- 接收切片。
- 进行轻量本地统计/ACK。
- 参与 iterate 和 reliability update 的通信回合。

算法聚合主要在 coordinator 中完成。worker 进程启动后已有稳定 RSS，方法运行期间没有明显额外增长，因此 `fleet_peak_delta_rss_mb_sum=0`。但这不表示 worker 没有内存占用；应同时看 `mean_fleet_peak_rss_mb_sum`，数值场景约为 `951MB`。

## 11. 有效性核验

本次实验完成后做了以下检查：

1. 数值结果：
   - `raw_runs.csv` 共 15 条记录。
   - 15 条均成功。
   - 所有记录 `agent_count=50`。
   - 所有记录 `selected_agent_count=50`。
   - 所有记录 `fleet_proc_count=50`。

2. 文本结果：
   - `raw_runs.csv` 共 3 条记录。
   - 3 条均成功。
   - 所有记录 `agent_count=50`。
   - 所有记录 `selected_agent_count=50`。
   - 所有记录 `fleet_proc_count=50`。
   - `use_sbert=True`。
   - 所有记录 `input_text_json=/home/gao/data/data_agent-50_news-300/demo.json`。
   - 所有记录 `input_item_count=250`，表示实际输入为 250 条新闻摘要文本。
   - 所有记录 `input_object_count=5`，表示 5 条新闻摘要对象。

3. 分布式核验：
   - 所有记录 `mode=dist-tcp`。
   - 所有记录 `transport=zmq-tcp`。
   - 所有记录 `serialization=msgpack`。
   - 日志中每个方法均显示 `[dist-tcp|workload ...] agents=50 ... status=OK`。

4. SBERT 核验：
   - 日志中没有 `Falling back`。
   - 日志中没有 `CUDA out`。
   - 日志中没有 `Failed to load SBERT`。
   - 日志中没有 `TimeoutError`。
   - 日志中没有 `status=FAIL`。

## 12. 结论

本次 exp5b 50-agent 实验表明：

1. 数值场景下，五种方法的内存 footprint 接近，主要受 50 个 worker 进程基线控制。
2. 数值场景下，时间差异主要来自通信轮数和 coordinator 聚合逻辑；BasicTruth 因多轮 trust-confidence 更新和最多 RPC 而最慢。
3. 文本场景下，SBERT 方法的内存 footprint 显著高于 BasicTruth，因为 50 个 worker 分别持有 SBERT/torch runtime。
4. 文本场景下，SenteTruth 比 SemanticTruth 更快，原因是其文本聚合逻辑更直接；SemanticTruth 的双通道/soft voting/affinity 更新更复杂。
5. BasicTruth 文本路径内存最低，但时间最高，因为它使用 TF-IDF implication 和 TruthFinder 多轮更新。

总体上，本实验符合方法理论预期：

```text
数值场景：通信轮数和聚合复杂度决定时间，worker 基线决定内存。
文本场景：SBERT 模型复制决定内存，文本聚合机制决定时间。
```
