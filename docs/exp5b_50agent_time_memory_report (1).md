# Exp5b 50-Agent Distributed Time/Memory Experiment

## 1. 实验目的

Exp5b 的目标是评估 truth discovery 方法在 50 个 agent 场景下的分布式可扩展性。本次全量实验只使用两个可扩展性指标：

- 时间开销：方法计算 + 真实跨进程通信/等待开销。
- 内存开销：coordinator 进程 + 50 个 worker 进程的总进程内存占用。

本实验不使用 source score、quality rank、composite rank 作为 exp5b 结论指标。source score 是方法内部对 source/agent 可靠性的估计，适合作为质量代理指标，但不适合作为本实验的可扩展性指标。

## 2. 实验环境

- 远端机器：`10.3.242.100`
- 远端用户：`gao`
- 远端工程目录：`/home/gao/TruthFinder_exp5b`
- 数据目录：`/home/gao/data/data_agent-50_news-300`
- Python：`/home/gao/anaconda3/bin/python`
- 文本编码模型：`sentence-transformers/all-MiniLM-L6-v2`
- 通信方式：ZeroMQ TCP，`msgpack` 序列化。
- 数值端口：`tcp://127.0.0.1:5560`
- 文本端口：`tcp://127.0.0.1:5588`

## 3. 分布式实验设计

### 3.1 分布式模型

本实验是单机多进程分布式模拟，不是单进程 in-process benchmark。

具体实现方式：

- coordinator 在主进程中运行。
- 每个 agent 对应一个独立 worker 进程。
- 50-agent 实验中，每次方法运行会启动 50 个 `agent_worker_entry.py` worker 进程。
- worker 通过 ZeroMQ DEALER socket 连接 coordinator。
- coordinator 通过 ROUTER socket 对 worker 执行 send、gather、broadcast。
- 通信走真实 TCP socket，并使用 `msgpack` 序列化 payload。

结果文件验证：

```text
mode = dist-tcp
transport = zmq-tcp
serialization = msgpack
agent_count = 50
selected_agent_count = 50
fleet_proc_count = 50
```

因此本实验不是本地函数调用模拟，而是包含真实跨进程通信、序列化和 worker 进程内存开销的分布式运行。

### 3.2 每个方法单独冷启动

为了避免内存指标受到方法顺序污染，本实验每个方法都单独启动一次完整 distributed run：

- 每个方法拥有独立 coordinator 运行窗口。
- 每个方法重新启动 50 个 worker 进程。
- 每个方法单独输出 `raw_runs.csv` 和 `summary_time_memory.csv`。
- 最后只合并 time/memory 结果。

这样可以避免前一个方法加载库或模型后影响后一个方法的 RSS 高水位，同时保留真实通信开销。

### 3.3 全量运行配置

本次更新后的实验明确跑完整 JSON 文件，而不是只跑部分 workload。

数值全量运行使用：

```text
input_json = /home/gao/data/data_agent-50_news-300/num_demo.json
batch_start = 1
batch_end = 300
batch_group_size = 300
num_workloads = 1
agent_count = 50
subset_samples = 1
```

文本全量运行使用：

```text
input_json = /home/gao/data/data_agent-50_news-300/num_demo.json
input_text_json = /home/gao/data/data_agent-50_news-300/demo.json
text_mode = True
use_sbert = True
batch_start = 1
batch_end = 300
batch_group_size = 300
num_workloads = 1
agent_count = 50
subset_samples = 1
```

文本 SBERT 方法使用 `CUDA_VISIBLE_DEVICES=""` 强制 CPU SBERT 运行，避免 GPU 显存不足导致 fallback。本次 iter1 全量实验从启动阶段即禁用 GPU，最终日志中没有 SBERT fallback、CUDA OOM、timeout 或失败记录。

### 3.4 原本真实部署应该怎么跑

如果按真实分布式部署来跑，50 个 agent 应该是 50 个独立参与方。它们可以位于不同机器或不同容器中，各自持有自己的观测数据和本地模型状态，通过网络向 coordinator 或协议中的聚合节点发送本地结果，再接收全局可靠性、truth estimate 或下一轮迭代消息。

真实部署中的理想流程是：

- 每个 agent 独立采集或持有自己的 claim。
- 每个 agent 在本地完成必要的预处理或局部计算，例如数值统计、文本 embedding、本地可靠性更新等。
- coordinator 或协议聚合节点通过网络收集 agent 输出。
- coordinator 执行全局 truth discovery 聚合，并把 reliability/truth 相关状态发回 agent。
- 对 BasicTruth 这类迭代式方法，真实算法可以展开多轮 trust-confidence 更新；但本次 Exp5b 仿真实验为了公平比较通信开销，所有方法都统一只执行 1 次 `iterate` 消息轮次。
- 对 SemanticTruth/SenteTruth 这类文本方法，每个 agent 如果部署了 SBERT，就需要在自己的进程或机器上持有模型副本。

也就是说，真实实验最理想的形式是跨机器或跨容器的多节点部署，网络、进程隔离、模型复制和消息序列化都真实存在。

### 3.5 本实验作为仿真实验是怎么跑的

本实验没有部署 50 台真实机器，而是在 `10.3.242.100` 单机上用 50 个独立 OS 进程模拟 50 个 agent。这样做的目的不是模拟单进程函数调用，而是在可控环境下保留真实进程隔离、真实 TCP socket、真实序列化和每个 worker 的独立内存 footprint。

实际仿真方式是：

- coordinator 在主 Python 进程中运行。
- 每个 agent 启动一个独立 `agent_worker_entry.py` worker 进程。
- 50-agent run 中，每次方法运行都有 50 个 worker 进程。
- coordinator 使用 ZeroMQ `ROUTER` socket。
- worker 使用 ZeroMQ `DEALER` socket。
- coordinator 和 worker 之间通过 `tcp://127.0.0.1:5560` 或 `tcp://127.0.0.1:5588` 通信。
- 消息使用 `msgpack` 序列化。
- 数据按照 agent id 切片后发给对应 worker。
- coordinator 复用现有 truth discovery 算法实现完成全局聚合，避免分布式实验和单进程算法实现出现两套逻辑。

因此，本实验是单机多进程分布式仿真：它没有跨物理机器的真实网络抖动和带宽限制，但它包含真实跨进程通信、真实序列化、真实 worker 进程 RSS，以及文本 SBERT 方法的模型复制内存。

### 3.6 RPC 调用的含义和调用内容

本文中的 `RPC` 不是区块链 RPC，也不是 HTTP/gRPC 接口。这里的 RPC 指 coordinator 和 agent worker 之间的一次 request/reply 消息往返。代码中用 ZeroMQ ROUTER/DEALER 实现，统计口径来自 coordinator 发送请求到收到对应 worker 回复的 latency sample。

本实验中的主要 RPC 消息类型如下：

| RPC 消息 | 发送方向 | 作用 |
|---|---|---|
| `process_batch` | coordinator -> worker | 数值场景。coordinator 把某个 agent 的数值 claim 切片发给对应 worker；worker 做轻量本地统计和计时后返回 ack。 |
| `process_text_batch` | coordinator -> worker | 文本场景。coordinator 把某个 agent 的新闻摘要 claim 切片发给 worker；BasicTruth worker 不做 SBERT 编码，SemanticTruth/SenteTruth worker 会执行本地文本编码并返回 ack。 |
| `iterate` | coordinator -> worker | 统一的一次迭代 tick。所有方法都执行 1 次，用于让各方法承受相同的 coordinator-worker 往返调度开销。 |
| `reliability_update` | coordinator -> worker | coordinator 把 source/agent reliability snapshot 广播给 worker；worker 存储快照并返回 ack。 |
| `shutdown` | coordinator -> worker | 方法结束后关闭 worker；用于进程生命周期管理，不计入方法主体的 `rpc_count`。 |

`rpc_count` 是按 agent 粒度计数的，所以一次发给 50 个 worker 并等待 50 个回复，会贡献 50 个 RPC latency sample。本次 iter1 实验统一所有方法的消息轮次，每个方法的主体 RPC 计数相同：

```text
all methods:
  process_batch/process_text_batch 50 次
  + 1 轮 iterate * 50 次
  + reliability_update 50 次
  = 150
```

worker 启动时发送的 `hello` 和结束时的 `shutdown` 主要用于进程管理，不作为方法主体性能指标中的 RPC 轮次解释。

### 3.7 本实验和区块链部分的关系

本实验和区块链部分没有直接关系。Exp5b 测的是 truth discovery 方法在 50-agent 分布式设置下的 off-chain 计算、通信和内存开销，不涉及链上交易、智能合约调用、共识延迟、区块确认、ChainMaker/EVM 部署，也不涉及 zk proof 的生成或链上验证。

它可以被理解为 blockchain oracle 系统中的 off-chain truth discovery 层可扩展性实验。区块链相关工作属于后续上链、证明或合约验证环节，应单独作为 proof/on-chain benchmark 评估，不能把本实验的 RPC、时间和内存结果解释为区块链性能结果。

## 4. 输入规模

### 4.1 数值场景

数值场景使用完整 `num_demo.json`：

```text
batches = 300
agents = 50
numeric objects = 3: BTC, DOGE, ETH
numeric claims = 45000
```

数值场景运行五种方法：

- `SenFeedTruth`
- `DecentTruth`
- `SenteTruth`
- `BasicTruth`
- `SemanticTruth`

总运行记录数为：

```text
5 methods * 1 full workload * 1 subset = 5 runs
```

所有 5 条记录均成功。

### 4.2 文本场景

文本场景使用完整 `demo.json`。文本 loader 只取 `response` 为字符串的新闻摘要记录，过滤掉 `response` 为 dict 的价格记录。

```text
batches = 300
agents = 50
text objects = 300
text claims = 15000
paired numeric objects = 3
paired numeric claims = 45000
```

文本场景运行三种方法：

- `BasicTruth`
- `SemanticTruth`
- `SenteTruth`

总运行记录数为：

```text
3 methods * 1 full workload * 1 subset = 3 runs
```

所有 3 条记录均成功。`paired_numeric_input_item_count=45000` 只表示同一批 batch 中配套存在的价格记录数量，不是文本方法的实际输入；文本方法的实际输入是 `input_item_count=15000`、`input_object_count=300`。

## 5. 指标定义

### 5.1 时间指标

| 指标 | 含义 |
|---|---|
| `mean_elapsed_with_comm_sec` | 方法计算时间 + 分布式通信/等待时间。该值是本文主时间指标。 |
| `mean_wall_time_sec` | 完整运行墙钟时间，包含冷启动、首次 import、worker 启动、模型加载等成本。 |
| `mean_network_sec` | coordinator 聚合之外的 send/gather/broadcast 等协议时间；文本 SBERT 场景中也包含等待 worker 本地编码后回复的时间。 |

代码中 distributed 模式的 `elapsed_sec` 定义为：

```text
elapsed_sec = method_compute_sec + network_sec
```

因此 `mean_elapsed_with_comm_sec` 可以反映方法本身和分布式协议共同造成的运行时间。

### 5.2 内存指标

| 指标 | 含义 |
|---|---|
| `mean_total_peak_rss_mb` | coordinator peak RSS + 50 个 worker peak RSS 之和。该值是本文主内存指标。 |
| `mean_total_peak_delta_rss_mb` | coordinator 和 worker 在采样窗口中的 peak delta 之和。 |
| `mean_coord_peak_rss_mb` | coordinator 进程 peak RSS。 |
| `mean_fleet_peak_rss_mb_sum` | 50 个 worker 进程 peak RSS 之和。 |
| `mean_fleet_peak_delta_rss_mb_sum` | 50 个 worker 在方法运行窗口内的 RSS 增量之和。 |

`total peak RSS` 更适合描述 50-agent 分布式部署的总体内存 footprint。`delta` 可作为增量参考，但它依赖采样起点；如果 worker 在采样前已经预加载 SBERT，模型常驻内存会主要体现在 peak RSS，而不一定完全体现在 delta。

## 6. 协议形状与理论预期

本次 iter1 实验统一所有方法的通信轮数，避免把方法时间差异混入不同数量的 RPC 往返开销。所有方法都使用同一协议形状：

| 方法 | 协议轮次设计 | RPC count |
|---|---|---:|
| `SenFeedTruth` | gather + 1 次 iterate + reliability broadcast | 150 |
| `DecentTruth` | gather + 1 次 iterate + reliability broadcast | 150 |
| `SenteTruth` | gather + 1 次 iterate + reliability broadcast | 150 |
| `BasicTruth` | gather + 1 次 iterate + reliability broadcast | 150 |
| `SemanticTruth` | gather + 1 次 iterate + reliability broadcast | 150 |

因此，本实验中的时间差异不应再解释为“某个方法通信轮数更多”。所有方法的 `rpc_count=150`，时间差异主要来自 coordinator 聚合逻辑、文本编码/相似度计算、TF-IDF implication、以及方法内部算法复杂度。

## 7. 数值实验结果

主结果表如下，均为完整 `num_demo.json`、50-agent distributed TCP 运行。

| 方法 | runs | success | wall time(s) | elapsed+comm(s) | network(s) | total peak RSS(MB) | total peak delta(MB) | fleet peak RSS sum(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SenFeedTruth` | 1 | 1 | 11.6887 | 6.9504 | 0.2072 | 1882.4 | 771.8 | 1014.5 |
| `DecentTruth` | 1 | 1 | 13.0528 | 8.2898 | 0.2050 | 1871.1 | 760.0 | 1015.0 |
| `SenteTruth` | 1 | 1 | 13.4852 | 8.5491 | 0.1954 | 1871.3 | 761.0 | 1014.5 |
| `BasicTruth` | 1 | 1 | 42.0038 | 36.9090 | 0.1933 | 1873.4 | 761.2 | 1014.6 |
| `SemanticTruth` | 1 | 1 | 7.9123 | 3.0002 | 0.2136 | 1871.6 | 761.9 | 1013.9 |

### 7.1 数值结果解析

数值场景中，五种方法的 total peak RSS 都在约 `1.87GB-1.88GB`。主要内存来自：

- 50 个 Python worker 进程的基础常驻内存，合计约 `1.014GB`。
- coordinator 进程加载数据、算法代码和运行时库的内存。

数值场景没有 SBERT 模型复制，因此不同方法之间的内存差异很小。

时间方面：

- `SemanticTruth` 的 `elapsed+comm` 最低，为 `3.0002s`。在数值场景中它不使用 SBERT，仅执行轻量数值一致性/软投票逻辑。
- `SenFeedTruth` 为 `6.9504s`，低于 `DecentTruth` 和 `SenteTruth`。
- `DecentTruth` 和 `SenteTruth` 接近，分别为 `8.2898s` 和 `8.5491s`。
- `BasicTruth` 最高，为 `36.9090s`。在统一 `rpc_count=150` 后，它仍然最慢，说明主要差异来自 TruthFinder trust-confidence 更新和全量数值聚合逻辑本身，而不是更多 RPC 轮次。

因此数值场景的主要结论是：

```text
完整 300-batch 数值输入下，内存主要由 50 个 worker 基线决定；在统一 RPC 轮次后，时间主要由方法聚合复杂度决定。
```

## 8. 文本实验结果

主结果表如下，均为完整 `demo.json`、50-agent distributed TCP 运行。`SemanticTruth` 和 `SenteTruth` 使用 CPU SBERT；`BasicTruth` 的文本路径使用 TF-IDF implication，不预加载 SBERT worker。

| 方法 | runs | success | wall time(s) | elapsed+comm(s) | network(s) | total peak RSS(MB) | total peak delta(MB) | fleet peak RSS sum(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BasicTruth` | 1 | 1 | 553.8999 | 549.0969 | 0.0770 | 2003.7 | 854.2 | 1014.3 |
| `SemanticTruth` | 1 | 1 | 99.0550 | 94.4821 | 34.9531 | 57507.9 | 13271.5 | 55958.0 |
| `SenteTruth` | 1 | 1 | 76.3015 | 71.4274 | 35.4299 | 57215.9 | 12982.1 | 55652.2 |

### 8.1 文本结果解析

文本场景的主要差异来自是否使用 SBERT，以及文本聚合逻辑的复杂度。

#### BasicTruth

`BasicTruth` 的文本路径使用 TF-IDF implication，不要求每个 worker 预加载 SBERT。因此：

- worker fleet peak RSS 约 `1012MB`，接近数值场景 worker 基线。
- total peak RSS 约 `2.02GB`，显著低于 SemanticTruth/SenteTruth。
- network time 只有 `0.0770s`，因为 worker 不执行 SBERT 编码，通信等待较低。

但 `BasicTruth` 的 `elapsed+comm` 最高，为 `549.0969s`。原因是完整 `demo.json` 包含 `300` 个新闻摘要 object、`15000` 条文本 claim，BasicTruth 要在全量文本 prefix 上构建 TF-IDF implication，并执行 TruthFinder 的 trust-confidence 更新逻辑。它表现为低内存、高时间。

#### SemanticTruth

`SemanticTruth` 使用 SBERT，并且文本模式下 50 个 worker 会预加载 SBERT，以反映真实分布式部署中模型复制带来的内存 footprint。因此：

- fleet peak RSS sum 达到约 `55958MB`。
- total peak RSS 达到约 `57508MB`。
- network time 为 `34.9531s`，明显高于 BasicTruth，因为协议时间包含等待 worker 端完成本地文本编码后回复。

时间上，`SemanticTruth` 的 `elapsed+comm` 为 `94.4821s`，低于 BasicTruth，高于 SenteTruth。原因是它不仅做语义 embedding，还包含双通道/软投票、source reliability、semantic affinity 等逻辑。

#### SenteTruth

`SenteTruth` 同样使用 SBERT，因此内存量级与 SemanticTruth 接近：

- fleet peak RSS sum 约 `55652MB`
- total peak RSS 约 `57216MB`

但它的 `elapsed+comm` 最低，为 `71.4274s`。原因是 SenteTruth 的文本聚合逻辑更直接：

- SBERT 编码文本。
- 计算 cosine similarity。
- 基于节点信誉进行加权选择。

相对 SemanticTruth，它少了更复杂的双通道融合和 affinity 更新逻辑，因此时间更低。

文本场景的主要结论是：

```text
完整 300-object 新闻摘要输入下，SBERT 方法承担模型复制带来的高内存；BasicTruth 不复制 SBERT，内存低但 TF-IDF implication 与 TruthFinder 更新使时间最高。
```

## 9. 为什么文本 SBERT 方法的内存远高于 BasicTruth

在 50-agent distributed 设置下，SemanticTruth 和 SenteTruth 的 worker 进程会各自预加载 SBERT。也就是说，模型不是在一个进程里共享一次，而是被 50 个 worker 进程分别持有。

这导致：

```text
50 workers * SBERT/torch runtime footprint => 约 55.7GB-56.0GB fleet peak RSS
```

BasicTruth 文本路径不预加载 SBERT，所以它的 worker fleet peak RSS 仍接近普通 Python worker 基线，约 `1.01GB`。

这正是分布式部署中模型型方法和非模型型方法在内存可扩展性上的核心差异。

## 10. 为什么数值场景 worker delta 较小

数值场景中 worker 的主要工作是：

- 接收切片。
- 进行轻量本地统计/ACK。
- 参与 iterate 和 reliability update 的通信回合。

算法聚合主要在 coordinator 中完成。完整 300-batch 输入下，worker fleet peak delta 约 `62MB-63MB`，相比 worker fleet peak RSS 约 `1.014GB` 仍然较小。也就是说，数值场景的总体内存仍主要由 50 个 worker 的基础常驻内存和 coordinator 的数据/算法运行时决定。


