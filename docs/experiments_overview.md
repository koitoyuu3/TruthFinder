# TruthFinder 实验汇总（基于 `examples/outputs/` 结果整理）

本文件基于 `decentralized_truth_discovery/examples/` 下的实验脚本及其已有
`outputs/` 目录内的结果文件（CSV / PNG / MD / proof bundle），对当前项目已完成
的全部实验进行系统整理，包括：实验目的、参数设置、输出文件、核心结果与结论。

---

## 0. 被对比方法

| 方法 | 类型 | 定位 |
|---|---|---|
| **SenteTruth** | 文本/数值 | 基于节点间可信度迭代的经典真值发现（SentiTruth 系列） |
| **BasicTruth** | 文本/数值 | 原始 TruthFinder（Yin et al.）实现作为 baseline |
| **SemanticTruth** | 文本+数值（混合） | 本文提出的混合真值发现，SBERT 语义 + 鲁棒 IRLS 数值 |
| **SenFeedTruth** | 数值 | SenFEED (CompositeTD) 增量 MAP 权重更新 |
| **DecentTruth** | 数值 | CBI-TD（INFOCOM 2023）单次前向在线更新 |

文本域通常对比 `{SenteTruth, BasicTruth, SemanticTruth}`；数值域通常对比
`{SenFeedTruth, DecentTruth, SenteTruth, BasicTruth, SemanticTruth}`。

数据集主要有两套：

- `data_agent-50_news-300/` 与 `data_agent-50_weather-300/`：50 agent × 300
  batch，附 `demo_Kbad_collude.json`（文本攻击）与
  `num_demo_Kbad_collude.json`（数值攻击），K ∈ {5,10,15,20,25,30,35,40}。
- `datas_CLLM/`：CLLM 风格的多 LLM 文本问答数据（gpt / google / chatglm /
  llama / tencent），用于 `compare_cllm_text_methods` 与
  `semantictruth_cllm_replay`。

---

## 1. E1：恶意节点比例扫描

### 1.1 `exp1_numeric.py`（10 agent，数值）
- 目的：固定数据集，扫恶意节点 k ∈ {0,1,2,3,4}（0%–40%），输出 **逐 batch
  Relative RMSE**。
- 攻击模式：`collude`（默认）或 `rotate`。
- 输出：`outputs/exp1_numeric/data_agent-50_news-300_collude_relative/`
  - `numeric_per_batch_rmse.csv`（逐 batch 误差）
  - `numeric_per_batch_rmse.png`

### 1.2 `exp1_text.py`（10 agent，文本）
- 目的：同上，但输出 **文本 DataAccuracy**（X=clean 输出，Y=noisy 输出，阈值
  0.85）。
- 输出：`outputs/exp1_text/data_agent-50_news-300/mixed_text.png` 等。

### 1.3 `exp1_numeric_50agent_0_80.py` / `exp1_text_50agent_0_80.py`
- 目的：扩展到 50 agent × 300 batch，扫 0%–80% 恶意比例（k=0,5,…,40）。
- 输出：`outputs/exp1_text/data_agent-50_news-300_0_80_new/` 与
  `data_agent-50_weather-300_0_80_new/`
  - `text_data_accuracy.csv`（含 `_score` 列 = hybrid 分数均值）
  - `mixed_text.png`、`mixed_text_score.png`

#### 关键结果（50 agent news，`text_data_accuracy.csv`）

| 恶意比例 | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|
| 0%  | 1.000 | 1.000 | 1.000 |
| 10% | 1.000 | 0.887 | 1.000 |
| 20% | 1.000 | 0.197 | 1.000 |
| 30% | 1.000 | 0.197 | 1.000 |
| 40% | 0.103 | 0.197 | 1.000 |
| 50% | 0.120 | 0.243 | **0.997** |
| 60% | 0.107 | 0.253 | 0.600 |
| 70% | 0.087 | 0.237 | 0.447 |
| 80% | 0.087 | 0.267 | 0.357 |

- **SemanticTruth 在 ≤50% 恶意下保持 0.99+ DataAccuracy**，即使超过 50% 仍有显著
  鲁棒性；SenteTruth 在 40% 时已崩盘；BasicTruth 20% 就已破坏严重。
- Weather 数据 (`data_agent-50_weather-300_0_80_new/text_data_accuracy.csv`)
  结论一致：SemanticTruth ≤50% 均为 1.0，60% 仍 0.973，80% 仍 0.79。

#### `data_agent-10` 基线结果（`outputs/exp1_numeric/.../numeric_per_batch_rmse.csv`）
数值 collude 场景下，`DecentTruth / SenFeedTruth / SenteTruth` Relative RMSE
在前 49 batch（0 bad）接近 0，进入 50 后阶跃明显；**BasicTruth** 始终处在
~0.0197 的高位；**SemanticTruth** 维持在 ~4e-4 量级，明显优于其他方法。

---

## 2. E3：分阶段恶意源激活（Temporal Attack）

场景：clean → k₁ bad → k₂ bad 阶段性切换。

### 2.1 `exp3_numeric.py`（10 agent，数值）
- 默认 schedule：batch 1–49 clean / 50–99 = 2 bad (20%) / 100–150 = 4 bad (40%)
- 指标：per-batch Relative RMSE vs `num_demo.json`
- 输出：`outputs/exp3_numeric/collude/2bad_at_50__4bad_at_100/`
  - `numeric.png`、`results.csv`

### 2.2 `exp3_text.py`（10 agent，文本）
- 同样的阶段切换，指标：per-batch `mean_hybrid_score` + `data_accuracy`
- 输出：`outputs/exp3_text/{agent,collude}/2bad_at_50__4bad_at_100/`
  - `mixed_text.png`、`results.csv`（含 batch_idx × method × metric）

### 2.3 `exp3_text_50agent.py`（50 agent 扩展）
- schedule：`1-49 clean / 50-99 = 10 bad (20%) / 100-199 = 20 bad (40%) /
  200-300 = 30 bad (60%)`
- 输出：`outputs/exp3_text/data_agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200/`
  - `mixed_text.png`、`mixed_text_accuracy.png`、`results.csv`（900 行）
- 同样存在 Weather 变体，见
  `outputs/exp3_text/data_agent-50_weather-300/results.csv`（900 行）。

### 2.4 `exp3_injection_time.py`
- 变量：注入时间 t ∈ {早，中，晚}（越晚，方法积累的可信度越多，抵御能力越
  强）
- 同时跑数值（Relative RMSE）和文本（DataAccuracy）。
- 目标目录 `outputs/exp3_injection_time/`（脚本入口）。

#### 关键结果（E3 数值，`exp3_numeric/collude/2bad_at_50__4bad_at_100/results.csv`，150 行 × 4 方法）
- **BasicTruth**：phase1 后阶跃至 ~0.0134 Relative RMSE，phase2 再阶跃至
  ~0.0197，始终最高。
- **SenFeedTruth**：phase1 ~0.004，phase2 ~0.008，衰减一半左右。
- **DecentTruth / SenteTruth**：phase1 ~0.002，phase2 也仅 ~0.002，几乎无阶跃。
- **SemanticTruth**：phase1 ~4.5e-4，phase2 ~3.9e-4，**反而随恶意数量上升下
  降**（IRLS 在多数 agent 报告相近值时更容易收敛）。

---

## 3. E4：Agent 数量扫描（固定恶意节点数）

### 3.1 `exp4_numeric.py` / `exp4_text.py` / `exp4_agent_count.py`（10 agent 基础）
- 固定 1 个恶意节点，总 agent 数 N ∈ {3,4,…,10}，实际恶意比例从 33% 降至 10%。
- 数据过滤：仅保留 agent-1..agent-N 的行。
- 预期：N 增大 → 恶意比例下降 → 指标应逐渐改善。
- 输出根目录：
  - `outputs/exp4_numeric/data_agent-50_news-300/bad_5/metric_relative/`
  - `outputs/exp4_text/data_agent-50_news-300/bad_5/`
  - `outputs/exp4_text/data_agent-50_weather-300/bad_5/`
  - `outputs/exp4_agent_count/demo_agent_text_collude/bad_1/metric_relative/`

### 3.2 `exp4_numeric_50agent.py` / `exp4_text_50agent.py`（50 agent 扩展）
- 固定 5 个恶意节点，N ∈ {15,20,25,30,35,40,45,50}。
- 恶意比例从 33% 降到 10%。
- 5 个方法 × 8 个 N = 40 行结果。

#### 关键结果（E4 数值 50 agent，`results.csv`，40 行）
- SenFeedTruth 始终 ~6.7e-3 Relative RMSE，不随 N 改善。
- DecentTruth / SenteTruth 维持 ~2.0e-3 水平。
- SemanticTruth 仍然显著最低（约 3.9e-4）。
- BasicTruth 在 0.013–0.02 之间波动。

#### 关键结果（E4 文本 50 agent，`results.csv`，24 行）
- SenteTruth / SemanticTruth 在 N≥15 时 DataAccuracy 均为 1.000。
- BasicTruth 在 N=15 时只有 0.367（news）/ 0.253（weather），说明其对高恶意比
  例非常脆弱。

### 3.3 `exp4_agent_count.py` 综合文本+数值
- `outputs/exp4_agent_count/demo_agent_text_collude/bad_1/metric_relative/results.csv`
  同时记录 `scene ∈ {numeric, mixed_text}`，包含 `rmse, mae, data_accuracy`。

---

## 4. E5：Agent 可扩展性基准（`agent_scalability_benchmark.py`）

- 目的：在纯数值输入域上统一对比五种方法的 **运行时间 / 内存** 及其随 agent 数
  量增长的 **幂律扩展指数**。
- 输出：`outputs/exp5_agent_scalability_benchmark/`
  - `agent_scalability_benchmark.md`（原始表格）
  - `agent_scalability_benchmark_analysis.md`（完整分析）

#### 核心数据

| method | avg_time(s) | avg_mem_delta(MB) | time_exp α | mem_exp |
|---|---:|---:|---:|---:|
| DecentTruth   | 0.0831 | 246.26 | 0.191 | 0.032 |
| SenFeedTruth  | 0.0877 | 246.05 | 0.185 | 0.031 |
| SenteTruth    | 0.0878 | 246.18 | 0.170 | 0.030 |
| SemanticTruth | 0.1004 | 246.12 | **0.052** | 0.033 |
| BasicTruth    | 0.2345 | 246.00 | **0.408** | 0.032 |

- **绝对最快**：DecentTruth ≈ SenFeedTruth ≈ SenteTruth，BasicTruth 慢 ~2.7
  倍（源于 `calculate_confidence / adjust_confidence` 的 O(N²) + 循环内
  `df.drop_duplicates` 等低效实现）。
- **扩展性最优**：SemanticTruth α≈0.05（近常数），得益于 IRLS 的 MAD 早退出 +
  numpy 向量化。
- **内存差异不显著**：~246 MB 的固定开销（Python + 库）主导。

---

## 5. E6：零知识证明可扩展性（`proof_scalability_benchmark.py`）

- 目的：检查五种方法在生成 / 验证 Schnorr-NIZK 证明 bundle 时的 **proof 尺寸 /
  验证耗时 / 内存**，以及随 batch_count 增长的扩展性。
- 后端：`schnorr_nizk_v1`。
- 输出：`outputs/proof_scalability/`
  - `manifest.json`
  - `raw_results.csv`（每个 `(domain, method, batch_count, repeat)` 一行，共
    48 行 = 5 数值方法 × 150 batch × 3 repeats + 3 文本方法 × 150 batch × 3
    repeats + 其余 batch 档）
  - `summary.csv`（汇总到 `(domain, method, batch_count)`）
  - `bundles_batch_{1,5,10,25,50,100,150}/` 每档的 8 个 proof bundle JSON
  - `outputs/exp6_proof/proof_scalability_summary_panels.{pdf,png}`
- 还有独立的 `outputs/proof_bundles/` 保存一份 8 个 bundle 的静态样本。

#### 核心数据（150 batch 档，`summary.csv`）

| domain | method | row_count | proof_size | verify_mean(s) |
|---|---|---:|---:|---:|
| numeric | SenFeedTruth  | 4500 | 1,786,358 B | 0.0715 |
| numeric | DecentTruth   | 4500 | 2,122,642 B | 0.0784 |
| numeric | SenteTruth    | 4500 | 1,706,306 B | 0.0662 |
| numeric | BasicTruth    | 4500 | 2,063,365 B | 0.0759 |
| numeric | SemanticTruth | 4500 | 2,146,903 B | 0.0818 |
| text    | SenteTruth    | 1500 | 1,102,796 B | 0.0422 |
| text    | BasicTruth    | 1500 | 1,181,295 B | 0.1343 |
| text    | SemanticTruth | 1500 | 1,300,834 B | 0.0604 |

- 所有 bundle `verified_all = True`、`result_digest_ok_all = True`、
  `experiment_digest_ok_all = True`。
- Proof 尺寸与 row_count 大致成线性：数值域 ~1.7–2.1 MB（4500 行），文本域
  ~1.1–1.3 MB（1500 行）。
- 数值域验证 66–82 ms 区间；文本域 42–134 ms（BasicTruth 文本验证高标准差
  σ≈0.126 s 是异常点，但仍通过校验）。

---

## 6. E7：Alpha 敏感度（`exp7_alpha_sensitivity.py`）

- 目的：扫 DataAcc 混合权重 α ∈ {0.0, 0.1, …, 1.0}，量化语义相似度与数值相似度
  在最终评估中的相对贡献。
- 数据：`data_agent-50_news-300`，bad ∈ {0, 5, 10, …, 40}（0%–80%）。
- 方法内部 α 固定为 0.5，只扫评估指标的 α。每个 (method, bad) 只做 1 次 SBERT
  编码（比朴素扫描快 11 倍）。
- 输出：`outputs/exp7_alpha_sensitivity/`
  - `results.csv`（3 方法 × 9 恶意比例 × 11 α = 297 行）
  - `alpha_sensitivity_by_method.png`（3 个子图，每条线一个恶意比例）
  - `alpha_sensitivity_summary.png`（3 方法对所有恶意比例求均值）
  - `README.md`（完整公式与绘图说明）

两个指标定义：

- `SemSim`：SBERT paraphrase-multilingual-MiniLM-L12-v2 的余弦相似度（无 SBERT
  时退化为 char-ngram TF-IDF）。
- `NumSim`：抽取文本数字，逐个检查相对误差 ≤ 0.02 的匹配率。
- `DataAcc(α) = mean( α·SemSim + (1−α)·NumSim ≥ 0.85 )`。

---

## 7. CLLM 数据集实验

### 7.1 `compare_cllm_text_methods.py`
- 对比 `SenteTruth / BasicTruth / SemanticTruth` 在 CLLM 数据集上的三种攻击：
  `random / incorrect / substitution`。
- 输出：`outputs/compare_cllm_text_methods/`
  - `BASE_incorrect_gpt_bad4.csv`：incorrect 攻击，target=gpt，bad=4/10
    - SenteTruth 0.767 / BasicTruth 0.800 / SemanticTruth **0.883**
  - `BASE_random_gpt_bad4.csv`：random 攻击
    - SenteTruth 0.850 / BasicTruth 0.883 / SemanticTruth **0.950**
  - `BASE_substitution_gpt_vs_google_bad4.csv`：substitution 攻击
    （gpt 被替换为 google 答案）
    - SenteTruth **0.833** / BasicTruth 0.767 / SemanticTruth 0.750

> substitution 场景下 SenteTruth 略优于 SemanticTruth；random / incorrect 场
> 景下 SemanticTruth 明显领先。

### 7.2 `semantictruth_cllm_replay.py`
- 目的：以 CLLM 原型的 **sequential oracle replay** 方式运行 SemanticTruth：
  每个 question 独立作为一个 batch，10 个 oracle node，按顺序流式处理，保留跨
  question 的节点可靠度状态。
- 支持注入攻击：`none / random / incorrect / substitution`。
- 输出：`outputs/semantictruth_cllm_replay/`
  - `BASE_incorrect_gpt_bad4.csv`
  - `BASE_none_gpt_bad4.csv`
  - 每行记录一个 question 的 `global_truth` + 各节点的
    `alpha / beta / reliability` 快照（Beta-Bernoulli 后验），用于可视化可靠度
    的动态演化。

---

## 8. 其它

### 8.1 `five_methods_comparison.py`
- 一站式跑五种方法 + 生成 ZK proof bundle 的 pipeline 入口，被
  `proof_scalability_benchmark.py` 复用（`generate_numeric_proof_bundles /
  generate_text_proof_bundles`）。
- 单次输出：`outputs/proof_bundles/*.proof_bundle.json`。
- `outputs/four_methods_benchmark/{agent,collude}/` 是历史入口，当前为空目录（已
  被 `exp3_text/` 下的 agent/collude 结果替代）。

### 8.2 `exp_utils.py`
- 集中提供所有实验共用的工具函数：数据加载（`load_batches`）、转换
  （`to_numeric_df / to_text_df`）、指标（`eval_numeric / eval_text /
  eval_text_metrics_per_batch`）、绘图（`plot_numeric / plot_text /
  plot_temporal_metric / plot_stage_metric_panels`）、方法列表
  （`TEXT_METHODS / NUM_METHODS`）、颜色映射（`COLORS / MARKERS`）等。

---

## 9. 输出目录索引

| 目录 | 对应脚本 | 核心结果文件 |
|---|---|---|
| `outputs/exp1_numeric/data_agent-50_news-300_collude_relative/` | `exp1_numeric.py` | `numeric_per_batch_rmse.{csv,png}` |
| `outputs/exp1_text/data_agent-50_news-300/` | `exp1_text.py` | `mixed_text.png` |
| `outputs/exp1_text/data_agent-50_news-300_0_80_new/` | `exp1_text_50agent_0_80.py` | `text_data_accuracy.csv`, `mixed_text{,_score}.png` |
| `outputs/exp1_text/data_agent-50_weather-300_0_80_new/` | 同上（weather） | 同上 |
| `outputs/exp3_numeric/collude/2bad_at_50__4bad_at_100/` | `exp3_numeric.py` | `results.csv`, `numeric.png` |
| `outputs/exp3_text/{agent,collude}/2bad_at_50__4bad_at_100/` | `exp3_text.py` | `results.csv`, `mixed_text.png` |
| `outputs/exp3_text/data_agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200/` | `exp3_text_50agent.py` | `results.csv`, `mixed_text{,_accuracy}.png` |
| `outputs/exp3_text/data_agent-50_weather-300/` | 同上（weather 变体） | 同上 |
| `outputs/exp4_agent_count/demo_agent_text_collude/bad_1/metric_relative/` | `exp4_agent_count.py` | `results.csv`（含文本+数值） |
| `outputs/exp4_numeric/data_agent-50_news-300/bad_5/metric_relative/` | `exp4_numeric_50agent.py` | `results.csv`, `numeric.png` |
| `outputs/exp4_text/data_agent-50_{news,weather}-300/bad_5/` | `exp4_text_50agent.py` | `results.csv`, `mixed_text.png` |
| `outputs/exp5_agent_scalability_benchmark/` | `agent_scalability_benchmark.py` | `agent_scalability_benchmark{,_analysis}.md` |
| `outputs/exp6_proof/` | `proof_scalability_benchmark.py`（图表） | `proof_scalability_summary_panels.{pdf,png}` |
| `outputs/exp7_alpha_sensitivity/` | `exp7_alpha_sensitivity.py` | `results.csv`, `alpha_sensitivity_{by_method,summary}.png`, `README.md` |
| `outputs/proof_scalability/` | `proof_scalability_benchmark.py` | `manifest.json`, `raw_results.csv`, `summary.csv`, `bundles_batch_{1,5,10,25,50,100,150}/` |
| `outputs/proof_bundles/` | `five_methods_comparison.py` | 8 个 `*.proof_bundle.json` 静态样本 |
| `outputs/compare_cllm_text_methods/` | `compare_cllm_text_methods.py` | `BASE_{attack}_gpt_bad4.csv` |
| `outputs/semantictruth_cllm_replay/` | `semantictruth_cllm_replay.py` | `BASE_{none,incorrect}_gpt_bad4.csv`（含 reliability 快照） |
| `outputs/four_methods_benchmark/{agent,collude}/` | 历史入口，已废弃 | *（空）* |

---

## 10. 一句话总结

- **语义鲁棒**：SemanticTruth 在文本 E1 50% 以下均保持 0.99+ DataAcc，在 80%
  极端恶意下仍能提供 0.79 的精度（其他方法早已降至 ~0.1–0.3）。
- **数值精度**：E3 分阶段攻击下 SemanticTruth Relative RMSE ~4e-4，比
  SenteTruth/DecentTruth 低一个量级，比 BasicTruth 低两个量级。
- **可扩展性**：SemanticTruth 时间指数 α≈0.05（近常数），最适合大规模 agent 场
  景；DecentTruth/SenFeedTruth/SenteTruth 绝对速度最快。
- **ZK proof**：五种方法的证明 bundle 均能 `verified_all=True`，验证耗时 40–130
  ms 范围，尺寸 1.1–2.2 MB（150 batch），schnorr_nizk_v1 后端性能可用。
- **CLLM 场景**：SemanticTruth 在 random/incorrect 攻击下优势明显；
  substitution 攻击下 SenteTruth 更稳。
