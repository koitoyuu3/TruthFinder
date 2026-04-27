# 实验详细说明文档

---

## 0. 共同背景知识（先读这一节）

### 0.1 项目在解决什么问题
**Truth Discovery（真相发现）**：很多 source（这里叫 agent）都在汇报同一件事，但有的 agent 是诚实的、有的是恶意的，每个 agent 报的值还可能略有差异。系统要从一堆"互相冲突的报告"里推断出"真值"，并顺带估出每个 agent 的可靠度。

举个例子：50 个 agent 同时报"BTC 的价格"，48 个报 90490 附近，2 个串谋报 80000。系统要：
1. 把真值定为 90490（不是简单平均，那样会被恶意值拉偏）
2. 给那 2 个串谋 agent 打低分

### 0.2 项目里的五种方法
| 方法 | 类型 | 一句话原理 |
|---|---|---|
| **BasicTruth** | TruthFinder 经典算法 | 信任度 ↔ 事实置信度互相迭代（trust-confidence） |
| **DecentTruth** | 论文新方法（数值） | 去中心化贝叶斯/MAP 聚合，每轮做一次可靠度广播 |
| **SenFeedTruth** | 论文新方法 | 增量 MAP + shock 检测，专门针对**时序攻击** |
| **SenteTruth** | 论文新方法 | 节点信誉 × 内部 phi 值的 vote-style 聚合 |
| **SemanticTruth** | 论文新方法（文本/混合） | SBERT 语义嵌入 + 数值通道双通道软投票 |

### 0.3 数据集
所有 exp1/3/4 共用 `data_agent-10` 这套小数据集（**10 个 agent × 150 个 batch**）：
- `num_demo.json`：纯数值（每条 item 的 response 是 `{BTC, ETH, DOGE}` 三个币种的价格）
- `demo.json`：混合文本（每个 batch 包含"新闻摘要" item 和"价格描述" item，response 是字符串）
- 文件名里 `_Kbad_` 表示有 K 个恶意 agent，`collude` 是协同攻击，`agent` 是单 agent 偏离

注：exp7 用更大的 `data_agent-50_news-300`（50 agent × 300 batch）。

### 0.4 两类攻击模式
- **collude（协同攻击）**：恶意 agent 互相对齐，一起报同一个偏离的"假真值"。这种攻击迷惑性最强，因为多个 agent 报相同的值会让系统误以为这是 majority truth。
- **agent / rotate（动态偏离）**：恶意 agent 按 4-profile 轮换报值，每个时刻偏离方向都不一样。这种攻击容易被识别但难以稳定收敛。

### 0.5 评估指标怎么算
- **数值场景**：Relative RMSE = $\sqrt{\frac{1}{N}\sum (pred - gt)^2 / |gt|^2}$（先按真值绝对值归一化再算 RMSE）。**越小越好**。
- **文本场景**：DataAccuracy = `mean(SemSim(clean_pred, noisy_pred) ≥ 0.85)`，即"在 clean 数据和 noisy 数据上方法输出多大比例是相似的"。**越大越好**。
  - SemSim = SBERT 余弦相似度（语义通道）+ 数字抽取匹配率（数值通道），按 alpha 混合
- **Mean Hybrid Score**（exp3_text 主图）：直接用混合分数的均值，不再做阈值化，能看到细微变化

---

## 1. Exp1-Numeric：恶意节点比例对数值方法的影响

### 1.1 这是什么实验？
**目的**：把恶意 agent 的比例从 0% 一直加到 40%，看 5 种数值真相发现方法的 RMSE 误差怎么变化。

回答的核心问题：**"当一伙恶意串谋者占据越来越多席位时，哪些方法依然能找到真值，哪些会被带偏？"**

### 1.2 实验是怎么做的？
对每个恶意比例 K ∈ {0, 1, 2, 3, 4}（即 0% / 10% / 20% / 30% / 40%）：

1. 加载 `num_demo_Kbad_collude.json`（被注入了 K 个 collude 恶意 agent 的数据）
2. 把数据喂给 5 个方法各跑一遍，每个 batch 输出"global_truth"（每币种的预测真值）
3. 拿这个预测真值和 `num_demo.json`（K=0 的干净版本）里的真实价格做对比
4. 每个 batch 算一个 Relative RMSE，再按 (bad_count, method) 平均

最后画一张分组柱状图：横轴 = 恶意比例，每组 5 根柱（5 个方法），纵轴 = 平均归一化 RMSE。

### 1.3 数据具体是怎么设置的？
**默认运行（10-agent 小数据集）**：
- 文件：`data_agent-10/num_demo*.json`
- batch 数：150
- agent 数：10（全用，不做 subset）
- 每 batch 中 item 数：10（一个 agent 一条），每条 response 含 3 个币种 → 每 batch 30 条 (agent, 币种) 数值
- 总数值条目：150 × 30 = **4500 条**
- 5 个 bad_count × 5 方法 × 150 batch = **3750 行 CSV**

**已有的 50-agent 大规模复现**（`outputs/exp1_numeric/data_agent-50_news-300_collude_relative/`）：
- 文件：`data_agent-50_news-300/num_demo_*bad_collude.json`
- batch 数：300，agent 数：50
- bad_count ∈ {0, 5, 10, 15, 20, 25, 30, 35, 40}（即 0%~80%）
- 5 方法 × 9 bad_count × 300 batch = **13500 行 CSV**

### 1.4 实验结果详细分析
（用 50-agent collude 数据的 mean RMSE，单位是相对误差）

| bad% | BasicTruth | DecentTruth | SenteTruth | SemanticTruth | SenFeedTruth |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000004 |
| 10% | **0.0197** | 0.0016 | 0.0018 | 0.0018 | 0.0020 |
| 20% | 0.0197 | 0.0020 | 0.0020 | 0.0011 | 0.0041 |
| 30% | 0.0197 | 0.0020 | 0.0020 | 0.0018 | 0.0061 |
| 40% | 0.0197 | 0.0020 | 0.0020 | 0.0010 | 0.0081 |
| 50% | 0.0197 | **0.0023** | **0.0200** | 0.0198 | 0.0100 |
| 60% | 0.0197 | 0.0207 | 0.0200 | 0.0198 | 0.0121 |
| 70% | 0.0197 | 0.0207 | 0.0200 | 0.0202 | 0.0141 |
| 80% | 0.0197 | 0.0199 | 0.0200 | 0.0199 | 0.0161 |

**为什么是这样的？**

1. **0% 时所有方法都接近 0**——干净数据下没有冲突，全员一致。SenFeedTruth 出现 4×10⁻⁶ 的微小残差是因为它的增量 MAP 内部有先验，不会精确收敛到点估计。

2. **BasicTruth 一上来就锁定在 0.0197**：经典 TruthFinder 用的是 trust × confidence 的对数迭代，对 collude 攻击没有任何鲁棒性机制——只要有 ≥1 个 collude agent 报相同的假值，方法就会把它当成可信信号，直接选错那条 fact。所以 10%~80% 它都是同一条水平线。

3. **DecentTruth / SenteTruth 在 ≤40% 时近乎完美（~0.002）**：它们都用了"相似度矩阵 + 节点可靠度"的二级过滤——少数派的 collude agent 在 phi（簇内相似度）这一维就被压低权重。但这两种方法用的是 **majority-vote 风格**，一旦恶意比例 ≥ 50%，恶意派变成多数，相似度矩阵反而把 honest 当成离群点，误差立刻跳到 0.02 量级。这就是经典的 **50% 拜占庭阈值**。

4. **SenFeedTruth 是单调线性增长**（0.0020 → 0.0161）：它的增量 MAP 每个 batch 都做一点权重更新，恶意越多每步偏移越多——这是个"被慢慢腐蚀"的曲线，而不是断崖式。它在 50%~60% 反而是**最好**的方法（0.010~0.012），因为先验抑制了多数派 takeover。

5. **SemanticTruth 行为像 DecentTruth**：50% 是清晰的转折点。原因是它的数值通道也是基于 majority similarity，跟 DecentTruth 同结构。

**核心结论**：
- 30% 以下选 DecentTruth / SenteTruth / SemanticTruth（误差最低）
- 50% 上下选 SenFeedTruth（唯一在多数派攻击下还能用的方法）
- BasicTruth 在 collude 攻击下完全失效，**不要用**

---

## 2. Exp1-Text：恶意节点比例对文本/混合方法的影响

### 2.1 这是什么实验？
跟 exp1_numeric 一模一样的设置，只是把数据换成了**文本/混合 demo**（response 是含中文新闻摘要 + 数字的字符串），评估 3 个文本方法（SenteTruth / BasicTruth / SemanticTruth）的 DataAccuracy。

### 2.2 实验是怎么做的？
1. 加载 clean 数据 → 跑方法 → 得到 `X = {(batch, object): clean_pred}`
2. 加载 K-bad noisy 数据 → 跑方法 → 得到 `Y = {(batch, object): noisy_pred}`
3. 算 `DataAcc = mean(HybridScore(X, Y) >= 0.85)`：方法在被污染数据上的输出与干净基线的吻合度

注意：这里 ground truth 的 proxy 是"方法自己在干净数据上的输出"，不是绝对真值。这样对齐论文 Section VI 的评估范式。

### 2.3 数据具体是怎么设置的？
默认走 `data_agent-10`，bad_counts = [0,1,2,3,4]，但已有的输出主要是 4 个不同设置的 50-agent 大数据复现：

| 子实验 | 数据集 | 攻击 | 阈值 | 恶意比例点 |
|---|---|---|---|---|
| 1.1 | data_agent-50_**news**-300 | collude | 0.85 | 0~80%（9 点） |
| 1.2 | data_agent-50_news-300 | numonly | 0.9 | 0~50%（6 点） |
| 1.3 | data_agent-50_**weather**-300 | collude | 0.85 | 0~80%（9 点） |
| 1.4 | data_agent-50_weather-300 | numonly | 0.9 | 0~80%（9 点）+ 分数 |

每个 batch 含 ~20 个 item（agent_count × 2 个 object：新闻摘要 + 价格描述）。

### 2.4 实验结果详细分析

**1.1 News + collude（最经典设置）**：

| bad% | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|
| 0% | 1.00 | 1.00 | 1.00 |
| 10% | 1.00 | 0.63 | 1.00 |
| 20% | 1.00 | 0.38 | 1.00 |
| 30% | 1.00 | 0.33 | 1.00 |
| 40% | 1.00 | 0.33 | 1.00 |
| 50% | 1.00 | 0.35 | 1.00 |
| 60% | **0.35** | 0.34 | 0.98 |
| 70% | 0.32 | 0.33 | **0.82** |
| 80% | 0.33 | 0.34 | 0.73 |

**为什么是这样？**

1. **BasicTruth 在 10% 就崩了**（0.63→0.33），跟数值实验同一个原因：trust-confidence 对 collude 没有防御。30% 以后稳定在 0.33 = "在 1/3 个 batch 上仍然碰对 majority"，剩下都被错的 collude 字符串带走。

2. **SenteTruth 撑到 50% 才崩**：节点信誉 × phi 的双重加权能挡住 ≤50% 的协同攻击，但一过 50% majority 就翻转。崩塌后稳定在 0.32~0.35（同 BasicTruth）。

3. **SemanticTruth 撑到 60% 都是 0.98**，70% 才开始下降：因为它的语义通道（SBERT 嵌入）会比较两条字符串在意义层面的相似度，即使恶意 agent 报的字符串不同但意思一致才能"协同"，这增加了攻击的语义成本。**这是论文最想强调的卖点：语义鲁棒性**。

**1.2 News + numonly（仅扰动数字）**：

| bad% | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|
| 0% | 1.00 | 1.00 | 1.00 |
| 10% | 0.92 | 0.50 | 0.99 |
| 30% | 0.92 | 0.28 | 0.99 |
| 40% | **0.08** | 0.28 | 0.93 |
| 50% | 0.07 | 0.31 | 0.83 |

**为什么 SenteTruth 在 40% 突然崩？** numonly 攻击只改数字、不改文字，所以 SenteTruth 的语义通道（句子结构）依然识别为高度相似——它会被骗着选择恶意 agent 的字符串。SemanticTruth 因为有数值通道做交叉验证，所以撑得更久。

**1.3 vs 1.4（news vs weather）**：
- weather 数据集恶意 agent 撑到 70% SenteTruth 才完全崩（0.61），比 news 健壮——可能因为 weather 文本用词更受限、相似度更容易区分
- BasicTruth 在 weather + collude 上同样在 30% 就崩

**核心结论**：
- 安全阈值：BasicTruth ≤0%、SenteTruth ≤50%（collude）/ ≤30%（numonly）、SemanticTruth ≤60%（collude）/ ≤40%（numonly）
- SemanticTruth 是**唯一在 numonly 攻击下还能保住 ≥0.9 准确率到 30% 的方法**

---

## 3. Exp3-Numeric：分阶段恶意源激活（时序攻击）

### 3.1 这是什么实验？
模拟"恶意 agent 不是一开始就坏，而是中途叛变"的场景，看哪些方法能**适应这种突变**。

时间线（默认 150 batch）：
- batch 1–49：全员诚实
- batch 50–99：突然 2 个 agent 变恶意
- batch 100–150：再追加 2 个变恶意（共 4 个）

回答的核心问题：**"方法在攻击突然出现时能否快速响应？长期会不会被慢慢腐蚀？"**

### 3.2 实验是怎么做的？
1. 加载 3 份数据：clean / 2bad / 4bad（同一时间序列、不同恶意配置）
2. 用 `build_phase_switched_df` 按时间点拼接：前 49 batch 取 clean、50-99 取 2bad、100-150 取 4bad
3. 拿 ground truth = clean 数据的真值
4. 跑 5 个数值方法，记录每个 batch 的 Relative RMSE
5. 画时序折线图，在 batch 50/100 处画两条竖线标"激活事件"

### 3.3 数据具体是怎么设置的？
默认走 `data_agent-10`：
- 150 batch，10 agent
- phase1_start=50, phase2_start=100, phase1_bad_count=2, phase2_bad_count=4
- 攻击模式：默认 collude

已有的 50-agent 输出（`outputs/exp3_numeric/data_agent-50_news-300/5bad_at_50__10bad_at_100__15bad_at_200/`）：
- 300 batch，50 agent
- 三阶段：clean(1-49) → 5bad(50-99) → 10bad(100-199) → 15bad(200-300)（即 0% → 10% → 20% → 30%）

### 3.4 实验结果详细分析

**50-agent 三阶段数据，各方法各阶段平均 Relative RMSE**：

| 方法 | P0 clean | P1 5bad(10%) | P2 10bad(20%) | P3 15bad(30%) |
|---|---:|---:|---:|---:|
| BasicTruth | 0.0000 | **0.0197** | 0.0197 | 0.0197 |
| DecentTruth | 0.0000 | 0.0016 | 0.0016 | 0.0016 |
| SenteTruth | 0.0000 | 0.0018 | 0.0020 | 0.0020 |
| SemanticTruth | 0.0000 | 0.0001 | 0.0002 | 0.0003 |
| SenFeedTruth | 0.0000 | 0.0020 | **0.0041** | **0.0061** |

**为什么是这样？**

1. **P0 全员诚实**：所有方法 RMSE ≈ 0，因为没有冲突。

2. **P1 突变那一刻（batch 50）**：
   - **BasicTruth 一步到顶**（0.0197），跟 exp1 的 collude 失败模式一样——只要有 collude agent 上线就立刻被错位
   - DecentTruth/SenteTruth/SemanticTruth 几乎纹丝不动（≤0.002），说明它们的 reliability 估计在线更新时能立刻把新出现的离群 agent 排除
   - SenFeedTruth 也只有 0.002，但是**注意它的 P2/P3 在涨**

3. **P2 → P3 持续偏移**：
   - DecentTruth/SenteTruth/SemanticTruth 几乎不变（数值在 10⁻³ 量级波动），说明它们的可靠度估计**记忆很短**——每个 batch 重新建相似度矩阵，过去阶段的污染不会累计
   - **SenFeedTruth 单调上升**（0.002 → 0.004 → 0.006）：增量 MAP 把过去的恶意观测都纳入了后验更新，所以越往后误差越大。这是它在静态多数派攻击下表现好（exp1 50%+）但在**渐进腐蚀**下不占优的原因
   - SemanticTruth 是**最稳的方法**（P3 仅 0.0003），因为它做了 batch 内独立聚合，不依赖跨 batch 的可靠度记忆

**核心结论**：
- 突变响应速度：DecentTruth ≈ SenteTruth ≈ SemanticTruth > SenFeedTruth >> BasicTruth
- 长期稳定性：SemanticTruth > DecentTruth/SenteTruth > SenFeedTruth >> BasicTruth
- SenFeedTruth 反而**不适合**渐进式攻击，它的优势在静态高比例攻击

---

## 4. Exp3-Text：分阶段恶意源激活（文本版）

### 4.1 这是什么实验？
exp3_numeric 的文本版。同样的"clean → 2bad → 4bad"分阶段攻击，但数据是文本/混合，评估 3 个文本方法的 per-batch DataAccuracy 和 Mean Hybrid Score。

### 4.2 实验是怎么做的？
1. 同 exp3_numeric 的拼接思路，分阶段构造 `phased_txt_df`
2. ground truth proxy = 方法自己在 clean 数据上的输出
3. 评估每个 batch 的 `DataAccuracy(X_b, Y_b)` 和 `Mean Hybrid Score(X_b, Y_b)`
4. 主图画 hybrid score（连续值，看细微变化），附图画 data accuracy（阈值化，看断点）

### 4.3 数据具体是怎么设置的？
默认 attack_profile=**agent**（不是 collude），phase1=50, phase2=100, bad=2/4。

实际跑的两个 50-agent 设置：
- `data_agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200`：news + collude，三阶段 10/20/30 bad（即 20%/40%/60%），300 batch
- `data_agent-50_weather-300/`：weather + collude，10/20/30 bad

### 4.4 实验结果详细分析

**News collude，DataAcc / HybridScore 按阶段聚合**：

| 方法 | P0 clean | P1 10bad(20%) | P2 20bad(40%) | P3 30bad(60%) |
|---|---|---|---|---|
| BasicTruth | 1.00 / 1.00 | **0.26** / 0.69 | 0.29 / 0.74 | 0.49 / 0.82 |
| SenteTruth | 1.00 / 1.00 | 1.00 / 1.00 | 0.84 / 0.93 | **0.49** / 0.79 |
| SemanticTruth | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 | **1.00** / 0.99 |

（格式：DataAcc / HybridScore）

**Weather collude**：

| 方法 | P0 clean | P1 10bad(20%) | P2 20bad(40%) | P3 30bad(60%) |
|---|---|---|---|---|
| BasicTruth | 1.00 / 1.00 | 0.62 / 0.87 | 0.16 / 0.72 | 0.32 / 0.75 |
| SenteTruth | 1.00 / 1.00 | 0.98 / 0.98 | 1.00 / 1.00 | **0.50** / 0.81 |
| SemanticTruth | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **0.95** / 0.98 |

**为什么是这样？**

1. **BasicTruth 在 P1 立刻断崖式跌到 0.26**：跟 exp1_text 一致，collude 攻击对 trust-confidence 是"开门即破"。HybridScore 还有 0.69 是因为有些 batch 的方法输出和 clean 输出在数字上还有部分重合（hybrid 包含数值通道）。

2. **SenteTruth 撑到 P2（40%）还能 0.84**：节点信誉 + phi 的双滤波在 ≤40% 仍能识别少数恶意。但 P3 60% 跨过 50% 阈值后立刻崩到 0.49——典型多数派 takeover。

3. **SemanticTruth 是 P3 60% 下唯一 ≥0.95 的方法**：
   - news 数据下保持 1.00（在 batch 级 DataAcc 上完美，hybrid 也只是从 1.00 微降到 0.99）
   - weather 数据下从 1.00 降到 0.95
   - 原因还是语义通道：collude 攻击需要恶意 agent 报"语义对齐"的字符串才能形成虚假 majority，但 SBERT 能识别这种 paraphrase 攻击

4. **HybridScore 比 DataAcc 信息量更大**：
   - 看 SenteTruth news P3：DataAcc 0.49 看起来像随机，但 HybridScore 0.79 说明它输出的字符串其实跟 clean 还是有相当相似度，只是没过 0.85 阈值
   - 这就是为什么 exp3_text 把 hybrid_score 设为主图

**核心结论**：
- 分阶段攻击下文本方法的鲁棒性顺序：SemanticTruth >> SenteTruth >> BasicTruth
- SemanticTruth 在 60% 恶意下依然保持 DataAcc=1.0（news），这是论文的核心结果之一
- HybridScore 比 DataAcc 更适合做诊断指标（避免阈值二值化造成的信息损失）

---

## 5. Exp4-Numeric：随 Agent 数量变化的鲁棒性

### 5.1 这是什么实验？
**固定 1 个恶意 agent**（agent-1），但**总 agent 数 N** 从 3 一直加到 10。这样恶意比例从 1/3 ≈ 33% 一路降到 1/10 = 10%。

回答的核心问题：**"诚实人变多时，方法的精度提升曲线是怎样的？是否存在某个 N 之后边际收益就消失？"**

### 5.2 实验是怎么做的？
1. 加载 `num_demo_1bad_agent.json`（1 个 agent 恶意，模式是 agent-style 而不是 collude）
2. 对每个 N ∈ {3, 4, …, 10}，只取 agent-1..agent-N 的子集
3. 跑 5 个方法，按 batch 算 RMSE/MAE，**只统计最后 10% 的 batch**（让方法收敛后再评估）
4. 画 RMSE 和 MAE 随 N 的曲线

### 5.3 数据具体是怎么设置的？
默认走 `data_agent-10`：
- 文件：`num_demo_1bad_agent.json`（注意是 agent 模式不是 collude）
- 150 batch，agent-1 恒定恶意
- N 扫描 {3, 4, 5, 6, 7, 8, 9, 10}，对应恶意比例 {33%, 25%, 20%, 17%, 14%, 12.5%, 11%, 10%}
- 评估窗口：last 10% batches = batch 136-150

已有的 50-agent 输出：
- 文件：`data_agent-50_news-300/num_demo_5bad_collude.json`（5 个恶意 collude）
- N 扫描 {15, 20, 25, …, 50}（恶意比例 33% → 10%）

### 5.4 实验结果详细分析

**50-agent 数据 5bad collude，最后 10% 的 mean RMSE**：

| N agents | bad% | SenFeedTruth | DecentTruth | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 33% | 0.0067 | 0.0020 | 0.0020 | 0.0197 | 0.0011 |
| 20 | 25% | 0.0050 | 0.0020 | 0.0020 | 0.0197 | 0.0019 |
| 25 | 20% | 0.0041 | 0.0020 | 0.0020 | 0.0197 | 0.0013 |
| 30 | 17% | 0.0033 | 0.0022 | 0.0020 | 0.0197 | 0.0019 |
| 35 | 14% | 0.0029 | 0.0020 | 0.0020 | 0.0197 | 0.0011 |
| 40 | 12.5% | 0.0025 | 0.0020 | 0.0018 | 0.0197 | 0.0019 |
| 45 | 11% | 0.0022 | 0.0020 | 0.0020 | 0.0197 | 0.0011 |
| 50 | 10% | 0.0020 | 0.0016 | 0.0018 | 0.0197 | 0.0019 |

**为什么是这样？**

1. **BasicTruth 完全水平在 0.0197**：跟前几个实验一致，trust-confidence 对 collude 不分 N 都失效。N=50 也没用——多 1 个 agent 不能解开 collude 锁定。

2. **SenFeedTruth 单调下降**（0.0067 → 0.0020）：诚实 agent 越多，先验拉得越准。这是**唯一对 N 敏感的方法**——其他方法靠相似度过滤，1 个恶意被识别后多少 agent 都不影响结果。

3. **DecentTruth / SenteTruth 几乎扁平**（恒在 0.002 左右）：它们用相似度矩阵 + reliability，1 个恶意 agent 无论 N 多大都会被压到 phi=0 附近，所以 RMSE 由"剩下 N-1 个诚实 agent 的随机噪声"决定，这部分跟 N 关系不大。

4. **SemanticTruth 在 0.001-0.002 之间随机抖**：最低（最好），但没有单调趋势——它的双通道软投票对 collude 有较强抑制，但具体值取决于 batch 内随机种子。

**核心结论**：
- N 增大时**只有 SenFeedTruth 真正受益**（曲线单调下降）
- DecentTruth / SenteTruth / SemanticTruth 在 N≥10 后基本饱和
- BasicTruth 完全不响应 N，再多 agent 也救不了

---

## 6. Exp4-Text：随 Agent 数量变化（文本版）

### 6.1 这是什么实验？
exp4_numeric 的文本版。同样固定 1 个恶意 agent、扫 N，但数据是文本，评估 3 个文本方法的 DataAccuracy。

### 6.2 实验是怎么做的？
1. 加载 `demo.json`（clean）和 `demo_1bad_collude.json`（1 个 collude 恶意）
2. 对每个 N，取 agent-1..agent-N 的子集
3. clean 方法输出 = X，noisy 方法输出 = Y
4. 算 DataAccuracy(X, Y)
5. 画曲线

### 6.3 数据具体是怎么设置的？
- 默认 `data_agent-10/demo*_collude.json`
- N ∈ {3,4,...,10}，恶意比例 {33%~10%}
- 默认阈值 0.85，可选 0.90（已跑两套对比）

50-agent 输出：
- `data_agent-50_news-300` / `data_agent-50_weather-300`，5bad collude
- N ∈ {15, 20, …, 50}

### 6.4 实验结果详细分析

**News 数据，5bad collude，DataAcc（threshold=0.85）**：

| N | bad% | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|---:|
| 15 | 33% | 1.00 | 0.37 | 1.00 |
| 25 | 20% | 1.00 | 0.75 | 1.00 |
| 35 | 14% | 1.00 | 0.91 | 1.00 |
| 50 | 10% | 1.00 | 0.53 | 1.00 |

**News 数据，threshold=0.90 时**（更严格）：

| N | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|
| 15 | 0.75 | 0.27 | 0.86 |
| 35 | 0.87 | 0.41 | 0.84 |
| 50 | 0.84 | 0.50 | 0.83 |

**Weather 数据，threshold=0.85**：

| N | SenteTruth | BasicTruth | SemanticTruth |
|---:|---:|---:|---:|
| 15 | 1.00 | 0.25 | 1.00 |
| 30 | 1.00 | 0.997 | 1.00 |
| 50 | 1.00 | 0.93 | 1.00 |

**为什么是这样？**

1. **SenteTruth / SemanticTruth 在 0.85 阈值下永远是 1.00**：5/N 始终 ≤ 33% < 50% 阈值，方法的鲁棒性给了完美 score。曲线没法看出方法间差异，**说明阈值 0.85 太松了**。

2. **BasicTruth 显示出不规则曲线**（news）：N=15 → 0.37，N=35 → 0.91，N=50 → 0.53。这种"先上升后下降"的非单调说明 BasicTruth 不仅对恶意 agent 敏感，对诚实 agent 总数也很敏感——更多的诚实噪声反而会干扰它的 trust 估计收敛。

3. **threshold=0.90 时方法间差距才显现**：SemanticTruth 和 SenteTruth 才分得出高下（SemanticTruth 在小 N 略高 0.86 vs 0.75）。**所以 0.90 才是更好的诊断阈值**。

4. **Weather vs News**：BasicTruth 在 weather 上 N=30 就接近 1.00（0.997），news 上 N=35 才到 0.91。weather 文本的 collude 攻击更容易被 trust-confidence 识别（可能是攻击的字符串多样性更低，更容易判断为离群）。

**核心结论**：
- 阈值 0.85 区分度不够，建议改用 0.90
- BasicTruth 的鲁棒性强烈依赖 N 和数据集，不可预测
- SemanticTruth / SenteTruth 在 1bad 攻击下从 N=15 起就饱和到 1.00

---

## 7. Exp7-Alpha Sensitivity：混合权重 α 的敏感度

### 7.1 这是什么实验？
DataAccuracy 内部用 `α·SemSim + (1-α)·NumSim` 混合两个通道。**α 这个超参怎么选？**

α=0：纯数值匹配；α=1：纯语义匹配；α=0.5：当前默认。
本实验扫描 α∈{0.0, 0.1, …, 1.0}（11 个点），看每个方法在不同恶意比例下哪个 α 最优。

注意：**不是改方法内部 α，只改评估指标 α**。模型内部固定 α=0.5。

### 7.2 实验是怎么做的？
关键优化：**预计算 SemSim 和 NumSim 各一次，扫 α 时零成本加权求和**（不重复跑 SBERT 编码）。

```
对每个 bad_count ∈ {0, 5, 10, ..., 40}:
  对每个 method:
    跑一次方法在 clean 数据 → x_preds
    跑一次方法在 noisy 数据 → y_preds
    一次 SBERT 编码 x_txts、y_txts → sem_scores（向量）
    一次正则抽数字 + 容差匹配 → num_scores
    对每个 α: combined = α·sem + (1-α)·num，DataAcc = mean(combined ≥ 0.85)
```

### 7.3 数据具体是怎么设置的？
- 数据集：`data_agent-50_news-300`（50 agent × 300 batch）
- bad_count: 0, 5, 10, 15, 20, 25, 30, 35, 40（即 0%, 10%, 20%, …, 80%）
- methods: SenteTruth / BasicTruth / SemanticTruth
- α: 11 个点（0.0~1.0 步长 0.1）
- 评估阈值: 0.85
- 总实验单元：3 method × 9 bad × 11 α = **297 行 CSV**
- 计算成本：3 × 9 = **27 次 method run + 27 次 SBERT 编码**（α 扫描零成本）

### 7.4 实验结果详细分析

**各方法跨 bad_count 的 mean DataAcc 随 α 变化**：

| α | BasicTruth | SemanticTruth | SenteTruth |
|---:|---:|---:|---:|
| 0.0（纯数值） | 0.43 | 0.81 | 0.61 |
| 0.5（默认） | 0.45 | **0.82** | 0.62 |
| 0.6 | 0.46 | **0.83** | 0.63 |
| 0.8 | 0.56 | 0.82 | 0.67 |
| 1.0（纯语义） | **0.81** | 0.81 | **0.78** |

**最优 α**：
- BasicTruth：α=1.0（纯语义最优）
- SemanticTruth：α=0.6（略偏语义最优）
- SenteTruth：α=1.0（纯语义最优）

**逐 bad_count 看更清楚**（DataAcc at α=0.0 / 0.5 / 1.0）：

**SemanticTruth**：
- bad=10~30：α=0/0.5 都 0.97，α=1 反而降到 0.69~0.76
- bad=35~40：α=0/0.5 跌到 0.32，α=1 反而 0.75
- → SemanticTruth 在低攻击靠数字通道精准识别（α 偏低好），高攻击时数字被严重污染、必须靠语义（α 偏高好）

**SenteTruth**：
- bad=10~15：α=0/0.5 都 0.98，α=1 是 0.76
- bad=20~40：α=0/0.5 暴跌到 0.30，α=1 才能维持 0.71~0.82
- → 攻击≥40% 时 SenteTruth 输出的数字通道几乎完全错乱，只能依赖语义

**BasicTruth**：
- bad=10：α=0/0.5 是 0.70，α=1 是 0.63（数值略胜）
- bad=20+：所有 α 全崩，但 α=1 维持 0.78~0.85（语义最高）
- → BasicTruth 自己就崩了，但语义通道至少能识别"输出字符串还在新闻范畴里"

**为什么是这样？**

1. **α 高（偏语义）的方法整体更鲁棒**：恶意 agent 数字编造容易，语义伪造（生成像样的新闻摘要）难。语义通道的攻击成本高得多。

2. **SemanticTruth 在 α=0.5/0.6 局部最优**：因为它本身已经是双通道方法，输出已经被语义对齐过，再用纯语义评估反而会"双重 smoothing"导致区分度下降。

3. **当攻击比例 < 50% 时数值通道仍然有用**（SemanticTruth/SenteTruth 在 bad=10/20 时 α=0 就有 0.97）：少数派恶意数字被 majority 数值压住，数字匹配率仍然反映真实情况。

4. **当攻击比例 > 50% 时数值通道完全失效**（SenteTruth bad=20 时 α=0 直接跌到 0.30，这是 50agent×0.4=20bad 的 majority takeover），此时 α=1 是唯一有用的信号。

**核心结论**：
- 默认 α=0.5 对所有方法都不是最优，但是是**最稳健的折中**：低攻击时跟 α=0 接近，高攻击时退化没 α=1 那么严重
- 如果**先验已知攻击比例低**（≤30%），用 α=0.4 更准
- 如果**面对高攻击或不可知场景**，用 α=0.7 更稳
- SemanticTruth 的 α 敏感度最低（曲线最平），印证了它的"双通道软投票"设计鲁棒

---

## 8. 七个实验串起来看

| 实验 | 自变量 | 固定 | 结论速记 |
|---|---|---|---|
| **Exp1-Num** | 恶意比例 0~40%（80%） | 全 batch、全 agent | 30% 以下选 DecentTruth；50%+ 选 SenFeedTruth；BasicTruth 永远不行 |
| **Exp1-Text** | 恶意比例 0~80% | 全 batch、全 agent | 60%+ 只剩 SemanticTruth；阈值 0.85 太松 |
| **Exp3-Num** | 时间（分阶段攻击激活） | 攻击 schedule 0/2/4 bad | 突变响应：DecentTruth/SenteTruth/SemanticTruth 立刻；SenFeedTruth 慢慢漂移 |
| **Exp3-Text** | 时间（分阶段攻击激活） | 同上 | 60% 阶段 SemanticTruth 唯一保 1.0 |
| **Exp4-Num** | Agent 总数 N，固定 1 bad | 比例随 N 33%→10% | 只有 SenFeedTruth 单调随 N 改善；其他方法饱和 |
| **Exp4-Text** | Agent 总数 N，固定 1 bad | 同上 | 0.85 阈值下方法都饱和到 1.0；建议升到 0.90 |
| **Exp7** | α 评估权重 | 50agent, 300batch, 9 bad 比例 | α=0.5 是稳健折中；高攻击场景应升 α 到 0.7+ |

### 跨实验关键发现

1. **BasicTruth 是反例基线**：所有 collude 实验中它都最先崩溃，论文用它来证明"经典 trust-confidence 在串谋下不够用"。
2. **SemanticTruth 是文本场景全局赢家**：exp1/3/7 文本部分它都是最稳的，因为有语义 + 数值双通道。
3. **SenFeedTruth 是数值场景的"50% 拜占庭破壁者"**：exp1 bad>50% 时它是唯一可用的，但 exp3/4 显示它对时序攻击和小 N 不友好。
4. **DecentTruth / SenteTruth 是数值场景的"低攻击最优解"**：在 ≤40% 区间精度极高、对 N 无要求，只在 ≥50% 突然失效。

### 用户怎么挑方法
- **数据是数值，攻击 < 30%**：DecentTruth（最简单且精度最高）
- **数据是数值，攻击 50%+**：SenFeedTruth（唯一选择）
- **数据是文本/混合**：SemanticTruth（无脑选）
- **跑 baseline / 论文复现**：加上 BasicTruth 做反例
