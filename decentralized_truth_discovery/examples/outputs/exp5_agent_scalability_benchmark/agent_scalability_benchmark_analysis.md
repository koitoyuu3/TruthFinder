# 实验五：Agent 可扩展性基准测试分析

## 一、实验概述

本实验评估五种真值发现方法在不同 agent 数量下的**时间性能**和**内存消耗**的可扩展性。实验在纯数值输入域上进行，使五种方法共享同一批 workload，确保对比的公平性。

实验通过在多个 workload 上、不同 agent 子集规模下运行各方法，收集运行时间和峰值内存增量，并拟合 **幂律扩展指数** 来量化各方法随 agent 规模增长的扩展行为。

## 二、实验结果

| method | avg_time(s) | avg_mem_delta(MB) | time_exp | mem_exp |
|---|---:|---:|---:|---:|
| DecentTruth | 0.0831 | 246.2603 | 0.1906 | 0.0323 |
| SenFeedTruth | 0.0877 | 246.0450 | 0.1850 | 0.0312 |
| SemanticTruth | 0.1004 | 246.1154 | 0.0519 | 0.0325 |
| SenteTruth | 0.0878 | 246.1788 | 0.1703 | 0.0302 |
| BasicTruth | 0.2345 | 246.0010 | 0.4080 | 0.0324 |

## 三、指标含义详解

### 3.1 `avg_time(s)` — 平均运行时间（秒）

在所有 workload × agent 子集组合上，每种方法单次运行的**平均耗时**。使用 `time.perf_counter()` 精确计时，覆盖从模型初始化到所有 batch 处理完毕的完整流程。

### 3.2 `avg_mem_delta(MB)` — 平均峰值内存增量（MB）

每次测试在独立子进程中运行。后台采样线程以 20ms 间隔采集进程的 RSS（Resident Set Size），计算：

$$\text{mem\_delta} = \text{peak\_rss} - \text{baseline\_rss}$$

即运行期间的**内存净增长量**。

### 3.3 `time_exp` — 时间扩展指数

对 $(N, T)$ 数据点做**对数-对数线性回归**：

$$\log(T) = \alpha \cdot \log(N) + \beta$$

其中 $N$ 为 agent 数量，$T$ 为运行时间，$\alpha$ 即 `time_exp`。它刻画了时间复杂度的幂律关系 $T \propto N^{\alpha}$：

| 指数范围 | 含义 | 扩展性评价 |
|---|---|---|
| $\alpha \approx 0$ | 时间几乎不随 $N$ 增长 | 常数级，极优 |
| $0 < \alpha < 1$ | 亚线性增长 | 良好 |
| $\alpha \approx 1$ | 线性增长 $T \propto N$ | 可接受 |
| $\alpha > 1$ | 超线性增长 | 较差 |

### 3.4 `mem_exp` — 内存扩展指数

同 `time_exp` 的拟合方法，但针对 `peak_delta_rss_mb`：

$$\log(M) = \alpha \cdot \log(N) + \beta$$

$\alpha$ 值刻画内存消耗随 agent 数量增长的幂律关系。

## 四、结果分析与方法层面的原因解释

### 4.1 绝对运行时间分析

**排名：DecentTruth (0.083s) ≈ SenFeedTruth (0.088s) ≈ SenteTruth (0.088s) < SemanticTruth (0.100s) << BasicTruth (0.235s)**

#### DecentTruth — 最快 (0.0831s)

DecentTruth 实现了 CBI-TD 算法（IEEE INFOCOM 2023），其核心设计是**单次前向在线更新**：

- **真值估计 (Eq.3)**：对每个 object，计算所有 source 的可靠度加权平均，复杂度 $O(N)$
- **可靠度更新 (Eq.4-6)**：每个 source 独立计算 batch 内 MSE → erfc 一致性 → 可靠度，复杂度 $O(N \cdot M)$（$M$ 为 object 数量）
- **无 batch 内迭代**：不做固定点收敛循环，每个 batch 仅执行一次前向更新

**总复杂度**：每个 batch $O(N \cdot M)$，**对 agent 数量线性**。操作简单（加权平均、减法平方、erfc），无矩阵运算，无排序。

#### SenFeedTruth — 接近最快 (0.0877s)

SenFeedTruth 实现了 SenFEED 论文的 CompositeTD 算法，常规路径也是线性的：

- **真值估计 (Eq.2/7)**：加权平均 + 历史真值融合，复杂度 $O(N)$ per object
- **权重更新 (Eq.6)**：增量 MAP 更新，使用累计误差统计量，复杂度 $O(N \cdot M)$
- **Shock 检测**：比较当前与历史权重的变化量，复杂度 $O(N)$
- **常规路径**（无 shock 触发）：单步增量更新，$O(N \cdot M)$

**总复杂度**：常规为 $O(N \cdot M)$，仅在权重突变时回退到 batch 内迭代（$O(I \cdot N \cdot M)$，$I$ 为迭代轮数）。略慢于 DecentTruth 的原因是额外维护了历史累计误差统计量和 shock 检测逻辑。

#### SenteTruth — 接近最快 (0.0878s)

SenteTruth 的数值适配版本 `_NumericSenteTruthFinder` 的核心操作：

- **相似度矩阵**：对每个 object 计算 $N \times N$ 的数值相似度矩阵（基于相对距离），复杂度 $O(N^2)$
- **可信度分数 $\phi(v_i)$ (Eq.2)**：每个节点与其他节点的平均相似度，$O(N)$ per node，总计 $O(N^2)$
- **真值聚合 (Eq.3)**：$\arg\max C_i \cdot \phi(v_i)$，$O(N)$
- **可信度更新 (Eq.4)**：归一化更新每个节点的可信度，$O(N)$

**总复杂度**：$O(M \cdot N^2)$。在当前实验规模下（$N \leq 10$），$N^2 = 100$ 很小，且每个操作是简单的浮点运算（无排序、无矩阵分解），因此绝对耗时与线性方法相当。

#### SemanticTruth — 略慢 (0.1004s)

SemanticTruth 使用鲁棒 IRLS（迭代重加权最小二乘）管道：

- **每个 object 的处理**：
  1. **可靠度门控预过滤**：$O(N)$，剔除 reliability 明显低于均值的 source
  2. **加权中位数**：$O(N \log N)$（需排序）
  3. **IRLS 循环**：每轮计算 MAD → Tukey 双权 → 加权平均，$O(N)$ per iteration
  4. **Consistency 评估**：高斯衰减打分，$O(N)$
- **可靠度更新**：Beta-Bernoulli 贝叶斯更新，$O(N)$

**总复杂度**：$O(M \cdot I_{\text{IRLS}} \cdot N \log N)$。虽然理论上涉及排序和多轮迭代，但在实验中：
- 当多数 agent 报告相同值时，MAD = 0，IRLS 直接返回中位数，循环提前退出
- 本实验用纯数值模式（`use_sbert=False`），文本管道被跳过

略慢的原因是 IRLS 管道的**初始化开销**较大（构建 numpy 数组、reliability 门控筛选等），以及 `process_batch` 需要同时接收数值 df 和空文本 df 并分别处理。

#### BasicTruth — 显著最慢 (0.2345s)

BasicTruth 是原始 TruthFinder 论文的实现，其数值适配版本 `_NumericBasicTruthFinder` 存在明显的计算瓶颈：

- **置信度计算 `calculate_confidence`**：对每个 object 内的每对 fact，调用 `implication` 函数（数值蕴含 → 相对距离的高斯衰减），复杂度 $O(N^2)$ per object
- **置信度调整 `adjust_confidence`**：对每行 × 每个唯一 fact 调用 implication，且在循环内执行 `df.drop_duplicates("fact")`（创建临时 DataFrame），复杂度 $O(N^2)$ per object，常数因子大
- **Sigmoid 归一化**：逐行 Python-level `math.exp()`，$O(N)$
- **可信度更新**：逐 website 的 DataFrame 过滤，$O(N)$

**总复杂度**：$O(M \cdot N^2)$，与 SenteTruth 相同阶次，但**常数因子显著更大**：
1. `adjust_confidence` 在循环内反复调用 `df.drop_duplicates("fact")`，产生大量 DataFrame 中间对象
2. 每次 `implication` 调用涉及 Python-level 的浮点运算（无向量化）
3. 使用 `df.iterrows()` 逐行遍历（pandas 最慢的遍历方式）
4. 逐元素 `df.at[i, ...]` 赋值，未利用 numpy/pandas 的批量操作

这些实现层面的低效叠加，使 BasicTruth 的实际耗时是其他方法的 **2.7 倍**。

### 4.2 时间扩展指数分析

**排名：SemanticTruth (0.052) << SenteTruth (0.170) < SenFeedTruth (0.185) < DecentTruth (0.191) << BasicTruth (0.408)**

#### SemanticTruth — 扩展性最优 ($\alpha = 0.052$，接近常数级)

SemanticTruth 的时间几乎不随 agent 数量增长，原因：

1. **IRLS 的 MAD 早退出机制**：对于数值数据，当多数 agent 给出相近值时，MAD 趋近于 0，IRLS 循环在第一轮就退出并返回加权中位数。增加 agent 数量时，多数派更强，MAD 更可能为 0，IRLS 反而更快收敛
2. **可靠度门控预过滤**：reliability 低于均值 × `rel_gate_frac` 的 source 被预先剔除，增加 agent 时被过滤的数量增多，参与 IRLS 计算的有效 agent 数趋于稳定
3. **向量化操作**：核心计算使用 numpy 数组操作（`np.average`, `np.median`, `np.abs`, `np.exp`），利用了 SIMD 加速，增加元素数量时开销增长极小

#### SenteTruth、SenFeedTruth、DecentTruth — 良好的亚线性扩展 ($\alpha \approx 0.17\text{-}0.19$)

三者的扩展指数接近，原因是在当前实验规模下，它们的核心循环均为 Python-level 遍历，agent 数翻倍时时间增长约 12-14%：

- **DecentTruth/SenFeedTruth**：虽然算法理论上是 $O(N \cdot M)$（线性），但 Python 的 `iterrows()` 遍历和 dict 操作有固定开销，使得实际增长略低于线性（numpy 的批量操作吸收了部分增量）
- **SenteTruth**：虽然包含 $O(N^2)$ 的相似度矩阵计算，但在小 $N$ 范围内（2-10），二次项和一次项的差别被常数项掩盖，实测指数反而与线性方法相近

#### BasicTruth — 扩展性最差 ($\alpha = 0.408$)

BasicTruth 的高扩展指数是其 $O(N^2)$ 算法与低效实现的共同结果：

1. **二次复杂度显性化**：`calculate_confidence` 和 `adjust_confidence` 均为双重循环（$O(N^2)$），agent 数翻倍时计算量增长约 $2^{0.41} \approx 1.33$ 倍（32% 增长）
2. **DataFrame 操作的超线性开销**：循环内的 `df.drop_duplicates()` 和 `df.loc[...]` 过滤操作的开销随数据量超线性增长（pandas 内部涉及哈希和索引重建）
3. **无向量化加速**：Python-level 的逐元素 `math.exp()` 和 `math.log()` 无法从数据规模增大中获得 SIMD 吞吐量提升

### 4.3 内存分析

**五种方法的内存指标几乎无差异**：
- `avg_mem_delta` 均约为 246 MB，最大差异仅 0.26 MB
- `mem_exp` 均约为 0.03，接近 0

**原因**：

1. **固定开销主导**：每次测试在独立子进程中运行（`subprocess.run()`），约 246 MB 的内存增量主要来自 **Python 解释器启动 + 依赖库加载**（NumPy、Pandas、SciPy、项目模块等），这是固定开销，与 agent 数量和算法选择无关
2. **算法增量内存可忽略**：在当前规模下（10 个 agent × 3 个 object × 10 个 batch），算法本身的数据结构（dict、DataFrame、相似度矩阵等）仅占几十 KB 到几 MB，相对于 246 MB 的基线可以忽略
3. **RSS 采样精度限制**：20ms 的采样间隔可能遗漏毫秒级的内存峰值，且 `ru_maxrss` 反映的是进程生命周期内的最大 RSS，受操作系统页分配策略影响

**结论**：在当前实验规模下，内存维度不具有区分度。如需观察真实的内存扩展差异，需要将 agent 数量提升至数百甚至数千级别，使算法数据结构的内存消耗显著超过 Python/库的固定开销。

## 五、总结

| 维度 | 最优方法 | 最差方法 | 关键差异原因 |
|---|---|---|---|
| 绝对速度 | DecentTruth (0.083s) | BasicTruth (0.235s) | 线性 $O(NM)$ vs 二次 $O(MN^2)$ + DataFrame 低效操作 |
| 时间可扩展性 | SemanticTruth ($\alpha=0.05$) | BasicTruth ($\alpha=0.41$) | IRLS 早退出 + 向量化 vs Python-level 双重循环 |
| 内存消耗 | 无显著差异 | 无显著差异 | Python/库加载的固定开销 (~246MB) 主导 |

**核心发现**：

1. **SemanticTruth 在时间可扩展性上表现最优**（$\alpha \approx 0.05$，接近常数级），得益于 IRLS 的自适应早退出机制和 numpy 向量化操作。这意味着在大规模 agent 部署场景下，SemanticTruth 的运行时间几乎不会随参与节点数增长
2. **DecentTruth 和 SenFeedTruth 在绝对速度上最快**，适合对单次延迟敏感的实时场景
3. **BasicTruth 在速度和可扩展性上均显著落后**，主要瓶颈不在算法设计（$O(N^2)$ 与 SenteTruth 相同），而在于实现层面的 DataFrame 逐行操作和循环内重复创建临时对象
4. **SenteTruth 取得了良好的平衡**——绝对速度接近线性方法，扩展指数也处于合理范围
