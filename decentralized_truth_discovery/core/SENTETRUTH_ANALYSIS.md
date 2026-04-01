# SenteTruth方法可信度差异分析报告

## 问题描述

在对比测试中，SenteTruth方法的可信度评估结果与其他两种方法（混合模式和原有方法）存在显著差异：

```
可靠性/可信度对比:
  Agent_1:
    混合模式 (EnhancedTruthFinder): 0.937010
    原有方法 (EnhancedTruthFinder, 非混合): 1.000000
    SenteTruth方法: 0.032567
    SenteTruth vs 混合模式: -96.52%

  Agent_2:
    混合模式 (EnhancedTruthFinder): 0.882465
    原有方法 (EnhancedTruthFinder, 非混合): 0.499991
    SenteTruth方法: 1.000000
    SenteTruth vs 混合模式: +13.32% ⚠️

  Agent_3:
    混合模式 (EnhancedTruthFinder): 0.812314
    原有方法 (EnhancedTruthFinder, 非混合): 0.499991
    SenteTruth方法: 0.163724
    SenteTruth vs 混合模式: -79.84%

  Agent_4:
    混合模式 (EnhancedTruthFinder): 0.499991
    原有方法 (EnhancedTruthFinder, 非混合): 0.499991
    SenteTruth方法: 0.010000
    SenteTruth vs 混合模式: -98.00%
```

## 核心问题分析

### 1. 公式4的数值不稳定问题（主要原因）

**公式4：**
```
C_i ← ( (Σ_{l=1}^n C_l) / (Σ_{l=1}^n C_l · φ(v_l)) ) · C_i · φ(v_i)
```

**问题机制：**

当所有节点的 φ(v_i)（语义相似度分数）普遍较小时，分母 `Σ C_l · φ(v_l)` 会变得很小，导致 `(Σ C_l) / (Σ C_l · φ(v_l))` 这个归一化因子变得异常大。

**数值示例：**

假设所有节点初始可信度 C_i = 0.5，有4个节点：

- 如果 φ(v_i) 都很小（比如 0.1-0.3），那么：
  - 分子：Σ C_l = 4 × 0.5 = 2.0
  - 分母：Σ C_l · φ(v_l) ≈ 4 × 0.5 × 0.2 = 0.4
  - **归一化因子：2.0 / 0.4 = 5.0**（放大5倍！）

- 如果某个节点的 φ(v_i) = 0.3（相对较高）：
  - 更新后：C_i = 5.0 × 0.5 × 0.3 = **0.75**（大幅提升）

- 如果某个节点的 φ(v_i) = 0.1（相对较低）：
  - 更新后：C_i = 5.0 × 0.5 × 0.1 = **0.25**（大幅降低）

**影响：**
- 可信度更新会出现异常放大或缩小
- 在迭代过程中，这种效应会累积，导致极端值
- Agent_2 的可信度被放大到 1.0（达到上限）
- Agent_1、Agent_3、Agent_4 的可信度被大幅降低

### 2. 文本语义相似度普遍较低

**测试数据特点：**

- **Agent_1**: "A new Agilent Technologies, Inc. (Symbol: A) option chain has opened with a $125 put priced at $4.50 and a $150 call at $8.10..."
- **Agent_2**: "Newly trading August 2024 options for Agilent Technologies Inc., including a $125 put with $4 premium and a $150 call at $8.10..."
- **Agent_3**: "A recent round of Agilent Technologies options... The August $150 put has a bid of $8.10..."（包含错误：$150 put应该是$150 call）
- **Agent_4**: "Ticker A is leading the tech stock rally, trading at all-time highs!"（完全不同的内容）

**相似度分析：**

- Agent_1 和 Agent_2：内容相似但表述方式不同，语义相似度可能在 0.4-0.6 之间
- Agent_3：包含错误信息，与其他节点的相似度较低（0.2-0.4）
- Agent_4：文本完全不同，相似度极低（0.1-0.2）

**影响：**

- 相似度矩阵中的值普遍较低（0.2-0.5）
- 导致 φ(v_i) 值也较低（0.15-0.35）
- 这进一步放大了公式4的不稳定性

### 3. 相似度归一化可能不合适

**当前实现：**
```python
similarity = (cosine_similarity + 1) / 2  # 从[-1, 1]归一化到[0, 1]
```

**问题：**

- 余弦相似度原本在 [-1, 1] 范围内
- 归一化到 [0, 1] 会压缩分布：
  - 原本 0.5 的相似度变成 0.75
  - 原本 -0.5 的相似度变成 0.25
  - 原本 0.0 的相似度变成 0.5

**影响：**

- 可能导致 φ(v_i) 值偏高或偏低
- 如果原始相似度分布不均匀，归一化后可能失真

### 4. Agent_2 可信度达到 1.0 的原因

**可能原因：**

1. **相对较高的 φ(v_i)**：
   - Agent_2 的文本与 Agent_1 相似度较高
   - 与其他节点的平均相似度相对较高
   - φ(v_i) 可能在 0.3-0.4 之间（相对较高）

2. **公式4的放大效应**：
   - 在低相似度环境下，归一化因子很大（如5.0）
   - Agent_2 的 φ(v_i) 相对较高，在放大效应下可信度快速提升
   - 由于代码中有 `min(1.0, new_C_i)` 的限制，可信度被截断到 1.0

3. **迭代累积**：
   - 第一次迭代后，Agent_2 的可信度可能已经提升到 0.7-0.8
   - 后续迭代中，由于可信度已经较高，φ(v_i) 计算时权重更大
   - 进一步放大，最终达到上限 1.0

### 5. 其他节点可信度极低的原因

**Agent_1 (0.032567)：**

- 可能 φ(v_i) 相对较低（虽然内容正确，但表述方式导致相似度不高）
- 在公式4的放大效应下，可信度被大幅降低
- 迭代过程中持续下降，最终接近最小值

**Agent_3 (0.163724)：**

- 包含错误信息（$150 put 应该是 $150 call）
- 与其他节点的相似度较低
- φ(v_i) 较低，在公式4作用下可信度下降
- 但由于仍有一些相似度，没有降到最低

**Agent_4 (0.010000)：**

- 文本完全不同（只说股票涨，没有期权信息）
- 与其他节点的相似度极低（接近0）
- φ(v_i) 接近 0
- 在公式4作用下，可信度被降到最小值（代码中设置的下限 0.01）

### 6. 与混合模式的对比

**混合模式（EnhancedTruthFinder）的优势：**

1. **结构化信息提取**：
   - 从文本中提取数值信息（价格、溢价等）
   - 提取离散信息（公司名、股票代码等）
   - 不依赖文本的语义相似度

2. **稳定的误差计算**：
   - 使用多维度误差计算（数值误差 + 离散误差）
   - 误差计算基于提取的结构化信息，更稳定
   - 不会出现公式4的放大效应

3. **更准确的评估**：
   - 能准确识别 Agent_1 和 Agent_2 提供的信息是正确的
   - 能识别 Agent_3 的错误信息
   - 能识别 Agent_4 的信息不完整

**SenteTruth 的局限性：**

1. **只依赖语义相似度**：
   - 对结构化信息不敏感
   - 即使内容正确，表述不同也可能被误判

2. **公式4的不稳定性**：
   - 在低相似度情况下不稳定
   - 容易出现极端值

3. **对文本表述差异敏感**：
   - Agent_1 和 Agent_2 虽然内容正确，但表述不同
   - 语义相似度可能不够高，导致可信度评估不准确

### 7. 迭代过程中的累积效应

**迭代过程：**

1. **第一次迭代**：
   - 所有节点初始可信度 = 0.5
   - 计算相似度矩阵和 φ(v_i)
   - 应用公式4更新可信度
   - 可信度开始分化

2. **后续迭代**：
   - 使用更新后的可信度
   - 重新计算相似度矩阵和 φ(v_i)
   - 再次应用公式4
   - **放大效应累积**，可信度差异越来越大

3. **收敛或达到上限**：
   - 某些节点的可信度达到上限（1.0）或下限（0.01）
   - 算法收敛或达到最大迭代次数

**累积效应的影响：**

- 小的初始差异会被放大
- Agent_2 的微小优势在迭代中被放大到 1.0
- Agent_1、Agent_3、Agent_4 的劣势被放大到极低值

## 总结

### 根本原因

**公式4在低相似度环境下的数值不稳定是主要原因：**

1. 当 φ(v_i) 普遍较小时，分母很小，导致归一化因子很大
2. 可信度更新会出现异常放大或缩小
3. 迭代过程中这种效应会累积，导致极端值

### 次要原因

1. **文本语义相似度普遍较低**：测试数据中文本表述差异大，相似度不高
2. **相似度归一化可能不合适**：从 [-1, 1] 归一化到 [0, 1] 可能压缩了分布
3. **只依赖语义相似度**：对结构化信息不敏感，不如混合模式准确

### 改进建议

1. **添加平滑因子**：
   - 在分母中添加小的平滑项，避免分母过小
   - 例如：`Σ C_l · φ(v_l) + ε`，其中 ε 是一个小的正数（如 0.01）

2. **限制更新幅度**：
   - 限制单次迭代中可信度的变化幅度
   - 例如：`new_C_i = C_i + α * (update - C_i)`，其中 α 是学习率（如 0.1）

3. **使用对数空间或归一化**：
   - 在更新前对可信度进行归一化
   - 或使用对数空间进行计算，提高数值稳定性

4. **结合与真值的相似度**：
   - 不仅依赖节点间相似度，还考虑与真值的相似度
   - 这样可以更准确地评估节点的可靠性

5. **改进相似度计算**：
   - 考虑使用更合适的相似度度量
   - 或者不进行归一化，直接使用余弦相似度

### 结论

这解释了为什么 SenteTruth 的性能不如混合模式：

- **混合模式**使用更稳定的误差计算机制，基于提取的结构化信息进行评估
- **SenteTruth**的公式4在低相似度情况下容易出现数值不稳定，导致可信度评估不准确

因此，对于包含结构化信息的文本数据，混合模式是更好的选择。SenteTruth 方法更适合处理纯文本数据，且节点间语义相似度较高的场景。


1. 他们先搭了一个本地区块链原型。环境是 Ganache + Solidity + Web3.py，系统里有 user contract、oracle contract 和一个由 10 个 oracle 节点组成的 oracle network。用户合约发起请求，节点去调 LLM，最后把结果回传链上。
  2. 被测对象其实是“聚合器”而不是 LLM 本身。每个节点拿到一条文本答案后，SenteTruth 先用 SBERT 把答案编码成向量，算每个答案和其他答案的平均语义相似度，再乘上该节点的历史信誉权重，选出分数最高的答案作为最终结果；下一轮再根据这轮表现更
     新节点权重。也就是“语义相似度 + 历史信誉”的组合。
  3. 数据集分三类。BASE 有 60 题，分成 事实一致性 Q1、逻辑一致性 Q2、开放问题 Q3，每类 20 题；MIX 有 100 题，覆盖 10 个领域；PRO 基于 C-Eval 的物理题，初中、高中、大学各 20 题，总共 60 题。论文还让 10 个节点分别去问 5 个模型：
     ChatGPT-4o-mini、Gemini-1.5-flash、Llama3-8b、ChatGLM-4-flash、Hunyuan-lite。
  4. 他们专门模拟了三种恶意节点。Random Response 是节点根本不认真调模型，直接回随机废话；Model Substitution 是本来该问指定模型，却偷偷换成更便宜的模型；Incorrect Response 是通过恶意 prompt 诱导模型输出错误内容。实验主表都用 40% 恶意
     节点，也就是 10 个节点里有 4 个作恶。
  5. 对比方法有三类。第一类是 oracle 里常见的 Majority Voting；第二类是 TF-IDF Similarity；第三类是 SBERT Similarity。有些表里还加了 TF-IDF + TD，相当于一个消融，看看“真值发现”如果配弱一点的文本表示会怎样。
  6. 他们把实验拆成四个研究问题。RQ1 先研究“不一致从哪来”，比较了不同模型、不同题型、中英文提问、temperature、以及“只输出答案”还是“答案+解释”这类 prompt 设计；结论是：题目越开放越不一致，模型差异很大，temperature 调低、更约束输出格
     式，都能明显提升一致性，题目难度反而影响不大。RQ2 再正式比性能，在三种攻击下看最终聚合准确率；这里的 Accuracy 不是传统 QA 正确率，而是“最终结果有没有被恶意篡改带偏”。RQ3 看机制为什么有效，他们跟踪多轮迭代里每个节点的权重变化，
     发现恶意节点的权重会逐步下降，所以哪怕某次它的答案表面上和别人挺像，最后也不容易被选中。RQ4 测可用性，统计链上 Gas 和可扩展性。
  7. 结果上，论文最想强调的是：在 40% 恶意节点时，SenteTruth 比同条件下最好的 baseline 平均再提升 17.74%。BASE 上，随机响应攻击下它基本都是 1.0；错误响应下大多是 0.983~1.0；模型替换下仍明显优于基线。附录里的 MIX 和 PRO 基本也维持在
     0.99~1.0。另外有两个细节值得你注意：第一，这篇论文的“truth”更接近“多数诚实节点的未篡改答案”，不完全等于外部客观真值；第二，Table 5 的 Gas 表按它自己给的公式算，单次调用更接近几美元，不太像正文说的“约 $0.3/次”，这里有明显数值不
     一致。



当前方法在 decentralized_truth_discovery/examples/compare_cllm_text_methods.py:1 里，特点完全不同：

  1. 它先把 datas_CLLM 读出来，变成统一的 clean_df / noisy_df
     见 decentralized_truth_discovery/examples/compare_cllm_text_methods.py:189
  2. 它显式注入三类攻击：
      - random
      - incorrect
      - substitution
        见 decentralized_truth_discovery/examples/compare_cllm_text_methods.py:255
  3. 它不是只跑一个聚合器，而是比较三种方法：
      - SenteTruth
      - BasicTruth
      - SemanticTruth
  4. 它不走链，不测合约回调，不测 gas，只测离线鲁棒性
  5. 它最后按 data_accuracy 打分并导出 CSV
     见 decentralized_truth_discovery/examples/compare_cllm_text_methods.py:375

  最核心的方法学差异

  1. CLLM 是“在线系统实验”，你现在的是“离线对比实验”

  - CLLM 关注：节点取数后，链上 oracle 最终回什么
  - 现在这个脚本关注：在同一批 clean/noisy 数据上，不同 truth-finder 谁更稳

  2. CLLM 默认只有一个聚合器，当前实验是多方法对比

  - CLLM 聚合逻辑本质上就是“平均语义相似度最高 + 节点权重修正” /Users/huabowen/CLLM/node/utils/aggre.py:102
  - 当前脚本把 SenteTruth / BasicTruth / SemanticTruth 都跑一遍

  3. CLLM 没有显式的 clean-vs-noisy 评估框架，当前方法有

  - 当前 eval_text() 明确把
      - X = clean 上的方法输出
      - Y = noisy 上的方法输出
        做对比
        见 decentralized_truth_discovery/examples/exp_utils.py:836

  4. 当前实验的指标比 CLLM 原型更严格、也更可复现

  - 当前 metric_data_accuracy() 明确计算语义一致性，必要时再混合数值一致性
    见 decentralized_truth_discovery/examples/exp_utils.py:646
  - CLLM 代码里没有一个成体系的“批量统计指标”实现，更多是运行原型

  5. 当前实验已经偏离了论文原始 baseline 设计

  - 论文原始对比更偏向 Majority / TF-IDF / SBERT / Ours
  - 你现在做的是 SenteTruth / BasicTruth / SemanticTruth
  - 所以现在这个实验回答的问题是：
    “在同一数据和攻击下注入下，三个 truth-finder 谁更稳？”
  - 不是：
    “原论文 Ours 相对传统 baseline 提升多少？”

  一句话概括

  - /Users/huabowen/CLLM 的实验方法更像“原论文系统原型的在线执行方式”
  - TruthFinder 里的当前方法更像“把同一份数据集离线化、标准化后做公平对比的 benchmark”

  所以两者不是谁替代谁，而是在回答不同问题：

  - CLLM：系统能不能跑起来，原始聚合器怎么在 oracle 网络里工作
  - 当前脚本：在固定攻击和固定数据下，三种方法的鲁棒性谁更强

Random Response (BASE, 40%)

  | Target Model | SenteTruth | BasicTruth | SemanticTruth |
  |---|---:|---:|---:|
  | ChatGPT | 0.8500 | 0.8667 | 0.9500 |
  | ChatGLM | 0.8000 | 0.9500 | 0.9500 |
  | Llama | 0.8833 | 0.7333 | 0.8167 |
  | Gemini | 0.7500 | 0.8667 | 0.9500 |
  | Hunyuan | 0.9333 | 0.9833 | 0.9833 |

  Incorrect Response (BASE, 40%)

  | Target Model | SenteTruth | BasicTruth | SemanticTruth |
  |---|---:|---:|---:|
  | ChatGPT | 0.7667 | 0.8000 | 0.8833 |
  | ChatGLM | 0.7500 | 0.8667 | 0.8833 |
  | Llama | 0.8833 | 0.7167 | 0.7000 |
  | Gemini | 0.8667 | 0.7500 | 0.7667 |
  | Hunyuan | 0.9167 | 0.9500 | 0.9333 |

  Model Substitution (BASE, 40%, Target = ChatGPT)

  | Substitute Model | SenteTruth | BasicTruth | SemanticTruth |
  |---|---:|---:|---:|
  | ChatGLM | 0.8333 | 0.8500 | 0.8500 |
  | Llama | 0.8333 | 0.6000 | 0.8833 |
  | Gemini | 0.8333 | 0.6833 | 0.7500 |
  | Hunyuan | 0.8167 | 0.6167 | 0.8000 |

  | 场景 | Majority | TF-IDF | SBERT | Ours |
  |---|---:|---:|---:|---:|
  | BASE random | 0.9767 | 0.9367 | 0.9767 | 0.9167 |
  | BASE incorrect | 0.9900 | 0.8800 | 0.9233 | 0.9167 |
  | BASE substitution | 0.9250 | 0.8083 | 0.9292 | 0.9292 |
  | MIX random | 0.9900 | 0.9640 | 0.9780 | 0.9780 |
  | MIX incorrect | 0.9900 | 0.9300 | 0.9600 | 0.9640 |
  | MIX substitution | 0.8250 | 0.7575 | 0.9775 | 0.9850 |
  | PRO random | 0.9867 | 0.9233 | 0.9433 | 0.9000 |
  | PRO substitution | 0.9417 | 0.7625 | 0.9500 | 0.9833 |
  | PRO-single incorrect | 0.9500 | 0.9000 | 0.8800 | 0.8833 |