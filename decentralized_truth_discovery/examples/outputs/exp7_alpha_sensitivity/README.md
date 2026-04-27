# E7: Alpha Sensitivity Analysis

## 实验目的

分析混合评估指标 DataAcc 中语义-数值混合权重 $\alpha$ 的敏感度。

$$
\text{DataAcc}(\alpha) = \frac{1}{N} \sum_{k} \mathbb{1}\Big[\alpha \cdot \text{SemSim}(x_k^{\text{clean}},\; y_k^{\text{noisy}}) + (1-\alpha) \cdot \text{NumSim}(x_k^{\text{clean}},\; y_k^{\text{noisy}}) \;\geq\; 0.85\Big]
$$

其中：
- $\alpha = 0.0$ 表示纯数值匹配
- $\alpha = 1.0$ 表示纯语义匹配
- $\alpha = 0.5$ 为当前系统默认值

---

## 实验设置

| 项目 | 值 |
|---|---|
| 数据集 | `data_agent-50_news-300` (50 agents, 300 batches) |
| 扫描变量 | $\alpha \in \{0.0, 0.1, 0.2, \ldots, 1.0\}$（11 个点）|
| 恶意节点数 | 0, 5, 10, 15, 20, 25, 30, 35, 40（对应 0%~80%）|
| 对比方法 | SenteTruth, BasicTruth, SemanticTruth |
| 评估阈值 | threshold = 0.85 |
| 模型内部 alpha | 固定 0.5（仅扫描评估指标中的混合权重）|

## 实验流程

```
对每个 bad_count ∈ {0, 5, 10, ..., 40}:
  1. 加载 clean 数据 (demo.json) 和 noisy 数据 (demo_{bad_count}bad_collude.json)
  2. 对每个方法（SenteTruth / BasicTruth / SemanticTruth）:
     a. 在 clean 数据上运行，得到 clean_preds: Dict[(batch_idx, object), str]
     b. 在 noisy 数据上运行，得到 noisy_preds: Dict[(batch_idx, object), str]
  3. 预计算语义和数值分数（仅一次 SBERT 编码）:
     - sem_scores: SBERT cosine similarity (clean vs noisy)
     - num_scores: 数字抽取匹配率 (clean vs noisy, tol=0.02)
  4. 对每个 alpha ∈ {0.0, 0.1, ..., 1.0}:
     - combined = alpha * sem_scores + (1-alpha) * num_scores
     - DataAcc = mean(combined >= 0.85)
```

### 效率优化

SBERT 编码是最耗时的步骤。通过预计算 `sem_scores` 和 `num_scores`，alpha 扫描阶段仅需执行简单的加权求和，无需重复编码。实际运行中，每个 (method, bad_ratio) 只做 1 次 SBERT 编码，而非 11 次。

---

## 输出文件

### `alpha_sensitivity_by_method.png`

**布局**: 3 个并列子图，每个子图对应一个方法 (SenteTruth / BasicTruth / SemanticTruth)。

**坐标轴**:
- X 轴: $\alpha$ (semantic weight)，范围 [0.0, 1.0]
- Y 轴: Data Accuracy，范围 [0.0, 1.0]

**线条**: 每条线代表一个恶意比例（0%~80%），使用 `YlOrRd` 顺序色带着色（浅黄 = 低攻击，深红 = 高攻击）。

**辅助线**: $\alpha = 0.5$ 处的灰色虚线标记当前默认值。

**绘图逻辑** (对应代码 `_plot_by_method()`):
```python
for ax, method in zip(axes, methods):
    for bad_k in bad_counts:
        y_vals = [results[method][bad_k][alpha] for alpha in alpha_values]
        ax.plot(alpha_values, y_vals, color=cmap(norm(bad_k)), marker="o")
```

### `alpha_sensitivity_summary.png`

**布局**: 单图。

**坐标轴**:
- X 轴: $\alpha$
- Y 轴: Mean Data Accuracy（对所有恶意比例取均值）

**线条**: 每条线代表一个方法，使用方法专属颜色（SenteTruth=绿, BasicTruth=橙, SemanticTruth=紫）。

**绘图逻辑** (对应代码 `_plot_summary()`):
```python
for method in TEXT_METHODS:
    mean_vals = []
    for alpha in alpha_values:
        accs = [results[method][bad_k][alpha] for bad_k in bad_counts]
        mean_vals.append(np.mean(accs))
    ax.plot(alpha_values, mean_vals, color=COLORS[method], marker=MARKERS[method])
```

### `results.csv`

| 列名 | 说明 |
|---|---|
| `method` | 方法名称 (SenteTruth / BasicTruth / SemanticTruth) |
| `bad_count` | 恶意节点数 (0, 5, 10, ..., 40) |
| `malicious_pct` | 恶意比例 (%) |
| `alpha` | 混合权重 (0.0 ~ 1.0) |
| `data_accuracy` | 对应的 DataAcc 值 |

共 3 方法 x 9 恶意比例 x 11 alpha = **297 行**。

---

## 核心指标计算

### SemSim（语义相似度）

使用 SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) 编码 clean 和 noisy 预测文本，计算归一化向量的点积（余弦相似度）:

$$
\text{SemSim}(x, y) = \frac{\text{SBERT}(x) \cdot \text{SBERT}(y)}{\|\text{SBERT}(x)\| \cdot \|\text{SBERT}(y)\|}
$$

若 SBERT 不可用，退化为 char-ngram TF-IDF 余弦相似度。

### NumSim（数值匹配率）

从 clean 文本中抽取所有数字作为 reference，从 noisy 文本中抽取数字作为 candidate。对每个 reference 数字，检查 candidate 中是否存在相对误差 $\leq 0.02$ 的匹配:

$$
\text{NumSim}(x, y) = \frac{\text{matched}(x, y)}{|\text{nums}(x)|}
$$

其中 $\text{matched}(x, y) = \sum_{r \in \text{nums}(x)} \mathbb{1}\left[\exists\, c \in \text{nums}(y): \frac{|r-c|}{\max(|r|, 10^{-8})} \leq 0.02\right]$

若 clean 文本不含数字，则 NumSim 返回 None，此时 DataAcc 退化为纯语义评估。

### 混合公式

$$
\text{score}_k = \begin{cases}
\alpha \cdot \text{SemSim}_k + (1-\alpha) \cdot \text{NumSim}_k & \text{if NumSim}_k \neq \text{None} \\
\text{SemSim}_k & \text{otherwise}
\end{cases}
$$

$$
\text{DataAcc}(\alpha) = \frac{1}{N}\sum_k \mathbb{1}[\text{score}_k \geq 0.85]
$$

---

## 复现

```bash
cd decentralized_truth_discovery/examples
python exp7_alpha_sensitivity.py
# 或指定数据/输出目录:
python exp7_alpha_sensitivity.py --data-dir <path> --out-dir <path>
```

源码: `decentralized_truth_discovery/examples/exp7_alpha_sensitivity.py`
