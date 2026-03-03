# Python Files Overview

本文件统一说明 `core` 与 `examples` 目录下主要 `.py` 文件的用途。

## 通用数据格式

TruthFinder 系列实现通常使用以下字段：
- `website`: 数据源/节点标识
- `fact`: 源提供的事实值（文本、离散值或数值）
- `object`: 被讨论对象标识

## `core` 目录

### `core/__init__.py`
- `core` 包入口文件。
- 统一导出对外使用的主要类和函数：
  - `EnhancedTruthFinder`
  - `SenteTruthFinder`
  - `BasicTruthFinder`

### `core/basic_truth_finder.py`
https://github.com/IshitaTakeshi/TruthFinder
- 原始 TruthFinder 思路的基础实现。
- 主要功能：
  - 迭代计算事实置信度（`fact_confidence`）
  - 迭代更新数据源可信度（`trustworthiness`）
  - 根据收敛阈值停止训练
- 关键接口：
  - `BasicTruthFinder.train(...)`
  - `create_implication_function_from_texts(...)`

### `core/sente_truth_finder.py`
https://dl.acm.org/doi/10.1145/3767296
- 基于 SenteTruth 的语义型真值发现实现（面向文本）。
- 主要功能：
  - SBERT（失败时回退 TF-IDF）文本编码
  - 节点间语义相似度计算
  - 全局真值聚合与节点可信度更新（`source_reliability`）
- 关键接口：
  - `SenteTruthFinder.train(...)`
  - `SenteTruthFinder.get_source_reliability(...)`
  - `SenteTruthFinder.reset_history()`

### `core/enhanced_truth_finder.py`
https://ieeexplore.ieee.org/document/10229019
- 增强版 TruthFinder，实现 DECENTRUTH 风格的可靠性更新机制。
- 主要功能：
  - 维护源可靠性历史（可靠性、误差、一致性）
  - 支持数值型、离散型和混合文本模式
  - 支持批量训练（`train`）和增量处理（`process_batch`）
- 关键接口：
  - `EnhancedTruthFinder.train(...)`
  - `EnhancedTruthFinder.process_batch(...)`
  - `EnhancedTruthFinder.get_source_reliability(...)`
  - `EnhancedTruthFinder.reset_history()`

## `examples` 目录

### `examples/enhanced_example.py`
- `EnhancedTruthFinder` 的基础使用示例（离散字符串数据）。
- 展示批量训练、增量处理和源可靠性查询。

### `examples/comparison_test.py`
- 对比 `EnhancedTruthFinder` 与原 `TruthFinder`（`truthdiscovery`）在数值场景下的表现。
- 通过构造拜占庭源，观察异常源识别与估计误差差异。

### `examples/hybrid_data_example.py`
- 混合文本数据示例，展示多方法对比：
  - `EnhancedTruthFinder`（混合模式）
  - `EnhancedTruthFinder`（非混合模式）
  - `SenteTruthFinder`
  - `BasicTruthFinder`

### `examples/evaluation_comparison.py`
- 评估脚本，用于比较混合模式与非混合模式。
- 指标包括：
  - 综合准确率（含混合模式下数值/离散准确率）
  - 源可靠性排名准确性（Spearman）
  - 恶意源检测指标（Precision/Recall/F1）
  - 收敛迭代次数
