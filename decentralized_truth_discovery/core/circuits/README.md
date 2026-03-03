# Hybrid Truth Finder 零知识证明电路

## 概述

本目录包含用于证明Hybrid Truth Finder算法正确性的Circom零知识证明电路。

## 文件说明

- `hybrid_truth_finder.circom`: 主电路文件，包含所有证明组件
- `test_zkp.js`: JavaScript测试脚本，用于验证电路的正确性
- `zkp_verification.md`: 详细的验证文档和数学证明
- `README.md`: 本文件

## 快速开始

### 1. 安装依赖

```bash
npm install circom circomlib snarkjs
```

### 2. 运行测试

```bash
node test_zkp.js
```

### 3. 编译电路

```bash
circom hybrid_truth_finder.circom --r1cs --wasm --sym
```

### 4. 生成和验证证明

参考 `zkp_verification.md` 中的详细说明。

## 电路组件

### 核心证明组件

1. **WeightedAverage**: 证明加权平均计算的正确性
2. **WeightedVoting**: 证明加权投票的正确性
3. **ReliabilityUpdate**: 证明可靠性更新的正确性
4. **HybridTruthFinderProof**: 组合所有证明的主电路

### 辅助组件

- 基本算术运算（乘法、加法、减法、除法）
- 比较运算（相等、大于）
- 聚合运算（求和、最大值查找）
- 特殊函数（绝对值、erfc近似）

## 正确性保证

### 完整性（Completeness）

如果计算正确，证明应该被接受。

### 可靠性（Soundness）

如果计算错误，证明应该被拒绝。

### 零知识性（Zero-Knowledge）

验证者除了知道证明为真外，不获得任何额外信息。

## 数学证明

详细的数学证明请参考 `zkp_verification.md`。

## 注意事项

1. **定点数精度**：电路使用定点数进行除法运算，需要注意精度问题
2. **erfc近似**：erfc函数使用多项式近似，可能存在一定误差
3. **约束数量**：电路约束数量较大，生成证明可能需要较长时间

## 性能优化

1. 减少约束数量
2. 使用更高效的组件
3. 并行计算
4. 近似优化

## 贡献

欢迎提交问题和改进建议。
