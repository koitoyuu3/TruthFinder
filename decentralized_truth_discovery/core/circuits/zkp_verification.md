# Hybrid Truth Finder 零知识证明验证文档

## 电路设计说明

### 1. 核心证明组件

#### WeightedAverage (加权平均证明)
**功能**：证明数值真值的加权平均计算正确性
- **输入**：权重数组 `weights[n]`，值数组 `values[n]`，声称的真值 `claimed_truth`
- **证明**：`truth = Σ(r_i × v_i) / Σ(r_i)`
- **验证**：
  - 计算加权和：`weighted_sum = Σ(weights[i] × values[i])`
  - 计算总权重：`total_weight = Σ(weights[i])`
  - 计算真值：`computed_truth = weighted_sum / total_weight`
  - 验证：`claimed_truth == computed_truth` 且 `total_weight > 0`

#### WeightedVoting (加权投票证明)
**功能**：证明离散真值的加权投票计算正确性
- **输入**：权重数组、值数组、候选值数组、声称的真值
- **证明**：`truth = argmax(Σ(r_i))`，其中值相同
- **验证**：
  - 对每个候选值，计算权重和
  - 找到权重和最大的候选值
  - 验证：声称的真值对应的权重和是否最大

#### ReliabilityUpdate (可靠性更新证明)
**功能**：证明源可靠性更新的计算正确性
- **输入**：alpha, beta, 上一轮误差/一致性, 当前MSE, 声称的新误差/一致性/可靠性
- **证明**：
  - `e_s = α × e_prev + (1-α) × MSE`
  - `k_s = erfc(β × |e_s - e_prev|)`
  - `r_s = (1/(1+e_s)) × k_s`
- **验证**：每个公式的计算是否正确

### 2. 辅助组件

- **Multiplier**: 乘法运算
- **Adder/Subtractor**: 加减运算
- **Divider**: 除法运算（使用定点数）
- **Sum**: 求和运算
- **IsEqual**: 相等比较
- **GreaterThan**: 大于比较
- **AbsoluteValue**: 绝对值计算
- **ErfcApprox**: erfc函数近似（使用多项式）

### 3. 主电路

**HybridTruthFinderProof**: 组合所有证明组件
- 证明数值真值计算
- 证明离散真值计算
- 证明可靠性更新
- 最终验证：所有证明都通过

## 正确性验证

### 完整性（Completeness）

**定义**：如果计算正确，证明应该被接受

**验证方法**：
1. 使用正确的输入数据生成证明
2. 验证电路输出 `is_valid = 1`

**测试用例**：
```python
# 测试数据
weights = [0.9, 0.8, 0.7, 0.6]  # 源可靠性
values = [100.5, 101.0, 99.8, 100.2]  # 源提供的值

# 计算真值
weighted_sum = 0.9*100.5 + 0.8*101.0 + 0.7*99.8 + 0.6*100.2
total_weight = 0.9 + 0.8 + 0.7 + 0.6
truth = weighted_sum / total_weight  # 应该约等于 100.4

# 验证：如果claimed_truth = truth，则is_valid = 1
```

### 可靠性（Soundness）

**定义**：如果计算错误，证明应该被拒绝

**验证方法**：
1. 使用错误的输入数据生成证明
2. 验证电路输出 `is_valid = 0`

**测试用例**：
```python
# 错误的真值
claimed_truth = 150.0  # 明显错误

# 验证：is_valid = 0（因为150.0 ≠ 100.4）
```

### 零知识性（Zero-Knowledge）

**定义**：验证者除了知道证明为真外，不获得任何额外信息

**实现方式**：
1. 使用承诺（Commitment）隐藏输入数据
2. 只公开必要的验证信息（如真值、可靠性）
3. 不泄露源数据的具体值

## 电路编译和测试

### 1. 安装依赖

```bash
npm install circom circomlib snarkjs
```

### 2. 编译电路

```bash
circom hybrid_truth_finder.circom --r1cs --wasm --sym
```

### 3. 生成证明

```bash
# 生成可信设置
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

# 生成证明密钥
snarkjs groth16 setup hybrid_truth_finder.r1cs pot12_final.ptau hybrid_truth_finder_0000.zkey
snarkjs zkey contribute hybrid_truth_finder_0000.zkey hybrid_truth_finder_0001.zkey --name="First contribution" -v
snarkjs zkey export verificationkey hybrid_truth_finder_0001.zkey verification_key.json

# 生成证明
snarkjs groth16 prove hybrid_truth_finder_0001.zkey witness.wtns proof.json public.json
```

### 4. 验证证明

```bash
snarkjs groth16 verify verification_key.json public.json proof.json
```

## 数学正确性证明

### 定理1：加权平均计算的正确性

**陈述**：如果 `truth = Σ(r_i × v_i) / Σ(r_i)`，且 `total_weight > 0`，则电路输出 `is_valid = 1`

**证明**：
1. 电路计算 `weighted_sum = Σ(r_i × v_i)`
2. 电路计算 `total_weight = Σ(r_i)`
3. 电路计算 `computed_truth = weighted_sum / total_weight`
4. 如果 `claimed_truth = computed_truth`，则 `check_truth.out = 1`
5. 如果 `total_weight > 0`，则 `check_weight.out = 1`
6. 因此 `is_valid = 1`

**QED**

### 定理2：加权投票的正确性

**陈述**：如果 `truth = argmax(Σ(r_i))`，则电路输出 `is_valid = 1`

**证明**：
1. 对每个候选值，电路计算权重和
2. 电路找到权重和最大的候选值
3. 如果 `claimed_truth` 对应的权重和等于最大值，则 `is_valid = 1`

**QED**

### 定理3：可靠性更新的正确性

**陈述**：如果可靠性更新公式正确执行，则电路输出 `is_valid = 1`

**证明**：
1. 误差更新：`e_s = α × e_prev + (1-α) × MSE`
   - 电路验证 `claimed_error = computed_error`
2. 一致性更新：`k_s = erfc(β × |e_s - e_prev|)`
   - 电路验证 `claimed_consistency = computed_consistency`
3. 可靠性更新：`r_s = (1/(1+e_s)) × k_s`
   - 电路验证 `claimed_reliability = computed_reliability`
4. 如果所有验证都通过，则 `is_valid = 1`

**QED**

## 安全性分析

### 1. 输入验证

- **范围检查**：确保权重、值在合理范围内
- **除零检查**：确保 `total_weight > 0`
- **溢出检查**：使用定点数避免溢出

### 2. 计算验证

- **算术运算**：所有运算都在有限域内进行
- **比较运算**：使用安全的比较器
- **近似函数**：erfc使用多项式近似，误差可控

### 3. 隐私保护

- **数据隐藏**：源数据通过承诺隐藏
- **部分公开**：只公开必要的验证信息
- **零知识性**：验证者无法从证明中推断源数据

## 性能优化建议

1. **减少约束数量**：优化电路结构，减少不必要的约束
2. **使用更高效的组件**：使用circomlib中的优化组件
3. **并行计算**：对独立的计算进行并行处理
4. **近似优化**：对erfc等复杂函数使用更高效的近似

## 总结

本零知识证明系统能够：
1. **证明计算正确性**：验证真值计算和可靠性更新的正确性
2. **保护数据隐私**：不泄露源数据的具体值
3. **提供可验证性**：任何人都可以验证证明的正确性

通过数学证明和实际测试，可以确保系统的完整性和可靠性。
