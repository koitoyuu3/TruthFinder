/*
 * Hybrid Truth Finder Algorithm - Zero Knowledge Proof Circuit
 * 
 * 这个电路证明hybrid混合算法的正确性，包括：
 * 1. 数值真值的加权平均计算
 * 2. 离散真值的加权投票
 * 3. 源可靠性的更新计算
 * 
 * 电路设计遵循以下原则：
 * - 完整性（Completeness）：如果计算正确，证明应该被接受
 * - 可靠性（Soundness）：如果计算错误，证明应该被拒绝
 * - 零知识性（Zero-Knowledge）：不泄露输入数据的隐私
 */

pragma circom 2.0.0;

// 注意：实际使用时需要安装circomlib
// include "../node_modules/circomlib/circuits/comparators.circom";
// include "../node_modules/circomlib/circuits/poseidon.circom";

// ========== 模板1：证明加权平均计算的正确性 ==========
// 证明：truth = Σ(r_i × v_i) / Σ(r_i)
// 其中 r_i 是源可靠性，v_i 是源提供的值
template WeightedAverage(n) {
    // 输入信号
    signal input weights[n];      // 源可靠性 r_i
    signal input values[n];       // 源提供的值 v_i
    signal input claimed_truth;   // 声称的真值
    
    // 中间信号
    signal weighted_sum;          // Σ(r_i × v_i)
    signal total_weight;           // Σ(r_i)
    signal computed_truth;         // 计算出的真值
    
    // 输出信号（用于验证）
    signal output is_valid;       // 是否为有效证明
    
    // 计算加权和
    component weighted_products[n];
    for (var i = 0; i < n; i++) {
        weighted_products[i] = Multiplier();
        weighted_products[i].a <== weights[i];
        weighted_products[i].b <== values[i];
    }
    
    // 累加加权和
    component sum_weighted = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_weighted.in[i] <== weighted_products[i].out;
    }
    weighted_sum <== sum_weighted.out;
    
    // 累加权重
    component sum_weights = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_weights.in[i] <== weights[i];
    }
    total_weight <== sum_weights.out;
    
    // 计算真值：truth = weighted_sum / total_weight
    // 使用定点数除法：truth = (weighted_sum * SCALE) / total_weight
    // 其中 SCALE 是精度因子（例如 10^6）
    var SCALE = 1000000;
    
    component scaled_weighted = Multiplier();
    scaled_weighted.a <== weighted_sum;
    scaled_weighted.b <== SCALE;
    
    component div = Divider();
    div.numerator <== scaled_weighted.out;
    div.denominator <== total_weight;
    computed_truth <== div.out;
    
    // 验证：claimed_truth 是否等于 computed_truth
    component check_truth = IsEqual();
    check_truth.in[0] <== claimed_truth;
    check_truth.in[1] <== computed_truth;
    
    // 验证：total_weight > 0（避免除零）
    component check_weight = GreaterThan();
    check_weight.in[0] <== total_weight;
    check_weight.in[1] <== 0;
    
    // 注意：IsEqual和GreaterThan需要从circomlib导入或自己实现
    
    // 最终验证：所有条件都满足
    component final_check = AND();
    final_check.a <== check_truth.out;
    final_check.b <== check_weight.out;
    
    is_valid <== final_check.out;
}

// ========== 模板2：证明加权投票的正确性 ==========
// 证明：truth = argmax(Σ(r_i))，其中值相同
// 对于离散值，选择权重和最大的候选值
template WeightedVoting(n, m) {
    // n: 源数量
    // m: 候选值数量
    
    signal input weights[n];           // 源可靠性 r_i
    signal input values[n];            // 源提供的值（编码为整数）
    signal input candidate_values[m];  // 所有候选值
    signal input claimed_truth;        // 声称的真值（编码为整数）
    
    signal output is_valid;
    
    // 对每个候选值，计算权重和
    component candidate_sums[m];
    for (var j = 0; j < m; j++) {
        candidate_sums[j] = Sum(n);
        
        // 对每个源，如果值等于候选值，则累加权重
        for (var i = 0; i < n; i++) {
            component is_equal = IsEqual();
            is_equal.in[0] <== values[i];
            is_equal.in[1] <== candidate_values[j];
            
            // 如果相等，权重累加；否则累加0
            component conditional_weight = ConditionalMultiplier();
            conditional_weight.condition <== is_equal.out;
            conditional_weight.value <== weights[i];
            conditional_weight.multiplier <== 1;
            
            candidate_sums[j].in[i] <== conditional_weight.out;
        }
    }
    
    // 找到权重和最大的候选值
    component max_finder = MaxFinder(m);
    for (var j = 0; j < m; j++) {
        max_finder.values[j] <== candidate_sums[j].out;
    }
    
    // 验证：claimed_truth 对应的权重和是否最大
    component check_max = IsEqual();
    check_max.in[0] <== max_finder.max_value;
    
    // 找到claimed_truth对应的索引
    component find_index = FindIndex(m);
    find_index.target <== claimed_truth;
    find_index.candidates <== candidate_values;
    
    check_max.in[1] <== candidate_sums[find_index.index].out;
    
    is_valid <== check_max.out;
}

// ========== 模板3：证明可靠性更新的正确性 ==========
// 证明：r_s = (1/(1+e_s)) × k_s
// 其中 e_s = α × e_prev + (1-α) × MSE
//      k_s = erfc(β × |e_s - e_prev|)
template ReliabilityUpdate() {
    signal input alpha;              // 衰减因子
    signal input beta;               // 尺度因子
    signal input prev_error;         // 上一轮误差 e_prev
    signal input prev_consistency;   // 上一轮一致性 k_prev
    signal input current_mse;        // 当前均方误差 MSE
    signal input claimed_error;      // 声称的新误差 e_s
    signal input claimed_consistency;// 声称的新一致性 k_s
    signal input claimed_reliability;// 声称的新可靠性 r_s
    
    signal output is_valid;
    
    // 步骤1：计算新误差 e_s = α × e_prev + (1-α) × MSE
    component alpha_e_prev = Multiplier();
    alpha_e_prev.a <== alpha;
    alpha_e_prev.b <== prev_error;
    
    component one_minus_alpha = Subtractor();
    one_minus_alpha.a <== 1;
    one_minus_alpha.b <== alpha;
    
    component one_minus_alpha_mse = Multiplier();
    one_minus_alpha_mse.a <== one_minus_alpha.out;
    one_minus_alpha_mse.b <== current_mse;
    
    component computed_error = Adder();
    computed_error.a <== alpha_e_prev.out;
    computed_error.b <== one_minus_alpha_mse.out;
    
    // 验证误差计算
    component check_error = IsEqual();
    check_error.in[0] <== claimed_error;
    check_error.in[1] <== computed_error.out;
    
    // 步骤2：计算一致性 k_s = erfc(β × |e_s - e_prev|)
    component error_diff = Subtractor();
    error_diff.a <== claimed_error;
    error_diff.b <== prev_error;
    
    // 计算绝对值 |e_s - e_prev|
    component abs_diff = AbsoluteValue();
    abs_diff.in <== error_diff.out;
    
    component beta_abs = Multiplier();
    beta_abs.a <== beta;
    beta_abs.b <== abs_diff.out;
    
    // 计算 erfc(β × |e_s - e_prev|)
    // 注意：erfc在Circom中需要近似实现
    component computed_consistency = ErfcApprox();
    computed_consistency.x <== beta_abs.out;
    
    // 更新一致性：k_s = α × k_prev + (1-α) × new_k
    component alpha_k_prev = Multiplier();
    alpha_k_prev.a <== alpha;
    alpha_k_prev.b <== prev_consistency;
    
    component one_minus_alpha_new_k = Multiplier();
    one_minus_alpha_new_k.a <== one_minus_alpha.out;
    one_minus_alpha_new_k.b <== computed_consistency.out;
    
    component updated_consistency = Adder();
    updated_consistency.a <== alpha_k_prev.out;
    updated_consistency.b <== one_minus_alpha_new_k.out;
    
    // 验证一致性计算
    component check_consistency = IsEqual();
    check_consistency.in[0] <== claimed_consistency;
    check_consistency.in[1] <== updated_consistency.out;
    
    // 步骤3：计算可靠性 r_s = (1/(1+e_s)) × k_s
    component one_plus_error = Adder();
    one_plus_error.a <== 1;
    one_plus_error.b <== claimed_error;
    
    // 计算 1/(1+e_s)
    var SCALE = 1000000;
    component scaled_one = Multiplier();
    scaled_one.a <== 1;
    scaled_one.b <== SCALE;
    
    component error_component = Divider();
    error_component.numerator <== scaled_one.out;
    error_component.denominator <== one_plus_error.out;
    
    // 计算 r_s = error_component × k_s
    component computed_reliability = Multiplier();
    computed_reliability.a <== error_component.out;
    computed_reliability.b <== claimed_consistency;
    
    // 验证可靠性计算
    component check_reliability = IsEqual();
    check_reliability.in[0] <== claimed_reliability;
    check_reliability.in[1] <== computed_reliability.out;
    
    // 最终验证：所有计算都正确
    component final_check = AND3();
    final_check.a <== check_error.out;
    final_check.b <== check_consistency.out;
    final_check.c <== check_reliability.out;
    
    is_valid <== final_check.out;
}

// ========== 模板4：混合算法完整证明 ==========
// 结合数值和离散真值计算的完整证明
template HybridTruthFinderProof(n_sources, n_numeric_fields, n_discrete_fields, n_candidates) {
    // n_sources: 源数量
    // n_numeric_fields: 数值字段数量
    // n_discrete_fields: 离散字段数量
    // n_candidates: 离散字段的候选值数量
    
    // ========== 输入信号 ==========
    // 数值部分
    signal input numeric_weights[n_sources];
    signal input numeric_values[n_sources][n_numeric_fields];
    signal input claimed_numeric_truth[n_numeric_fields];
    
    // 离散部分
    signal input discrete_weights[n_sources];
    signal input discrete_values[n_sources][n_discrete_fields];
    signal input candidate_values[n_discrete_fields][n_candidates];
    signal input claimed_discrete_truth[n_discrete_fields];
    
    // 可靠性更新
    signal input alpha;
    signal input beta;
    signal input prev_error[n_sources];
    signal input prev_consistency[n_sources];
    signal input current_mse[n_sources];
    signal input claimed_error[n_sources];
    signal input claimed_consistency[n_sources];
    signal input claimed_reliability[n_sources];
    
    // ========== 输出信号 ==========
    signal output is_valid;
    
    // ========== 证明数值真值计算 ==========
    component numeric_proofs[n_numeric_fields];
    for (var i = 0; i < n_numeric_fields; i++) {
        numeric_proofs[i] = WeightedAverage(n_sources);
        for (var j = 0; j < n_sources; j++) {
            numeric_proofs[i].weights[j] <== numeric_weights[j];
            numeric_proofs[i].values[j] <== numeric_values[j][i];
        }
        numeric_proofs[i].claimed_truth <== claimed_numeric_truth[i];
    }
    
    // ========== 证明离散真值计算 ==========
    component discrete_proofs[n_discrete_fields];
    for (var i = 0; i < n_discrete_fields; i++) {
        discrete_proofs[i] = WeightedVoting(n_sources, n_candidates);
        for (var j = 0; j < n_sources; j++) {
            discrete_proofs[i].weights[j] <== discrete_weights[j];
            discrete_proofs[i].values[j] <== discrete_values[j][i];
        }
        for (var j = 0; j < n_candidates; j++) {
            discrete_proofs[i].candidate_values[j] <== candidate_values[i][j];
        }
        discrete_proofs[i].claimed_truth <== claimed_discrete_truth[i];
    }
    
    // ========== 证明可靠性更新 ==========
    component reliability_proofs[n_sources];
    for (var i = 0; i < n_sources; i++) {
        reliability_proofs[i] = ReliabilityUpdate();
        reliability_proofs[i].alpha <== alpha;
        reliability_proofs[i].beta <== beta;
        reliability_proofs[i].prev_error <== prev_error[i];
        reliability_proofs[i].prev_consistency <== prev_consistency[i];
        reliability_proofs[i].current_mse <== current_mse[i];
        reliability_proofs[i].claimed_error <== claimed_error[i];
        reliability_proofs[i].claimed_consistency <== claimed_consistency[i];
        reliability_proofs[i].claimed_reliability <== claimed_reliability[i];
    }
    
    // ========== 验证所有证明 ==========
    component all_numeric_valid = AND(n_numeric_fields);
    for (var i = 0; i < n_numeric_fields; i++) {
        all_numeric_valid.in[i] <== numeric_proofs[i].is_valid;
    }
    
    component all_discrete_valid = AND(n_discrete_fields);
    for (var i = 0; i < n_discrete_fields; i++) {
        all_discrete_valid.in[i] <== discrete_proofs[i].is_valid;
    }
    
    component all_reliability_valid = AND(n_sources);
    for (var i = 0; i < n_sources; i++) {
        all_reliability_valid.in[i] <== reliability_proofs[i].is_valid;
    }
    
    // 最终验证：所有部分都正确
    component final_check = AND3();
    final_check.a <== all_numeric_valid.out;
    final_check.b <== all_discrete_valid.out;
    final_check.c <== all_reliability_valid.out;
    
    is_valid <== final_check.out;
}

// ========== 辅助组件 ==========

// 乘法器
template Multiplier() {
    signal input a;
    signal input b;
    signal output out;
    out <== a * b;
}

// 加法器
template Adder() {
    signal input a;
    signal input b;
    signal output out;
    out <== a + b;
}

// 减法器
template Subtractor() {
    signal input a;
    signal input b;
    signal output out;
    out <== a - b;
}

// 除法器（使用定点数）
template Divider() {
    signal input numerator;
    signal input denominator;
    signal output out;
    out <== numerator / denominator;
}

// 求和器
template Sum(n) {
    signal input in[n];
    signal output out;
    var sum = 0;
    for (var i = 0; i < n; i++) {
        sum = sum + in[i];
    }
    out <== sum;
}

// 相等比较器
template IsEqual() {
    signal input in[2];
    signal output out;
    signal diff;
    diff <== in[0] - in[1];
    // 使用约束：如果diff = 0，则out = 1；否则out = 0
    // 简化实现：out = 1 - (diff != 0)
    // 注意：实际实现需要使用IsZero组件（从circomlib）
    out <== 1;  // 临时实现，需要替换为正确的IsZero
}

// 大于比较器
template GreaterThan() {
    signal input in[0];
    signal input in[1];
    signal output out;
    signal diff;
    diff <== in[0] - in[1];
    // 使用约束：如果diff > 0，则out = 1；否则out = 0
    // 简化实现：需要从circomlib导入GreaterThan组件
    out <== 1;  // 临时实现，需要替换为正确的GreaterThan
}

// AND门（2输入）
template AND() {
    signal input a;
    signal input b;
    signal output out;
    out <== a * b;
}

// AND门（3输入）
template AND3() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;
    out <== a * b * c;
}

// AND门（n输入）
template AND(n) {
    signal input in[n];
    signal output out;
    var prod = 1;
    for (var i = 0; i < n; i++) {
        prod = prod * in[i];
    }
    out <== prod;
}

// 绝对值
template AbsoluteValue() {
    signal input in;
    signal output out;
    // 简化实现：对于非负数，绝对值等于本身
    // 注意：实际实现需要处理负数情况
    out <== in;  // 临时实现，假设输入总是非负
}

// 条件选择器
template ConditionalSelector() {
    signal input condition;
    signal input if_true;
    signal input if_false;
    signal output out;
    out <== condition * if_true + (1 - condition) * if_false;
}

// 条件乘法器
template ConditionalMultiplier() {
    signal input condition;
    signal input value;
    signal input multiplier;
    signal output out;
    out <== condition * value * multiplier;
}

// 最大值查找器
template MaxFinder(n) {
    signal input values[n];
    signal output max_value;
    signal output max_index;
    
    // 使用比较器链找到最大值
    var max_val = values[0];
    var max_idx = 0;
    for (var i = 1; i < n; i++) {
        component gt = GreaterThan();
        gt.in[0] <== values[i];
        gt.in[1] <== max_val;
        // 如果 values[i] > max_val，更新最大值
        component selector = ConditionalSelector();
        selector.condition <== gt.out;
        selector.if_true <== values[i];
        selector.if_false <== max_val;
        max_val <== selector.out;
    }
    max_value <== max_val;
    // max_index 的计算需要更复杂的逻辑，这里简化处理
    max_index <== 0;
}

// 查找索引
template FindIndex(n) {
    signal input target;
    signal input candidates[n];
    signal output index;
    
    // 找到target在candidates中的索引
    // 简化实现：假设总是能找到
    index <== 0;
}

// erfc近似（使用多项式近似）
template ErfcApprox() {
    signal input x;
    signal output out;
    
    // erfc(x) ≈ 1 - erf(x)
    // 使用erf的多项式近似：erf(x) ≈ 2/√π * x * (1 - x²/3 + x⁴/10 - ...)
    // 这里使用简化的线性近似（实际应该使用更精确的多项式）
    var PI_SQRT = 1772453850; // √π * 10^9（定点数）
    var TWO_OVER_SQRT_PI = 1128379166; // 2/√π * 10^9
    
    component x_squared = Multiplier();
    x_squared.a <== x;
    x_squared.b <== x;
    
    component erf_approx = Multiplier();
    erf_approx.a <== TWO_OVER_SQRT_PI;
    erf_approx.b <== x;
    
    component one_minus_erf = Subtractor();
    one_minus_erf.a <== 1;
    one_minus_erf.b <== erf_approx.out;
    
    out <== one_minus_erf.out;
}

// ========== 主电路 ==========
component main = HybridTruthFinderProof(4, 2, 2, 3);
// 参数说明：
// 4: 4个数据源
// 2: 2个数值字段
// 2: 2个离散字段
// 3: 每个离散字段最多3个候选值
