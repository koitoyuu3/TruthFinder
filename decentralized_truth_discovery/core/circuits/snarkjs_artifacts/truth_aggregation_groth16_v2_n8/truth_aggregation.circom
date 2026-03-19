template Sum(n) {
    signal input in[n];
    signal output out;
    signal partial[n+1];

    partial[0] <== 0;
    for (var i = 0; i < n; i++) {
        partial[i+1] <== partial[i] + in[i];
    }
    out <== partial[n];
}

template TruthAggregationProof(n) {
    signal input values[n];
    signal input weights[n];
    signal input claimed_weighted_sum;
    signal input claimed_total_weight;
    signal input statement_hash;

    signal output public_statement_hash;
    signal output public_weighted_sum;
    signal output public_total_weight;
    signal output is_valid;

    signal products[n];
    for (var i = 0; i < n; i++) {
        products[i] <== values[i] * weights[i];
    }

    component sum_products = Sum(n);
    component sum_weights = Sum(n);
    for (var j = 0; j < n; j++) {
        sum_products.in[j] <== products[j];
        sum_weights.in[j] <== weights[j];
    }

    claimed_weighted_sum === sum_products.out;
    claimed_total_weight === sum_weights.out;

    // 绑定哈希（线性承诺）：把私有输入与公开 statement_hash 绑定
    signal hash_terms[18];
    component sum_hash = Sum(18);
    for (var k = 0; k < n; k++) {
        hash_terms[2*k] <== values[k] * (k + 1);
        hash_terms[2*k + 1] <== weights[k] * (k + 1001);
    }
    hash_terms[16] <== claimed_weighted_sum;
    hash_terms[17] <== claimed_total_weight;
    for (var t = 0; t < 18; t++) {
        sum_hash.in[t] <== hash_terms[t];
    }
    statement_hash === sum_hash.out;

    public_statement_hash <== statement_hash;
    public_weighted_sum <== claimed_weighted_sum;
    public_total_weight <== claimed_total_weight;
    is_valid <== 1;
}

component main = TruthAggregationProof(8);