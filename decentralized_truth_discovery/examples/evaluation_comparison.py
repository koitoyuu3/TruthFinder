"""
评估指标和对比实验
对比混合模式（新方法）和原有方法的性能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from decentralized_truth_discovery.core import EnhancedTruthFinder


def calculate_metrics(finder, dataframe, result_df, true_reliability=None):
    """
    计算评估指标
    
    Args:
        finder: EnhancedTruthFinder实例
        dataframe: 原始数据
        result_df: 训练结果DataFrame
        true_reliability: 真实的源可靠性（用于计算排名准确性）
                        格式：{'Agent_1': 0.95, 'Agent_2': 0.90, ...}
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 1. 综合准确率（Combined Accuracy）
    # 对于混合模式，需要从真值中提取信息进行对比
    if finder.data_type == 'hybrid':
        # 计算数值准确率和离散准确率
        numeric_accuracies = []
        discrete_accuracies = []
        
        for obj in result_df['object'].unique():
            obj_data = result_df[result_df['object'] == obj]
            truth_value = obj_data['global_truth'].iloc[0]
            
            if isinstance(truth_value, dict):
                # 对每个源，计算其与真值的相似度
                for _, row in obj_data.iterrows():
                    source_text = str(row['fact'])
                    extracted = finder._extract_structured_info(source_text)
                    
                    # 数值准确率：计算提取的数值与真值的匹配度
                    if 'numeric' in truth_value and truth_value['numeric']:
                        numeric_errors = []
                        for key, truth_val in truth_value['numeric'].items():
                            if key in extracted['numeric']:
                                extracted_val = extracted['numeric'][key]
                                if abs(truth_val) > 1e-6:
                                    relative_error = abs(extracted_val - truth_val) / abs(truth_val)
                                    numeric_errors.append(1.0 - min(1.0, relative_error))
                        if numeric_errors:
                            numeric_accuracies.append(np.mean(numeric_errors))
                    
                    # 离散准确率：计算语义相似度
                    if 'discrete' in truth_value and truth_value['discrete']:
                        truth_text_parts = []
                        if 'company' in truth_value['discrete']:
                            truth_text_parts.append(truth_value['discrete']['company'])
                        if 'symbol' in truth_value['discrete']:
                            truth_text_parts.append(truth_value['discrete']['symbol'])
                        truth_text_summary = ' '.join(truth_text_parts)
                        
                        if truth_text_summary:
                            similarity = finder._calculate_semantic_similarity(
                                source_text, truth_text_summary
                            )
                            discrete_accuracies.append(similarity)
        
        numeric_accuracy = np.mean(numeric_accuracies) if numeric_accuracies else 0.0
        discrete_accuracy = np.mean(discrete_accuracies) if discrete_accuracies else 0.0
        combined_accuracy = finder.numeric_weight * numeric_accuracy + (1 - finder.numeric_weight) * discrete_accuracy
        
        metrics['numeric_accuracy'] = numeric_accuracy
        metrics['discrete_accuracy'] = discrete_accuracy
        metrics['combined_accuracy'] = combined_accuracy
    else:
        # 对于非混合模式，使用简单的准确率
        # 这里简化处理，主要关注可靠性评估
        metrics['combined_accuracy'] = 0.5  # 占位符
    
    # 2. 源可靠性排名准确性（Spearman相关系数）
    if true_reliability:
        reliability_info = finder.get_source_reliability()
        predicted_ranks = []
        true_ranks = []
        
        # 获取所有源的可靠性
        predicted_reliability = {w: reliability_info[w]['r'] for w in reliability_info.keys()}
        
        # 计算排名
        predicted_sorted = sorted(predicted_reliability.items(), key=lambda x: x[1], reverse=True)
        true_sorted = sorted(true_reliability.items(), key=lambda x: x[1], reverse=True)
        
        # 构建排名列表
        for website in predicted_reliability.keys():
            if website in true_reliability:
                pred_rank = next(i for i, (w, _) in enumerate(predicted_sorted) if w == website)
                true_rank = next(i for i, (w, _) in enumerate(true_sorted) if w == website)
                predicted_ranks.append(pred_rank)
                true_ranks.append(true_rank)
        
        if len(predicted_ranks) > 1:
            spearman_corr, p_value = spearmanr(predicted_ranks, true_ranks)
            metrics['ranking_accuracy'] = spearman_corr
            metrics['ranking_p_value'] = p_value
        else:
            metrics['ranking_accuracy'] = 0.0
            metrics['ranking_p_value'] = 1.0
    else:
        metrics['ranking_accuracy'] = None
    
    # 3. 恶意源检测F1分数
    # 假设可靠性低于0.6的源为恶意源
    reliability_info = finder.get_source_reliability()
    threshold = 0.6
    
    if true_reliability:
        true_malicious = [w for w, r in true_reliability.items() if r < threshold]
        pred_malicious = [w for w, info in reliability_info.items() if info['r'] < threshold]
        
        true_positives = len(set(true_malicious) & set(pred_malicious))
        false_positives = len(set(pred_malicious) - set(true_malicious))
        false_negatives = len(set(true_malicious) - set(pred_malicious))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['malicious_detection_precision'] = precision
        metrics['malicious_detection_recall'] = recall
        metrics['malicious_detection_f1'] = f1_score
    else:
        metrics['malicious_detection_f1'] = None
    
    # 4. 收敛迭代次数（从训练过程中获取）
    metrics['convergence_iterations'] = getattr(finder, 'last_iteration_count', finder.max_iterations)
    
    return metrics


def main():
    """主函数：运行对比实验"""
    
    print("=" * 80)
    print("评估指标和对比实验")
    print("=" * 80)
    
    # 创建测试数据（4个Agent）
    dataframe = pd.DataFrame([
        ["Agent_1", 
         "A new Agilent Technologies, Inc. (Symbol: A) option chain has opened with a $125 put priced at $4.50 and a $150 call at $8.10, offering opportunities for premium capturing via selling puts or opening covered calls respectively.",
         "Agilent_Options"],
        ["Agent_2",
         "Newly trading August 2024 options for Agilent Technologies Inc., including a $125 put with $4 premium and a $150 call at $8.10, offer opportunities to sellers of puts or calls for higher premiums due to upcoming expiration, while noting slightly lower implied volatilities compared to similar contracts.",
         "Agilent_Options"],
        ["Agent_3",
         "A recent round of Agilent Technologies options, trading at 241 days until expiry, now offers potential upside for option sellers and lower cost entry points for call buyers. The August $150 put has a bid of $8.10 with an implied volatility of 32%, and the August $125 call comes in at $9.10 with a premium of 13.75%.",
         "Agilent_Options"],
        ["Agent_4",
         "Ticker A is leading the tech stock rally, trading at all-time highs!",
         "Agilent_Options"],
    ], columns=["website", "fact", "object"])
    
    # 定义真实的源可靠性（用于评估）
    # Agent_1和Agent_2提供准确信息，可靠性高
    # Agent_3有部分错误（价格和百分比错误），可靠性中等
    # Agent_4信息不完整且不相关，可靠性低
    true_reliability = {
        'Agent_1': 0.95,  # 准确
        'Agent_2': 0.90,  # 准确
        'Agent_3': 0.60,  # 部分错误
        'Agent_4': 0.30   # 不相关
    }
    
    print("\n真实源可靠性:")
    for agent, rel in true_reliability.items():
        print(f"  {agent}: {rel:.2f}")
    
    # ========== 测试新方法（混合模式）==========
    print("\n" + "=" * 80)
    print("测试新方法（混合模式）")
    print("=" * 80)
    
    finder_hybrid = EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        numeric_weight=0.5,
        enable_hybrid_mode=True
    )
    
    result_hybrid = finder_hybrid.train(dataframe.copy())
    
    print("\n新方法 - 源可靠性:")
    reliability_hybrid = finder_hybrid.get_source_reliability()
    for agent, info in reliability_hybrid.items():
        print(f"  {agent}: {info['r']:.6f} (误差: {info['e']:.6f}, 一致性: {info['k']:.6f})")
    
    metrics_hybrid = calculate_metrics(finder_hybrid, dataframe, result_hybrid, true_reliability)
    
    print("\n新方法 - 评估指标:")
    print(f"  综合准确率: {metrics_hybrid.get('combined_accuracy', 0):.6f}")
    if metrics_hybrid.get('numeric_accuracy') is not None:
        print(f"  数值准确率: {metrics_hybrid.get('numeric_accuracy', 0):.6f}")
        print(f"  离散准确率: {metrics_hybrid.get('discrete_accuracy', 0):.6f}")
    if metrics_hybrid.get('ranking_accuracy') is not None:
        print(f"  排名准确性 (Spearman): {metrics_hybrid.get('ranking_accuracy', 0):.6f}")
    if metrics_hybrid.get('malicious_detection_f1') is not None:
        print(f"  恶意源检测F1: {metrics_hybrid.get('malicious_detection_f1', 0):.6f}")
    print(f"  收敛迭代次数: {metrics_hybrid.get('convergence_iterations', 0)}")
    
    # ========== 测试原方法（非混合模式）==========
    print("\n" + "=" * 80)
    print("测试原方法（非混合模式）")
    print("=" * 80)
    
    finder_original = EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        enable_hybrid_mode=False
    )
    
    result_original = finder_original.train(dataframe.copy())
    
    print("\n原方法 - 源可靠性:")
    reliability_original = finder_original.get_source_reliability()
    for agent, info in reliability_original.items():
        print(f"  {agent}: {info['r']:.6f} (误差: {info['e']:.6f}, 一致性: {info['k']:.6f})")
    
    metrics_original = calculate_metrics(finder_original, dataframe, result_original, true_reliability)
    
    print("\n原方法 - 评估指标:")
    print(f"  综合准确率: {metrics_original.get('combined_accuracy', 0):.6f}")
    if metrics_original.get('ranking_accuracy') is not None:
        print(f"  排名准确性 (Spearman): {metrics_original.get('ranking_accuracy', 0):.6f}")
    if metrics_original.get('malicious_detection_f1') is not None:
        print(f"  恶意源检测F1: {metrics_original.get('malicious_detection_f1', 0):.6f}")
    print(f"  收敛迭代次数: {metrics_original.get('convergence_iterations', 0)}")
    
    # ========== 对比总结 ==========
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    
    print("\n指标对比:")
    print(f"{'指标':<30} {'新方法':<15} {'原方法':<15} {'提升':<15}")
    print("-" * 75)
    
    if metrics_hybrid.get('combined_accuracy') is not None and metrics_original.get('combined_accuracy') is not None:
        ca_hybrid = metrics_hybrid['combined_accuracy']
        ca_original = metrics_original['combined_accuracy']
        ca_improvement = (ca_hybrid - ca_original) / ca_original * 100 if ca_original > 0 else 0
        print(f"{'综合准确率':<30} {ca_hybrid:<15.6f} {ca_original:<15.6f} {ca_improvement:+.2f}%")
    
    if metrics_hybrid.get('ranking_accuracy') is not None and metrics_original.get('ranking_accuracy') is not None:
        ra_hybrid = metrics_hybrid['ranking_accuracy']
        ra_original = metrics_original['ranking_accuracy']
        ra_improvement = (ra_hybrid - ra_original) / abs(ra_original) * 100 if ra_original != 0 else 0
        print(f"{'排名准确性 (Spearman)':<30} {ra_hybrid:<15.6f} {ra_original:<15.6f} {ra_improvement:+.2f}%")
    
    if metrics_hybrid.get('malicious_detection_f1') is not None and metrics_original.get('malicious_detection_f1') is not None:
        f1_hybrid = metrics_hybrid['malicious_detection_f1']
        f1_original = metrics_original['malicious_detection_f1']
        f1_improvement = (f1_hybrid - f1_original) / f1_original * 100 if f1_original > 0 else 0
        print(f"{'恶意源检测F1':<30} {f1_hybrid:<15.6f} {f1_original:<15.6f} {f1_improvement:+.2f}%")
    
    if metrics_hybrid.get('convergence_iterations') is not None and metrics_original.get('convergence_iterations') is not None:
        iter_hybrid = metrics_hybrid['convergence_iterations']
        iter_original = metrics_original['convergence_iterations']
        iter_improvement = (iter_original - iter_hybrid) / iter_original * 100 if iter_original > 0 else 0
        print(f"{'收敛迭代次数':<30} {iter_hybrid:<15} {iter_original:<15} {iter_improvement:+.2f}% (减少)")
    
    # 判断是否满足要求
    print("\n" + "=" * 80)
    print("结果判断")
    print("=" * 80)
    
    all_improved = True
    improvements = []
    
    if metrics_hybrid.get('ranking_accuracy') is not None and metrics_original.get('ranking_accuracy') is not None:
        ra_hybrid = metrics_hybrid['ranking_accuracy']
        ra_original = metrics_original['ranking_accuracy']
        if ra_hybrid < ra_original:
            all_improved = False
            improvements.append("排名准确性未提升")
        elif ra_hybrid > ra_original:
            improvements.append("✓ 排名准确性提升")
        else:
            # 如果排名准确性相同，检查评估的可靠性是否更接近真实值
            # 计算与真实可靠性的均方误差（MSE）
            if true_reliability:
                reliability_hybrid = finder_hybrid.get_source_reliability()
                reliability_original = finder_original.get_source_reliability()
                
                mse_hybrid = np.mean([(reliability_hybrid.get(w, {}).get('r', 0) - true_reliability.get(w, 0))**2 
                                     for w in true_reliability.keys()])
                mse_original = np.mean([(reliability_original.get(w, {}).get('r', 0) - true_reliability.get(w, 0))**2 
                                      for w in true_reliability.keys()])
                
                if mse_hybrid < mse_original:
                    improvements.append(f"✓ 排名准确性相同，但评估更准确（MSE: {mse_hybrid:.6f} vs {mse_original:.6f}）")
                elif mse_hybrid > mse_original:
                    all_improved = False
                    improvements.append(f"排名准确性相同，但评估准确性较低（MSE: {mse_hybrid:.6f} vs {mse_original:.6f}）")
                else:
                    improvements.append("✓ 排名准确性相同")
            else:
                improvements.append("✓ 排名准确性相同")
    
    if metrics_hybrid.get('malicious_detection_f1') is not None and metrics_original.get('malicious_detection_f1') is not None:
        f1_hybrid = metrics_hybrid['malicious_detection_f1']
        f1_original = metrics_original['malicious_detection_f1']
        if f1_hybrid <= f1_original:
            all_improved = False
            improvements.append("恶意源检测F1未提升")
        else:
            improvements.append("✓ 恶意源检测F1提升")
    
    for imp in improvements:
        print(f"  {imp}")
    
    if all_improved:
        print("\n✓ 新方法在所有关键指标上都优于原方法！")
    else:
        print("\n✗ 新方法在某些指标上未优于原方法，需要优化。")
    
    return all_improved, metrics_hybrid, metrics_original


if __name__ == '__main__':
    success, metrics_h, metrics_o = main()
    exit(0 if success else 1)
