"""
混合数据类型示例（文本+数值）
演示如何使用EnhancedTruthFinder处理包含数值和离散信息的文本数据
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from decentralized_truth_discovery.core import (
    EnhancedTruthFinder, 
    SenteTruthFinder,
    BasicTruthFinder,
    create_implication_function_from_texts
)


def main():
    """混合数据类型使用示例"""

    print("=" * 80)
    print("EnhancedTruthFinder 混合数据类型示例")
    print("=" * 80)

    # 创建示例数据（4个Agent提供的文本，包含数值和离散信息）
    # 真实情况：Agent_1和Agent_2提供准确信息，Agent_3和Agent_4有部分错误
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

    print("\n1. 初始数据:")
    for idx, row in dataframe.iterrows():
        print(f"\n{row['website']}:")
        print(f"  {row['fact'][:100]}...")  # 只显示前100个字符

    # 创建EnhancedTruthFinder实例（启用混合模式）
    print("\n2. 初始化EnhancedTruthFinder（混合模式）...")
    finder_hybrid = EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        numeric_weight=0.5,  # 数值和离散误差各占50%
        enable_hybrid_mode=True  # 启用混合模式
    )

    # 训练模型（批量处理）
    print("\n3. 训练模型（混合模式）...")
    result_hybrid = finder_hybrid.train(dataframe.copy())

    print("\n4. 训练结果（混合模式）:")
    print(result_hybrid[["website", "source_reliability", "error_metric", "consistency_metric"]])

    # 显示真值估计
    print("\n5. 真值估计（混合模式）:")
    for obj in result_hybrid['object'].unique():
        obj_data = result_hybrid[result_hybrid['object'] == obj]
        truth_value = obj_data['global_truth'].iloc[0]
        print(f"\n对象: {obj}")
        if isinstance(truth_value, dict):
            print(f"  数值信息: {truth_value.get('numeric', {})}")
            print(f"  离散信息: {truth_value.get('discrete', {})}")
        else:
            print(f"  真值: {truth_value}")

    # 显示源可靠性信息
    print("\n6. 源可靠性详情（混合模式）:")
    reliability_info = finder_hybrid.get_source_reliability()
    for website, metrics in reliability_info.items():
        print(f"  {website}:")
        print(f"    可靠性 (r): {metrics['r']:.6f}")
        print(f"    误差 (e): {metrics['e']:.6f}")
        print(f"    一致性 (k): {metrics['k']:.6f}")

    # ========== 对比：使用原有方法（非混合模式）==========
    print("\n" + "=" * 80)
    print("对比：原有方法（非混合模式，按离散类型处理）")
    print("=" * 80)

    finder_original = EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5,
        enable_hybrid_mode=False  # 不启用混合模式
    )

    print("\n7. 训练模型（原有方法）...")
    result_original = finder_original.train(dataframe.copy())

    print("\n8. 训练结果（原有方法）:")
    print(result_original[["website", "source_reliability", "error_metric", "consistency_metric"]])

    print("\n9. 源可靠性详情（原有方法）:")
    reliability_info_original = finder_original.get_source_reliability()
    for website, metrics in reliability_info_original.items():
        print(f"  {website}:")
        print(f"    可靠性 (r): {metrics['r']:.6f}")
        print(f"    误差 (e): {metrics['e']:.6f}")
        print(f"    一致性 (k): {metrics['k']:.6f}")

    # ========== 对比总结 ==========
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print("\n可靠性对比:")
    for website in reliability_info.keys():
        r_hybrid = reliability_info[website]['r']
        r_original = reliability_info_original[website]['r']
        improvement = (r_hybrid - r_original) / r_original * 100 if r_original > 0 else 0
        print(f"  {website}:")
        print(f"    混合模式: {r_hybrid:.6f}")
        print(f"    原有方法: {r_original:.6f}")
        print(f"    变化: {improvement:+.2f}%")

    print("\n关键观察:")
    print("- 混合模式能够从文本中提取数值和离散信息，更准确地评估源可靠性")
    print("- Agent_1和Agent_2提供准确信息，应该具有较高的可靠性")
    print("- Agent_3和Agent_4有错误信息，应该具有较低的可靠性")
    print("- 混合模式通过多维度误差计算，能够更细致地评估每个源的表现")

    # ========== 对比：使用SenteTruth方法 ==========
    print("\n" + "=" * 80)
    print("对比：SenteTruth方法（基于论文的语义相似度方法）")
    print("=" * 80)

    finder_sente = SenteTruthFinder(
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_credibility=0.5,
        use_sbert=True  # 尝试使用SBERT，如果没有则回退到TF-IDF
    )

    print("\n10. 训练模型（SenteTruth方法）...")
    result_sente = finder_sente.train(dataframe.copy())

    print("\n11. 训练结果（SenteTruth方法）:")
    print(result_sente[["website", "source_reliability", "global_truth"]])

    print("\n12. 源可靠性详情（SenteTruth方法）:")
    reliability_info_sente = finder_sente.get_source_reliability()
    for website, metrics in reliability_info_sente.items():
        print(f"  {website}:")
        print(f"    可信度 (credibility): {metrics['credibility']:.6f}")

    # ========== 三种方法对比总结 ==========
    print("\n" + "=" * 80)
    print("三种方法对比总结")
    print("=" * 80)
    print("\n可靠性/可信度对比:")
    for website in reliability_info.keys():
        r_hybrid = reliability_info[website]['r']
        r_original = reliability_info_original[website]['r']
        c_sente = reliability_info_sente[website]['credibility']
        
        print(f"  {website}:")
        print(f"    混合模式 (EnhancedTruthFinder): {r_hybrid:.6f}")
        print(f"    原有方法 (EnhancedTruthFinder, 非混合): {r_original:.6f}")
        print(f"    SenteTruth方法: {c_sente:.6f}")
        
        # 计算相对于混合模式的变化
        diff_hybrid_sente = (c_sente - r_hybrid) / r_hybrid * 100 if r_hybrid > 0 else 0
        print(f"    SenteTruth vs 混合模式: {diff_hybrid_sente:+.2f}%")
        
        # 验证：SenteTruth的性能不应该优于混合模式
        if c_sente > r_hybrid:
            print(f"    ⚠️  注意：SenteTruth可信度高于混合模式，这可能不符合预期")
        else:
            print(f"    ✓ SenteTruth可信度低于或等于混合模式（符合预期）")

    print("\n方法特点说明:")
    print("- 混合模式（EnhancedTruthFinder + hybrid_mode=True）:")
    print("  * 从文本中提取结构化信息（数值+离散）")
    print("  * 使用多维度误差计算（数值误差+离散误差）")
    print("  * 能够更精确地评估源可靠性")
    print("- SenteTruth方法:")
    print("  * 直接使用文本的语义相似度")
    print("  * 不提取结构化信息")
    print("  * 适用于LLM生成的文本数据")
    print("  * 性能可能不如混合模式（因为缺少结构化信息）")
    print("- 预期结果:")
    print("  * 混合模式应该能更准确地区分可靠和不可靠的源")
    print("  * SenteTruth方法应该能工作，但性能可能略低于混合模式")

    # ========== 对比：使用BasicTruthFinder方法 ==========
    print("\n" + "=" * 80)
    print("对比：BasicTruthFinder方法（基于原始TruthFinder论文）")
    print("=" * 80)

    # 创建implication函数（使用TF-IDF + 余弦相似度）
    print("\n13. 创建implication函数（TF-IDF + 余弦相似度）...")
    texts = [str(row['fact']) for _, row in dataframe.iterrows()]
    implication_func = create_implication_function_from_texts(texts)

    finder_basic = BasicTruthFinder(
        implication=implication_func,
        dampening_factor=0.8,  # 与main.py中的示例保持一致
        influence_related=0.6  # 与main.py中的示例保持一致
    )

    print("\n14. 训练模型（BasicTruthFinder方法）...")
    result_basic = finder_basic.train(
        dataframe.copy(),
        max_iterations=200,
        threshold=1e-6,
        initial_trustworthiness=0.9  # 与truthdiscovery.py中的默认值保持一致
    )

    print("\n15. 训练结果（BasicTruthFinder方法）:")
    print(result_basic[["website", "trustworthiness", "fact_confidence", "object"]])

    print("\n16. 源可靠性详情（BasicTruthFinder方法）:")
    # 从DataFrame中提取trustworthiness
    reliability_info_basic = {}
    for website in result_basic['website'].unique():
        website_data = result_basic[result_basic['website'] == website]
        trustworthiness = website_data['trustworthiness'].iloc[0]
        reliability_info_basic[website] = {'trustworthiness': trustworthiness}
        print(f"  {website}:")
        print(f"    可靠性 (trustworthiness): {trustworthiness:.6f}")

    # ========== 四种方法对比总结 ==========
    print("\n" + "=" * 80)
    print("四种方法对比总结")
    print("=" * 80)
    print("\n可靠性/可信度对比:")
    for website in reliability_info.keys():
        r_hybrid = reliability_info[website]['r']
        r_original = reliability_info_original[website]['r']
        c_sente = reliability_info_sente[website]['credibility']
        t_basic = reliability_info_basic[website]['trustworthiness']
        
        print(f"  {website}:")
        print(f"    混合模式 (EnhancedTruthFinder): {r_hybrid:.6f}")
        print(f"    原有方法 (EnhancedTruthFinder, 非混合): {r_original:.6f}")
        print(f"    SenteTruth方法: {c_sente:.6f}")
        print(f"    BasicTruthFinder方法: {t_basic:.6f}")
        
        # 计算相对于混合模式的变化
        diff_hybrid_basic = (t_basic - r_hybrid) / r_hybrid * 100 if r_hybrid > 0 else 0
        print(f"    BasicTruthFinder vs 混合模式: {diff_hybrid_basic:+.2f}%")

    print("\n方法特点说明:")
    print("- 混合模式（EnhancedTruthFinder + hybrid_mode=True）:")
    print("  * 从文本中提取结构化信息（数值+离散）")
    print("  * 使用多维度误差计算（数值误差+离散误差）")
    print("  * 能够更精确地评估源可靠性")
    print("- SenteTruth方法:")
    print("  * 直接使用文本的语义相似度")
    print("  * 不提取结构化信息")
    print("  * 适用于LLM生成的文本数据")
    print("  * 性能可能不如混合模式（因为缺少结构化信息）")
    print("- BasicTruthFinder方法（原始TruthFinder）:")
    print("  * 基于原始TruthFinder论文实现")
    print("  * 使用TF-IDF + 余弦相似度计算事实之间的关联")
    print("  * 迭代更新trustworthiness和fact_confidence")
    print("  * 考虑事实之间的关联（implication）")
    print("  * 适用于离散文本数据")
    print("- 预期结果:")
    print("  * 混合模式应该能更准确地区分可靠和不可靠的源")
    print("  * BasicTruthFinder和SenteTruth方法应该能工作，但性能可能不如混合模式")


if __name__ == '__main__':
    main()
