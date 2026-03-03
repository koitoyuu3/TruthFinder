"""
对比测试：EnhancedTruthFinder vs 原TruthFinder
展示两者的区别和优势
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from truthdiscovery import TruthFinder
from decentralized_truth_discovery.core import EnhancedTruthFinder


def create_test_data():
    """创建测试数据：模拟拜占庭源"""
    # 真实值
    true_values = {
        "stock_A": 100.0,
        "stock_B": 50.0
    }

    # 创建数据：包含一个拜占庭源（website_c提供错误数据）
    dataframe = pd.DataFrame([
        ["website_a", 100.2, "stock_A"],  # 正常源，接近真实值
        ["website_a", 50.1, "stock_B"],
        ["website_b", 99.8, "stock_A"],   # 正常源
        ["website_b", 49.9, "stock_B"],
        ["website_c", 120.0, "stock_A"],  # 拜占庭源，大幅偏离
        ["website_c", 30.0, "stock_B"],   # 拜占庭源
        ["website_d", 100.1, "stock_A"],  # 正常源
        ["website_d", 50.2, "stock_B"],
    ], columns=["website", "fact", "object"])

    return dataframe, true_values


def test_original_truthfinder():
    """测试原TruthFinder（注意：原TruthFinder主要针对字符串fact，数值型可能不稳定）"""
    print("\n" + "=" * 80)
    print("原 TruthFinder 测试")
    print("=" * 80)
    print("注意：原TruthFinder主要设计用于字符串fact，数值型数据可能不稳定")

    dataframe, true_values = create_test_data()

    # 创建简单的implication函数（对于数值型数据，使用距离）
    def implication(f1, f2):
        # 对于数值，使用距离的倒数作为相似度
        try:
            diff = abs(float(f1) - float(f2))
            return 1.0 / (1.0 + diff) if diff > 0 else 1.0
        except:
            return 0.0

    try:
        finder = TruthFinder(implication, dampening_factor=0.8, influence_related=0.6)
        result = finder.train(dataframe.copy(), max_iterations=50, threshold=1e-4)

        print("\n结果:")
        print(result[["website", "fact", "object", "trustworthiness", "fact_confidence"]])

        # 计算RMSE
        rmse_by_object = {}
        for obj in result['object'].unique():
            obj_data = result[result['object'] == obj]
            # 使用fact_confidence作为权重计算加权平均
            weighted_sum = (obj_data['fact'] * obj_data['fact_confidence']).sum()
            total_weight = obj_data['fact_confidence'].sum()
            if total_weight > 0:
                estimated = weighted_sum / total_weight
            else:
                estimated = obj_data['fact'].mean()
            true_val = true_values[obj]
            rmse_by_object[obj] = abs(estimated - true_val)

        print(f"\n估计误差 (|估计值 - 真实值|):")
        for obj, error in rmse_by_object.items():
            print(f"  {obj}: {error:.4f}")

        print(f"\n网站可信度:")
        for website in result['website'].unique():
            trust = result[result['website'] == website]['trustworthiness'].iloc[0]
            print(f"  {website}: {trust:.4f}")

        return result, rmse_by_object
    except Exception as e:
        print(f"\n原TruthFinder处理数值型数据时出错: {e}")
        print("这是预期的，因为原TruthFinder主要设计用于字符串fact")
        return None, {}


def test_enhanced_truthfinder():
    """测试EnhancedTruthFinder"""
    print("\n" + "=" * 80)
    print("EnhancedTruthFinder 测试")
    print("=" * 80)

    dataframe, true_values = create_test_data()

    finder = EnhancedTruthFinder(
        alpha=0.9,
        beta=100.0,
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5
    )

    result = finder.train(dataframe.copy())

    print("\n结果:")
    print(result[["website", "fact", "object",
                  "source_reliability", "error_metric",
                  "consistency_metric", "global_truth"]])

    # 计算RMSE
    rmse_by_object = {}
    for obj in result['object'].unique():
        estimated = result[result['object'] == obj]['global_truth'].iloc[0]
        true_val = true_values[obj]
        rmse_by_object[obj] = abs(estimated - true_val)

    print(f"\n估计误差 (|估计值 - 真实值|):")
    for obj, error in rmse_by_object.items():
        print(f"  {obj}: {error:.4f}")

    print(f"\n源可靠性详情:")
    reliability_info = finder.get_source_reliability()
    for website, metrics in reliability_info.items():
        print(f"  {website}:")
        print(f"    可靠性 (r): {metrics['r']:.4f}")
        print(f"    误差 (e): {metrics['e']:.4f}")
        print(f"    一致性 (k): {metrics['k']:.4f}")

    return result, rmse_by_object


def main():
    """主函数"""
    print("=" * 80)
    print("EnhancedTruthFinder vs 原TruthFinder 对比测试")
    print("=" * 80)
    print("\n测试场景：包含一个拜占庭源（website_c提供大幅偏离的数据）")
    print("真实值: stock_A=100.0, stock_B=50.0")
    print("website_c提供: stock_A=120.0, stock_B=30.0 (明显错误)")

    # 测试原TruthFinder
    result1, rmse1 = test_original_truthfinder()

    # 测试EnhancedTruthFinder
    result2, rmse2 = test_enhanced_truthfinder()

    # 对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)

    if rmse1:
        print("\n1. 估计精度对比:")
        print("   原TruthFinder:")
        for obj, error in rmse1.items():
            print(f"     {obj}: {error:.4f}")
        print("   EnhancedTruthFinder:")
        for obj, error in rmse2.items():
            print(f"     {obj}: {error:.4f}")
    else:
        print("\n1. 估计精度（EnhancedTruthFinder）:")
        for obj, error in rmse2.items():
            print(f"     {obj}: {error:.4f}")

    print("\n2. 拜占庭源检测:")
    if result1 is not None:
        print("   原TruthFinder:")
        website_c_trust1 = result1[result1['website'] == 'website_c']['trustworthiness'].iloc[0]
        print(f"     website_c可信度: {website_c_trust1:.4f}")
    print("   EnhancedTruthFinder:")
    website_c_reliability = result2[result2['website'] == 'website_c']['source_reliability'].iloc[0]
    website_c_error = result2[result2['website'] == 'website_c']['error_metric'].iloc[0]
    website_c_consistency = result2[result2['website'] == 'website_c']['consistency_metric'].iloc[0]
    print(f"     website_c可靠性: {website_c_reliability:.4f}")
    print(f"     website_c误差: {website_c_error:.4f}")
    print(f"     website_c一致性: {website_c_consistency:.4f}")

    print("\n3. 关键优势:")
    print("   - EnhancedTruthFinder通过误差和一致性度量，能更清晰地识别拜占庭源")
    print("   - 提供更详细的源质量信息（误差、一致性）")
    print("   - 支持增量处理，可以持续追踪源的历史表现")
    print("   - 基于DECENTRUTH论文的公式，具有更强的理论保证")


if __name__ == '__main__':
    main()

