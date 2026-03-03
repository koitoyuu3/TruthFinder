"""
EnhancedTruthFinder使用示例（离散类型数据）
演示如何使用基于DECENTRUTH公式的增强版TruthFinder处理离散类型（字符串）数据
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from decentralized_truth_discovery.core import EnhancedTruthFinder

# 设置pandas显示选项，确保显示所有列（避免省略号）
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)         # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制每列的最大宽度


def main():
    """基本使用示例（离散类型数据）"""

    print("=" * 80)
    print("EnhancedTruthFinder 示例 - 离散类型数据")
    print("=" * 80)

    # 创建示例数据（离散型fact - 字符串）
    # 模拟：多个网站提供科学理论的发现者信息
    # 真实情况：相对论的发现者是Einstein，万有引力的发现者是Newton
    dataframe = pd.DataFrame([
        ["website_a", "Einstein", "Special relativity"],      # website, fact(发现者), object(理论)
        ["website_a", "Newton", "Universal gravitation"],
        ["website_b", "Einstein", "Special relativity"],
        ["website_b", "Newton", "Universal gravitation"],
        ["website_c", "Newton", "Special relativity"],       # website_c提供错误数据（应该是Einstein）
        ["website_c", "Galilei", "Universal gravitation"],   # website_c提供错误数据（应该是Newton）
        ["website_d", "Einstein", "Special relativity"],
        ["website_d", "Newton", "Universal gravitation"],
    ], columns=["website", "fact", "object"])

    print("\n1. 初始数据:")
    print(dataframe)

    # 创建EnhancedTruthFinder实例
    print("\n2. 初始化EnhancedTruthFinder...")
    finder = EnhancedTruthFinder(
        alpha=0.9,           # 衰减因子
        beta=100.0,          # 尺度因子
        convergence_threshold=1e-6,
        max_iterations=200,
        initial_reliability=0.5
    )

    # 训练模型（批量处理）
    print("\n3. 训练模型（批量处理）...")
    result = finder.train(dataframe)

    print("\n4. 训练结果:")
    print(result[["website", "fact", "object",
                  "source_reliability", "error_metric",
                  "consistency_metric", "global_truth"]])

    # 显示源可靠性信息
    print("\n5. 源可靠性详情:")
    reliability_info = finder.get_source_reliability()
    for website, metrics in reliability_info.items():
        print(f"  {website}:")
        print(f"    可靠性 (r): {metrics['r']:.6f}")
        print(f"    误差 (e): {metrics['e']:.6f}")
        print(f"    一致性 (k): {metrics['k']:.6f}")

    # 显示每个对象的真值估计
    print("\n5.1 真值估计结果:")
    for obj in result['object'].unique():
        obj_data = result[result['object'] == obj]
        truth_value = obj_data['global_truth'].iloc[0]
        print(f"  {obj}: {truth_value}")

    # 增量处理示例
    print("\n" + "=" * 80)
    print("增量处理示例（离散类型数据）")
    print("=" * 80)

    # 重置历史（可选）
    finder.reset_history()

    # 第一批数据
    print("\n6. 处理第一批数据...")
    batch1 = pd.DataFrame([
        ["website_a", "Einstein", "Special relativity"],
        ["website_b", "Einstein", "Special relativity"],
        ["website_c", "Newton", "Special relativity"],  # website_c提供错误数据
    ], columns=["website", "fact", "object"])

    result1 = finder.process_batch(batch1, epoch=0)
    print(result1[["website", "fact", "object", "source_reliability", "global_truth"]])

    # 第二批数据
    print("\n7. 处理第二批数据...")
    batch2 = pd.DataFrame([
        ["website_a", "Newton", "Universal gravitation"],
        ["website_b", "Newton", "Universal gravitation"],
        ["website_c", "Galilei", "Universal gravitation"],  # website_c提供错误数据（应该是Newton）
    ], columns=["website", "fact", "object"])

    result2 = finder.process_batch(batch2, epoch=1)
    print(result2[["website", "fact", "object", "source_reliability", "global_truth"]])

    # 观察源可靠性的变化
    print("\n8. 源可靠性变化（增量处理）:")
    reliability_info = finder.get_source_reliability()
    for website, metrics in reliability_info.items():
        print(f"  {website}: 可靠性 = {metrics['r']:.6f}, "
              f"误差 = {metrics['e']:.6f}, "
              f"一致性 = {metrics['k']:.6f}")

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
    print("\n关键观察:")
    print("- website_c提供的数据与大多数源不一致，其可靠性应该较低")
    print("- 对于离散类型数据，算法使用加权投票选择真值（权重和最大的值）")
    print("- 误差度量反映不匹配率（0=完全匹配，1=完全不匹配）")
    print("- 增量处理可以持续更新源可靠性")


if __name__ == '__main__':
    main()

