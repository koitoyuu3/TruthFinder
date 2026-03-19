"""
Basic TruthFinder
基于原始TruthFinder论文实现的真值发现算法

这个文件实现了基础的TruthFinder算法，严格按照truthdiscovery.py中的方法实现。
主要功能是：
1. 从多个可能不可靠的数据源中发现真实值
2. 自动评估每个数据源的可靠性（trustworthiness）
3. 计算每个事实的置信度（fact_confidence）
4. 使用implication函数处理事实之间的关联

核心公式：
- 公式3：trustworthiness_score = -log(1 - trustworthiness)
- 公式5：fact_confidence = Σ trustworthiness_score(website)
- 公式6：fact_confidence = rho * Σ(fact_confidence * implication) + fact_confidence
- 公式：fact_confidence = sigmoid(gamma * fact_confidence)
- 公式：trustworthiness = mean(fact_confidence)

数据结构：DataFrame with columns: website, fact, object
- website: 数据源标识
- fact: 输入值（文本）
- object: 对象标识
"""
import numpy as np
from numpy.linalg import norm
import math
import pandas as pd
from typing import Dict, Optional, Callable
from sklearn.feature_extraction.text import TfidfVectorizer

from .zk_proof import ZKProofEngine, attach_grouped_proofs
from .text_vectorization import build_tfidf_vectorizer


def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + math.exp(-x))


class BasicTruthFinder:
    """
    基础TruthFinder真值发现算法
    
    严格按照原始TruthFinder论文和truthdiscovery.py中的实现。
    这个类的主要作用是：
    - 输入：多个数据源（website）提供的关于不同对象（object）的值（fact）
    - 输出：每个数据源的可靠性评分（trustworthiness）和每个事实的置信度（fact_confidence）
    
    工作流程：
    1. 初始化所有数据源的trustworthiness
    2. 对每个object：
       a. 计算fact_confidence（基于trustworthiness）
       b. 调整fact_confidence（考虑事实之间的关联）
       c. 应用sigmoid函数
    3. 更新trustworthiness（基于fact_confidence的平均值）
    4. 迭代直到收敛
    """

    def __init__(self, 
                 implication: Callable[[str, str], float],
                 dampening_factor: float = 0.3,
                 influence_related: float = 0.5,
                 enable_zk_proof: bool = True,
                 zk_proof_secret: Optional[str] = None):
        """
        初始化BasicTruthFinder
        
        Args:
            implication: 函数，接受两个fact作为参数，返回它们之间的关联度
                        implication(f1, f2) 应该返回 imp(f1 -> f2) ∈ [-1, 1]
                        通常使用相似度函数，如TF-IDF + 余弦相似度
            dampening_factor: 衰减因子 gamma ∈ (0, 1)
                            - 用于sigmoid函数，控制fact_confidence的缩放
                            - 默认0.3
            influence_related: 关联影响因子 rho ∈ [0, 1]
                              - 控制事实之间关联对confidence的影响程度
                              - 默认0.5
            enable_zk_proof: 是否为结果附加零知识证明
            zk_proof_secret: 证明私钥种子（可选）
        """
        assert 0 < dampening_factor < 1, "dampening_factor must be in (0, 1)"
        assert 0 <= influence_related <= 1, "influence_related must be in [0, 1]"
        
        self.implication = implication
        self.dampening_factor = dampening_factor
        self.influence_related = influence_related
        self.enable_zk_proof = enable_zk_proof
        self.last_iteration_count = 0
        self.zk_proof_engine = ZKProofEngine(
            prover_id="BasicTruthFinder",
            secret_seed=zk_proof_secret,
        )

    def calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个事实的置信度（公式5）
        
        fact_confidence = Σ trustworthiness_score(website)
        其中 trustworthiness_score = -log(1 - trustworthiness)  # 公式3
        
        Args:
            df: 单个object的数据子集
            
        Returns:
            更新后的DataFrame（添加了fact_confidence列）
        """
        # 公式3：trustworthiness_score = -log(1 - trustworthiness)
        # 添加边界检查，避免 trustworthiness >= 1.0 导致 math domain error
        def trustworthiness_score(x):
            # 限制 x 的最大值为 0.999，避免 log(0) 或 log(负数)
            x = min(0.999, max(0.001, x))
            return -math.log(1 - x)
        
        # 公式5：对每个fact，累加所有提供该fact的website的trustworthiness_score
        for i, row in df.iterrows():
            # 获取所有提供这个fact的website的trustworthiness
            ts = df.loc[df["fact"] == row["fact"], "trustworthiness"]
            # 计算fact_confidence = Σ trustworthiness_score
            v = sum(trustworthiness_score(t) for t in ts)
            df.at[i, "fact_confidence"] = v
        
        return df

    def adjust_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        调整事实置信度（公式6）
        
        fact_confidence = rho * Σ(fact_confidence * implication) + fact_confidence
        
        考虑事实之间的关联，如果两个事实相关，一个事实的置信度会影响另一个。
        
        Args:
            df: 单个object的数据子集
            
        Returns:
            更新后的DataFrame（更新了fact_confidence列）
        """
        update = {}
        
        for i, row1 in df.iterrows():
            f1 = row1["fact"]
            s = 0
            
            # 对每个不同的fact，计算关联影响
            for j, row2 in df.drop_duplicates("fact").iterrows():
                f2 = row2["fact"]
                if f1 == f2:
                    continue
                # 累加：fact_confidence * implication(f2 -> f1)
                s += row2["fact_confidence"] * self.implication(f2, f1)
            
            # 公式6：fact_confidence = rho * Σ(关联影响) + 原始fact_confidence
            update[i] = self.influence_related * s + row1["fact_confidence"]
        
        # 更新fact_confidence
        for i, row1 in df.iterrows():
            df.at[i, "fact_confidence"] = update[i]
        
        return df

    def compute_fact_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用sigmoid函数计算最终的事实置信度
        
        fact_confidence = sigmoid(gamma * fact_confidence)
        
        Args:
            df: 单个object的数据子集
            
        Returns:
            更新后的DataFrame（更新了fact_confidence列）
        """
        f = lambda x: sigmoid(self.dampening_factor * x)
        for i, row in df.iterrows():
            df.at[i, "fact_confidence"] = f(row["fact_confidence"])
        return df

    def update_fact_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        更新所有事实的置信度
        
        对每个object分别处理：
        1. 计算confidence
        2. 调整confidence（考虑关联）
        3. 应用sigmoid
        
        Args:
            df: 完整数据DataFrame
            
        Returns:
            更新后的DataFrame
        """
        for object_ in df["object"].unique():
            indices = df["object"] == object_
            d = df.loc[indices].copy()
            d = self.calculate_confidence(d)
            d = self.adjust_confidence(d)
            df.loc[indices] = self.compute_fact_confidence(d)
        return df

    def update_website_trustworthiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        更新数据源的可靠性
        
        trustworthiness = mean(fact_confidence)
        对每个website，计算它提供的所有fact的confidence的平均值。
        
        Args:
            df: 完整数据DataFrame
            
        Returns:
            更新后的DataFrame（更新了trustworthiness列）
        """
        for website in df["website"].unique():
            indices = df["website"] == website
            cs = df.loc[indices, "fact_confidence"]
            # trustworthiness = 平均fact_confidence
            # 限制在 [0.001, 0.999] 范围内，避免后续计算中的 math domain error
            trustworthiness = sum(cs) / len(cs)
            trustworthiness = max(0.001, min(0.999, trustworthiness))
            df.loc[indices, "trustworthiness"] = trustworthiness
        return df

    def iteration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行一次迭代
        
        1. 更新fact_confidence
        2. 更新trustworthiness
        
        Args:
            df: 输入DataFrame
            
        Returns:
            更新后的DataFrame
        """
        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df

    def stop_condition(self, t1: pd.Series, t2: pd.Series, threshold: float) -> bool:
        """
        检查是否满足停止条件
        
        当trustworthiness的变化小于阈值时，认为已收敛。
        
        Args:
            t1: 上一轮的trustworthiness
            t2: 当前轮的trustworthiness
            threshold: 收敛阈值
            
        Returns:
            True表示已收敛，False表示未收敛
        """
        return norm(t2 - t1) < threshold

    def _estimate_global_truth_text(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """
        基于fact_confidence提取每个object的代表性真值文本。
        """
        truth_map: Dict[str, str] = {}
        for object_id in dataframe["object"].unique():
            obj_df = dataframe[dataframe["object"] == object_id]
            if obj_df.empty:
                continue
            best_idx = obj_df["fact_confidence"].astype(float).idxmax()
            truth_map[str(object_id)] = str(dataframe.loc[best_idx, "fact"])
        return truth_map

    def train(self, 
              dataframe: pd.DataFrame,
              max_iterations: int = 200,
              threshold: float = 1e-6,
              initial_trustworthiness: float = 0.9) -> pd.DataFrame:
        """
        训练模型（批量处理）
        
        反复迭代直到收敛或达到最大迭代次数。
        
        Args:
            dataframe: 输入DataFrame，必须包含website, fact, object列
            max_iterations: 最大迭代次数
            threshold: 收敛阈值
            initial_trustworthiness: 初始数据源可靠性
            
        Returns:
            更新后的DataFrame，包含trustworthiness和fact_confidence列
        """
        # 创建副本
        dataframe = dataframe.copy()
        
        # 初始化trustworthiness和fact_confidence
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))

        # 迭代
        converged_at = None
        for i in range(max_iterations):
            # 保存上一轮的trustworthiness
            t1 = dataframe.drop_duplicates("website")["trustworthiness"]

            # 执行一次迭代
            dataframe = self.iteration(dataframe)

            # 获取当前轮的trustworthiness
            t2 = dataframe.drop_duplicates("website")["trustworthiness"]

            # 检查收敛
            if self.stop_condition(t1, t2, threshold):
                converged_at = i + 1
                break

        self.last_iteration_count = converged_at if converged_at is not None else max_iterations

        if self.enable_zk_proof:
            truth_map = self._estimate_global_truth_text(dataframe)
            dataframe["global_truth"] = dataframe["object"].map(truth_map)
            source_scores = (
                dataframe[["website", "trustworthiness"]]
                .drop_duplicates(subset=["website"])
                .set_index("website")["trustworthiness"]
                .astype(float)
                .to_dict()
            )
            dataframe = attach_grouped_proofs(
                dataframe,
                method_name="BasicTruthFinder",
                source_scores=source_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
            )

        return dataframe

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:
        """
        获取数据源可靠性信息
        
        Args:
            website: 指定数据源（可选）
            
        Returns:
            如果指定website，返回这个源的可靠性
            如果不指定，返回所有源的可靠性字典
        """
        # 这个方法需要从DataFrame中提取，所以需要先训练
        # 为了保持接口一致性，返回空字典
        # 实际使用时应该从训练后的DataFrame中获取
        return {}


def create_implication_function_from_texts(texts: list, 
                                          vectorizer: Optional[TfidfVectorizer] = None) -> Callable[[str, str], float]:
    """
    从文本列表创建implication函数（使用TF-IDF + 余弦相似度）
    
    这是一个辅助函数，用于创建基于文本相似度的implication函数。
    
    Args:
        texts: 文本列表（用于训练TF-IDF向量化器）
        vectorizer: 可选的TF-IDF向量化器（如果不提供，会创建一个新的）
        
    Returns:
        implication函数：接受两个文本，返回相似度
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(texts, stop_words=None)
        vectorizer.fit(texts)
    
    def similarity(w1: str, w2: str) -> float:
        """计算两个文本的余弦相似度"""
        V = vectorizer.transform([w1, w2])
        v1, v2 = np.asarray(V.todense())
        dot_product = np.dot(v1, v2)
        norm1 = norm(v1)
        norm2 = norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0
    
    def implication(f1: str, f2: str) -> float:
        """implication函数：返回两个文本的相似度"""
        return similarity(f1.lower(), f2.lower())
    
    return implication
