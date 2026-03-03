"""
SenteTruth TruthFinder
基于论文"Connecting Large Language Models with Blockchain: Making Smart Contracts Smarter"
中的SenteTruth方法实现的真值发现算法

这个文件实现了SenteTruth真值发现算法，主要功能是：
1. 使用SBERT进行语义编码
2. 计算节点数据之间的语义相似度
3. 结合节点可信度和语义相似度进行真相聚合
4. 迭代更新节点可信度

核心公式：
- 公式2：φ(v_i) = (1/n) * Σ(j=1, j≠i to n) cosine_similarity(v_i, v_j)  # 语义相似度
- 公式3：D ← argmax_i C_i · φ(v_i)  # 真相聚合
- 公式4：C_i ← ( (Σ_{l=1}^n C_l) / (Σ_{l=1}^n C_l · φ(v_l)) ) · C_i · φ(v_i)  # 可信度更新
"""
import numpy as np
from numpy.linalg import norm
import pandas as pd
from typing import Dict, Optional, Union, List
import warnings
from sentence_transformers import SentenceTransformer
HAS_SBERT = True


class SenteTruthFinder:
    """
    SenteTruth真值发现算法
    
    基于论文中的SenteTruth方法，专门用于处理文本数据的真值发现。
    与EnhancedTruthFinder的混合模式不同，这个方法：
    - 不提取结构化信息（数值/离散）
    - 直接使用文本的语义相似度
    - 适用于LLM生成的文本数据
    
    数据结构：DataFrame with columns: website, fact, object
    - website: 数据源标识（例如："Agent_1"、"节点1"等）
    - fact: 输入文本（例如：完整的句子或段落）
    - object: 对象标识（例如："Agilent_Options"等）
    
    工作流程：
    1. 使用SBERT将所有文本编码为向量
    2. 计算语义相似度矩阵
    3. 计算每个节点的可信度分数 φ(v_i)
    4. 选择 C_i · φ(v_i) 最大的节点数据作为真值
    5. 根据与真值的相似度更新节点可信度
    6. 迭代直到收敛
    """

    def __init__(self,
                 convergence_threshold: float = 1e-6,
                 max_iterations: int = 200,
                 initial_credibility: float = 0.5,
                 sbert_model: Optional[str] = None,
                 use_sbert: bool = True):
        """
        初始化SenteTruthFinder
        
        Args:
            convergence_threshold: 收敛阈值
                                 - 当节点可信度的变化小于这个值时，算法认为已经收敛
            max_iterations: 最大迭代次数
            initial_credibility: 初始节点可信度 (0-1之间)
                               - 所有节点开始时都有这个可信度
            sbert_model: SBERT模型名称（可选）
                        - 如果不提供，使用默认模型 'all-MiniLM-L6-v2'
                        - 如果use_sbert=False，忽略此参数
            use_sbert: 是否使用SBERT（如果False，使用TF-IDF作为备选）
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.initial_credibility = initial_credibility
        self.use_sbert = use_sbert and HAS_SBERT
        
        # 节点可信度历史
        # 格式: {website: credibility}
        # 例如: {'Agent_1': 0.7, 'Agent_2': 0.8, ...}
        self.node_credibility: Dict[str, float] = {}
        
        # 全局真值历史（按object分组）
        # 格式: {object: truth_text}
        # 例如: {'Agilent_Options': 'A new Agilent Technologies...'}
        self.global_truth_history: Dict[str, str] = {}
        
        # 句子编码器
        if self.use_sbert:
            model_name = sbert_model or 'all-MiniLM-L6-v2'
            try:
                self.encoder = SentenceTransformer(model_name)
                print(f"Using SBERT model: {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load SBERT model {model_name}: {e}. Falling back to TF-IDF.")
                self.use_sbert = False
                self.encoder = None
        else:
            self.encoder = None
        
        # TF-IDF向量化器（备选方案）
        if not self.use_sbert:
            self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
            print("Using TF-IDF as fallback for semantic encoding")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量矩阵，shape=(n_texts, embedding_dim)
        """
        if self.use_sbert and self.encoder is not None:
            # 使用SBERT编码
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            return embeddings
        else:
            # 使用TF-IDF作为备选
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=1000
                )
            
            vectors = self.tfidf_vectorizer.fit_transform(texts)
            return vectors.toarray()

    def _calculate_semantic_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算语义相似度矩阵（公式2的基础）
        
        计算所有节点之间的余弦相似度
        
        Args:
            embeddings: 向量矩阵，shape=(n_nodes, embedding_dim)
            
        Returns:
            相似度矩阵，shape=(n_nodes, n_nodes)
            - S[i, j] = cosine_similarity(embedding_i, embedding_j)
            - 对角线元素为1.0（自己与自己的相似度）
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 计算余弦相似度
                    v_i = embeddings[i]
                    v_j = embeddings[j]
                    dot_product = np.dot(v_i, v_j)
                    norm_i = norm(v_i)
                    norm_j = norm(v_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        # 确保相似度在[0, 1]范围内（余弦相似度通常在[-1, 1]，我们归一化到[0, 1]）
                        similarity = (similarity + 1) / 2
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix

    def _calculate_trustworthiness(self, similarity_matrix: np.ndarray, node_idx: int) -> float:
        """
        计算节点的可信度分数 φ(v_i)（公式2）
        
        φ(v_i) = (1/n) * Σ(j=1, j≠i to n) S[i, j]
        其中 S[i, j] 是节点i和节点j的相似度
        
        Args:
            similarity_matrix: 相似度矩阵
            node_idx: 节点索引
            
        Returns:
            可信度分数 φ(v_i)，范围[0, 1]
        """
        n = len(similarity_matrix)
        if n <= 1:
            return 1.0
        
        # 计算节点i与其他所有节点的平均相似度（排除自己）
        similarities = []
        for j in range(n):
            if j != node_idx:
                similarities.append(similarity_matrix[node_idx, j])
        
        if len(similarities) > 0:
            phi = np.mean(similarities)
        else:
            phi = 1.0
        
        return float(phi)

    def _aggregate_truth(self, 
                        df: pd.DataFrame,
                        embeddings: np.ndarray,
                        similarity_matrix: np.ndarray) -> Dict[str, str]:
        """
        真相聚合（公式3）
        
        D ← argmax_i C_i · φ(v_i)
        选择 C_i · φ(v_i) 最大的节点数据作为真值
        
        Args:
            df: 单个object的数据子集
            embeddings: 向量矩阵
            similarity_matrix: 相似度矩阵
            
        Returns:
            {object: truth_text}  # 真值文本
        """
        if df.empty:
            return {}
        
        object_id = df['object'].iloc[0]
        n = len(df)
        
        # 计算每个节点的加权分数
        scores = []
        texts = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            text = str(row['fact'])
            
            # 获取节点可信度
            C_i = self.node_credibility.get(website, self.initial_credibility)
            
            # 计算可信度分数
            phi_i = self._calculate_trustworthiness(similarity_matrix, idx)
            
            # 计算加权分数
            score = C_i * phi_i
            
            scores.append(score)
            texts.append(text)
        
        # 选择分数最大的节点数据作为真值
        best_idx = np.argmax(scores)
        truth_text = texts[best_idx]
        
        return {object_id: truth_text}

    def _update_credibility(self,
                           df: pd.DataFrame,
                           embeddings: np.ndarray,
                           similarity_matrix: np.ndarray,
                           global_truth: Dict[str, str]):
        """
        更新节点可信度（公式4）
        
        C_i ← ( (Σ_{l=1}^n C_l) / (Σ_{l=1}^n C_l · φ(v_l)) ) · C_i · φ(v_i)
        
        严格按照论文公式4实现，不添加额外优化。
        这个公式根据节点数据与其他节点的相似度来更新可信度。
        如果节点的数据与其他节点相似度高（φ(v_i)大），可信度会增加；否则会降低。
        
        Args:
            df: 单个object的数据子集
            embeddings: 向量矩阵
            similarity_matrix: 相似度矩阵
            global_truth: 全局真值 {object: truth_text}（用于参考，但不直接用于更新）
        """
        if df.empty:
            return
        
        # 计算所有节点的可信度和可信度分数
        credibilities = []
        trustworthiness_scores = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            C_i = self.node_credibility.get(website, self.initial_credibility)
            phi_i = self._calculate_trustworthiness(similarity_matrix, idx)
            
            credibilities.append(C_i)
            trustworthiness_scores.append(phi_i)
        
        # 计算公式4中的归一化因子
        sum_C = sum(credibilities)
        sum_C_phi = sum(C_i * phi_i for C_i, phi_i in zip(credibilities, trustworthiness_scores))
        
        # 更新每个节点的可信度（严格按照公式4）
        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            C_i = self.node_credibility.get(website, self.initial_credibility)
            phi_i = self._calculate_trustworthiness(similarity_matrix, idx)
            
            # 公式4：C_i ← ( (Σ_{l=1}^n C_l) / (Σ_{l=1}^n C_l · φ(v_l)) ) · C_i · φ(v_i)
            if sum_C_phi > 0:
                new_C_i = (sum_C / sum_C_phi) * C_i * phi_i
            else:
                # 如果分母为0，保持原值
                new_C_i = C_i
            
            # 确保可信度在合理范围内（避免极端值）
            new_C_i = max(0.01, min(1.0, new_C_i))
            
            self.node_credibility[website] = new_C_i

    def _initialize_credibility(self, dataframe: pd.DataFrame):
        """
        初始化节点可信度
        
        Args:
            dataframe: 输入DataFrame
        """
        websites = dataframe['website'].unique()
        for website in websites:
            if website not in self.node_credibility:
                self.node_credibility[website] = self.initial_credibility

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        执行一次迭代
        
        工作流程：
        1. 按object分组
        2. 对每个object：
           a. 编码所有节点的文本
           b. 计算相似度矩阵
           c. 聚合真值
           d. 更新节点可信度
        
        Args:
            dataframe: 输入DataFrame
            
        Returns:
            更新后的DataFrame（添加了source_reliability和global_truth列）
        """
        # 初始化可信度
        self._initialize_credibility(dataframe)
        
        # 按object分组处理
        all_truth = {}
        
        for object_id in dataframe['object'].unique():
            obj_data = dataframe[dataframe['object'] == object_id].copy()
            
            # 获取所有节点的文本
            texts = [str(row['fact']) for _, row in obj_data.iterrows()]
            
            # 编码文本
            embeddings = self._encode_texts(texts)
            
            # 计算相似度矩阵
            similarity_matrix = self._calculate_semantic_similarity_matrix(embeddings)
            
            # 聚合真值
            truth = self._aggregate_truth(obj_data, embeddings, similarity_matrix)
            all_truth.update(truth)
            
            # 更新可信度
            self._update_credibility(obj_data, embeddings, similarity_matrix, all_truth)
        
        # 更新DataFrame
        dataframe['source_reliability'] = dataframe['website'].map(
            lambda w: self.node_credibility.get(w, self.initial_credibility)
        )
        dataframe['global_truth'] = dataframe['object'].map(all_truth)
        
        return dataframe

    def train(self,
              dataframe: pd.DataFrame,
              max_iterations: Optional[int] = None,
              threshold: Optional[float] = None) -> pd.DataFrame:
        """
        训练模型（批量处理）
        
        反复迭代直到收敛或达到最大迭代次数
        
        Args:
            dataframe: 输入DataFrame，必须包含website, fact, object列
            max_iterations: 最大迭代次数（可选）
            threshold: 收敛阈值（可选）
            
        Returns:
            更新后的DataFrame，包含source_reliability和global_truth列
        """
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        conv_threshold = threshold if threshold is not None else self.convergence_threshold
        
        # 创建副本
        dataframe = dataframe.copy()
        
        # 确保fact列是字符串
        dataframe['fact'] = dataframe['fact'].astype(str)
        
        # 初始化
        self._initialize_credibility(dataframe)
        
        # 迭代
        for i in range(max_iter):
            print(f'Iteration {i + 1}/{max_iter}')
            
            # 保存上一轮的可信度
            prev_credibility = self.node_credibility.copy()
            
            # 执行一次迭代
            dataframe = self.iteration(dataframe)
            
            # 检查收敛
            max_change = 0.0
            all_websites = set(list(prev_credibility.keys()) + list(self.node_credibility.keys()))
            
            for website in all_websites:
                prev_val = prev_credibility.get(website, 0)
                curr_val = self.node_credibility.get(website, 0)
                change = abs(curr_val - prev_val)
                max_change = max(max_change, change)
            
            if max_change < conv_threshold:
                print(f'Convergence reached at iteration {i + 1}/{max_iter}')
                break
        
        return dataframe

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:
        """
        获取节点可信度信息
        
        Args:
            website: 指定节点（可选）
            
        Returns:
            如果指定website，返回这个节点的可信度
            如果不指定，返回所有节点的可信度字典
        """
        if website is not None:
            return {'credibility': self.node_credibility.get(website, self.initial_credibility)}
        else:
            return {w: {'credibility': c} for w, c in self.node_credibility.items()}

    def reset_history(self):
        """重置历史状态"""
        self.node_credibility = {}
        self.global_truth_history = {}
