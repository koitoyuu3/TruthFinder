"""
SemanticTruthFinder

Cross-batch robust truth discovery for numeric and text data.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .hybrid_text_alignment import (
    DEFAULT_NUMERIC_MATCH_TOL,
    DEFAULT_SEMANTIC_NUMERIC_ALPHA,
    blend_semantic_numeric_scores,
    extract_numeric_values,
    numeric_match_ratio,
    symmetric_numeric_match_ratio,
)
from .text_vectorization import build_tfidf_vectorizer
from .zk_proof import ZKProofEngine, attach_grouped_proofs

try:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:  # pragma: no cover - import environment dependent
    snapshot_download = None
    SentenceTransformer = None
    HAS_SBERT = False



class SemanticTruthFinder:
    def __init__(
        self,
        tukey_c: float = 1.5,
        irls_max_iter: int = 10,
        weiszfeld_max_iter: int = 10,
        inner_iter: int = 3,
        eta: float = 1.0,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        convergence_tol: float = 1e-8,
        rel_gate_frac: float = 0.3,
        diversity_bonus: float = 1.0,
        use_sbert: bool = True,
        sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        numeric_tol: float = DEFAULT_NUMERIC_MATCH_TOL,
        semantic_numeric_alpha: float = DEFAULT_SEMANTIC_NUMERIC_ALPHA,
        local_files_only: Optional[bool] = None,
        enable_zk_proof: bool = False,
        zk_proof_secret: Optional[str] = None,
    ):
        # Fix 2: tukey_c 从 4.685（高斯噪声调参）降至 1.5（拜占庭对抗场景）
        # c=1.5 时偏离 0.9σ 的点权重仅 3.7%，能有效排除 adversarial outlier
        self.tukey_c = max(float(tukey_c), 1e-8)
        self.irls_max_iter = max(1, int(irls_max_iter))
        self.weiszfeld_max_iter = max(1, int(weiszfeld_max_iter))
        # 批内迭代轮数：每轮用上一轮的 consistency 更新权重后重算质心
        self.inner_iter = max(1, int(inner_iter))
        self.eta = max(float(eta), 0.0)
        self.initial_alpha = max(float(initial_alpha), 1e-8)
        self.initial_beta = max(float(initial_beta), 1e-8)
        self.convergence_tol = max(float(convergence_tol), 1e-12)
        # Fix 3: reliability 门控比例——reliability < mean_rel * rel_gate_frac 的源被预过滤
        # 在跨 batch 学到可信度差异后生效；初始阶段（全等）不会错误排除任何 agent
        self.rel_gate_frac = max(float(rel_gate_frac), 0.0)
        self.semantic_cluster_threshold = 0.80
        self.numeric_tol = max(float(numeric_tol), 0.0)
        # Default to equal semantic/numeric weight to align winner selection,
        # consistency updates, and DataAcc evaluation.
        self.semantic_numeric_alpha = float(np.clip(semantic_numeric_alpha, 0.0, 1.0))
        self.consistency_sem_weight = self.semantic_numeric_alpha
        self.consistency_num_weight = 1.0 - self.semantic_numeric_alpha
        self.use_sbert = bool(use_sbert) and HAS_SBERT
        self.sbert_model = sbert_model
        if local_files_only is None:
            local_files_only = (
                os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            )
        self.local_files_only = bool(local_files_only)
        self.enable_zk_proof = bool(enable_zk_proof)
        self.zk_proof_engine = ZKProofEngine(
            prover_id="SemanticTruthFinder",
            secret_seed=zk_proof_secret,
        )

        self.source_state: Dict[str, Dict[str, float]] = {}
        self._encoder = None
        self._tfidf_vectorizer = None

        if self.use_sbert:
            try:
                model_source = self.sbert_model
                if (
                    self.local_files_only
                    and snapshot_download is not None
                    and not os.path.isdir(self.sbert_model)
                ):
                    repo_id = (
                        self.sbert_model
                        if "/" in self.sbert_model
                        else f"sentence-transformers/{self.sbert_model}"
                    )
                    model_source = snapshot_download(
                        repo_id,
                        local_files_only=True,
                    )
                old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
                if self.local_files_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                try:
                    self._encoder = SentenceTransformer(
                        model_source,
                        local_files_only=self.local_files_only,
                    )
                finally:
                    if self.local_files_only:
                        if old_hf_offline is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                        if old_tf_offline is None:
                            os.environ.pop("TRANSFORMERS_OFFLINE", None)
                        else:
                            os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
            except Exception as exc:  # pragma: no cover - model availability dependent
                warnings.warn(
                    f"Failed to load SBERT model {self.sbert_model}: {exc}. Falling back to TF-IDF."
                )
                self.use_sbert = False
                self._encoder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(
        self,
        numeric_df: pd.DataFrame,
        text_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        numeric_df = self._normalize_numeric_df(numeric_df)
        text_df = self._normalize_text_df(text_df)

        sources = sorted(
            set(numeric_df.get("website", pd.Series(dtype=object)).astype(str).tolist())
            | set(text_df.get("website", pd.Series(dtype=object)).astype(str).tolist())
        )
        self._ensure_sources(sources)
        reliabilities = {
            source: self._reliability_from_state(self.source_state[source])
            for source in sources
        }

        numeric_truth, cons_num = self._numeric_pipeline(numeric_df, reliabilities)
        text_truth, cons_txt = self._text_pipeline(text_df, reliabilities)
        self._update_reliability(sources, cons_num, cons_txt)

        numeric_out = self._decorate_output(
            numeric_df,
            numeric_truth,
            cons_num,
            truth_column="global_truth",
        )
        text_out = self._decorate_output(
            text_df,
            text_truth,
            cons_txt,
            truth_column="global_truth",
        )

        if self.enable_zk_proof:
            numeric_out = self._attach_proofs(numeric_out, "SemanticTruthFinder[numeric]")
            text_out = self._attach_proofs(text_out, "SemanticTruthFinder[text]")

        return numeric_out, text_out

    def get_reliability(self, source: Optional[str] = None) -> Dict:
        if source is not None:
            state = self.source_state.get(str(source), {})
            if not state:
                return {}
            return {
                "alpha": float(state["alpha"]),
                "beta": float(state["beta"]),
                "reliability": float(self._reliability_from_state(state)),
            }

        return {
            str(src): {
                "alpha": float(state["alpha"]),
                "beta": float(state["beta"]),
                "reliability": float(self._reliability_from_state(state)),
            }
            for src, state in self.source_state.items()
        }

    def reset(self) -> None:
        self.source_state = {}
        self._tfidf_vectorizer = None

    # ------------------------------------------------------------------
    # Numeric pipeline
    # ------------------------------------------------------------------

    def _numeric_pipeline(
        self,
        df: pd.DataFrame,
        reliabilities: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        if df.empty:
            return {}, {}

        truth_map: Dict[str, float] = {}
        source_scores: Dict[str, List[float]] = {}

        for object_id, group in df.groupby("object", sort=True):
            working = group.copy()
            working["fact"] = pd.to_numeric(working["fact"], errors="coerce")
            working = working.dropna(subset=["fact"])
            if working.empty:
                continue

            values = working["fact"].astype(float).to_numpy()
            sources = working["website"].astype(str).tolist()
            base_weights = np.asarray(
                [max(reliabilities.get(source, self._default_reliability()), 1e-8) for source in sources],
                dtype=float,
            )

            # Fix 3: 跨 batch reliability 门控预过滤
            # 剔除 reliability 明显低于均值的源（仅在多 batch 累积后生效）
            mean_rel = float(np.mean(base_weights))
            gate = max(mean_rel * self.rel_gate_frac, 1e-8)
            gate_mask = base_weights >= gate
            if gate_mask.sum() >= 1 and gate_mask.sum() < len(values):
                v_irls = values[gate_mask]
                s_irls = [s for s, m in zip(sources, gate_mask) if m]
                w_irls = base_weights[gate_mask]
            else:
                v_irls, s_irls, w_irls = values, sources, base_weights

            truth_estimate = float(np.average(v_irls, weights=w_irls))
            for _ in range(self.irls_max_iter):
                robust_center = self._weighted_median(v_irls, w_irls)
                mad = float(np.median(np.abs(v_irls - robust_center)))
                scale = max(1.4826 * mad, self.convergence_tol)

                if scale <= self.convergence_tol:
                    # Fix 1: MAD=0 时直接返回中位数，而非加权均值
                    # 原因：MAD=0 说明多数 agent 报告相同值（少数 outlier 不影响中位数）
                    # 返回均值会将 Byzantine 值混入结果；返回中位数天然鲁棒
                    truth_estimate = robust_center
                    break

                u = (v_irls - robust_center) / (self.tukey_c * scale + 1e-12)
                robust_weights = w_irls * self._tukey_weight(u)
                weight_sum = float(robust_weights.sum())
                if weight_sum <= 1e-12:
                    truth_estimate = robust_center
                    break

                updated_truth = float(np.sum(robust_weights * v_irls) / weight_sum)
                if abs(updated_truth - truth_estimate) < self.convergence_tol:
                    truth_estimate = updated_truth
                    break
                truth_estimate = updated_truth

            truth_map[str(object_id)] = truth_estimate

            # consistency 基于全量 values（含门控外的源），用于更新跨 batch reliability
            residuals = values - truth_estimate
            residual_mad = float(np.median(np.abs(residuals - np.median(residuals))))
            score_scale = max(1.4826 * residual_mad, np.std(residuals), self.convergence_tol)
            consistency = np.exp(-0.5 * (residuals / score_scale) ** 2)
            consistency = np.clip(consistency, 0.0, 1.0)

            for source, score in zip(sources, consistency):
                source_scores.setdefault(source, []).append(float(score))

        return truth_map, {
            source: float(np.mean(scores))
            for source, scores in source_scores.items()
            if scores
        }

    def _weighted_median(self, v: np.ndarray, w: np.ndarray) -> float:
        """Weighted median: argmin_m Σ w_i |v_i - m|.
        With equal weights this equals np.median. With unequal weights it
        shifts toward the cluster whose agents have accumulated higher
        cross-batch reliability, enabling faster convergence.
        """
        total_w = float(w.sum())
        if total_w <= 0 or len(v) == 0:
            return float(np.median(v)) if len(v) > 0 else 0.0
        order = np.argsort(v)
        v_s = v[order]
        w_s = w[order]
        cum_w = np.cumsum(w_s)
        half = total_w / 2.0
        idx = int(np.searchsorted(cum_w, half, side="left"))
        idx = min(idx, len(v_s) - 1)
        return float(v_s[idx])

    def _tukey_weight(self, u: np.ndarray) -> np.ndarray:
        abs_u = np.abs(np.asarray(u, dtype=float))
        inside = abs_u <= 1.0
        weights = np.zeros_like(abs_u, dtype=float)
        weights[inside] = (1.0 - abs_u[inside] ** 2) ** 2
        return weights

    # ------------------------------------------------------------------
    # Text pipeline
    # ------------------------------------------------------------------

    def _find_connected_components(self, adj: np.ndarray) -> List[List[int]]:
        """BFS 找邻接矩阵中所有连通分量，返回各分量的节点索引列表。"""
        n = len(adj)
        visited = [False] * n
        components: List[List[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            cluster: List[int] = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                cluster.append(node)
                for neighbor in range(n):
                    if not visited[neighbor] and bool(adj[node, neighbor]):
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(cluster)
        return components

    def _extract_numeric_values(self, text: str) -> List[float]:
        return extract_numeric_values(text)

    def _semantic_affinity_matrix(
        self,
        peer_sims: np.ndarray,
        report_numeric_values: List[List[float]],
    ) -> np.ndarray:
        """
        两路信号：语义相似度（SBERT 或 TF-IDF）+ 数值匹配率。
        语言无关，不依赖 token / 停用词。
        - 含数字：affinity = alpha * sem_sim + (1-alpha) * numeric_sim
        - 无数字：affinity = sem_sim（纯语义判断）
        """
        n = peer_sims.shape[0]
        affinity = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if report_numeric_values[i] or report_numeric_values[j]:
                    numeric_similarity = self._numeric_similarity(
                        report_numeric_values[i], report_numeric_values[j]
                    )
                    pair_affinity = blend_semantic_numeric_scores(
                        sem_score=float(peer_sims[i, j]),
                        num_score=numeric_similarity,
                        alpha=self.semantic_numeric_alpha,
                    )
                else:
                    pair_affinity = float(peer_sims[i, j])
                affinity[i, j] = pair_affinity
                affinity[j, i] = pair_affinity
        return np.clip(affinity, 0.0, 1.0)

    def _build_semantic_clusters(self, affinity: np.ndarray) -> Tuple[List[List[int]], Dict[int, int]]:
        n = affinity.shape[0]
        adj = np.eye(n, dtype=bool)
        threshold = self.semantic_cluster_threshold
        if not (self.use_sbert and self._encoder is not None):
            threshold = 0.55
        adj |= affinity >= threshold
        clusters = self._find_connected_components(adj)
        membership: Dict[int, int] = {}
        for cluster_idx, members in enumerate(clusters):
            for member in members:
                membership[member] = cluster_idx
        return clusters, membership

    def _numeric_similarity(self, left: List[float], right: List[float]) -> float:
        return symmetric_numeric_match_ratio(left, right, tol=self.numeric_tol)

    def _numeric_match_ratio(self, reference: List[float], candidate: List[float]) -> float:
        return numeric_match_ratio(reference, candidate, tol=self.numeric_tol)

    def _cluster_numeric_consensus(self, numeric_lists: List[List[float]]) -> List[float]:
        """
        簇内数值共识：对每个数值位置取中位数。
        只使用拥有最众数长度的成员做位置对齐，避免数值个数不同导致位置错位。
        若簇内文本均不含数字，返回空列表（不惩罚纯文本对象）。
        """
        non_empty = [lst for lst in numeric_lists if lst]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            return list(non_empty[0])

        mode_len = Counter(len(lst) for lst in non_empty).most_common(1)[0][0]
        aligned = [lst for lst in non_empty if len(lst) == mode_len]
        if not aligned:
            min_len = min(len(lst) for lst in non_empty)
            aligned = [lst[:min_len] for lst in non_empty]

        return [float(np.median([lst[i] for lst in aligned])) for i in range(len(aligned[0]))]

    def _combine_consistency_components(
        self,
        sem_score: float,
        num_score: Optional[float],
    ) -> float:
        """
        Consistency = 语义相似度 + 数值匹配率（语言无关，无 token 依赖）。
        无数字文本：100% 语义相似度。
        含数字文本：默认 0.5 语义 + 0.5 数值，与 DataAcc 对齐。
        """
        return blend_semantic_numeric_scores(
            sem_score=sem_score,
            num_score=num_score,
            alpha=self.semantic_numeric_alpha,
        )

    def _text_pipeline(
        self,
        df: pd.DataFrame,
        reliabilities: Dict[str, float],
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        软投票真值选择（语言无关，无硬切簇阈值）：

          score(i) = Σ_{j≠i} affinity(i, j) × reliability(j)
          affinity(i, j) = alpha * semantic_sim(i, j) + (1-alpha) * numeric_match(i, j)

        胜出者 = argmax score。
        正确 agent 互相给出高 affinity，自然形成多数支撑，无论表述略有差异。
        协同恶意 agent 只能依靠同类支撑，数量少时总分低于正确侧。

        相比 Sente：affinity 同时看语义和数值匹配；语义相近但数值错的恶意 agent
        对正确 agent 的贡献会被数值通道压低，Sente 的纯 SBERT 无法识别此类攻击。

        跨 batch consistency：
          sem_score = SBERT_sim(i, winner)
          num_score = numeric_match(i, 全局数值中位共识)
        """
        if df.empty:
            return {}, {}

        truth_map: Dict[str, str] = {}
        source_scores: Dict[str, List[float]] = {}

        for object_id, group in df.groupby("object", sort=True):
            working = group.copy()
            working["fact"] = working["fact"].astype(str)
            if working.empty:
                continue

            texts = working["fact"].tolist()
            sources = working["website"].astype(str).tolist()
            embeddings = self._encode_texts(texts)
            if embeddings.size == 0:
                continue

            n = len(texts)
            if n == 1:
                truth_map[str(object_id)] = texts[0]
                source_scores.setdefault(sources[0], []).append(1.0)
                continue

            peer_sims = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
            np.fill_diagonal(peer_sims, 1.0)

            report_numeric_values = [self._extract_numeric_values(text) for text in texts]
            affinity = self._semantic_affinity_matrix(peer_sims, report_numeric_values)

            # 软投票：score(i) = Σ_{j≠i} affinity(i,j) × reliability(j)
            default_rel = self._default_reliability()
            rel_weights = np.array(
                [reliabilities.get(sources[j], default_rel) for j in range(n)],
                dtype=float,
            )
            masked = affinity.copy()
            np.fill_diagonal(masked, 0.0)
            vote_scores = masked @ rel_weights  # shape (n,)

            local_best = int(np.argmax(vote_scores))
            truth_map[str(object_id)] = texts[local_best]

            # Consistency 更新：语义对比胜出者；数值对比全局中位共识
            truth_emb = embeddings[local_best]
            global_num_consensus = self._cluster_numeric_consensus(report_numeric_values)
            for idx, source in enumerate(sources):
                sem_score = float(np.clip(float(embeddings[idx] @ truth_emb), 0.0, 1.0))
                num_score = None
                if global_num_consensus or report_numeric_values[idx]:
                    num_score = self._numeric_match_ratio(
                        global_num_consensus, report_numeric_values[idx]
                    )
                score = self._combine_consistency_components(
                    sem_score=sem_score,
                    num_score=num_score,
                )
                source_scores.setdefault(source, []).append(float(np.clip(score, 0.0, 1.0)))

        return truth_map, {
            source: float(np.mean(scores))
            for source, scores in source_scores.items()
            if scores
        }

    # ------------------------------------------------------------------
    # Reliability update
    # ------------------------------------------------------------------

    def _update_reliability(
        self,
        sources: List[str],
        cons_num: Dict[str, float],
        cons_txt: Dict[str, float],
    ) -> None:
        for source in sources:
            self._ensure_sources([source])
            available_scores = []
            if source in cons_num:
                available_scores.append(float(cons_num[source]))
            if source in cons_txt:
                available_scores.append(float(cons_txt[source]))
            if not available_scores:
                continue

            cons = float(np.mean(available_scores))
            cons = float(np.clip(cons, 0.0, 1.0))
            self.source_state[source]["alpha"] += self.eta * cons
            self.source_state[source]["beta"] += self.eta * (1.0 - cons)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _normalize_numeric_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
        out = df.copy()
        for column in ["website", "object"]:
            out[column] = out[column].astype(str)
        out["fact"] = pd.to_numeric(out["fact"], errors="coerce")
        if "batch_index" not in out.columns:
            out["batch_index"] = 0
        return out[["website", "object", "fact", "batch_index"]]

    def _normalize_text_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
        out = df.copy()
        for column in ["website", "object", "fact"]:
            out[column] = out[column].astype(str)
        if "batch_index" not in out.columns:
            out["batch_index"] = 0
        return out[["website", "object", "fact", "batch_index"]]

    def _ensure_sources(self, sources: List[str]) -> None:
        for source in sources:
            if source not in self.source_state:
                self.source_state[source] = {
                    "alpha": self.initial_alpha,
                    "beta": self.initial_beta,
                }

    def _default_reliability(self) -> float:
        return self.initial_alpha / (self.initial_alpha + self.initial_beta)

    @staticmethod
    def _reliability_from_state(state: Dict[str, float]) -> float:
        total = float(state["alpha"]) + float(state["beta"])
        if total <= 0:
            return 0.5
        return float(state["alpha"]) / total

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)

        if self.use_sbert and self._encoder is not None:
            vectors = self._encoder.encode(texts, convert_to_numpy=True)
            return self._normalize_embeddings(np.asarray(vectors, dtype=float))

        self._tfidf_vectorizer = build_tfidf_vectorizer(texts, max_features=1000)
        matrix = self._tfidf_vectorizer.fit_transform(texts)
        return self._normalize_embeddings(matrix.toarray())

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings, dtype=float),
            where=norms > 0,
        )

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return np.zeros_like(vector, dtype=float)
        return vector / norm

    def _decorate_output(
        self,
        df: pd.DataFrame,
        truth_map: Dict[str, object],
        consistency_map: Dict[str, float],
        *,
        truth_column: str,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        out = df.copy()
        out[truth_column] = out["object"].map(lambda obj: truth_map.get(str(obj)))
        out["source_reliability"] = out["website"].map(
            lambda src: self._reliability_from_state(self.source_state[str(src)])
        )
        out["consistency_score"] = out["website"].map(
            lambda src: float(consistency_map.get(str(src), np.nan))
        )
        out["alpha_state"] = out["website"].map(
            lambda src: float(self.source_state[str(src)]["alpha"])
        )
        out["beta_state"] = out["website"].map(
            lambda src: float(self.source_state[str(src)]["beta"])
        )
        return out

    def _attach_proofs(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        if df.empty or "global_truth" not in df.columns:
            return df
        source_scores = {
            source: self._reliability_from_state(state)
            for source, state in self.source_state.items()
        }
        iteration_count = None
        if "batch_index" in df.columns and not df["batch_index"].dropna().empty:
            unique_batches = sorted(df["batch_index"].dropna().unique().tolist())
            if len(unique_batches) == 1:
                iteration_count = int(unique_batches[0])
        return attach_grouped_proofs(
            df,
            method_name=method_name,
            source_scores=source_scores,
            group_columns=("object",),
            truth_column="global_truth",
            iteration_count=iteration_count,
            proof_engine=self.zk_proof_engine,
        )


SemanticTruthDiscovery = SemanticTruthFinder
