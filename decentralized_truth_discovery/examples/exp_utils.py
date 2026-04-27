"""
exp_utils.py — 在线实验共享工具库
================================================
评估框架（单次在线前向，对齐按 batch 顺序持续更新的协议）:
  preds = 方法在实验数据（含恶意节点）上的真值发现结果
  gt    = 干净数据（0 恶意节点）中的真实值

  数值场景 → RMSE(preds_last10%, gt_last10%), MAE(...)
            仅取最后 10% 的 batch（对齐论文：last 10% batches for long-term accuracy）
  混合场景 → DataAccuracy(X_clean, Y_noisy)
            = proportion of (batch, object) where SBERT_cosine(X, Y) >= threshold
            对齐论文 SenteTruth paper 的 Data Accuracy 指标：衡量含噪场景下能否选出
            与干净场景语义一致的正确答案。使用 SBERT 语义相似度而非 TF-IDF 字面距离。

方法适配范围:
  数值场景 → SenFeedTruth / DecentTruth / SenteTruth / BasicTruth / SemanticTruth
  混合场景 → SenteTruth / BasicTruth / SemanticTruth
"""
from __future__ import annotations

import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sentence_transformers import SentenceTransformer

_SBERT_ENCODER = None
_SBERT_ENCODER_INIT_FAILED = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.core import (
    BasicTruthFinder,
    EnhancedTruthFinder,
    SemanticTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
    attach_grouped_proofs,
)
from decentralized_truth_discovery.core.hybrid_text_alignment import (
    DEFAULT_NUMERIC_MATCH_TOL,
    DEFAULT_SEMANTIC_NUMERIC_ALPHA,
    blend_semantic_numeric_scores,
    extract_numeric_values,
    numeric_match_ratio,
)

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "decentralized_truth_discovery" / "data_agent-10"

DEMO_FILES: Dict[int, Path] = {
    0: DATA_DIR / "demo.json",
    1: DATA_DIR / "demo_1bad_agent.json",
    2: DATA_DIR / "demo_2bad_agent.json",
    3: DATA_DIR / "demo_3bad_agent.json",
    4: DATA_DIR / "demo_4bad_agent.json",
    5: DATA_DIR / "demo_5bad_agent.json",
    6: DATA_DIR / "demo_6bad_agent.json",
}

NUMERIC_DEMO_FILES: Dict[int, Path] = {
    0: DATA_DIR / "num_demo.json",
    1: DATA_DIR / "num_demo_1bad_agent.json",
    2: DATA_DIR / "num_demo_2bad_agent.json",
    3: DATA_DIR / "num_demo_3bad_agent.json",
    4: DATA_DIR / "num_demo_4bad_agent.json",
    5: DATA_DIR / "num_demo_5bad_agent.json",
    6: DATA_DIR / "num_demo_6bad_agent.json",
}

NUMERIC_COLLUDE_DEMO_FILES: Dict[int, Path] = {
    0: DATA_DIR / "num_demo.json",
    1: DATA_DIR / "num_demo_1bad_collude.json",
    2: DATA_DIR / "num_demo_2bad_collude.json",
    3: DATA_DIR / "num_demo_3bad_collude.json",
    4: DATA_DIR / "num_demo_4bad_collude.json",
    5: DATA_DIR / "num_demo_5bad_collude.json",
    6: DATA_DIR / "num_demo_6bad_collude.json",
}

COLLUDE_DEMO_FILES: Dict[int, Path] = {
    0: DATA_DIR / "demo.json",
    1: DATA_DIR / "demo_1bad_collude.json",
    2: DATA_DIR / "demo_2bad_collude.json",
    3: DATA_DIR / "demo_3bad_collude.json",
    4: DATA_DIR / "demo_4bad_collude.json",
    5: DATA_DIR / "demo_5bad_collude.json",
    6: DATA_DIR / "demo_6bad_collude.json",
}

ATTACK_PROFILE_DEMO_FILES: Dict[str, Dict[int, Path]] = {
    "agent": NUMERIC_DEMO_FILES,
    "collude": COLLUDE_DEMO_FILES,
}

NUMERIC_AGENT_DEMO_FILES: Dict[int, Path] = NUMERIC_DEMO_FILES

TEXT_AGENT_DEMO_FILES: Dict[int, Path] = DEMO_FILES
TEXT_COLLUDE_DEMO_FILES: Dict[int, Path] = COLLUDE_DEMO_FILES
EXP1_BAD_COUNTS: List[int] = [0, 1, 2, 3, 4]

N_AGENTS = 10

# ── 绘图样式 ──────────────────────────────────────────────────────────────────
NUM_METHODS  = ["SenFeedTruth", "DecentTruth", "SenteTruth", "BasicTruth", "SemanticTruth"]
TEXT_METHODS = ["SenteTruth", "BasicTruth", "SemanticTruth"]

COLORS = {
    "SenFeedTruth": "#e74c3c",
    "DecentTruth":  "#3498db",
    "SenteTruth":   "#2ecc71",
    "BasicTruth":   "#f39c12",
    "SemanticTruth": "#9b59b6",
}
MARKERS = {
    "SenFeedTruth": "o",
    "DecentTruth":  "^",
    "SenteTruth":   "s",
    "BasicTruth":   "D",
    "SemanticTruth": "*",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def get_demo_files(attack_profile: str = "agent") -> Dict[int, Path]:
    profile = str(attack_profile).strip().lower()
    if profile not in ATTACK_PROFILE_DEMO_FILES:
        supported = ", ".join(sorted(ATTACK_PROFILE_DEMO_FILES))
        raise ValueError(f"Unsupported attack profile: {attack_profile}. Expected one of: {supported}")
    return ATTACK_PROFILE_DEMO_FILES[profile]


def get_numeric_demo_files(mode: str = "collude") -> Dict[int, Path]:
    """Return numeric experiment files. mode='collude' (default) or 'agent'."""
    if mode == "collude":
        return NUMERIC_COLLUDE_DEMO_FILES
    return NUMERIC_AGENT_DEMO_FILES


def get_text_demo_files(mode: str = "collude") -> Dict[int, Path]:
    """Return text experiment files. mode='collude' (default) or 'agent'."""
    if mode == "agent":
        return TEXT_AGENT_DEMO_FILES
    return TEXT_COLLUDE_DEMO_FILES


def get_shared_bad_counts() -> List[int]:
    """
    Bad-count values available to both numeric and text experiment datasets.
    Exp1 is currently restricted to the <=40% malicious regime.
    """
    shared = set(NUMERIC_AGENT_DEMO_FILES) & set(TEXT_COLLUDE_DEMO_FILES)
    return [k for k in EXP1_BAD_COUNTS if k in shared]

# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_batches(filepath: Path) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["job"]["batches"]


def to_numeric_df(
    batches: List[Dict],
    agent_subset: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """
    提取数值条目：response 为 dict（BTC/ETH/DOGE 价格）。
    每个币种扩展为一行，object = 币名（"BTC"/"ETH"/"DOGE"）。
    返回列: website, fact(float), object, batch_index
    """
    rows: List[Dict] = []
    for b in (batches[:max_batches] if max_batches else batches):
        for item in b["items"]:
            if agent_subset and item["agent"] not in agent_subset:
                continue
            if isinstance(item["response"], dict):
                for coin, price in item["response"].items():
                    rows.append({
                        "website":     item["agent"],
                        "object":      str(coin),
                        "fact":        float(price),
                        "batch_index": int(b["batchIndex"]),
                    })
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["website", "object", "fact", "batch_index"]))


def to_text_df(
    batches: List[Dict],
    agent_subset: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """
    提取混合文本条目：response 为 str（新闻摘要，含数字+文字）。
    返回列: website, fact(str), object, batch_index
    """
    rows: List[Dict] = []
    for b in (batches[:max_batches] if max_batches else batches):
        for item in b["items"]:
            if agent_subset and item["agent"] not in agent_subset:
                continue
            if isinstance(item["response"], str):
                rows.append({
                    "website":     item["agent"],
                    "object":      str(item["object"]),
                    "fact":        item["response"],
                    "batch_index": int(b["batchIndex"]),
                })
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["website", "object", "fact", "batch_index"]))


def build_gt_numeric(
    batches: List[Dict],
    max_batches: Optional[int] = None,
) -> Dict[Tuple[int, str], float]:
    """
    从无恶意节点的干净数据中提取数值真值（对齐论文 Section VI）。
    返回 {(batch_index, coin) → true_price}。
    干净数据中所有 agent 报告相同价格，取第一个即为 ground truth。
    """
    gt: Dict[Tuple[int, str], float] = {}
    for b in (batches[:max_batches] if max_batches else batches):
        for item in b["items"]:
            if isinstance(item["response"], dict):
                for coin, price in item["response"].items():
                    key = (int(b["batchIndex"]), str(coin))
                    if key not in gt:
                        gt[key] = float(price)
                break  # 干净数据中所有 agent 值相同，取第一个即可
    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# 二、方法运行器  →  每个方法的输出均为 {(batch_index, object): value}
# ═══════════════════════════════════════════════════════════════════════════════

def _preds_numeric(out_df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """从方法输出 DataFrame 提取数值预测，key = (batch_index, object)。"""
    preds: Dict[Tuple[int, str], float] = {}
    if "global_truth" not in out_df.columns:
        return preds
    bi_col = "timestamp" if "timestamp" in out_df.columns else "batch_index"
    for _, row in out_df.drop_duplicates(["object", bi_col]).iterrows():
        try:
            preds[(int(row[bi_col]), str(row["object"]))] = float(row["global_truth"])
        except (ValueError, TypeError):
            pass
    return preds


def _preds_text(out_df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """从方法输出 DataFrame 提取文本预测，key = (batch_index, object)。"""
    preds: Dict[Tuple[int, str], str] = {}
    if "global_truth" not in out_df.columns:
        return preds
    bi_col = "batch_index" if "batch_index" in out_df.columns else "timestamp"
    for _, row in out_df.drop_duplicates(["object", bi_col]).iterrows():
        v = row["global_truth"]
        if pd.notna(v):
            preds[(int(row[bi_col]), str(row["object"]))] = str(v)
    return preds


def _run_silently(fn):
    """Suppress noisy model stdout during experiment loops."""
    with redirect_stdout(io.StringIO()):
        return fn()


def _basic_truth_map(out_df: pd.DataFrame) -> Dict[str, str]:
    """Extract one best fact per object from BasicTruth fact_confidence."""
    truth_map: Dict[str, str] = {}
    if out_df.empty or "fact_confidence" not in out_df.columns:
        return truth_map
    for object_id in out_df["object"].unique():
        obj_df = out_df[out_df["object"] == object_id]
        if obj_df.empty:
            continue
        best_idx = obj_df["fact_confidence"].astype(float).idxmax()
        truth_map[str(object_id)] = str(out_df.loc[best_idx, "fact"])
    return truth_map


def _scalar_numeric_similarity(
    left: float,
    right: float,
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:
    """Smooth numeric similarity based on relative distance."""
    denom = max(abs(float(left)), abs(float(right)), 1e-8)
    rel_error = abs(float(left) - float(right)) / denom
    bandwidth = max(float(tol), 1e-8)
    return float(np.exp(-rel_error / bandwidth))


class _NumericSenteTruthFinder(SenteTruthFinder):
    """Numeric adapter for SenteTruth that operates directly on floats."""

    def __init__(self, numeric_tol: float = DEFAULT_NUMERIC_MATCH_TOL, **kwargs):
        kwargs.setdefault("use_sbert", False)
        super().__init__(**kwargs)
        self.numeric_tol = max(float(numeric_tol), 1e-8)

    def _calculate_numeric_similarity_matrix(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n == 0:
            return np.zeros((0, 0), dtype=float)

        matrix = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = _scalar_numeric_similarity(values[i], values[j], tol=self.numeric_tol)
                matrix[i, j] = similarity
                matrix[j, i] = similarity
        return matrix

    def _aggregate_truth(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> Dict[str, float]:
        if df.empty:
            return {}

        object_id = str(df["object"].iloc[0])
        phi_values = self._calculate_trustworthiness_scores(similarity_matrix)
        scores: List[float] = []
        values: List[float] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            website = str(row["website"])
            value = float(row["fact"])
            credibility = self.node_credibility.get(website, self.initial_credibility)
            scores.append(float(credibility) * float(phi_values[idx]))
            values.append(value)

        best_idx = int(np.argmax(scores))
        return {object_id: float(values[best_idx])}

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self._initialize_credibility(dataframe)
        all_truth: Dict[str, float] = {}

        for object_id in dataframe["object"].unique():
            obj_data = dataframe[dataframe["object"] == object_id].copy()
            values = obj_data["fact"].astype(float).to_numpy()
            similarity_matrix = self._calculate_numeric_similarity_matrix(values)
            truth = self._aggregate_truth(obj_data, values.reshape(-1, 1), similarity_matrix)
            all_truth.update(truth)
            self._update_credibility(obj_data, values.reshape(-1, 1), similarity_matrix, all_truth)

        self.global_truth_history.update({str(k): float(v) for k, v in all_truth.items()})
        dataframe["source_reliability"] = dataframe["website"].map(
            lambda website: self.node_credibility.get(str(website), self.initial_credibility)
        )
        dataframe["global_truth"] = dataframe["object"].map(all_truth)
        return dataframe

    def process_batch(
        self,
        dataframe: pd.DataFrame,
        epoch: Optional[int] = None,
    ) -> pd.DataFrame:
        if dataframe is None or dataframe.empty:
            columns = ["website", "fact", "object", "source_reliability", "global_truth"]
            if dataframe is not None:
                columns = list(dict.fromkeys(list(dataframe.columns) + columns))
            return pd.DataFrame(columns=columns)

        dataframe = dataframe.copy()
        dataframe["website"] = dataframe["website"].astype(str)
        dataframe["object"] = dataframe["object"].astype(str)
        dataframe["fact"] = pd.to_numeric(dataframe["fact"], errors="coerce")
        dataframe = dataframe.dropna(subset=["fact"])
        if dataframe.empty:
            return pd.DataFrame(columns=["website", "fact", "object", "source_reliability", "global_truth"])

        if "batch_index" not in dataframe.columns and epoch is not None:
            dataframe["batch_index"] = int(epoch)

        dataframe = self.iteration(dataframe)
        self.last_iteration_count = 1

        if self.enable_zk_proof:
            source_scores = {website: float(score) for website, score in self.node_credibility.items()}
            dataframe = attach_grouped_proofs(
                dataframe,
                method_name="SenteTruthFinder[numeric]",
                source_scores=source_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
                extra_public_data={"numeric_tol": float(self.numeric_tol)},
            )

        return dataframe


class _NumericBasicTruthFinder(BasicTruthFinder):
    """Numeric adapter for BasicTruth that avoids text implication fitting."""

    def __init__(self, numeric_tol: float = DEFAULT_NUMERIC_MATCH_TOL, **kwargs):
        super().__init__(implication=lambda _f1, _f2: 0.0, **kwargs)
        self.numeric_tol = max(float(numeric_tol), 1e-8)
        self.implication = self._numeric_implication

    def _numeric_implication(self, left: float, right: float) -> float:
        return _scalar_numeric_similarity(left, right, tol=self.numeric_tol)

    def calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        def trustworthiness_score(x: float) -> float:
            x = min(0.999, max(0.001, float(x)))
            return -np.log(1 - x)

        rows = [
            (idx, float(row["fact"]), trustworthiness_score(float(row["trustworthiness"])))
            for idx, row in df.iterrows()
        ]

        for idx_i, fact_i, _ in rows:
            confidence = 0.0
            for _, fact_j, trust_score_j in rows:
                confidence += trust_score_j * self.implication(fact_j, fact_i)
            df.at[idx_i, "fact_confidence"] = float(confidence)
        return df

    def _estimate_global_truth_numeric(self, dataframe: pd.DataFrame) -> Dict[str, float]:
        truth_map: Dict[str, float] = {}
        for object_id in dataframe["object"].unique():
            obj_df = dataframe[dataframe["object"] == object_id]
            if obj_df.empty:
                continue
            best_idx = obj_df["fact_confidence"].astype(float).idxmax()
            truth_map[str(object_id)] = float(dataframe.loc[best_idx, "fact"])
        return truth_map

    def process_batch(
        self,
        dataframe: pd.DataFrame,
        implication_texts: Optional[List[str]] = None,
        initial_trustworthiness: float = 0.5,
    ) -> pd.DataFrame:
        del implication_texts

        if dataframe is None or dataframe.empty:
            columns = ["website", "fact", "object", "trustworthiness", "fact_confidence", "global_truth"]
            if dataframe is not None:
                columns = list(dict.fromkeys(list(dataframe.columns) + columns))
            return pd.DataFrame(columns=columns)

        dataframe = dataframe.copy()
        dataframe["website"] = dataframe["website"].astype(str)
        dataframe["object"] = dataframe["object"].astype(str)
        dataframe["fact"] = pd.to_numeric(dataframe["fact"], errors="coerce")
        dataframe = dataframe.dropna(subset=["fact"])
        if dataframe.empty:
            return pd.DataFrame(columns=["website", "fact", "object", "trustworthiness", "fact_confidence", "global_truth"])

        self.implication = self._numeric_implication
        init = self.website_trustworthiness if self.website_trustworthiness else initial_trustworthiness
        dataframe["trustworthiness"] = self._initialize_trustworthiness(
            dataframe,
            init,
            initial_trustworthiness,
        )
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))

        out = self.iteration(dataframe)
        self.last_iteration_count = 1

        truth_map = self._estimate_global_truth_numeric(out)
        out["global_truth"] = out["object"].map(truth_map)

        latest_scores = (
            out[["website", "trustworthiness"]]
            .drop_duplicates(subset=["website"], keep="last")
            .set_index("website")["trustworthiness"]
            .astype(float)
            .to_dict()
        )
        self.website_trustworthiness.update(latest_scores)
        out["source_reliability"] = out["website"].map(
            lambda website: float(self.website_trustworthiness[str(website)])
        )

        if self.enable_zk_proof:
            out = attach_grouped_proofs(
                out,
                method_name="BasicTruthFinder[numeric]",
                source_scores=latest_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
                extra_public_data={"numeric_tol": float(self.numeric_tol)},
            )
        return out


# ── 数值方法 ──────────────────────────────────────────────────────────────────

def _run_senfeed(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SenFeedTruth (CompositeTD): batch_index → timestamp，
    一次 train() 按 timestamp 顺序运行论文式 CompositeTD：
    - 首个 timestamp 做 iterative 初始化
    - 常规阶段做 incremental MAP
    - 当 source weight evolution 突变时回退到 batch 内迭代收敛
    """
    if df.empty:
        return {}
    inp = df[["website", "fact", "object", "batch_index"]].rename(
        columns={"batch_index": "timestamp"})
    try:
        model = SenFeedTruthDiscovery(enable_zk_proof=False)
        out   = model.train(inp)
        return _preds_numeric(out)
    except Exception as e:
        warnings.warn(f"SenFeedTruth 失败: {e}")
        return {}


def _run_decent(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    DecentTruth (CBI-TD): 按 batch_index 顺序调用 process_batch()，
    跨 batch 共享 source_reliability_history（增量更新 Eq.4-6）。
    """
    if df.empty:
        return {}
    model = EnhancedTruthFinder(enable_zk_proof=False)
    preds: Dict[Tuple[int, str], float] = {}
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object"]].copy()
        try:
            out = model.process_batch(grp, epoch=int(bi))
            out["batch_index"] = bi
            preds.update(_preds_numeric(out))
        except Exception as e:
            warnings.warn(f"DecentTruth batch {bi}: {e}")
    return preds


def _run_sente_num(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SenteTruth (数值模式): 直接对 float 计算相对距离相似度。
    按论文语义跨 batch 保留 node_credibility，每个 batch 只做一次更新。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], float] = {}
    model = _run_silently(
        lambda: _NumericSenteTruthFinder(enable_zk_proof=False)
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        try:
            def _process():
                return model.process_batch(grp)

            out = _run_silently(_process)
            preds.update(_preds_numeric(out))
        except Exception as e:
            warnings.warn(f"SenteTruth-num batch {bi}: {e}")
    return preds


def _run_basic_num(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    BasicTruth (数值模式): 直接对 float 计算数值蕴含与簇支持度。
    跨 batch 保留 website -> trustworthiness，不再拟合文本 implication。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], float] = {}
    model = _run_silently(
        lambda: _NumericBasicTruthFinder(enable_zk_proof=False)
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        try:
            def _process():
                return model.process_batch(grp)

            out = _run_silently(_process)
            preds.update(_preds_numeric(out))
        except Exception as e:
            warnings.warn(f"BasicTruth-num batch {bi}: {e}")
    return preds


# ── 文本方法 ──────────────────────────────────────────────────────────────────

def _run_sente_txt(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    SenteTruth (文本模式): SBERT 语义编码 + credibility 单步更新。
    按论文语义跨 batch 保留 node_credibility，并仅从本地缓存加载 SBERT。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], str] = {}
    model = _run_silently(
        lambda: SenteTruthFinder(
            enable_zk_proof=False,
            use_sbert=True,
            local_files_only=True,
        )
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        try:
            def _process():
                return model.process_batch(grp)

            out = _run_silently(_process)
            preds.update(_preds_text(out))
        except Exception as e:
            warnings.warn(f"SenteTruth-txt batch {bi}: {e}")
    return preds


def _run_basic_txt(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    BasicTruth (文本模式): TF-IDF implication + trustworthiness 迭代。
    跨 batch 保留 website -> trustworthiness，implication 仅用当前 prefix 拟合。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], str] = {}
    model = _run_silently(
        lambda: BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False)
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        prefix_texts = df[df["batch_index"] <= bi]["fact"].astype(str).tolist()
        try:
            def _process():
                return model.process_batch(
                    grp,
                    implication_texts=prefix_texts,
                )

            out = _run_silently(_process)
            truth_map = _basic_truth_map(out)
            for object_id, truth in truth_map.items():
                preds[(int(bi), object_id)] = truth
        except Exception as e:
            warnings.warn(f"BasicTruth-txt batch {bi}: {e}")
    return preds


def _run_semantic_num(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SemanticTruth（数值模式）:
    使用跨 batch 状态的鲁棒 IRLS 管道，累计源可信度。
    """
    if df.empty:
        return {}
    empty_text_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
    model = SemanticTruthFinder(enable_zk_proof=False, use_sbert=False)
    preds: Dict[Tuple[int, str], float] = {}
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        try:
            out_num, _ = model.process_batch(grp, empty_text_df.copy())
            preds.update(_preds_numeric(out_num))
        except Exception as e:
            warnings.warn(f"SemanticTruth-num batch {bi}: {e}")
    return preds


def _run_semantic_txt(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
) -> Tuple[Dict[Tuple[int, str], str], Dict[Tuple[int, str], str]]:
    """
    SemanticTruth（文本模式）:
    使用跨 batch 状态的几何中位数 + 共谋惩罚管道，clean/noisy 各自独立累计源可信度。
    """
    empty_numeric_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])

    def _run(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
        if df.empty:
            return {}
        model = SemanticTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
        preds: Dict[Tuple[int, str], str] = {}
        for bi in sorted(df["batch_index"].unique()):
            grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
            try:
                _, out_txt = model.process_batch(empty_numeric_df.copy(), grp)
                preds.update(_preds_text(out_txt))
            except Exception as e:
                warnings.warn(f"SemanticTruth-txt batch {bi}: {e}")
        return preds

    return _run(clean_df), _run(noisy_df)


# ═══════════════════════════════════════════════════════════════════════════════
# 三、指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def metric_rmse(
    x_preds: Dict[Tuple[int, str], float],
    y_preds: Dict[Tuple[int, str], float],
) -> float:
    """RMSE(X, Y): 共同 key 上的均方根误差。"""
    errs = [(x_preds[k] - y_preds[k]) ** 2
            for k in x_preds if k in y_preds]
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def metric_mae(
    x_preds: Dict[Tuple[int, str], float],
    y_preds: Dict[Tuple[int, str], float],
) -> float:
    """MAE(X, Y): 共同 key 上的平均绝对误差。"""
    errs = [abs(x_preds[k] - y_preds[k])
            for k in x_preds if k in y_preds]
    return float(np.mean(errs)) if errs else float("nan")


def metric_normalized_rmse(
    x_preds: Dict[Tuple[int, str], float],
    y_preds: Dict[Tuple[int, str], float],
    eps: float = 1e-12,
) -> float:
    """
    归一化 RMSE：先按每个对象的 |ground truth| 归一化误差，再计算 RMSE。
    这等价于按币种的相对误差聚合，可避免 BTC 的绝对量级主导指标。
    """
    errs = [((x_preds[k] - y_preds[k]) / max(abs(y_preds[k]), eps)) ** 2
            for k in x_preds if k in y_preds]
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def metric_normalized_mae(
    x_preds: Dict[Tuple[int, str], float],
    y_preds: Dict[Tuple[int, str], float],
    eps: float = 1e-12,
) -> float:
    """
    归一化 MAE：先按每个对象的 |ground truth| 归一化误差，再计算 MAE。
    """
    errs = [abs(x_preds[k] - y_preds[k]) / max(abs(y_preds[k]), eps)
            for k in x_preds if k in y_preds]
    return float(np.mean(errs)) if errs else float("nan")


def _filter_last_pct(
    d: Dict[Tuple[int, str], float],
    pct: float = 0.1,
) -> Dict[Tuple[int, str], float]:
    """
    保留最后 pct 比例的 batch（论文 Section VI：last 10% batches）。
    对齐在线协议：只取后段 batch 评估长期表现，避免冷启动阶段主导指标。
    """
    if not d:
        return d
    all_batches = sorted({k[0] for k in d})
    n_last = max(1, int(np.ceil(len(all_batches) * pct)))
    last_set = set(all_batches[-n_last:])
    return {k: v for k, v in d.items() if k[0] in last_set}


def metric_semantic_dist(
    x_preds: Dict[Tuple[int, str], str],
    y_preds: Dict[Tuple[int, str], str],
) -> float:
    """
    SemanticDist(X, Y) = 1 - mean(CosineSim(X_txt, Y_txt))
    基于 TF-IDF char-ngram 余弦相似度；值越大表示语义偏移越大。
    """
    keys = [k for k in x_preds if k in y_preds]
    if not keys:
        return float("nan")
    x_txts = [x_preds[k] for k in keys]
    y_txts = [y_preds[k] for k in keys]
    try:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat  = vect.fit_transform(x_txts + y_txts)
        n    = len(keys)
        sims = sk_cosine(mat[:n], mat[n:]).diagonal()
        return float(1.0 - np.mean(sims))
    except Exception:
        # 退化为精确匹配距离
        mismatches = sum(
            x.strip() != y.strip() for x, y in zip(x_txts, y_txts)
        )
        return float(mismatches) / len(keys)


def _extract_numbers(text: str) -> List[float]:
    """从文本中提取所有数字（含小数）。"""
    return extract_numeric_values(text)


def _numeric_text_sim(ref: str, pred: str, tol: float = DEFAULT_NUMERIC_MATCH_TOL) -> float:
    """
    数值相似度 B：比较两段文本中抽取的数字集合。
    对 ref 中每个数字，在 pred 中寻找相对误差 ≤ tol 的匹配。
    返回匹配比例 [0, 1]；ref 无数字时返回 1.0（不惩罚）。
    """
    ref_nums = _extract_numbers(ref)
    pred_nums = _extract_numbers(pred)
    return numeric_match_ratio(ref_nums, pred_nums, tol=tol)


def _numeric_text_score_for_accuracy(
    ref: str,
    pred: str,
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> Optional[float]:
    """
    Data Accuracy 专用数值通道。

    如果 clean/reference 文本本身不含数字，则认为该样本没有可验证的数值事实，
    关闭数值通道，仅按语义相似度评估，不再自动给 1.0 的数值奖励。
    """
    ref_nums = _extract_numbers(ref)
    if not ref_nums:
        return None
    pred_nums = _extract_numbers(pred)
    return numeric_match_ratio(ref_nums, pred_nums, tol=tol)


def metric_data_accuracy(
    x_preds: Dict[Tuple[int, str], str],
    y_preds: Dict[Tuple[int, str], str],
    threshold: float = 0.85,
    alpha: float = DEFAULT_SEMANTIC_NUMERIC_ALPHA,
    num_tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:
    """
    HybridAccuracy(X, Y) = proportion of (batch, object) where
        alpha * SBERT_cosine(X[k], Y[k]) + (1-alpha) * NumericSim(X[k], Y[k]) >= threshold

    A = SBERT 语义相似度（捕捉主题/结构一致性）
    B = 数字抽取匹配率（捕捉事实数值一致性）
    加权组合 alpha*A + (1-alpha)*B 兼顾语义和数值两个维度。

    X = 方法在干净数据上的输出（ground truth 代理）
    Y = 方法在含噪数据上的输出（预测结果）
    """
    combined = _hybrid_text_scores(x_preds, y_preds, alpha=alpha, num_tol=num_tol)
    if combined.size == 0:
        return float("nan")
    return float(np.mean(combined >= threshold))


def metric_mean_hybrid_score(
    x_preds: Dict[Tuple[int, str], str],
    y_preds: Dict[Tuple[int, str], str],
    alpha: float = DEFAULT_SEMANTIC_NUMERIC_ALPHA,
    num_tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:
    """
    返回 clean/noisy 输出之间的平均混合分数，不做阈值化。
    用于更平滑地观察文本鲁棒性随 batch 的变化。
    """
    combined = _hybrid_text_scores(x_preds, y_preds, alpha=alpha, num_tol=num_tol)
    if combined.size == 0:
        return float("nan")
    return float(np.mean(combined))


def _hybrid_text_scores(
    x_preds: Dict[Tuple[int, str], str],
    y_preds: Dict[Tuple[int, str], str],
    alpha: float = DEFAULT_SEMANTIC_NUMERIC_ALPHA,
    num_tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> np.ndarray:
    keys = [k for k in x_preds if k in y_preds]
    if not keys:
        return np.array([], dtype=float)
    x_txts = [x_preds[k] for k in keys]
    y_txts = [y_preds[k] for k in keys]

    encoder = _get_metric_sbert_encoder()
    if encoder is not None:
        x_emb = encoder.encode(x_txts, convert_to_numpy=True, show_progress_bar=False)
        y_emb = encoder.encode(y_txts, convert_to_numpy=True, show_progress_bar=False)
        x_norm = x_emb / (np.linalg.norm(x_emb, axis=1, keepdims=True) + 1e-12)
        y_norm = y_emb / (np.linalg.norm(y_emb, axis=1, keepdims=True) + 1e-12)
        sem_scores = np.clip((x_norm * y_norm).sum(axis=1), -1.0, 1.0)
    else:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat = vect.fit_transform(x_txts + y_txts)
        n = len(keys)
        sem_scores = np.clip(sk_cosine(mat[:n], mat[n:]).diagonal(), 0.0, 1.0)

    return np.array([
        blend_semantic_numeric_scores(
            sem_score=float(sem_score),
            num_score=_numeric_text_score_for_accuracy(x, y, tol=num_tol),
            alpha=alpha,
        )
        for sem_score, x, y in zip(sem_scores, x_txts, y_txts)
    ], dtype=float)


def _get_metric_sbert_encoder():
    """懒加载本地 SBERT 缓存，避免 import 阶段触发网络探测。"""
    global _SBERT_ENCODER, _SBERT_ENCODER_INIT_FAILED
    if _SBERT_ENCODER is not None:
        return _SBERT_ENCODER
    if _SBERT_ENCODER_INIT_FAILED:
        return None

    try:
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            model_source = snapshot_download(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                local_files_only=True,
            )
            _SBERT_ENCODER = SentenceTransformer(
                model_source,
                local_files_only=True,
            )
        finally:
            if old_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline
            if old_tf_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
    except Exception as e:
        warnings.warn(
            f"Failed to load local SBERT cache for text metrics: {e}. "
            "Falling back to char-ngram TF-IDF similarity."
        )
        _SBERT_ENCODER_INIT_FAILED = True
        return None

    return _SBERT_ENCODER


# ═══════════════════════════════════════════════════════════════════════════════
# 四、统一评估入口  (X-vs-Y 框架)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_numeric(
    noisy_df: pd.DataFrame,
    gt: Dict[Tuple[int, str], float],
    last_pct: float = 0.1,
    normalize: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    对各数值方法计算 RMSE/MAE(preds, gt)（对齐论文 Section VI）：
      preds = 方法在 noisy_df（含恶意节点）上的真值发现结果
      gt    = 干净数据中的真实值（由 build_gt_numeric 构建）
      当 last_pct < 1.0 时，仅评估最后 last_pct 比例的 batch
      当 last_pct = 1.0 时，评估全程所有 batch
      当 normalize = True 时，先按每个 (batch, object) 的 |gt| 做归一化，
      即计算相对误差上的 RMSE/MAE，避免不同币种的面值量级主导指标
    返回 {method: {rmse, mae}}。
    """
    runners = {
        "SenFeedTruth": _run_senfeed,
        "DecentTruth":  _run_decent,
        "SenteTruth":   _run_sente_num,
        "BasicTruth":   _run_basic_num,
    }
    out: Dict[str, Dict[str, float]] = {}
    gt_filtered = _filter_last_pct(gt, last_pct)
    rmse_metric = metric_normalized_rmse if normalize else metric_rmse
    mae_metric = metric_normalized_mae if normalize else metric_mae
    for name, fn in runners.items():
        print(f"    [{name}] numeric ...", end=" ", flush=True)
        preds = fn(noisy_df)
        preds_filtered = _filter_last_pct(preds, last_pct)
        r = rmse_metric(preds_filtered, gt_filtered)
        m = mae_metric(preds_filtered, gt_filtered)
        out[name] = {"rmse": r, "mae": m}
        print(f"RMSE={r:.4f}  MAE={m:.4f}")

    print("    [SemanticTruth] numeric ...", end=" ", flush=True)
    preds_sem = _run_semantic_num(noisy_df)
    preds_sem_filtered = _filter_last_pct(preds_sem, last_pct)
    r_sem = rmse_metric(preds_sem_filtered, gt_filtered)
    m_sem = mae_metric(preds_sem_filtered, gt_filtered)
    out["SemanticTruth"] = {"rmse": r_sem, "mae": m_sem}
    print(f"RMSE={r_sem:.4f}  MAE={m_sem:.4f}")
    return out


def eval_numeric_per_batch(
    noisy_df: pd.DataFrame,
    gt: Dict[Tuple[int, str], float],
    normalize: bool = False,
) -> Dict[str, Dict[int, float]]:
    """
    对各数值方法按 batch 逐个计算 RMSE(preds_b, gt_b)。
    当 normalize = True 时，对每个对象先除以该对象的 |gt|，
    再聚合为 batch 级 RMSE。
    返回 {method: {batch_index: rmse}}。
    """
    runners = {
        "SenFeedTruth": _run_senfeed,
        "DecentTruth":  _run_decent,
        "SenteTruth":   _run_sente_num,
        "BasicTruth":   _run_basic_num,
    }
    gt_by_batch: Dict[int, Dict[Tuple[int, str], float]] = {}
    for key, value in gt.items():
        gt_by_batch.setdefault(int(key[0]), {})[key] = value

    out: Dict[str, Dict[int, float]] = {}
    batch_indices = sorted(gt_by_batch)
    rmse_metric = metric_normalized_rmse if normalize else metric_rmse
    for name, fn in runners.items():
        print(f"    [{name}] numeric per-batch ...", end=" ", flush=True)
        preds = fn(noisy_df)
        preds_by_batch: Dict[int, Dict[Tuple[int, str], float]] = {}
        for key, value in preds.items():
            preds_by_batch.setdefault(int(key[0]), {})[key] = value
        out[name] = {
            bi: rmse_metric(preds_by_batch.get(bi, {}), gt_by_batch.get(bi, {}))
            for bi in batch_indices
        }
        print(f"{len(out[name])} batches")

    print("    [SemanticTruth] numeric per-batch ...", end=" ", flush=True)
    preds_sem = _run_semantic_num(noisy_df)
    preds_sem_by_batch: Dict[int, Dict[Tuple[int, str], float]] = {}
    for key, value in preds_sem.items():
        preds_sem_by_batch.setdefault(int(key[0]), {})[key] = value
    out["SemanticTruth"] = {
        bi: rmse_metric(preds_sem_by_batch.get(bi, {}), gt_by_batch.get(bi, {}))
        for bi in batch_indices
    }
    print(f"{len(out['SemanticTruth'])} batches")
    return out


def _group_preds_by_batch(
    preds: Dict[Tuple[int, str], object],
) -> Dict[int, Dict[Tuple[int, str], object]]:
    grouped: Dict[int, Dict[Tuple[int, str], object]] = {}
    for key, value in preds.items():
        grouped.setdefault(int(key[0]), {})[key] = value
    return grouped


def eval_text(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    accuracy_threshold: float = 0.85,
) -> Dict[str, Dict[str, float]]:
    """
    对各文本方法计算 HybridAccuracy(X, Y)：
      X = 方法在 clean_df 上的输出（ground truth 代理）
      Y = 方法在 noisy_df 上的输出（预测结果）
      HybridAccuracy = proportion of (batch, object) where
          0.5 * SBERT_cosine(X, Y) + 0.5 * NumericSim(X, Y) >= accuracy_threshold
    返回 {method: {data_accuracy}}。
    """
    runners = {
        "SenteTruth": _run_sente_txt,
        "BasicTruth": _run_basic_txt,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in runners.items():
        print(f"    [{name}] text ...", end=" ", flush=True)
        x = fn(clean_df)
        y = fn(noisy_df)
        acc = metric_data_accuracy(x, y, threshold=accuracy_threshold)
        out[name] = {"data_accuracy": acc}
        print(f"DataAcc={acc:.4f}")

    print("    [SemanticTruth] text ...", end=" ", flush=True)
    x_sem, y_sem = _run_semantic_txt(clean_df, noisy_df)
    acc_sem = metric_data_accuracy(x_sem, y_sem, threshold=accuracy_threshold)
    out["SemanticTruth"] = {"data_accuracy": acc_sem}
    print(f"DataAcc={acc_sem:.4f}")
    return out


def eval_text_per_batch(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    accuracy_threshold: float = 0.85,
) -> Dict[str, Dict[int, float]]:
    """
    对各文本方法按 batch 逐个计算 DataAccuracy(X_b, Y_b)。
    返回 {method: {batch_index: data_accuracy}}。
    """
    runners = {
        "SenteTruth": _run_sente_txt,
        "BasicTruth": _run_basic_txt,
    }
    batch_indices = sorted(
        set(clean_df.get("batch_index", pd.Series(dtype=int)).astype(int).tolist())
        | set(noisy_df.get("batch_index", pd.Series(dtype=int)).astype(int).tolist())
    )
    out: Dict[str, Dict[int, float]] = {}
    for name, fn in runners.items():
        print(f"    [{name}] text per-batch ...", end=" ", flush=True)
        x = fn(clean_df)
        y = fn(noisy_df)
        x_by_batch = _group_preds_by_batch(x)
        y_by_batch = _group_preds_by_batch(y)
        out[name] = {
            bi: metric_data_accuracy(
                x_by_batch.get(bi, {}),
                y_by_batch.get(bi, {}),
                threshold=accuracy_threshold,
            )
            for bi in batch_indices
        }
        print(f"{len(out[name])} batches")

    print("    [SemanticTruth] text per-batch ...", end=" ", flush=True)
    x_sem, y_sem = _run_semantic_txt(clean_df, noisy_df)
    x_sem_by_batch = _group_preds_by_batch(x_sem)
    y_sem_by_batch = _group_preds_by_batch(y_sem)
    out["SemanticTruth"] = {
        bi: metric_data_accuracy(
            x_sem_by_batch.get(bi, {}),
            y_sem_by_batch.get(bi, {}),
            threshold=accuracy_threshold,
        )
        for bi in batch_indices
    }
    print(f"{len(out['SemanticTruth'])} batches")
    return out


def eval_text_metrics_per_batch(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    accuracy_threshold: float = 0.85,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    对各文本方法按 batch 逐个计算：
      1. DataAccuracy(X_b, Y_b)
      2. MeanHybridScore(X_b, Y_b)
    返回 (accuracy_results, hybrid_score_results)。
    """
    runners = {
        "SenteTruth": _run_sente_txt,
        "BasicTruth": _run_basic_txt,
    }
    batch_indices = sorted(
        set(clean_df.get("batch_index", pd.Series(dtype=int)).astype(int).tolist())
        | set(noisy_df.get("batch_index", pd.Series(dtype=int)).astype(int).tolist())
    )
    acc_out: Dict[str, Dict[int, float]] = {}
    score_out: Dict[str, Dict[int, float]] = {}

    for name, fn in runners.items():
        print(f"    [{name}] text per-batch ...", end=" ", flush=True)
        x = fn(clean_df)
        y = fn(noisy_df)
        x_by_batch = _group_preds_by_batch(x)
        y_by_batch = _group_preds_by_batch(y)
        acc_out[name] = {}
        score_out[name] = {}
        for bi in batch_indices:
            x_b = x_by_batch.get(bi, {})
            y_b = y_by_batch.get(bi, {})
            acc_out[name][bi] = metric_data_accuracy(
                x_b,
                y_b,
                threshold=accuracy_threshold,
            )
            score_out[name][bi] = metric_mean_hybrid_score(x_b, y_b)
        print(f"{len(acc_out[name])} batches")

    print("    [SemanticTruth] text per-batch ...", end=" ", flush=True)
    x_sem, y_sem = _run_semantic_txt(clean_df, noisy_df)
    x_sem_by_batch = _group_preds_by_batch(x_sem)
    y_sem_by_batch = _group_preds_by_batch(y_sem)
    acc_out["SemanticTruth"] = {}
    score_out["SemanticTruth"] = {}
    for bi in batch_indices:
        x_b = x_sem_by_batch.get(bi, {})
        y_b = y_sem_by_batch.get(bi, {})
        acc_out["SemanticTruth"][bi] = metric_data_accuracy(
            x_b,
            y_b,
            threshold=accuracy_threshold,
        )
        score_out["SemanticTruth"][bi] = metric_mean_hybrid_score(x_b, y_b)
    print(f"{len(acc_out['SemanticTruth'])} batches")

    return acc_out, score_out


# ═══════════════════════════════════════════════════════════════════════════════
# 五、可视化 & CSV 导出
# ═══════════════════════════════════════════════════════════════════════════════

def plot_numeric(
    x_vals: List,
    results: Dict[str, Dict[str, List]],
    xlabel: str,
    title_prefix: str,
    out_dir: Path,
    metric_scope_label: str = "last 10% batches",
    metric_name_prefix: str = "",
) -> None:
    """绘制数值实验图（两列：RMSE | MAE）。"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    rmse_name = f"{metric_name_prefix}RMSE".strip()
    mae_name = f"{metric_name_prefix}MAE".strip()
    for m in NUM_METHODS:
        kw = dict(color=COLORS[m], marker=MARKERS[m], label=m,
                  linewidth=1.8, markersize=6)
        ax1.plot(x_vals, results[m]["rmse"], **kw)
        ax2.plot(x_vals, results[m]["mae"],  **kw)
    for ax, ylabel, suffix in [
        (ax1, f"{rmse_name} (vs ground truth, {metric_scope_label})", f"{rmse_name}（数值场景）"),
        (ax2, f"{mae_name} (vs ground truth, {metric_scope_label})",  f"{mae_name}（数值场景）"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} — {suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "numeric.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 已保存 {path}")


def plot_text(
    x_vals: List,
    results: Dict[str, Dict[str, List]],
    xlabel: str,
    title_prefix: str,
    out_dir: Path,
    threshold: float = 0.85,
) -> None:
    """绘制文本实验图（Data Accuracy）。"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in TEXT_METHODS:
        ax.plot(x_vals, results[m]["data_accuracy"],
                color=COLORS[m], marker=MARKERS[m], label=m,
                linewidth=1.8, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Data Accuracy (hybrid score ≥ {threshold:g})")
    ax.set_title(f"{title_prefix} — 文本数据准确率（混合场景）")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "mixed_text.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 已保存 {path}")


def save_csv(
    x_col: str,
    x_vals: List,
    num_res: Dict[str, Dict[str, List]],
    txt_res: Dict[str, Dict[str, List]],
    out_dir: Path,
    metric_scale: Optional[str] = None,
) -> None:
    """将所有指标合并为一个 CSV 文件。"""
    rows = []
    for i, xv in enumerate(x_vals):
        for m in NUM_METHODS:
            rows.append({
                x_col:       xv,
                "method":    m,
                "scene":     "numeric",
                "rmse":      num_res[m]["rmse"][i]     if i < len(num_res[m]["rmse"])     else float("nan"),
                "mae":       num_res[m]["mae"][i]      if i < len(num_res[m]["mae"])      else float("nan"),
                "data_accuracy":  float("nan"),
                "metric_scale": metric_scale,
            })
        for m in TEXT_METHODS:
            rows.append({
                x_col:       xv,
                "method":    m,
                "scene":     "mixed_text",
                "rmse":      float("nan"),
                "mae":       float("nan"),
                "data_accuracy":  txt_res[m]["data_accuracy"][i] if i < len(txt_res[m]["data_accuracy"]) else float("nan"),
                "metric_scale": metric_scale,
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


def plot_stage_metric_panels(
    stages: List[Tuple[str, int]],
    results_by_stage: Dict[str, Dict[str, Dict[int, float]]],
    methods: List[str],
    ylabel: str,
    title_prefix: str,
    out_path: Path,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot one subplot per injection stage, with batch on x-axis and metric on y-axis.
    """
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n_cols = max(1, len(stages))
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.8), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, (stage_label, injection_start) in zip(axes, stages):
        stage_results = results_by_stage.get(stage_label, {})
        for method in methods:
            series = stage_results.get(method, {})
            x_vals = sorted(series)
            y_vals = [series[x] for x in x_vals]
            ax.plot(
                x_vals,
                y_vals,
                color=COLORS[method],
                marker=MARKERS[method],
                label=method,
                linewidth=1.8,
                markersize=4,
            )
        ax.axvline(
            injection_start,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.set_title(f"{stage_label} (start={injection_start})")
        ax.set_xlabel("Batch")
        ax.grid(True, alpha=0.3)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {out_path}")


def plot_temporal_metric(
    results_by_method: Dict[str, Dict[int, float]],
    methods: List[str],
    ylabel: str,
    title: str,
    out_path: Path,
    event_lines: Optional[List[Tuple[int, str]]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot a single temporal metric curve panel with optional attack-event markers.
    """
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for method in methods:
        series = results_by_method.get(method, {})
        x_vals = sorted(series)
        y_vals = [series[x] for x in x_vals]
        ax.plot(
            x_vals,
            y_vals,
            color=COLORS[method],
            marker=MARKERS[method],
            label=method,
            linewidth=1.8,
            markersize=4,
        )

    if event_lines:
        for x_pos, label in event_lines:
            ax.axvline(
                x_pos,
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
            ax.text(
                x_pos + 1,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=9,
                color="#444444",
            )

    ax.set_xlabel("Batch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、E3 专用：构造注入数据集
# ═══════════════════════════════════════════════════════════════════════════════

def get_injection_stages(n_batches: int) -> List[Tuple[str, int]]:
    """
    Return coarse injection stages for Exp3.

    Fixed injection starts:

    - 早: 第 20 个 batch
    - 中: 第 70 个 batch
    - 晚: 第 120 个 batch
    """
    n_batches = max(int(n_batches), 0)
    if n_batches <= 0:
        return []

    candidates = [
        ("早", min(n_batches, 20)),
        ("中", min(n_batches, 70)),
        ("晚", min(n_batches, 120)),
    ]

    stages: List[Tuple[str, int]] = []
    seen_starts = set()
    for label, start in candidates:
        if start in seen_starts:
            continue
        stages.append((label, start))
        seen_starts.add(start)
    return stages

def build_injection_df(
    clean_batches: List[Dict],
    mal_batches:   List[Dict],
    inj_start:     int,
    n_total:       int,
    mode:          str,          # "numeric" | "text"
) -> pd.DataFrame:
    """
    构造注入数据集：
      batch_index < inj_start → 取干净数据
      batch_index >= inj_start → 取恶意数据
    mode 决定调用 to_numeric_df 还是 to_text_df。
    """
    fn = to_numeric_df if mode == "numeric" else to_text_df

    clean_idx = {b["batchIndex"]: b for b in clean_batches}
    mal_idx   = {b["batchIndex"]: b for b in mal_batches}

    parts: List[pd.DataFrame] = []
    for k in range(1, n_total + 1):
        src   = clean_idx if k < inj_start else mal_idx
        batch = src.get(k)
        if batch is None:
            continue
        parts.append(fn([batch]))

    valid = [p for p in parts if not p.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame(
        columns=["website", "object", "fact", "batch_index"])


def build_phase_switched_df(
    phase_batches: List[Tuple[int, List[Dict]]],
    n_total: int,
    mode: str,
) -> pd.DataFrame:
    """
    构造多阶段数据集：
      选择满足 start_batch <= k 的最后一个阶段数据源作为第 k 个 batch 的输入。

    Example:
      [(1, clean), (50, bad_2), (100, bad_4)]
    表示：
      1-49   -> clean
      50-99  -> bad_2
      100+   -> bad_4
    """
    fn = to_numeric_df if mode == "numeric" else to_text_df

    normalized_phases: List[Tuple[int, Dict[int, Dict]]] = []
    for start_batch, batches in sorted(phase_batches, key=lambda item: int(item[0])):
        normalized_phases.append(
            (int(start_batch), {int(batch["batchIndex"]): batch for batch in batches})
        )

    parts: List[pd.DataFrame] = []
    for k in range(1, int(n_total) + 1):
        current_batches: Optional[Dict[int, Dict]] = None
        for start_batch, indexed_batches in normalized_phases:
            if k >= start_batch:
                current_batches = indexed_batches
            else:
                break
        if current_batches is None:
            continue
        batch = current_batches.get(k)
        if batch is None:
            continue
        parts.append(fn([batch]))

    valid = [p for p in parts if not p.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame(
        columns=["website", "object", "fact", "batch_index"])
