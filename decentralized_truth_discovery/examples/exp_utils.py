"""
exp_utils.py — 四实验共享工具库
================================================
评估框架（对齐论文 Section VI）:
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
    create_implication_function_from_texts,
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

TEXT_COLLUDE_DEMO_FILES: Dict[int, Path] = COLLUDE_DEMO_FILES

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
    """Return numeric experiment files. mode='collude' (default) or 'rotate'."""
    if mode == "collude":
        return NUMERIC_COLLUDE_DEMO_FILES
    return NUMERIC_AGENT_DEMO_FILES


def get_text_demo_files() -> Dict[int, Path]:
    """Text experiments always use demo_*bad_collude.json files."""
    return TEXT_COLLUDE_DEMO_FILES


def get_shared_bad_counts() -> List[int]:
    """
    Bad-count values available to both numeric and text experiment datasets.
    This keeps combined experiment plots aligned on the same x-axis.
    """
    return sorted(set(NUMERIC_AGENT_DEMO_FILES) & set(TEXT_COLLUDE_DEMO_FILES))

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


# ── 数值方法 ──────────────────────────────────────────────────────────────────

def _run_senfeed(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SenFeedTruth (CompositeTD): batch_index → timestamp，
    一次 train() 跨 batch 增量处理（Eq.2/3 初始化 + Eq.6 MAP 更新）。
    """
    if df.empty:
        return {}
    inp = df[["website", "fact", "object", "batch_index"]].rename(
        columns={"batch_index": "timestamp"})
    try:
        model = SenFeedTruthDiscovery(mode="composite", enable_zk_proof=False)
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
    SenteTruth (数值模式): 数值→字符串，TF-IDF 相似度选多数值，再还原为 float。
    按论文语义跨 batch 保留 node_credibility，每个 batch 只做一次更新。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], float] = {}
    model = _run_silently(
        lambda: SenteTruthFinder(enable_zk_proof=False, use_sbert=False)
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi].copy()
        grp["fact"] = grp["fact"].map(lambda x: f"{float(x):.8g}")
        try:
            def _process():
                return model.process_batch(grp[["website", "fact", "object", "batch_index"]])

            out = _run_silently(_process)
            for _, row in out.drop_duplicates("object").iterrows():
                v = row.get("global_truth")
                if pd.notna(v):
                    try:
                        preds[(int(bi), str(row["object"]))] = float(str(v))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            warnings.warn(f"SenteTruth-num batch {bi}: {e}")
    return preds


def _run_basic_num(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    BasicTruth (数值模式): 数值→字符串，TF-IDF implication 选置信度最高值。
    跨 batch 保留 website -> trustworthiness，implication 仅用当前 prefix 拟合。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], float] = {}
    model = _run_silently(
        lambda: BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False)
    )
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi].copy()
        grp["fact"] = grp["fact"].map(lambda x: f"{float(x):.8g}")
        prefix_texts = (
            df[df["batch_index"] <= bi]["fact"]
            .map(lambda x: f"{float(x):.8g}")
            .astype(str)
            .tolist()
        )
        try:
            def _process():
                return model.process_batch(
                    grp[["website", "fact", "object", "batch_index"]],
                    implication_texts=prefix_texts,
                )

            out = _run_silently(_process)
            truth_map = _basic_truth_map(out)
            for object_id, truth in truth_map.items():
                try:
                    preds[(int(bi), object_id)] = float(truth)
                except (ValueError, TypeError):
                    pass
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
    对齐论文：只用收敛后的稳态结果评估长期精度，过滤冷启动阶段的噪声。
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
    keys = [k for k in x_preds if k in y_preds]
    if not keys:
        return float("nan")
    x_txts = [x_preds[k] for k in keys]
    y_txts = [y_preds[k] for k in keys]

    # A: SBERT 语义相似度；若本地缓存模型不可用，则退化为 char-ngram TF-IDF 余弦
    encoder = _get_metric_sbert_encoder()
    if encoder is not None:
        x_emb = encoder.encode(x_txts, convert_to_numpy=True, show_progress_bar=False)
        y_emb = encoder.encode(y_txts, convert_to_numpy=True, show_progress_bar=False)
        x_norm = x_emb / (np.linalg.norm(x_emb, axis=1, keepdims=True) + 1e-12)
        y_norm = y_emb / (np.linalg.norm(y_emb, axis=1, keepdims=True) + 1e-12)
        A = np.clip((x_norm * y_norm).sum(axis=1), -1.0, 1.0)
    else:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat = vect.fit_transform(x_txts + y_txts)
        n = len(keys)
        A = np.clip(sk_cosine(mat[:n], mat[n:]).diagonal(), 0.0, 1.0)

    # B: 数值匹配率。若 clean/reference 本身不含数字，则关闭数值通道，仅按语义评估。
    combined = np.array([
        blend_semantic_numeric_scores(
            sem_score=float(sem_score),
            num_score=_numeric_text_score_for_accuracy(x, y, tol=num_tol),
            alpha=alpha,
        )
        for sem_score, x, y in zip(A, x_txts, y_txts)
    ])
    return float(np.mean(combined >= threshold))


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
) -> None:
    """绘制数值实验图（两列：RMSE | MAE）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in NUM_METHODS:
        kw = dict(color=COLORS[m], marker=MARKERS[m], label=m,
                  linewidth=1.8, markersize=6)
        ax1.plot(x_vals, results[m]["rmse"], **kw)
        ax2.plot(x_vals, results[m]["mae"],  **kw)
    for ax, ylabel, suffix in [
        (ax1, f"RMSE (vs ground truth, {metric_scope_label})", "RMSE（数值场景）"),
        (ax2, f"MAE (vs ground truth, {metric_scope_label})",  "MAE（数值场景）"),
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
) -> None:
    """绘制文本实验图（Data Accuracy）。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in TEXT_METHODS:
        ax.plot(x_vals, results[m]["data_accuracy"],
                color=COLORS[m], marker=MARKERS[m], label=m,
                linewidth=1.8, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Data Accuracy (SBERT cosine ≥ 0.85)")
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
            })
        for m in TEXT_METHODS:
            rows.append({
                x_col:       xv,
                "method":    m,
                "scene":     "mixed_text",
                "rmse":      float("nan"),
                "mae":       float("nan"),
                "data_accuracy":  txt_res[m]["data_accuracy"][i] if i < len(txt_res[m]["data_accuracy"]) else float("nan"),
            })
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 已保存 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、E3 专用：构造注入数据集
# ═══════════════════════════════════════════════════════════════════════════════

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
