#!/usr/bin/env python3
"""
四种真值发现方法对比基准实验
================================================
方法: SenFeedTruth | SenteTruth | DecentTruth | BasicTruth
实验:
  E1 - 改变恶意节点比例 (0% → 40%)
  E2 - 随着 Batch 数量变化 (1 → N)
  E3 - 随着恶意节点注入时间变化
  E4 - 随着 Agent 数量变化 (3 → 10)
数据类型: 纯数值 (Numeric) + 文本离散 (Text-Discrete)
数据源: decentralized_truth_discovery/data_agent-10/

用法:
  python four_methods_benchmark.py            # 运行全部实验
  python four_methods_benchmark.py --exp 1    # 只运行实验1
  python four_methods_benchmark.py --exp 2,3  # 运行实验2和3
"""
from __future__ import annotations

import argparse
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

# ── 项目根目录 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decentralized_truth_discovery.core import (
    BasicTruthFinder,
    EnhancedTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
    create_implication_function_from_texts,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 常量与路径
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = ROOT / "decentralized_truth_discovery" / "data_agent-10"
BASE_OUT_DIR = Path(__file__).parent / "outputs" / "four_methods_benchmark"
OUT_DIR = BASE_OUT_DIR / "collude"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#: 数值实验文件  {bad_count → filepath}
NUM_FILES: Dict[int, Path] = {
    0: DATA_DIR / "num_demo.json",
    1: DATA_DIR / "num_demo_1bad_agent.json",
    2: DATA_DIR / "num_demo_2bad_agent.json",
    3: DATA_DIR / "num_demo_3bad_agent.json",
    4: DATA_DIR / "num_demo_4bad_agent.json",
}
#: 文本实验文件  {bad_count → filepath}
TXT_FILES_AGENT: Dict[int, Path] = {
    0: DATA_DIR / "demo.json",
    1: DATA_DIR / "demo_1bad_agent.json",
    2: DATA_DIR / "demo_2bad_agent.json",
    3: DATA_DIR / "demo_3bad_agent.json",
    4: DATA_DIR / "demo_4bad_agent.json",
}

TXT_FILES_COLLUDE: Dict[int, Path] = {
    0: DATA_DIR / "demo.json",
    1: DATA_DIR / "demo_1bad_collude.json",
    2: DATA_DIR / "demo_2bad_collude.json",
    3: DATA_DIR / "demo_3bad_collude.json",
    4: DATA_DIR / "demo_4bad_collude.json",
}

TXT_FILES = TXT_FILES_COLLUDE

N_AGENTS = 10
METHODS  = ["SenFeedTruth", "SenteTruth", "DecentTruth", "BasicTruth"]
COLORS   = {
    "SenFeedTruth": "#e74c3c",
    "SenteTruth":   "#2ecc71",
    "DecentTruth":  "#3498db",
    "BasicTruth":   "#f39c12",
}
MARKERS  = {
    "SenFeedTruth": "o",
    "SenteTruth":   "s",
    "DecentTruth":  "^",
    "BasicTruth":   "D",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_batches(filepath: Path) -> List[Dict]:
    """加载 JSON 文件，返回 batches 列表。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["job"]["batches"]


def get_text_files(attack_profile: str) -> Dict[int, Path]:
    profile = str(attack_profile).strip().lower()
    if profile == "agent":
        return TXT_FILES_AGENT
    if profile == "collude":
        return TXT_FILES_COLLUDE
    raise ValueError(f"Unsupported text attack profile: {attack_profile}")


def batches_to_numeric_df(
    batches: List[Dict],
    agent_subset: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """
    解析 batches → DataFrame[website, fact, object, batch_index]
    只保留数值类型 response（dict 格式，如 BTC/ETH/DOGE 价格）。
    每个币种作为独立 object 行。
    """
    rows: List[Dict] = []
    for b in (batches[:max_batches] if max_batches else batches):
        for item in b["items"]:
            if agent_subset and item["agent"] not in agent_subset:
                continue
            resp = item["response"]
            if isinstance(resp, dict):
                for coin, price in resp.items():
                    rows.append({
                        "website":     item["agent"],
                        "object":      str(coin),
                        "fact":        float(price),
                        "batch_index": int(b["batchIndex"]),
                    })
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["website", "object", "fact", "batch_index"]))


def batches_to_text_df(
    batches: List[Dict],
    agent_subset: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """
    解析 batches → DataFrame[website, fact, object, batch_index]
    只保留文本类型 response（str 格式，如新闻摘要）。
    """
    rows: List[Dict] = []
    for b in (batches[:max_batches] if max_batches else batches):
        for item in b["items"]:
            if agent_subset and item["agent"] not in agent_subset:
                continue
            resp = item["response"]
            if isinstance(resp, str):
                rows.append({
                    "website":     item["agent"],
                    "object":      str(item["object"]),
                    "fact":        resp,
                    "batch_index": int(b["batchIndex"]),
                })
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["website", "object", "fact", "batch_index"]))


def build_gt_numeric(
    clean_batches: List[Dict],
    max_batches: Optional[int] = None,
) -> Dict[Tuple[int, str], float]:
    """
    从无恶意节点的干净数据中提取数值真值。
    返回 {(batch_index, coin) → true_price}。
    """
    gt: Dict[Tuple[int, str], float] = {}
    for b in (clean_batches[:max_batches] if max_batches else clean_batches):
        for item in b["items"]:
            if isinstance(item["response"], dict):
                for coin, price in item["response"].items():
                    key = (int(b["batchIndex"]), str(coin))
                    if key not in gt:
                        gt[key] = float(price)
                break  # 干净数据中所有 agent 值相同，取第一个即可
    return gt


def build_gt_text(
    clean_batches: List[Dict],
    max_batches: Optional[int] = None,
) -> Dict[Tuple[int, str], str]:
    """
    从干净数据中提取文本真值。
    返回 {(batch_index, object_name) → correct_text}。
    """
    gt: Dict[Tuple[int, str], str] = {}
    for b in (clean_batches[:max_batches] if max_batches else clean_batches):
        for item in b["items"]:
            if isinstance(item["response"], str):
                key = (int(b["batchIndex"]), str(item["object"]))
                if key not in gt:
                    gt[key] = item["response"]
    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# 二、方法运行器
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── 公共辅助 ────────────────────────────────────

def _extract_num_preds(out_df: pd.DataFrame, bi: int) -> Dict[Tuple[int, str], float]:
    """从方法输出 DataFrame 中提取数值预测 {(batch_index, object) → value}。"""
    preds: Dict[Tuple[int, str], float] = {}
    if "global_truth" not in out_df.columns:
        return preds
    for _, row in out_df.drop_duplicates("object").iterrows():
        try:
            preds[(bi, str(row["object"]))] = float(row["global_truth"])
        except (ValueError, TypeError):
            pass
    return preds


def _extract_txt_preds(out_df: pd.DataFrame, bi: int) -> Dict[Tuple[int, str], str]:
    """从方法输出 DataFrame 中提取文本预测 {(batch_index, object) → text}。"""
    preds: Dict[Tuple[int, str], str] = {}
    if "global_truth" not in out_df.columns:
        return preds
    for _, row in out_df.drop_duplicates("object").iterrows():
        val = row["global_truth"]
        if pd.notna(val):
            preds[(bi, str(row["object"]))] = str(val)
    return preds


def _run_silently(fn):
    """Suppress noisy model stdout during experiment loops."""
    with redirect_stdout(io.StringIO()):
        return fn()


# ───────────────────────────── 数值方法 ──────────────────────────────────────

def run_senfeed_numeric(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SenFeedTruth 数值版：利用 timestamp=batch_index 进行跨 batch 增量学习。
    CompositeTD 模式：t=0 迭代式初始化，后续 MAP 增量更新，权重波动大时回退。
    """
    if df.empty:
        return {}
    inp = df[["website", "fact", "object", "batch_index"]].copy()
    inp = inp.rename(columns={"batch_index": "timestamp"})
    try:
        model = SenFeedTruthDiscovery(mode="composite", enable_zk_proof=False)
        out = model.train(inp)
        preds: Dict[Tuple[int, str], float] = {}
        for _, row in out.iterrows():
            bi_val = int(row.get("timestamp", 0))
            try:
                preds[(bi_val, str(row["object"]))] = float(row["global_truth"])
            except (ValueError, TypeError):
                pass
        return preds
    except Exception as e:
        warnings.warn(f"SenFeedTruth numeric failed: {e}")
        return {}


def run_decent_numeric(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    DecentTruth 数值版：CBI-TD，每个 batch 独立训练（Eq.3-9）。
    跨 batch 共享 source_reliability_history 实现增量。
    """
    if df.empty:
        return {}
    model = EnhancedTruthFinder(enable_zk_proof=False)
    preds: Dict[Tuple[int, str], float] = {}
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object"]].copy()
        try:
            out = model.process_batch(grp, epoch=int(bi))
            preds.update(_extract_num_preds(out, int(bi)))
        except Exception as e:
            warnings.warn(f"DecentTruth numeric batch {bi}: {e}")
    return preds


def run_sente_numeric(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    SenteTruth 数值版：数值转字符串，TF-IDF 相似度选多数值，再还原为浮点。
    （SenteTruth 原为文本设计；此处用 TF-IDF 模拟数值字符串的相似度分析）
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
        grp["fact"] = grp["fact"].map(lambda x: f"{float(x):.6g}")
        try:
            out = model.process_batch(grp[["website", "fact", "object", "batch_index"]].copy())
            for _, row in out.drop_duplicates("object").iterrows():
                gt_val = row.get("global_truth")
                if pd.notna(gt_val):
                    try:
                        preds[(int(bi), str(row["object"]))] = float(str(gt_val))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            warnings.warn(f"SenteTruth numeric batch {bi}: {e}")
    return preds


def run_basic_numeric(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """
    BasicTruth 数值版：数值转字符串，TF-IDF implication 选置信度最高值。
    跨 batch 保留 website -> trustworthiness，implication 仅用当前 prefix 拟合。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], float] = {}
    model = BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=True)
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi].copy()
        grp["fact"] = grp["fact"].map(lambda x: f"{float(x):.6g}")
        prefix_texts = (
            df[df["batch_index"] <= bi]["fact"]
            .map(lambda x: f"{float(x):.6g}")
            .astype(str)
            .tolist()
        )
        try:
            out = model.process_batch(
                grp[["website", "fact", "object", "batch_index"]].copy(),
                implication_texts=prefix_texts,
            )
            for _, row in out.drop_duplicates("object").iterrows():
                gt_val = row.get("global_truth")
                if pd.notna(gt_val):
                    try:
                        preds[(int(bi), str(row["object"]))] = float(str(gt_val))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            warnings.warn(f"BasicTruth numeric batch {bi}: {e}")
    return preds


# ───────────────────────────── 文本方法 ──────────────────────────────────────

def _get_text_encoder():
    """懒加载 SBERT 编码器，失败时返回 None（将使用 TF-IDF 备选）。"""
    try:
        from sentence_transformers import SentenceTransformer
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            model_source = snapshot_download(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                local_files_only=True,
            )
            return SentenceTransformer(model_source, local_files_only=True)
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
        warnings.warn(f"SBERT 加载失败，将使用 TF-IDF 代替: {e}")
        return None


def _compute_proxy_scores(
    texts: List[str],
    encoder,
) -> List[float]:
    """
    将文本列表计算为"共识代理分数"（avgCosine）。
    用于将 SenFeedTruth/DecentTruth（纯数值算法）适配到文本数据。
    代理分数 ∈ [-1, 1]，越高表示与多数节点越一致。
    """
    n = len(texts)
    if n <= 1:
        return [1.0] * n
    if encoder is not None:
        embs = encoder.encode(texts, normalize_embeddings=True)
        sim_mat = embs @ embs.T                                # (n, n)
    else:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat = vect.fit_transform(texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        mat = mat / norms
        sim_mat = mat @ mat.T
    scores = []
    for i in range(n):
        avg = (sim_mat[i].sum() - 1.0) / max(n - 1, 1)
        scores.append(float(avg))
    return scores


def _proxy_df_for_text(
    df: pd.DataFrame,
    encoder,
) -> pd.DataFrame:
    """
    将文本 DataFrame 转换为代理数值 DataFrame。
    保留 _text 列以便最终还原所选文本。
    """
    rows: List[Dict] = []
    for (bi, obj), grp in df.groupby(["batch_index", "object"]):
        texts  = grp["fact"].tolist()
        agents = grp["website"].tolist()
        scores = _compute_proxy_scores(texts, encoder)
        for agent, score, text in zip(agents, scores, texts):
            rows.append({
                "website":     agent,
                "object":      str(obj),
                "fact":        score,
                "_text":       text,
                "batch_index": int(bi),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["website", "object", "fact", "_text", "batch_index"])


def run_senfeed_text(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    SenFeedTruth 文本版：
    SBERT avgCosine 作为代理数值 → SenFeedTruth 计算 source_weight
    → 选取权重最高节点的原始文本。
    """
    if df.empty:
        return {}
    encoder = _get_text_encoder()
    proxy   = _proxy_df_for_text(df, encoder)
    preds: Dict[Tuple[int, str], str] = {}

    for bi in sorted(proxy["batch_index"].unique()):
        grp = proxy[proxy["batch_index"] == bi]
        for obj, obj_grp in grp.groupby("object"):
            inp = obj_grp[["website", "fact", "object"]].copy()
            try:
                model = SenFeedTruthDiscovery(mode="composite", enable_zk_proof=False)
                out   = model.train(inp)
                w_map = {row["website"]: row.get("source_weight", 1.0)
                         for _, row in out.iterrows()}
                best  = max(w_map, key=w_map.get)
                text  = obj_grp[obj_grp["website"] == best]["_text"].iloc[0]
            except Exception as e:
                warnings.warn(f"SenFeedTruth text {bi}/{obj}: {e}")
                # 备选：取最高代理分数节点的文本
                best_row = obj_grp.loc[obj_grp["fact"].idxmax()]
                text = best_row["_text"]
            preds[(int(bi), str(obj))] = text
    return preds


def run_decent_text(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    DecentTruth 文本版：
    SBERT avgCosine 代理 → DecentTruth CBI-TD 计算 source_reliability
    → 选取可靠度最高节点的原始文本。
    """
    if df.empty:
        return {}
    encoder = _get_text_encoder()
    proxy   = _proxy_df_for_text(df, encoder)
    preds: Dict[Tuple[int, str], str] = {}
    model = EnhancedTruthFinder(enable_zk_proof=False)  # 跨 batch 共享 history

    for bi in sorted(proxy["batch_index"].unique()):
        grp = proxy[proxy["batch_index"] == bi]
        for obj, obj_grp in grp.groupby("object"):
            inp = obj_grp[["website", "fact", "object"]].copy()
            try:
                out   = model.process_batch(inp, epoch=int(bi))
                r_map = {row["website"]: row.get("source_reliability", 0.5)
                         for _, row in out.iterrows()}
                best  = max(r_map, key=r_map.get)
                text  = obj_grp[obj_grp["website"] == best]["_text"].iloc[0]
            except Exception as e:
                warnings.warn(f"DecentTruth text {bi}/{obj}: {e}")
                best_row = obj_grp.loc[obj_grp["fact"].idxmax()]
                text = best_row["_text"]
            preds[(int(bi), str(obj))] = text
    return preds


def run_sente_text(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    SenteTruth 文本版：原生 SBERT + 单步 credibility 更新。
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
            out = model.process_batch(grp)
            preds.update(_extract_txt_preds(out, int(bi)))
        except Exception as e:
            warnings.warn(f"SenteTruth text batch {bi}: {e}")
    return preds


def run_basic_text(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    """
    BasicTruth 文本版：TF-IDF implication + trustworthiness 迭代真值发现。
    跨 batch 保留 website -> trustworthiness，implication 仅用当前 prefix 拟合。
    """
    if df.empty:
        return {}
    preds: Dict[Tuple[int, str], str] = {}
    model = BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=True)
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        prefix_texts = df[df["batch_index"] <= bi]["fact"].astype(str).tolist()
        try:
            out = model.process_batch(
                grp,
                implication_texts=prefix_texts,
            )
            preds.update(_extract_txt_preds(out, int(bi)))
        except Exception as e:
            warnings.warn(f"BasicTruth text batch {bi}: {e}")
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
# 三、评价指标
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_last_pct(d: Dict, pct: float = 0.1) -> Dict:
    """
    保留最后 pct 比例的 batch（对齐论文 Section VI：last 10% batches）。
    key 格式为 (batch_index, object)。
    """
    if not d:
        return d
    all_batches = sorted({k[0] for k in d})
    n_last = max(1, int(np.ceil(len(all_batches) * pct)))
    last_set = set(all_batches[-n_last:])
    return {k: v for k, v in d.items() if k[0] in last_set}


def metric_rmse(preds: Dict, gt: Dict) -> float:
    """均方根误差（数值实验）。"""
    errs = [(preds[k] - gt[k]) ** 2 for k in gt if k in preds]
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def metric_mae(preds: Dict, gt: Dict) -> float:
    """平均绝对误差（数值实验）。"""
    errs = [abs(preds[k] - gt[k]) for k in gt if k in preds]
    return float(np.mean(errs)) if errs else float("nan")


def metric_nrmse(preds: Dict, gt: Dict) -> float:
    """归一化 RMSE，除以 gt 均值（便于跨价格量纲比较）。"""
    errs = [(preds[k] - gt[k]) ** 2 for k in gt if k in preds]
    if not errs:
        return float("nan")
    gt_mean = float(np.mean([gt[k] for k in gt if k in preds]))
    if gt_mean == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errs))) / gt_mean


def metric_accuracy(
    preds: Dict[Tuple, str],
    gt: Dict[Tuple, str],
    sim_threshold: float = 0.80,
) -> float:
    """
    文本准确率（Text Accuracy）。
    判定标准：TF-IDF char-ngram 余弦相似度 ≥ sim_threshold 则视为"正确"。
    """
    keys = [k for k in gt if k in preds]
    if not keys:
        return float("nan")
    gt_texts   = [gt[k]   for k in keys]
    pred_texts = [preds[k] for k in keys]
    try:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat  = vect.fit_transform(gt_texts + pred_texts)
        n    = len(keys)
        sims = sk_cosine(mat[:n], mat[n:]).diagonal()
        return float(np.mean(sims >= sim_threshold))
    except Exception:
        exact = sum(g.strip() == p.strip() for g, p in zip(gt_texts, pred_texts))
        return float(exact) / len(keys)


# ═══════════════════════════════════════════════════════════════════════════════
# 四、统一评估接口
# ═══════════════════════════════════════════════════════════════════════════════

NumericRes = Dict[str, Dict[str, float]]   # {method: {rmse, mae, nrmse}}
TextRes    = Dict[str, Dict[str, float]]   # {method: {accuracy}}


def eval_all_numeric(df: pd.DataFrame, gt: Dict, last_pct: float = 0.1) -> NumericRes:
    """
    运行全部四种方法（数值模式），返回各方法指标。
    对齐论文 Section VI：仅对最后 last_pct 比例的 batch 计算 RMSE/MAE（默认 10%）。
    """
    runners = {
        "SenFeedTruth": run_senfeed_numeric,
        "SenteTruth":   run_sente_numeric,
        "DecentTruth":  run_decent_numeric,
        "BasicTruth":   run_basic_numeric,
    }
    gt_filtered = _filter_last_pct(gt, last_pct)
    result: NumericRes = {}
    for name, fn in runners.items():
        print(f"    [{name}] numeric ...", end=" ", flush=True)
        preds = fn(df)
        preds_filtered = _filter_last_pct(preds, last_pct)
        result[name] = {
            "rmse":  metric_rmse(preds_filtered, gt_filtered),
            "mae":   metric_mae(preds_filtered, gt_filtered),
            "nrmse": metric_nrmse(preds_filtered, gt_filtered),
        }
        print(f"RMSE={result[name]['rmse']:.4f}  MAE={result[name]['mae']:.4f}")
    return result


def eval_all_text(df: pd.DataFrame, gt: Dict) -> TextRes:
    """运行全部四种方法（文本模式），返回各方法准确率。"""
    runners = {
        "SenFeedTruth": run_senfeed_text,
        "SenteTruth":   run_sente_text,
        "DecentTruth":  run_decent_text,
        "BasicTruth":   run_basic_text,
    }
    result: TextRes = {}
    for name, fn in runners.items():
        print(f"    [{name}] text ...", end=" ", flush=True)
        preds = fn(df)
        acc = metric_accuracy(preds, gt)
        result[name] = {"accuracy": acc}
        print(f"Accuracy={acc:.4f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 五、四个实验
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────── E1: 改变恶意节点比例 ────────────────────────────────────

def exp1_malicious_ratio() -> None:
    """
    实验1: 改变恶意节点比例 (0%, 10%, 20%, 30%, 40%)

    设计说明：
    - 10 个 agent，恶意节点数 k ∈ {0,1,2,3,4}，即比例 {0,10,20,30,40}%
    - 使用对应 num_demo_Xbad / demo_Xbad 文件
    - 对所有 batch 一次性处理，取全局平均 RMSE/MAE/Accuracy
    - 图表：折线图，x=恶意节点比例，y=指标
    """
    print("\n" + "="*60)
    print("E1: 恶意节点比例实验")
    print("="*60)

    clean_num = load_batches(NUM_FILES[0])
    clean_txt = load_batches(TXT_FILES[0])
    gt_num = build_gt_numeric(clean_num)
    gt_txt = build_gt_text(clean_txt)

    bad_counts = sorted(NUM_FILES.keys())   # [0,1,2,3,4]
    ratios     = [k / N_AGENTS * 100 for k in bad_counts]

    num_res: Dict[str, Dict[str, List]] = {m: {"rmse": [], "mae": [], "nrmse": []} for m in METHODS}
    txt_res: Dict[str, Dict[str, List]] = {m: {"accuracy": []} for m in METHODS}

    for bad_k, ratio in zip(bad_counts, ratios):
        print(f"\n  bad={bad_k} ({ratio:.0f}%)")
        num_df = batches_to_numeric_df(load_batches(NUM_FILES[bad_k]))
        txt_df = batches_to_text_df(load_batches(TXT_FILES[bad_k]))

        for m, v in eval_all_numeric(num_df, gt_num).items():
            for metric, val in v.items():
                num_res[m][metric].append(val)
        for m, v in eval_all_text(txt_df, gt_txt).items():
            txt_res[m]["accuracy"].append(v["accuracy"])

    _plot_e1(ratios, num_res, txt_res)
    _save_csv("malicious_ratio_pct", ratios, num_res, txt_res, "e1_malicious_ratio.csv")
    print("\n[E1] 完成")


# ─────────────────── E2: 随着 Batch 变化 ─────────────────────────────────────

def exp2_batch_variation() -> None:
    """
    实验2: 随着 batch 数量增加衡量四种方法的表现

    设计说明：
    - 固定 40% 恶意节点（最严苛场景）
    - 逐步增加 batch 数量 k = 1, 2, ..., N
    - 每一步处理 batches 1..k，计算累计平均指标
    - 图表：折线图，x=batch 数量，y=指标
    - 目的：观察随数据积累各方法的收敛与鲁棒性趋势
    """
    print("\n" + "="*60)
    print("E2: Batch 数量变化实验")
    print("="*60)

    bad_k     = 4   # 40% malicious
    clean_num = load_batches(NUM_FILES[0])
    clean_txt = load_batches(TXT_FILES[0])
    mal_num   = load_batches(NUM_FILES[bad_k])
    mal_txt   = load_batches(TXT_FILES[bad_k])

    n_batches   = min(len(clean_num), len(mal_num))
    batch_range = list(range(1, n_batches + 1))

    num_res: Dict[str, Dict[str, List]] = {m: {"rmse": [], "mae": [], "nrmse": []} for m in METHODS}
    txt_res: Dict[str, Dict[str, List]] = {m: {"accuracy": []} for m in METHODS}

    for k in batch_range:
        print(f"\n  batches 1..{k}")
        num_df = batches_to_numeric_df(mal_num, max_batches=k)
        gt_num = build_gt_numeric(clean_num, max_batches=k)
        txt_df = batches_to_text_df(mal_txt, max_batches=k)
        gt_txt = build_gt_text(clean_txt, max_batches=k)

        for m, v in eval_all_numeric(num_df, gt_num).items():
            for metric, val in v.items():
                num_res[m][metric].append(val)

        if not txt_df.empty:
            for m, v in eval_all_text(txt_df, gt_txt).items():
                txt_res[m]["accuracy"].append(v["accuracy"])
        else:
            for m in METHODS:
                txt_res[m]["accuracy"].append(float("nan"))

    _plot_e2(batch_range, num_res, txt_res)
    _save_csv("batch_count", batch_range, num_res, txt_res, "e2_batch_variation.csv")
    print("\n[E2] 完成")


# ─────────────────── E3: 恶意节点注入时间 ────────────────────────────────────

def exp3_injection_time() -> None:
    """
    实验3: 随着恶意节点注入时间（开始攻击的 batch）变化

    设计说明：
    - 总 batch 数 N，注入时间点 t ∈ {1, 2, ..., N}
    - t 之前 (batch < t)：使用干净数据（0 恶意节点）
    - t 开始 (batch ≥ t)：使用恶意数据（40% 恶意节点）
    - 注入越晚：方法已积累足够置信度，抵御攻击能力越强
    - 图表：折线图，x=注入开始 batch，y=全局最终指标
    """
    print("\n" + "="*60)
    print("E3: 恶意节点注入时间实验")
    print("="*60)

    bad_k     = 4
    clean_num = load_batches(NUM_FILES[0])
    clean_txt = load_batches(TXT_FILES[0])
    mal_num   = load_batches(NUM_FILES[bad_k])
    mal_txt   = load_batches(TXT_FILES[bad_k])

    n_batches        = min(len(clean_num), len(mal_num))
    injection_points = list(range(1, n_batches + 1))

    gt_num = build_gt_numeric(clean_num)
    gt_txt = build_gt_text(clean_txt)

    num_res: Dict[str, Dict[str, List]] = {m: {"rmse": [], "mae": [], "nrmse": []} for m in METHODS}
    txt_res: Dict[str, Dict[str, List]] = {m: {"accuracy": []} for m in METHODS}

    # 建立 batchIndex → batch 的快速查找
    cn_idx = {b["batchIndex"]: b for b in clean_num}
    mn_idx = {b["batchIndex"]: b for b in mal_num}
    ct_idx = {b["batchIndex"]: b for b in clean_txt}
    mt_idx = {b["batchIndex"]: b for b in mal_txt}

    for inj_start in injection_points:
        print(f"\n  注入时间点 = batch {inj_start}")
        # 组合数据集：[clean_1..inj-1] + [malicious_inj..N]
        num_batches = [
            (cn_idx if k < inj_start else mn_idx)[k]
            for k in range(1, n_batches + 1)
            if k in (cn_idx if k < inj_start else mn_idx)
        ]
        txt_batches = [
            (ct_idx if k < inj_start else mt_idx)[k]
            for k in range(1, n_batches + 1)
            if k in (ct_idx if k < inj_start else mt_idx)
        ]

        num_df = batches_to_numeric_df(num_batches)
        txt_df = batches_to_text_df(txt_batches)

        for m, v in eval_all_numeric(num_df, gt_num).items():
            for metric, val in v.items():
                num_res[m][metric].append(val)

        if not txt_df.empty:
            for m, v in eval_all_text(txt_df, gt_txt).items():
                txt_res[m]["accuracy"].append(v["accuracy"])
        else:
            for m in METHODS:
                txt_res[m]["accuracy"].append(float("nan"))

    _plot_e3(injection_points, num_res, txt_res)
    _save_csv("injection_start_batch", injection_points, num_res, txt_res, "e3_injection_time.csv")
    print("\n[E3] 完成")


# ─────────────────── E4: Agent 数量变化 ──────────────────────────────────────

def exp4_agent_count() -> None:
    """
    实验4: 随着 agent（节点）总数变化

    设计说明：
    - 固定 1 个恶意节点（agent-1），改变总 agent 数 N ∈ {3,4,5,6,7,8,9,10}
    - 实际恶意比例随 N 变化：N=3→33%，N=5→20%，N=10→10%
    - 通过过滤 {agent-1, ..., agent-N} 子集实现不同 N
    - 图表：折线图，x=agent 总数，y=指标
    - 目的：观察总节点数增加时各方法的拜占庭容错能力变化
    """
    print("\n" + "="*60)
    print("E4: Agent 数量变化实验")
    print("="*60)

    bad_k       = 1   # 固定 1 个恶意节点
    clean_num   = load_batches(NUM_FILES[0])
    clean_txt   = load_batches(TXT_FILES[0])
    mal_num     = load_batches(NUM_FILES[bad_k])
    mal_txt     = load_batches(TXT_FILES[bad_k])

    gt_num = build_gt_numeric(clean_num)
    gt_txt = build_gt_text(clean_txt)

    agent_counts = list(range(3, N_AGENTS + 1))   # 3,4,...,10
    all_agents   = [f"agent-{i}" for i in range(1, N_AGENTS + 1)]

    num_res: Dict[str, Dict[str, List]] = {m: {"rmse": [], "mae": [], "nrmse": []} for m in METHODS}
    txt_res: Dict[str, Dict[str, List]] = {m: {"accuracy": []} for m in METHODS}

    for n_ag in agent_counts:
        subset = all_agents[:n_ag]   # agent-1 为恶意节点，agent-2..n_ag 为诚实节点
        ratio  = 1.0 / n_ag * 100
        print(f"\n  n_agents={n_ag}, malicious_ratio={ratio:.1f}%")

        num_df = batches_to_numeric_df(mal_num, agent_subset=subset)
        txt_df = batches_to_text_df(mal_txt,   agent_subset=subset)

        for m, v in eval_all_numeric(num_df, gt_num).items():
            for metric, val in v.items():
                num_res[m][metric].append(val)

        if not txt_df.empty:
            for m, v in eval_all_text(txt_df, gt_txt).items():
                txt_res[m]["accuracy"].append(v["accuracy"])
        else:
            for m in METHODS:
                txt_res[m]["accuracy"].append(float("nan"))

    _plot_e4(agent_counts, num_res, txt_res)
    _save_csv("agent_count", agent_counts, num_res, txt_res, "e4_agent_count.csv")
    print("\n[E4] 完成")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、可视化
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "figure.dpi":      120,
})


def _plot_three_panels(
    xdata: List,
    num_res: Dict,
    txt_res: Dict,
    xlabel: str,
    exp_tag: str,
    x_invert: bool = False,
) -> None:
    """
    绘制三格子图：左=RMSE，中=MAE，右=Accuracy。
    同时生成数值和文本两张独立图（更清晰）。
    """
    # ── 数值双指标图 ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHODS:
        kw = dict(color=COLORS[m], marker=MARKERS[m], label=m,
                  linewidth=1.8, markersize=6)
        ax1.plot(xdata, num_res[m]["rmse"],  **kw)
        ax2.plot(xdata, num_res[m]["mae"],   **kw)
    for ax, ylabel, title_suffix in [
        (ax1, "RMSE",    "RMSE  (Numeric)"),
        (ax2, "MAE",     "MAE   (Numeric)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"[{exp_tag}] {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if x_invert:
            ax.invert_xaxis()
    plt.tight_layout()
    path = OUT_DIR / f"{exp_tag}_numeric.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → 保存 {path}")

    # ── 文本准确率图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        ax.plot(xdata, txt_res[m]["accuracy"],
                color=COLORS[m], marker=MARKERS[m], label=m,
                linewidth=1.8, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"[{exp_tag}] Text Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if x_invert:
        ax.invert_xaxis()
    plt.tight_layout()
    path = OUT_DIR / f"{exp_tag}_text.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → 保存 {path}")


def _plot_e1(ratios, num_res, txt_res):
    _plot_three_panels(ratios, num_res, txt_res,
                       "恶意节点比例 (%)", "E1_malicious_ratio")


def _plot_e2(batch_range, num_res, txt_res):
    _plot_three_panels(batch_range, num_res, txt_res,
                       "Batch 数量", "E2_batch_variation")


def _plot_e3(inj_pts, num_res, txt_res):
    # 注入越晚性能越好，所以 x 轴较小=更早注入=更难
    _plot_three_panels(inj_pts, num_res, txt_res,
                       "恶意注入开始 Batch", "E3_injection_time")


def _plot_e4(agent_counts, num_res, txt_res):
    _plot_three_panels(agent_counts, num_res, txt_res,
                       "Agent 总数", "E4_agent_count")


# ═══════════════════════════════════════════════════════════════════════════════
# 七、CSV 导出
# ═══════════════════════════════════════════════════════════════════════════════

def _save_csv(
    x_col: str,
    x_vals: List,
    num_res: Dict,
    txt_res: Dict,
    fname: str,
) -> None:
    rows = []
    for i, xv in enumerate(x_vals):
        for m in METHODS:
            rows.append({
                x_col:      xv,
                "method":   m,
                "rmse":     num_res[m]["rmse"][i] if i < len(num_res[m]["rmse"]) else float("nan"),
                "mae":      num_res[m]["mae"][i]  if i < len(num_res[m]["mae"])  else float("nan"),
                "nrmse":    num_res[m]["nrmse"][i] if i < len(num_res[m]["nrmse"]) else float("nan"),
                "accuracy": txt_res[m]["accuracy"][i] if i < len(txt_res[m]["accuracy"]) else float("nan"),
            })
    path = OUT_DIR / fname
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 保存 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 八、主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global OUT_DIR, TXT_FILES
    parser = argparse.ArgumentParser(
        description="四种真值发现方法对比基准实验"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        help='运行哪些实验，逗号分隔，如 "1,3" 或 "all"',
    )
    parser.add_argument(
        "--text-attack-profile",
        choices=["agent", "collude"],
        default="collude",
        help="文本实验使用的攻击画像。数值实验固定使用 num_demo_*bad_agent.json。",
    )
    args = parser.parse_args()

    TXT_FILES = get_text_files(args.text_attack_profile)
    OUT_DIR = BASE_OUT_DIR / args.text_attack_profile
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"text_attack_profile = {args.text_attack_profile}")

    selected = {str(i) for i in range(1, 5)} if args.exp == "all" \
               else set(args.exp.split(","))

    if "1" in selected:
        exp1_malicious_ratio()
    if "2" in selected:
        exp2_batch_variation()
    if "3" in selected:
        exp3_injection_time()
    if "4" in selected:
        exp4_agent_count()

    print(f"\n所有结果已保存至: {OUT_DIR}")


if __name__ == "__main__":
    main()
