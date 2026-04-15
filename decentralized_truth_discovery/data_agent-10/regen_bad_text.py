#!/usr/bin/env python3
"""
regen_bad_text.py

Support two malicious-text generation modes:
1. noise   : force large semantic deviation (legacy easy benchmark)
2. collude : assign malicious agents to explicit attack directions with
             fixed group sizes so each colluding group maps to one and only
             one semantic direction
"""

import argparse
import copy
import json
import os
import random
import re
from difflib import SequenceMatcher

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Mutation tables ───────────────────────────────────────────────────────────

DIRECTION_FLIPS = [
    ("上涨", "下跌"), ("下跌", "上涨"),
    ("上升", "下降"), ("下降", "上升"),
    ("增长", "萎缩"), ("萎缩", "增长"),
    ("增加", "减少"), ("减少", "增加"),
    ("超过", "低于"), ("低于", "超过"),
    ("高于", "低于"), ("低于", "高于"),
    ("看好", "看空"), ("看空", "看好"),
    ("最佳", "最差"), ("最差", "最佳"),
    ("看涨", "看跌"), ("看跌", "看涨"),
    ("领涨", "领跌"), ("领跌", "领涨"),
    ("好于", "差于"), ("差于", "好于"),
]

RATING_FLIPS = [
    ("买入", "卖出"), ("卖出", "买入"),
    ("增持", "减持"), ("减持", "增持"),
    ("outperform", "underperform"),
    ("underperform", "outperform"),
    ("Outperform", "Underperform"),
    ("Underperform", "Outperform"),
    ("最高", "最低"), ("最低", "最高"),
    ("上调", "下调"), ("下调", "上调"),
    ("启动覆盖", "终止覆盖"), ("终止覆盖", "启动覆盖"),
]

BULLISH_DIRECTION_BOOSTS = [
    ("下跌", "上涨"),
    ("下降", "上升"),
    ("萎缩", "增长"),
    ("减少", "增加"),
    ("低于", "高于"),
    ("看跌", "看涨"),
    ("领跌", "领涨"),
    ("差于", "好于"),
]

BULLISH_RATING_BOOSTS = [
    ("卖出", "买入"),
    ("减持", "增持"),
    ("underperform", "outperform"),
    ("Underperform", "Outperform"),
    ("最低", "最高"),
    ("下调", "上调"),
    ("终止覆盖", "启动覆盖"),
]

# False conclusions — used when numeric mutations are insufficient
# Must be long enough (≥40 chars) to push SBERT similarity below 0.85 even for longer texts
FALSE_CONCLUSIONS = [
    "然而根据最新财报，该公司营收同比下降18.3%，净利润录得重大亏损，分析师纷纷下调目标价至历史低位，建议投资者立即离场规避风险。",
    "但市场最新数据显示，该公司财务状况持续恶化，核心业务萎缩明显，机构投资者大规模减持，股价面临进一步下行压力。",
    "然而监管机构已对该公司展开正式调查，多项核心业务被迫暂停，公司发布盈利预警，预计当季亏损幅度将超过预期。",
    "但实际情况与官方表态截然相反，公司内部已出现重大债务违约风险，评级机构已将其信用评级列入负面观察名单。",
]


COLLUDE_FALSE_CLAUSE_FAMILIES = [
    [
        "，但全年指引已下修。",
        "，但公司已下修全年指引。",
        "，不过全年展望已被下调。",
        "，但最新口径已把全年预期下修。",
        "，不过公司已下修全年目标。",
        "，但全年经营指引已被下调。",
    ],
    [
        "，但需求指引已转弱。",
        "，但最新披露已显示需求转弱。",
        "，不过需求展望已转弱。",
        "，但新增披露显示需求势头回落。",
        "，不过最新更新显示需求明显放缓。",
        "，但最新信息反映需求已转弱。",
    ],
    [
        "，但利润率指引已下调。",
        "，但公司已提示利润率承压。",
        "，不过利润率目标已被下调。",
        "，但最新口径已提示利润率走弱。",
        "，不过公司已提示毛利率承压。",
        "，但新增说明已指向利润率承压。",
    ],
    [
        "，但新增指引已转为保守。",
        "，但最新指引已经转弱。",
        "，不过公司最新指引转向保守。",
        "，但管理层最新口径偏保守。",
        "，不过新增展望已更趋谨慎。",
        "，但最新经营展望转为谨慎。",
    ],
]


COLLUDE_BULLISH_FALSE_CLAUSE_FAMILIES = [
    [
        "，但全年指引已上修。",
        "，但公司已上修全年指引。",
        "，不过全年展望已被上调。",
        "，但最新口径已把全年预期上修。",
        "，不过公司已上修全年目标。",
        "，但全年经营指引已被上调。",
    ],
    [
        "，但需求指引已明显转强。",
        "，但最新披露已显示需求走强。",
        "，不过需求展望已显著改善。",
        "，但新增披露显示需求势头增强。",
        "，不过最新更新显示需求明显回升。",
        "，但最新信息反映需求已转强。",
    ],
    [
        "，但利润率指引已上调。",
        "，但公司已提示利润率改善。",
        "，不过利润率目标已被上调。",
        "，但最新口径已提示利润率走强。",
        "，不过公司已提示毛利率改善。",
        "，但新增说明已指向利润率改善。",
    ],
    [
        "，但新增指引已转为积极。",
        "，但最新指引已经走强。",
        "，不过公司最新指引转向积极。",
        "，但管理层最新口径更为乐观。",
        "，不过新增展望已明显改善。",
        "，但最新经营展望转向积极。",
    ],
]


DEFAULT_COLLUDE_GROUP_SPECS = {
    1: [1],
    2: [1, 1],
    3: [1, 1, 1],
    4: [2, 1, 1],
    5: [2, 2, 1],
    6: [2, 2, 2],
}


COLLUDE_DIRECTION_LIBRARY = [
    {
        "name": "guidance_cut",
        "bias": "bearish",
        "factor_even": 0.78,
        "factor_odd": 0.88,
        "clauses": COLLUDE_FALSE_CLAUSE_FAMILIES[0],
        "surface_profile_ids": [0, 5],
    },
    {
        "name": "guidance_raise",
        "bias": "bullish",
        "factor_even": 1.18,
        "factor_odd": 1.28,
        "clauses": COLLUDE_BULLISH_FALSE_CLAUSE_FAMILIES[0],
        "surface_profile_ids": [5, 0],
    },
    {
        "name": "demand_slump",
        "bias": "bearish",
        "factor_even": 0.74,
        "factor_odd": 0.84,
        "clauses": COLLUDE_FALSE_CLAUSE_FAMILIES[1],
        "surface_profile_ids": [1, 4],
    },
    {
        "name": "demand_rebound",
        "bias": "bullish",
        "factor_even": 1.12,
        "factor_odd": 1.22,
        "clauses": COLLUDE_BULLISH_FALSE_CLAUSE_FAMILIES[1],
        "surface_profile_ids": [4, 1],
    },
    {
        "name": "margin_pressure",
        "bias": "bearish",
        "factor_even": 0.69,
        "factor_odd": 0.79,
        "clauses": COLLUDE_FALSE_CLAUSE_FAMILIES[2],
        "surface_profile_ids": [2, 3],
    },
    {
        "name": "margin_expansion",
        "bias": "bullish",
        "factor_even": 1.09,
        "factor_odd": 1.17,
        "clauses": COLLUDE_BULLISH_FALSE_CLAUSE_FAMILIES[2],
        "surface_profile_ids": [3, 2],
    },
    {
        "name": "cautious_outlook",
        "bias": "bearish",
        "factor_even": 0.72,
        "factor_odd": 0.82,
        "clauses": COLLUDE_FALSE_CLAUSE_FAMILIES[3],
        "surface_profile_ids": [0, 1],
    },
    {
        "name": "constructive_outlook",
        "bias": "bullish",
        "factor_even": 1.14,
        "factor_odd": 1.24,
        "clauses": COLLUDE_BULLISH_FALSE_CLAUSE_FAMILIES[3],
        "surface_profile_ids": [1, 0],
    },
]


SURFACE_PROFILES = [
    [("将于", "将在"), ("当前", "目前"), ("预计", "预期")],
    [("当前", "现阶段"), ("第四季度", "第四财季"), ("预计", "预计")],
    [("当前", "目前"), ("可能", "或将"), ("第四季度", "第四季度")],
    [("将于", "将在"), ("当前", "当前"), ("预计", "预计"), ("有", "约有")],
    [("当前", "现时"), ("预计", "预期"), ("可能", "可能"), ("将于", "将于")],
    [("当前", "目前"), ("预计", "展望"), ("第四季度", "第四财季")],
]

HONEST_AGENT_PREFIXES = [
    "",
    "公开信息显示，",
    "从最新披露看，",
    "根据现有资料，",
    "综合公告内容来看，",
    "从市场信息看，",
    "就目前公开内容看，",
    "结合已披露资料，",
    "从现有消息看，",
    "按照公开信息，",
]

HONEST_SURFACE_PROFILES = [
    [
        ("将于", "将在"),
        ("当前", "目前"),
        ("最近", "近期"),
        ("宣布", "公布"),
        ("给予", "给出"),
        ("启动覆盖", "开始覆盖"),
        ("上涨至", "升至"),
        ("下跌至", "回落至"),
        ("已超过", "已高于"),
    ],
    [
        ("当前", "现阶段"),
        ("新期权合约", "新的期权合约"),
        ("开始交易", "开始挂牌交易"),
        ("最近", "近期"),
        ("增强了", "升级了"),
        ("上涨至", "涨至"),
        ("下跌至", "跌至"),
        ("未披露明确财务数字", "暂未披露明确财务数据"),
    ],
    [
        ("涉及的", "相关"),
        ("看跌和看涨期权", "看跌与看涨期权"),
        ("可能为卖方带来更高溢价", "对卖方而言可能带来更高溢价"),
        ("尽管", "虽然"),
        ("股价已超过", "股价已经高于"),
        ("前景看好", "前景仍被看好"),
        ("新增", "加入"),
    ],
    [
        ("将于", "计划于"),
        ("当前", "最新"),
        ("开始交易", "开启交易"),
        ("最近宣布", "最新宣布"),
        ("上涨至", "上行至"),
        ("下跌至", "下探至"),
        ("未披露明确财务数字", "没有披露明确财务数字"),
    ],
    [
        ("当前", "目前"),
        ("最近", "近来"),
        ("宣布", "披露"),
        ("股价已超过", "股价现已高于"),
        ("新增", "新增了"),
        ("公司在全球", "公司在全球范围内"),
        ("可能", "或"),
    ],
]


def _apply_flips_atomic(text: str, flips: list) -> tuple[str, int]:
    """
    Apply all flip rules in one regex pass to avoid double-flipping.
    (e.g. "看涨和看跌" → "看跌和看涨", not "看涨和看涨")
    """
    flip_dict = dict(flips)
    keys = sorted(flip_dict, key=len, reverse=True)
    if not keys:
        return text, 0
    pattern = re.compile("|".join(re.escape(k) for k in keys))
    count = [0]

    def replacer(m):
        count[0] += 1
        return flip_dict[m.group(0)]

    new_text = pattern.sub(replacer, text)
    return new_text, count[0]


def _apply_all_direction_flips(text: str) -> tuple[str, int]:
    return _apply_flips_atomic(text, DIRECTION_FLIPS)


def _apply_all_rating_flips(text: str) -> tuple[str, int]:
    return _apply_flips_atomic(text, RATING_FLIPS)


def _apply_all_bullish_direction_boosts(text: str) -> tuple[str, int]:
    return _apply_flips_atomic(text, BULLISH_DIRECTION_BOOSTS)


def _apply_all_bullish_rating_boosts(text: str) -> tuple[str, int]:
    return _apply_flips_atomic(text, BULLISH_RATING_BOOSTS)


def _mutate_percentage(m: re.Match) -> str:
    val = float(m.group(1))
    # Invert: 77% → 23%, 8.8% → 88%→ use (100-val) always for strong change
    new_val = round(100.0 - val, 2) if val <= 95 else round(val * 0.3, 2)
    if "." in m.group(1):
        decimals = len(m.group(1).split(".")[1])
        return f"{new_val:.{decimals}f}%"
    return f"{int(new_val)}%"


def _make_dollar_replacer(factor: float):
    def _replace(m: re.Match) -> str:
        val = float(m.group(1))
        new_val = round(val * factor, 2)
        return f"{new_val}{m.group(2)}"
    return _replace


def _make_price_replacer(factor: float):
    def _replace(m: re.Match) -> str:
        raw = m.group(0)
        # Skip years
        if "." not in raw:
            try:
                v = int(raw)
                if 1900 <= v <= 2100:
                    return raw
            except Exception:
                pass
        val = float(raw)
        if val < 1:
            return str(round(val * factor, 6))
        return str(round(val * factor, 2))
    return _replace


def _stable_object_seed(batch_index: int, object_name: str) -> int:
    return int(batch_index) * 1009 + sum(ord(ch) for ch in str(object_name))


def _replace_first_symbol_style(text: str, target_reference: str) -> str:
    if "(Symbol:" in target_reference:
        return re.sub(r"\(([A-Z][A-Z0-9.\-]*)\)", r"(Symbol: \1)", text, count=1)
    return text


def _apply_surface_profile(text: str, target_reference: str, profile_id: int) -> str:
    result = text
    for src, dst in SURFACE_PROFILES[profile_id % len(SURFACE_PROFILES)]:
        if src != dst:
            result = result.replace(src, dst)

    if "将在" in target_reference:
        result = result.replace("将于", "将在")
    if "第四财季" in target_reference:
        result = result.replace("第四季度", "第四财季")
    if "目前" in target_reference:
        result = result.replace("当前", "目前")
    if "现阶段" in target_reference:
        result = result.replace("当前", "现阶段")
    if "预期" in target_reference:
        result = result.replace("预计", "预期")

    return _replace_first_symbol_style(result, target_reference)


def _append_clause(text: str, clause: str) -> str:
    return text.rstrip("。！？") + clause


def _normalize_option_pair_terms(text: str) -> str:
    replacements = [
        ("看涨和看涨期权", "看涨和看跌期权"),
        ("看跌和看跌期权", "看跌和看涨期权"),
        ("看涨与看涨期权", "看涨与看跌期权"),
        ("看跌与看跌期权", "看跌与看涨期权"),
    ]
    result = text
    for src, dst in replacements:
        result = result.replace(src, dst)
    return result


def _apply_phrase_profile(text: str, replacements: list[tuple[str, str]]) -> str:
    result = text
    for src, dst in replacements:
        if src != dst:
            result = result.replace(src, dst)
    return result


def _agent_number(agent_name: str) -> int:
    try:
        return int(str(agent_name).split("-")[1])
    except (IndexError, ValueError, TypeError):
        return 1


def diversify_honest_text(
    text: str,
    agent_name: str,
) -> str:
    """
    Apply a deterministic, semantics-preserving style rewrite.

    The rewrite keeps all numeric literals unchanged and mainly changes
    discourse markers / phrasing so honest agents no longer collapse into
    large exact-match clusters.
    """
    agent_num = _agent_number(agent_name)
    profile = HONEST_SURFACE_PROFILES[(agent_num - 1) % len(HONEST_SURFACE_PROFILES)]
    prefix = HONEST_AGENT_PREFIXES[(agent_num - 1) % len(HONEST_AGENT_PREFIXES)]

    result = _apply_phrase_profile(text, profile)
    if prefix and not result.startswith(prefix):
        result = prefix + result
    return result


def summarize_text_diversity(data) -> dict:
    unique_counts = []
    largest_clusters = []
    for batch in data["job"]["batches"]:
        grouped = {}
        for item in batch["items"]:
            if isinstance(item["response"], str):
                grouped.setdefault(item["object"], []).append(item["response"])
        for texts in grouped.values():
            distinct = len(set(texts))
            unique_counts.append(distinct)
            counts = {}
            for text in texts:
                counts[text] = counts.get(text, 0) + 1
            largest_clusters.append(max(counts.values()))
    return {
        "avg_unique_texts": sum(unique_counts) / len(unique_counts) if unique_counts else 0.0,
        "min_unique_texts": min(unique_counts) if unique_counts else 0,
        "max_unique_texts": max(unique_counts) if unique_counts else 0,
        "avg_largest_cluster": sum(largest_clusters) / len(largest_clusters) if largest_clusters else 0.0,
    }


def diversify_clean_demo_data(demo_data):
    """
    Break large exact-text honest clusters while keeping semantics unchanged.

    Agent-1 remains the canonical clean text per object. Other agents with the
    same exact response are rewritten using deterministic agent-specific styles.
    """
    data = copy.deepcopy(demo_data)
    samples = []

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        grouped = {}
        for item in batch["items"]:
            if isinstance(item["response"], str):
                key = (item["object"], item["response"])
                grouped.setdefault(key, []).append(item)

        for (obj, original_text), items in grouped.items():
            if len(items) <= 1:
                continue

            sorted_items = sorted(items, key=lambda entry: (_agent_number(entry["agent"]), str(entry["agent"])))
            used_texts = {entry["response"] for entry in batch["items"] if isinstance(entry["response"], str) and entry["object"] == obj}

            for item in sorted_items[1:]:
                before = item["response"]
                after = diversify_honest_text(before, item["agent"])
                if after in used_texts:
                    alt_prefix = f"{HONEST_AGENT_PREFIXES[(_agent_number(item['agent']) + bi) % len(HONEST_AGENT_PREFIXES)]}"
                    if alt_prefix and not after.startswith(alt_prefix):
                        after = alt_prefix + after
                if after == before:
                    continue
                item["response"] = after
                used_texts.add(after)
                if len(samples) < 12:
                    samples.append((bi, obj, item["agent"], before, after))

    return data, samples


def _mutate_percentage_collude(m: re.Match) -> str:
    val = float(m.group(1))
    if val >= 50:
        new_val = round(max(5.0, val * 0.75), 2)
    else:
        new_val = round(min(95.0, val * 1.35 + 3.0), 2)
    if "." in m.group(1):
        decimals = len(m.group(1).split(".")[1])
        return f"{new_val:.{decimals}f}%"
    return f"{int(new_val)}%"


def _mutate_percentage_collude_bullish(m: re.Match) -> str:
    val = float(m.group(1))
    if val >= 50:
        new_val = round(min(99.0, val * 1.12 + 3.0), 2)
    else:
        new_val = round(min(95.0, val * 1.8 + 4.0), 2)
    if "." in m.group(1):
        decimals = len(m.group(1).split(".")[1])
        return f"{new_val:.{decimals}f}%"
    return f"{int(new_val)}%"


def default_collude_group_sizes(num_bad_agents: int) -> list[int]:
    if num_bad_agents <= 0:
        return []
    if num_bad_agents in DEFAULT_COLLUDE_GROUP_SPECS:
        return list(DEFAULT_COLLUDE_GROUP_SPECS[num_bad_agents])

    group_count = min(3, num_bad_agents)
    base, remainder = divmod(num_bad_agents, group_count)
    return [base + (1 if idx < remainder else 0) for idx in range(group_count)]


def collude_direction_names(direction_count: int) -> list[str]:
    if direction_count < 0:
        raise ValueError("direction_count must be non-negative")
    if direction_count > len(COLLUDE_DIRECTION_LIBRARY):
        raise ValueError(
            f"Requested {direction_count} directions, but only "
            f"{len(COLLUDE_DIRECTION_LIBRARY)} unique direction templates are defined."
        )
    return [COLLUDE_DIRECTION_LIBRARY[idx]["name"] for idx in range(direction_count)]


def _collude_direction_spec(direction_name: str) -> dict:
    for spec in COLLUDE_DIRECTION_LIBRARY:
        if spec["name"] == direction_name:
            return spec
    raise ValueError(f"Unknown collude direction: {direction_name}")


def _validate_collude_group_sizes(group_sizes: list[int], bad_agents: list[str]) -> list[int]:
    cleaned = [int(size) for size in group_sizes if int(size) > 0]
    if sum(cleaned) != len(bad_agents):
        raise ValueError(
            f"group_sizes {cleaned} do not sum to bad agent count {len(bad_agents)}"
        )
    return cleaned


def _collude_variant_id(
    agent_name: str,
    bad_agents: list[str],
    object_seed: int,
    group_sizes: list[int],
) -> int:
    if not group_sizes or not bad_agents:
        return 0
    try:
        agent_index = bad_agents.index(agent_name)
    except ValueError:
        agent_index = 0

    rotated_index = (agent_index + object_seed) % len(bad_agents)
    boundary = 0
    for variant_id, size in enumerate(group_sizes):
        boundary += size
        if rotated_index < boundary:
            return variant_id
    return len(group_sizes) - 1


def corrupt_text_collude(
    canonical_text: str,
    style_reference: str,
    object_seed: int,
    variant_id: int,
    attack_direction: str = "bearish",
    direction_name: str = None,
) -> str:
    """
    Generate a coordinated false text:
    - shared wrong slots come from object_seed
    - each colluding group is assigned to a unique semantic direction
    - surface form stays close enough to the clean text to remain plausible
    """
    if direction_name is None:
        direction_name = "guidance_raise" if attack_direction == "bullish" else "guidance_cut"

    direction_spec = _collude_direction_spec(direction_name)
    direction_bias = direction_spec["bias"]
    result = canonical_text
    mutation_count = 0

    if direction_bias == "bullish":
        num_factor = (
            direction_spec["factor_even"]
            if object_seed % 2 == 0
            else direction_spec["factor_odd"]
        )
        result, n = _apply_all_bullish_direction_boosts(result)
        mutation_count += n
        result, n = _apply_all_bullish_rating_boosts(result)
        mutation_count += n
        pct_replacer = _mutate_percentage_collude_bullish
    else:
        num_factor = (
            direction_spec["factor_even"]
            if object_seed % 2 == 0
            else direction_spec["factor_odd"]
        )
        result, n = _apply_all_direction_flips(result)
        mutation_count += n
        result, n = _apply_all_rating_flips(result)
        mutation_count += n
        pct_replacer = _mutate_percentage_collude

    new_result, n_subs = re.subn(r"(\d+\.?\d*)%", pct_replacer, result)
    if n_subs > 0:
        result = new_result
        mutation_count += n_subs

    new_result, n_subs = re.subn(
        r"(\d+\.?\d*)(美元|美金|\$)",
        _make_dollar_replacer(num_factor),
        result,
    )
    if n_subs > 0:
        result = new_result
        mutation_count += n_subs

    new_result = re.sub(
        r"(?<![\d,])\d+\.\d+(?!\d)",
        _make_price_replacer(num_factor),
        result,
    )
    if new_result != result:
        result = new_result
        mutation_count += 1

    clause = direction_spec["clauses"][object_seed % len(direction_spec["clauses"])]
    result = _append_clause(result, clause)

    surface_profile_ids = direction_spec["surface_profile_ids"]
    result = _apply_surface_profile(
        result,
        target_reference=style_reference,
        profile_id=surface_profile_ids[object_seed % len(surface_profile_ids)],
    )
    return _normalize_option_pair_terms(result)


def corrupt_text(text: str, seed: int) -> str:
    """
    Apply genuine fact mutations to *text* using *seed* for reproducibility.
    Returns a plausible-but-wrong version with no meta-labels.
    Target: SBERT cosine with clean text < 0.85.
    """
    rng = random.Random(seed)
    # Factor alternates between down (0.65) and up (1.45) based on seed
    num_factor = 0.65 if seed % 2 == 0 else 1.45

    result = text
    mutation_count = 0
    direction_count = 0
    rating_count = 0

    # 1. Flip ALL direction words
    result, n = _apply_all_direction_flips(result)
    direction_count = n
    mutation_count += n

    # 2. Flip ALL rating words
    result, n = _apply_all_rating_flips(result)
    rating_count = n
    mutation_count += n

    # 3. Mutate percentages (invert: 77%→23%, 8.8%→91.2%)
    new_result, n_subs = re.subn(r"(\d+\.?\d*)%", _mutate_percentage, result)
    if n_subs > 0:
        result = new_result
        mutation_count += n_subs

    # 4. Mutate explicit dollar amounts with large factor
    new_result, n_subs = re.subn(
        r"(\d+\.?\d*)(美元|美金|\$)",
        _make_dollar_replacer(num_factor),
        result,
    )
    if n_subs > 0:
        result = new_result
        mutation_count += n_subs

    # 5. Mutate ALL plain decimal numbers (prices, indices, etc.)
    # Note: use negative lookaround instead of \b because CJK chars are \w in Python 3
    # Also skip digits after comma (part of comma-formatted large numbers like 15,858.26)
    new_result = re.sub(r"(?<![\d,])\d+\.\d+(?!\d)", _make_price_replacer(num_factor), result)
    if new_result != result:
        result = new_result
        mutation_count += 1

    # 6. Always append a false conclusion.
    # Rationale: pure word-flips and even number changes alone keep SBERT similarity
    # above 0.85 for most texts (tested: avg sim ≈ 0.92 without conclusion).
    # Adding a ~60-char false conclusion reliably pushes similarity below 0.85.
    conclusion = FALSE_CONCLUSIONS[seed % len(FALSE_CONCLUSIONS)]
    result = result.rstrip("。！？") + conclusion

    return result


# ─── Main logic ────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def build_clean_text_map(demo_data) -> dict[tuple[int, str], str]:
    """
    Build a mapping of (batchIndex, object_name) → agent-1 clean text
    from demo.json.  Only text responses (str) are included.
    """
    mapping = {}
    for batch in demo_data["job"]["batches"]:
        bi = batch["batchIndex"]
        for item in batch["items"]:
            if item["agent"] == "agent-1" and isinstance(item["response"], str):
                mapping[(bi, item["object"])] = item["response"]
    return mapping


def build_clean_agent_text_map(demo_data) -> dict[tuple[int, str, str], str]:
    mapping = {}
    for batch in demo_data["job"]["batches"]:
        bi = batch["batchIndex"]
        for item in batch["items"]:
            if isinstance(item["response"], str):
                mapping[(bi, item["object"], item["agent"])] = item["response"]
    return mapping


def regenerate_file(
    bad_file_path: str,
    demo_data,
    clean_map: dict,
    bad_agents: list[str],
):
    """
    Build a legacy noise dataset from the diversified clean demo data.
    Honest agents keep their diversified clean responses; malicious agents are
    overwritten with corrupted responses derived from the canonical clean text.

    bad_agents: list like ["agent-1"] or ["agent-1","agent-2"] etc.
    """
    data = copy.deepcopy(demo_data)
    samples = []  # (batch_index, agent, before, after)

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        for item in batch["items"]:
            agent = item["agent"]
            if agent not in bad_agents:
                continue
            if not isinstance(item["response"], str):
                continue

            obj = item["object"]
            clean_text = clean_map.get((bi, obj))
            if clean_text is None:
                # fallback: use the item's own text as base
                clean_text = item["response"]

            # Derive a unique seed per (batch, agent)
            agent_num = _agent_number(agent)
            seed = bi * 10 + agent_num

            before = item["response"]
            corrupted = corrupt_text(clean_text, seed)
            item["response"] = corrupted

            if bi == 1 and len(samples) < len(bad_agents):
                samples.append((bi, agent, before, corrupted))

    save_json(bad_file_path, data)
    return samples


def regenerate_collude_file(
    out_file_path: str,
    demo_data,
    canonical_clean_map: dict,
    clean_agent_map: dict,
    bad_agents: list[str],
    group_sizes: list[int] = None,
    direction_names: list[str] = None,
):
    """
    Build a collusive malicious dataset from clean demo data.
    Each malicious group has a fixed size and maps to exactly one attack
    direction so theoretical analyses over m1, m2, ... can be reproduced.
    """
    if group_sizes is None:
        group_sizes = default_collude_group_sizes(len(bad_agents))
    group_sizes = _validate_collude_group_sizes(group_sizes, bad_agents)
    if direction_names is None:
        direction_names = collude_direction_names(len(group_sizes))
    if len(direction_names) != len(group_sizes):
        raise ValueError(
            f"direction_names length {len(direction_names)} does not match "
            f"group_sizes length {len(group_sizes)}"
        )

    data = copy.deepcopy(demo_data)
    samples = []

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        object_groups = {}
        for item in batch["items"]:
            agent = item["agent"]
            if agent in bad_agents and isinstance(item["response"], str):
                object_groups.setdefault(item["object"], []).append(item)

        for obj, items in object_groups.items():
            canonical_text = canonical_clean_map.get((bi, obj), items[0]["response"])
            object_seed = _stable_object_seed(bi, obj)

            exemplar_style_refs = {}
            for variant_id in range(len(group_sizes)):
                exemplar = next(
                    (
                        item for item in items
                        if _collude_variant_id(
                            item["agent"],
                            bad_agents,
                            object_seed,
                            group_sizes,
                        ) == variant_id
                    ),
                    items[0],
                )
                exemplar_style_refs[variant_id] = clean_agent_map.get(
                    (bi, obj, exemplar["agent"]),
                    canonical_text,
                )

            for item in items:
                agent = item["agent"]
                variant_id = _collude_variant_id(
                    agent,
                    bad_agents,
                    object_seed,
                    group_sizes,
                )
                style_reference = exemplar_style_refs[variant_id]

                before = item["response"]
                after = corrupt_text_collude(
                    canonical_text=canonical_text,
                    style_reference=style_reference,
                    object_seed=object_seed,
                    variant_id=variant_id,
                    direction_name=direction_names[variant_id],
                )
                item["response"] = after

                if bi == 1 and len(samples) < len(bad_agents):
                    samples.append((bi, agent, before, after))

    save_json(out_file_path, data)
    return data, samples


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def summarize_collude_quality(
    data,
    clean_agent_map: dict,
    bad_agents: list[str],
) -> dict:
    clean_bad_ratios = []
    bad_bad_ratios = []
    unique_bad_counts = []

    for batch in data["job"]["batches"]:
        grouped = {}
        bi = batch["batchIndex"]
        for item in batch["items"]:
            if item["agent"] not in bad_agents or not isinstance(item["response"], str):
                continue
            grouped.setdefault(item["object"], []).append((item["agent"], item["response"]))

        for obj, pairs in grouped.items():
            texts = [text for _, text in pairs]
            unique_bad_counts.append(len(set(texts)))
            for agent, bad_text in pairs:
                clean_text = clean_agent_map.get((bi, obj, agent))
                if clean_text is not None:
                    clean_bad_ratios.append(_ratio(clean_text, bad_text))
            for idx in range(len(texts)):
                for jdx in range(idx + 1, len(texts)):
                    bad_bad_ratios.append(_ratio(texts[idx], texts[jdx]))

    return {
        "avg_clean_bad_ratio": sum(clean_bad_ratios) / len(clean_bad_ratios) if clean_bad_ratios else 0.0,
        "avg_bad_bad_ratio": sum(bad_bad_ratios) / len(bad_bad_ratios) if bad_bad_ratios else 0.0,
        "avg_unique_bad_texts": sum(unique_bad_counts) / len(unique_bad_counts) if unique_bad_counts else 0.0,
        "min_unique_bad_texts": min(unique_bad_counts) if unique_bad_counts else 0,
    }


def _parse_bad_counts(spec: str) -> list[int]:
    counts = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        counts.append(int(part))
    return sorted(set(counts))


def main():
    parser = argparse.ArgumentParser(description="Regenerate malicious text datasets.")
    parser.add_argument(
        "--mode",
        choices=["noise", "collude", "both"],
        default="both",
        help="Generation mode. 'noise' keeps legacy large-deviation attacks; "
             "'collude' creates high-similarity coordinated false texts.",
    )
    parser.add_argument(
        "--bad-counts",
        default="1,2,3,4,5,6",
        help="Comma-separated malicious agent counts for collude mode.",
    )
    args = parser.parse_args()

    demo_path = os.path.join(DATA_DIR, "demo.json")
    print(f"Loading clean texts from {demo_path} …")
    demo_data = load_json(demo_path)
    diverse_demo_data, clean_samples = diversify_clean_demo_data(demo_data)
    save_json(demo_path, diverse_demo_data)
    canonical_clean_map = build_clean_text_map(diverse_demo_data)
    clean_agent_map = build_clean_agent_text_map(diverse_demo_data)
    diversity_summary = summarize_text_diversity(diverse_demo_data)
    print(f"  → {len(canonical_clean_map)} canonical (batchIndex, object) entries loaded.")
    print(f"  → {len(clean_agent_map)} agent-style (batchIndex, object, agent) entries loaded.")
    print(
        "  → diversified clean demo:"
        f" avg(unique/object)={diversity_summary['avg_unique_texts']:.2f}"
        f"  avg(largest exact cluster)={diversity_summary['avg_largest_cluster']:.2f}\n"
    )

    for bi, obj, agent, before, after in clean_samples[:8]:
        print(f"  [clean batch {bi}] {agent} / {obj}")
        print(f"  BEFORE: {before[:160]}")
        print(f"  AFTER : {after[:160]}")
        print()

    if args.mode in {"noise", "both"}:
        for bad_count in _parse_bad_counts(args.bad_counts):
            bad_agents = [f"agent-{idx}" for idx in range(1, bad_count + 1)]
            filename = f"demo_{bad_count}bad_agent.json"
            path = os.path.join(DATA_DIR, filename)
            print(f"{'='*70}")
            print(f"Processing {filename}  (legacy noise, bad agents: {bad_agents})")
            print(f"{'='*70}")

            samples = regenerate_file(path, diverse_demo_data, canonical_clean_map, bad_agents)

            for bi, agent, before, after in samples:
                print(f"\n  [batch {bi}] {agent}")
                print(f"  BEFORE: {before[:160]}")
                print(f"  AFTER : {after[:160]}")

            print(f"\n  Saved → {path}\n")

    if args.mode in {"collude", "both"}:
        for bad_count in _parse_bad_counts(args.bad_counts):
            bad_agents = [f"agent-{idx}" for idx in range(1, bad_count + 1)]
            group_sizes = default_collude_group_sizes(len(bad_agents))
            direction_names = collude_direction_names(len(group_sizes))
            filename = f"demo_{bad_count}bad_collude.json"
            path = os.path.join(DATA_DIR, filename)

            print(f"{'='*70}")
            print(
                f"Processing {filename}  (collude, bad agents: {bad_agents}, "
                f"group_sizes={group_sizes}, directions={direction_names})"
            )
            print(f"{'='*70}")

            data, samples = regenerate_collude_file(
                out_file_path=path,
                demo_data=diverse_demo_data,
                canonical_clean_map=canonical_clean_map,
                clean_agent_map=clean_agent_map,
                bad_agents=bad_agents,
                group_sizes=group_sizes,
                direction_names=direction_names,
            )
            summary = summarize_collude_quality(data, clean_agent_map, bad_agents)

            for bi, agent, before, after in samples:
                print(f"\n  [batch {bi}] {agent}")
                print(f"  BEFORE: {before[:160]}")
                print(f"  AFTER : {after[:160]}")

            print(
                "\n  Quality summary:"
                f" avg(clean,bad)={summary['avg_clean_bad_ratio']:.3f}"
                f"  avg(bad,bad)={summary['avg_bad_bad_ratio']:.3f}"
                f"  avg(unique_bad/group)={summary['avg_unique_bad_texts']:.2f}"
                f"  min(unique_bad/group)={summary['min_unique_bad_texts']}"
            )
            print(f"\n  Saved → {path}\n")


if __name__ == "__main__":
    main()
