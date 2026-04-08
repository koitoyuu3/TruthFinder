"""
实验4: 随着 Agent 数量变化
================================================
自变量: 总 agent 数 N ∈ {3, 4, 5, 6, 7, 8, 9, 10}
场景设定:
  固定 1 个恶意节点（agent-1），总数 N 变化
  → 实际恶意比例随 N 从 33% 降至 10%
  → 节点越多，诚实节点比例越高，方法 RMSE 应逐渐降低

数据过滤: 只保留 agent-1..agent-N 的数据

评估框架（对齐论文 Section VI）:
  preds = 方法在 demo_1bad_agent.json（agent subset, 1 恶意）上的输出
  gt    = demo.json 中的真实值（价格与 agent subset 无关）
  数值场景 → RMSE(preds_last10%, gt_last10%), MAE(...)
           可选 relative metric：先按 |gt| 归一化误差，再聚合 RMSE/MAE
  混合场景 → DataAccuracy(X_clean, Y_noisy)

运行: python3 exp4_agent_count.py [--metric-scale raw|relative]
"""
import argparse
from pathlib import Path

from exp_utils import (
    N_AGENTS,
    NUM_METHODS, TEXT_METHODS,
    get_numeric_demo_files,
    get_text_demo_files,
    load_batches,
    to_numeric_df, to_text_df,
    build_gt_numeric,
    eval_numeric, eval_text,
    plot_numeric, plot_text, save_csv,
)

BASE_OUT_DIR = Path(__file__).parent / "outputs" / "exp4_agent_count"

BAD_COUNT   = 1          # 固定 1 个恶意节点
TITLE       = "E4: Agent 数量变化"
X_LABEL     = "Agent 总数"
MIN_AGENTS  = 3          # 最少 3 个 agent（1 恶 + 2 诚实，满足拜占庭容错最低要求）


def main() -> None:
    parser = argparse.ArgumentParser(description="实验4: 随着 Agent 数量变化")
    parser.add_argument(
        "--attack-profile",
        default="collude",
        help="已弃用，仅为兼容旧命令保留。实验固定使用：数值=num_demo_*bad_agent.json，文本=demo_*bad_collude.json。",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=BAD_COUNT,
        help="固定恶意节点数量。",
    )
    parser.add_argument(
        "--metric-scale",
        choices=["raw", "relative"],
        default="relative",
        help="raw: 原始绝对误差; relative: 按每个币种的 |ground truth| 归一化后再算 RMSE/MAE（默认）",
    )
    args = parser.parse_args()
    normalize_metric = args.metric_scale == "relative"

    num_files = get_numeric_demo_files()
    text_files = get_text_demo_files()
    if args.bad_count not in num_files:
        available = ", ".join(str(k) for k in sorted(num_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for numeric files. "
            f"Available bad_count values: {available}"
        )
    if args.bad_count not in text_files:
        available = ", ".join(str(k) for k in sorted(text_files))
        raise ValueError(
            f"bad_count={args.bad_count} is unavailable for text files. "
            f"Available bad_count values: {available}"
        )
    out_dir = BASE_OUT_DIR / "demo_agent_text_collude" / f"bad_{args.bad_count}" / f"metric_{args.metric_scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "datasets = numeric:num_demo_*bad_agent.json, "
        f"text:demo_*bad_collude.json, bad_count = {args.bad_count}, metric_scale = {args.metric_scale}"
    )

    clean_num_batches = load_batches(num_files[0])
    mal_num_batches   = load_batches(num_files[args.bad_count])
    clean_txt_batches = load_batches(text_files[0])
    mal_txt_batches   = load_batches(text_files[args.bad_count])
    gt_num        = build_gt_numeric(clean_num_batches)   # 价格真值与 agent subset 无关

    all_agents  = [f"agent-{i}" for i in range(1, N_AGENTS + 1)]
    agent_range = list(range(MIN_AGENTS, N_AGENTS + 1))   # [3,4,...,10]

    num_results = {m: {"rmse": [], "mae": []} for m in NUM_METHODS}
    txt_results = {m: {"data_accuracy": []}        for m in TEXT_METHODS}

    for n_ag in agent_range:
        subset = all_agents[:n_ag]                        # agent-1..agent-N
        ratio  = 1.0 / n_ag * 100
        print(f"\n  n_agents={n_ag}  恶意比例≈{ratio:.1f}%  "
              f"subset={subset}")

        noisy_num_df = to_numeric_df(mal_num_batches,   agent_subset=subset)
        clean_txt_df = to_text_df(clean_txt_batches,    agent_subset=subset)
        noisy_txt_df = to_text_df(mal_txt_batches,      agent_subset=subset)

        # ── 数值场景 ──────────────────────────────────────────────
        res_num = eval_numeric(noisy_num_df, gt_num, normalize=normalize_metric)
        for m in NUM_METHODS:
            num_results[m]["rmse"].append(res_num[m]["rmse"])
            num_results[m]["mae"].append(res_num[m]["mae"])

        # ── 混合场景（文本） ───────────────────────────────────────
        if not clean_txt_df.empty and not noisy_txt_df.empty:
            res_txt = eval_text(clean_txt_df, noisy_txt_df)
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(res_txt[m]["data_accuracy"])
        else:
            for m in TEXT_METHODS:
                txt_results[m]["data_accuracy"].append(float("nan"))

    plot_numeric(
        agent_range,
        num_results,
        X_LABEL,
        TITLE,
        out_dir,
        metric_scope_label=(
            "last 10% batches, relative error normalized by |ground truth|"
            if normalize_metric else
            "last 10% batches"
        ),
        metric_name_prefix="Normalized " if normalize_metric else "",
    )
    plot_text(agent_range, txt_results, X_LABEL, TITLE, out_dir)
    save_csv("agent_count", agent_range, num_results, txt_results, out_dir, metric_scale=args.metric_scale)

    print(f"\n[E4] 完成，结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
