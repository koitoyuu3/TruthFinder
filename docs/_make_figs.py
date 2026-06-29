"""Regenerate all figures referenced by exp1_3_4_7_detailed_report.md.

Style rules (per user request):
- No titles.
- All text in English.
- exp1_numeric: bar charts WITHOUT value labels on bars.
- exp3 phase boundary labels do NOT overlap data lines (placed at top inside).
- exp4_numeric y-axis labels are simply 'RMSE' / 'MAE'.

Each figure is saved as <basename>_clean.png next to its source CSV so the
originals are not overwritten.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/huabowen/TruthFinder/decentralized_truth_discovery/examples/outputs")
DOC_FIGS = Path("/Users/huabowen/TruthFinder/docs/figs")
DOC_FIGS.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "BasicTruth":    "#FF8C42",  # orange
    "DecentTruth":   "#1F77B4",  # blue
    "SenFeedTruth":  "#9467BD",  # purple
    "SenteTruth":    "#2CA02C",  # green
    "SemanticTruth": "#7F3FBF",  # violet
}
METHOD_MARKERS = {
    "BasicTruth":    "o",
    "DecentTruth":   "s",
    "SenFeedTruth":  "D",
    "SenteTruth":    "^",
    "SemanticTruth": "v",
}
METHOD_ORDER = ["BasicTruth", "DecentTruth", "SenFeedTruth", "SenteTruth", "SemanticTruth"]
TEXT_METHOD_ORDER = ["BasicTruth", "SenteTruth", "SemanticTruth"]

DISPLAY_NAMES = {
    "SemanticTruth": "LASO(Ours)",
}


def _disp(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    out = DOC_FIGS / name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(DOC_FIGS.parent.parent)}")
    print(f"  -> {pdf_out.relative_to(DOC_FIGS.parent.parent)}")


# ---------------------------------------------------------------------------
# EXP1 NUMERIC — grouped bar chart per bad_count, mean RMSE per method.
# ---------------------------------------------------------------------------

def fig_exp1_numeric(csv_path: Path, out_name: str, title_unused: str = ""):
    rows = _read(csv_path)
    bad_col = "bad_count" if "bad_count" in rows[0] else list(rows[0].keys())[0]
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    bads = sorted({int(r[bad_col]) for r in rows})

    means = {m: [] for m in methods}
    for b in bads:
        for m in methods:
            vs = [float(r["rmse"]) for r in rows if int(r[bad_col]) == b and r["method"] == m]
            means[m].append(np.mean(vs) if vs else 0.0)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.0 * len(bads) + 2), 4.6))
    x = np.arange(len(bads))
    width = 0.78 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            means[m], width=width,
            label=_disp(m), color=METHOD_COLORS[m], edgecolor="white", linewidth=0.6,
        )
    ax.set_xticks(x)
    if max(bads) <= 4:
        ax.set_xticklabels([f"{b}" for b in bads])
        ax.set_xlabel("Number of malicious agents (out of 10)")
    else:
        pct = [f"{b} ({int(round(100*b/50))}%)" for b in bads]
        ax.set_xticklabels(pct)
        ax.set_xlabel("Malicious agents (count and ratio)")
    ax.set_ylabel("Relative RMSE")
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, 1.10), fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP1 TEXT — line chart of DataAcc vs malicious_ratio_pct.
# ---------------------------------------------------------------------------

def fig_exp1_text(csv_path: Path, out_name: str):
    rows = _read(csv_path)
    # Restrict to ratios <= 50% per user request.
    rows = [r for r in rows if float(r["malicious_ratio_pct"]) <= 50.0]
    ratios = [float(r["malicious_ratio_pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in TEXT_METHOD_ORDER:
        if m not in rows[0]:
            continue
        ys = [float(r[m]) for r in rows]
        ax.plot(ratios, ys, label=_disp(m), color=METHOD_COLORS[m],
                marker=METHOD_MARKERS[m], markersize=6, linewidth=1.6)
    ax.set_xlabel("Malicious agent ratio (%)")
    ax.set_ylabel("Data Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(ratios)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP3 NUMERIC — per-batch RMSE timeseries with three phase markers.
# ---------------------------------------------------------------------------

def fig_exp3_numeric(csv_path: Path, out_name: str):
    rows = _read(csv_path)
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    series = {m: ([], []) for m in methods}
    for r in rows:
        m = r["method"]
        if m not in series:
            continue
        series[m][0].append(int(r["batch_idx"]))
        series[m][1].append(float(r["relative_rmse"]))

    p1 = int(rows[0]["phase1_start_batch"]); p1b = int(rows[0]["phase1_bad_count"])
    p2 = int(rows[0]["phase2_start_batch"]); p2b = int(rows[0]["phase2_bad_count"])
    p3 = int(rows[0]["phase3_start_batch"]); p3b = int(rows[0]["phase3_bad_count"])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    xmax = 0
    for m in methods:
        xs, ys = series[m]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        xmax = max(xmax, int(xs.max()))
        ax.plot(xs, ys, label=_disp(m), color=METHOD_COLORS[m], linewidth=1.2, alpha=0.9)

    for x in (p1, p2, p3):
        ax.axvline(x, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    # Phase labels at the very top of the axes (above all data lines).
    ymax = ax.get_ylim()[1]
    labels = [
        ((1 + p1) / 2,  "no attack"),
        ((p1 + p2) / 2, f"{p1b} bad ({int(round(100*p1b/50))}%)"),
        ((p2 + p3) / 2, f"{p2b} bad ({int(round(100*p2b/50))}%)"),
        ((p3 + xmax) / 2, f"{p3b} bad ({int(round(100*p3b/50))}%)"),
    ]
    ax.set_ylim(top=ymax * 1.22)
    for x, txt in labels:
        ax.text(x, ax.get_ylim()[1] * 0.96, txt, ha="center", va="top",
                fontsize=9, color="dimgrey",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="lightgrey", lw=0.5))
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Relative RMSE")
    ax.set_xlim(1, xmax)
    ax.legend(ncol=len(methods), loc="upper left",
              bbox_to_anchor=(0, -0.13), fontsize=9)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP3 TEXT — per-batch DataAcc + mean_hybrid_score with phase markers.
# ---------------------------------------------------------------------------

def fig_exp3_text(csv_path: Path, out_name: str):
    rows = _read(csv_path)
    methods = [m for m in TEXT_METHOD_ORDER if any(r["method"] == m for r in rows)]
    p1 = int(rows[0]["phase1_start_batch"]); p1b = int(rows[0]["phase1_bad_count"])
    p2 = int(rows[0]["phase2_start_batch"]); p2b = int(rows[0]["phase2_bad_count"])
    p3 = int(rows[0]["phase3_start_batch"]); p3b = int(rows[0]["phase3_bad_count"])

    series = {m: ([], []) for m in methods}
    for r in rows:
        m = r["method"]
        if m not in series:
            continue
        series[m][0].append(int(r["batch_idx"]))
        series[m][1].append(float(r["mean_hybrid_score"]))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    xmax = 0
    for m in methods:
        xs, ys = series[m]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        xmax = max(xmax, int(xs.max()))
        # Plot raw data — no smoothing — so trends like the SemanticTruth
        # value at the final batches reflect the actual CSV exactly.
        ax.plot(xs, ys, label=_disp(m), color=METHOD_COLORS[m], linewidth=1.0, alpha=0.9)

    for x in (p1, p2, p3):
        ax.axvline(x, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 1.22)  # leave room for labels above data
    labels = [
        ((1 + p1) / 2,  "no attack"),
        ((p1 + p2) / 2, f"{p1b} bad ({int(round(100*p1b/50))}%)"),
        ((p2 + p3) / 2, f"{p2b} bad ({int(round(100*p2b/50))}%)"),
        ((p3 + xmax) / 2, f"{p3b} bad ({int(round(100*p3b/50))}%)"),
    ]
    for x, txt in labels:
        ax.text(x, 1.17, txt, ha="center", va="top", fontsize=9, color="dimgrey",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="lightgrey", lw=0.5))
    ax.axhline(0.85, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(xmax * 0.012, 0.86, "threshold = 0.85", color="black", alpha=0.55, fontsize=9)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Mean Hybrid Score")
    ax.set_xlim(1, xmax)
    ax.legend(ncol=len(methods), loc="upper left",
              bbox_to_anchor=(0, -0.13), fontsize=9)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP4 NUMERIC — line chart of mean RMSE & MAE vs agent_count.
# Y-axis labels are exactly 'RMSE' and 'MAE' (per user request).
# ---------------------------------------------------------------------------

def fig_exp4_numeric(csv_path: Path, out_name: str):
    rows = _read(csv_path)
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    Ns = sorted({int(r["agent_count"]) for r in rows})

    rmse = {m: [np.mean([float(r["rmse"]) for r in rows
                          if int(r["agent_count"]) == n and r["method"] == m]) for n in Ns]
            for m in methods}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in methods:
        ax.plot(Ns, rmse[m], label=_disp(m), color=METHOD_COLORS[m],
                marker=METHOD_MARKERS[m], markersize=6, linewidth=1.6)
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(Ns)
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, 1.10), fontsize=9)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP4 TEXT — line chart of DataAcc vs agent_count.
# ---------------------------------------------------------------------------

def fig_exp4_text(csv_path: Path, out_name: str):
    rows = _read(csv_path)
    methods = [m for m in TEXT_METHOD_ORDER if any(r["method"] == m for r in rows)]
    Ns = sorted({int(r["agent_count"]) for r in rows})
    series = {m: [np.mean([float(r["data_accuracy"]) for r in rows
                            if int(r["agent_count"]) == n and r["method"] == m]) for n in Ns]
              for m in methods}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in methods:
        ax.plot(Ns, series[m], label=_disp(m), color=METHOD_COLORS[m],
                marker=METHOD_MARKERS[m], markersize=6, linewidth=1.6)
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("Data Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(Ns)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# EXP7 — 3-subplot per-method alpha sweep + summary line plot.
# ---------------------------------------------------------------------------

def fig_exp7(csv_path: Path):
    rows = _read(csv_path)
    methods = TEXT_METHOD_ORDER
    bads = sorted({int(r["bad_count"]) for r in rows})
    alphas = sorted({float(r["alpha"]) for r in rows})

    nested = {m: {b: {a: None for a in alphas} for b in bads} for m in methods}
    for r in rows:
        nested[r["method"]][int(r["bad_count"])][float(r["alpha"])] = float(r["data_accuracy"])

    # by-method — leave headroom at the top of each subplot so the method
    # label sits above all curves and never overlaps the data lines.
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, m in zip(axes, methods):
        for b in bads:
            ys = [nested[m][b][a] for a in alphas]
            color = cmap(0.15 + 0.8 * (b - min(bads)) / max(1, max(bads) - min(bads)))
            ax.plot(alphas, ys, color=color, marker="o", markersize=4,
                    linewidth=1.4, label=f"{int(round(100*b/50))}% bad")
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$\alpha$ (semantic weight)")
        ax.set_ylim(0, 1.18)  # headroom above curves for method label
        ax.text(0.5, 1.10, _disp(m), transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#333333")
    axes[0].set_ylabel("Data Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(bads), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "exp7_alpha_by_method.png")

    # summary
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in methods:
        means = [np.mean([nested[m][b][a] for b in bads]) for a in alphas]
        ax.plot(alphas, means, label=_disp(m), color=METHOD_COLORS[m],
                marker=METHOD_MARKERS[m], markersize=6, linewidth=1.6)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label=r"default $\alpha=0.5$")
    ax.set_xlabel(r"$\alpha$ (semantic weight)")
    ax.set_ylabel("Mean Data Accuracy")
    ax.set_ylim(0.3, 0.95)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "exp7_alpha_summary.png")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    print("[exp1_numeric]")
    fig_exp1_numeric(ROOT / "exp1_numeric/numeric_per_batch_rmse.csv",
                     "exp1_numeric_default.png")
    fig_exp1_numeric(ROOT / "exp1_numeric/data_agent-50_news-300_collude_relative/numeric_per_batch_rmse.csv",
                     "exp1_numeric_50agent.png")

    print("[exp1_text]")
    for sub in ["1.1-agent-50-news-300-80%-0.85-collude",
                "1.2-agent-50-news-300-50%-0.9-numonly",
                "1.3-agent-50-weather-300-80%-0.85-collude",
                "1.4-agent-50-weather-300-50%-0.9-numonly"]:
        tag = sub.split("-")[0]  # "1.1"
        fig_exp1_text(ROOT / f"exp1_text/{sub}/text_data_accuracy.csv",
                      f"exp1_text_{tag.replace('.','_')}.png")

    print("[exp3_numeric]")
    fig_exp3_numeric(ROOT / "exp3_numeric/data_agent-50_news-300/5bad_at_50__10bad_at_100__15bad_at_200/results.csv",
                     "exp3_numeric.png")

    print("[exp3_text]")
    fig_exp3_text(ROOT / "exp3_text/data_agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200/results.csv",
                  "exp3_text_news.png")
    fig_exp3_text(ROOT / "exp3_text/data_agent-50_weather-300/results.csv",
                  "exp3_text_weather.png")

    print("[exp4_numeric]")
    fig_exp4_numeric(ROOT / "exp4_numeric/data_agent-50_news-300/bad_5/metric_relative/results.csv",
                     "exp4_numeric.png")

    print("[exp4_text]")
    pairs = [
        ("exp4_text/data_agent-50_news-300/bad_5/results.csv",                 "exp4_text_news_t085.png"),
        ("exp4_text/data_agent-50_news-300/bad_5/threshold_0.90/results.csv",  "exp4_text_news_t090.png"),
        ("exp4_text/data_agent-50_weather-300/bad_5/results.csv",              "exp4_text_weather_t085.png"),
        ("exp4_text/data_agent-50_weather-300/bad_5/threshold_0.90/results.csv","exp4_text_weather_t090.png"),
    ]
    for src, out in pairs:
        fig_exp4_text(ROOT / src, out)

    print("[exp7]")
    fig_exp7(ROOT / "exp7_alpha_sensitivity/results.csv")


if __name__ == "__main__":
    main()
