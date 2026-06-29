"""Regenerate every figure listed by the user with an academic-paper style
(IEEE/ACM look — serif font, fine dashed gridlines, small markers, no title).

Outputs:
    docs/figs_v2/<basename>.png
    docs/figs_v2/<basename>.pdf
    docs/figs_v2/figures_compiled.pdf  (all PNGs bundled into one PDF)

All in-figure text is in English, "SemanticTruth" is renamed to "LASO(Ours)",
and titles are removed.  Axis labels are kept short.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path("/Users/huabowen/TruthFinder/decentralized_truth_discovery/examples/outputs")
OUT = Path("/Users/huabowen/TruthFinder/docs/figs_v2")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — paper-style: serif font, dashed gridlines, all spines visible.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#222222",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.color": "#bdbdbd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# Academic-paper palette modeled on the reference screenshots: muted teal /
# coral / orange / lavender / navy.  LASO(Ours) gets the strongest tone
# (dark navy) so it pops against the four baseline methods.
METHOD_COLORS = {
    "BasicTruth":    "#E76F51",   # coral red
    "DecentTruth":   "#2A9D8F",   # teal
    "SenFeedTruth":  "#B8A8D8",   # lavender
    "SenteTruth":    "#F4A261",   # warm orange
    "SemanticTruth": "#1F6FEB",   # bright blue -> highlights LASO(Ours)
}
METHOD_MARKERS = {
    "BasicTruth":    "o",
    "DecentTruth":   "s",
    "SenFeedTruth":  "D",
    "SenteTruth":    "^",
    "SemanticTruth": "v",
}
# Distinct line styles ensure B&W readability even if colors collapse to grey.
METHOD_LINESTYLES = {
    "BasicTruth":    (0, (1, 1.5)),     # dotted
    "DecentTruth":   (0, (5, 2)),       # dashed
    "SenFeedTruth":  (0, (3, 1.5, 1, 1.5)),  # dash-dot
    "SenteTruth":    (0, (6, 2, 1, 2, 1, 2)), # dash-dot-dot
    "SemanticTruth": "-",                # solid (Ours stands out)
}
# Hatch patterns make grouped bar charts readable in grayscale prints.
METHOD_HATCHES = {
    "BasicTruth":    "////",
    "DecentTruth":   "\\\\\\\\",
    "SenFeedTruth":  "xxxx",
    "SenteTruth":    "....",
    "SemanticTruth": "",  # solid black fill, no hatch -> most striking bar
}
# Marker face fill — open (white) for baselines, filled (color) for Ours.
METHOD_MFC = {
    "BasicTruth":    "white",
    "DecentTruth":   "white",
    "SenFeedTruth":  "white",
    "SenteTruth":    "white",
    "SemanticTruth": None,  # use the line color (filled)
}
NUMERIC_ORDER = ["BasicTruth", "DecentTruth", "SenFeedTruth", "SenteTruth", "SemanticTruth"]
TEXT_ORDER = ["BasicTruth", "SenteTruth", "SemanticTruth"]

DISPLAY = {"SemanticTruth": "LASO(Ours)"}


def disp(m: str) -> str:
    return DISPLAY.get(m, m)


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, name: str):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# EXP1 NUMERIC — grouped bar chart of mean RMSE per method per bad_count.
# ---------------------------------------------------------------------------

def fig_exp1_numeric(csv_path: Path, name: str):
    rows = read_csv(csv_path)
    bad_col = "bad_count"
    methods = [m for m in NUMERIC_ORDER if any(r["method"] == m for r in rows)]
    bads = sorted({int(r[bad_col]) for r in rows})
    # Only show 0–50% malicious ratio range.
    if bads:
        n_total = max(bads) * 2 if max(bads) > 4 else 10
        bads = [b for b in bads if b * 100 / n_total <= 50]
    means = {m: [np.mean([float(r["rmse"]) for r in rows
                          if int(r[bad_col]) == b and r["method"] == m]) for b in bads]
             for m in methods}

    # Pick a label format. If any value is below 0.1 or above 1e4, switch
    # the y-axis and the per-bar labels to a fixed power-of-10 multiplier
    # (e.g. "1.6" with a "×10⁻³" axis offset) instead of long decimals.
    all_vals = [v for m in methods for v in means[m] if np.isfinite(v) and v > 0]
    vmax = max(all_vals) if all_vals else 1.0
    use_sci = vmax < 0.1 or vmax >= 1e4
    # Pick a single shared exponent so every label sits on the same scale
    # as the y-axis offset.  vmax = 0.02 -> exponent -3 (so 0.02 displays as
    # 20).  vmax = 8e-6 -> exponent -6 (so 1.5e-6 displays as 1.5).
    if use_sci:
        sci_exp = int(np.floor(np.log10(vmax)))
        # Prefer a "1, 10, 100" style band: exponent giving values in [1, 100).
        if vmax / 10 ** sci_exp < 2:
            sci_exp -= 1
    else:
        sci_exp = 0
    sci_scale = 10 ** sci_exp

    def fmt(v: float) -> str:
        if not np.isfinite(v) or v <= 0:
            return "0"
        if use_sci:
            return f"{v / sci_scale:.2g}"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.2f}"
        return f"{v:.3f}"

    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * len(bads) + 2.0), 3.4))
    x = np.arange(len(bads))
    width = 0.78 / len(methods)
    # Slightly lighter facecolor + dark edge so the hatch pattern shows up
    # clearly when the chart is printed in grayscale.
    bar_artists = []
    for i, m in enumerate(methods):
        face = "#ffffff" if m != "SemanticTruth" else METHOD_COLORS[m]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, means[m],
                      width=width, label=disp(m),
                      facecolor=face,
                      edgecolor=METHOD_COLORS[m],
                      hatch=METHOD_HATCHES[m],
                      linewidth=0.8)
        bar_artists.append((m, bars))
    ax.set_xticks(x)
    if max(bads) <= 4:
        ax.set_xticklabels([str(b) for b in bads])
        ax.set_xlabel("Malicious agents (of 10)")
    else:
        ax.set_xticklabels([f"{int(round(100*b/50))}%" for b in bads])
        ax.set_xlabel("Malicious ratio")
    ax.set_ylabel("Relative RMSE")

    # When the values are tiny, swap the y-axis tick formatter so ticks are
    # shown rescaled by `sci_scale`, with a "x10^k" offset above the axis.
    if use_sci:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v / sci_scale:g}"))
        ax.text(0.0, 1.01, rf"$\times 10^{{{sci_exp}}}$",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color="#222222")

    # Headroom for value labels above each bar.
    ax.set_ylim(0, vmax * 1.10)

    # Annotate each bar with its value, drawn horizontally above the bar.
    # Bars at bad_count == 0 are skipped per request — that column is the
    # noise floor and the digits crowd the rest of the chart.
    for m, bars in bar_artists:
        for j, rect in enumerate(bars):
            if bads[j] == 0:
                continue
            h = rect.get_height()
            if not np.isfinite(h) or h <= 0:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2,
                    h + vmax * 0.012, fmt(h),
                    ha="center", va="bottom", rotation=0,
                    fontsize=6.0, color="#222222", clip_on=False)

    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, 1.16), fontsize=8.5,
              columnspacing=1.2, handletextpad=0.5, handlelength=1.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP1 TEXT — DataAcc vs malicious ratio.
# ---------------------------------------------------------------------------

def fig_exp1_text(csv_path: Path, name: str):
    rows = read_csv(csv_path)
    # Only show 0–50% malicious ratio range.
    rows = [r for r in rows if float(r["malicious_ratio_pct"]) <= 50]
    ratios = [float(r["malicious_ratio_pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in TEXT_ORDER:
        if m not in rows[0]:
            continue
        ys = [float(r[m]) for r in rows]
        mfc = METHOD_MFC[m] if METHOD_MFC[m] is not None else METHOD_COLORS[m]
        ax.plot(ratios, ys, label=disp(m),
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linestyle=METHOD_LINESTYLES[m],
                markersize=6, linewidth=1.6 if m == "SemanticTruth" else 1.3,
                markerfacecolor=mfc,
                markeredgewidth=1.0,
                markeredgecolor=METHOD_COLORS[m])
    ax.set_xlabel("Malicious ratio (%)")
    ax.set_ylabel("DataAcc")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(ratios)
    ax.legend(loc="lower left", fontsize=8.5, ncol=1,
              handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP3 NUMERIC — per-batch RMSE timeseries with three phase markers.
# ---------------------------------------------------------------------------

def fig_exp3_numeric(csv_path: Path, name: str):
    rows = read_csv(csv_path)
    methods = [m for m in NUMERIC_ORDER if any(r["method"] == m for r in rows)]
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

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xmax = 0
    for m in methods:
        xs, ys = series[m]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        xmax = max(xmax, int(xs.max()))
        ax.plot(xs, ys, label=disp(m), color=METHOD_COLORS[m],
                linestyle=METHOD_LINESTYLES[m],
                linewidth=1.4 if m == "SemanticTruth" else 0.95,
                alpha=0.95)
    for x in (p1, p2, p3):
        ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.25)
    labels = [
        ((1 + p1) / 2,            "no attack"),
        ((p1 + p2) / 2,           f"{int(round(100*p1b/50))}%"),
        ((p2 + p3) / 2,           f"{int(round(100*p2b/50))}%"),
        ((p3 + xmax) / 2,         f"{int(round(100*p3b/50))}%"),
    ]
    y_lo, y_hi = ax.get_ylim()
    y_mid = (y_lo + y_hi) / 2
    for x, txt in labels:
        ax.text(x, y_mid, txt, ha="center", va="center",
                fontsize=11.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.26", fc="white",
                          ec="#bdbdbd", lw=0.5))
    ax.set_xlabel("Object")
    ax.set_ylabel("Relative RMSE")
    ax.set_xlim(1, xmax)
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8.5,
              columnspacing=1.2, handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP3 TEXT — per-batch mean hybrid score timeseries with phase markers.
# ---------------------------------------------------------------------------

def fig_exp3_text(csv_path: Path, name: str, tau: float = 0.85):
    rows = read_csv(csv_path)
    methods = [m for m in TEXT_ORDER if any(r["method"] == m for r in rows)]
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

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xmax = 0
    for m in methods:
        xs, ys = series[m]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        xmax = max(xmax, int(xs.max()))
        ax.plot(xs, ys, label=disp(m), color=METHOD_COLORS[m],
                linestyle=METHOD_LINESTYLES[m],
                linewidth=1.2 if m == "SemanticTruth" else 0.85,
                alpha=0.9)
    for x in (p1, p2, p3):
        ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1.05)
    labels = [
        ((1 + p1) / 2,    "no attack"),
        ((p1 + p2) / 2,   f"{int(round(100*p1b/50))}%"),
        ((p2 + p3) / 2,   f"{int(round(100*p2b/50))}%"),
        ((p3 + xmax) / 2, f"{int(round(100*p3b/50))}%"),
    ]
    for x, txt in labels:
        ax.text(x, 0.12, txt, ha="center", va="bottom", fontsize=11.5,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.26", fc="white",
                          ec="#bdbdbd", lw=0.5))
    ax.axhline(tau, color="black", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.text(xmax * 0.012, tau + 0.01, rf"$\tau={tau:.2f}$",
            color="black", alpha=0.65, fontsize=10)
    ax.set_xlabel("Object", fontsize=10)
    ax.set_ylabel("Similarity score", fontsize=10)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_xlim(1, xmax)
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=10,
              columnspacing=1.2, handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP3 TEXT (τ=0.90 view) — rolling-mean per-batch DataAcc with phase markers.
# Mean hybrid score is τ-independent, so a stricter threshold only shows up in
# the per-batch DataAcc.  Smooth with a centred rolling window so each method
# is a continuous line instead of an unreadable 0/1 spike train.
# ---------------------------------------------------------------------------

def fig_exp3_text_acc(csv_path: Path, name: str, tau: float = 0.90,
                      window: int = 20):
    rows = read_csv(csv_path)
    methods = [m for m in TEXT_ORDER if any(r["method"] == m for r in rows)]
    p1 = int(rows[0]["phase1_start_batch"]); p1b = int(rows[0]["phase1_bad_count"])
    p2 = int(rows[0]["phase2_start_batch"]); p2b = int(rows[0]["phase2_bad_count"])
    p3 = int(rows[0]["phase3_start_batch"]); p3b = int(rows[0]["phase3_bad_count"])

    series = {m: ([], []) for m in methods}
    for r in rows:
        m = r["method"]
        if m not in series:
            continue
        series[m][0].append(int(r["batch_idx"]))
        series[m][1].append(float(r["data_accuracy"]))

    def rolling(y, w):
        y = np.asarray(y, dtype=float)
        if len(y) < w:
            return y
        kernel = np.ones(w) / w
        # 'same' length, edge-pad replicate to keep window stable at boundaries
        pad = w // 2
        ext = np.concatenate([np.full(pad, y[0]), y, np.full(w - 1 - pad, y[-1])])
        return np.convolve(ext, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xmax = 0
    for m in methods:
        xs, ys = series[m]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ys_s = rolling(ys, window)
        xmax = max(xmax, int(xs.max()))
        ax.plot(xs, ys_s, label=disp(m), color=METHOD_COLORS[m],
                linestyle=METHOD_LINESTYLES[m],
                linewidth=1.6 if m == "SemanticTruth" else 1.2,
                alpha=0.95)
    for x in (p1, p2, p3):
        ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylim(-0.04, 1.22)
    labels = [
        ((1 + p1) / 2,    "no attack"),
        ((p1 + p2) / 2,   f"{int(round(100*p1b/50))}%"),
        ((p2 + p3) / 2,   f"{int(round(100*p2b/50))}%"),
        ((p3 + xmax) / 2, f"{int(round(100*p3b/50))}%"),
    ]
    for x, txt in labels:
        ax.text(x, 1.17, txt, ha="center", va="top", fontsize=8.5,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#bdbdbd", lw=0.4))
    ax.set_xlabel("Object")
    ax.set_ylabel(f"DataAcc (rolling, " + r"$\tau=" + f"{tau:.2f}" + r"$)")
    ax.set_xlim(1, xmax)
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8.5,
              columnspacing=1.2, handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP4 NUMERIC — mean RMSE vs N agents.
# ---------------------------------------------------------------------------

def fig_exp4_numeric(csv_path: Path, name: str):
    rows = read_csv(csv_path)
    methods = [m for m in NUMERIC_ORDER if any(r["method"] == m for r in rows)]
    Ns = sorted({int(r["agent_count"]) for r in rows})
    rmse = {m: [np.mean([float(r["rmse"]) for r in rows
                         if int(r["agent_count"]) == n and r["method"] == m]) for n in Ns]
            for m in methods}

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in methods:
        mfc = METHOD_MFC[m] if METHOD_MFC[m] is not None else METHOD_COLORS[m]
        ax.plot(Ns, rmse[m], label=disp(m),
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linestyle=METHOD_LINESTYLES[m],
                markersize=5.5, linewidth=1.6 if m == "SemanticTruth" else 1.2,
                markerfacecolor=mfc,
                markeredgewidth=1.0,
                markeredgecolor=METHOD_COLORS[m])
    ax.set_xlabel("Agents (N)")
    ax.set_ylabel("Relative RMSE")
    ax.set_xticks(Ns)
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, 1.16), fontsize=8.5,
              columnspacing=1.0, handletextpad=0.4, handlelength=2.2)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP4 TEXT — DataAcc vs N agents.
# ---------------------------------------------------------------------------

def fig_exp4_text(csv_path: Path, name: str):
    rows = read_csv(csv_path)
    methods = [m for m in TEXT_ORDER if any(r["method"] == m for r in rows)]
    Ns = sorted({int(r["agent_count"]) for r in rows})
    series = {m: [np.mean([float(r["data_accuracy"]) for r in rows
                           if int(r["agent_count"]) == n and r["method"] == m]) for n in Ns]
              for m in methods}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in methods:
        mfc = METHOD_MFC[m] if METHOD_MFC[m] is not None else METHOD_COLORS[m]
        ax.plot(Ns, series[m], label=disp(m),
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linestyle=METHOD_LINESTYLES[m],
                markersize=6, linewidth=1.6 if m == "SemanticTruth" else 1.3,
                markerfacecolor=mfc,
                markeredgewidth=1.0,
                markeredgecolor=METHOD_COLORS[m])
    ax.set_xlabel("Agents (N)")
    ax.set_ylabel("DataAcc")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(Ns)
    ax.legend(loc="lower right", fontsize=8.5,
              handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EXP7 — alpha sensitivity sweep (per-method 3-subplot + summary).
# ---------------------------------------------------------------------------

def fig_exp7(csv_path: Path):
    rows = read_csv(csv_path)
    methods = TEXT_ORDER
    bads = sorted({int(r["bad_count"]) for r in rows})
    alphas = sorted({float(r["alpha"]) for r in rows})

    nested = {m: {b: {a: None for a in alphas} for b in bads} for m in methods}
    for r in rows:
        nested[r["method"]][int(r["bad_count"])][float(r["alpha"])] = float(r["data_accuracy"])

    # ----- summary line plot -----
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in methods:
        means = [np.mean([nested[m][b][a] for b in bads]) for a in alphas]
        mfc = METHOD_MFC[m] if METHOD_MFC[m] is not None else METHOD_COLORS[m]
        ax.plot(alphas, means, label=disp(m),
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linestyle=METHOD_LINESTYLES[m],
                markersize=5.5, linewidth=1.6 if m == "SemanticTruth" else 1.2,
                markerfacecolor=mfc,
                markeredgewidth=1.0,
                markeredgecolor=METHOD_COLORS[m])
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8,
               label=r"default $\alpha=0.5$")
    ax.set_xlabel(r"$\alpha$ (semantic weight)")
    ax.set_ylabel("Mean DataAcc")
    ax.set_ylim(0.3, 0.95)
    ax.legend(loc="lower right", fontsize=8.5,
              handletextpad=0.5, handlelength=2.4)
    fig.tight_layout()
    save(fig, "exp7_alpha_summary")


# ---------------------------------------------------------------------------
# Driver — generate every PNG the user listed, in `docs/figs_v2/`.
# ---------------------------------------------------------------------------

JOBS = [
    # ---- EXP1 NUMERIC ----
    ("exp1_numeric_50agent",
     lambda: fig_exp1_numeric(
         ROOT / "exp1_numeric/data_agent-50_news-300_collude_relative/numeric_per_batch_rmse.csv",
         "exp1_numeric_50agent")),
    # ---- EXP1 TEXT (agent-50) ----
    ("exp1_text_1_2_agent",
     lambda: fig_exp1_text(
         ROOT / "exp1_text/1.2-agent-50-news-300-50%-0.9-numonly/text_data_accuracy.csv",
         "exp1_text_1_2_agent")),
    ("exp1_text_1_4_agent",
     lambda: fig_exp1_text(
         ROOT / "exp1_text/1.4-agent-50-weather-300-50%-0.9-numonly/text_data_accuracy.csv",
         "exp1_text_1_4_agent")),
    # ---- EXP1 TEXT (llama) ----
    ("exp1_text_1_2_llama",
     lambda: fig_exp1_text(
         ROOT / "exp1_text/1.2-llama-news-300-50%-0.9-numonly/text_data_accuracy.csv",
         "exp1_text_1_2_llama")),
    ("exp1_text_1_4_llama",
     lambda: fig_exp1_text(
         ROOT / "exp1_text/1.4-llama-weather-300-50%-0.9-numonly/text_data_accuracy.csv",
         "exp1_text_1_4_llama")),
    # ---- EXP3 ----
    ("exp3_numeric_news",
     lambda: fig_exp3_numeric(
         ROOT / "exp3_numeric/data_agent-50_news-300/5bad_at_50__10bad_at_100__15bad_at_200/results.csv",
         "exp3_numeric_news")),
    ("exp3_text_news_t090",
     lambda: fig_exp3_text(
         ROOT / "exp3_text/data_agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200__thr0.90/results.csv",
         "exp3_text_news_t090", tau=0.90)),
    ("exp3_text_weather_t090",
     lambda: fig_exp3_text(
         ROOT / "exp3_text/data_agent-50_weather-300/10bad_at_50__20bad_at_100__30bad_at_200__thr0.90/results.csv",
         "exp3_text_weather_t090", tau=0.90)),
    ("exp3_text_news_t090_llama",
     lambda: fig_exp3_text(
         ROOT / "exp3_text/data-llama-agent-50_news-300/10bad_at_50__20bad_at_100__30bad_at_200__thr0.90/results.csv",
         "exp3_text_news_t090_llama", tau=0.90)),
    ("exp3_text_weather_t090_llama",
     lambda: fig_exp3_text(
         ROOT / "exp3_text/data-llama-agent-50_weather-300/10bad_at_50__20bad_at_100__30bad_at_200__thr0.90/results.csv",
         "exp3_text_weather_t090_llama", tau=0.90)),
    # ---- EXP4 ----
    ("exp4_numeric_news",
     lambda: fig_exp4_numeric(
         ROOT / "exp4_numeric/data_agent-50_news-300/bad_5/metric_relative/results.csv",
         "exp4_numeric_news")),
    ("exp4_text_news_t090",
     lambda: fig_exp4_text(
         ROOT / "exp4_text/data_agent-50_news-300/bad_5/threshold_0.90/results.csv",
         "exp4_text_news_t090")),
    ("exp4_text_weather_t090",
     lambda: fig_exp4_text(
         ROOT / "exp4_text/data_agent-50_weather-300/bad_5/threshold_0.90/results.csv",
         "exp4_text_weather_t090")),
]


def main():
    pngs = []
    for name, fn in JOBS:
        png = fn()
        print(f"  -> {png.relative_to(OUT.parent.parent)}")
        pngs.append((name, png))

    # EXP7 writes two figures in one call; append after the JOBS loop.
    fig_exp7(ROOT / "exp7_alpha_sensitivity/results.csv")
    for tag in ("exp7_alpha_summary",):
        png = OUT / f"{tag}.png"
        print(f"  -> {png.relative_to(OUT.parent.parent)}")
        pngs.append((tag, png))

    # Bundle all PNGs into a single PDF (one figure per page).
    bundle = OUT / "figures_compiled.pdf"
    with PdfPages(bundle) as pdf:
        for name, png in pngs:
            img = plt.imread(png)
            h, w = img.shape[:2]
            # 8.5 x 11 letter, fit by aspect.
            page_w = 8.0
            page_h = page_w * h / w
            fig = plt.figure(figsize=(page_w, page_h))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.set_axis_off()
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
    print(f"  -> {bundle.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
