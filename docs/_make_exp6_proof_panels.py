"""Render the proof-bundle summary panels at batch=150 in the same
academic paper style as docs/_make_figs_v2.py (serif font, dashed grid,
hatched bars, LASO(Ours) highlighted with a filled bar).

Source data:
    decentralized_truth_discovery/examples/outputs/exp6_proof_v2/raw_results.csv

For the noisy verifier metrics (verify_elapsed_sec, peak_delta_rss_mb)
we aggregate the 50 repeats per (domain, method) into a robust
median + interquartile range, and draw an error bar from q25 to q75.
The size metrics (proof_size_bytes, proof_payload_bytes) are constant
across repeats so we just take the first value.

Outputs:
    docs/figs_v2/exp6_proof_scalability_summary_panels.png
    docs/figs_v2/exp6_proof_scalability_summary_panels.pdf
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
INPUT = Path(
    "/Users/huabowen/TruthFinder/decentralized_truth_discovery/examples/outputs/"
    "exp6_proof_v2/raw_results.csv"
)
OUT = Path("/Users/huabowen/TruthFinder/docs/figs_v2")
OUT.mkdir(parents=True, exist_ok=True)

BATCH = 150

# ---------------------------------------------------------------------------
# Style — matches docs/_make_figs_v2.py exactly.
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
    "legend.frameon": False,
    "legend.fontsize": 9,
})

METHOD_COLORS = {
    "BasicTruth":    "#E76F51",
    "DecentTruth":   "#2A9D8F",
    "SenFeedTruth":  "#B8A8D8",
    "SenteTruth":    "#F4A261",
    "SemanticTruth": "#1F6FEB",
}
METHOD_HATCHES = {
    "BasicTruth":    "////",
    "DecentTruth":   "\\\\\\\\",
    "SenFeedTruth":  "xxxx",
    "SenteTruth":    "....",
    "SemanticTruth": "",
}
DISPLAY = {
    "SenFeedTruth":  "SenFeedTruth",
    "DecentTruth":   "DecentTruth",
    "SenteTruth":    "SenteTruth",
    "BasicTruth":    "BasicTruth",
    "SemanticTruth": "LASO(Ours)",
}
NUMERIC_ORDER = ["SenFeedTruth", "DecentTruth", "SenteTruth", "BasicTruth", "SemanticTruth"]
TEXT_ORDER    = ["SenteTruth", "BasicTruth", "SemanticTruth"]

# (raw column, panel title prefix, formatter kind, zoom_to_data_range,
#  aggregation: "first" for size metrics, "median" for noisy metrics)
METRICS = [
    ("proof_size_bytes",      "Bundle Size",           "bytes", False, "first",  "bytes"),
    ("proof_payload_bytes",   "Proof Payload Size",    "int",   True,  "first",  "bytes"),
    ("peak_delta_rss_mb",     "Verify Peak RSS Delta", "mb",    False, "median", "MB"),
    ("verify_elapsed_sec",    "Verify Time",           "sec",   False, "median", "sec"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_rows() -> list[dict[str, str]]:
    with open(INPUT, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def aggregate(rows, methods, metric, agg):
    """Aggregate raw repeats per method.

    Returns parallel lists (values, lower_err, upper_err). For "first"
    metrics the err lists contain zeros (no spread to show).
    """
    grouped = defaultdict(list)
    for r in rows:
        try:
            grouped[r["method"]].append(float(r[metric]))
        except (KeyError, ValueError):
            continue

    values, lower, upper = [], [], []
    for m in methods:
        samples = grouped.get(m, [])
        if not samples:
            values.append(float("nan"))
            lower.append(0.0)
            upper.append(0.0)
            continue
        if agg == "first":
            values.append(samples[0])
            lower.append(0.0)
            upper.append(0.0)
        else:  # median + IQR
            arr = np.asarray(samples, dtype=float)
            med = float(np.median(arr))
            q25 = float(np.quantile(arr, 0.25))
            q75 = float(np.quantile(arr, 0.75))
            values.append(med)
            lower.append(max(med - q25, 0.0))
            upper.append(max(q75 - med, 0.0))
    return values, lower, upper


def fmt_bytes(v: float) -> str:
    av = abs(v)
    if av >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if av >= 10_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def fmt_int(v: float) -> str:
    return f"{v:.0f}"


def fmt_mb(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


def fmt_sec(v: float) -> str:
    # Use 3 decimals so 0.064 vs 0.078 don't both round to 0.07/0.08.
    return f"{v:.3f}"


FMT = {"bytes": fmt_bytes, "int": fmt_int, "mb": fmt_mb, "sec": fmt_sec}


def axis_formatter(kind: str):
    if kind == "bytes":
        return FuncFormatter(lambda v, _: fmt_bytes(v))
    if kind == "int":
        return FuncFormatter(lambda v, _: fmt_int(v))
    if kind == "mb":
        return FuncFormatter(lambda v, _: fmt_mb(v))
    return FuncFormatter(lambda v, _: fmt_sec(v))


def compute_ylim(values: list[float], *, zoom: bool) -> tuple[float, float]:
    vs = [v for v in values if np.isfinite(v)]
    if not vs:
        return 0.0, 1.0
    vmin, vmax = min(vs), max(vs)
    span = vmax - vmin
    if zoom and vmin > 0 and span / max(abs(vmax), 1.0) <= 0.08:
        pad = max(span * 0.45, vmax * 0.005, 1.0)
        return max(vmin - pad, 0.0), vmax + pad
    return 0.0, vmax * 1.18 if vmax > 0 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_panel(ax, rows, methods, metric, title, fmt_kind, zoom, agg):
    values, lower, upper = aggregate(rows, methods, metric, agg)

    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        v = values[i]
        if not np.isfinite(v):
            continue
        face = METHOD_COLORS[m] if m == "SemanticTruth" else "#ffffff"
        ax.bar(x[i], v,
               width=0.66,
               facecolor=face,
               edgecolor=METHOD_COLORS[m],
               hatch=METHOD_HATCHES[m],
               linewidth=0.9,
               zorder=3)

    # IQR error bars on noisy metrics (median panels).
    if agg == "median":
        finite = [(xi, vi, lo, up) for xi, vi, lo, up
                  in zip(x, values, lower, upper) if np.isfinite(vi)]
        if finite:
            xs = [t[0] for t in finite]
            vs = [t[1] for t in finite]
            lo = [t[2] for t in finite]
            up = [t[3] for t in finite]
            ax.errorbar(xs, vs, yerr=[lo, up],
                        fmt="none", ecolor="#222222",
                        elinewidth=0.9, capsize=3.0, capthick=0.9,
                        zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(axis_formatter(fmt_kind))
    ax.set_title(title, pad=6)

    # Compute ylim including error-bar tops so caps are not clipped.
    upper_envelope = [v + u for v, u in zip(values, upper) if np.isfinite(v)]
    lo, hi = compute_ylim(upper_envelope or values, zoom=zoom)
    ax.set_ylim(lo, hi)

    # Per-bar value labels (positioned above the error bar cap).
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        cap = v + (upper[i] if agg == "median" else 0.0)
        ax.text(x[i], cap + (hi - lo) * 0.022, FMT[fmt_kind](v),
                ha="center", va="bottom",
                fontsize=8.0, color="#222222", clip_on=False, zorder=5)

    ax.margins(x=0.04)


def main() -> None:
    rows = read_rows()
    rows = [r for r in rows if int(r["batch_count"]) == BATCH]

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.055, right=0.985,
                        wspace=0.28, hspace=0.28)

    domain_methods = [("numeric", NUMERIC_ORDER), ("text", TEXT_ORDER)]
    for row_idx, (domain, methods) in enumerate(domain_methods):
        domain_rows = [r for r in rows if r["domain"] == domain]
        row_label = "Numeric" if domain == "numeric" else "Text"
        for col_idx, (metric, prefix, fmt_kind, zoom, agg, unit) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            plot_panel(ax, domain_rows, methods, metric,
                       f"{prefix} [{unit}] ({domain})", fmt_kind, zoom, agg)
            if col_idx == 0:
                ax.set_ylabel(row_label)

    # Single figure-level legend (shows hatch swatches for all 5 methods).
    legend_handles = []
    for m in NUMERIC_ORDER:
        face = METHOD_COLORS[m] if m == "SemanticTruth" else "#ffffff"
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=face,
                          edgecolor=METHOD_COLORS[m],
                          hatch=METHOD_HATCHES[m],
                          linewidth=0.9,
                          label=DISPLAY[m])
        )
    fig.legend(handles=legend_handles,
               loc="upper center",
               ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 0.985),
               frameon=False,
               handlelength=2.2,
               columnspacing=1.6,
               handletextpad=0.6)

    png = OUT / "exp6_proof_scalability_summary_panels.png"
    pdf = OUT / "exp6_proof_scalability_summary_panels.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
