"""Render the §6.4 four-stage ChainMaker bar chart.

Each phase uses a single shared bar style (one fill colour + one hatch);
step names are annotated directly on / above their bars rather than via a
separate legend.  Layout follows the on-chain phases in
``decentralized_truth_discovery/result/figures/流程图.svg``:

  1. Query Phase            — one bar per step, side by side
  2. Evidence Collection    — single stacked bar (4 serial callbacks)
  3. Truth Discovery        — one bar per step, side by side
  4. On-chain Verification  — one bar per step, side by side

(ZK Proof phase is off-chain in the workflow diagram and is not measured
in §6.4.)

All values are 10-run averages from
``docs/fisco_vs_chainmaker_benchmark.md`` §4 (ChainMaker column).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


OUT = Path("/Users/huabowen/TruthFinder/docs/figs_v2")
OUT.mkdir(parents=True, exist_ok=True)


CALLBACK_AVG = 2107.1 / 4  # 526.775 ms — 4 serial agent callbacks


# Each phase has ONE shared style.  Step labels follow the flowchart names.
STAGES = [
    {
        "title": "Query Phase",
        "layout": "split",
        "face":  "#FFB59E",
        "edge":  "#C84A2C",
        "hatch": "////",
        "intra_gap": 0.20,  # tighter than the other split phases
        "steps": [
            ("submit query", 520.5),
            ("query event",    9.3),
        ],
    },
    {
        "title": "Evidence Collection",
        "layout": "stacked",
        "face":  "#A8DCD3",
        "edge":  "#1F6F66",
        "hatch": "\\\\\\\\",
        "steps": [
            ("claim report (agent-1)", CALLBACK_AVG),
            ("claim report (agent-2)", CALLBACK_AVG),
            ("claim report (agent-3)", CALLBACK_AVG),
            ("claim report (agent-4)", CALLBACK_AVG),
        ],
    },
    {
        "title": "Truth Discovery",
        "layout": "split",
        "face":  "#D2B7E5",
        "edge":  "#5B2A8D",
        "hatch": "xxxx",
        "gap_before": 0.55,  # pull phase closer to Evidence Collection callout
        "steps": [
            ("claim reports",          63.2),
            ("result + reliabilities",  8.9),
        ],
    },
    {
        "title": "On-chain Verification",
        "layout": "split",
        "face":  "#AFCBF5",
        "edge":  "#1A4FA6",
        "hatch": "....",
        "intra_gap": 0.25,   # tighter packing for the three steps
        "gap_before": 0.55,  # pull phase closer to Truth Discovery
        # ``submissionKey`` (8.8 ms) is not a separate step in 流程图.svg —
        # it is an internal helper of ``submitProof`` (a client-side key
        # derivation done immediately before the submit call), so its time
        # is folded into ``store results``: 1536.5 + 8.8 = 1545.3 ms.
        "steps": [
            ("VerifyProof",      523.7),
            ("store results",   1545.3),
            ("output available",   7.8),
        ],
    },
]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 21,
    "axes.labelsize": 23,
    "axes.titlesize": 23,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#222222",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.35,
    "grid.color": "#bdbdbd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 3,
    "hatch.linewidth": 0.6,
})


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v:.0f}"
    return f"{v:.1f}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

BAR_WIDTH         = 0.55   # width of each individual bar
INTRA_GAP         = 0.55   # gap between bars within the same phase
PHASE_GAP         = 0.95   # default gap between phases
STACKED_RIGHT_GAP = 1.30   # room to the right of a stacked phase for its callout


def main() -> None:
    # Plan x positions; record phase centres for x-tick labels.
    plan: list[dict] = []
    phase_centers: list[float] = []
    cursor = 0.45  # shift Query Phase slightly right of the y-axis
    last_bar_right = 0.0  # right edge of the right-most rendered bar
    n_stages = len(STAGES)
    for i, stage in enumerate(STAGES):
        is_last_stage = (i == n_stages - 1)
        steps = stage["steps"]
        if stage["layout"] == "split":
            intra_gap = stage.get("intra_gap", INTRA_GAP)
            xs = []
            for name, v in steps:
                plan.append({
                    "kind": "single", "x": cursor,
                    "name": name, "value": v, "stage": stage,
                })
                xs.append(cursor)
                cursor += BAR_WIDTH + intra_gap
            cursor -= intra_gap
            phase_centers.append((xs[0] + xs[-1]) / 2)
            last_bar_right = xs[-1] + BAR_WIDTH / 2
        else:
            plan.append({
                "kind": "stack", "x": cursor,
                "steps": steps, "stage": stage,
            })
            phase_centers.append(cursor)
            cursor += BAR_WIDTH
            last_bar_right = cursor - BAR_WIDTH / 2
            # Stacked phases reserve extra room on the right for callouts,
            # unless the next phase explicitly overrides the gap.
            if not is_last_stage:
                next_gap = STAGES[i + 1].get("gap_before", STACKED_RIGHT_GAP)
                cursor += next_gap
            continue
        if not is_last_stage:
            next_gap = STAGES[i + 1].get("gap_before", PHASE_GAP)
            cursor += next_gap
    total_x = cursor

    # Compute y-axis scale.
    all_vals: list[float] = []
    for stage in STAGES:
        if stage["layout"] == "stacked":
            all_vals.append(sum(v for _, v in stage["steps"]))
        else:
            all_vals.extend(v for _, v in stage["steps"])
    vmax = max(all_vals) * 1.18

    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.07, right=0.995)

    # ----- bars -----
    for item in plan:
        stage = item["stage"]
        if item["kind"] == "single":
            v = item["value"]
            name = item["name"]
            ax.bar(
                item["x"], v, width=BAR_WIDTH,
                facecolor=stage["face"], edgecolor=stage["edge"],
                hatch=stage["hatch"], linewidth=0.9, zorder=3,
            )
            # Step name above the bar, value below it.
            ax.text(
                item["x"], v + vmax * 0.012,
                f"{name}\n{fmt_ms(v)} ms",
                ha="center", va="bottom",
                fontsize=19.5, color="#1a1a1a", linespacing=1.25,
                zorder=4,
            )
        else:  # stacked: one consolidated annotation to the right
            bottom = 0.0
            for name, v in item["steps"]:
                ax.bar(
                    item["x"], v, width=BAR_WIDTH,
                    bottom=bottom,
                    facecolor=stage["face"], edgecolor=stage["edge"],
                    hatch=stage["hatch"], linewidth=0.9, zorder=3,
                )
                bottom += v
            # Single consolidated callout — all four segments share the same
            # name pattern and value, so one block is enough.
            per_step = item["steps"][0][1]
            x_right = item["x"] + BAR_WIDTH / 2
            ax.annotate(
                f"claim report\n(agent-1..4)\n≈ {fmt_ms(per_step)} ms each",
                xy=(x_right, bottom / 2),
                xytext=(x_right + 0.06, bottom / 2),
                ha="left", va="center",
                fontsize=19.5, color="#1a1a1a", linespacing=1.3,
                arrowprops=dict(
                    arrowstyle="-",
                    color=stage["edge"],
                    lw=0.7,
                    shrinkA=0, shrinkB=2,
                ),
                zorder=4,
            )
            ax.text(
                item["x"], bottom + vmax * 0.012,
                f"Σ {fmt_ms(bottom)} ms",
                ha="center", va="bottom",
                fontsize=21.0, fontweight="bold", color="#1a1a1a",
                zorder=4,
            )

    # ----- x-axis: phase names only -----
    ax.set_xticks(phase_centers)
    ax.set_xticklabels(
        [s["title"] for s in STAGES],
        fontsize=22.0, fontweight="bold",
    )
    ax.tick_params(axis="x", pad=6)

    # ----- y-axis -----
    ax.set_ylim(0, vmax)
    ax.set_xlim(-BAR_WIDTH * 0.15, last_bar_right + BAR_WIDTH * 0.75)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylabel("Elapsed time (ms)")

    png = OUT / "exp64_chain_stage_bars.png"
    pdf = OUT / "exp64_chain_stage_bars.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
