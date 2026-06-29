"""Generate exp7 figures from the th0.90 alpha-sensitivity sweep into docs/figs_v2/.

Outputs:
    docs/figs_v2/alpha_sensitivity_heatmap.{png,pdf}  -- new heatmap view
    docs/figs_v2/exp7_alpha_summary.{png,pdf}         -- Image #1 style (overwrite)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Paths ---------------------------------------------------------------
ROOT = Path("/Users/huabowen/TruthFinder")
CSV = ROOT / "decentralized_truth_discovery/examples/outputs/exp7_alpha_sensitivity_th0.90/results.csv"
OUT = ROOT / "docs/figs_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Style (matches _make_figs_v2.py academic look) ----------------------
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
    "SenteTruth":    "#F4A261",
    "LASO(Ours)":    "#1F6FEB",
}
METHOD_MARKERS = {
    "BasicTruth":    "o",
    "SenteTruth":    "^",
    "LASO(Ours)":    "v",
}
METHOD_LINESTYLES = {
    "BasicTruth":    (0, (1, 1.5)),
    "SenteTruth":    (0, (6, 2, 1, 2, 1, 2)),
    "LASO(Ours)":    "-",
}
METHOD_MFC = {
    "BasicTruth":    "white",
    "SenteTruth":    "white",
    "LASO(Ours)":    None,  # filled with line color
}
TEXT_ORDER = ["BasicTruth", "SenteTruth", "LASO(Ours)"]


def disp(m: str) -> str:
    return m


def save(fig, name: str) -> Path:
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


# ---- Load data -----------------------------------------------------------

def _load():
    df = pd.read_csv(CSV)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


# ---- Summary plot (Image #1 style) --------------------------------------

def make_summary(df: pd.DataFrame) -> Path:
    alphas = sorted(df["alpha"].unique())
    bads = sorted(df["bad_count"].unique())
    nested = {
        m: {
            int(b): {float(a): None for a in alphas} for b in bads
        }
        for m in TEXT_ORDER
    }
    for _, r in df.iterrows():
        nested[r["method"]][int(r["bad_count"])][float(r["alpha"])] = float(r["data_accuracy"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for m in TEXT_ORDER:
        means = [
            float(np.mean([nested[m][int(b)][float(a)] for b in bads]))
            for a in alphas
        ]
        mfc = METHOD_MFC[m] if METHOD_MFC[m] is not None else METHOD_COLORS[m]
        ax.plot(
            alphas, means,
            label=disp(m),
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            linestyle=METHOD_LINESTYLES[m],
            markersize=5.5,
            linewidth=1.6 if m == "LASO(Ours)" else 1.2,
            markerfacecolor=mfc,
            markeredgewidth=1.0,
            markeredgecolor=METHOD_COLORS[m],
        )
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8,
               label=r"default $\omega=0.5$")
    ax.set_xlabel(r"$\omega$ (semantic weight)")
    ax.set_ylabel("Mean DataAcc")
    ax.set_ylim(0.3, 0.95)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        fontsize=8.5,
        columnspacing=1.2,
        handletextpad=0.5,
        handlelength=2.4,
        frameon=False,
    )
    fig.tight_layout()
    return save(fig, "exp7_alpha_summary")


# ---- Heatmap ------------------------------------------------------------

def make_heatmap(df: pd.DataFrame, threshold: float = 0.9) -> Path:
    methods_disp = [(m, disp(m)) for m in TEXT_ORDER]
    alphas = sorted(df["alpha"].unique())
    pcts = sorted(df["malicious_pct"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), sharey=True)

    cmap = plt.get_cmap("viridis")
    vmin, vmax = 0.0, 1.0
    im = None

    for ax, (m, m_disp) in zip(axes, methods_disp):
        sub = df[df["method"] == m]
        mat = (
            sub.pivot(index="malicious_pct", columns="alpha", values="data_accuracy")
            .reindex(index=pcts, columns=alphas)
            .values
        )
        mat_disp = mat[::-1, :]
        pcts_disp = pcts[::-1]

        im = ax.imshow(
            mat_disp, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )

        for i in range(mat_disp.shape[0]):
            for j in range(mat_disp.shape[1]):
                val = mat_disp[i, j]
                if np.isnan(val):
                    continue
                txt_color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.1f}" for a in alphas], fontsize=8)
        ax.set_yticks(range(len(pcts_disp)))
        ax.set_yticklabels([f"{p:.0f}%" for p in pcts_disp], fontsize=8)
        ax.set_xlabel(r"$\omega$ (semantic weight)")
        ax.text(0.5, 1.06, m_disp, transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#222222")
        ax.tick_params(axis="both", which="both", length=0)
        if 0.5 in alphas:
            j_mid = alphas.index(0.5)
            ax.axvline(j_mid, color="white", linewidth=0.9,
                       alpha=0.6, linestyle="--")

    axes[0].set_ylabel("Malicious %")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label(f"DataAcc (threshold={threshold})", fontsize=9)

    return save(fig, "exp7_alpha_sensitivity_heatmap")


def main() -> None:
    df = _load()
    p1 = make_heatmap(df)
    print(f"  -> {p1.relative_to(ROOT)}")
    p2 = make_summary(df)
    print(f"  -> {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
