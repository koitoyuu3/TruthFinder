#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEFAULT_INPUT = Path(__file__).with_name('数值-随着攻击时间变化.csv')
DEFAULT_PNG = Path(__file__).with_name('figures') / '数值-随着攻击时间变化_attack_induced_shock_panels.png'
DEFAULT_PDF = Path(__file__).with_name('figures') / '数值-随着攻击时间变化_attack_induced_shock_panels.pdf'

SCENARIOS = ['early', 'mid', 'late']
TITLE_MAP = {'early': 'Early Attack', 'mid': 'Mid Attack', 'late': 'Late Attack'}

# Ordered by descending max value in the numeric dataset
METHOD_ORDER = [
    'EnhancedTruthFinder-NonHybrid',
    'SenFeedTruthFinder-CompositeTD',
    'SenteTruthFinder',
    'BasicTruthFinder',
    'EnhancedTruthFinder-Hybrid',
]
METHOD_LABELS = {
    'EnhancedTruthFinder-Hybrid':         'Enhanced-Hybrid',
    'EnhancedTruthFinder-NonHybrid':      'Enhanced-NonHybrid',
    'SenteTruthFinder':                   'Sente',
    'BasicTruthFinder':                   'Basic',
    'SenFeedTruthFinder-CompositeTD':     'SenFeed-CompositeTD',
}
METHOD_STYLES: dict[str, dict] = {
    'EnhancedTruthFinder-Hybrid': {
        'color': '#1f77b4', 'ls': '-',                   'lw': 1.7, 'marker': 'o',  'ms': 3.2, 'zorder': 4,
    },
    'EnhancedTruthFinder-NonHybrid': {
        'color': '#ff7f0e', 'ls': '--',                  'lw': 1.7, 'marker': 's',  'ms': 3.0, 'zorder': 5,
    },
    'SenteTruthFinder': {
        'color': '#2ca02c', 'ls': '-.',                  'lw': 1.7, 'marker': '^',  'ms': 3.2, 'zorder': 3,
    },
    'BasicTruthFinder': {
        'color': '#d62728', 'ls': (0, (4, 1, 1, 1, 1, 1)), 'lw': 2.2, 'marker': 'X', 'ms': 4.2, 'zorder': 3,
    },
    'SenFeedTruthFinder-CompositeTD': {
        'color': '#9467bd', 'ls': '-',                   'lw': 2.0, 'marker': 'v',  'ms': 3.2, 'zorder': 2,
    },
}

LINTHRESH = 0.001
YTICKS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot numeric attack-induced shock mean across rounds for early/mid/late attack timing.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--png',   type=Path, default=DEFAULT_PNG)
    parser.add_argument('--pdf',   type=Path, default=DEFAULT_PDF)
    parser.add_argument('--dpi',   type=int,  default=260)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        9.5,
        'axes.titlesize':   12,
        'axes.labelsize':   10,
        'xtick.labelsize':  8.5,
        'ytick.labelsize':  8.5,
        'axes.linewidth':   0.8,
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'savefig.bbox':     'tight',
        'pdf.fonttype':     42,
        'ps.fonttype':      42,
    })


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'scenario', 'round', 'attack_induced_shock_mean'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if 'method' not in df.columns:
        df['method'] = 'EnhancedTruthFinder-Hybrid'
    return df


def _symlog_fmt(val: float, _pos: int) -> str:
    if val == 0:
        return '0'
    if abs(val) < 0.01:
        return f'{val:.3f}'
    if abs(val) < 0.1:
        return f'{val:.2f}'
    return f'{val:.1f}'


def plot_panels(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    plot_df = df[df['scenario'].isin(SCENARIOS)].copy()
    if 'track' in plot_df.columns:
        plot_df = plot_df[plot_df['track'] == 'attacked'].copy()

    y_max = float(plot_df['attack_induced_shock_mean'].max()) if not plot_df.empty else 1.0
    y_upper = max(y_max * 1.12, 0.01)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.0, 4.4),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.20, top=0.72, wspace=0.10)

    handles_for_legend: list = []
    labels_for_legend:  list = []

    for col_idx, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        scenario_df = plot_df[plot_df['scenario'] == scenario].copy()
        round_min = int(scenario_df['round'].min()) if not scenario_df.empty else 1
        round_max = int(scenario_df['round'].max()) if not scenario_df.empty else 50

        for method in METHOD_ORDER:
            method_df = scenario_df[scenario_df['method'] == method].sort_values('round')
            if method_df.empty:
                continue
            st = METHOD_STYLES.get(method, {'color': '#555555', 'ls': '-', 'lw': 1.5,
                                             'marker': None, 'ms': 3.0, 'zorder': 2})
            rounds = method_df['round'].to_numpy()
            shock  = method_df['attack_induced_shock_mean'].to_numpy()
            markevery = max(1, len(rounds) // 8)

            line, = ax.plot(
                rounds, shock,
                color=st['color'],
                linestyle=st['ls'],
                linewidth=st['lw'],
                marker=st.get('marker'),
                markersize=st.get('ms', 3.0),
                markevery=markevery,
                markeredgewidth=0.6,
                markeredgecolor='white',
                alpha=0.92,
                solid_capstyle='round',
                label=METHOD_LABELS.get(method, method),
                zorder=st.get('zorder', 2),
            )
            if col_idx == 0:
                handles_for_legend.append(line)
                labels_for_legend.append(METHOD_LABELS.get(method, method))

        # ── symlog y-axis ──────────────────────────────────────────────────
        ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=0.5)
        ax.set_ylim(0.0, y_upper)
        ax.yaxis.set_major_locator(mticker.FixedLocator(
            [t for t in YTICKS if t <= y_upper * 1.01]
        ))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_symlog_fmt))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

        ax.axhline(LINTHRESH, color='#bbbbbb', linewidth=0.6, linestyle='--', zorder=0)
        ax.text(
            round_max - 0.5, LINTHRESH * 1.18,
            f'linear|log\n({LINTHRESH})',
            fontsize=6.2, color='#999999', ha='right', va='bottom',
        )

        # ── x-axis ────────────────────────────────────────────────────────
        ax.set_xlim(float(round_min), float(round_max))
        ax.set_xticks([t for t in [1, 10, 20, 30, 40, 50]
                       if round_min <= t <= round_max] or [round_min, round_max])

        # ── labels & decoration ───────────────────────────────────────────
        ax.set_title(TITLE_MAP[scenario], pad=7, fontsize=12, fontweight='semibold')
        ax.set_xlabel('Round', labelpad=4)
        ax.grid(axis='y', linestyle='--', linewidth=0.65, alpha=0.28, zorder=0)
        ax.grid(axis='x', linestyle=':', linewidth=0.5,  alpha=0.18, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', width=0.8, length=3.0, pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#4a4a4a')
        ax.spines['bottom'].set_color('#4a4a4a')

    axes[0].set_ylabel('Attack-Induced Shock Mean\n(symlog scale)', labelpad=6)

    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=5,
        frameon=False,
        fontsize=8.8,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    fig.suptitle(
        'Attack-Induced Shock Mean Across Rounds',
        fontsize=14, y=1.08, fontweight='semibold',
    )

    fig.savefig(png_path, dpi=dpi, facecolor='white')
    fig.savefig(pdf_path, facecolor='white')
    plt.close(fig)
    print(f'Saved PNG: {png_path}')
    print(f'Saved PDF: {pdf_path}')


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    plot_panels(df, args.png, args.pdf, args.dpi)


if __name__ == '__main__':
    main()
