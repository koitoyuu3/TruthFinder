#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

DEFAULT_INPUT = Path(__file__).with_name('混合-随着攻击时间变化.csv')
DEFAULT_PNG = Path(__file__).with_name('figures') / '混合-随着攻击时间变化_attack_induced_shock_panels.png'
DEFAULT_PDF = Path(__file__).with_name('figures') / '混合-随着攻击时间变化_attack_induced_shock_panels.pdf'
SCENARIOS = ['early', 'mid', 'late']
TITLE_MAP = {'early': 'Early', 'mid': 'Mid', 'late': 'Late'}
LINE_COLOR = '#00BFA5'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot attack-induced shock mean across rounds for early/mid/late attack timing.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Input CSV path.')
    parser.add_argument('--png', type=Path, default=DEFAULT_PNG, help='Output PNG path.')
    parser.add_argument('--pdf', type=Path, default=DEFAULT_PDF, help='Output PDF path.')
    parser.add_argument('--dpi', type=int, default=260, help='PNG DPI.')
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 10.5,
            'xtick.labelsize': 8.5,
            'ytick.labelsize': 8.5,
            'axes.linewidth': 0.9,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.bbox': 'tight',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'scenario', 'round', 'attack_induced_shock_mean'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df


def plot_panels(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    plot_df = df[df['scenario'].isin(SCENARIOS)].copy()
    y_max = float(plot_df['attack_induced_shock_mean'].max()) if not plot_df.empty else 0.0
    y_upper = max(y_max * 1.08, 0.001)
    y_lower = -max(y_upper * 0.05, 0.0008)

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.23, top=0.80, wspace=0.12)

    for ax, scenario in zip(axes, SCENARIOS):
        scenario_df = plot_df[plot_df['scenario'] == scenario].sort_values('round')
        ax.plot(
            scenario_df['round'],
            scenario_df['attack_induced_shock_mean'],
            color=LINE_COLOR,
            linewidth=2.5,
            solid_capstyle='round',
        )
        ax.set_title(TITLE_MAP[scenario], pad=6)
        ax.set_xlabel('Round')
        ax.set_xlim(float(scenario_df['round'].min()), float(scenario_df['round'].max()))
        ax.set_ylim(y_lower, y_upper)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(axis='y', linestyle='--', linewidth=0.75, alpha=0.30)
        ax.grid(axis='x', linestyle='--', linewidth=0.55, alpha=0.14)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', width=0.9, length=3.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#4a4a4a')
        ax.spines['bottom'].set_color('#4a4a4a')
        ax.spines['left'].set_linewidth(0.9)
        ax.spines['bottom'].set_linewidth(0.9)

    axes[0].set_ylabel('attack_induced_shock_mean')
    fig.suptitle('Attack-Induced Shock Mean Across Rounds', fontsize=14.5, y=0.965)

    fig.savefig(png_path, dpi=dpi, facecolor='white')
    fig.savefig(pdf_path, facecolor='white')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    plot_panels(df, args.png, args.pdf, args.dpi)
    print(f'Saved PNG: {args.png}')
    print(f'Saved PDF: {args.pdf}')


if __name__ == '__main__':
    main()
