#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DEFAULT_INPUT = Path(__file__).with_name('proof_可扩展性.txt')
DEFAULT_PNG = Path(__file__).with_name('figures') / 'proof_可扩展性_panels.png'
DEFAULT_PDF = Path(__file__).with_name('figures') / 'proof_可扩展性_panels.pdf'

DISPLAY_NAMES = {
    'EnhancedTruthFinder-Hybrid': 'Hybrid',
    'BasicTruthFinder': 'Basic',
    'SenteTruthFinder': 'Sente',
    'SenFeedTruthFinder-CompositeTD': 'SenFeed',
    'EnhancedTruthFinder-NonHybrid': 'NonHybrid',
}

LEGEND_NAMES = {
    'EnhancedTruthFinder-Hybrid': 'EnhancedTruthFinder (Hybrid)',
    'BasicTruthFinder': 'BasicTruthFinder',
    'SenteTruthFinder': 'SenteTruthFinder',
    'SenFeedTruthFinder-CompositeTD': 'SenFeedTruthFinder',
    'EnhancedTruthFinder-NonHybrid': 'EnhancedTruthFinder (w/o Hybrid)',
}

COLOR_MAP = {
    'EnhancedTruthFinder-Hybrid': '#1f77b4',
    'BasicTruthFinder': '#d62728',
    'SenteTruthFinder': '#2ca02c',
    'SenFeedTruthFinder-CompositeTD': '#9467bd',
    'EnhancedTruthFinder-NonHybrid': '#ff7f0e',
}

METRICS = [
    ('composite_score', 'Composite Score', '{:.1f}'),
    ('proof_bundle_bytes', 'Proof Bundle Bytes', '{:,.0f}'),
    ('verify_total_elapsed_sec', 'Verification Time (s)', '{:.3f}'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot 1x3 proof scalability bar charts from proof_可扩展性.txt.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Input TXT path.')
    parser.add_argument('--png', type=Path, default=DEFAULT_PNG, help='Output PNG path.')
    parser.add_argument('--pdf', type=Path, default=DEFAULT_PDF, help='Output PDF path.')
    parser.add_argument('--dpi', type=int, default=280, help='PNG DPI.')
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8.5,
            'ytick.labelsize': 8.5,
            'axes.linewidth': 0.9,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'legend.fontsize': 8.5,
            'savefig.bbox': 'tight',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_fwf(path)
    df.columns = [str(column).strip() for column in df.columns]
    required = {'method', 'composite_score', 'proof_bundle_bytes', 'verify_total_elapsed_sec'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df


def format_bytes(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return f'{value/1000:.0f}k'
    return f'{value:.0f}'


def plot_panels(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    methods = df['method'].astype(str).tolist()
    x_labels = [DISPLAY_NAMES.get(method, method) for method in methods]
    colors = [COLOR_MAP.get(method, '#4c4c4c') for method in methods]
    x_positions = list(range(len(methods)))

    fig, axes = plt.subplots(1, 3, figsize=(13.3, 3.7))
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.82, wspace=0.28)

    legend_handles = None
    legend_labels = None

    for ax, (metric, title, text_fmt) in zip(axes, METRICS):
        values = pd.to_numeric(df[metric], errors='coerce').astype(float).tolist()
        bars = ax.bar(x_positions, values, color=colors, width=0.68, edgecolor='none', zorder=3)
        if legend_handles is None:
            legend_handles = bars
            legend_labels = [LEGEND_NAMES.get(method, method) for method in methods]

        ymax = max(values) if values else 0.0
        ax.set_title(title, pad=7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=18, ha='right')
        ax.grid(axis='y', linestyle='--', linewidth=0.75, alpha=0.30)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#4a4a4a')
        ax.spines['bottom'].set_color('#4a4a4a')
        ax.spines['left'].set_linewidth(0.9)
        ax.spines['bottom'].set_linewidth(0.9)
        ax.tick_params(axis='both', width=0.85, length=3)
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

        if metric == 'proof_bundle_bytes':
            ax.yaxis.set_major_formatter(FuncFormatter(format_bytes))

        offset = max(ymax * 0.025, 0.005)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                text_fmt.format(value),
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=0,
            )

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.45,
        borderpad=0.4,
    )

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
