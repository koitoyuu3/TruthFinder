#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    ('proof_size', 'Proof Bundle Bytes', '{:,.0f}'),
    ('verify_memory', 'Peak Verify RSS (MB)', '{:.3f}'),
    ('quality', 'Quality (%)', '{:.1f}'),
]
WEIGHT_SWEEP_CONFIGS = [
    {
        'title': 'Fixed: w_proof = 0.1',
        'fixed_key': 'proof',
        'fixed_weight': 0.1,
        'x_key': 'mem',
        'residual_key': 'qual',
    },
    {
        'title': 'Fixed: w_mem = 0.1',
        'fixed_key': 'mem',
        'fixed_weight': 0.1,
        'x_key': 'proof',
        'residual_key': 'qual',
    },
    {
        'title': 'Fixed: w_qual = 0.1',
        'fixed_key': 'qual',
        'fixed_weight': 0.1,
        'x_key': 'proof',
        'residual_key': 'mem',
    },
]
WEIGHT_LABELS = {
    'proof': 'w_proof',
    'mem': 'w_mem',
    'qual': 'w_qual',
}
WEIGHT_SWEEP_X = np.linspace(0.0, 0.9, 50)
WEIGHT_SWEEP_TICKS = np.linspace(0.0, 0.9, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot proof scalability charts from proof_可扩展性.txt.')
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
    required = {'method', 'proof_size', 'verify_memory', 'quality'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df


def format_proof_size(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return f'{value/1000:.0f}k'
    return f'{value:.0f}'


def _normalize_metric(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors='coerce')
    valid_series = numeric_series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid_series.empty:
        return result
    minimum = float(valid_series.min())
    maximum = float(valid_series.max())
    if np.isclose(minimum, maximum, rtol=1e-12, atol=1e-12):
        result.loc[valid_series.index] = 100.0
        return result
    if higher_is_better:
        result.loc[valid_series.index] = 100.0 * (valid_series - minimum) / (maximum - minimum)
    else:
        result.loc[valid_series.index] = 100.0 * (maximum - valid_series) / (maximum - minimum)
    return result


def build_weight_sweep_df(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df['proof_norm'] = _normalize_metric(plot_df['proof_size'], higher_is_better=False)
    plot_df['mem_norm'] = _normalize_metric(plot_df['verify_memory'], higher_is_better=False)
    plot_df['qual_norm'] = pd.to_numeric(plot_df['quality'], errors='coerce')
    return plot_df.reset_index(drop=True)


def compute_weight_sweep_curve(row: pd.Series, config: dict[str, object]) -> np.ndarray:
    score_map = {
        'proof': float(row['proof_norm']),
        'mem': float(row['mem_norm']),
        'qual': float(row['qual_norm']),
    }
    fixed_key = str(config['fixed_key'])
    x_key = str(config['x_key'])
    residual_key = str(config['residual_key'])
    fixed_weight = float(config['fixed_weight'])
    return (
        fixed_weight * score_map[fixed_key]
        + WEIGHT_SWEEP_X * score_map[x_key]
        + (0.9 - WEIGHT_SWEEP_X) * score_map[residual_key]
    )


def compute_shared_y_limits(plot_df: pd.DataFrame) -> tuple[float, float]:
    all_curves: list[np.ndarray] = []
    for config in WEIGHT_SWEEP_CONFIGS:
        for _, row in plot_df.iterrows():
            all_curves.append(compute_weight_sweep_curve(row, config))
    values = np.concatenate(all_curves) if all_curves else np.array([0.0, 100.0], dtype=float)
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    span = ymax - ymin
    pad = max(span * 0.12, 4.0)
    return max(0.0, ymin - pad * 0.35), ymax + pad


def add_dual_weight_axis(ax: plt.Axes, x_key: str, residual_key: str) -> None:
    ax.set_xlim(0.0, 0.9)
    ax.set_xticks(WEIGHT_SWEEP_TICKS)
    ax.set_xticklabels([f'{0.9 - tick:.2f}' for tick in WEIGHT_SWEEP_TICKS])
    ax.set_xlabel(f"{WEIGHT_LABELS[residual_key]} = 0.9 - {WEIGHT_LABELS[x_key]}", labelpad=8)

    top_ax = ax.secondary_xaxis('top', functions=(lambda value: value, lambda value: value))
    top_ax.set_xticks(WEIGHT_SWEEP_TICKS)
    top_ax.set_xticklabels([f'{tick:.2f}' for tick in WEIGHT_SWEEP_TICKS])
    top_ax.set_xlabel(WEIGHT_LABELS[x_key], labelpad=8)
    top_ax.tick_params(axis='x', pad=4, labelsize=8.5)


def add_equal_split_annotation(ax: plt.Axes, split_values: list[float]) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    max_split = max(split_values) if split_values else y_max
    y_pos = min(y_max - y_span * 0.07, max_split + y_span * 0.05)

    ax.annotate(
        'equal split',
        xy=(0.47, y_pos),
        xytext=(0, 0),
        textcoords='offset points',
        ha='left',
        va='bottom',
        fontsize=8.3,
        color='#666666',
        bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.95},
        zorder=6,
    )


def plot_bar_panel(
    ax: plt.Axes,
    methods: list[str],
    x_positions: list[int],
    x_labels: list[str],
    colors: list[str],
    values: list[float],
    title: str,
    text_fmt: str,
    metric: str,
) -> None:
    bars = ax.bar(x_positions, values, color=colors, width=0.68, edgecolor='none', zorder=3)
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

    if metric == 'proof_size':
        ax.yaxis.set_major_formatter(FuncFormatter(format_proof_size))

    offset = max(ymax * 0.025, 0.005)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            text_fmt.format(value),
            ha='center',
            va='bottom',
            fontsize=8,
        )


def plot_weight_sweep_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    config: dict[str, object],
    *,
    y_limits: tuple[float, float],
    show_ylabel: bool,
    add_legend: bool,
) -> None:
    handles = []
    labels = []
    split_values = []
    for _, row in plot_df.iterrows():
        method = str(row['method'])
        color = COLOR_MAP.get(method, '#4c4c4c')
        curve = compute_weight_sweep_curve(row, config)
        line, = ax.plot(WEIGHT_SWEEP_X, curve, color=color, linewidth=2.2, alpha=0.98)
        handles.append(line)
        labels.append(LEGEND_NAMES.get(method, method))
        split_values.append(float(np.interp(0.45, WEIGHT_SWEEP_X, curve)))

    ax.axvline(0.45, linestyle='--', linewidth=1.0, color='#8c8c8c', alpha=0.9)
    ax.set_title(str(config['title']), pad=8)
    add_dual_weight_axis(ax, str(config['x_key']), str(config['residual_key']))
    ax.grid(axis='y', linestyle='--', linewidth=0.75, alpha=0.28)
    ax.grid(axis='x', linestyle=':', linewidth=0.7, alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4a4a4a')
    ax.spines['bottom'].set_color('#4a4a4a')
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(axis='both', width=0.85, length=3)
    ax.set_ylim(*y_limits)
    add_equal_split_annotation(ax, split_values)
    if show_ylabel:
        ax.set_ylabel('Total Score')
    else:
        ax.tick_params(axis='y', labelleft=False)

    if add_legend:
        ax.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title='Method',
            fontsize=8.5,
            title_fontsize=9,
        )


def plot_panels(df: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    methods = df['method'].astype(str).tolist()
    x_labels = [DISPLAY_NAMES.get(method, method) for method in methods]
    colors = [COLOR_MAP.get(method, '#4c4c4c') for method in methods]
    x_positions = list(range(len(methods)))

    fig = plt.figure(figsize=(16.6, 8.2), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.12])
    top_axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    bottom_axes = []
    for index in range(3):
        shared_axis = bottom_axes[0] if bottom_axes else None
        bottom_axes.append(fig.add_subplot(gs[1, index], sharey=shared_axis))

    for ax, (metric, title, text_fmt) in zip(top_axes, METRICS):
        values = pd.to_numeric(df[metric], errors='coerce').astype(float).tolist()
        plot_bar_panel(ax, methods, x_positions, x_labels, colors, values, title, text_fmt, metric)

    sweep_df = build_weight_sweep_df(df)
    y_min, y_max = compute_shared_y_limits(sweep_df)
    for index, (axis, config) in enumerate(zip(bottom_axes, WEIGHT_SWEEP_CONFIGS)):
        plot_weight_sweep_panel(
            axis,
            sweep_df,
            config,
            y_limits=(y_min, y_max),
            show_ylabel=index == 0,
            add_legend=index == len(bottom_axes) - 1,
        )

    fig.text(
        0.5,
        0.01,
        'Top row: proof size / verify RSS / quality | Bottom row: weight sweeps with one metric fixed at 0.1',
        ha='center',
        fontsize=9,
        color='#666666',
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
