"""
Functions for comparing TabPFN and XGBoost performance across counties.

This module provides utilities to:
1. Load experimental results from multiple CSV files
2. Aggregate results at the county level
3. Generate statistical comparison plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Tuple


def load_all_results(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all county result CSVs from a directory into a single DataFrame.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing county_*_results.csv files

    Returns
    -------
    pd.DataFrame
        Combined dataframe with all results
    """
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob("county_*_results.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} files with {len(combined_df)} total rows")

    return combined_df


def aggregate_by_county(df: pd.DataFrame, metric: str = 'mae') -> pd.DataFrame:
    """
    Aggregate results to county level by computing mean across repetitions and folds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results dataframe from load_all_results()
    metric : str, default='mae'
        Performance metric to aggregate (e.g., 'mae', 'rmse', 'r2')

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with columns: fips, model, {metric}_mean
    """
    # Group by county (fips) and model, compute mean of the metric
    agg_df = df.groupby(['fips', 'model'])[metric].mean().reset_index()
    agg_df.columns = ['fips', 'model', f'{metric}_mean']

    # Pivot to have tabpfn and xgboost as separate columns for easier comparison
    pivot_df = agg_df.pivot(index='fips', columns='model', values=f'{metric}_mean').reset_index()
    pivot_df.columns.name = None

    return pivot_df


def plot_paired_scatter(
    agg_df: pd.DataFrame,
    metric: str = 'mae',
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create paired comparison scatter plot (county-level aggregates).

    Each point represents one county. Points below the diagonal indicate
    TabPFN performs better (lower is better for MAE/RMSE, higher for R²).

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated dataframe from aggregate_by_county()
    metric : str, default='mae'
        Metric name for axis labels
    figsize : tuple, default=(8, 8)
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine if lower is better (mae, rmse, mse) or higher is better (r2)
    lower_is_better = metric.lower() in ['mae', 'rmse', 'mse']

    # Scatter plot
    ax.scatter(agg_df['tabpfn'], agg_df['xgboost'], alpha=0.6, s=80, edgecolors='black', linewidths=0.5)

    # Diagonal reference line
    min_val = min(agg_df['tabpfn'].min(), agg_df['xgboost'].min())
    max_val = max(agg_df['tabpfn'].max(), agg_df['xgboost'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Equality')

    # Labels and title
    metric_upper = metric.upper()
    ax.set_xlabel(f'TabPFN {metric_upper}', fontsize=12)
    ax.set_ylabel(f'XGBoost {metric_upper}', fontsize=12)
    ax.set_title(f'County-Level Performance Comparison ({metric_upper})', fontsize=14, fontweight='bold')

    # Add text indicating which direction is better
    if lower_is_better:
        better_region = "Lower = Better\n(TabPFN wins below diagonal)"
    else:
        better_region = "Higher = Better\n(TabPFN wins above diagonal)"

    ax.text(0.05, 0.95, better_region, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Count wins
    if lower_is_better:
        tabpfn_wins = (agg_df['tabpfn'] < agg_df['xgboost']).sum()
    else:
        tabpfn_wins = (agg_df['tabpfn'] > agg_df['xgboost']).sum()

    xgb_wins = len(agg_df) - tabpfn_wins
    ax.text(0.05, 0.05, f'TabPFN wins: {tabpfn_wins}\nXGBoost wins: {xgb_wins}',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_bland_altman(
    agg_df: pd.DataFrame,
    metric: str = 'mae',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create Bland-Altman plot (difference vs. average).

    Shows systematic bias and whether differences vary with performance level.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated dataframe from aggregate_by_county()
    metric : str, default='mae'
        Metric name for axis labels
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate mean and difference
    mean_performance = (agg_df['tabpfn'] + agg_df['xgboost']) / 2
    difference = agg_df['tabpfn'] - agg_df['xgboost']

    # Calculate statistics
    mean_diff = difference.mean()
    std_diff = difference.std()

    # Scatter plot
    ax.scatter(mean_performance, difference, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)

    # Horizontal lines
    ax.axhline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f'Mean difference: {mean_diff:.3f}')
    ax.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', linewidth=1.5,
               label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.3f}')
    ax.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', linewidth=1.5,
               label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.3f}')
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)

    # Labels and title
    metric_upper = metric.upper()
    ax.set_xlabel(f'Mean {metric_upper} (TabPFN + XGBoost) / 2', fontsize=12)
    ax.set_ylabel(f'Difference in {metric_upper} (TabPFN - XGBoost)', fontsize=12)
    ax.set_title(f'Bland-Altman Plot: Agreement Between Models ({metric_upper})', fontsize=14, fontweight='bold')

    # Determine if lower is better
    lower_is_better = metric.lower() in ['mae', 'rmse', 'mse']
    if lower_is_better:
        interpretation = "Positive difference = TabPFN worse\nNegative difference = TabPFN better"
    else:
        interpretation = "Positive difference = TabPFN better\nNegative difference = TabPFN worse"

    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_raincloud(
    agg_df: pd.DataFrame,
    metric: str = 'mae',
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create raincloud/violin plot with paired lines connecting each county.

    Shows both overall distribution and individual county trajectories.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated dataframe from aggregate_by_county()
    metric : str, default='mae'
        Metric name for axis labels
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for seaborn
    plot_data = pd.DataFrame({
        'County': list(agg_df['fips']) * 2,
        'Model': ['TabPFN'] * len(agg_df) + ['XGBoost'] * len(agg_df),
        'Performance': list(agg_df['tabpfn']) + list(agg_df['xgboost'])
    })

    # Violin plot
    violin_parts = ax.violinplot(
        [agg_df['tabpfn'], agg_df['xgboost']],
        positions=[0, 1],
        widths=0.7,
        showmeans=True,
        showextrema=True,
        showmedians=True
    )

    # Color the violins
    colors = ['#1f77b4', '#ff7f0e']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    # Add paired lines connecting each county
    for idx, row in agg_df.iterrows():
        ax.plot([0, 1], [row['tabpfn'], row['xgboost']],
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)

    # Overlay strip plot for individual counties
    np.random.seed(42)
    jitter_strength = 0.1
    for i, (model, values) in enumerate([(0, agg_df['tabpfn']), (1, agg_df['xgboost'])]):
        x_jittered = np.random.normal(i, jitter_strength, size=len(values))
        ax.scatter(x_jittered, values, alpha=0.5, s=50, color=colors[i],
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Labels and title
    metric_upper = metric.upper()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['TabPFN', 'XGBoost'], fontsize=12)
    ax.set_ylabel(f'{metric_upper}', fontsize=12)
    ax.set_title(f'Performance Distribution with Paired County Results ({metric_upper})',
                 fontsize=14, fontweight='bold')

    # Add statistics
    tabpfn_mean = agg_df['tabpfn'].mean()
    xgb_mean = agg_df['xgboost'].mean()
    tabpfn_std = agg_df['tabpfn'].std()
    xgb_std = agg_df['xgboost'].std()

    stats_text = f'TabPFN: μ={tabpfn_mean:.3f}, σ={tabpfn_std:.3f}\nXGBoost: μ={xgb_mean:.3f}, σ={xgb_std:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_all_plots(
    results_dir: Union[str, Path],
    metric: str = 'mae',
    save_dir: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Convenience function to load data and create all three plots.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing county_*_results.csv files
    metric : str, default='mae'
        Performance metric to plot
    save_dir : str or Path, optional
        Directory to save plots. If None, plots are not saved.

    Returns
    -------
    tuple of matplotlib.figure.Figure
        (scatter_fig, bland_altman_fig, raincloud_fig)
    """
    # Load and aggregate data
    df = load_all_results(results_dir)
    agg_df = aggregate_by_county(df, metric=metric)

    # Create save paths if save_dir provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        scatter_path = save_dir / f'paired_scatter_{metric}.png'
        bland_altman_path = save_dir / f'bland_altman_{metric}.png'
        raincloud_path = save_dir / f'raincloud_{metric}.png'
    else:
        scatter_path = None
        bland_altman_path = None
        raincloud_path = None

    # Create plots
    fig1 = plot_paired_scatter(agg_df, metric=metric, save_path=scatter_path)
    fig2 = plot_bland_altman(agg_df, metric=metric, save_path=bland_altman_path)
    fig3 = plot_raincloud(agg_df, metric=metric, save_path=raincloud_path)

    return fig1, fig2, fig3
