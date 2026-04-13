"""
Aggregation utilities for experimental results.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregate experimental results at different levels."""

    @staticmethod
    def aggregate_folds(
        results_df: pd.DataFrame,
        groupby_cols: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate fold-level results to higher level.

        Args:
            results_df: DataFrame with fold-level results
            groupby_cols: Columns to group by (e.g., ['county', 'model'])

        Returns:
            Aggregated DataFrame with mean and std
        """
        # Identify metric columns (numeric columns that aren't in groupby_cols)
        metric_cols = [
            col for col in results_df.select_dtypes(include=[np.number]).columns
            if col not in groupby_cols
        ]

        # Compute mean and std for each metric
        agg_funcs = {}
        for col in metric_cols:
            agg_funcs[col] = ['mean', 'std', 'count']

        aggregated = results_df.groupby(groupby_cols).agg(agg_funcs)

        # Flatten column names
        aggregated.columns = [
            f"{col}_{stat}" if stat != 'mean' else col
            for col, stat in aggregated.columns
        ]

        aggregated = aggregated.reset_index()

        return aggregated

    @staticmethod
    def aggregate_by_bin(
        results_df: pd.DataFrame,
        bin_col: str = 'bin_name',
        model_col: str = 'model'
    ) -> pd.DataFrame:
        """
        Aggregate results by county size bin.

        Args:
            results_df: DataFrame with county-level results
            bin_col: Column name for bin
            model_col: Column name for model

        Returns:
            Aggregated DataFrame by bin and model
        """
        return ResultsAggregator.aggregate_folds(
            results_df,
            groupby_cols=[bin_col, model_col]
        )

    @staticmethod
    def aggregate_overall(
        results_df: pd.DataFrame,
        model_col: str = 'model'
    ) -> pd.DataFrame:
        """
        Aggregate results across all counties.

        Args:
            results_df: DataFrame with county-level results
            model_col: Column name for model

        Returns:
            Overall aggregated DataFrame
        """
        return ResultsAggregator.aggregate_folds(
            results_df,
            groupby_cols=[model_col]
        )

    @staticmethod
    def create_comparison_table(
        results_df: pd.DataFrame,
        baseline_model: str = 'xgboost',
        comparison_model: str = 'tabpfn',
        metric: str = 'r2',
        groupby_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create a comparison table showing performance differences.

        Args:
            results_df: Aggregated results DataFrame
            baseline_model: Name of baseline model
            comparison_model: Name of comparison model
            metric: Metric to compare
            groupby_cols: Columns to group by (e.g., ['bin_name'])

        Returns:
            Comparison DataFrame
        """
        if groupby_cols is None:
            groupby_cols = []

        # Filter for the two models
        baseline_df = results_df[results_df['model'] == baseline_model].copy()
        comparison_df = results_df[results_df['model'] == comparison_model].copy()

        # Merge on groupby columns
        if groupby_cols:
            merged = pd.merge(
                baseline_df,
                comparison_df,
                on=groupby_cols,
                suffixes=('_baseline', '_comparison')
            )
        else:
            # No grouping, just direct comparison
            if len(baseline_df) != 1 or len(comparison_df) != 1:
                raise ValueError("Expected single row for each model when no grouping")

            merged = pd.DataFrame({
                f'{metric}_baseline': [baseline_df[metric].iloc[0]],
                f'{metric}_comparison': [comparison_df[metric].iloc[0]],
            })

        # Compute difference
        merged[f'{metric}_diff'] = (
            merged[f'{metric}_comparison'] - merged[f'{metric}_baseline']
        )

        # Compute relative improvement (in percentage)
        higher_is_better = metric in ['r2']  # Add more if needed
        if higher_is_better:
            merged[f'{metric}_rel_improvement_pct'] = (
                merged[f'{metric}_diff'] / merged[f'{metric}_baseline'].abs() * 100
            )
        else:
            merged[f'{metric}_rel_improvement_pct'] = (
                -merged[f'{metric}_diff'] / merged[f'{metric}_baseline'].abs() * 100
            )

        return merged

    @staticmethod
    def save_aggregated_results(
        fold_results: pd.DataFrame,
        output_dir: str,
        experiment_name: str = 'within_county'
    ):
        """
        Save aggregated results at multiple levels.

        Args:
            fold_results: Fold-level results DataFrame
            output_dir: Directory to save results
            experiment_name: Name of experiment
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save fold-level results
        fold_file = output_path / f"{experiment_name}_fold_results.csv"
        fold_results.to_csv(fold_file, index=False)
        logger.info(f"Saved fold-level results to {fold_file}")

        # Aggregate by county
        if 'fips' in fold_results.columns:
            county_agg = ResultsAggregator.aggregate_folds(
                fold_results,
                groupby_cols=['fips', 'model']
            )
            county_file = output_path / f"{experiment_name}_county_aggregated.csv"
            county_agg.to_csv(county_file, index=False)
            logger.info(f"Saved county-aggregated results to {county_file}")

            # Aggregate by bin
            if 'bin_name' in fold_results.columns:
                bin_agg = ResultsAggregator.aggregate_by_bin(fold_results)
                bin_file = output_path / f"{experiment_name}_bin_aggregated.csv"
                bin_agg.to_csv(bin_file, index=False)
                logger.info(f"Saved bin-aggregated results to {bin_file}")

            # Aggregate overall
            overall_agg = ResultsAggregator.aggregate_overall(fold_results)
            overall_file = output_path / f"{experiment_name}_overall_aggregated.csv"
            overall_agg.to_csv(overall_file, index=False)
            logger.info(f"Saved overall-aggregated results to {overall_file}")
