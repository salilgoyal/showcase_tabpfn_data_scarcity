#!/usr/bin/env python3
"""
Recompute per-county MAPE values from saved predictions.

This script loads predictions parquet files and recomputes per-county metrics
(especially MAPE) that were missing due to a bug in the original experiment code.

Usage:
    python experiments/scripts/recompute_per_county_mape.py \
        --config experiments/configs/finetuning/large_scale.yaml
"""

from __future__ import print_function, division
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import CleanedDataLoader
from src.data.split_strategies import (
    load_test_set_result,
    load_train_set_result,
)
from src.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load experiment configuration and substitute template variables."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute template variables (e.g., {train_version})
    train_version = config.get('experiment', {}).get('train_version', 'train_v6')

    # Recursively substitute in strings
    def substitute_vars(obj):
        if isinstance(obj, dict):
            return {k: substitute_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_vars(item) for item in obj]
        elif isinstance(obj, str):
            return obj.format(train_version=train_version)
        else:
            return obj

    config = substitute_vars(config)
    return config


def recompute_per_county_mape(config_path):
    """
    Recompute per-county MAPE values from saved predictions.

    Args:
        config_path: Path to experiment config YAML
    """
    logger.info("=" * 80)
    logger.info("RECOMPUTING PER-COUNTY MAPE VALUES")
    logger.info("=" * 80)

    # Load config
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Get results directory
    results_dir = Path(config['output']['results_dir'])
    results_file = results_dir / 'results.csv'
    output_file = results_dir / 'mape_values_filled_in.csv'

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Loading results from: {results_file}")

    # Load existing results
    df_results = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df_results)} rows from results.csv")

    # Check if we have per_county rows
    per_county_rows = df_results[df_results['result_type'] == 'per_county']
    logger.info(f"Found {len(per_county_rows)} per-county rows")

    if len(per_county_rows) == 0:
        logger.warning("No per-county rows found. Nothing to recompute.")
        return

    # Determine if log-transformed
    # Try to infer from data metadata
    data_loader = CleanedDataLoader(
        cleaned_data_path=config['data']['cleaned_data_path'],
        target_column=config['data']['target_column'],
        phase2_config={}
    )
    log_transformed = data_loader.is_target_log_transformed()
    logger.info(f"Target log-transformed: {log_transformed}")

    # Load test set to get county mappings
    logger.info("Loading test set info...")
    if 'splits' in config:
        # Pre-generated splits
        test_split_dir = config['splits']['test_set_dir']
        train_split_dir = config['splits']['train_set_dir']
        test_result = load_test_set_result(test_split_dir)
        train_result = load_train_set_result(train_split_dir)

        # Load only the test data indices we need
        logger.info("Loading test data for county mapping...")
        test_indices = test_result.test_indices

        # Check if debug mode
        debug_max_rows = None
        if 'debug' in config and 'sample_size' in config['debug']:
            debug_max_rows = config['debug']['sample_size']

        # Note: We'll load the full FIPS array later for test_index mapping
        if debug_max_rows is not None:
            # Debug mode: just read first N rows
            df = data_loader.load_data_by_indices(np.array([]), max_rows=debug_max_rows)
            test_indices = np.arange(min(debug_max_rows // 2, len(df)))
            test_fips = df.iloc[test_indices]['fips'].values
        else:
            # For fallback only (if test_index column missing in predictions)
            import pyarrow.parquet as pq
            data_file = data_loader.data_file
            logger.info(f"Loading 'fips' column for test indices (fallback)...")
            parquet_file = pq.ParquetFile(data_file)
            test_fips = parquet_file.read(columns=['fips']).to_pandas()['fips'].iloc[test_result.test_indices].values
            logger.info(f"Loaded fips for {len(test_fips)} test samples")

    else:
        raise NotImplementedError("Only pre-generated splits mode is supported")

    logger.info(f"Test set has {len(test_result.test_counties)} counties")

    # Build the unique_indices mapping for finetuning predictions
    # Finetuning code loads a subset (test + train_pool + train), and test_index is iloc within that subset
    # We need to map these iloc positions back to original data.parquet row indices
    logger.info("Building index mapping for finetuning predictions...")
    unique_indices = np.unique(np.concatenate([
        test_result.test_indices,
        test_result.train_pool_indices,
        train_result.train_indices
    ]))
    logger.info(f"unique_indices: {len(unique_indices)} rows (range {unique_indices.min()}-{unique_indices.max()})")

    # Load full FIPS array from data.parquet (needed to map data row index -> FIPS)
    import pyarrow.parquet as pq
    data_file = data_loader.data_file
    logger.info("Loading FIPS column from data.parquet for FIPS mapping...")
    parquet_file = pq.ParquetFile(data_file)
    fips_all = parquet_file.read(columns=['fips']).to_pandas()['fips'].values
    logger.info(f"Loaded {len(fips_all)} FIPS values")

    # Get list of models from per_county rows
    models = per_county_rows['model'].unique()
    logger.info(f"Models to process: {models.tolist()}")

    # Recompute metrics for each model
    for model_name in models:
        logger.info(f"\nProcessing {model_name}...")

        # Load predictions
        pred_file = results_dir / f'predictions_{model_name}.parquet'
        if not pred_file.exists():
            logger.warning(f"  Predictions file not found: {pred_file}")
            logger.warning(f"  Skipping {model_name}")
            continue

        pred_df = pd.read_parquet(pred_file)
        logger.info(f"  Loaded {len(pred_df)} predictions")

        # Map test_index to FIPS for correct county grouping
        # Finetuning predictions: test_index is iloc position in subset DataFrame
        # Need to convert: test_index -> data.parquet row index -> FIPS
        if 'test_index' in pred_df.columns:
            # Convert test_index (iloc in subset) to data.parquet row index
            data_row_indices = unique_indices[pred_df['test_index'].values]
            # Map data row index to FIPS
            pred_df['fips'] = fips_all[data_row_indices]
            logger.info(f"  Mapped test_index -> data row index -> FIPS for {len(pred_df)} predictions")
        else:
            logger.warning(f"  No test_index column found - assuming sequential order")
            pred_df['fips'] = test_fips

        # Compute MAPE per county
        for fips in test_result.test_counties:
            # Filter predictions for this county using FIPS
            county_mask = pred_df['fips'] == fips

            if not np.any(county_mask):
                continue

            # Get predictions for this county
            y_true_county = pred_df.loc[county_mask, 'y_true'].values
            y_pred_county = pred_df.loc[county_mask, 'y_pred'].values

            # Compute metrics (including MAPE)
            metrics = compute_metrics(
                y_true=y_true_county,
                y_pred=y_pred_county,
                log_transformed=log_transformed
            )

            # Update the results dataframe
            mask = (
                (df_results['model'] == model_name) &
                (df_results['result_type'] == 'per_county') &
                (df_results['test_fips'] == fips)
            )

            if mask.sum() == 0:
                logger.warning(f"  No matching row for {model_name}, county {fips}")
                continue

            if mask.sum() > 1:
                logger.warning(f"  Multiple matching rows for {model_name}, county {fips}")

            # Update MAPE value
            df_results.loc[mask, 'mape'] = metrics['mape']

            logger.debug(f"  County {fips}: MAPE = {metrics['mape']:.2f}")

        logger.info(f"  Completed {model_name}")

    # Save updated results
    logger.info(f"\nSaving updated results to: {output_file}")
    df_results.to_csv(output_file, index=False)

    # Summary
    per_county_rows_updated = df_results[df_results['result_type'] == 'per_county']
    n_with_mape = per_county_rows_updated['mape'].notna().sum()
    n_total = len(per_county_rows_updated)

    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Updated {n_with_mape}/{n_total} per-county rows with MAPE values")
    logger.info(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Recompute per-county MAPE values from saved predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )

    args = parser.parse_args()

    recompute_per_county_mape(args.config)


if __name__ == '__main__':
    main()
