"""
Evaluate calibration of TabPFN quantile predictions.

This script loads calibration data from pickle files and evaluates:
1. Coverage: Does the Xth percentile prediction contain X% of true values?
2. Per-fold variance: How stable is calibration across folds?
3. Per-county calibration: Which counties have better/worse calibration?
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_files(results_dir: Path) -> List[Dict]:
    """
    Load all calibration pickle files from results directory.

    Args:
        results_dir: Directory containing calibration pickle files

    Returns:
        List of calibration data dictionaries
    """
    calibration_files = list(results_dir.glob("*_calibration.pkl"))

    if not calibration_files:
        raise ValueError(f"No calibration files found in {results_dir}")

    logger.info(f"Found {len(calibration_files)} calibration files")

    all_data = []
    for cal_file in calibration_files:
        with open(cal_file, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)

    return all_data


def compute_coverage(y_true: np.ndarray, y_pred_quantile: np.ndarray) -> float:
    """
    Compute coverage: fraction of true values below predicted quantile.

    Args:
        y_true: True values
        y_pred_quantile: Predicted quantile values

    Returns:
        Coverage (fraction between 0 and 1)
    """
    return np.mean(y_true <= y_pred_quantile)


def evaluate_calibration_aggregate(all_data: List[Dict]) -> pd.DataFrame:
    """
    Evaluate calibration aggregated across all folds and counties.

    Args:
        all_data: List of calibration data dicts

    Returns:
        DataFrame with coverage results per quantile
    """
    logger.info("Computing aggregate calibration...")

    # Get quantiles from first file
    quantiles = all_data[0]['quantiles']

    # Aggregate all predictions and true values
    all_y_true = []
    all_quantile_preds = {q: [] for q in quantiles}

    for data in all_data:
        # Handle both within-county (folds) and cross-county (iterations)
        fold_key = 'folds' if 'folds' in data else 'iterations'

        for fold_data in data[fold_key]:
            all_y_true.append(fold_data['y_true'])

            for q in quantiles:
                all_quantile_preds[q].append(fold_data['quantile_predictions'][q])

    # Concatenate all arrays
    all_y_true = np.concatenate(all_y_true)
    for q in quantiles:
        all_quantile_preds[q] = np.concatenate(all_quantile_preds[q])

    # Compute coverage for each quantile
    results = []
    for q in quantiles:
        coverage = compute_coverage(all_y_true, all_quantile_preds[q])

        results.append({
            'quantile': q,
            'expected_coverage': q,
            'observed_coverage': coverage,
            'absolute_error': abs(coverage - q),
            'n_predictions': len(all_y_true)
        })

    results_df = pd.DataFrame(results)

    # Add summary statistics
    mean_abs_error = results_df['absolute_error'].mean()
    max_abs_error = results_df['absolute_error'].max()

    logger.info(f"Mean absolute calibration error: {mean_abs_error:.4f}")
    logger.info(f"Max absolute calibration error: {max_abs_error:.4f}")

    return results_df


def evaluate_calibration_per_fold(all_data: List[Dict]) -> pd.DataFrame:
    """
    Evaluate calibration per fold to assess variance.

    Args:
        all_data: List of calibration data dicts

    Returns:
        DataFrame with per-fold coverage statistics
    """
    logger.info("Computing per-fold calibration statistics...")

    quantiles = all_data[0]['quantiles']

    # Compute coverage for each fold separately
    fold_results = []

    for data in all_data:
        fips = data.get('fips') or data.get('target_fips')
        fold_key = 'folds' if 'folds' in data else 'iterations'

        for fold_data in data[fold_key]:
            fold_num = fold_data.get('fold', fold_data.get('iteration'))
            y_true = fold_data['y_true']

            for q in quantiles:
                y_pred_q = fold_data['quantile_predictions'][q]
                coverage = compute_coverage(y_true, y_pred_q)

                fold_results.append({
                    'fips': fips,
                    'fold': fold_num,
                    'quantile': q,
                    'coverage': coverage,
                    'error': abs(coverage - q)
                })

    fold_results_df = pd.DataFrame(fold_results)

    # Compute statistics per quantile
    summary = fold_results_df.groupby('quantile').agg({
        'coverage': ['mean', 'std', 'min', 'max'],
        'error': ['mean', 'std', 'max']
    }).round(4)

    logger.info("Per-quantile variance across folds:")
    logger.info(f"\n{summary}")

    return fold_results_df


def evaluate_calibration_per_county(all_data: List[Dict]) -> pd.DataFrame:
    """
    Evaluate calibration per county.

    Args:
        all_data: List of calibration data dicts

    Returns:
        DataFrame with per-county calibration statistics
    """
    logger.info("Computing per-county calibration...")

    quantiles = all_data[0]['quantiles']

    # Aggregate by county
    county_results = []

    for data in all_data:
        fips = data.get('fips') or data.get('target_fips')
        fold_key = 'folds' if 'folds' in data else 'iterations'

        # Aggregate all folds for this county
        county_y_true = []
        county_quantile_preds = {q: [] for q in quantiles}

        for fold_data in data[fold_key]:
            county_y_true.append(fold_data['y_true'])
            for q in quantiles:
                county_quantile_preds[q].append(fold_data['quantile_predictions'][q])

        # Concatenate
        county_y_true = np.concatenate(county_y_true)
        for q in quantiles:
            county_quantile_preds[q] = np.concatenate(county_quantile_preds[q])

        # Compute coverage for each quantile
        for q in quantiles:
            coverage = compute_coverage(county_y_true, county_quantile_preds[q])

            county_results.append({
                'fips': fips,
                'quantile': q,
                'coverage': coverage,
                'error': abs(coverage - q),
                'n_predictions': len(county_y_true)
            })

    county_results_df = pd.DataFrame(county_results)

    # Compute mean error per county
    county_summary = county_results_df.groupby('fips')['error'].mean().sort_values(ascending=False)

    logger.info(f"Counties with worst calibration (top 5):")
    logger.info(f"\n{county_summary.head()}")

    return county_results_df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate calibration of TabPFN quantile predictions'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing calibration pickle files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save analysis results'
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TabPFN Calibration Evaluation")
    logger.info("=" * 60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load data
    all_data = load_calibration_files(results_dir)

    # 1. Aggregate calibration (main result)
    logger.info("")
    logger.info("=" * 60)
    logger.info("1. AGGREGATE CALIBRATION")
    logger.info("=" * 60)
    aggregate_df = evaluate_calibration_aggregate(all_data)

    aggregate_file = output_dir / "calibration_aggregate.csv"
    aggregate_df.to_csv(aggregate_file, index=False)
    logger.info(f"\nAggregate results saved to: {aggregate_file}")
    logger.info(f"\n{aggregate_df.to_string(index=False)}")

    # 2. Per-fold variance
    logger.info("")
    logger.info("=" * 60)
    logger.info("2. PER-FOLD CALIBRATION VARIANCE")
    logger.info("=" * 60)
    fold_df = evaluate_calibration_per_fold(all_data)

    fold_file = output_dir / "calibration_per_fold.csv"
    fold_df.to_csv(fold_file, index=False)
    logger.info(f"\nPer-fold results saved to: {fold_file}")

    # 3. Per-county calibration
    logger.info("")
    logger.info("=" * 60)
    logger.info("3. PER-COUNTY CALIBRATION")
    logger.info("=" * 60)
    county_df = evaluate_calibration_per_county(all_data)

    county_file = output_dir / "calibration_per_county.csv"
    county_df.to_csv(county_file, index=False)
    logger.info(f"\nPer-county results saved to: {county_file}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
