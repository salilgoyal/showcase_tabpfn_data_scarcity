"""
Generate baseline results CSV for a test set using per-county adjustment ratios.

This pre-computes baseline predictions and metrics once in results.csv format,
so they can be loaded and appended to experiment results without recomputing.

The adjustment ratio is computed PER-COUNTY from ALL available data (test + train_pool)
for maximum stability, since the baseline represents ground truth (county assessor values)
rather than a predictive model. Each county's assessed values are adjusted by its own
median ratio of SALE_AMOUNT / CALCULATED_TOTAL_VALUE.

Usage:
    python experiments/scripts/generate_baseline_results.py \
        --test_split_dir /scratch/.../test_v1/ \
        --output_file /scratch/.../test_v1/baseline_results.csv

Output:
    - baseline_results.csv: overall + per-county results with metrics
    - baseline_predictions.parquet: individual predictions with raw assessment values
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models import BaselineModel
from src.evaluation import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata(split_dir: Path) -> dict:
    """Load metadata from split directory."""
    metadata_file = split_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Also load source breakdown if available (for train sets)
    source_file = split_dir / "source_breakdown.json"
    if source_file.exists():
        with open(source_file, 'r') as f:
            source_breakdown = json.load(f)
            metadata['n_train_source_test_history'] = source_breakdown.get('test_counties_historical', 0)
            metadata['n_train_source_external'] = source_breakdown.get('external_counties', 0)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute baseline results for a test/train split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--test_split_dir',
        type=str,
        required=True,
        help='Directory containing test set files'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output CSV file path (default: <test_split_dir>/baseline_results.csv)'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='baseline_precomputed',
        help='Experiment name to include in results'
    )

    parser.add_argument(
        '--experiment_description',
        type=str,
        default='Pre-computed baseline results',
        help='Experiment description to include in results'
    )

    args = parser.parse_args()

    # Default output file
    if args.output_file is None:
        args.output_file = str(Path(args.test_split_dir) / "baseline_results.csv")

    test_split_dir = Path(args.test_split_dir)

    logger.info("=" * 80)
    logger.info("GENERATING BASELINE RESULTS (PER-COUNTY RATIOS)")
    logger.info("=" * 80)
    logger.info(f"Test split: {test_split_dir}")
    logger.info(f"Output: {args.output_file}")
    logger.info("")

    # Load metadata
    test_metadata = load_metadata(test_split_dir)

    # Load saved test data
    logger.info("Loading baseline and sale values for test set...")
    test_baseline = np.load(test_split_dir / "test_baseline_values.npy")
    test_sales = np.load(test_split_dir / "test_sale_amounts.npy")
    test_indices = np.load(test_split_dir / "test_indices.npy")

    # Also load train_pool data for more stable ratio estimation
    logger.info("Loading baseline and sale values for train pool...")
    train_pool_baseline = np.load(test_split_dir / "train_pool_baseline_values.npy")
    train_pool_sales = np.load(test_split_dir / "train_pool_sale_amounts.npy")
    train_pool_indices = np.load(test_split_dir / "train_pool_indices.npy")

    # Combine test + train_pool for ratio computation (baseline is ground truth, not a model)
    all_baseline = np.concatenate([test_baseline, train_pool_baseline])
    all_sales = np.concatenate([test_sales, train_pool_sales])
    all_indices = np.concatenate([test_indices, train_pool_indices])

    logger.info(f"Test samples: {len(test_baseline):,}")
    logger.info(f"Train pool samples: {len(train_pool_baseline):,}")
    logger.info(f"Total samples for ratio computation: {len(all_baseline):,}")

    # Detect log transformation
    log_transformed = (all_sales.mean() < 100)  # log(sale) is typically 10-15
    logger.info(f"Target log-transformed: {log_transformed}")

    # Get original-scale sale amounts for ratio computation
    if log_transformed:
        all_sales_original = np.exp(all_sales)
        test_sales_original = np.exp(test_sales)
    else:
        all_sales_original = all_sales
        test_sales_original = test_sales

    # Load FIPS for all samples (needed for per-county ratios)
    logger.info("Loading FIPS for all samples...")
    import pyarrow.parquet as pq
    data_path = test_split_dir.parent / "data.parquet"
    table = pq.read_table(data_path, columns=['fips'])
    fips_all_data = table.column('fips').to_numpy()
    fips_all = fips_all_data[all_indices]
    fips_test = fips_all_data[test_indices]
    unique_fips = np.unique(fips_all)
    logger.info(f"All samples span {len(unique_fips)} counties")

    # Compute global adjustment ratio (as fallback)
    global_valid = (all_baseline > 0) & (all_sales_original > 0) & np.isfinite(all_baseline) & np.isfinite(all_sales_original)
    if global_valid.sum() == 0:
        logger.error("No valid baseline/sales pairs found!")
        return
    global_ratio = float(np.median(all_sales_original[global_valid] / all_baseline[global_valid]))
    logger.info(f"Global adjustment ratio (fallback): {global_ratio:.4f}")

    # Compute per-county adjustment ratios using ALL available data (test + train_pool)
    logger.info("Computing per-county adjustment ratios from all available data...")
    per_county_ratios = {}
    per_county_n_samples = {}
    for fips in unique_fips:
        county_mask = fips_all == fips
        county_baseline = all_baseline[county_mask]
        county_sales = all_sales_original[county_mask]

        valid = (county_baseline > 0) & (county_sales > 0) & np.isfinite(county_baseline) & np.isfinite(county_sales)
        if valid.sum() > 0:
            county_ratios = county_sales[valid] / county_baseline[valid]
            per_county_ratios[int(fips)] = float(np.median(county_ratios))
            per_county_n_samples[int(fips)] = valid.sum()
        else:
            per_county_ratios[int(fips)] = global_ratio
            per_county_n_samples[int(fips)] = 0
            logger.warning(f"  County {fips}: no valid pairs, using global ratio")

    ratio_values = list(per_county_ratios.values())
    logger.info(f"Per-county ratios: min={min(ratio_values):.4f}, median={np.median(ratio_values):.4f}, max={max(ratio_values):.4f}")
    logger.info(f"Sample sizes for ratio computation: min={min(per_county_n_samples.values())}, median={int(np.median(list(per_county_n_samples.values())))}, max={max(per_county_n_samples.values())}")

    # Generate predictions using per-county ratios
    logger.info("Generating per-county baseline predictions...")
    fit_start = time.time()
    y_pred = np.full(len(test_baseline), np.nan)
    per_sample_ratio = np.full(len(test_baseline), np.nan)

    for fips in unique_fips:
        county_mask = fips_test == fips
        ratio = per_county_ratios[int(fips)]
        per_sample_ratio[county_mask] = ratio
        county_pred = test_baseline[county_mask] * ratio
        if log_transformed:
            county_pred = np.where(county_pred > 0, np.log(county_pred), np.nan)
        y_pred[county_mask] = county_pred

    fit_time = time.time() - fit_start
    pred_time = 0.0  # predictions computed inline with fitting

    # Load county info for size_bucket metadata
    logger.info("Loading county info for metadata...")
    county_info_file = test_split_dir / "county_info.json"
    county_info = {}
    if county_info_file.exists():
        with open(county_info_file, 'r') as f:
            county_info = json.load(f)
            county_info = {int(k): v for k, v in county_info.items()}

    # Create size_bucket array for each test sample
    test_size_buckets = np.array([
        county_info.get(int(fips), {}).get('size_bucket', 'unknown')
        for fips in fips_test
    ])

    # Save individual predictions with both raw assessment and adjusted values
    logger.info("Saving individual predictions...")
    predictions_file = Path(args.output_file).parent / "baseline_predictions.parquet"
    pred_df = pd.DataFrame({
        'test_index': test_indices,
        'fips': fips_test,
        'size_bucket': test_size_buckets,
        'y_true': test_sales,
        'y_pred': y_pred,
        'baseline_raw': test_baseline,
        'adjustment_ratio': per_sample_ratio
    })
    pred_df.to_parquet(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file} ({len(pred_df):,} records)")

    # Compute overall metrics (filter out NaN predictions)
    logger.info("Computing metrics...")
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(test_sales)
    n_valid = valid_mask.sum()
    n_total = len(y_pred)
    if n_valid < n_total:
        logger.warning(f"Filtered out {n_total - n_valid} samples with NaN predictions ({100*(n_total-n_valid)/n_total:.2f}%)")

    metrics = compute_metrics(test_sales[valid_mask], y_pred[valid_mask], log_transformed=log_transformed)

    # Build results list
    results = []

    # Overall result
    overall_result = {
        'model': 'baseline',
        'train_size': '',  # N/A - baseline doesn't train
        'test_size': float(len(test_baseline)),
        'n_features': 0.0,
        'fit_time': fit_time,
        'pred_time': pred_time,
        'r2': metrics.get('r2'),
        'mae': metrics.get('mae'),
        'rmse': metrics.get('rmse'),
        'mape': metrics.get('mape'),
        'mse': metrics.get('mse'),
        'test_set_version': test_metadata.get('version', 'unknown'),
        'train_set_version': '',  # N/A - not specific to any train set
        'n_test_counties': test_metadata.get('n_test_counties', 0),
        'n_train_source_test_history': '',  # N/A
        'n_train_source_external': '',  # N/A
        'status': 'success',
        'experiment_name': args.experiment_name,
        'experiment_description': args.experiment_description,
        'adjustment_ratio': '',  # Per-county ratios used; see per-county rows
        'result_type': '',
        'test_fips': '',
        'size_bucket': '',
        'county_test_size': '',
        'county_train_pool_size': '',
        'county_train_used': ''
    }
    results.append(overall_result)

    # Per-county results
    logger.info("Computing per-county metrics...")

    # County info already loaded above for predictions file

    # Load test counties
    test_counties_file = test_split_dir / "test_counties.json"
    with open(test_counties_file, 'r') as f:
        test_counties = json.load(f)

    # fips_test already loaded above
    for county_fips in test_counties:
        county_mask = fips_test == county_fips
        y_pred_county = y_pred[county_mask]
        y_true_county = test_sales[county_mask]

        if len(y_pred_county) == 0:
            continue

        # Filter out NaN predictions for this county
        valid_mask_county = ~np.isnan(y_pred_county) & ~np.isnan(y_true_county)
        if valid_mask_county.sum() == 0:
            continue

        # Compute metrics for this county
        county_metrics = compute_metrics(
            y_true_county[valid_mask_county],
            y_pred_county[valid_mask_county],
            log_transformed=log_transformed
        )

        # Get county info
        info = county_info.get(county_fips, {})

        county_result = {
            'model': 'baseline',
            'train_size': '',
            'test_size': '',
            'n_features': '',
            'fit_time': '',
            'pred_time': '',
            'r2': county_metrics.get('r2'),
            'mae': county_metrics.get('mae'),
            'rmse': county_metrics.get('rmse'),
            'mape': county_metrics.get('mape'),
            'mse': county_metrics.get('mse'),
            'test_set_version': test_metadata.get('version', 'unknown'),
            'train_set_version': '',  # N/A
            'n_test_counties': '',
            'n_train_source_test_history': '',
            'n_train_source_external': '',
            'status': '',
            'experiment_name': args.experiment_name,
            'experiment_description': f'Per-county baseline (ratio from all available data)',
            'adjustment_ratio': per_county_ratios.get(county_fips, ''),
            'adjustment_ratio_n_samples': per_county_n_samples.get(county_fips, ''),
            'result_type': 'per_county',
            'test_fips': float(county_fips),
            'size_bucket': info.get('size_bucket', ''),
            'county_test_size': float(info.get('test_size', valid_mask_county.sum())),
            'county_train_pool_size': '',  # N/A
            'county_train_used': ''  # N/A
        }
        results.append(county_result)

    # Convert to DataFrame and save
    logger.info(f"Saving {len(results)} result rows to {args.output_file}...")
    results_df = pd.DataFrame(results)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE RESULTS GENERATED")
    logger.info("=" * 80)
    logger.info(f"Total results: {len(results)}")
    logger.info(f"Overall results: 1")
    logger.info(f"Per-county results: {len(results) - 1}")
    logger.info(f"Per-county ratios: {len(per_county_ratios)} counties")
    logger.info(f"Global ratio (fallback): {global_ratio:.4f}")
    logger.info(f"Overall R2: {metrics.get('r2', np.nan):.4f}")
    logger.info(f"Overall MAE: {metrics.get('mae', np.nan):.2f}")
    logger.info(f"Aggregate results: {args.output_file}")
    logger.info(f"Individual predictions: {predictions_file}")


if __name__ == "__main__":
    main()
