"""
Utility functions for comparing model predictions against county baseline assessments.

This module provides utilities for loading baseline values and computing
comparison metrics between model predictions and county baseline assessments.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_baseline_data(test_split_dir: str, train_split_dir: str) -> Dict[str, np.ndarray]:
    """
    Load baseline values and adjustment ratio for a specific train/test split.

    Args:
        test_split_dir: Path to test split directory (e.g., 'test_v1/')
        train_split_dir: Path to train split directory (e.g., 'test_v1/train_v2/')

    Returns:
        Dictionary containing:
            - 'test_baseline': Baseline values for test set
            - 'test_sales': Actual sale amounts for test set
            - 'adjustment_ratio': Ratio to adjust baseline values
            - 'train_baseline': Baseline values for training set (optional, for analysis)
            - 'train_sales': Sale amounts for training set (optional, for analysis)
    """
    test_path = Path(test_split_dir)
    train_path = Path(train_split_dir)

    result = {}

    # Load test baseline values
    test_baseline_file = test_path / "test_baseline_values.npy"
    if test_baseline_file.exists():
        result['test_baseline'] = np.load(test_baseline_file)
    else:
        raise FileNotFoundError(f"Test baseline values not found at {test_baseline_file}")

    # Load test sale amounts
    test_sales_file = test_path / "test_sale_amounts.npy"
    if test_sales_file.exists():
        result['test_sales'] = np.load(test_sales_file)
    else:
        raise FileNotFoundError(f"Test sale amounts not found at {test_sales_file}")

    # Load adjustment ratio from train metadata
    metadata_file = train_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            result['adjustment_ratio'] = metadata.get('baseline_adjustment_ratio', 1.0)
    else:
        raise FileNotFoundError(f"Train metadata not found at {metadata_file}")

    # Optionally load train baseline values (for analysis)
    train_baseline_file = train_path / "train_baseline_values.npy"
    if train_baseline_file.exists():
        result['train_baseline'] = np.load(train_baseline_file)

    train_sales_file = train_path / "train_sale_amounts.npy"
    if train_sales_file.exists():
        result['train_sales'] = np.load(train_sales_file)

    return result


def get_adjusted_baseline_predictions(baseline_values: np.ndarray, adjustment_ratio: float) -> np.ndarray:
    """
    Apply adjustment ratio to baseline values.

    Args:
        baseline_values: Raw baseline values (CALCULATED_TOTAL_VALUE)
        adjustment_ratio: Adjustment ratio from training data

    Returns:
        Adjusted baseline predictions
    """
    return baseline_values * adjustment_ratio


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, MAPE, and R2
    """
    # Filter out any NaN or infinite values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}

    # Compute MAPE, filtering out zeros to avoid division by zero
    mape_mask = y_true_clean != 0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
    else:
        mape = np.nan

    return {
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'mape': mape,
        'r2': r2_score(y_true_clean, y_pred_clean)
    }


def compare_to_baseline(
    model_predictions: np.ndarray,
    test_split_dir: str,
    train_split_dir: str,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare model predictions to baseline assessments.

    Args:
        model_predictions: Your model's predictions
        test_split_dir: Path to test split directory
        train_split_dir: Path to train split directory
        verbose: Whether to print comparison results

    Returns:
        Dictionary with metrics for:
            - 'model': Model performance
            - 'baseline_raw': Raw baseline performance
            - 'baseline_adjusted': Adjusted baseline performance
    """
    # Load baseline data
    baseline_data = load_baseline_data(test_split_dir, train_split_dir)

    test_sales = baseline_data['test_sales']
    test_baseline_raw = baseline_data['test_baseline']
    adjustment_ratio = baseline_data['adjustment_ratio']

    # Get adjusted baseline
    test_baseline_adjusted = get_adjusted_baseline_predictions(test_baseline_raw, adjustment_ratio)

    # Compute metrics
    model_metrics = compute_metrics(test_sales, model_predictions)
    baseline_raw_metrics = compute_metrics(test_sales, test_baseline_raw)
    baseline_adjusted_metrics = compute_metrics(test_sales, test_baseline_adjusted)

    results = {
        'model': model_metrics,
        'baseline_raw': baseline_raw_metrics,
        'baseline_adjusted': baseline_adjusted_metrics,
        'adjustment_ratio': adjustment_ratio
    }

    if verbose:
        print("=" * 80)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 80)
        print(f"\nAdjustment ratio: {adjustment_ratio:.4f}")
        print(f"\nModel Performance:")
        print(f"  MAE:  {model_metrics['mae']:>12,.2f}")
        print(f"  RMSE: {model_metrics['rmse']:>12,.2f}")
        print(f"  MAPE: {model_metrics['mape']:>12.2f}%")
        print(f"  R²:   {model_metrics['r2']:>12.4f}")

        print(f"\nRaw Baseline (unadjusted):")
        print(f"  MAE:  {baseline_raw_metrics['mae']:>12,.2f}")
        print(f"  RMSE: {baseline_raw_metrics['rmse']:>12,.2f}")
        print(f"  MAPE: {baseline_raw_metrics['mape']:>12.2f}%")
        print(f"  R²:   {baseline_raw_metrics['r2']:>12.4f}")

        print(f"\nAdjusted Baseline (recommended):")
        print(f"  MAE:  {baseline_adjusted_metrics['mae']:>12,.2f}")
        print(f"  RMSE: {baseline_adjusted_metrics['rmse']:>12,.2f}")
        print(f"  MAPE: {baseline_adjusted_metrics['mape']:>12.2f}%")
        print(f"  R²:   {baseline_adjusted_metrics['r2']:>12.4f}")

        # Compute improvement percentages
        if baseline_adjusted_metrics['mae'] > 0:
            mae_improvement = (baseline_adjusted_metrics['mae'] - model_metrics['mae']) / baseline_adjusted_metrics['mae'] * 100
            print(f"\nImprovement over adjusted baseline:")
            print(f"  MAE: {mae_improvement:>12.1f}%")

        print("=" * 80)

    return results


def analyze_baseline_quality(test_split_dir: str, train_split_dir: str) -> Dict[str, float]:
    """
    Analyze the quality of baseline assessments (before adjustment).

    Args:
        test_split_dir: Path to test split directory
        train_split_dir: Path to train split directory

    Returns:
        Dictionary with statistics about baseline quality
    """
    baseline_data = load_baseline_data(test_split_dir, train_split_dir)

    # Compute ratios on test set
    test_ratios = baseline_data['test_sales'] / baseline_data['test_baseline']

    # Filter out invalid ratios
    valid_mask = np.isfinite(test_ratios) & (test_ratios > 0) & (baseline_data['test_baseline'] > 0)
    test_ratios_clean = test_ratios[valid_mask]

    stats = {
        'n_valid': int(valid_mask.sum()),
        'n_total': len(test_ratios),
        'median_ratio': float(np.median(test_ratios_clean)),
        'mean_ratio': float(np.mean(test_ratios_clean)),
        'p25_ratio': float(np.percentile(test_ratios_clean, 25)),
        'p75_ratio': float(np.percentile(test_ratios_clean, 75)),
        'adjustment_ratio': baseline_data['adjustment_ratio'],
    }

    print("=" * 80)
    print("BASELINE QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nValid samples: {stats['n_valid']:,} / {stats['n_total']:,}")
    print(f"\nRatio of sale_amount / baseline (test set):")
    print(f"  25th percentile: {stats['p25_ratio']:.4f}")
    print(f"  Median:          {stats['median_ratio']:.4f}")
    print(f"  Mean:            {stats['mean_ratio']:.4f}")
    print(f"  75th percentile: {stats['p75_ratio']:.4f}")
    print(f"\nAdjustment ratio (from training data): {stats['adjustment_ratio']:.4f}")

    # Check if test and train distributions are similar
    ratio_diff = abs(stats['median_ratio'] - stats['adjustment_ratio'])
    if ratio_diff > 0.1 * stats['adjustment_ratio']:
        print(f"\nWARNING: Test set median ratio differs significantly from training adjustment ratio")
        print(f"         Difference: {ratio_diff:.4f} ({ratio_diff / stats['adjustment_ratio'] * 100:.1f}%)")
        print(f"         This may indicate temporal drift in assessment practices")

    print("=" * 80)

    return stats


# Example usage
if __name__ == "__main__":
    # Example: Compare TabPFN predictions to baseline
    import sys

    if len(sys.argv) < 4:
        print("Usage: python baseline_comparison.py <predictions.npy> <test_split_dir> <train_split_dir>")
        print("\nExample:")
        print("  python baseline_comparison.py \\")
        print("    results/predictions.npy \\")
        print("    /scratch/.../preprocessed/v1_no_onehot/test_v1 \\")
        print("    /scratch/.../preprocessed/v1_no_onehot/test_v1/train_v2")
        sys.exit(1)

    predictions_file = sys.argv[1]
    test_split_dir = sys.argv[2]
    train_split_dir = sys.argv[3]

    # Load predictions
    predictions = np.load(predictions_file)

    # Analyze baseline quality
    print("\n")
    analyze_baseline_quality(test_split_dir, train_split_dir)

    # Compare to baseline
    print("\n")
    results = compare_to_baseline(predictions, test_split_dir, train_split_dir)
