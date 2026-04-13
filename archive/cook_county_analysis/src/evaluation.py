"""
Evaluation metrics and utilities for model comparison.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, log_transformed=False):
    """
    Calculate regression metrics including ratio statistics.

    Args:
        y_true: True target values (may be log-transformed)
        y_pred: Predicted target values (may be log-transformed)
        log_transformed: If True, inverse-transform (exp) both y_true and y_pred before computing metrics

    Returns:
        Dictionary with metrics:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - r2: R² Score
        - mean_ratio: Mean of (y_pred / y_true)
        - median_ratio: Median of (y_pred / y_true)
        - std_ratio: Standard deviation of (y_pred / y_true)
    """
    # ====================================================================
    # MODIFICATION FOR EVELYN PREPROCESSING: INVERSE LOG TRANSFORMATION
    # If targets are log-transformed, apply exp() to get back to original scale
    # ====================================================================
    if log_transformed:
        y_true_original = np.exp(y_true.values if hasattr(y_true, 'values') else y_true)
        y_pred_original = np.exp(y_pred)
    else:
        y_true_original = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred_original = y_pred

    mae = mean_absolute_error(y_true_original, y_pred_original)
    mse = mean_squared_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)

    # Ratio metrics (use original scale)
    ratios = y_pred_original / y_true_original
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    std_ratio = np.std(ratios)

    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'std_ratio': std_ratio
    }


def create_result_dict(seed, train_size, test_type, n_train, n_test,
                       n_train_cbgs, n_test_cbgs, metrics,
                       train_time, pred_time, tune_time=None, status='success'):
    """
    Create a standardized result dictionary.

    Args:
        seed: Random seed used
        train_size: Number of training samples
        test_type: 'full' or 'cbg_matched'
        n_train: Actual training samples
        n_test: Actual test samples
        n_train_cbgs: Number of unique CBGs in training
        n_test_cbgs: Number of unique CBGs in test
        metrics: Dictionary from calculate_metrics()
        train_time: Training time in seconds
        pred_time: Prediction time in seconds
        tune_time: Hyperparameter tuning time (optional, for XGBoost)
        status: 'success' or error message

    Returns:
        Dictionary with all experiment info
    """
    result = {
        'seed': seed,
        'train_size': train_size,
        'test_type': test_type,
        'n_train': n_train,
        'n_test': n_test,
        'n_train_cbgs': n_train_cbgs,
        'n_test_cbgs': n_test_cbgs,
        'mae': metrics.get('mae', np.nan),
        'mse': metrics.get('mse', np.nan),
        'r2': metrics.get('r2', np.nan),
        'mean_ratio': metrics.get('mean_ratio', np.nan),
        'median_ratio': metrics.get('median_ratio', np.nan),
        'std_ratio': metrics.get('std_ratio', np.nan),
        'train_time': train_time,
        'pred_time': pred_time,
        'status': status
    }

    if tune_time is not None:
        result['tune_time'] = tune_time

    return result
