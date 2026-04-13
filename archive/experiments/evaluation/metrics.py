"""
Evaluation metrics for model performance.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, log_transformed: bool = False) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True target values (may be log-transformed)
        y_pred: Predicted target values (may be log-transformed)
        log_transformed: If True, inverse-transform (exp) both y_true and y_pred
                        before computing metrics to get results on original scale

    Returns:
        Dictionary of metric name -> value
    """
    # ====================================================================
    # MODIFICATION FOR EVELYN PREPROCESSING: INVERSE LOG TRANSFORMATION
    # If targets are log-transformed, apply exp() to get back to original scale
    # ====================================================================
    if log_transformed:
        y_true_original = np.exp(y_true)
        y_pred_original = np.exp(y_pred)
    else:
        y_true_original = y_true
        y_pred_original = y_pred

    metrics = {}

    try:
        metrics['r2'] = r2_score(y_true_original, y_pred_original)
    except Exception as e:
        logger.warning(f"Error computing R2: {e}")
        metrics['r2'] = np.nan

    try:
        metrics['mae'] = mean_absolute_error(y_true_original, y_pred_original)
    except Exception as e:
        logger.warning(f"Error computing MAE: {e}")
        metrics['mae'] = np.nan

    try:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    except Exception as e:
        logger.warning(f"Error computing RMSE: {e}")
        metrics['rmse'] = np.nan

    # Also compute MSE for completeness
    try:
        metrics['mse'] = mean_squared_error(y_true_original, y_pred_original)
    except Exception as e:
        logger.warning(f"Error computing MSE: {e}")
        metrics['mse'] = np.nan

    return metrics


def compute_relative_performance(
    baseline_metric: float,
    comparison_metric: float,
    higher_is_better: bool = True
) -> float:
    """
    Compute relative performance improvement.

    Args:
        baseline_metric: Baseline model metric value
        comparison_metric: Comparison model metric value
        higher_is_better: Whether higher metric values are better

    Returns:
        Relative improvement (positive = comparison is better)
    """
    if baseline_metric == 0:
        return np.nan

    if higher_is_better:
        # For R2, etc.
        relative = (comparison_metric - baseline_metric) / abs(baseline_metric)
    else:
        # For MAE, RMSE, etc.
        relative = (baseline_metric - comparison_metric) / abs(baseline_metric)

    return relative * 100  # Return as percentage
