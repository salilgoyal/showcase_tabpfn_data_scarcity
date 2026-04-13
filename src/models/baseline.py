"""
Baseline model using county assessments (CALCULATED_TOTAL_VALUE).

This model doesn't train - it uses county assessments adjusted by the
median ratio of SALE_AMOUNT / CALCULATED_TOTAL_VALUE computed on the training set.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class BaselineModel(BaseModel):
    """
    Baseline model that uses county assessments multiplied by adjustment ratio.

    This model computes an adjustment ratio from training data:
        ratio = median(y_train / baseline_train)

    Then predicts on test data:
        y_pred = baseline_test * ratio

    The adjustment ratio accounts for counties where assessed values are
    systematically different from market values (e.g., Cook County where
    assessments are 10% of market value).
    """

    def __init__(
        self,
        baseline_values_train: Optional[np.ndarray] = None,
        baseline_values_test: Optional[np.ndarray] = None,
        adjustment_ratio: Optional[float] = None,
        log_transformed: bool = False,
        random_state: int = 42
    ):
        """
        Initialize baseline model.

        Args:
            baseline_values_train: Baseline values for training set (optional)
            baseline_values_test: Baseline values for test set (optional)
            adjustment_ratio: Pre-computed adjustment ratio (optional)
            log_transformed: If True, return predictions in log space
            random_state: Random seed (not used, for interface compatibility)
        """
        super().__init__(random_state)

        self.baseline_values_train = baseline_values_train
        self.baseline_values_test = baseline_values_test
        self.adjustment_ratio = adjustment_ratio
        self.log_transformed = log_transformed

        # Will be computed during fit if not provided
        self._computed_ratio = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Compute adjustment ratio from training data.

        Args:
            X_train: Training features (not used, for interface compatibility)
            y_train: Training targets (sale amounts)
        """
        if self.adjustment_ratio is not None:
            # Use pre-computed ratio
            self._computed_ratio = self.adjustment_ratio
            logger.debug(f"Using pre-computed adjustment ratio: {self._computed_ratio:.4f}")
            return

        if self.baseline_values_train is None:
            raise ValueError(
                "baseline_values_train must be provided if adjustment_ratio is not pre-computed"
            )

        # Compute adjustment ratio as median(sale_amount / baseline)
        # Filter out zeros and invalid values
        y_train_vals = y_train.values
        baseline_train = self.baseline_values_train

        # If target is log-transformed, exp-transform it to get original scale
        if self.log_transformed:
            y_train_original = np.exp(y_train_vals)
        else:
            y_train_original = y_train_vals

        valid_mask = (baseline_train > 0) & (y_train_original > 0) & np.isfinite(baseline_train) & np.isfinite(y_train_original)

        if valid_mask.sum() == 0:
            logger.warning("No valid training samples for computing adjustment ratio. Using ratio=1.0")
            self._computed_ratio = 1.0
        else:
            ratios = y_train_original[valid_mask] / baseline_train[valid_mask]
            self._computed_ratio = float(np.median(ratios))
            logger.debug(
                f"Computed adjustment ratio from {valid_mask.sum()} valid samples: "
                f"{self._computed_ratio:.4f}"
            )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using baseline values * adjustment ratio.

        Args:
            X_test: Test features (not used, for interface compatibility)

        Returns:
            Predictions (baseline * ratio), optionally log-transformed
        """
        if self._computed_ratio is None:
            raise RuntimeError("Model must be fitted before making predictions")

        if self.baseline_values_test is None:
            raise ValueError("baseline_values_test must be provided for prediction")

        # Predict: baseline * adjustment_ratio
        predictions = self.baseline_values_test * self._computed_ratio

        # If target is log-transformed, return predictions in log space
        if self.log_transformed:
            # Filter out zeros and invalid values before log transform
            predictions = np.where(predictions > 0, np.log(predictions), np.nan)

        return predictions

    def get_name(self) -> str:
        """Get model name."""
        return "baseline"

    def get_hyperparameters(self) -> Optional[dict]:
        """Return adjustment ratio as a 'hyperparameter'."""
        if self._computed_ratio is not None:
            return {'adjustment_ratio': self._computed_ratio}
        return None


def load_baseline_data(
    test_set_dir: str,
    train_set_dir: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load baseline values and adjustment ratio from split directories.

    Args:
        test_set_dir: Directory containing test set files
        train_set_dir: Directory containing train set files (optional)

    Returns:
        Tuple of (test_baseline_values, train_baseline_values, adjustment_ratio)

    Raises:
        FileNotFoundError: If required baseline files don't exist
    """
    test_dir = Path(test_set_dir)

    # Load test baseline values
    test_baseline_file = test_dir / "test_baseline_values.npy"
    if not test_baseline_file.exists():
        raise FileNotFoundError(
            f"Test baseline values not found: {test_baseline_file}\n"
            "Make sure you regenerated the test set after enabling baseline saving."
        )

    test_baseline = np.load(test_baseline_file)
    logger.debug(f"Loaded test baseline values: {len(test_baseline)} samples")

    # Load train baseline values if train_set_dir provided
    train_baseline = None
    adjustment_ratio = None

    if train_set_dir:
        train_dir = Path(train_set_dir)

        train_baseline_file = train_dir / "train_baseline_values.npy"
        if train_baseline_file.exists():
            train_baseline = np.load(train_baseline_file)
            logger.debug(f"Loaded train baseline values: {len(train_baseline)} samples")

        # Load adjustment ratio from metadata
        metadata_file = train_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            # Try both keys for backwards compatibility
            adjustment_ratio = metadata.get('baseline_adjustment_ratio') or metadata.get('adjustment_ratio')
            if adjustment_ratio:
                logger.debug(f"Loaded adjustment ratio from metadata: {adjustment_ratio:.4f}")

    return test_baseline, train_baseline, adjustment_ratio
