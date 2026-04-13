"""
Phase 2 Preprocessing Utilities.

This module provides utilities for per-experiment preprocessing that must be
fit on training data and applied to both train and test to avoid data leakage.

Phase 2 includes:
- Winsorization (clip outliers based on train percentiles)
- Normalization (standardize using train mean/std)
- Imputation (fill missing with train median/mean)

Usage:
    # Fit on train data
    preprocessor = Phase2Preprocessor(config)
    preprocessor.fit(X_train, y_train)

    # Transform both train and test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    y_train_processed = preprocessor.transform_target(y_train)
    y_test_processed = preprocessor.transform_target(y_test)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class WinsorizerParams:
    """Parameters for winsorization."""
    lower_bounds: Dict[str, float] = field(default_factory=dict)
    upper_bounds: Dict[str, float] = field(default_factory=dict)
    target_lower: Optional[float] = None
    target_upper: Optional[float] = None


@dataclass
class ImputerParams:
    """Parameters for imputation."""
    fill_values: Dict[str, float] = field(default_factory=dict)


class Phase2Preprocessor:
    """
    Phase 2 preprocessing: winsorization, normalization, imputation.

    All transformations are fit on training data only and applied to both
    train and test to prevent data leakage.
    """

    def __init__(self, config: dict):
        """
        Initialize Phase 2 preprocessor.

        Args:
            config: Phase 2 preprocessing configuration with keys:
                - winsorize: bool
                - winsorize_percentile: int (e.g., 1 for 1st/99th percentile)
                - normalize_continuous: bool
                - impute_method: str ("median", "mean", "zero", "none")
        """
        self.config = config
        self.winsorize = config.get("winsorize", True)
        self.winsorize_percentile = config.get("winsorize_percentile", 1)
        self.normalize = config.get("normalize_continuous", True)
        self.impute_method = config.get("impute_method", "median")

        # Fitted parameters (set during fit())
        self.winsorizer_params: Optional[WinsorizerParams] = None
        self.scaler: Optional[StandardScaler] = None
        self.imputer_params: Optional[ImputerParams] = None
        self.continuous_cols: List[str] = []

        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        continuous_cols: Optional[List[str]] = None
    ) -> "Phase2Preprocessor":
        """
        Fit preprocessing parameters on training data.

        Args:
            X_train: Training features
            y_train: Training target (for winsorization)
            continuous_cols: List of continuous column names (auto-detected if None)

        Returns:
            self (for method chaining)
        """
        logger.info("Fitting Phase 2 preprocessor on training data...")

        # Identify continuous columns
        if continuous_cols is not None:
            self.continuous_cols = [c for c in continuous_cols if c in X_train.columns]
        else:
            self.continuous_cols = self._detect_continuous_cols(X_train)

        logger.info(f"  Continuous columns: {len(self.continuous_cols)}")

        # Fit winsorizer
        if self.winsorize:
            self.winsorizer_params = self._fit_winsorizer(X_train, y_train)
            logger.info(f"  Fitted winsorizer at {self.winsorize_percentile}/{100 - self.winsorize_percentile} percentiles")

        # Fit imputer (before scaler, since scaler needs complete data)
        if self.impute_method != "none":
            self.imputer_params = self._fit_imputer(X_train)
            logger.info(f"  Fitted imputer with method: {self.impute_method}")

        # Fit scaler
        if self.normalize and len(self.continuous_cols) > 0:
            self.scaler = self._fit_scaler(X_train)
            logger.info(f"  Fitted StandardScaler on {len(self.continuous_cols)} columns")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to features.

        Order: winsorize -> impute -> normalize

        Args:
            X: Features DataFrame

        Returns:
            Transformed features
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        X = X.copy()

        # 1. Winsorize
        if self.winsorize and self.winsorizer_params is not None:
            X = self._apply_winsorizer(X, self.winsorizer_params)

        # 2. Impute
        if self.impute_method != "none" and self.imputer_params is not None:
            X = self._apply_imputer(X, self.imputer_params)

        # 3. Normalize
        if self.normalize and self.scaler is not None:
            X = self._apply_scaler(X, self.scaler)

        return X

    def transform_target(self, y: pd.Series) -> pd.Series:
        """
        Apply winsorization to target variable.

        Note: Log transformation is done in Phase 1, not here.

        Args:
            y: Target series

        Returns:
            Transformed target
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        y = y.copy()

        # Only winsorize target
        if self.winsorize and self.winsorizer_params is not None:
            if self.winsorizer_params.target_lower is not None:
                y = np.clip(
                    y,
                    self.winsorizer_params.target_lower,
                    self.winsorizer_params.target_upper
                )

        return y

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        continuous_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit and transform training data.

        Args:
            X_train: Training features
            y_train: Training target
            continuous_cols: List of continuous column names

        Returns:
            Tuple of (transformed X_train, transformed y_train)
        """
        self.fit(X_train, y_train, continuous_cols)
        X_transformed = self.transform(X_train)
        y_transformed = self.transform_target(y_train) if y_train is not None else None
        return X_transformed, y_transformed

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _detect_continuous_cols(self, X: pd.DataFrame) -> List[str]:
        """Auto-detect continuous columns based on dtype."""
        continuous = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Skip binary columns
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) > 2:
                    continuous.append(col)
        return continuous

    def _fit_winsorizer(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series]
    ) -> WinsorizerParams:
        """Compute winsorization bounds from training data."""
        params = WinsorizerParams()

        lower_pct = self.winsorize_percentile
        upper_pct = 100 - self.winsorize_percentile

        # Feature bounds
        for col in self.continuous_cols:
            if col in X_train.columns:
                values = X_train[col].dropna()
                if len(values) > 0:
                    params.lower_bounds[col] = np.percentile(values, lower_pct)
                    params.upper_bounds[col] = np.percentile(values, upper_pct)

        # Target bounds
        if y_train is not None:
            values = y_train.dropna()
            if len(values) > 0:
                params.target_lower = np.percentile(values, lower_pct)
                params.target_upper = np.percentile(values, upper_pct)

        return params

    def _apply_winsorizer(
        self,
        X: pd.DataFrame,
        params: WinsorizerParams
    ) -> pd.DataFrame:
        """Apply winsorization to features."""
        for col in self.continuous_cols:
            if col in X.columns and col in params.lower_bounds:
                X[col] = np.clip(
                    X[col],
                    params.lower_bounds[col],
                    params.upper_bounds[col]
                )
        return X

    def _fit_imputer(self, X_train: pd.DataFrame) -> ImputerParams:
        """Compute imputation values from training data."""
        params = ImputerParams()

        for col in X_train.columns:
            if X_train[col].isna().any():
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    if self.impute_method == "median":
                        params.fill_values[col] = X_train[col].median()
                    elif self.impute_method == "mean":
                        params.fill_values[col] = X_train[col].mean()
                    elif self.impute_method == "zero":
                        params.fill_values[col] = 0.0
                else:
                    # For non-numeric, use mode
                    mode = X_train[col].mode()
                    params.fill_values[col] = mode[0] if len(mode) > 0 else 0

        return params

    def _apply_imputer(
        self,
        X: pd.DataFrame,
        params: ImputerParams
    ) -> pd.DataFrame:
        """Apply imputation to features."""
        for col, fill_value in params.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
        return X

    def _fit_scaler(self, X_train: pd.DataFrame) -> StandardScaler:
        """Fit StandardScaler on training data."""
        # Get continuous columns that exist
        cols_to_scale = [c for c in self.continuous_cols if c in X_train.columns]

        if not cols_to_scale:
            return None

        scaler = StandardScaler()
        # Handle any remaining NaNs by filling with 0 temporarily
        X_for_fit = X_train[cols_to_scale].fillna(0)
        scaler.fit(X_for_fit)

        return scaler

    def _apply_scaler(
        self,
        X: pd.DataFrame,
        scaler: StandardScaler
    ) -> pd.DataFrame:
        """Apply StandardScaler to features."""
        cols_to_scale = [c for c in self.continuous_cols if c in X.columns]

        if not cols_to_scale or scaler is None:
            return X

        # Scale
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

        return X

    def get_params(self) -> dict:
        """Get all fitted parameters as a dictionary."""
        if not self._is_fitted:
            return {}

        return {
            "winsorizer": {
                "lower_bounds": self.winsorizer_params.lower_bounds if self.winsorizer_params else {},
                "upper_bounds": self.winsorizer_params.upper_bounds if self.winsorizer_params else {},
                "target_lower": self.winsorizer_params.target_lower if self.winsorizer_params else None,
                "target_upper": self.winsorizer_params.target_upper if self.winsorizer_params else None,
            } if self.winsorize else None,
            "imputer": {
                "fill_values": self.imputer_params.fill_values if self.imputer_params else {}
            } if self.impute_method != "none" else None,
            "scaler": {
                "mean": self.scaler.mean_.tolist() if self.scaler else None,
                "std": self.scaler.scale_.tolist() if self.scaler else None,
                "columns": self.continuous_cols
            } if self.normalize else None,
        }


# ==============================================================================
# Convenience functions
# ==============================================================================

def apply_phase2_preprocessing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
    continuous_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function to apply Phase 2 preprocessing.

    Fits on train, transforms both train and test.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        config: Phase 2 config dict
        continuous_cols: Optional list of continuous column names

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) - all transformed
    """
    preprocessor = Phase2Preprocessor(config)
    preprocessor.fit(X_train, y_train, continuous_cols)

    X_train_out = preprocessor.transform(X_train)
    y_train_out = preprocessor.transform_target(y_train)
    X_test_out = preprocessor.transform(X_test)
    y_test_out = preprocessor.transform_target(y_test)

    return X_train_out, y_train_out, X_test_out, y_test_out
