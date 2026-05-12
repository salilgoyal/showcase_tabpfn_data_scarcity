"""
TabICL model wrapper.
"""

import pandas as pd
import numpy as np
import logging
import gc
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class TabICLModel(BaseModel):
    """TabICL regressor wrapper."""

    def __init__(self, device: str = None, n_estimators: int = 8, random_state: int = 42):
        """
        Initialize TabICL model.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            n_estimators: Number of ensemble members (more = better but slower)
            random_state: Random seed (not directly supported by TabICL, stored for metadata)
        """
        super().__init__(random_state)
        self.device = device
        self.n_estimators = n_estimators

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit TabICL model. Downloads checkpoint on first call (~500MB, no auth required).

        Args:
            X_train: Training features
            y_train: Training targets
        """
        from tabicl import TabICLRegressor

        logger.info(
            f"Starting TabICL fit with {len(X_train)} samples, "
            f"{X_train.shape[1]} features, n_estimators={self.n_estimators}"
        )

        self.model = TabICLRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
        )
        self.model.fit(X_train, y_train)

        logger.info("TabICL fit complete")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict with TabICL model.

        Args:
            X_test: Test features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info(f"TabICL predicting on {len(X_test)} samples")
        return self.model.predict(X_test)

    def get_name(self) -> str:
        """Get model name."""
        return "tabicl"

    def get_hyperparameters(self) -> dict:
        """Return configuration used."""
        return {
            "n_estimators": self.n_estimators,
            "device": self.device,
        }

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
