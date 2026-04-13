"""
Abstract base class for model wrappers.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict


class BaseModel(ABC):
    """Abstract interface for model wrappers."""

    def __init__(self, random_state: int = 42):
        """
        Initialize model.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the model on training data.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get model name.

        Returns:
            Model name string
        """
        pass

    def get_hyperparameters(self) -> Optional[Dict]:
        """
        Get the hyperparameters used.

        Returns:
            Dictionary of hyperparameters, or None if not applicable
        """
        return self.best_params

    def cleanup(self) -> None:
        """
        Clean up resources (GPU memory, etc.).
        Override if needed.
        """
        pass
