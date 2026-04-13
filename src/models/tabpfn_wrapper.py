"""
TabPFN model wrapper.
"""

import pandas as pd
import numpy as np
import torch
import gc
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class TabPFNModel(BaseModel):
    """TabPFN model wrapper."""

    def __init__(self, device: str = 'cuda', version: str = 'v2', random_state: int = 42):
        """
        Initialize TabPFN model.

        Args:
            device: Device to use ('cuda' or 'cpu')
            version: TabPFN version ('v2' or 'v2.5')
            random_state: Random seed
        """
        super().__init__(random_state)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.version = version

        if self.device == 'cpu' and device == 'cuda':
            logger.warning("CUDA not available, using CPU for TabPFN")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit TabPFN model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        import sys
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        logger.info(f"Starting TabPFN {self.version} fit with {len(X_train)} samples, {X_train.shape[1]} features")
        sys.stdout.flush()
        sys.stderr.flush()

        # Map version string to ModelVersion enum
        version_map = {
            'v2': ModelVersion.V2,
            'v2.5': ModelVersion.V2_5
        }
        model_version = version_map.get(self.version, ModelVersion.V2)

        logger.info(f"Creating TabPFN {self.version} model instance...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Use TabPFN with device and pretraining limits configuration
        # Initialize with batched mode for fast inference (uses InferenceEngineBatchedNoPreprocessing)
        # rather than fit_preprocessors mode which uses slow InferenceEngineOnDemand
        self.model = TabPFNRegressor.create_default_for_version(
            model_version,
            device=self.device,
            fit_mode="batched",
            ignore_pretraining_limits=True
        )

        logger.info("Model instance created. Switching to fit_preprocessors mode...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Temporarily switch to fit_preprocessors mode to call fit()
        # This is required because fit() expects fit_preprocessors mode
        self.model.fit_mode = "fit_preprocessors"

        logger.info(f"Calling fit() on {len(X_train)} samples (this may take several minutes)...")
        sys.stdout.flush()
        sys.stderr.flush()

        self.model.fit(X_train, y_train)

        logger.info("Fit completed. Switching back to batched mode for inference...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Switch back to batched mode for fast inference
        # This matches the approach used in tabpfn_finetuning.py (line 660)
        self.model.fit_mode = "batched"

        logger.info("TabPFN fit complete and ready for predictions")
        sys.stdout.flush()
        sys.stderr.flush()

    def predict(self, X_test: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
        """
        Predict with TabPFN model.

        Args:
            X_test: Test features
            batch_size: Batch size for prediction (to avoid OOM)

        Returns:
            Predictions array
        """
        import sys

        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info(f"Starting prediction on {len(X_test)} samples with batch_size={batch_size}")
        sys.stdout.flush()
        sys.stderr.flush()

        # If test set is small, predict all at once
        if len(X_test) <= batch_size:
            logger.info(f"Predicting all {len(X_test)} samples in single batch...")
            sys.stdout.flush()
            sys.stderr.flush()
            return self.model.predict(X_test)

        # Batch prediction for large test sets
        logger.info(f"Predicting in batches of {batch_size} ({len(X_test) // batch_size + 1} batches)")
        sys.stdout.flush()
        sys.stderr.flush()

        predictions = []

        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            X_batch = X_test.iloc[i:batch_end]

            logger.info(f"  Batch {i // batch_size + 1}/{len(X_test) // batch_size + 1}: samples {i}-{batch_end}")
            sys.stdout.flush()
            sys.stderr.flush()

            batch_pred = self.model.predict(X_batch)
            predictions.append(batch_pred)

            # Clear cache after each batch
            if self.device == 'cuda' and i + batch_size < len(X_test):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        logger.info("Prediction complete")
        sys.stdout.flush()
        sys.stderr.flush()

        return np.concatenate(predictions)

    def predict_quantiles(
        self,
        X_test: pd.DataFrame,
        quantiles: list[float],
        batch_size: int = 1000
    ) -> dict[float, np.ndarray]:
        """
        Predict quantiles with TabPFN model.

        Args:
            X_test: Test features
            quantiles: List of quantiles to predict (e.g., [0.1, 0.2, ..., 0.9])
            batch_size: Batch size for prediction (to avoid OOM)

        Returns:
            Dictionary mapping quantile -> predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        logger.debug(f"Predicting quantiles: {quantiles}")

        # If test set is small, predict all at once
        if len(X_test) <= batch_size:
            quantile_preds = self.model.predict(
                X_test,
                output_type="quantiles",
                quantiles=quantiles
            )
            return {q: pred for q, pred in zip(quantiles, quantile_preds)}

        # Batch prediction for large test sets
        logger.debug(f"Predicting quantiles in batches of {batch_size}")
        batch_results = {q: [] for q in quantiles}

        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            X_batch = X_test.iloc[i:batch_end]

            # Get quantile predictions for batch
            batch_preds = self.model.predict(
                X_batch,
                output_type="quantiles",
                quantiles=quantiles
            )

            # Store each quantile's predictions
            for q, pred in zip(quantiles, batch_preds):
                batch_results[q].append(pred)

            # Clear cache after each batch
            if self.device == 'cuda' and i + batch_size < len(X_test):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Concatenate batches for each quantile
        return {q: np.concatenate(preds) for q, preds in batch_results.items()}

    def get_name(self) -> str:
        """Get model name."""
        return "tabpfn"

    def get_hyperparameters(self) -> dict:
        """TabPFN has no tunable hyperparameters in our setup."""
        return {
            "device": self.device,
            "version": self.version
        }

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
