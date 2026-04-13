"""
Model runner abstractions for TabPFN and XGBoost.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import gc
import logging
import time
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseModelRunner(ABC):
    """Abstract base class for model runners."""

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions."""
        pass

    @abstractmethod
    def get_name(self):
        """Return model name."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (GPU memory, etc.)."""
        pass


class TabPFNRunner(BaseModelRunner):
    """
    TabPFN model runner with GPU memory management.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None

    def fit(self, X_train, y_train):
        """
        Fit TabPFN model.

        Args:
            X_train: Training features (should not contain object columns)
            y_train: Training targets
        """
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self.model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        self.model.device = self.device
        self.model.ignore_pretraining_limits = True
        self.model.fit(X_train, y_train)

    def predict(self, X_test, batch_size=1000):
        """
        Predict with TabPFN model in batches to avoid OOM.

        Args:
            X_test: Test features
            batch_size: Number of samples to predict at once (default: 1000)

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # If test set is small, predict all at once
        if len(X_test) <= batch_size:
            return self.model.predict(X_test)

        # Batch prediction for large test sets
        logger.info(f"Predicting in batches of {batch_size} (total: {len(X_test)} samples)")
        predictions = []

        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            X_batch = X_test.iloc[i:batch_end] if hasattr(X_test, 'iloc') else X_test[i:batch_end]

            batch_pred = self.model.predict(X_batch)
            predictions.append(batch_pred)

            # Clear cache after each batch
            if i + batch_size < len(X_test):  # Don't clear on last batch (will be cleared in cleanup)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            if (i // batch_size + 1) % 5 == 0:  # Log every 5 batches
                logger.info(f"Predicted {batch_end}/{len(X_test)} samples")

        return np.concatenate(predictions)

    def get_name(self):
        return "tabpfn"

    def cleanup(self):
        """Clear GPU memory and delete model."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class XGBoostRunner(BaseModelRunner):
    """
    XGBoost model runner with hyperparameter tuning.
    """

    def __init__(self, n_iter=100, cv=3, use_gpu=True, random_state=42):
        """
        Initialize XGBoost runner.

        Args:
            n_iter: Number of hyperparameter combinations to try (default: 10)
            cv: Number of cross-validation folds (default: 3)
            use_gpu: Use GPU acceleration if available (default: True)
            random_state: Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.cv = cv
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.tune_time = 0.0

    def _suggest_params(self, trial, train_size):
        """
        Suggest hyperparameters for Optuna trial based on training size.

        For small train sizes (<100), use more constrained ranges to avoid overfitting.

        Args:
            trial: Optuna trial object
            train_size: Number of training samples

        Returns:
            Dictionary of suggested hyperparameters
        """
        if train_size < 100:
            # Simpler models for very small training sets
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10, log=True),
                'gamma': trial.suggest_float('gamma', 0.1, 2.0, log=True),
            }
        else:
            # Full search space for larger training sets
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 2.0, log=True),
            }

        return params

    def fit(self, X_train, y_train):
        """
        Fit XGBoost model with Optuna hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        import xgboost as xgb

        train_size = len(X_train)
        logger.info(f"Training XGBoost with Optuna hyperparameter search (n_trials={self.n_iter}, cv={self.cv})")

        # Base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        if self.use_gpu and torch.cuda.is_available():
            base_params['tree_method'] = 'gpu_hist'
            base_params['gpu_id'] = 0
            logger.info("Using GPU acceleration for XGBoost")
        else:
            base_params['tree_method'] = 'hist'
            logger.info("Using CPU for XGBoost")

        # Define objective function for Optuna
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        def objective(trial):
            # Suggest hyperparameters
            params = self._suggest_params(trial, train_size)

            # Create model with suggested parameters
            model = xgb.XGBRegressor(**{**base_params, **params})

            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv,
                scoring=mae_scorer,
                n_jobs=1  # XGBoost already uses multiple cores
            )

            # Return mean MAE (Optuna minimizes, scorer returns negative MAE)
            return -scores.mean()

        # Run Optuna optimization
        start_time = time.time()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_iter, show_progress_bar=False)

        self.tune_time = time.time() - start_time
        self.best_params = study.best_params

        # Train final model with best parameters
        self.model = xgb.XGBRegressor(**{**base_params, **self.best_params})
        self.model.fit(X_train, y_train)

        logger.info(f"Best XGBoost params: {self.best_params}")
        logger.info(f"Best CV MAE: {study.best_value:.2f}")
        logger.info(f"Tuning completed in {self.tune_time:.2f}s")

    def predict(self, X_test):
        """
        Predict with tuned XGBoost model.

        Args:
            X_test: Test features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X_test)

    def get_name(self):
        return "xgboost"

    def get_tune_time(self):
        """Return hyperparameter tuning time."""
        return self.tune_time

    def cleanup(self):
        """Clear memory and delete model."""
        del self.model
        self.model = None
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
