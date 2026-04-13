"""
XGBoost model wrapper with Optuna hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import torch
import gc
import time
import logging
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostModel(BaseModel):
    """XGBoost model wrapper with Optuna tuning."""

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 3,
        use_gpu: bool = True,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.

        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of CV folds for hyperparameter tuning
            use_gpu: Use GPU acceleration if available
            random_state: Random seed
        """
        super().__init__(random_state)
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tune_time = 0.0

        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available, using CPU for XGBoost")
            self.use_gpu = False

    def _suggest_params(self, trial: optuna.Trial, train_size: int) -> dict:
        """
        Suggest hyperparameters for Optuna trial.

        Args:
            trial: Optuna trial object
            train_size: Number of training samples

        Returns:
            Dictionary of suggested hyperparameters
        """
        if train_size < 100:
            # Constrained search for very small datasets
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.01, 2.0, log=True),
            }
        else:
            # Full search space
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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit XGBoost model with Optuna hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        import xgboost as xgb

        train_size = len(X_train)
        logger.debug(
            f"Fitting XGBoost with Optuna tuning: "
            f"{train_size} samples, {X_train.shape[1]} features, "
            f"{self.n_trials} trials, {self.cv_folds}-fold CV"
        )

        # Base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        # Try GPU if requested, with fallback to CPU
        if self.use_gpu:
            try:
                # Test if GPU is actually available for XGBoost
                # Note: Use tree_method='hist' with device='cuda' (new syntax as of XGBoost 2.0)
                test_model = xgb.XGBRegressor(
                    tree_method='hist',
                    device='cuda',
                    n_estimators=1
                )
                # Quick test fit to verify GPU works
                test_X = X_train.iloc[:min(5, len(X_train))]
                test_y = y_train.iloc[:min(5, len(y_train))]
                test_model.fit(test_X, test_y)

                # GPU works, use it
                base_params['tree_method'] = 'hist'  # Use 'hist' not 'gpu_hist' (deprecated)
                base_params['device'] = 'cuda'       # GPU is controlled by device parameter
                logger.debug("Using GPU acceleration for XGBoost")
            except Exception as e:
                logger.warning(f"GPU requested but not available for XGBoost: {e}. Falling back to CPU.")
                self.use_gpu = False

        # Define Optuna objective
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        def objective(trial):
            params = self._suggest_params(trial, train_size)
            model = xgb.XGBRegressor(**{**base_params, **params})

            # Cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv_folds,
                scoring=mae_scorer,
                n_jobs=1  # XGBoost already parallelizes
            )

            return -scores.mean()  # Return negative MAE (Optuna minimizes)

        # Run Optuna optimization
        start_time = time.time()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.tune_time = time.time() - start_time
        self.best_params = study.best_params

        logger.debug(
            f"Optuna tuning completed in {self.tune_time:.2f}s, "
            f"best MAE: {study.best_value:.2f}"
        )

        # Train final model with best parameters
        self.model = xgb.XGBRegressor(**{**base_params, **self.best_params})
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
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

    def get_name(self) -> str:
        """Get model name."""
        return "xgboost"

    def get_hyperparameters(self) -> dict:
        """Get best hyperparameters found."""
        return self.best_params if self.best_params else {}

    def get_tune_time(self) -> float:
        """Get hyperparameter tuning time."""
        return self.tune_time

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.use_gpu:
            torch.cuda.empty_cache()

        gc.collect()
