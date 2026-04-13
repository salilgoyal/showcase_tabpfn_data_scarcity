"""
Data splitting utilities for cross-validation and train/test splits.
"""

import numpy as np
from sklearn.model_selection import KFold
from typing import List, Tuple, Generator
import logging

logger = logging.getLogger(__name__)


class RepeatedKFoldSplitter:
    """Generate repeated K-fold cross-validation splits."""

    def __init__(self, n_splits: int, n_repeats: int, random_state: int = 42):
        """
        Initialize splitter.

        Args:
            n_splits: Number of folds (k)
            n_repeats: Number of repetitions (R)
            random_state: Base random seed
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, n_samples: int) -> Generator[Tuple[int, int, np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for repeated k-fold CV.

        Args:
            n_samples: Number of samples in dataset

        Yields:
            Tuple of (repetition, fold, train_indices, test_indices)
        """
        for rep in range(self.n_repeats):
            # Use different random state for each repetition
            seed = self.random_state + 100 * rep

            kfold = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_samples))):
                yield rep, fold, train_idx, test_idx

    def get_total_splits(self) -> int:
        """Get total number of train/test splits."""
        return self.n_splits * self.n_repeats


class PooledDataSplitter:
    """
    Splitter for pooled cross-county experiments.
    Holds out test samples from one county, trains on rest of pooled data.
    """

    def __init__(
        self,
        test_fraction: float = 0.2,
        min_test_samples: int = 5,
        n_iterations: int = 10,
        random_state: int = 42
    ):
        """
        Initialize pooled data splitter.

        Args:
            test_fraction: Fraction of target county to use as test set
            min_test_samples: Minimum number of test samples
            n_iterations: Number of different test samples to draw
            random_state: Base random seed
        """
        self.test_fraction = test_fraction
        self.min_test_samples = min_test_samples
        self.n_iterations = n_iterations
        self.random_state = random_state

    def compute_test_size(self, county_size: int) -> int:
        """
        Compute number of test samples to draw from county.

        Args:
            county_size: Number of samples in county

        Returns:
            Number of test samples
        """
        test_size = max(
            self.min_test_samples,
            int(county_size * self.test_fraction)
        )

        # Don't take more than 80% as test set
        test_size = min(test_size, int(county_size * 0.8))

        return test_size

    def generate_test_indices(
        self,
        county_size: int,
        iteration: int
    ) -> np.ndarray:
        """
        Generate test indices for a specific iteration.

        Args:
            county_size: Number of samples in county
            iteration: Iteration number (0 to n_iterations-1)

        Returns:
            Array of test indices
        """
        test_size = self.compute_test_size(county_size)

        # Use different seed for each iteration
        seed = self.random_state + iteration
        rng = np.random.RandomState(seed)

        # Sample without replacement
        test_indices = rng.choice(
            county_size,
            size=test_size,
            replace=False
        )

        return test_indices

    def create_pooled_split(
        self,
        county_data_dict: dict,
        target_county_fips: int,
        iteration: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split for pooled experiment.

        Args:
            county_data_dict: Dict mapping FIPS -> (X, y) tuples
            target_county_fips: FIPS code of county to test on
            iteration: Which iteration (for random seed)

        Returns:
            Tuple of (X_train_pool, y_train_pool, X_test, y_test)
        """
        if target_county_fips not in county_data_dict:
            raise ValueError(f"County {target_county_fips} not in data dict")

        # Get target county data
        X_target, y_target = county_data_dict[target_county_fips]

        # Generate test indices
        test_indices = self.generate_test_indices(len(X_target), iteration)

        # Create test set
        X_test = X_target.iloc[test_indices].copy()
        y_test = y_target.iloc[test_indices].copy()

        # Create train set: all other counties + remaining samples from target county
        train_indices = np.array([i for i in range(len(X_target)) if i not in test_indices])

        X_train_list = []
        y_train_list = []

        # Add remaining samples from target county
        if len(train_indices) > 0:
            X_train_list.append(X_target.iloc[train_indices])
            y_train_list.append(y_target.iloc[train_indices])

        # Add all samples from other counties
        for fips, (X, y) in county_data_dict.items():
            if fips != target_county_fips:
                X_train_list.append(X)
                y_train_list.append(y)

        # Concatenate
        import pandas as pd
        X_train_pool = pd.concat(X_train_list, ignore_index=True)
        y_train_pool = pd.concat(y_train_list, ignore_index=True)

        logger.debug(
            f"Pooled split: train_size={len(X_train_pool)}, "
            f"test_size={len(X_test)}, target_county={target_county_fips}"
        )

        return X_train_pool, y_train_pool, X_test, y_test
