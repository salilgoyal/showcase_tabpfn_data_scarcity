"""
Data sampling strategies for cross-county experiments.

This module provides different sampling strategies for creating train/test splits
across multiple counties. Each sampler implements a standard interface for consistency.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CountyAllocation:
    """Represents allocation of data samples for a county."""

    def __init__(self, fips: int, n_samples: int, source: str):
        """
        Initialize county allocation.

        Args:
            fips: County FIPS code
            n_samples: Number of samples to use from this county
            source: Source category (e.g., 'small_county', 'other_county', 'remaining_county')
        """
        self.fips = fips
        self.n_samples = n_samples
        self.source = source

    def __repr__(self):
        return f"CountyAllocation(fips={self.fips}, n_samples={self.n_samples}, source='{self.source}')"


class SamplingResult:
    """Result of a sampling strategy containing train and test allocations."""

    def __init__(
        self,
        train_allocations: List[CountyAllocation],
        test_allocations: List[CountyAllocation]
    ):
        """
        Initialize sampling result.

        Args:
            train_allocations: List of county allocations for training/in-context
            test_allocations: List of county allocations for testing
        """
        self.train_allocations = train_allocations
        self.test_allocations = test_allocations

    @property
    def total_train_samples(self) -> int:
        """Total number of training samples."""
        return sum(alloc.n_samples for alloc in self.train_allocations)

    @property
    def total_test_samples(self) -> int:
        """Total number of test samples."""
        return sum(alloc.n_samples for alloc in self.test_allocations)

    @property
    def train_counties(self) -> List[int]:
        """List of county FIPS codes in training set."""
        return [alloc.fips for alloc in self.train_allocations]

    @property
    def test_counties(self) -> List[int]:
        """List of county FIPS codes in test set."""
        return [alloc.fips for alloc in self.test_allocations]

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert allocations to DataFrames.

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = pd.DataFrame([
            {'fips': a.fips, 'rows': a.n_samples, 'source': a.source}
            for a in self.train_allocations
        ])

        test_df = pd.DataFrame([
            {'fips': a.fips, 'rows': a.n_samples, 'source': a.source}
            for a in self.test_allocations
        ])

        return train_df, test_df

    def __repr__(self):
        return (
            f"SamplingResult(train={self.total_train_samples} samples from "
            f"{len(self.train_allocations)} counties, "
            f"test={self.total_test_samples} samples from "
            f"{len(self.test_allocations)} counties)"
        )


class BaseCountySampler(ABC):
    """Base class for county sampling strategies."""

    @abstractmethod
    def sample(self) -> SamplingResult:
        """
        Execute the sampling strategy.

        Returns:
            SamplingResult containing train and test allocations
        """
        pass


class SmallCountyInContextSampler(BaseCountySampler):
    """
    Sampler for 10K in-context learning experiment with small counties.

    Strategy:
    1. Sample N small counties randomly
    2. Split each small county into in-context and test portions
    3. Fill up to target size for in-context by sampling from other counties
    4. Fill up to target size for test by sampling from remaining counties

    This ensures:
    - Small test counties have data in both in-context and test sets
    - No overlap between counties used for filling train vs test
    - Reproducible sampling with random seeds
    """

    def __init__(
        self,
        small_county_metadata_file: str,
        all_county_metadata_file: str,
        n_small_counties: int = 50,
        small_county_test_ratio: float = 0.5,
        target_train_size: int = 10000,
        target_test_size: int = 10000,
        other_county_train_ratio: float = 0.2,
        remaining_county_test_ratio: float = 0.5,
        random_seed: int = 42
    ):
        """
        Initialize sampler.

        Args:
            small_county_metadata_file: Path to small county metadata CSV
            all_county_metadata_file: Path to all county metadata CSV
            n_small_counties: Number of small counties to sample as test set
            small_county_test_ratio: Fraction of each small county for test (rest for train)
            target_train_size: Target number of training/in-context samples
            target_test_size: Target number of test samples
            other_county_train_ratio: Fraction to sample from other counties for training
            remaining_county_test_ratio: Fraction to sample from remaining counties for test
            random_seed: Random seed for reproducibility
        """
        self.small_county_metadata_file = Path(small_county_metadata_file)
        self.all_county_metadata_file = Path(all_county_metadata_file)
        self.n_small_counties = n_small_counties
        self.small_county_test_ratio = small_county_test_ratio
        self.target_train_size = target_train_size
        self.target_test_size = target_test_size
        self.other_county_train_ratio = other_county_train_ratio
        self.remaining_county_test_ratio = remaining_county_test_ratio
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)

    def sample(self) -> SamplingResult:
        """
        Execute the sampling strategy.

        Returns:
            SamplingResult containing train and test allocations
        """
        logger.info("Starting SmallCountyInContextSampler")

        # Load metadata
        logger.info(f"Loading small county metadata from {self.small_county_metadata_file}")
        small_counties = pd.read_csv(self.small_county_metadata_file)

        logger.info(f"Loading all county metadata from {self.all_county_metadata_file}")
        all_counties = pd.read_csv(self.all_county_metadata_file)

        logger.info(f"Total small counties available: {len(small_counties)}")
        logger.info(f"Total counties available: {len(all_counties)}")

        # Step 1: Sample small counties
        sampled_small_counties = small_counties.sample(
            n=self.n_small_counties,
            random_state=self.random_seed
        )
        sampled_small_fips = set(sampled_small_counties['fips'].values)

        logger.info(f"Sampled {self.n_small_counties} small counties")

        # Step 2: Split each small county into train and test
        train_allocations = []
        test_allocations = []

        for _, row in sampled_small_counties.iterrows():
            fips = row['fips']
            total_rows = row['row_count']

            # Split based on test ratio
            test_rows = int(total_rows * self.small_county_test_ratio)
            train_rows = total_rows - test_rows

            test_allocations.append(
                CountyAllocation(fips, test_rows, 'small_county')
            )
            train_allocations.append(
                CountyAllocation(fips, train_rows, 'small_county')
            )

        current_train_size = sum(a.n_samples for a in train_allocations)
        current_test_size = sum(a.n_samples for a in test_allocations)

        logger.info(f"Training rows from small counties: {current_train_size}")
        logger.info(f"Test rows from small counties: {current_test_size}")

        # Step 3: Fill training set to target size
        remaining_train_needed = self.target_train_size - current_train_size
        logger.info(f"Need {remaining_train_needed} more training rows to reach {self.target_train_size}")

        # Filter out small counties
        other_counties = all_counties[~all_counties['fips'].isin(sampled_small_fips)].copy()
        other_counties = other_counties.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        counties_used_for_train = set()

        if remaining_train_needed > 0:
            logger.info(f"Sampling from {len(other_counties)} other counties for training...")

            accumulated_rows = 0
            for _, row in other_counties.iterrows():
                if accumulated_rows >= remaining_train_needed:
                    break

                fips = row['fips']
                total_rows = row['row_count']

                # Take specified ratio from this county
                sample_rows = int(total_rows * self.other_county_train_ratio)
                sample_rows = min(sample_rows, remaining_train_needed - accumulated_rows)

                if sample_rows > 0:
                    train_allocations.append(
                        CountyAllocation(fips, sample_rows, 'other_county')
                    )
                    accumulated_rows += sample_rows
                    counties_used_for_train.add(fips)

            logger.info(f"Added {accumulated_rows} rows from {len(counties_used_for_train)} other counties to training")

        # Step 4: Fill test set to target size
        remaining_test_needed = self.target_test_size - current_test_size
        logger.info(f"Need {remaining_test_needed} more test rows to reach {self.target_test_size}")

        if remaining_test_needed > 0:
            # Get remaining counties (not in small sample or used for training)
            all_used_fips = sampled_small_fips.union(counties_used_for_train)
            remaining_counties = all_counties[~all_counties['fips'].isin(all_used_fips)].copy()
            remaining_counties = remaining_counties.sample(
                frac=1,
                random_state=self.random_seed + 1
            ).reset_index(drop=True)

            logger.info(f"Sampling from {len(remaining_counties)} remaining counties for test...")

            accumulated_test_rows = 0
            counties_used_for_test = 0

            # Cycle through remaining counties until target reached
            while accumulated_test_rows < remaining_test_needed and len(remaining_counties) > 0:
                for _, row in remaining_counties.iterrows():
                    if accumulated_test_rows >= remaining_test_needed:
                        break

                    fips = row['fips']
                    total_rows = row['row_count']

                    # Take specified ratio from this county
                    sample_rows = int(total_rows * self.remaining_county_test_ratio)
                    sample_rows = min(sample_rows, remaining_test_needed - accumulated_test_rows)

                    if sample_rows > 0:
                        test_allocations.append(
                            CountyAllocation(fips, sample_rows, 'remaining_county')
                        )
                        accumulated_test_rows += sample_rows
                        counties_used_for_test += 1

                # Break if we've cycled through once
                if accumulated_test_rows < remaining_test_needed:
                    logger.warning(
                        f"Exhausted all remaining counties. Test set has "
                        f"{current_test_size + accumulated_test_rows} rows."
                    )
                    break

            logger.info(f"Added {accumulated_test_rows} rows from {counties_used_for_test} remaining counties to test")

        result = SamplingResult(train_allocations, test_allocations)

        logger.info("=" * 60)
        logger.info("SAMPLING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Training set: {result.total_train_samples} samples from {len(train_allocations)} counties")
        logger.info(f"Test set: {result.total_test_samples} samples from {len(test_allocations)} counties")

        return result
