"""
County registry: filter and bin counties by size.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class CountyRegistry:
    """Manages county metadata and filtering."""

    def __init__(self, metadata_file: str, county_bins: List[Dict]):
        """
        Initialize registry.

        Args:
            metadata_file: Path to county_row_counts.csv
            county_bins: List of bin configs, each with min_size, max_size, name, k_folds
        """
        self.metadata_file = metadata_file
        self.county_bins = county_bins
        self.metadata_df = None
        self.load_metadata()

    def load_metadata(self):
        """Load county metadata from CSV."""
        logger.info(f"Loading county metadata from {self.metadata_file}")
        self.metadata_df = pd.read_csv(self.metadata_file)
        logger.info(f"Loaded metadata for {len(self.metadata_df)} counties")

    def filter_counties_by_bin(self, bin_name: str) -> pd.DataFrame:
        """
        Filter counties by size bin.

        Args:
            bin_name: Name of the bin (e.g., "small")

        Returns:
            DataFrame with filtered counties
        """
        # Find bin config
        bin_config = None
        for b in self.county_bins:
            if b['name'] == bin_name:
                bin_config = b
                break

        if bin_config is None:
            raise ValueError(f"Bin '{bin_name}' not found in configuration")

        min_size = bin_config['min_size']
        max_size = bin_config['max_size']

        # Filter by size
        filtered = self.metadata_df[
            (self.metadata_df['row_count'] >= min_size) &
            (self.metadata_df['row_count'] <= max_size)
        ].copy()

        # Add bin info
        filtered['bin_name'] = bin_name
        filtered['k_folds'] = bin_config['k_folds']

        logger.info(
            f"Bin '{bin_name}' ({min_size}-{max_size} rows): "
            f"{len(filtered)} counties"
        )

        return filtered

    def get_all_small_counties(self) -> pd.DataFrame:
        """
        Get all counties in all configured bins.

        Returns:
            DataFrame with all counties in configured bins
        """
        all_counties = []

        for bin_config in self.county_bins:
            bin_df = self.filter_counties_by_bin(bin_config['name'])
            all_counties.append(bin_df)

        combined = pd.concat(all_counties, ignore_index=True)
        logger.info(f"Total counties across all bins: {len(combined)}")

        return combined

    def get_county_info(self, fips: int) -> Dict:
        """
        Get metadata for a specific county.

        Args:
            fips: FIPS code

        Returns:
            Dictionary with county metadata
        """
        row = self.metadata_df[self.metadata_df['fips'] == fips]

        if len(row) == 0:
            raise ValueError(f"County {fips} not found in metadata")

        return row.iloc[0].to_dict()

    def save_filtered_metadata(self, output_path: str):
        """
        Save filtered county metadata with bin assignments.

        Args:
            output_path: Path to save CSV
        """
        filtered = self.get_all_small_counties()
        filtered = filtered.sort_values(['bin_name', 'row_count'])

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        filtered.to_csv(output_file, index=False)
        logger.info(f"Saved filtered metadata to {output_file}")

        # Log summary statistics
        logger.info("\nSummary by bin:")
        for bin_name in filtered['bin_name'].unique():
            bin_df = filtered[filtered['bin_name'] == bin_name]
            logger.info(f"  {bin_name}: {len(bin_df)} counties, "
                       f"size range: {bin_df['row_count'].min()}-{bin_df['row_count'].max()}")
