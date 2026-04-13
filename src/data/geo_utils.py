"""
Geographic utility functions for county proximity calculations.

Provides haversine distance computation and neighbor lookup
using county centroid coordinates.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def haversine_miles(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in miles between two points."""
    R = 3958.8  # Earth radius in miles
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def load_centroids(csv_path: str) -> pd.DataFrame:
    """
    Load county centroid coordinates.

    Args:
        csv_path: Path to CSV with columns fips_code, name, lng, lat

    Returns:
        DataFrame with columns fips (int), name, lng, lat
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'fips_code': 'fips'})
    # Ensure fips is integer (drop leading zeros)
    df['fips'] = df['fips'].astype(int)
    logger.info(f"Loaded centroids for {len(df)} counties from {csv_path}")
    return df


def get_neighbors(
    target_fips: int,
    candidate_fips: List[int],
    centroids_df: pd.DataFrame,
    max_k: Optional[int] = None,
    max_distance_miles: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """
    Get geographically nearest neighbors for a county.

    Args:
        target_fips: FIPS code of the target county
        candidate_fips: List of FIPS codes to consider as neighbors
        centroids_df: DataFrame with fips, lat, lng columns
        max_k: Maximum number of neighbors to return
        max_distance_miles: Maximum distance cutoff in miles

    Returns:
        List of (fips, distance_miles) tuples, sorted by distance
    """
    target_row = centroids_df[centroids_df['fips'] == target_fips]
    if len(target_row) == 0:
        logger.warning(f"County {target_fips} not found in centroids")
        return []

    lat0 = target_row.iloc[0]['lat']
    lon0 = target_row.iloc[0]['lng']

    # Filter centroids to candidates only
    candidates = centroids_df[
        (centroids_df['fips'].isin(candidate_fips))
        & (centroids_df['fips'] != target_fips)
    ].copy()

    if len(candidates) == 0:
        return []

    candidates['distance_miles'] = haversine_miles(
        lat0, lon0, candidates['lat'].values, candidates['lng'].values
    )
    candidates = candidates.sort_values('distance_miles')

    if max_distance_miles is not None:
        candidates = candidates[candidates['distance_miles'] <= max_distance_miles]

    if max_k is not None:
        candidates = candidates.head(max_k)

    return list(zip(candidates['fips'].values, candidates['distance_miles'].values))
