#!/usr/bin/env python3
"""
Create registry of small counties and save metadata with feature information.
"""

import sys
from pathlib import Path
import yaml
import logging
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CountyRegistry, CountyDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_feature_consistency(config, small_counties_df):
    """
    Check feature consistency across small counties and add to metadata.

    Args:
        config: Configuration dictionary
        small_counties_df: DataFrame with small counties

    Returns:
        DataFrame with added feature information
    """
    logger.info("\nChecking feature consistency across small counties...")

    loader = CountyDataLoader(
        county_csvs_dir=config['data']['county_csvs_dir'],
        target_column=config['data']['target_column']
    )

    # Get features for each county
    feature_info = []
    all_feature_sets = []

    for _, row in tqdm(small_counties_df.iterrows(), total=len(small_counties_df), desc="Loading counties"):
        fips = row['fips']
        try:
            df = loader.load_county(fips, drop_missing_target=False)
            features = set(df.columns)
            all_feature_sets.append(features)

            feature_info.append({
                'fips': fips,
                'num_features': len(features),
                'feature_list': ','.join(sorted(features))
            })
        except Exception as e:
            logger.warning(f"Could not load county {fips}: {e}")
            feature_info.append({
                'fips': fips,
                'num_features': 0,
                'feature_list': ''
            })

    # Analyze feature consistency
    if all_feature_sets:
        common_features = set.intersection(*all_feature_sets)
        all_unique_features = set.union(*all_feature_sets)

        logger.info(f"\nFeature Consistency Analysis:")
        logger.info(f"  Common features across all counties: {len(common_features)}")
        logger.info(f"  Total unique features: {len(all_unique_features)}")

        if len(common_features) == len(all_unique_features):
            logger.info(f"  ✓ All counties have identical features!")
        else:
            logger.warning(f"  ⚠ Feature sets differ across counties!")
            missing_count = len(all_unique_features) - len(common_features)
            logger.warning(f"  {missing_count} features are not present in all counties")

            # Find counties with different features
            inconsistent_counties = []
            for features in all_feature_sets:
                if features != common_features:
                    missing = all_unique_features - features
                    extra = features - common_features
                    if missing or extra:
                        inconsistent_counties.append({
                            'missing': len(missing),
                            'extra': len(extra)
                        })

            if inconsistent_counties:
                logger.warning(f"  {len(inconsistent_counties)} counties have different feature sets")

    # Merge feature info with county metadata
    feature_df = pd.DataFrame(feature_info)
    enhanced_df = small_counties_df.merge(feature_df, on='fips', how='left')

    return enhanced_df


def main():
    # Load config
    config_dir = Path(__file__).parent.parent / 'config'
    with open(config_dir / 'base_config.yaml') as f:
        config = yaml.safe_load(f)

    # Print data source information
    print("\n" + "="*80)
    print("DATA SOURCE INFORMATION")
    print("="*80)
    print(f"County CSV files: {config['data']['county_csvs_dir']}")
    print(f"Metadata file: {config['data']['county_metadata_file']}")
    print(f"Target variable: {config['data']['target_column']}")
    print()

    # Create registry
    registry = CountyRegistry(
        metadata_file=config['data']['county_metadata_file'],
        county_bins=config['county_bins']
    )

    # Get all small counties
    small_counties = registry.get_all_small_counties()

    # Check feature consistency and add feature info
    enhanced_metadata = check_feature_consistency(config, small_counties)

    # Save enhanced metadata
    output_file = Path(config['data']['county_metadata_file']).parent / 'small_county_metadata.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enhanced_metadata.to_csv(output_file, index=False)

    logger.info(f"\nRegistry created successfully!")
    logger.info(f"Total small counties: {len(enhanced_metadata)}")
    logger.info(f"Enhanced metadata saved to: {output_file}")

    # Print summary by bin
    print("\n" + "="*80)
    print("SMALL COUNTY REGISTRY SUMMARY")
    print("="*80)
    for bin_name in enhanced_metadata['bin_name'].unique():
        bin_df = enhanced_metadata[enhanced_metadata['bin_name'] == bin_name]
        k_folds = bin_df['k_folds'].iloc[0]
        print(f"\nBin: {bin_name}")
        print(f"  Counties: {len(bin_df)}")
        print(f"  Size range: {bin_df['row_count'].min()} - {bin_df['row_count'].max()}")
        print(f"  K-folds: {k_folds}")
        print(f"  Feature count range: {bin_df['num_features'].min()} - {bin_df['num_features'].max()}")
        print(f"  Sample FIPS codes: {sorted(bin_df['fips'].tolist())[:10]}{'...' if len(bin_df) > 10 else ''}")

    print("\n" + "="*80)
    print("Metadata columns:")
    print(f"  {', '.join(enhanced_metadata.columns)}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
