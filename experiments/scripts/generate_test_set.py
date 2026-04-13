"""
Generate test set from cleaned pooled data.

This script creates a fixed test set based on size-stratified county selection
with temporal splits. The test set is saved to disk and can be reused across
multiple experiments to ensure reproducibility.

Usage:
    python preprocessing/scripts/generate_test_set.py \
        --config experiments/configs/test_sets/test_v1.yaml \
        --data_path /scratch/.../cleaned_datasets/v1_no_onehot/data.parquet \
        --output_dir experiments/splits/test_v1/

Output:
    experiments/splits/test_v1/
    ├── test_indices.npy           # Row indices for test
    ├── train_pool_indices.npy     # Row indices available for training
    ├── test_counties.json         # List of test county FIPS
    ├── county_info.json           # Per-county statistics
    ├── size_buckets.json          # Counties in each size bucket
    ├── metadata.json              # Split metadata
    └── summary_report.txt         # Human-readable summary
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.split_strategies import (
    create_test_set,
    save_test_set_result,
    load_test_set_config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_summary_report(test_result, output_dir: str):
    """Create human-readable summary report."""
    output_path = Path(output_dir)

    with open(output_path / "summary_report.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST SET SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        f.write("Version: {}\n".format(test_result.metadata.get('version', 'unknown')))
        f.write("Description: {}\n".format(test_result.metadata.get('description', '')))
        f.write("Random seed: {}\n".format(test_result.metadata.get('random_seed', 'N/A')))
        f.write("\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total test counties: {test_result.metadata['n_test_counties']}\n")
        f.write(f"Total test samples: {test_result.metadata['n_test_samples']:,}\n")
        f.write(f"Total train pool samples: {test_result.metadata['n_train_pool_samples']:,}\n")
        split_method = test_result.metadata.get('split_method', 'temporal')
        if split_method == 'random':
            f.write(f"Split method: random (fraction={test_result.metadata.get('test_fraction', 0.2)}, "
                    f"split_seed={test_result.metadata.get('split_seed', 'N/A')})\n")
        else:
            f.write(f"Split method: temporal (test_percentile={test_result.metadata.get('test_percentile', 'N/A')}%)\n")
        f.write("\n")

        # Size buckets
        f.write("SIZE BUCKETS\n")
        f.write("-" * 80 + "\n")
        for bucket_name, counties in test_result.size_buckets.items():
            f.write(f"\n{bucket_name.upper()}: {len(counties)} counties\n")
            for fips in sorted(counties):
                info = test_result.county_info[fips]
                f.write(f"  {fips:5d}: {info['total_rows']:6d} rows "
                       f"({info['test_rows']:5d} test, {info['train_pool_rows']:5d} train pool)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Summary report saved to {output_path / 'summary_report.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test set from cleaned pooled data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to test set configuration YAML file'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to cleaned pooled data (parquet file)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save test set files'
    )

    parser.add_argument(
        '--fips_column',
        type=str,
        default='fips',
        help='Name of FIPS column (default: fips)'
    )

    parser.add_argument(
        '--date_column',
        type=str,
        default='sale_date',
        help='Name of date column (default: sale_date)'
    )

    parser.add_argument(
        '--split_seed',
        type=int,
        default=None,
        help='Random seed for within-county split (overrides county_selection.random_seed; '
             'only used when random_split.enabled is true in config)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("GENERATING TEST SET")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    if args.split_seed is not None:
        logger.info(f"Split seed: {args.split_seed}")
    logger.info("")

    # Load configuration
    logger.info("Loading configuration...")
    config = load_test_set_config(args.config)
    logger.info(f"Test set version: {config.get('version', 'unknown')}")
    logger.info(f"Description: {config.get('description', '')}")

    # Load data
    logger.info(f"\nLoading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Get random seed
    random_seed = config.get('county_selection', {}).get('random_seed', 42)

    # Create test set
    logger.info("\nCreating test set...")
    test_result = create_test_set(
        df=df,
        config=config,
        fips_column=args.fips_column,
        date_column=args.date_column,
        random_seed=random_seed,
        split_seed=args.split_seed
    )

    # Save test set
    logger.info(f"\nSaving test set to {args.output_dir}...")
    save_test_set_result(test_result, args.output_dir, df=df)

    # Create summary report
    logger.info("\nCreating summary report...")
    create_summary_report(test_result, args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TEST SET GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Test counties: {len(test_result.test_counties)}")
    logger.info(f"Test samples: {len(test_result.test_indices):,}")
    logger.info(f"Train pool samples: {len(test_result.train_pool_indices):,}")
    logger.info(f"\nFiles saved to: {args.output_dir}")
    logger.info(f"Review summary: {Path(args.output_dir) / 'summary_report.txt'}")


if __name__ == '__main__':
    main()
