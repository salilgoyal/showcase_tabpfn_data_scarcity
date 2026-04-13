"""
Generate train set from cleaned pooled data using a pre-generated test set.

This script creates a training set based on various strategies (test county history,
external counties, mixed, etc.). The train set is saved to disk and can be reused
across multiple experiment runs to ensure reproducibility.

Usage:
    python preprocessing/scripts/generate_train_set.py \
        --config experiments/configs/train_sets/train_v2.yaml \
        --test_split_dir experiments/splits/test_v1/ \
        --data_path /scratch/.../cleaned_datasets/v1_no_onehot/data.parquet \
        --output_dir experiments/splits/test_v1/train_v2/

Output:
    experiments/splits/test_v1/train_v2/
    ├── train_indices.npy          # Row indices for training
    ├── source_breakdown.json      # Samples from each source
    ├── county_distribution.json   # Samples per county
    ├── metadata.json              # Train set metadata
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
    create_train_set,
    save_train_set_result,
    load_train_set_config,
    load_test_set_result
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_summary_report(train_result, test_result, output_dir: str):
    """Create human-readable summary report."""
    output_path = Path(output_dir)

    with open(output_path / "summary_report.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAIN SET SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        f.write("Version: {}\n".format(train_result.metadata.get('version', 'unknown')))
        f.write("Description: {}\n".format(train_result.metadata.get('description', '')))
        f.write("Random seed: {}\n".format(train_result.metadata.get('random_seed', 'N/A')))
        f.write("\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total train samples: {train_result.metadata['n_train_samples']:,}\n")
        f.write(f"Counties used: {train_result.metadata['n_counties_used']}\n")
        f.write("\n")

        # Source breakdown
        f.write("SOURCE BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        for source, count in train_result.source_breakdown.items():
            percentage = 100 * count / train_result.metadata['n_train_samples']
            f.write(f"  {source:30s}: {count:8,} ({percentage:5.1f}%)\n")
        f.write("\n")

        # Top counties by sample count
        f.write("TOP 20 COUNTIES BY SAMPLE COUNT\n")
        f.write("-" * 80 + "\n")
        sorted_counties = sorted(
            train_result.county_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for fips, count in sorted_counties:
            # Check if in test set
            in_test = "(test county)" if fips in test_result.test_counties else ""
            f.write(f"  {fips:5d}: {count:6,} samples {in_test}\n")

        if len(train_result.county_distribution) > 20:
            f.write(f"  ... and {len(train_result.county_distribution) - 20} more counties\n")

        f.write("\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Summary report saved to {output_path / 'summary_report.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train set from cleaned pooled data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to train set configuration YAML file'
    )

    parser.add_argument(
        '--test_split_dir',
        type=str,
        required=True,
        help='Directory containing pre-generated test set'
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
        help='Directory to save train set files'
    )

    parser.add_argument(
        '--fips_column',
        type=str,
        default='fips',
        help='Name of FIPS column (default: fips)'
    )

    parser.add_argument(
        '--log_transformed_target',
        action='store_true',
        default=True,
        help='Target variable is log-transformed (default: True)'
    )

    parser.add_argument(
        '--no_log_transformed_target',
        action='store_false',
        dest='log_transformed_target',
        help='Target variable is NOT log-transformed'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("GENERATING TRAIN SET")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Test split: {args.test_split_dir}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Load test set
    logger.info("Loading pre-generated test set...")
    test_result = load_test_set_result(args.test_split_dir)

    # Load configuration
    logger.info("\nLoading train set configuration...")
    config = load_train_set_config(args.config)
    logger.info(f"Train set version: {config.get('version', 'unknown')}")
    logger.info(f"Description: {config.get('description', '')}")

    # Load data
    logger.info(f"\nLoading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Get random seed
    random_seed = config.get('sampling', {}).get('random_seed', 42)
    if 'random_seed' not in config.get('sampling', {}):
        # Use test set's random seed if not specified
        random_seed = test_result.metadata.get('random_seed', 42)

    # Create train set
    logger.info("\nCreating train set...")
    train_result = create_train_set(
        df=df,
        config=config,
        test_result=test_result,
        fips_column=args.fips_column,
        random_seed=random_seed
    )

    # Save train set
    logger.info(f"\nSaving train set to {args.output_dir}...")
    save_train_set_result(train_result, args.output_dir, df=df, log_transformed_target=args.log_transformed_target)

    # Create summary report
    logger.info("\nCreating summary report...")
    create_summary_report(train_result, test_result, args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TRAIN SET GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Train samples: {len(train_result.train_indices):,}")
    logger.info(f"Counties used: {len(train_result.county_distribution)}")
    logger.info(f"Source breakdown: {train_result.source_breakdown}")
    logger.info(f"\nFiles saved to: {args.output_dir}")
    logger.info(f"Review summary: {Path(args.output_dir) / 'summary_report.txt'}")


if __name__ == '__main__':
    main()
