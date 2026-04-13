#!/usr/bin/env python3
"""
Aggregate experimental results from individual job outputs.
"""

import sys
from pathlib import Path
import pandas as pd
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import ResultsAggregator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_within_county_results(results_dir: Path):
    """
    Aggregate within-county experiment results.

    Args:
        results_dir: Directory containing individual county result files
    """
    logger.info("Aggregating within-county results...")

    # Find all county result files
    result_files = list(results_dir.glob("county_*_results.csv"))

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return

    logger.info(f"Found {len(result_files)} county result files")

    # Load and concatenate all results
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    if not all_results:
        logger.error("No results loaded successfully")
        return

    combined_results = pd.concat(all_results, ignore_index=True)
    logger.info(f"Combined {len(combined_results)} fold-level results")

    # Save and aggregate
    ResultsAggregator.save_aggregated_results(
        fold_results=combined_results,
        output_dir=str(results_dir),
        experiment_name='within_county'
    )

    logger.info("Within-county aggregation complete!")


def aggregate_cross_county_results(results_dir: Path):
    """
    Aggregate cross-county experiment results.

    Args:
        results_dir: Directory containing individual (county, iteration) result files
    """
    logger.info("Aggregating cross-county results...")

    # Find all result files
    result_files = list(results_dir.glob("county_*_iter_*_results.csv"))

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return

    logger.info(f"Found {len(result_files)} result files")

    # Load and concatenate all results
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    if not all_results:
        logger.error("No results loaded successfully")
        return

    combined_results = pd.concat(all_results, ignore_index=True)
    logger.info(f"Combined {len(combined_results)} iteration-level results")

    # Rename columns to match expected format
    combined_results = combined_results.rename(columns={'target_fips': 'fips'})

    # Save and aggregate
    ResultsAggregator.save_aggregated_results(
        fold_results=combined_results,
        output_dir=str(results_dir),
        experiment_name='cross_county'
    )

    logger.info("Cross-county aggregation complete!")


def print_summary(results_dir: Path, experiment_name: str):
    """
    Print summary statistics from aggregated results.

    Args:
        results_dir: Results directory
        experiment_name: Name of experiment
    """
    overall_file = results_dir / f"{experiment_name}_overall_aggregated.csv"

    if not overall_file.exists():
        logger.warning(f"Overall results file not found: {overall_file}")
        return

    df = pd.read_csv(overall_file)

    print("\n" + "="*80)
    print(f"SUMMARY: {experiment_name.upper()}")
    print("="*80)

    for _, row in df.iterrows():
        model = row['model']
        print(f"\n{model.upper()}:")
        print(f"  R² Score:  {row['r2']:.4f} ± {row['r2_std']:.4f}")
        print(f"  MAE:       {row['mae']:.2f} ± {row['mae_std']:.2f}")
        print(f"  RMSE:      {row['rmse']:.2f} ± {row['rmse_std']:.2f}")
        print(f"  N:         {int(row['r2_count'])}")

    # Compute comparison
    if len(df) == 2:
        tabpfn_row = df[df['model'] == 'tabpfn'].iloc[0]
        xgboost_row = df[df['model'] == 'xgboost'].iloc[0]

        r2_diff = tabpfn_row['r2'] - xgboost_row['r2']
        r2_rel = (r2_diff / abs(xgboost_row['r2'])) * 100

        mae_diff = xgboost_row['mae'] - tabpfn_row['mae']  # Lower is better
        mae_rel = (mae_diff / xgboost_row['mae']) * 100

        print(f"\nCOMPARISON (TabPFN vs XGBoost):")
        print(f"  R² difference:     {r2_diff:+.4f} ({r2_rel:+.2f}%)")
        print(f"  MAE improvement:   {mae_diff:+.2f} ({mae_rel:+.2f}%)")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experimental results'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['within_county', 'cross_county'],
        help='Which experiment to aggregate'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Results directory (default: inferred from experiment type)'
    )

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        project_root = Path(__file__).parent.parent.parent
        results_dir = project_root / 'results' / args.experiment

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Aggregate based on experiment type
    if args.experiment == 'within_county':
        aggregate_within_county_results(results_dir)
    elif args.experiment == 'cross_county':
        aggregate_cross_county_results(results_dir)

    # Print summary
    print_summary(results_dir, args.experiment)


if __name__ == '__main__':
    main()
