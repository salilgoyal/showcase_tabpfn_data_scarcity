"""
Generate baseline predictions for a test/train split combination.

This pre-computes baseline predictions once so they can be reused across
multiple experiments without recomputing.

Usage:
    python experiments/scripts/generate_baseline_predictions.py \
        --test_split_dir /scratch/.../test_v1/ \
        --train_split_dir /scratch/.../test_v1/train_v5/ \
        --output_file /scratch/.../test_v1/train_v5/baseline_predictions.npz

Output:
    baseline_predictions.npz containing:
        - predictions: baseline predictions (in same space as y_test)
        - test_indices: indices of test samples
        - adjustment_ratio: the ratio used
        - log_transformed: whether predictions are log-transformed
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models import BaselineModel, load_baseline_data
from src.data import CleanedDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute baseline predictions for a test/train split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--test_split_dir',
        type=str,
        required=True,
        help='Directory containing test set files'
    )

    parser.add_argument(
        '--train_split_dir',
        type=str,
        required=True,
        help='Directory containing train set files'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to cleaned data (for loading y_test)'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file path (default: <train_split_dir>/baseline_predictions.npz)'
    )

    args = parser.parse_args()

    # Default output file
    if args.output_file is None:
        args.output_file = str(Path(args.train_split_dir) / "baseline_predictions.npz")

    logger.info("=" * 80)
    logger.info("GENERATING BASELINE PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Test split: {args.test_split_dir}")
    logger.info(f"Train split: {args.train_split_dir}")
    logger.info(f"Output: {args.output_file}")
    logger.info("")

    # Load data
    logger.info("Loading data...")
    data_loader = CleanedDataLoader(args.data_path)
    df = data_loader.load()
    log_transformed = data_loader.is_target_log_transformed()
    logger.info(f"Target log-transformed: {log_transformed}")

    # Load baseline data
    logger.info("Loading baseline data...")
    test_baseline, train_baseline, adjustment_ratio = load_baseline_data(
        test_set_dir=args.test_split_dir,
        train_set_dir=args.train_split_dir
    )
    logger.info(f"Adjustment ratio: {adjustment_ratio:.4f}")

    # Load test indices
    test_indices = np.load(Path(args.test_split_dir) / "test_indices.npy")
    logger.info(f"Test samples: {len(test_indices):,}")

    # Load y_test
    target_column = "SALE_AMOUNT"  # Default
    y_test = df.iloc[test_indices][target_column]

    # Create and fit baseline model
    logger.info("Creating baseline model...")
    model = BaselineModel(
        baseline_values_train=train_baseline,
        baseline_values_test=test_baseline,
        adjustment_ratio=adjustment_ratio,
        log_transformed=log_transformed
    )

    # Fit (uses pre-computed ratio)
    import pandas as pd
    model.fit(pd.DataFrame(), y_test)

    # Predict
    logger.info("Generating predictions...")
    predictions = model.predict(pd.DataFrame())

    # Save predictions
    logger.info(f"Saving predictions to {args.output_file}...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        predictions=predictions,
        test_indices=test_indices,
        adjustment_ratio=adjustment_ratio,
        log_transformed=log_transformed
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE PREDICTIONS GENERATED")
    logger.info("=" * 80)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Adjustment ratio: {adjustment_ratio:.4f}")
    logger.info(f"Log-transformed: {log_transformed}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("")
    logger.info("To use in experiments, the experiment code will check for this file")
    logger.info("and load pre-computed predictions if available.")


if __name__ == "__main__":
    main()
