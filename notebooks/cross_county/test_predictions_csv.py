#!/usr/bin/env python
"""
Quick test script to verify test predictions CSV generation works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_predictions_csv():
    """Test that the predictions CSV is generated correctly."""

    # Load test config (use main config but override iterations to 1 for testing)
    config_path = Path(__file__).parent.parent.parent / 'experiments' / 'configs' / 'cross_county' / 'small_in_context_10k.yaml'

    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override for faster testing
    config['sampling']['parameters']['n_small_counties'] = 5
    config['sampling']['parameters']['target_train_size'] = 500
    config['sampling']['parameters']['target_test_size'] = 500
    config['models'] = [{'name': 'xgboost', 'enabled': True}]  # Only XGBoost for speed
    config['xgboost']['optuna_trials'] = 2  # Fewer trials
    config['output']['results_dir'] = '/tmp/test_predictions_csv/'

    # Import experiment runner
    from experiments.experiment_types.cross_county import CrossCountyExperiment

    # Create runner
    logger.info("Creating experiment runner...")
    runner = CrossCountyExperiment(config)

    # Load counties
    logger.info("Loading counties...")
    county_data = runner.load_all_counties()

    if len(county_data) == 0:
        logger.error("No counties loaded!")
        return False

    logger.info(f"Loaded {len(county_data)} counties")

    # Run a single iteration
    logger.info("Running single iteration...")
    results, cal_data, pred_data = runner.run_single_iteration(
        county_data=county_data,
        target_fips=None,  # Use sampler
        iteration=0
    )

    # Check results
    logger.info(f"Got {len(results)} results")
    for result in results:
        logger.info(f"  {result['model']}: R2={result.get('r2', 'N/A'):.4f}")

    # Check if CSV was created
    output_dir = Path(config['output']['results_dir'])
    csv_file = output_dir / 'test_predictions_iter0.csv'

    if csv_file.exists():
        logger.info(f"✓ CSV file created: {csv_file}")

        # Read and inspect
        import pandas as pd
        df = pd.read_csv(csv_file)
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  First few rows:")
        logger.info(df.head())

        # Check required columns
        required_cols = ['fips', 'y_true']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"  ✗ Missing required column: {col}")
                return False

        # Check for model prediction columns
        model_cols = [col for col in df.columns if col.endswith('_pred')]
        logger.info(f"  Model prediction columns: {model_cols}")

        if len(model_cols) == 0:
            logger.error("  ✗ No model prediction columns found!")
            return False

        logger.info("✓ All checks passed!")
        return True
    else:
        logger.error(f"✗ CSV file not created: {csv_file}")
        return False

if __name__ == '__main__':
    success = test_predictions_csv()
    sys.exit(0 if success else 1)
