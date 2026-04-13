#!/usr/bin/env python
"""
Unified experiment runner CLI.

This is the main entry point for running experiments. It dispatches to the
appropriate experiment type based on the --experiment_type argument.

Usage:
    # Data scaling experiment (replaces cook_county_runner.py)
    python run_experiment.py \\
        --experiment_type data_scaling \\
        --config config/experiments/cook_county_with_preprocessing.yaml

    # Within-county CV (backward compatible)
    python run_experiment.py \\
        --experiment_type within_county \\
        --fips 1011 \\
        --bin_name small \\
        --k_folds 5 \\
        --config config/experiments/with_preprocessing.yaml

    # Future: In-context pooling
    python run_experiment.py \\
        --experiment_type in_context_pooling \\
        --target_fips 1011 \\
        --pool_sizes 0,5,10,20 \\
        --config config/experiments/pooling.yaml
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment_types import DataScalingExperiment

logger = logging.getLogger(__name__)


def load_config(config_path: str, base_config_path: str = None) -> dict:
    """
    Load experiment configuration.

    Args:
        config_path: Path to experiment config file
        base_config_path: Optional path to base config (for merging)

    Returns:
        Merged configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Optionally merge with base config (similar to existing runners)
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Deep merge base with experiment config
        from runners.within_county_runner import deep_merge
        config = deep_merge(base_config, config)

    return config


def setup_logging(config: dict, output_dir: Path):
    """
    Setup logging to file and console.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for log file
    """
    log_file = output_dir / 'experiment.log'
    log_level = config.get('logging', {}).get('level', 'INFO')

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Logging to: {log_file}")


def handle_output_dir_templating(config: dict, experiment_type: str) -> Path:
    """
    Handle output directory path templating.

    Args:
        config: Configuration dictionary
        experiment_type: Type of experiment

    Returns:
        Resolved output directory path
    """
    experiment_name = config.get('experiment', {}).get('name', 'default')
    output_dir = config['output']['results_dir']

    if '{experiment_type}' in output_dir or '{experiment_name}' in output_dir:
        output_dir = output_dir.format(
            experiment_type=experiment_type,
            experiment_name=experiment_name
        )
        config['output']['results_dir'] = output_dir
        logger.info(f"Output directory (after templating): {output_dir}")

    return Path(output_dir)


def run_data_scaling(config: dict, args: argparse.Namespace):
    """
    Run data scaling experiment.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'data_scaling'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    # Run experiment
    runner = DataScalingExperiment(config)
    results_df, calibration_data, predictions_data = runner.run_experiment()

    # Save results
    runner.save_results(results_df, output_dir, 'results.csv')

    # Save calibration data if available
    if calibration_data:
        runner.save_calibration_data(
            calibration_data,
            output_dir,
            'calibration.pkl'
        )

    # Save predictions if available
    if predictions_data:
        runner.save_predictions(
            predictions_data,
            output_dir,
            'predictions.parquet'
        )

    logger.info(f"All outputs saved to: {output_dir}")


def run_within_county(config: dict, args: argparse.Namespace):
    """
    Run within-county CV experiment (backward compatible).

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Import existing within_county_runner for backward compatibility
    from runners.within_county_runner import WithinCountyRunner

    experiment_type = 'within_county'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"FIPS: {args.fips}, Bin: {args.bin_name}, Folds: {args.k_folds}")
    logger.info("=" * 80)

    # Run experiment using existing runner
    runner = WithinCountyRunner(config)
    results_df, calibration_data, predictions_data = runner.run_county(
        fips=args.fips,
        bin_name=args.bin_name,
        k_folds=args.k_folds,
        n_repeats=config['experiment']['repetitions']
    )

    # Save results
    output_file = output_dir / f"county_{args.fips}_results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    # Save calibration data if available
    if calibration_data:
        import pickle
        calibration_file = output_dir / f"county_{args.fips}_calibration.pkl"
        cal_output = {
            'fips': args.fips,
            'experiment_name': config.get('experiment', {}).get('name'),
            'quantiles': config['calibration']['quantiles'],
            'folds': calibration_data
        }
        with open(calibration_file, 'wb') as f:
            pickle.dump(cal_output, f)
        logger.info(f"Calibration data saved to {calibration_file}")

    # Save predictions if available
    if predictions_data:
        pred_format = config.get('predictions', {}).get('predictions_format', 'parquet')
        if pred_format == 'parquet':
            import pandas as pd
            predictions_file = output_dir / f"county_{args.fips}_predictions.parquet"
            pred_df = pd.DataFrame(predictions_data)
            pred_df.to_parquet(predictions_file, index=False)
        else:
            import pickle
            predictions_file = output_dir / f"county_{args.fips}_predictions.pkl"
            pred_output = {
                'fips': args.fips,
                'experiment_name': config.get('experiment', {}).get('name'),
                'predictions': predictions_data
            }
            with open(predictions_file, 'wb') as f:
                pickle.dump(pred_output, f)
        logger.info(f"Predictions saved to {predictions_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for TabPFN data scarcity experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--experiment_type',
        type=str,
        required=True,
        choices=['data_scaling', 'within_county', 'cross_county', 'in_context_pooling'],
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )

    # Optional base config
    parser.add_argument(
        '--base_config',
        type=str,
        default='config/base_config.yaml',
        help='Path to base configuration file (default: config/base_config.yaml)'
    )

    # Within-county specific arguments
    parser.add_argument(
        '--fips',
        type=int,
        help='County FIPS code (for within_county experiments)'
    )
    parser.add_argument(
        '--bin_name',
        type=str,
        help='Size bin name (for within_county experiments)'
    )
    parser.add_argument(
        '--k_folds',
        type=int,
        help='Number of CV folds (for within_county experiments)'
    )

    # Output override
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Override output directory from config'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.base_config)

    # Override output dir if specified
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir

    # Dispatch to appropriate experiment handler
    if args.experiment_type == 'data_scaling':
        run_data_scaling(config, args)

    elif args.experiment_type == 'within_county':
        # Validate required arguments
        if not all([args.fips, args.bin_name, args.k_folds]):
            parser.error("--fips, --bin_name, and --k_folds are required for within_county experiments")
        run_within_county(config, args)

    elif args.experiment_type == 'cross_county':
        raise NotImplementedError("cross_county experiment type not yet refactored")

    elif args.experiment_type == 'in_context_pooling':
        raise NotImplementedError("in_context_pooling experiment type not yet implemented")

    else:
        parser.error(f"Unknown experiment type: {args.experiment_type}")


if __name__ == '__main__':
    main()
