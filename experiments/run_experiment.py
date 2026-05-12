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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_types import (
    GeoPoolingExperiment,
    GlobalFinetuningExperiment,
    SingleCountyScalingExperiment,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load experiment configuration with variable substitution.

    Args:
        config_path: Path to experiment config file

    Returns:
        Configuration dictionary with template variables resolved

    Note:
        Configs must be complete - no inheritance or merging.
        This ensures errors are caught early.

        Supports template variables like {train_version}, {experiment_name}, etc.
        defined in the experiment section.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply template variable substitution
    config = apply_template_substitution(config)

    return config


def apply_template_substitution(config: dict) -> dict:
    """
    Apply template variable substitution throughout config.

    Supports variables defined in the experiment section, e.g.:
    - {train_version}
    - {experiment_name}
    - {experiment_type}

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with all template variables resolved
    """
    import json

    # Extract template variables from experiment section
    experiment_config = config.get('experiment', {})
    template_vars = {}

    # Common template variables
    for key in ['train_version', 'test_version', 'experiment_type', 'experiment_name']:
        if key in experiment_config:
            template_vars[key] = experiment_config[key]

    if not template_vars:
        return config

    # Convert config to JSON string for easy substitution
    config_str = json.dumps(config)

    # Apply substitutions
    for var_name, var_value in template_vars.items():
        config_str = config_str.replace(f'{{{var_name}}}', str(var_value))

    # Convert back to dict
    config = json.loads(config_str)

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
    Run within-county CV experiment.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'within_county'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"FIPS: {args.fips}, Bin: {args.bin_name}, Folds: {args.k_folds}")
    logger.info("=" * 80)

    # Run experiment
    # TODO: WithinCountyExperiment needs migration to CleanedDataLoader
    raise NotImplementedError(
        "WithinCountyExperiment is not yet implemented. "
        "It needs to be migrated to use CleanedDataLoader."
    )
    # runner = WithinCountyExperiment(config)
    # results_df, calibration_data, predictions_data = runner.run_county(
    #     fips=args.fips,
    #     bin_name=args.bin_name,
    #     k_folds=args.k_folds,
    #     n_repeats=config['experiment']['repetitions']
    # )

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


def run_cross_county(config: dict, args: argparse.Namespace):
    """
    Run cross-county generalization experiment.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'cross_county'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    # Override county list if provided via command line
    if args.county_list:
        county_fips_list = [int(x.strip()) for x in args.county_list.split(',')]
        config['county_fips_list'] = county_fips_list

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    if 'county_fips_list' in config:
        logger.info(f"Counties: {config['county_fips_list']}")
    elif 'sampling' in config:
        logger.info(f"Sampling Strategy: {config['sampling']['strategy']}")
        logger.info(f"Sampling Parameters: {config['sampling'].get('parameters', {})}")
    else:
        logger.info("Counties: Determined dynamically")
    logger.info("=" * 80)

    # Run experiment
    runner = CrossCountyExperiment(config)
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


def run_finetuning(config: dict, args: argparse.Namespace):
    """
    Run fine-tuning experiment.

    This experiment fine-tunes TabPFN on large-scale pooled county data
    and compares performance with XGBoost.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'finetuning'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also create checkpoint directory if specified
    checkpoint_dir = config.get('output', {}).get('checkpoint_dir')
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    # Log key configuration
    split_config = config.get('split', {})
    train_size = split_config.get('train_size', 'N/A')
    val_size = split_config.get('val_size', 'N/A')
    test_size = split_config.get('test_size', 'N/A')
    logger.info(f"Target train size: {train_size:,}" if isinstance(train_size, int) else f"Target train size: {train_size}")
    logger.info(f"Target val size: {val_size:,}" if isinstance(val_size, int) else f"Target val size: {val_size}")
    logger.info(f"Target test size: {test_size:,}" if isinstance(test_size, int) else f"Target test size: {test_size}")

    ft_config = config.get('finetuning', {})
    logger.info(f"Learning rate: {ft_config.get('learning_rate', 'N/A')}")
    logger.info(f"Max epochs: {ft_config.get('max_epochs', 'N/A')}")
    logger.info(f"Batch size: {ft_config.get('batch_size', 'N/A')}")

    # Run experiment
    runner = FinetuningExperiment(config)
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

    logger.info(f"All outputs saved to: {output_dir}")


def run_per_county_scaling(config: dict, args: argparse.Namespace):
    """
    Run per-county scaling experiment.

    Trains separate models per county with varying training set sizes
    to build per-county learning curves.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'per_county_scaling'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    # Pass single county FIPS if provided (SLURM array mode)
    if args.county_fips:
        config['_single_county_fips'] = args.county_fips

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    if args.county_fips:
        logger.info(f"Single county mode: FIPS {args.county_fips}")
    logger.info("=" * 80)

    # Run experiment
    runner = PerCountyScalingExperiment(config)
    results_df, calibration_data, predictions_data = runner.run_experiment()

    logger.info(f"All outputs saved to: {output_dir}")


def run_geo_pooling(config: dict, args: argparse.Namespace):
    """
    Run geographic pooling experiment.

    Trains per-county models using local + geographically nearby county data.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'geo_pooling'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    # Pass array job parameters if provided
    if args.county_index is not None:
        config['_county_index'] = args.county_index
    if args.county_chunk_size is not None:
        config['_county_chunk_size'] = args.county_chunk_size
    if args.n_chunks is not None:
        config['_n_chunks'] = args.n_chunks

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    if args.county_index is not None:
        logger.info(f"Array job: chunk {args.county_index}/{args.n_chunks}, "
                    f"chunk_size={args.county_chunk_size}")
    logger.info("=" * 80)

    # Run experiment
    runner = GeoPoolingExperiment(config)
    results_df, calibration_data, predictions_data = runner.run_experiment()

    logger.info(f"All outputs saved to: {output_dir}")


def run_global_finetuning(config: dict, args: argparse.Namespace):
    """
    Run global finetuning experiment.

    Finetunes TabPFN on a large pooled dataset and saves the checkpoint.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'global_finetuning'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    # Run experiment
    runner = GlobalFinetuningExperiment(config)
    results_df, _, _ = runner.run_experiment()

    logger.info(f"All outputs saved to: {output_dir}")


def run_single_county_scaling(config: dict, args: argparse.Namespace):
    """
    Run single-county data scaling experiment.

    Trains models on varying training set sizes for a single county
    to build learning curves comparing model performance under data scarcity.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_type = 'single_county_scaling'
    output_dir = handle_output_dir_templating(config, experiment_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    county_fips = config.get('single_county_scaling', {}).get('county_fips', 'N/A')
    logger.info("=" * 80)
    logger.info(f"Experiment Type: {experiment_type}")
    logger.info(f"Config: {args.config}")
    logger.info(f"County FIPS: {county_fips}")
    logger.info("=" * 80)

    # Run experiment
    runner = SingleCountyScalingExperiment(config)
    results_df, calibration_data, predictions_data = runner.run_experiment()

    logger.info(f"All outputs saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for TabPFN data scarcity experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--experiment_type',
        type=str,
        required=False,
        choices=['data_scaling', 'within_county', 'cross_county', 'finetuning', 'per_county_scaling', 'geo_pooling', 'global_finetuning', 'in_context_pooling', 'single_county_scaling'],
        help='Type of experiment to run (optional - reads from config if not specified)'
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

    # Cross-county specific arguments (optional - can also be in config)
    parser.add_argument(
        '--county_list',
        type=str,
        help='Comma-separated list of county FIPS codes (for cross_county experiments)'
    )

    # Per-county scaling specific arguments
    parser.add_argument(
        '--county_fips',
        type=int,
        help='Single county FIPS code for SLURM array mode (for per_county_scaling experiments)'
    )

    # Geo pooling specific arguments (SLURM array job)
    parser.add_argument(
        '--county_index',
        type=int,
        default=None,
        help='County chunk index for SLURM array mode (for geo_pooling experiments)'
    )
    parser.add_argument(
        '--county_chunk_size',
        type=int,
        default=None,
        help='Number of counties per chunk (for geo_pooling array jobs)'
    )
    parser.add_argument(
        '--n_chunks',
        type=int,
        default=None,
        help='Total number of chunks; chunk_size is computed automatically (for geo_pooling array jobs)'
    )

    # Output override
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Override output directory from config'
    )

    # Model toggle override
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated model names to enable, overriding the config models section. '
             'E.g., --models tabpfn,tabicl or --models tabpfn_v2.5,xgboost'
    )

    # Force overwrite
    parser.add_argument(
        '--overwrite_existing_results',
        action='store_true',
        default=False,
        help='Ignore checkpoint and re-run all enabled models from scratch, overwriting existing results.'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine experiment type (from CLI or config)
    experiment_type = args.experiment_type
    config_type = config.get('experiment', {}).get('type')

    if experiment_type is None:
        # Read from config
        if config_type is None:
            parser.error("Experiment type not specified. Provide --experiment_type or add 'type' field to config.")
        experiment_type = config_type
        logger.info(f"Experiment type from config: {experiment_type}")
    else:
        # Validate CLI matches config if both provided
        if config_type and experiment_type != config_type:
            parser.error(
                f"Experiment type mismatch: CLI specifies '{experiment_type}' "
                f"but config has '{config_type}'"
            )

    # Override output dir if specified
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir

    # Override enabled models if specified
    if args.models:
        enabled_names = [m.strip() for m in args.models.split(',')]
        config['models'] = [{'name': name, 'enabled': True} for name in enabled_names]
        logger.info(f"Model override from --models flag: {enabled_names}")

    # Override checkpoint resume if --overwrite_existing_results
    if args.overwrite_existing_results:
        config.setdefault('checkpointing', {})['resume'] = False
        logger.info("--overwrite_existing_results: checkpoint resume disabled, will re-run from scratch")

    # Dispatch to appropriate experiment handler
    if experiment_type == 'data_scaling':
        run_data_scaling(config, args)

    elif experiment_type == 'within_county':
        # Validate required arguments
        if not all([args.fips, args.bin_name, args.k_folds]):
            parser.error("--fips, --bin_name, and --k_folds are required for within_county experiments")
        run_within_county(config, args)

    elif experiment_type == 'cross_county':
        run_cross_county(config, args)

    elif experiment_type == 'finetuning':
        run_finetuning(config, args)

    elif experiment_type == 'per_county_scaling':
        run_per_county_scaling(config, args)

    elif experiment_type == 'geo_pooling':
        run_geo_pooling(config, args)

    elif experiment_type == 'global_finetuning':
        run_global_finetuning(config, args)

    elif experiment_type == 'single_county_scaling':
        run_single_county_scaling(config, args)

    elif experiment_type == 'in_context_pooling':
        raise NotImplementedError("in_context_pooling experiment type not yet implemented")

    else:
        parser.error(f"Unknown experiment type: {experiment_type}")


if __name__ == '__main__':
    main()
