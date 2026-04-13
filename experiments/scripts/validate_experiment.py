#!/usr/bin/env python
"""
Quick validation script to catch errors before running experiments.

This performs "smoke tests" - quick checks that imports work, configs are valid,
and basic setup is correct. Catches most common errors in seconds rather than
after waiting for SLURM jobs to start.

Usage:
    # Validate all experiment types
    python experiments/scripts/validate_experiment.py

    # Validate specific config
    python experiments/scripts/validate_experiment.py --config experiments/configs/data_scaling/cook_county_with_preprocessing.yaml

    # Validate all configs in a directory
    python experiments/scripts/validate_experiment.py --config-dir experiments/configs/data_scaling
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports() -> Tuple[bool, List[str]]:
    """Test that all required imports work."""
    errors = []

    print("Testing imports...")

    # Test core imports
    try:
        from src.data import CountyDataLoader
        print("  ✓ src.data.CountyDataLoader")
    except ImportError as e:
        errors.append(f"  ✗ src.data.CountyDataLoader: {e}")

    try:
        from src.data.preprocessing import Preprocess
        print("  ✓ src.data.preprocessing.Preprocess")
    except ImportError as e:
        errors.append(f"  ✗ src.data.preprocessing.Preprocess: {e}")

    try:
        from src.models import TabPFNModel, XGBoostModel
        print("  ✓ src.models (TabPFN, XGBoost)")
    except ImportError as e:
        errors.append(f"  ✗ src.models: {e}")

    # Test experiment types
    try:
        from experiments.experiment_types import (
            DataScalingExperiment,
            WithinCountyExperiment,
            CrossCountyExperiment
        )
        print("  ✓ experiments.experiment_types.DataScalingExperiment")
        print("  ✓ experiments.experiment_types.WithinCountyExperiment")
        print("  ✓ experiments.experiment_types.CrossCountyExperiment")
    except ImportError as e:
        errors.append(f"  ✗ experiments.experiment_types: {e}")

    return len(errors) == 0, errors


def validate_config(config_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single config file."""
    errors = []

    print(f"\nValidating config: {config_path.relative_to(PROJECT_ROOT)}")

    # Check file exists
    if not config_path.exists():
        errors.append(f"  ✗ Config file not found: {config_path}")
        return False, errors

    # Load YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("  ✓ YAML loads successfully")
    except yaml.YAMLError as e:
        errors.append(f"  ✗ Invalid YAML: {e}")
        return False, errors

    # Check required sections
    required_sections = ['experiment', 'data', 'models', 'output']
    for section in required_sections:
        if section not in config:
            errors.append(f"  ✗ Missing required section: '{section}'")
        else:
            print(f"  ✓ Has section: '{section}'")

    # Check experiment metadata
    if 'experiment' in config:
        exp_config = config['experiment']

        # Check for type field
        if 'type' not in exp_config:
            errors.append("  ✗ Missing 'experiment.type' field")
        else:
            exp_type = exp_config['type']
            valid_types = ['data_scaling', 'within_county', 'cross_county']
            if exp_type not in valid_types:
                errors.append(f"  ✗ Invalid experiment type: '{exp_type}' (must be one of {valid_types})")
            else:
                print(f"  ✓ Experiment type: '{exp_type}'")

        # Check for name
        if 'name' not in exp_config:
            errors.append("  ✗ Missing 'experiment.name' field")
        else:
            print(f"  ✓ Experiment name: '{exp_config['name']}'")

        # Check for random_seed
        if 'random_seed' not in exp_config:
            errors.append("  ✗ Missing 'experiment.random_seed' field")

    # Check models
    if 'models' in config:
        models = config['models']
        if not isinstance(models, list) or len(models) == 0:
            errors.append("  ✗ 'models' must be a non-empty list")
        else:
            print(f"  ✓ Has {len(models)} model(s) configured")

    # Try to instantiate experiment runner (dry run)
    if len(errors) == 0 and 'experiment' in config and 'type' in config['experiment']:
        try:
            exp_type = config['experiment']['type']

            if exp_type == 'data_scaling':
                from experiments.experiment_types import DataScalingExperiment
                runner = DataScalingExperiment(config)
                print(f"  ✓ DataScalingExperiment instantiated successfully")

            elif exp_type == 'within_county':
                from experiments.experiment_types import WithinCountyExperiment
                runner = WithinCountyExperiment(config)
                print(f"  ✓ WithinCountyExperiment instantiated successfully")

            elif exp_type == 'cross_county':
                from experiments.experiment_types import CrossCountyExperiment
                runner = CrossCountyExperiment(config)
                print(f"  ✓ CrossCountyExperiment instantiated successfully")

        except Exception as e:
            error_msg = str(e)
            # Data directory not found is expected when validating locally
            if 'not found' in error_msg.lower() and ('directory' in error_msg.lower() or 'file' in error_msg.lower()):
                print(f"  ⚠ Warning: Data path not found (expected if validating locally): {error_msg}")
            else:
                errors.append(f"  ✗ Failed to instantiate experiment: {e}")

    return len(errors) == 0, errors


def validate_all_configs(config_dir: Path) -> Tuple[int, int]:
    """Validate all configs in a directory."""
    config_files = list(config_dir.glob("**/*.yaml"))

    if not config_files:
        print(f"No .yaml files found in {config_dir}")
        return 0, 0

    passed = 0
    failed = 0

    for config_file in config_files:
        success, errors = validate_config(config_file)
        if success:
            passed += 1
        else:
            failed += 1
            for error in errors:
                print(error)

    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description='Validate experiment setup before running'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Specific config file to validate'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory of configs to validate'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT VALIDATION")
    print("=" * 80)

    # Test imports first (always)
    import_success, import_errors = test_imports()
    if not import_success:
        print("\n" + "=" * 80)
        print("IMPORT ERRORS FOUND:")
        for error in import_errors:
            print(error)
        print("=" * 80)
        print("\n❌ Import validation FAILED - fix import errors before validating configs")
        sys.exit(1)

    print("\n✅ All imports successful")

    # Validate configs
    if args.config:
        config_path = PROJECT_ROOT / args.config
        success, errors = validate_config(config_path)

        print("\n" + "=" * 80)
        if success:
            print("✅ CONFIG VALID - Ready to run!")
        else:
            print("❌ CONFIG VALIDATION FAILED:")
            for error in errors:
                print(error)
            print("=" * 80)
            sys.exit(1)

    elif args.config_dir:
        config_dir = PROJECT_ROOT / args.config_dir
        passed, failed = validate_all_configs(config_dir)

        print("\n" + "=" * 80)
        print(f"Validated {passed + failed} configs: {passed} passed, {failed} failed")
        if failed > 0:
            print("❌ Some configs have errors")
            sys.exit(1)
        else:
            print("✅ All configs valid")

    else:
        # Validate all configs
        config_base = PROJECT_ROOT / "experiments" / "configs"

        for subdir in ['data_scaling', 'within_county', 'cross_county', 'calibration']:
            config_dir = config_base / subdir
            if config_dir.exists():
                print(f"\n{'=' * 80}")
                print(f"Validating {subdir} configs...")
                print('=' * 80)
                passed, failed = validate_all_configs(config_dir)

                if failed > 0:
                    print(f"\n❌ {subdir}: {failed} config(s) failed")
                    sys.exit(1)
                else:
                    print(f"\n✅ {subdir}: All {passed} config(s) valid")

    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE - All checks passed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
