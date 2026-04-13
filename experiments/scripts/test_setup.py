#!/usr/bin/env python3
"""
Test script to verify the experiment setup.
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("Testing Experiment Setup")
print("="*80)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.data import CountyDataLoader
    from src.models import TabPFNModel, XGBoostModel
    from src.evaluation import compute_metrics
    from experiments.experiment_types import DataScalingExperiment
    print("   ✓ All modules imported successfully")
    print("   Note: CountyRegistry, RepeatedKFoldSplitter, ResultsAggregator not implemented")
    print("   Note: WithinCountyRunner, CrossCountyRunner being migrated")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Load configs
print("\n2. Testing configuration files...")
try:
    config_dir = Path(__file__).parent.parent / 'configs'
    with open(config_dir / 'base_config.yaml') as f:
        base_config = yaml.safe_load(f)
    print("   ✓ Base config loaded successfully")
    print(f"   Config directory: {config_dir}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Check data paths
print("\n3. Testing data paths...")
try:
    county_csvs_dir = Path(base_config['data']['county_csvs_dir'])
    metadata_file = Path(base_config['data']['county_metadata_file'])

    if not county_csvs_dir.exists():
        print(f"   ⚠ County CSVs directory not found: {county_csvs_dir}")
    else:
        print(f"   ✓ County CSVs directory exists: {county_csvs_dir}")

    if not metadata_file.exists():
        print(f"   ⚠ Metadata file not found: {metadata_file}")
        print("   → Run experiments/scripts/setup/create_county_registry.py first")
    else:
        print(f"   ✓ Metadata file exists: {metadata_file}")
except Exception as e:
    print(f"   ✗ Path error: {e}")

# Test 4: Test county metadata (if exists)
print("\n4. Testing county metadata...")
try:
    metadata_file = Path(base_config['data']['county_metadata_file'])
    if metadata_file.exists():
        import pandas as pd
        metadata = pd.read_csv(metadata_file)
        print(f"   ✓ County metadata loaded")
        print(f"   → Found {len(metadata)} counties")
        print(f"   → Row count range: {metadata['row_count'].min()}-{metadata['row_count'].max()}")
    else:
        print("   ⚠ Skipping (metadata file not found)")
        print(f"   → Run: python experiments/scripts/setup/create_county_registry.py")
except Exception as e:
    print(f"   ✗ Metadata error: {e}")

# Test 5: Test data loader with a sample county
print("\n5. Testing data loader...")
try:
    # Test with no preprocessing (raw data)
    loader = CountyDataLoader(
        county_csvs_dir=base_config['data']['county_csvs_dir'],
        target_column=base_config['data']['target_column'],
        preprocessing_config=None  # No preprocessing for test
    )

    # Try to load a small county if we know any
    metadata_file = Path(base_config['data']['county_metadata_file'])
    if metadata_file.exists():
        import pandas as pd
        metadata = pd.read_csv(metadata_file)
        small_counties = metadata[
            (metadata['row_count'] >= 10) & (metadata['row_count'] <= 100)
        ]

        if len(small_counties) > 0:
            fips = small_counties.iloc[0]['fips']
            print(f"   Testing with county {fips}...")

            df = loader.load_county(fips, drop_missing_target=True)
            X, y = loader.preprocess_for_training(df)

            print(f"   ✓ Data loader works")
            print(f"     - Loaded {len(X)} samples with {X.shape[1]} features (no preprocessing)")
        else:
            print("   ⚠ No small counties found in metadata")
    else:
        print("   ⚠ Skipping (metadata file not found)")
except Exception as e:
    print(f"   ✗ Loader error: {e}")

# Test 6: Check GPU availability
print("\n6. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"     - CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ No GPU available (will use CPU)")
except Exception as e:
    print(f"   ✗ GPU check error: {e}")

# Summary
print("\n" + "="*80)
print("Setup Test Complete")
print("="*80)
print("\nNext steps:")
print("1. If metadata file not found, run: python experiments/scripts/setup/create_county_registry.py")
print("2. Launch experiments using: python experiments/run_experiment.py")
print("3. See experiments/docs/README.md for configuration details")
print()
