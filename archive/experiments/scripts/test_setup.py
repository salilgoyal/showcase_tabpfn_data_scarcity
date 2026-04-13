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
    from data import CountyRegistry, CountyDataLoader, RepeatedKFoldSplitter
    from models import TabPFNModel, XGBoostModel
    from evaluation import compute_metrics, ResultsAggregator
    from runners import WithinCountyRunner, CrossCountyRunner
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Load configs
print("\n2. Testing configuration files...")
try:
    config_dir = Path(__file__).parent.parent / 'config'
    with open(config_dir / 'base_config.yaml') as f:
        base_config = yaml.safe_load(f)
    with open(config_dir / 'within_county_config.yaml') as f:
        within_config = yaml.safe_load(f)
    with open(config_dir / 'cross_county_config.yaml') as f:
        cross_config = yaml.safe_load(f)
    print("   ✓ All config files loaded successfully")
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
        print("   → Run 00_create_county_registry.py first")
    else:
        print(f"   ✓ Metadata file exists: {metadata_file}")
except Exception as e:
    print(f"   ✗ Path error: {e}")

# Test 4: Test county registry (if metadata exists)
print("\n4. Testing county registry...")
try:
    metadata_file = Path(base_config['data']['county_metadata_file'])
    if metadata_file.exists():
        registry = CountyRegistry(
            metadata_file=str(metadata_file),
            county_bins=base_config['county_bins']
        )
        small_counties = registry.get_all_small_counties()
        print(f"   ✓ County registry works")
        print(f"   → Found {len(small_counties)} small counties")

        # Show bin summary
        for bin_name in small_counties['bin_name'].unique():
            bin_df = small_counties[small_counties['bin_name'] == bin_name]
            print(f"     - Bin '{bin_name}': {len(bin_df)} counties")
    else:
        print("   ⚠ Skipping (metadata file not found)")
except Exception as e:
    print(f"   ✗ Registry error: {e}")

# Test 5: Test data loader with a sample county
print("\n5. Testing data loader...")
try:
    loader = CountyDataLoader(
        county_csvs_dir=base_config['data']['county_csvs_dir'],
        target_column=base_config['data']['target_column']
    )

    # Try to load a small county if we know any
    metadata_file = Path(base_config['data']['county_metadata_file'])
    if metadata_file.exists():
        import pandas as pd
        metadata = pd.read_csv(metadata_file)
        small_county = metadata[
            (metadata['row_count'] >= 10) & (metadata['row_count'] <= 100)
        ].iloc[0]

        fips = small_county['fips']
        print(f"   Testing with county {fips}...")

        df = loader.load_county(fips, drop_missing_target=True)
        X, y = loader.preprocess_for_training(df)

        print(f"   ✓ Data loader works")
        print(f"     - Loaded {len(X)} samples with {X.shape[1]} features")
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
print("1. If metadata file not found, run: python 00_create_county_registry.py")
print("2. Then launch experiments using nlprun or SLURM scripts")
print("3. See experiments/docs/nlprun_commands.md for details")
print()
