#!/usr/bin/env python
"""
Smoke test for per-county scaling experiment with minimal fake data.

Creates tiny fake dataset (~100 rows) that can run on interactive nodes with
limited memory (4-8GB). Tests the full pipeline end-to-end.

Usage:
    python experiments/scripts/smoke_test_per_county_scaling.py

This will:
1. Create fake data.parquet with 2 tiny counties (50 rows each)
2. Create fake test set result files
3. Create smoke test config
4. Run the experiment for 1 county with 2 train sizes, 1 seed, 1 model
5. Verify results were generated correctly
6. Clean up temporary files

Expected runtime: ~2-3 minutes on interactive node
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import json
import yaml

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Per-County Scaling Experiment - Smoke Test")
print("=" * 80)

# Create temporary directory for test files
temp_dir = Path(tempfile.mkdtemp(prefix="per_county_scaling_smoke_"))
print(f"\nTemporary directory: {temp_dir}")

try:
    # ========================================
    # Step 1: Create fake data.parquet
    # ========================================
    print("\n[1/6] Creating fake data.parquet...")

    data_dir = temp_dir / "fake_data"
    data_dir.mkdir(parents=True)

    # Create 2 fake counties with 50 rows each
    np.random.seed(42)
    n_rows_per_county = 50
    fips_list = [31007, 31009]  # 2 tiny counties

    data_rows = []
    for fips in fips_list:
        for i in range(n_rows_per_county):
            row = {
                'fips': fips,
                'sale_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i * 7),
                'SALE_AMOUNT': 12.0 + np.random.randn() * 0.5,  # log-transformed
                # Property features
                'BEDROOMS': np.random.randint(1, 5),
                'TOTAL_BATHS': np.random.randint(1, 4),
                'BUILDING_SQUARE_FEET': np.random.randint(800, 3000),
                'LOT_SQUARE_FEET': np.random.randint(3000, 10000),
                'YEAR_BUILT': np.random.randint(1950, 2020),
                # Temporal features
                'sale_year': 2020,
                'sale_month': np.random.randint(1, 13),
                'sale_day': np.random.randint(1, 29),
                # Categorical
                'PROPERTY_INDICATOR': np.random.choice(['A', 'B', 'C']),
                'ASSESSED_IMPROVEMENT_PERCENT': np.random.randint(0, 100),
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df.to_parquet(data_dir / "data.parquet", index=False)

    print(f"  Created data.parquet with {len(df)} rows, {len(fips_list)} counties")

    # Create metadata.json
    metadata = {
        'version': 'smoke_test',
        'n_rows': len(df),
        'n_features': len(df.columns),
        'target_log_transformed': True,
        'counties': fips_list,
    }
    with open(data_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # ========================================
    # Step 2: Create fake test set result
    # ========================================
    print("\n[2/6] Creating fake test set result...")

    test_set_dir = temp_dir / "test_v1"
    test_set_dir.mkdir(parents=True)

    # Split each county: first 30 rows = train pool, last 20 rows = test
    test_indices = []
    train_pool_indices = []
    county_info = {}

    for i, fips in enumerate(fips_list):
        start_idx = i * n_rows_per_county
        county_train_pool = list(range(start_idx, start_idx + 30))
        county_test = list(range(start_idx + 30, start_idx + 50))

        train_pool_indices.extend(county_train_pool)
        test_indices.extend(county_test)

        county_info[str(fips)] = {
            'size_bucket': 'tiny',
            'train_pool_rows': len(county_train_pool),
            'test_rows': len(county_test),
        }

    # Save npy files
    np.save(test_set_dir / "test_indices.npy", np.array(test_indices))
    np.save(test_set_dir / "train_pool_indices.npy", np.array(train_pool_indices))

    # Save JSON files
    with open(test_set_dir / "test_counties.json", 'w') as f:
        json.dump(fips_list, f)

    with open(test_set_dir / "county_info.json", 'w') as f:
        json.dump(county_info, f)

    size_buckets = {'tiny': fips_list}
    with open(test_set_dir / "size_buckets.json", 'w') as f:
        json.dump(size_buckets, f)

    test_metadata = {
        'version': 'smoke_test_v1',
        'n_test_counties': len(fips_list),
        'n_test_samples': len(test_indices),
        'n_train_pool_samples': len(train_pool_indices),
    }
    with open(test_set_dir / "metadata.json", 'w') as f:
        json.dump(test_metadata, f)

    print(f"  Created test set: {len(test_indices)} test, {len(train_pool_indices)} train pool")

    # ========================================
    # Step 3: Create smoke test config
    # ========================================
    print("\n[3/6] Creating smoke test config...")

    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True)

    config = {
        'experiment': {
            'type': 'per_county_scaling',
            'name': 'smoke_test',
            'description': 'Smoke test with minimal fake data',
            'random_seed': 42,
        },
        'data': {
            'cleaned_data_path': str(data_dir),
            'target_column': 'SALE_AMOUNT',
        },
        'splits': {
            'test_set_dir': str(test_set_dir),
        },
        'target_buckets': ['tiny'],
        'train_sizes': [5, 10],  # Just 2 train sizes for speed
        'n_seeds': 1,  # Just 1 seed
        'preprocessing': {
            'phase2_steps': {
                'winsorize': True,
                'winsorize_percentile': 1,
                'normalize_continuous': True,
                'impute_method': 'median',
            },
        },
        'models': [
            {'name': 'tabpfn', 'enabled': True},
            {'name': 'xgboost', 'enabled': False},  # Disable XGB for speed
        ],
        'tabpfn': {
            'version': 'v2',
            'device': 'cpu',  # Use CPU for smoke test
        },
        'xgboost': {
            'optuna_trials': 5,  # Minimal tuning if enabled
            'optuna_cv_folds': 2,
            'use_gpu': False,
        },
        'checkpointing': {
            'enabled': True,
            'interval': 10,
            'resume': True,
        },
        'output': {
            'results_dir': str(temp_dir / "results"),
        },
        'logging': {
            'level': 'INFO',
        },
    }

    config_file = config_dir / "smoke_test.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Created config: {config_file}")

    # ========================================
    # Step 4: Run experiment for first county only
    # ========================================
    print("\n[4/6] Running experiment for first county...")
    print(f"  County FIPS: {fips_list[0]}")
    print(f"  Train sizes: {config['train_sizes']}")
    print(f"  Models: tabpfn only")
    print(f"  Expected combos: {len(config['train_sizes'])} train_sizes × 1 seed × 1 model = {len(config['train_sizes'])}")

    # Import and run
    from experiments.experiment_types import PerCountyScalingExperiment

    # Add single county mode
    config['_single_county_fips'] = fips_list[0]

    runner = PerCountyScalingExperiment(config)
    results_df, _, _ = runner.run_experiment()

    print(f"\n  Experiment completed!")

    # ========================================
    # Step 5: Verify results
    # ========================================
    print("\n[5/6] Verifying results...")

    if results_df is None or len(results_df) == 0:
        print("  ❌ FAIL: No results generated")
        sys.exit(1)

    expected_rows = len(config['train_sizes']) * config['n_seeds']  # 2 × 1 = 2
    actual_rows = len(results_df)

    if actual_rows != expected_rows:
        print(f"  ❌ FAIL: Expected {expected_rows} result rows, got {actual_rows}")
        sys.exit(1)

    # Check required columns
    required_cols = [
        'fips', 'size_bucket', 'county_train_pool_size', 'county_test_size',
        'requested_train_size', 'actual_train_size', 'seed', 'model',
        'n_features', 'fit_time', 'pred_time', 'r2', 'mae', 'rmse', 'mape', 'mse',
        'status', 'experiment_name',
    ]

    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"  ❌ FAIL: Missing columns: {missing_cols}")
        sys.exit(1)

    # Check all succeeded
    n_success = (results_df['status'] == 'success').sum()
    if n_success != actual_rows:
        print(f"  ❌ FAIL: Only {n_success}/{actual_rows} runs succeeded")
        print(results_df[['train_size', 'seed', 'model', 'status']])
        sys.exit(1)

    # Check metrics are reasonable
    if results_df['r2'].isna().any():
        print(f"  ❌ FAIL: Some R2 values are NaN")
        sys.exit(1)

    # Check results file was saved
    results_file = temp_dir / "results" / f"county_{fips_list[0]}" / "results.csv"
    if not results_file.exists():
        print(f"  ❌ FAIL: Results file not found: {results_file}")
        sys.exit(1)

    print(f"  ✓ Generated {actual_rows} result rows (expected {expected_rows})")
    print(f"  ✓ All runs succeeded")
    print(f"  ✓ All required columns present")
    print(f"  ✓ Results file saved: {results_file}")

    # Print sample results
    print("\n  Sample results:")
    print(results_df[['fips', 'requested_train_size', 'seed', 'model', 'r2', 'mae', 'status']].to_string(index=False))

    # ========================================
    # Step 6: Cleanup
    # ========================================
    print("\n[6/6] Cleaning up...")
    shutil.rmtree(temp_dir)
    print(f"  Removed temporary directory: {temp_dir}")

    print("\n" + "=" * 80)
    print("✓ SMOKE TEST PASSED")
    print("=" * 80)
    print("\nThe per-county scaling experiment is working correctly!")
    print("You can now run the full experiment with:")
    print("  sbatch experiments/slurm/per_county_scaling.sh experiments/configs/per_county_scaling/tiny_small.yaml")

except Exception as e:
    print(f"\n❌ SMOKE TEST FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

    # Cleanup on failure
    if temp_dir.exists():
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    sys.exit(1)
