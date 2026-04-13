"""
Test in-context data loading to debug column mismatch issues.

This script simulates the exact preprocessing pipeline used in the finetuning experiment
to verify that in-context data can be loaded and preprocessed correctly.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loading import CleanedDataLoader
from src.data.preprocessing_utils import Phase2Preprocessor
from src.data.split_strategies import get_train_test_data

print("=" * 80)
print("TESTING IN-CONTEXT DATA LOADING")
print("=" * 80)

# Configuration
DATA_PATH = "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/"
TRAIN_V2_PATH = "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v2/"
TARGET_COLUMN = "SALE_AMOUNT"

# Phase 2 config (from finetuning.yaml)
PHASE2_CONFIG = {
    "winsorize": True,
    "winsorize_percentile": 1,
    "normalize_continuous": True,
    "impute_method": "median"
}

print("\n1. Loading small sample of data (first 10000 rows)...")
data_loader = CleanedDataLoader(DATA_PATH, target_column=TARGET_COLUMN, phase2_config=PHASE2_CONFIG)
# Use max_rows to efficiently load only first 10K rows instead of full 17M dataset
df_sample = data_loader.load_data_by_indices(indices=np.array([]), max_rows=10000)
print(f"   Sample shape: {df_sample.shape}")
print(f"   Columns: {df_sample.columns.tolist()}")

print("\n2. Loading train_v2 indices...")
train_indices = np.load(f"{TRAIN_V2_PATH}/train_indices.npy")
print(f"   Train indices shape: {train_indices.shape}")
print(f"   First 10 indices: {train_indices[:10]}")

# For testing, use indices that exist in our sample
test_indices_sample = np.arange(1000, 2000)  # Use rows 1000-2000 as test
train_indices_sample = np.arange(0, 1000)  # Use first 1000 as train

print(f"\n3. Creating train/test split (using sample data)...")
print(f"   Train: {len(train_indices_sample)} samples")
print(f"   Test: {len(test_indices_sample)} samples")

# Simulate get_train_test_data with default exclude_columns
exclude_columns = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]

print(f"\n4. Extracting X_train, y_train (before Phase 2)...")
train_df = df_sample.iloc[train_indices_sample]
y_train = train_df[TARGET_COLUMN]

# Drop target and excluded columns
columns_to_drop = [TARGET_COLUMN] + [c for c in exclude_columns if c in train_df.columns]
print(f"   Dropping {len(columns_to_drop)} columns: {columns_to_drop}")
X_train = train_df.drop(columns=columns_to_drop)

# Drop object columns
object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"   Dropping {len(object_cols)} object columns: {object_cols}")
    X_train = X_train.drop(columns=object_cols)

print(f"   X_train shape before Phase 2: {X_train.shape}")
print(f"   X_train columns: {X_train.columns.tolist()[:10]}...")  # Show first 10

print(f"\n5. Applying Phase 2 preprocessing to X_train...")
preprocessor = Phase2Preprocessor(PHASE2_CONFIG)
preprocessor.fit(X_train, y_train)
X_train_transformed = preprocessor.transform(X_train)
print(f"   X_train shape after Phase 2: {X_train_transformed.shape}")

# Check if preprocessor has continuous_cols_
if hasattr(preprocessor, 'continuous_cols_'):
    print(f"   Preprocessor has continuous_cols_: {len(preprocessor.continuous_cols_)} columns")
    print(f"   First 10: {preprocessor.continuous_cols_[:10]}")
else:
    print(f"   WARNING: Preprocessor does not have continuous_cols_ attribute!")

print(f"\n6. Simulating in-context data loading (using train_v2 indices)...")
# For this test, use indices from our sample dataset (0-9999) that weren't used in train
# In the real experiment, these would be loaded from train_v2 indices
context_indices = np.arange(2000, 2100)  # Use rows 2000-2100 as in-context samples
print(f"   Using {len(context_indices)} in-context samples")

context_df = df_sample.iloc[context_indices]
y_context = context_df[TARGET_COLUMN]

print(f"\n7. Preprocessing in-context data (same as X_train)...")
# Drop target and excluded columns
columns_to_drop = [TARGET_COLUMN] + [c for c in exclude_columns if c in context_df.columns]
print(f"   Dropping {len(columns_to_drop)} columns")
X_context = context_df.drop(columns=columns_to_drop)

# Drop object columns
object_cols_context = X_context.select_dtypes(include=['object']).columns.tolist()
if object_cols_context:
    print(f"   Dropping {len(object_cols_context)} object columns: {object_cols_context}")
    X_context = X_context.drop(columns=object_cols_context)

print(f"   X_context shape before Phase 2: {X_context.shape}")
print(f"   X_context columns: {X_context.columns.tolist()[:10]}...")  # Show first 10

print(f"\n8. Checking column alignment...")
missing_in_context = set(X_train.columns) - set(X_context.columns)
extra_in_context = set(X_context.columns) - set(X_train.columns)

if missing_in_context:
    print(f"   ERROR: X_context missing columns that X_train has: {missing_in_context}")
if extra_in_context:
    print(f"   ERROR: X_context has extra columns that X_train doesn't: {extra_in_context}")
if not missing_in_context and not extra_in_context:
    print(f"   ✓ Column sets match! Both have {len(X_train.columns)} columns")

    # Check if column ORDER matches
    if (X_train.columns == X_context.columns).all():
        print(f"   ✓ Column order also matches!")
    else:
        print(f"   WARNING: Column order differs, will need to reorder")
        X_context = X_context[X_train.columns]
        print(f"   ✓ Reordered X_context columns to match X_train")

print(f"\n9. Applying Phase 2 preprocessing to X_context...")
try:
    X_context_transformed = preprocessor.transform(X_context)
    print(f"   ✓ SUCCESS! X_context transformed without errors")
    print(f"   X_context shape after Phase 2: {X_context_transformed.shape}")

    # Verify shapes match
    if X_train_transformed.shape[1] == X_context_transformed.shape[1]:
        print(f"   ✓ Feature dimensions match: {X_context_transformed.shape[1]} features")
    else:
        print(f"   ERROR: Feature dimensions don't match!")
        print(f"     X_train: {X_train_transformed.shape[1]} features")
        print(f"     X_context: {X_context_transformed.shape[1]} features")

except Exception as e:
    print(f"   ✗ FAILED to transform X_context!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nIf this test passes, the in-context loading logic should work in the full experiment.")
print("If it fails, we can see exactly where the column mismatch occurs.")
