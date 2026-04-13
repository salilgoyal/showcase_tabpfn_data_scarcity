#!/usr/bin/env python
"""
Verify that the implementation changes are syntactically correct and properly structured.
"""

import sys
from pathlib import Path
import inspect

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def verify_implementation():
    """Verify the implementation without needing actual data."""

    print("1. Checking imports...")
    from experiments.experiment_types.cross_county import CrossCountyExperiment
    print("   ✓ CrossCountyExperiment imports successfully")

    print("\n2. Checking method signatures...")

    # Check create_sampled_split signature
    sig = inspect.signature(CrossCountyExperiment.create_sampled_split)
    return_annotation = sig.return_annotation
    print(f"   create_sampled_split return type: {return_annotation}")

    if "np.ndarray" in str(return_annotation) or "ndarray" in str(return_annotation):
        print("   ✓ create_sampled_split returns test_fips_labels")
    else:
        print("   ✗ create_sampled_split may not return test_fips_labels correctly")
        return False

    # Check _save_test_predictions_csv exists
    if hasattr(CrossCountyExperiment, '_save_test_predictions_csv'):
        print("   ✓ _save_test_predictions_csv method exists")

        sig = inspect.signature(CrossCountyExperiment._save_test_predictions_csv)
        params = list(sig.parameters.keys())
        print(f"     Parameters: {params}")

        expected_params = ['self', 'X_test', 'y_test', 'test_fips_labels', 'model_predictions', 'iteration']
        for param in expected_params:
            if param not in params:
                print(f"   ✗ Missing parameter: {param}")
                return False
        print("   ✓ All expected parameters present")
    else:
        print("   ✗ _save_test_predictions_csv method not found")
        return False

    print("\n3. Checking method implementation...")

    # Read source code and verify key changes
    source_file = Path(__file__).parent.parent.parent / 'experiments' / 'experiment_types' / 'cross_county.py'
    source = source_file.read_text()

    # Check for test_fips_labels tracking
    if 'test_fips_labels = []' in source:
        print("   ✓ test_fips_labels list initialized")
    else:
        print("   ✗ test_fips_labels list not found")
        return False

    if 'test_fips_labels.extend' in source:
        print("   ✓ test_fips_labels being populated")
    else:
        print("   ✗ test_fips_labels not being populated")
        return False

    if 'test_fips_array = np.array(test_fips_labels)' in source:
        print("   ✓ test_fips_labels converted to array")
    else:
        print("   ✗ test_fips_labels not converted to array")
        return False

    # Check for model predictions collection
    if 'model_predictions = {}' in source:
        print("   ✓ model_predictions dictionary initialized")
    else:
        print("   ✗ model_predictions dictionary not found")
        return False

    if 'model_predictions[model_name] = y_pred' in source:
        print("   ✓ predictions being collected")
    else:
        print("   ✗ predictions not being collected")
        return False

    # Check for CSV saving call
    if '_save_test_predictions_csv(' in source:
        print("   ✓ CSV saving method being called")
    else:
        print("   ✗ CSV saving method not being called")
        return False

    if 'df_predictions.to_csv' in source:
        print("   ✓ DataFrame being saved to CSV")
    else:
        print("   ✗ DataFrame not being saved to CSV")
        return False

    print("\n4. Summary:")
    print("   ✓ All implementation checks passed!")
    print("\n   The implementation should:")
    print("   - Track which FIPS each test row came from")
    print("   - Collect predictions from all models")
    print("   - Save a CSV with columns: fips, y_true, <model>_pred, [features...]")
    print("   - One CSV per iteration: test_predictions_iter{N}.csv")
    print("   - CSVs saved to the same directory as results.csv")

    return True

if __name__ == '__main__':
    success = verify_implementation()
    sys.exit(0 if success else 1)
