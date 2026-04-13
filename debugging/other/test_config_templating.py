#!/usr/bin/env python
"""
Test script to verify config template substitution works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.run_experiment import load_config

def test_finetuning_config():
    """Test finetuning config template substitution."""
    config_path = "experiments/configs/finetuning/finetuning.yaml"

    print("Loading config:", config_path)
    config = load_config(config_path)

    print("\n" + "=" * 80)
    print("TEMPLATE SUBSTITUTION TEST")
    print("=" * 80)

    train_version = config['experiment']['train_version']
    print(f"\nTrain version: {train_version}")

    print(f"\nResolved values:")
    print(f"  Experiment name:  {config['experiment']['name']}")
    print(f"  Description:      {config['experiment']['description']}")
    print(f"  Train set dir:    {config['splits']['train_set_dir']}")
    print(f"  Results dir:      {config['output']['results_dir']}")
    print(f"  Checkpoint dir:   {config['output']['checkpoint_dir']}")

    # Verify substitution worked
    expected_name = f"tabpfn_finetuning_{train_version}"
    assert config['experiment']['name'] == expected_name, \
        f"Expected name '{expected_name}', got '{config['experiment']['name']}'"

    expected_train_dir = f"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/{train_version}/"
    assert config['splits']['train_set_dir'] == expected_train_dir, \
        f"Expected train_set_dir '{expected_train_dir}', got '{config['splits']['train_set_dir']}'"

    print("\n✅ All template substitutions working correctly!")
    print("=" * 80)

if __name__ == "__main__":
    test_finetuning_config()
