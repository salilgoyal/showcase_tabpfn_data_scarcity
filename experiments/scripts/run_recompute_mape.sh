#!/bin/bash
# Wrapper script to run recompute_per_county_mape.py with proper environment
# Example to run:
# bash experiments/scripts/run_recompute_mape.sh --config experiments/configs/finetuning/large_scale.yaml

# Load required modules
ml python/3.12
ml py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH

# Run the script
python experiments/scripts/recompute_per_county_mape.py "$@"
