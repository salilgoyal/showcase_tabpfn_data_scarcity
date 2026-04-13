#!/bin/bash
# Generate test_v4_rand splits for seeds 0-4.
#
# Each seed produces an independent random 20% split of the same counties
# (selected by test_v4_rand.yaml with county_seed=42), enabling variability
# analysis over split choices.
#
# Usage:
#   bash experiments/scripts/generate_test_set_rand.sh
#
# Output dirs:
#   /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s{0..4}/

set -e

# Load required modules
ml python/3.12
ml py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH

CONFIG=experiments/configs/test_sets/test_v4_rand.yaml
DATA=/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/data.parquet
OUTPUT_BASE=/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot

for S in 0 1 2 3 4; do
    echo "============================================================"
    echo "Generating test_v4_rand_s${S} (split_seed=${S})"
    echo "============================================================"
    python experiments/scripts/generate_test_set.py \
        --config "$CONFIG" \
        --data_path "$DATA" \
        --output_dir "${OUTPUT_BASE}/test_v4_rand_s${S}/" \
        --split_seed "$S"
done

echo ""
echo "Done. Splits saved to ${OUTPUT_BASE}/test_v4_rand_s{0..4}/"
