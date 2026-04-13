#!/bin/bash
#SBATCH --job-name=gen_test_rand
#SBATCH --output=logs/splits/gen_test_%j.out
#SBATCH --error=logs/splits/gen_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# =============================================================================
# Generate test_v4_rand splits for seeds 0-4
# =============================================================================
#
# Each seed produces an independent random 20% split of the same counties
# (selected by test_v4_rand.yaml with county_seed=42), enabling variability
# analysis over split choices.
#
# Usage:
#   sbatch experiments/scripts/generate_test_set_rand.sbatch
#
# Output dirs:
#   /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s{0..4}/
# =============================================================================

set -e

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"

# Load required modules
module load python/3.12
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create logs directory if needed
mkdir -p "${PROJECT_HOME}/logs/splits"

# Print job info
echo "======================================"
echo "GENERATING TEST SET SPLITS"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Memory requested: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK}"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

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
echo "======================================"
echo "Done. Splits saved to ${OUTPUT_BASE}/test_v4_rand_s{0..4}/"
echo "Finished at: $(date)"
echo "======================================"
