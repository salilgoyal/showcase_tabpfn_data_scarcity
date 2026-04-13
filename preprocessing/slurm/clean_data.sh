#!/bin/bash
#SBATCH --job-name=preprocess_pooled_data
#SBATCH --output=preprocessing/logs/%j.out
#SBATCH --error=preprocessing/logs/%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# =============================================================================
# Phase 1 Preprocessing: Clean and Pool County Data
# =============================================================================
#
# This script runs the Phase 1 preprocessing pipeline that:
#   1. Loads all county CSV files
#   2. Applies data cleaning (drop nulls, duplicates, etc.)
#   3. Generates temporal features
#   4. Label encodes categoricals (no one-hot encoding)
#   5. Log transforms the target
#   6. Saves as a single parquet file
#
# Usage:
#   sbatch preprocessing/slurm/clean_data.sh <CONFIG_FILE>
#
# Examples:
#   sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v1_no_onehot.yaml
#
# Output:
#   /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/<version>/
#     - data.parquet: Cleaned pooled data
#     - config.yaml: Copy of config used
#     - metadata.json: Statistics and column info
#     - preprocessing_log.txt: Detailed log
# =============================================================================

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Config file (required argument)
CONFIG_FILE="${1:-}"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: Config file required as first argument"
    echo "Usage: sbatch preprocessing/slurm/clean_data.sh <config_file>"
    echo "Example: sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v1_no_onehot.yaml"
    exit 1
fi

# Load modules
module load python/3.12
module load devel
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create logs directory if needed
mkdir -p "${PROJECT_HOME}/preprocessing/logs"

# Print job info
echo "======================================"
echo "PHASE 1 PREPROCESSING"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $CONFIG_FILE"
echo "Memory requested: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK}"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run preprocessing
echo "Starting preprocessing pipeline..."
echo ""

python preprocessing/scripts/clean_pooled_data.py \
    --config "$CONFIG_FILE" \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
