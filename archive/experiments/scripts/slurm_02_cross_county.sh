#!/bin/bash
#SBATCH --job-name=cross_county
#SBATCH --output=/scratch/users/salilg/property_tax/logs/cross_county_%A_%a.out
#SBATCH --error=/scratch/users/salilg/property_tax/logs/cross_county_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-499  # Adjust: (num_counties * num_iterations) - 1

# SLURM Array Job for Cross-County Experiment
#
# This script runs the cross-county experiment using SLURM array jobs.
# Each array task processes one (county, iteration) pair.
#
# Usage:
#   1. First, create the county registry:
#      cd /home/users/salilg/tabpfn_data_scarcity/experiments/scripts
#      python 00_create_county_registry.py
#
#   2. Calculate total jobs:
#      num_counties = (lines in small_county_metadata.csv) - 1
#      num_iterations = 10 (default, set in config)
#      total_jobs = num_counties * num_iterations
#
#   3. Submit job array:
#      sbatch --array=0-N slurm_02_cross_county.sh
#      where N = total_jobs - 1
#
# Example:
#   # Run with default config
#   sbatch --array=0-499 slurm_02_cross_county.sh
#
#   # Run with experiment config
#   sbatch --array=0-499 slurm_02_cross_county.sh experiments/with_preprocessing.yaml

# Configuration
N_ITERATIONS=10  # Must match config file

# Experiment config (optional first argument)
EXPERIMENT_CONFIG="${1:-}"

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Load modules for Sherlock
module load python/3.12
module load cuda

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set paths
METADATA_FILE="${PROJECT_HOME}/small_county_metadata.csv"
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/runners/cross_county_runner.py"
OUTPUT_DIR="${SCRATCH_DIR}/results/cross_county"

echo "======================================"
echo "Cross-County Experiment"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""

# Check if metadata file exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    echo "Please run 00_create_county_registry.py first"
    exit 1
fi

# Get all FIPS codes as comma-separated list (first column only)
FIPS_LIST=$(tail -n +2 "$METADATA_FILE" | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')

# Calculate which county and iteration this task corresponds to
NUM_COUNTIES=$(tail -n +2 "$METADATA_FILE" | wc -l)
COUNTY_IDX=$((SLURM_ARRAY_TASK_ID / N_ITERATIONS))
ITERATION=$((SLURM_ARRAY_TASK_ID % N_ITERATIONS))

# Get county info for this index (skip header, then get line COUNTY_IDX+2)
LINE_NUM=$((COUNTY_IDX + 2))
COUNTY_LINE=$(sed -n "${LINE_NUM}p" "$METADATA_FILE")

if [ -z "$COUNTY_LINE" ]; then
    echo "ERROR: No data for county index ${COUNTY_IDX}"
    echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Calculated: county_idx=${COUNTY_IDX}, iteration=${ITERATION}"
    exit 1
fi

# Extract fields (only first 7 columns)
TARGET_FIPS=$(echo "$COUNTY_LINE" | cut -d',' -f1)
BIN_NAME=$(echo "$COUNTY_LINE" | cut -d',' -f6)

echo "Processing:"
echo "  Target County FIPS: $TARGET_FIPS"
echo "  Iteration: $ITERATION"
echo "  Bin: $BIN_NAME"
echo "  Total counties in pool: $NUM_COUNTIES"
if [ -n "$EXPERIMENT_CONFIG" ]; then
    echo "  Experiment config: $EXPERIMENT_CONFIG"
fi
echo ""

# Run experiment
python "$RUNNER_SCRIPT" \
    --target_fips "$TARGET_FIPS" \
    --fips_list "$FIPS_LIST" \
    --iteration "$ITERATION" \
    --bin_name "$BIN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    ${EXPERIMENT_CONFIG:+--experiment_config "$EXPERIMENT_CONFIG"}

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
