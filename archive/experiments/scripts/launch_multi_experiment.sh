#!/bin/bash
# Launch multiple experiment configs in parallel
#
# This script submits jobs for multiple experiment configurations at once.
# Each experiment will have its own set of array jobs and output directories.
#
# Usage:
#   ./launch_multi_experiment.sh [EXPERIMENT_TYPE]
#
# Arguments:
#   EXPERIMENT_TYPE: "within_county" or "cross_county" (default: within_county)
#
# Example:
#   ./launch_multi_experiment.sh within_county
#   ./launch_multi_experiment.sh cross_county

set -e  # Exit on error

# Configuration
EXPERIMENT_TYPE="${1:-within_county}"

# Experiment configs to run
EXPERIMENT_CONFIGS=(
    "experiments/with_preprocessing.yaml"
    # "experiments/no_preprocessing.yaml"
)

# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine log directory based on experiment type
# If any config contains "calibration", use calibration log dir
if echo "${EXPERIMENT_CONFIGS[@]}" | grep -q "calibration"; then
    LOG_DIR="/scratch/users/salilg/property_tax/calibration/logs"
    echo "Detected calibration experiment(s) - using calibration log directory"
else
    LOG_DIR="/scratch/users/salilg/property_tax/logs"
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Multi-Experiment Launcher"
echo "=========================================="
echo "Experiment type: $EXPERIMENT_TYPE"
echo "Number of configs: ${#EXPERIMENT_CONFIGS[@]}"
echo ""

# Determine which SLURM script to use
if [ "$EXPERIMENT_TYPE" == "within_county" ]; then
    SLURM_SCRIPT="${SCRIPT_DIR}/slurm_01_within_county.sh"
elif [ "$EXPERIMENT_TYPE" == "cross_county" ]; then
    SLURM_SCRIPT="${SCRIPT_DIR}/slurm_02_cross_county.sh"
else
    echo "ERROR: Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Must be 'within_county' or 'cross_county'"
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Submit jobs for each experiment config
JOB_IDS=()
for CONFIG in "${EXPERIMENT_CONFIGS[@]}"; do
    echo "----------------------------------------"
    echo "Launching experiment: $CONFIG"
    echo "----------------------------------------"

    # Extract experiment name from config path
    EXP_NAME=$(basename "$CONFIG" .yaml)

    # Submit job
    OUTPUT=$(sbatch \
        --job-name="${EXPERIMENT_TYPE}_${EXP_NAME}" \
        --output="${LOG_DIR}/${EXPERIMENT_TYPE}_${EXP_NAME}_%A_%a.out" \
        --error="${LOG_DIR}/${EXPERIMENT_TYPE}_${EXP_NAME}_%A_%a.err" \
        "$SLURM_SCRIPT" "$CONFIG")

    # Extract job ID
    JOB_ID=$(echo "$OUTPUT" | grep -oP '(?<=Submitted batch job )\d+')
    JOB_IDS+=("$JOB_ID")

    echo "  Job ID: $JOB_ID"
    echo "  Job name: ${EXPERIMENT_TYPE}_${EXP_NAME}"
    echo "  Logs: ${LOG_DIR}/${EXPERIMENT_TYPE}_${EXP_NAME}_${JOB_ID}_*.{out,err}"
    echo ""

    # Small delay between submissions
    sleep 2
done

echo "=========================================="
echo "All experiments submitted!"
echo "=========================================="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job:"
for ((i=0; i<${#EXPERIMENT_CONFIGS[@]}; i++)); do
    CONFIG="${EXPERIMENT_CONFIGS[$i]}"
    JOB_ID="${JOB_IDS[$i]}"
    EXP_NAME=$(basename "$CONFIG" .yaml)
    echo "  squeue -j ${JOB_ID}  # ${EXP_NAME}"
done
echo ""
echo "View logs:"
for ((i=0; i<${#EXPERIMENT_CONFIGS[@]}; i++)); do
    CONFIG="${EXPERIMENT_CONFIGS[$i]}"
    JOB_ID="${JOB_IDS[$i]}"
    EXP_NAME=$(basename "$CONFIG" .yaml)
    echo "  tail -f ${LOG_DIR}/${EXPERIMENT_TYPE}_${EXP_NAME}_${JOB_ID}_0.out"
done
