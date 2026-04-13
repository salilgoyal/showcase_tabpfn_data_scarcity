#!/bin/bash
#SBATCH --job-name=gen_baseline
#SBATCH --output=logs/splits/gen_baseline_%j.out
#SBATCH --error=logs/splits/gen_baseline_%j.err
#SBATCH --partition=deho
#SBATCH --time=0:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM job script for generating baseline results with per-county adjustment ratios
# Usage: sbatch experiments/slurm/splits/generate_baseline_results.sh <test_split_dir>
# Example: sbatch experiments/slurm/splits/generate_baseline_results.sh /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/
#
# Note: This computes a separate adjustment ratio for each county based on that county's
#       median(SALE_AMOUNT / CALCULATED_TOTAL_VALUE) using ALL available data (test + train_pool)
#       for maximum stability, since the baseline represents ground truth rather than a model.

echo "======================================"
echo "GENERATING BASELINE RESULTS"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Get command line arguments
TEST_SPLIT_DIR=${1:-"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"}

# Construct paths
OUTPUT_FILE="${TEST_SPLIT_DIR}baseline_results.csv"

echo "Test split dir: $TEST_SPLIT_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Load modules
module load python/3.12
module load devel
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create logs directory if needed
mkdir -p "${PROJECT_HOME}/logs/splits"

# Change to project directory
cd "$PROJECT_HOME"

# Check directory exists
if [ ! -d "$TEST_SPLIT_DIR" ]; then
    echo "ERROR: Test split directory not found: $TEST_SPLIT_DIR"
    echo "Generate test set first:"
    echo "  sbatch experiments/slurm/splits/generate_test_set.sh"
    exit 1
fi

# Run baseline results generation
echo "Running baseline results generation..."
echo ""

python experiments/scripts/generate_baseline_results.py \
    --test_split_dir "$TEST_SPLIT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --experiment_name "baseline_precomputed" \
    --experiment_description "Pre-computed baseline (per-county ratios from all available data)"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Baseline results generated successfully!"
    echo "Output: ${OUTPUT_FILE}"
    echo ""
    echo "This baseline is computed from the test set itself and can be used"
    echo "with ANY train set for this test set."
    echo ""
    echo "To use in experiments, load and concatenate with results:"
    echo "  baseline_df = pd.read_csv('${OUTPUT_FILE}')"
    echo "  results_df = pd.concat([results_df, baseline_df], ignore_index=True)"
fi

exit $EXIT_CODE
