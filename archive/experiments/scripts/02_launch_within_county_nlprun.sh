#!/bin/bash
# Helper script to launch within-county experiments using nlprun
# This manages parallelism by launching jobs in batches

set -e

# Default settings
MAX_PARALLEL=10
QUEUE="jag"
PARTITION="standard"
CONDA_ENV="pfn_env"
GPU=1
CPU=8
MEM="32G"
SKIP_COMPLETED=true  # Set to false to rerun all counties

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --no-skip)
            SKIP_COMPLETED=false
            shift
            ;;
        --rerun-all)
            SKIP_COMPLETED=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--max-parallel N] [--queue Q] [--partition P] [--no-skip]"
            exit 1
            ;;
    esac
done

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
METADATA_FILE="${PROJECT_ROOT}/small_county_metadata.csv"
RUNNER_SCRIPT="${SCRIPT_DIR}/../runners/within_county_runner.py"
OUTFILES_DIR="${SCRIPT_DIR}/../outfiles"
RESULTS_DIR="${PROJECT_ROOT}/results/within_county"

mkdir -p "$OUTFILES_DIR"
mkdir -p "$RESULTS_DIR"

# Check if metadata exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    echo "Please run 00_create_county_registry.py first"
    exit 1
fi

echo "========================================"
echo "Launching Within-County Experiments"
echo "========================================"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Queue: $QUEUE"
echo "Partition: $PARTITION"
echo "Skip completed: $SKIP_COMPLETED"
echo "Metadata file: $METADATA_FILE"
echo ""

# Count total counties (skip header)
TOTAL_COUNTIES=$(tail -n +2 "$METADATA_FILE" | wc -l)
echo "Total counties to process: $TOTAL_COUNTIES"

# Count already completed counties
if [ "$SKIP_COMPLETED" = true ]; then
    COMPLETED_COUNT=$(ls -1 "$RESULTS_DIR"/county_*_results.csv 2>/dev/null | wc -l)
    echo "Already completed: $COMPLETED_COUNT"
    echo "Remaining: $((TOTAL_COUNTIES - COMPLETED_COUNT))"
fi
echo ""

# Read counties and launch jobs
LAUNCHED=0
ACTIVE_JOBS=0

# Function to count active jobs
count_active_jobs() {
    # Count number of jobs with our name pattern that are running
    nlpjobs 2>/dev/null | grep -c "within-county-" || echo "0"
}

# Process each county - extract only needed columns (fips, bin_name, k_folds)
# Columns: fips(1),filename(2),row_count(3),file_size_bytes(4),file_size_mb(5),bin_name(6),k_folds(7),num_features(8),feature_list(9)
tail -n +2 "$METADATA_FILE" | while IFS= read -r line; do
    # Extract first 7 fields before the feature_list (which has commas)
    FIPS=$(echo "$line" | cut -d',' -f1)
    BIN_NAME=$(echo "$line" | cut -d',' -f6)
    K_FOLDS=$(echo "$line" | cut -d',' -f7)

    # Skip if parsing failed
    if [ -z "$FIPS" ] || [ -z "$BIN_NAME" ] || [ -z "$K_FOLDS" ]; then
        continue
    fi

    # Check if this county is already completed
    RESULT_FILE="${RESULTS_DIR}/county_${FIPS}_results.csv"
    if [ "$SKIP_COMPLETED" = true ] && [ -f "$RESULT_FILE" ]; then
        echo "Skipping county $FIPS (already completed)"
        continue
    fi

    # Wait if we've hit the parallel limit
    while [ $(count_active_jobs) -ge $MAX_PARALLEL ]; do
        echo "Reached max parallel jobs ($MAX_PARALLEL), waiting..."
        sleep 30
    done

    JOB_NAME="within-county-${FIPS}"
    OUT_FILE="${OUTFILES_DIR}/within_county_${FIPS}.out"

    echo "Launching county $FIPS (bin: $BIN_NAME, k=$K_FOLDS)..."

    nlprun -q "$QUEUE" -p "$PARTITION" -a "$CONDA_ENV" \
        -g $GPU -c $CPU -r $MEM \
        -n "$JOB_NAME" \
        -o "$OUT_FILE" \
        "cd ${SCRIPT_DIR}/../runners && python within_county_runner.py --fips $FIPS --bin_name $BIN_NAME --k_folds $K_FOLDS"

    LAUNCHED=$((LAUNCHED + 1))
    echo "  Launched job $LAUNCHED/$TOTAL_COUNTIES"
    echo ""

    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "========================================"
echo "All jobs launched!"
echo "Total jobs: $LAUNCHED"
echo "========================================"
echo ""
echo "Monitor jobs with: nlpjobs"
echo "Check output files in: $OUTFILES_DIR"
