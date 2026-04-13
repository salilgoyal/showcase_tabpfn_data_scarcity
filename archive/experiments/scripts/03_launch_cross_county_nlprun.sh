#!/bin/bash
# Helper script to launch cross-county experiments using nlprun
# This manages parallelism by launching jobs in batches

set -e

# Default settings
MAX_PARALLEL=10
QUEUE="jag"
PARTITION="standard"
CONDA_ENV="pfn_env"
GPU=1
CPU=8
MEM="64G"
N_ITERATIONS=1
SKIP_COMPLETED=true  # Set to false to rerun all jobs

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
        --iterations)
            N_ITERATIONS="$2"
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
            echo "Usage: $0 [--max-parallel N] [--queue Q] [--partition P] [--iterations N] [--no-skip]"
            exit 1
            ;;
    esac
done

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
METADATA_FILE="${PROJECT_ROOT}/small_county_metadata.csv"
RUNNER_SCRIPT="${SCRIPT_DIR}/../runners/cross_county_runner.py"
OUTFILES_DIR="${SCRIPT_DIR}/../outfiles"
RESULTS_DIR="${PROJECT_ROOT}/results/cross_county"

mkdir -p "$OUTFILES_DIR"
mkdir -p "$RESULTS_DIR"

# Check if metadata exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    echo "Please run 00_create_county_registry.py first"
    exit 1
fi

echo "========================================"
echo "Launching Cross-County Experiments"
echo "========================================"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Iterations per county: $N_ITERATIONS"
echo "Queue: $QUEUE"
echo "Partition: $PARTITION"
echo "Skip completed: $SKIP_COMPLETED"
echo "Metadata file: $METADATA_FILE"
echo ""

# Get all FIPS codes as comma-separated list (extract first column only)
FIPS_LIST=$(tail -n +2 "$METADATA_FILE" | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')
echo "FIPS list: $FIPS_LIST"
echo ""

# Count total counties
TOTAL_COUNTIES=$(tail -n +2 "$METADATA_FILE" | wc -l)
TOTAL_JOBS=$((TOTAL_COUNTIES * N_ITERATIONS))
echo "Total counties: $TOTAL_COUNTIES"
echo "Total jobs to launch: $TOTAL_JOBS"

# Count already completed jobs
if [ "$SKIP_COMPLETED" = true ]; then
    COMPLETED_COUNT=$(ls -1 "$RESULTS_DIR"/county_*_iter_*_results.csv 2>/dev/null | wc -l)
    echo "Already completed: $COMPLETED_COUNT"
    echo "Remaining: $((TOTAL_JOBS - COMPLETED_COUNT))"
fi
echo ""

# Function to count active jobs
count_active_jobs() {
    nlpjobs 2>/dev/null | grep -c "cross-county-" || echo "0"
}

# Process each county and iteration
LAUNCHED=0

# Extract only needed columns (fips, bin_name) from first 7 fields
tail -n +2 "$METADATA_FILE" | while IFS= read -r line; do
    FIPS=$(echo "$line" | cut -d',' -f1)
    BIN_NAME=$(echo "$line" | cut -d',' -f6)

    # Skip if parsing failed
    if [ -z "$FIPS" ] || [ -z "$BIN_NAME" ]; then
        continue
    fi

    for ITER in $(seq 0 $((N_ITERATIONS - 1))); do
        # Check if this job is already completed
        RESULT_FILE="${RESULTS_DIR}/county_${FIPS}_iter_${ITER}_results.csv"
        if [ "$SKIP_COMPLETED" = true ] && [ -f "$RESULT_FILE" ]; then
            echo "Skipping county $FIPS, iteration $ITER (already completed)"
            continue
        fi

        # Wait if we've hit the parallel limit
        while [ $(count_active_jobs) -ge $MAX_PARALLEL ]; do
            echo "Reached max parallel jobs ($MAX_PARALLEL), waiting..."
            sleep 30
        done

        JOB_NAME="cross-county-${FIPS}-iter${ITER}"
        OUT_FILE="${OUTFILES_DIR}/cross_county_${FIPS}_iter${ITER}.out"

        echo "Launching county $FIPS, iteration $ITER..."

        nlprun -q "$QUEUE" -p "$PARTITION" -a "$CONDA_ENV" \
            -g $GPU -c $CPU -r $MEM \
            -n "$JOB_NAME" \
            -o "$OUT_FILE" \
            "cd ${SCRIPT_DIR}/../runners && python cross_county_runner.py --target_fips $FIPS --fips_list '$FIPS_LIST' --iteration $ITER --bin_name $BIN_NAME"

        LAUNCHED=$((LAUNCHED + 1))
        echo "  Launched job $LAUNCHED/$TOTAL_JOBS"
        echo ""

        # Small delay to avoid overwhelming the scheduler
        sleep 2
    done
done

echo ""
echo "========================================"
echo "All jobs launched!"
echo "Total jobs: $LAUNCHED"
echo "========================================"
echo ""
echo "Monitor jobs with: nlpjobs"
echo "Check output files in: $OUTFILES_DIR"
