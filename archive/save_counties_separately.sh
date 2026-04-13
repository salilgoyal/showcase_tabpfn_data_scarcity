#!/bin/bash
#SBATCH --job-name=save_counties
#SBATCH --output=/home/users/salilg/tabpfn_data_scarcity/save_counties_%j.out
#SBATCH --error=/home/users/salilg/tabpfn_data_scarcity/save_counties_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# SBATCH script to split corelogic_census CSV by county
#
# This script runs save_counties_separately.py which:
# - Reads: /oak/stanford/groups/deho/proptax/clean/corelogic_census_2018_2023.csv
# - Writes: Individual county files to /scratch/users/salilg/property_tax/county_csvs/
# - Processes in chunks to avoid OOM errors
#
# Usage:
#   sbatch save_counties_separately.sh

echo "=========================================="
echo "County Data Splitting Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""

# Set up paths
PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
SCRIPT="${PROJECT_HOME}/save_counties_separately.py"
OUTPUT_DIR="/scratch/users/salilg/property_tax/county_csvs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load Python module
module load python/3.12
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

echo "Configuration:"
echo "  Script: ${SCRIPT}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Memory: 64G"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

# Check if input file exists
INPUT_FILE="/oak/stanford/groups/deho/proptax/clean/corelogic_census_2018_2023.csv"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please verify the path is correct"
    exit 1
fi

echo "Input file found: $INPUT_FILE"
echo "File size: $(du -h "$INPUT_FILE" | cut -f1)"
echo ""

# Run the script
echo "Starting county data splitting..."
echo "This may take several hours for large files."
echo "Progress will be logged to:"
echo "  - ${PROJECT_HOME}/save_counties_separately.log"
echo "  - ${SLURM_SUBMIT_DIR}/save_counties_${SLURM_JOB_ID}.out"
echo ""

cd "$PROJECT_HOME"
python "$SCRIPT"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully!"
    echo ""
    echo "County files saved to: $OUTPUT_DIR"
    echo "Number of county files created: $(ls -1 ${OUTPUT_DIR}/fips_*.csv 2>/dev/null | wc -l)"
    echo ""
    echo "Next steps:"
    echo "  1. Verify files: ls -lh ${OUTPUT_DIR}/"
    echo "  2. Check log: cat ${PROJECT_HOME}/save_counties_separately.log"
    echo "  3. Create county registry: cd experiments/scripts && python 00_create_county_registry.py"
else
    echo "Job failed with exit code: $EXIT_CODE"
    echo "Check error log: save_counties_${SLURM_JOB_ID}.err"
fi
echo "=========================================="
echo "Finished at: $(date)"
