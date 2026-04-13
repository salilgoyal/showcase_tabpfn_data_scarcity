#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=notebooks/feature_investigation/logs/feature_analysis_%j.out
#SBATCH --error=notebooks/feature_investigation/logs/feature_analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# SLURM job to analyze feature coverage across all county CSV files

# Paths
PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
COUNTY_CSVS_DIR="/scratch/users/salilg/property_tax/county_csvs"
OUTPUT_DIR="${PROJECT_HOME}/notebooks/feature_investigation/output"

# Load Python
module load python/3.12

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_HOME}/notebooks/feature_investigation/logs"

echo "======================================"
echo "Feature Coverage Analysis"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "County CSVs: ${COUNTY_CSVS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Run the analysis
cd "${PROJECT_HOME}"
python notebooks/feature_investigation/analyze_county_features.py \
    "${COUNTY_CSVS_DIR}" \
    "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}

