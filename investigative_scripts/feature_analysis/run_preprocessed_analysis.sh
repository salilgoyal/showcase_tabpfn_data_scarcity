#!/bin/bash
#SBATCH --job-name=preproc_feat_analysis
#SBATCH --output=notebooks/feature_investigation/logs/preproc_analysis_%j.out
#SBATCH --error=notebooks/feature_investigation/logs/preproc_analysis_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# SLURM job to analyze feature coverage AFTER preprocessing each county

# Paths
PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
COUNTY_CSVS_DIR="/scratch/users/salilg/property_tax/county_csvs"
OUTPUT_DIR="${PROJECT_HOME}/notebooks/feature_investigation/output_preprocessed"
CONFIG_FILE="${PROJECT_HOME}/experiments/configs/cross_county/small_in_context_10k.yaml"

# Load modules
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_HOME}/notebooks/feature_investigation/logs"

echo "======================================"
echo "Preprocessed Feature Coverage Analysis"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "County CSVs: ${COUNTY_CSVS_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "NOTE: This will preprocess ALL ~2850 counties"
echo "      Estimated runtime: 2-4 hours"
echo ""

# Run the analysis
cd "${PROJECT_HOME}"
python notebooks/feature_investigation/analyze_preprocessed_features.py \
    "${COUNTY_CSVS_DIR}" \
    "${OUTPUT_DIR}" \
    "${CONFIG_FILE}"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}
