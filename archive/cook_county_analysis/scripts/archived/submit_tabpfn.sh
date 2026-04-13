#!/bin/bash
#SBATCH --job-name=tabpfn-propertyonly-original-20seeds
#SBATCH --output=../outfiles/propertyonly-original-20seeds/tabpfn.out
#SBATCH --error=../outfiles/propertyonly-original-20seeds/tabpfn.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Get experiment name from command line or use default
EXPERIMENT_NAME=${1:-"propertyonly-original-20seeds"}

###############################
# Submit TabPFN experiment
# Usage: sbatch submit_tabpfn.sh <experiment_name>
# Example: sbatch submit_tabpfn.sh propertyonly-original-20seeds
###############################

echo "=================================="
echo "Starting TabPFN Experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=================================="

# Activate conda environment
# . /nlp/scr/salilg/miniconda3/etc/profile.d/conda.sh
# conda activate pfn_env

# for Sherlock
module load python/3.12
module load cuda
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate 

# Create output directories
mkdir -p ../outfiles/$EXPERIMENT_NAME
mkdir -p ../results/$EXPERIMENT_NAME
mkdir -p ../logs/$EXPERIMENT_NAME

# Run experiment
python3 run_tabpfn_experiment.py \
    --experiment_name $EXPERIMENT_NAME \
    --data_path ../data/cook_county.csv \
    --output_dir ../results \
    --no_predictions

exit_code=$?

echo ""
echo "=================================="
echo "TabPFN Experiment Complete"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "=================================="

if [ $exit_code -eq 0 ]; then
    echo "✓ Results saved to: ../results/$EXPERIMENT_NAME/tabpfn.csv"
else
    echo "✗ Experiment failed!"
fi

exit $exit_code
