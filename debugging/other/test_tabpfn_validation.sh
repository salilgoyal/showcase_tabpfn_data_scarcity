#!/bin/bash
#SBATCH --job-name=test_tabpfn_val
#SBATCH --output=logs/test_tabpfn_val_%j.out
#SBATCH --error=logs/test_tabpfn_val_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Test script to verify TabPFN validation behavior

# Load modules
module load python/3.12

# Activate environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Run test
cd /home/users/salilg/tabpfn_data_scarcity
python test_tabpfn_finetuning.py
