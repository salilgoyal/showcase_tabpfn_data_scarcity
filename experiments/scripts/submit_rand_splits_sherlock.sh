#!/bin/bash
# Submit geo_pooling randsplit array jobs on the Sherlock cluster.
#
# Mirror of submit_rand_splits.sh but uses experiments/slurm/sherlock/geo_pooling.sh
# and the Sherlock-specific configs under experiments/configs/geo_pooling/sherlock/v2_no_onehot/.
#
# Prerequisites (run on Sherlock before submitting):
#   bash experiments/scripts/generate_test_set_rand.sh
#   (generates test_v4_rand_s{0..4} under /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/)
#
# Usage (run from project root on Sherlock login node):
#   # Submit all variant folders:
#   bash experiments/scripts/submit_rand_splits_sherlock.sh
#
#   # Submit a single variant folder:
#   bash experiments/scripts/submit_rand_splits_sherlock.sh experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits

cd /home/users/salilg/tabpfn_data_scarcity

CONFIG_ROOT="experiments/configs/geo_pooling/sherlock/v2_no_onehot"

# If a folder is passed as argument, use only that; otherwise glob all *_randsplits/
if [ -n "$1" ]; then
    FOLDERS=("$1")
else
    FOLDERS=("$CONFIG_ROOT"/*_randsplits/)
fi

for FOLDER in "${FOLDERS[@]}"; do
    for CONFIG in "${FOLDER%/}"/*.yaml; do
        JOB_NAME=$(basename "$CONFIG" .yaml)
        echo "Submitting: $CONFIG (job name: $JOB_NAME)"
        N_CHUNKS=4 sbatch --job-name="$JOB_NAME" --array=0-3 \
            experiments/slurm/sherlock/geo_pooling.sh "$CONFIG"
    done
done
