#!/bin/bash
# Submit geo_pooling array jobs for all randsplit configs.
#
# Iterates over every *_randsplits/ subfolder under v2_no_onehot, submits each
# config as a 4-chunk SLURM array job, and passes --job-name derived from the
# config filename so that log files are named meaningfully (%x in geo_pooling.sh).
#
# Usage:
#   # Submit all variant folders:
#   bash experiments/scripts/submit_rand_splits_nlp.sh
#
#   # Submit a single variant folder:
#   bash experiments/scripts/submit_rand_splits_nlp.sh experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits

cd /sailhome/salilg/tabpfn_data_scarcity

CONFIG_ROOT="experiments/configs/geo_pooling/nlp/v2_no_onehot"

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
            experiments/slurm/nlp/geo_pooling.sh "$CONFIG"
    done
done
