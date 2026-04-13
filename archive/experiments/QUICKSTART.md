# Quick Start Guide

Get your experiments running in 3 steps:

## Step 1: Setup and Test

```bash
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts

# Test the setup
python test_setup.py

# Create county registry (identifies all small counties)
python 00_create_county_registry.py
```

Expected output: You should see a list of small counties (10-100 observations) and their bin assignments.

## Step 2: Launch Experiments

### Option A: Using nlprun (Recommended)

```bash
# Launch within-county experiment (10 jobs in parallel)
bash 02_launch_within_county_nlprun.sh --max-parallel 10

# Launch cross-county experiment (10 jobs in parallel)
bash 03_launch_cross_county_nlprun.sh --max-parallel 10
```

Monitor progress:
```bash
nlpjobs  # Check running jobs
tail -f ../../../logs/within_county_*.out  # Watch a specific job
```

### Option B: Using SLURM

```bash
# Get number of counties
NUM_COUNTIES=$(tail -n +2 /sailhome/salilg/tabpfn_data_scarcity/small_county_metadata.csv | wc -l)

# Launch within-county experiment
sbatch --array=0-$((NUM_COUNTIES-1)) slurm_01_within_county.sh

# Launch cross-county experiment (10 iterations per county)
sbatch --array=0-$((NUM_COUNTIES*10-1)) slurm_02_cross_county.sh
```

## Step 3: Aggregate and Analyze Results

After jobs complete:

```bash
# Aggregate results
python 04_aggregate_results.py --experiment within_county
python 04_aggregate_results.py --experiment cross_county
```

Results will be saved to:
- `/sailhome/salilg/tabpfn_data_scarcity/results/within_county/`
- `/sailhome/salilg/tabpfn_data_scarcity/results/cross_county/`

## Adjusting Configuration

Edit config files to customize:
- `config/base_config.yaml` - County size bins, models, metrics
- `config/within_county_config.yaml` - Nested CV settings
- `config/cross_county_config.yaml` - Pooled training settings

## Common Issues

**"No GPU available"**: Models will run on CPU (slower but will work)

**Out of memory**: Increase memory in launch scripts:
- nlprun: change `-r 32G` to `-r 64G`
- SLURM: change `#SBATCH --mem=32G` to `#SBATCH --mem=64G`

**Job failures**: Check logs in `/sailhome/salilg/tabpfn_data_scarcity/logs/`

## Next Steps

See `README.md` for full documentation and `docs/nlprun_commands.md` for detailed command examples.
