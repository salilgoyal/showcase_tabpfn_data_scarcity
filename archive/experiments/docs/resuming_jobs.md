# Resuming Interrupted Jobs

The helper scripts support automatic resumption of interrupted experiments by skipping already-completed counties.

## How It Works

Both `02_launch_within_county_nlprun.sh` and `03_launch_cross_county_nlprun.sh` check for existing result files and automatically skip counties/iterations that have already completed.

### What Gets Checked:

**Within-County Experiment:**
- Looks for `results/within_county/county_{FIPS}_results.csv`
- If the file exists, that county is skipped

**Cross-County Experiment:**
- Looks for `results/cross_county/county_{FIPS}_iter_{N}_results.csv`
- If the file exists, that (county, iteration) pair is skipped

## Usage

### Default Behavior: Skip Completed Jobs

Simply run the script normally and it will automatically skip completed work:

```bash
# Within-county: Resumes where you left off
bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 10

# Cross-county: Resumes where you left off
bash experiments/scripts/03_launch_cross_county_nlprun.sh --max-parallel 10
```

The script will show you progress:
```
Total counties to process: 101
Already completed: 23
Remaining: 78
```

### Force Rerun All Jobs

If you want to rerun everything (ignoring completed results):

```bash
# Rerun all counties
bash experiments/scripts/02_launch_within_county_nlprun.sh --rerun-all

# Or use --no-skip (same thing)
bash experiments/scripts/02_launch_within_county_nlprun.sh --no-skip
```

## Example Scenarios

### Scenario 1: Job Got Interrupted

You started with:
```bash
bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 10
```

Only 23 out of 101 counties completed before something went wrong. Simply run the same command again:

```bash
bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 10
```

The script will:
- ✓ Check all 101 counties
- ✓ Skip the 23 that already have results
- ✓ Launch jobs for the remaining 78 counties

### Scenario 2: Want to Add More Parallel Jobs

You initially ran with 5 parallel jobs, but now you have more resources:

```bash
# Initial run (some completed)
bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 5

# Later: Resume with more parallelism
bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 20
```

This will skip completed counties and launch the remaining ones with higher parallelism.

### Scenario 3: A Few Jobs Failed

Some jobs failed due to issues (OOM, data problems, etc.). You fixed the issues and want to retry only the failed jobs:

1. **Identify failed jobs** (no result file created)
2. **Delete their output files** (optional, for clean logs):
   ```bash
   rm experiments/outfiles/within_county_XXXX.out
   ```
3. **Rerun the script** - it will automatically retry missing results:
   ```bash
   bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 10
   ```

### Scenario 4: Need to Rerun Specific Counties

If you need to rerun specific counties (e.g., to test changes):

1. **Delete their result files**:
   ```bash
   rm results/within_county/county_1007_results.csv
   rm results/within_county/county_1011_results.csv
   ```
2. **Rerun the script**:
   ```bash
   bash experiments/scripts/02_launch_within_county_nlprun.sh --max-parallel 10
   ```

Only the counties with missing result files will be rerun.

## Checking Status

To see what's completed vs. remaining:

```bash
# Count completed within-county jobs
ls -1 results/within_county/county_*_results.csv | wc -l

# Count completed cross-county jobs
ls -1 results/cross_county/county_*_iter_*_results.csv | wc -l

# Total counties
tail -n +2 small_county_metadata.csv | wc -l
```

## Important Notes

1. **Result files are the source of truth**: The script checks for CSV result files, not output logs
2. **Failed jobs without results**: If a job failed before writing results, it will be retried automatically
3. **Partial results**: If a job crashed mid-execution, delete its result file to rerun it
4. **No result file = job needs to run**: The script will launch jobs for any county without a result file

## Disabling Auto-Skip

If you want to disable the auto-skip feature permanently, edit the script and change:

```bash
SKIP_COMPLETED=true  # Change to false
```

Or use `--no-skip` flag when running.
