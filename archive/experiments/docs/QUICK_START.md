# Quick Start: Using Evelyn's Preprocessing

This is a quick reference for enabling Evelyn's preprocessing in your experiments.

## TL;DR

1. Edit `experiments/config/base_config.yaml`:
   ```yaml
   preprocessing:
     use_evelyn_preprocessing: true
     include_property_chars: false  # Start with minimal features
   ```

2. Run your experiments normally - everything else is handled automatically

3. Results will be on the original price scale (log transformation is handled internally)

## What Changes

### With `use_evelyn_preprocessing: true`

✅ **Automatically applied:**
- Winsorization (1st/99th percentile)
- Log transformation of target
- Feature normalization (StandardScaler)
- Temporal feature generation
- Outlier removal
- One-hot encoding

✅ **No code changes needed:**
- Metrics automatically inverse-transformed
- Results on original price scale
- Compatible with existing analysis code

### Feature Sets

| Setting | Features | Count |
|---------|----------|-------|
| `include_property_chars: false` | Assessed value + census + time | ~21 |
| `include_property_chars: true` | Above + property characteristics | ~110 |

## Example Workflow

### Step 1: Enable Evelyn's Preprocessing

```yaml
# experiments/config/base_config.yaml
preprocessing:
  use_evelyn_preprocessing: true
  include_property_chars: false
```

### Step 2: Run Experiments

```bash
# Within-county
cd experiments/scripts
bash 02_launch_within_county_nlprun.sh

# Cross-county
bash 03_launch_cross_county_nlprun.sh
```

### Step 3: Analyze Results

Results are automatically saved to the configured output directory with metrics on the original price scale.

## Comparing Preprocessing Methods

To compare original vs. Evelyn's preprocessing:

1. Run experiments with `use_evelyn_preprocessing: false` → Save results to `results/original/`
2. Run experiments with `use_evelyn_preprocessing: true` → Save results to `results/evelyn/`
3. Use `04_aggregate_results.py` to compare

Both result sets will have metrics on the same scale, making them directly comparable.

## Important Notes

⚠️ **Target is log-transformed** - The framework handles this automatically, but be aware:
- Models train on log(price)
- Predictions are inverse-transformed before metrics
- All reported metrics are on original scale

✅ **Metrics are comparable** - You can directly compare MAE, R², RMSE between preprocessing methods

📊 **Feature counts differ** - Original keeps all numeric columns, Evelyn uses curated set

## Need More Details?

See the full guide: `experiments/docs/PREPROCESSING_GUIDE.md`

## Common Questions

**Q: Do I need to change my code?**
- No - just update the config file

**Q: Are the metrics on log scale?**
- No - metrics are automatically computed on original price scale

**Q: Which feature set should I use?**
- Start with minimal (`include_property_chars: false`)
- Try full features if you want to include property characteristics

**Q: Can I switch between methods?**
- Yes - just toggle `use_evelyn_preprocessing` in the config
