# Training Set Versions

This directory contains training set configurations for experiments. Each version defines a different strategy for constructing the training data.

## Overview

All training sets are designed to work with the test set defined in `../test_sets/test_v1.yaml`, which:
- Selects ~36 counties across 5 size buckets (tiny → xlarge)
- Uses top 50% by date (most recent) as test data
- Reserves bottom 50% (historical) as potential training data

## Version Descriptions

### train_v1: Test-County Historical Only

**File**: `train_v1.yaml`

**Strategy**: Use ONLY the historical (bottom 50% by date) data from test counties. No external county data.

**Research Question**: Can the model learn to generalize from a county's own historical data? Tests temporal generalization within counties.

**Expected Performance**:
- Good for large test counties with substantial history
- May struggle for tiny counties with limited data (~25-50 samples)

**Use Case**: Baseline for temporal transfer within counties

---

### train_v2: Mixed (50% Test History, 50% External)

**File**: `train_v2.yaml`

**Strategy**: Use 50% of training budget from test counties' historical data, 50% from random external counties. Capped at 10K samples.

**Research Question**: Does mixing county-specific history with cross-county data help? Tests balanced approach.

**Expected Performance**:
- More robust than v1 for small test counties
- Balances county-specific and general patterns

**Use Case**: Balanced approach combining temporal and cross-county transfer

---

### train_v3: External Counties Only (Random)

**File**: `train_v3.yaml`

**Strategy**: Use ONLY data from external counties, randomly sampled. No test county data. Capped at 10K samples.

**Research Question**: How well does the model generalize to completely unseen counties? Tests pure cross-county transfer.

**Expected Performance**:
- Hardest generalization scenario
- Performance depends on transferability of patterns

**Use Case**: Pure cross-county baseline (no county-specific info)

---

### train_v4: Stratified External Counties

**File**: `train_v4.yaml`

**Strategy**: Sample equally from many external counties (~50 samples each from 200 counties = 10K). Maximizes diversity.

**Research Question**: Does diversity in training counties improve generalization? Compare with v3 (random).

**Expected Performance**:
- Should be robust across different test county sizes
- Trades off volume for diversity

**Use Case**: Maximum diversity training set

---

### train_v5: Large-Scale (For Fine-tuning)

**File**: `train_v5.yaml`

**Strategy**: Use ALL available data from external counties. No sampling limit (millions of samples).

**Research Question**: How does performance scale with training data size? What is the ceiling?

**Expected Performance**:
- NOT suitable for vanilla TabPFN (10K limit)
- Suitable for XGBoost and fine-tuned TabPFN
- Best for understanding scaling limits

**Use Case**: Large-scale experiments, fine-tuning

---

## Summary Table

| Version | Test County Data | External Data | Max Samples | TabPFN Compatible |
|---------|------------------|---------------|-------------|-------------------|
| v1      | 100% historical  | None          | Unlimited   | Depends on size   |
| v2      | 50%              | 50% random    | 10K         | ✅ Yes            |
| v3      | None             | 100% random   | 10K         | ✅ Yes            |
| v4      | None             | 100% stratified| 10K        | ✅ Yes            |
| v5      | None             | 100% all      | Unlimited   | ❌ No (use XGB)   |

## Recommended Experiments

### Initial Comparison
Run v1, v2, v3 to understand the spectrum:
- v1: Temporal transfer baseline
- v2: Balanced approach
- v3: Cross-county transfer baseline

### Diversity Analysis
Compare v3 vs v4 to understand if stratification helps.

### Scaling Analysis
Use v5 with XGBoost to understand large-scale performance ceiling.

## Creating New Versions

To create a new training set version:

1. Copy an existing config: `cp train_v3.yaml train_v6.yaml`
2. Modify the strategy in the new file
3. Update the version number and description
4. Add documentation to this README
5. Create experiment config: `../cross_county/test_v1_train_v6.yaml`
