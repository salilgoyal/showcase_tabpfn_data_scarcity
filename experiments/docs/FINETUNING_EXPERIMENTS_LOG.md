# TabPFN Finetuning Experiments Log

This log covers experiments since the Yandex-style direct finetuning implementation (v2).
Anything prior to that (v1, API-based approaches) is not documented here.

## Implementation: Yandex-style Direct Finetuning (v2)

**Code**: `src/models/tabpfn_finetuning_v2.py`

Key design:
- Loads TabPFN v2 checkpoint directly (bypasses high-level API)
- Standard PyTorch training loop with stable optimizer references
- Context = ALL remaining training samples (not subsampled)
- Query = `seq_len_pred` samples sampled per step
- Bar distribution loss (5000 bins) for regression
- Full model finetuning (all 11M parameters trainable)
- Flash Attention + bfloat16 AMP enabled

Reference: Yandex tabpfn-finetuning paper (https://arxiv.org/abs/2506.08982)

---

## Experiment Runs

### Run: test_v4_train_v1 — Job 14558970 (Feb 19 2026, NLP cluster, sphinx8)
**Config**: `experiments/configs/finetuning/nlp/test_v4_train_v1.yaml`
**Status**: FAILED — CUDA OOM

Settings:
- `seq_len_pred: 1024`, `batch_size: 1`, `epoch_size: 10`, `max_epochs: 100`
- Train: 10,000 samples (8000 train / 2000 val after split), 66 counties
- Test: 25,281 samples, 525 counties
- GPU: A100-SXM4-80GB (81,920 MiB)

Result: OOM during first training step — 78.97 GiB in use, tried to allocate 552 MiB
more in the einsum op in multi-head attention. `seq_len_pred=1024` fills essentially
all of an 80GB GPU.

Fix applied: Reduced `seq_len_pred` to 256.

---

### Run: test_v4_train_v1 — Job 14559307 (Feb 19 2026, NLP cluster)
**Config**: `experiments/configs/finetuning/nlp/test_v4_train_v1.yaml`
**Status**: FAILED — CUDA OOM

Settings:
- `seq_len_pred: 256`, `batch_size: 1`, `epoch_size: 10`, `max_epochs: 100`
- Same train/test split as above
- GPU: 47.40 GiB total (A6000 or A100-40GB — NOT the 80GB sphinx8 node)
- Zero-shot val R²: **0.6592** (before any finetuning)

Result: OOM during first training step — 46.28 GiB in use, tried to allocate 1.24 GiB
in GELU op in MLP layer. Even with `seq_len_pred=256`, context of 8000 samples is
too large for a 47GB GPU.

Note: This ran on a smaller GPU than intended. The 80GB A100 (`sphinx[3-6,8]`) should
have enough headroom for `seq_len_pred=256` — need to confirm by running on sphinx8
explicitly.

---

### Run: test_v4_train_v1 — Job 17801629 (Mar 4 2026, Sherlock cluster, sh03-11n12)
**Config**: `experiments/configs/finetuning/sherlock/test_v4_train_v1.yaml`
**Status**: FAILED — CUDA OOM

Settings:
- `seq_len_pred: 256`, `batch_size: 1`, `epoch_size: 10`, `max_epochs: 100`
- Train: 10,000 samples (8000 train / 2000 val after split), 66 counties
- Test: 25,281 samples, 525 counties
- GPU: NVIDIA A100-SXM4-80GB (81,920 MiB)

Result: OOM during first training step — 78.96 GiB in use, tried to allocate 308 MiB
in einsum op in multi-head attention. Even with `seq_len_pred=256`, the context size of
~7744 samples (8000 - 256 query samples) creates a sequence length of 8000 tokens,
which fills the 80GB GPU.

**Key insight**: The bottleneck is **context size**, not query size. The Yandex approach
uses ALL remaining training samples as context on each forward pass, resulting in
O(n²) memory for attention where n = train_size.

Fix applied: Added `max_context_size: 512` to config to cap context at 512 samples.

---

### Run: test_v4_train_v1 — Job 17804433 (Mar 4 2026, Sherlock cluster, sh03-11n12)
**Config**: `experiments/configs/finetuning/sherlock/test_v4_train_v1.yaml`
**Status**: SUCCESS

Settings:
- `seq_len_pred: 256`, `max_context_size: 512`, `batch_size: 1`, `epoch_size: 10`
- Train: 10,000 samples (8000 train / 2000 val after split), 66 counties
- Test: 25,281 samples, 525 counties
- GPU: NVIDIA A100-SXM4-80GB (81,920 MiB)
- Total sequence length per forward pass: 512 (context) + 256 (query) = 768 tokens

**Training dynamics**:
- Zero-shot val R²: 0.6577
- Training loss: 0.7986 → -0.4127 (epoch 43, consistent decrease)
- Validation R²: 0.6484 → 0.6664 (best at epoch 27, then plateaus)
- Early stopping at epoch 43 (16 epochs without improvement)
- **Net improvement: +0.0087 absolute (+1.3% relative)**

**Test set performance**:
- TabPFN (finetuned): R² = 0.526, MAE = 72,689, RMSE = 132,273
- XGBoost (baseline): R² = 0.476, MAE = 74,753, RMSE = 139,097
- County baseline: R² = -1.52, MAE = 103,023, RMSE = 305,050

**Analysis**:
- Memory issue resolved by capping context at 512 samples
- Training loss decreases but val R² barely improves → **overfitting**
- Test R² (0.526) lower than val R² (0.666) → distribution shift or overfitting
- **Critical limitation**: Using only 512/8000 = 6.4% of training data per step
- TabPFN outperforms XGBoost by ~5 R² points

**Next steps to try**:
1. Increase `max_context_size` to 1024-2048 to utilize more training data
2. Enable gradient checkpointing (`recompute_each_layer=True`) for ~40-50% memory savings
3. Combine larger context + checkpointing to potentially fit 2000-4000 context samples
4. If overfitting persists: lower LR (5e-5), add weight decay (0.01), reduce seq_len_pred to 128
5. Consider whether finetuning complexity is justified vs zero-shot TabPFN

---

### Run: test_v4_train_v1 — Job 17811957 (Mar 4 2026, Sherlock cluster, sh03-11n12)
**Config**: `experiments/configs/finetuning/sherlock/test_v4_train_v1.yaml`
**Status**: SUCCESS (with concerns)

**Changes from previous run**:
- Enabled gradient checkpointing: `recompute_each_layer=True` in `src/models/tabpfn_finetuning_v2.py:247`
- Increased context size: `max_context_size: 2000` (was 512)

Settings:
- `seq_len_pred: 256`, `max_context_size: 2000`, `batch_size: 1`, `epoch_size: 10`
- Train: 10,000 samples (8000 train / 2000 val after split), 66 counties
- Test: 25,281 samples, 525 counties
- GPU: NVIDIA A100-SXM4-80GB (81,920 MiB)
- Total sequence length per forward pass: 2000 (context) + 256 (query) = 2256 tokens

**Training dynamics**:
- Zero-shot val R²: 0.6586
- Training loss: 0.8128 → -0.6714 (epoch 54, consistent decrease)
- Validation R²: 0.6586 → 0.6779 (best at epoch 38)
- Early stopping at epoch 54 (16 epochs without improvement)
- **Net improvement: +0.0193 absolute (+2.9% relative)**
- Training speed: ~5.8s/epoch (vs ~1.2s with 512 context) = **4.8x slower due to gradient checkpointing**
- Total runtime: 12.77 minutes

**Test set performance**:
- TabPFN (finetuned): R² = 0.500, MAE = 73,906, RMSE = 135,820
- XGBoost (baseline): R² = 0.476, MAE = 74,753, RMSE = 139,097
- County baseline: R² = -1.52, MAE = 103,023, RMSE = 305,050

**Analysis - CONCERNING RESULT**:
- ✅ Memory: No OOM with 2256 sequence length (gradient checkpointing works)
- ✅ Validation: Better val R² (0.6779 vs 0.6664 with 512 context)
- ❌ **Test performance WORSE: R² dropped from 0.526 → 0.500 (2.6 point drop)**
- ❌ **Overfitting INCREASED: val-test gap grew from 0.14 → 0.178**

**Key insight**: Larger context (2000 vs 512) improved validation metrics but **degraded generalization**.
The model is learning training-county-specific patterns that don't transfer to test counties.

**Comparison with 512 context run**:
| Metric | Context=512 | Context=2000 | Change |
|--------|-------------|--------------|--------|
| Val R² | 0.6664 | 0.6779 | +0.0115 ✅ |
| Test R² | 0.526 | 0.500 | -0.026 ❌ |
| Overfitting gap | 0.140 | 0.178 | +0.038 ❌ |
| Training time | 19 min | 13 min | -6 min ✅ |

**Conclusion**: More context ≠ better generalization. The model needs regularization:
- Weight decay to prevent overfitting
- Lower learning rate
- Or accept that 512 context may be the sweet spot for this dataset

---

## Open Questions / Next Steps

**Key finding**: Larger context (2000) improved validation but **hurt test performance** (0.500 vs 0.526 with 512 context). The model overfits more with more training data per step.

**Next experiments to try**:
1. **Add regularization with 2000 context**:
   - Weight decay: 0.01 or 0.05
   - Lower learning rate: 5e-5 (currently 1e-4)
   - Earlier stopping (patience: 8 instead of 16)
2. **Try intermediate context sizes**: 1000, 1500 to find sweet spot
3. **Return to 512 context** as baseline (best test R² so far)
4. **Consider zero-shot TabPFN** without finetuning - may generalize better

**Open questions**:
- Why does more context hurt generalization? County distribution shift?
- Is 512 the optimal context size for this dataset?
- Is finetuning worth the complexity vs zero-shot TabPFN?
- Should we reduce training set diversity (single county source) to reduce distribution shift?
