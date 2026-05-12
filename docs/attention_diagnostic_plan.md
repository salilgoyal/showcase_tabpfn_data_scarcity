# Attention Score Diagnostic Plan

## Goal

Extract final-layer attention scores from TabPFN to assess whether finetuning
improves attention-based similarity signals. Based on the Yandex paper's analysis
of how TabPFN uses attention to weight training examples.

## Background

TabPFN uses in-context learning: training data is passed as context, and
predictions are made via attention over the context. The attention weights
indicate which training examples the model considers most relevant for each
test prediction.

The diagnostic compares:
1. **Zero-shot TabPFN**: attention patterns from the pre-trained model
2. **Finetuned TabPFN**: attention patterns after global finetuning

If finetuning improves the model, we'd expect:
- More focused attention on truly relevant training examples
- Better correlation between attention-weighted target averages and actual test targets

## Architecture Details

### TabPFN v2 Attention

- Custom `MultiHeadAttention` in `src/models/tabpfn_lib/multi_head_attention.py`
- `PerFeatureEncoderLayer` in `src/models/tabpfn_lib/layer.py` (line 95)
  - Has both **feature-level** and **item-level** attention
  - Item-level attention (`self_attn_between_items`) is what we care about
- Three backends: Flash Attention, SDPA, manual softmax
- **Only the manual softmax path** computes explicit attention weights:
  ```python
  # multi_head_attention.py, lines 722-730
  logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
  logits *= scale
  ps = torch.softmax(logits, dim=2)   # <-- attention weights
  attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)
  ```

### Approach: Monkey-Patch Extraction

Rather than modifying the core attention code (risky, affects all users), use
a standalone diagnostic script that temporarily patches the attention to capture
weights:

```python
from src.models.tabpfn_lib.multi_head_attention import MultiHeadAttention

def extract_attention_weights(model, x_num, x_cat, y_train, device):
    """Run inference and capture last-layer item attention weights."""
    captured = {}
    original_fn = MultiHeadAttention.scaled_dot_product_attention_simple_general

    @staticmethod
    def patched_fn(q, k, v, dropout_p=0.0, softmax_scale=None,
                   share_kv_across_n_heads=1):
        # Force manual softmax path (no Flash/SDPA)
        k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
        v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)

        d_k = q.shape[-1]
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= (1.0 / d_k) ** 0.5 if softmax_scale is None else softmax_scale
        ps = torch.softmax(logits, dim=2)

        # Capture attention weights from the call
        captured['attn_weights'] = ps.detach().cpu()

        attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)
        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        nhead = q.shape[2]
        d_v = v.shape[-1]
        return attention_head_outputs.reshape(batch_size, seqlen_q, nhead, d_v)

    # Patch, run, un-patch
    MultiHeadAttention.scaled_dot_product_attention_simple_general = patched_fn
    try:
        with torch.inference_mode():
            model.eval()
            logits = model(x_num=x_num, x_cat=x_cat, y_train=y_train)
    finally:
        MultiHeadAttention.scaled_dot_product_attention_simple_general = original_fn

    return captured.get('attn_weights')
```

**Note**: This captures attention from ALL layers (the last call to
`scaled_dot_product_attention_simple_general` within each forward pass). To
capture only the **final layer's item-level attention**, we'd need to track
which layer is calling the function. One approach: use a counter or register
a forward hook specifically on the last layer's `self_attn_between_items`.

### Alternative: Forward Hook Approach

```python
captured = {}

def hook_fn(module, input, output):
    # MultiHeadAttention doesn't return attention weights,
    # so we'd need the monkey-patch approach above
    pass

# Better: hook on the PerFeatureEncoderLayer to capture intermediate state
last_layer = model.transformer_encoder.layers[-1]
handle = last_layer.register_forward_hook(...)
```

The hook approach is cleaner but requires the attention module to expose
weights, which it currently doesn't. The monkey-patch approach is simpler.

## Diagnostic Computation

For each test sample i with attention weights `attn[i, j]` over training
samples j (averaged across heads):

```python
# Average attention across heads: (n_test, n_train)
attn_avg = attn_weights.mean(dim=-1)  # average over heads

# Only look at test→train attention (not test→test)
# attn_avg[:, :train_size] are the weights over training samples
train_attn = attn_avg[:, :train_size]

# Normalize to sum to 1
train_attn = train_attn / train_attn.sum(dim=1, keepdim=True)

# Weighted average of training targets using attention
weighted_pred = (train_attn * y_train.unsqueeze(0)).sum(dim=1)

# Compare to actual test target
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

r2_attn = r2_score(y_test, weighted_pred)
corr, pval = pearsonr(y_test.numpy(), weighted_pred.numpy())

diagnostic = {
    'r2_attention_weighted': r2_attn,
    'correlation': corr,
    'p_value': pval,
}
```

## Files to Create

- `src/diagnostics/__init__.py`
- `src/diagnostics/attention_analysis.py` — main diagnostic functions
- `scripts/run_attention_diagnostic.py` — standalone CLI script

## Usage (Proposed)

```bash
# Compare zero-shot vs finetuned attention for a specific county
python scripts/run_attention_diagnostic.py \
  --data_path /nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/ \
  --test_set_dir /nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/test_v4/ \
  --fips 6037 \
  --finetuned_checkpoint /nlp/scr/.../global_finetuned_external/ \
  --output_dir results/attention_diagnostic/
```

## Performance Notes

- Manual softmax attention is O(n^2) vs Flash Attention's O(n).
- For a county with 500 training + 100 test samples, the attention matrix
  is 600x600x6 (heads) which is ~8.6M floats (~34MB). Very manageable.
- Run on a few representative counties, not all 525.
- GPU required (model is large), but inference is fast for small batches.

## Expected Outputs

1. **Attention heatmap**: For a few test samples, show which training samples
   receive highest attention weight
2. **R² comparison table**:
   | Model | Attention R² | Actual R² | Gap |
   |-------|-------------|-----------|-----|
   | Zero-shot | 0.XX | 0.XX | 0.XX |
   | Finetuned | 0.XX | 0.XX | 0.XX |
3. **Scatter plots**: attention-weighted predictions vs actual test targets
