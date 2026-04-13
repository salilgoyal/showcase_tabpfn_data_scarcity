"""
Direct TabPFN v2 finetuning, adapted from the Yandex tabpfn-finetuning approach.

This module loads the TabPFN v2 checkpoint directly (bypassing the high-level API)
and runs a standard PyTorch training loop. This keeps parameter references stable
so the optimizer actually updates the model weights.

Source: https://github.com/yandex-research/tabpfn-finetuning
Paper: https://arxiv.org/abs/2506.08982
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from torch import Tensor

from .base_model import BaseModel
from .tabpfn_lib.bar_distribution import FullSupportBarDistribution
from .tabpfn_lib.border_utils import _transform_borders_one, translate_probs_across_borders
from .tabpfn_lib.layer import PerFeatureEncoderLayer

logger = logging.getLogger(__name__)


# =============================================================================
# Config and History (compatible with existing experiment runner)
# =============================================================================

@dataclass
class FinetuningConfigV2:
    """Configuration for Yandex-style direct TabPFN v2 finetuning."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 16
    epoch_size: int = 10
    seq_len_pred: int = 1024
    max_context_size: Optional[int] = None  # None = use all training samples as context
    min_context_size: int = 5  # minimum context size for diversity sampling (log-uniform)
    batch_size: int = 1
    gradient_clip: float = 1.0
    use_amp: bool = True
    finetune_mode: str = "full"  # "full", "embeds", "head", "lora"
    lora_rank: int = 0        # 0 = no LoRA; >0 = apply LoRA with this rank
    lora_alpha: float = 16.0  # LoRA scaling factor (effective lr ~ alpha/rank)
    target_transform: Optional[str] = None  # None, "power", "quantile"
    checkpoint_path: Optional[str] = None
    n_lr_warmup_epochs: int = 0
    softmax_temperature: float = 0.9
    val_fraction: float = 0.2
    eval_batch_size: int = 4096
    device: str = "cuda"
    random_state: int = 42
    training_mode: str = "global"  # "global" or "per_county"
    min_county_size: int = 5       # minimum county size for per_county mode
    context_fraction_range: Tuple[float, float] = (0.3, 0.7)  # random ctx/query split fraction
    # Spike diagnostics (gated — zero overhead when False)
    spike_diagnostics: bool = False
    spike_threshold: float = 100.0  # log step details when loss exceeds this


@dataclass
class TrainingHistory:
    """Tracks training metrics over epochs."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    train_r2: List[float] = field(default_factory=list)
    val_r2: List[float] = field(default_factory=list)
    val_mae: List[float] = field(default_factory=list)
    val_county_metrics: List[Dict] = field(default_factory=list)
    spike_count: int = 0
    spike_epochs: List[int] = field(default_factory=list)
    zeroshot_val_loss: float = float('nan')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'val_mae': self.val_mae,
            'val_county_metrics': self.val_county_metrics,
            'spike_count': self.spike_count,
            'spike_epochs': self.spike_epochs,
            'zeroshot_val_loss': self.zeroshot_val_loss,
        }


# =============================================================================
# LayerStack (from Yandex bin/tabpfnv2_finetune.py)
# =============================================================================

class LayerStack(nn.Module):
    """Stack of transformer layers with optional layer dropout."""

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            n_layers = self.num_layers // 2
        else:
            if self.training:
                n_layers = torch.randint(
                    low=self.min_num_layers_layer_dropout,
                    high=self.num_layers + 1,
                    size=(1,),
                ).item()
            else:
                n_layers = self.num_layers

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)

        return x


# =============================================================================
# TabPFN2 Model (from Yandex bin/tabpfnv2_finetune.py)
# =============================================================================

class TabPFN2(nn.Module):
    """TabPFN v2 with direct checkpoint loading for finetuning.

    Uses tied embeddings (untie_value_embeddings=False, untie_pos_embeddings=False)
    which is the simplest and recommended default from the Yandex paper.
    """

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: int,
        is_regression: bool,
        checkpoint_path: str,
        tabpfn_config: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.is_regression = is_regression
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities

        # Default TabPFN v2 config
        if tabpfn_config is None:
            tabpfn_config = {
                "emsize": 192,
                "nhead": 6,
                "nlayers": 12,
                "nhid_factor": 4,
            }

        state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]

        extract_state_dict = lambda pref: {
            k.removeprefix(pref): v
            for k, v in state_dict.items()
            if k.startswith(pref)
        }

        emsize = tabpfn_config["emsize"]

        # Feature embeddings (tied mode = shared linear layer)
        if n_num_features > 0:
            self.m_num = nn.Linear(1, emsize, bias=False)
            weight_init = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
            self.m_num.weight.data = weight_init
        else:
            self.m_num = None

        # Categorical embeddings (tied mode = reuse m_num or create new)
        if cat_cardinalities:
            if n_num_features > 0:
                self.m_cat = self.m_num
            else:
                self.m_cat = nn.Linear(1, emsize, bias=False)
                weight_init = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
                self.m_cat.weight.data = weight_init
        else:
            self.m_cat = None

        # Positional embeddings (subspace)
        self.pos_embs = nn.Linear(48, emsize)
        self.pos_embs.load_state_dict(
            extract_state_dict("feature_positional_embedding_embeddings.")
        )

        # Target embeddings
        layer_key = "1" if is_regression else "2"
        self.y_embedding_weight = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 0]
        )
        self.y_embedding_nan_ind = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 1]
        )
        self.y_embedding_bias = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.bias"]
        )

        # Transformer encoder
        ninp = emsize
        nhead = tabpfn_config["nhead"]
        nhid = emsize * tabpfn_config["nhid_factor"]
        nlayers = tabpfn_config["nlayers"]

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation="gelu",
            zero_init=False,
            precomputed_kv=None,
            multiquery_item_attention_for_test_set=True,
            layer_norm_with_elementwise_affine=False,
        )

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers,
            recompute_each_layer=True,  # Enable gradient checkpointing to save memory
            min_num_layers_layer_dropout=None,
        )
        self.transformer_encoder.load_state_dict(
            extract_state_dict("transformer_encoder."), strict=False
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_classes),
        )
        self.decoder.load_state_dict(extract_state_dict("decoder_dict.standard."))

    def forward(
        self,
        *,
        x_num: Optional[Tensor] = None,
        x_cat: Optional[Tensor] = None,
        y_train: Tensor,
    ) -> Tensor:
        bs = y_train.shape[0]
        train_size = y_train.shape[1]

        # Feature embeddings
        x = []
        if x_num is not None:
            x.append(self.m_num(x_num.unsqueeze(-1)))

        if x_cat is not None:
            assert self.m_cat is not None
            # Min-max scale categoricals using train set max
            x_cat_max = x_cat[:, :train_size].max(dim=1, keepdim=True).values
            x_cat_scaled = (x_cat / x_cat_max.clamp(min=1)).unsqueeze(-1)
            x.append(self.m_cat(x_cat_scaled))

        x_inp = torch.cat(x, dim=2)
        total_size = x_inp.shape[1]

        # Target embeddings
        y_mult = y_train.mean(dim=1, keepdim=True)
        if not self.is_regression:
            y_mult = torch.round(y_mult)
        y_test = x_inp.new_ones(bs, total_size - train_size) * y_mult
        nan_ind = x_inp.new_zeros(bs, total_size)
        nan_ind[:, train_size:] = -2.0
        y_emb = (
            torch.cat([y_train, y_test], dim=1).view(bs, -1, 1, 1).float()
            * self.y_embedding_weight.view(1, 1, 1, -1)
            + nan_ind.view(bs, -1, 1, 1)
            * self.y_embedding_nan_ind.view(1, 1, 1, -1)
            + self.y_embedding_bias.view(1, 1, 1, -1)
        )

        # Subspace positional embeddings
        if self.pos_embs is not None:
            _, _, n_features, d_emb = x_inp.shape
            x_inp = (
                x_inp
                + self.pos_embs(
                    torch.randn(n_features, d_emb // 4, device=x_inp.device)
                )[None, None]
            )

        # Transformer forward pass
        x_inp = torch.cat([x_inp, y_emb], dim=2)
        encoder_out = self.transformer_encoder(
            x_inp,
            half_layers=False,
            cache_trainset_representation=False,
            single_eval_pos=train_size,
        )

        return self.decoder(encoder_out[:, train_size:, -1])


# =============================================================================
# CandidateQueue (from Yandex bin/tabpfnv2_finetune.py)
# =============================================================================

class CandidateQueue:
    """Efficient shuffled sampling of training indices without replacement."""

    def __init__(
        self, train_size: int, n_candidates: int, device: torch.device
    ) -> None:
        assert train_size > 0
        assert 0 < n_candidates < train_size
        self._n_candidates = n_candidates
        self._train_size = train_size
        self._candidate_queue = torch.tensor([], dtype=torch.int64, device=device)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._candidate_queue) < self._n_candidates:
            self._candidate_queue = torch.cat([
                self._candidate_queue,
                torch.randperm(self._train_size, device=self._candidate_queue.device),
            ])
        candidate_indices, self._candidate_queue = self._candidate_queue.split(
            [self._n_candidates, len(self._candidate_queue) - self._n_candidates]
        )
        return candidate_indices


# =============================================================================
# LoRA (Low-Rank Adaptation)
# =============================================================================

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear.

    Freezes the original weight and adds trainable low-rank matrices A and B
    so that the effective weight becomes W + (B @ A) * (alpha / rank).
    B is zero-initialized so the model starts as the original.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # Low-rank decomposition: out = original(x) + x @ A^T @ B^T * scaling
        # Create on same device as original to avoid device mismatch after .to(device)
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero → LoRA contribution starts at zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_out = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A),
            self.lora_B,
        )
        return result + lora_out * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_names: Optional[list] = None,
) -> None:
    """Replace targeted nn.Linear layers with LoRA-wrapped versions in-place.

    Targets MLP layers and decoder by default. Attention Q/K/V/out are raw
    nn.Parameter tensors (not nn.Linear) and are left untouched — this
    preserves the ICL mechanism in the pretrained attention weights.

    Args:
        model: The TabPFN2 model to modify in-place.
        rank: LoRA rank (number of low-rank dimensions).
        alpha: LoRA scaling factor.
        target_names: Substrings to match in module names. Defaults to
            ["mlp.linear1", "mlp.linear2", "decoder.0", "decoder.2"].
    """
    if target_names is None:
        target_names = ["mlp.linear1", "mlp.linear2", "decoder.0", "decoder.2"]

    # Collect replacements (can't modify dict during iteration)
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            replacements.append((name, module))

    for name, module in replacements:
        # Navigate to parent module and replace child
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))

    # Freeze all non-LoRA parameters
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA applied: {n_lora:,} trainable / {n_total:,} total params "
                f"(rank={rank}, alpha={alpha})")


# =============================================================================
# Regression output transform (from Yandex)
# =============================================================================

def regression_output_transform(
    target_transform,
    criterion: FullSupportBarDistribution,
    renormalized_criterion: FullSupportBarDistribution,
    softmax_temperature: float = 0.9,
    device: torch.device = torch.device("cpu"),
):
    """Transform model logits into original-scale predictions for regression."""
    std_borders = criterion.borders.cpu().numpy()
    logit_cancel_mask, descending_borders, borders_t = _transform_borders_one(
        std_borders,
        target_transform=target_transform,
        repair_nan_borders_after_transform=True,
    )
    if descending_borders:
        borders_t = borders_t[::-1].copy()

    def transform(out):
        logits = translate_probs_across_borders(
            out.float() / softmax_temperature,
            frm=torch.as_tensor(borders_t, device=device),
            to=criterion.borders.to(device),
        )
        if logit_cancel_mask is not None:
            out = out.clone()
            out[..., logit_cancel_mask] = float("-inf")

        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        return renormalized_criterion.mean(logits)

    return transform


# =============================================================================
# DirectFineTunedTabPFNModel (main wrapper)
# =============================================================================

class DirectFineTunedTabPFNModel(BaseModel):
    """Direct TabPFN v2 finetuning using the Yandex approach.

    Loads the checkpoint directly, constructs context/query batches,
    and runs a standard PyTorch training loop with stable optimizer references.
    """

    def __init__(self, config: FinetuningConfigV2):
        super().__init__(random_state=config.random_state)
        self.config = config
        self.history = TrainingHistory()
        self.model: Optional[TabPFN2] = None
        self.is_fitted = False

        # Data storage for prediction
        self._train_data: Optional[Dict[str, Tensor]] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._target_transform = None
        self._pred_transform = None
        self._criterion = None  # raw bar distribution (normalized borders)
        self._continuous_cols: Optional[List[str]] = None
        self._all_columns: Optional[List[str]] = None
        self._cat_cardinalities: Optional[List[int]] = None
        self._best_model_state: Optional[Dict] = None

    def get_name(self) -> str:
        return "tabpfn_v2_finetuned"

    def get_training_history(self) -> TrainingHistory:
        return self.history

    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------

    def _prepare_features(
        self,
        X: pd.DataFrame,
        continuous_cols: List[str],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], List[int]]:
        """Split DataFrame into numerical and categorical tensors."""
        all_cols = list(X.columns)
        cat_cols = [c for c in all_cols if c not in continuous_cols]

        x_num = None
        if continuous_cols:
            present_continuous = [c for c in continuous_cols if c in X.columns]
            if present_continuous:
                x_num = torch.tensor(
                    X[present_continuous].fillna(0.0).values, dtype=torch.float32
                )

        x_cat = None
        cat_cardinalities = []
        if cat_cols:
            present_cat = [c for c in cat_cols if c in X.columns]
            if present_cat:
                # Fill NaN before casting — NaN from float columns becomes
                # overflow when cast directly to int64, corrupting cardinalities
                cat_data_float = X[present_cat].values.astype(float)
                cat_data = np.where(
                    np.isnan(cat_data_float), 0, cat_data_float
                ).astype(np.int64)
                x_cat = torch.tensor(cat_data, dtype=torch.float32)
                # Compute cardinalities (max value + 2 for unknown token)
                cat_cardinalities = [
                    int(cat_data[:, i].max()) + 2 for i in range(cat_data.shape[1])
                ]

        return x_num, x_cat, cat_cardinalities

    # -------------------------------------------------------------------------
    # Fit (Training Loop)
    # -------------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        continuous_cols: Optional[List[str]] = None,
        county_ids: Optional[pd.Series] = None,
    ) -> None:
        """Finetune TabPFN v2 using the Yandex direct-checkpoint approach.

        If X_val/y_val are not provided, an internal validation split is created
        using config.val_fraction and config.random_state for reproducibility.

        If continuous_cols is not provided, continuous columns are auto-detected
        from X_train dtypes (numeric columns with more than 2 unique values).

        If county_ids is provided and config.training_mode == "per_county",
        each training step uses one county's data for both context and query.
        """
        config = self.config
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        logger.info("=" * 60)
        logger.info("Starting Yandex-style direct TabPFN v2 finetuning")
        logger.info(f"  Training mode: {config.training_mode}")
        logger.info("=" * 60)

        # --- Auto-detect continuous columns if not provided ---
        if continuous_cols is None:
            continuous_cols = [
                col for col in X_train.columns
                if pd.api.types.is_numeric_dtype(X_train[col])
                and X_train[col].dropna().nunique() > 2
            ]
            logger.info(f"  Auto-detected {len(continuous_cols)} continuous columns")

        # --- Dispatch to per-county mode if configured ---
        if config.training_mode == "per_county" and county_ids is not None:
            return self._fit_per_county(X_train, y_train, county_ids, continuous_cols)
        elif config.training_mode == "per_county" and county_ids is None:
            logger.warning("training_mode='per_county' but no county_ids provided — falling back to global mode")

        # --- Internal val split if no val set provided ---
        # Save full dataset before split — used as inference context after training
        X_full, y_full = X_train, y_train
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=config.val_fraction,
                random_state=config.random_state,
            )
            logger.info(
                f"  Internal val split: {len(X_train)} train, {len(X_val)} val "
                f"(val_fraction={config.val_fraction}, seed={config.random_state})"
            )

        # --- Store column info ---
        self._all_columns = list(X_train.columns)
        self._continuous_cols = continuous_cols

        # --- Prepare features ---
        x_num_train, x_cat_train, cat_cardinalities = self._prepare_features(
            X_train, self._continuous_cols
        )
        self._cat_cardinalities = cat_cardinalities

        n_num_features = x_num_train.shape[1] if x_num_train is not None else 0
        logger.info(f"  Numerical features: {n_num_features}")
        logger.info(f"  Categorical features: {len(cat_cardinalities)}")
        logger.info(f"  Training samples: {len(X_train)}")

        # --- Per-batch y normalization: store raw targets (no global standardize) ---
        # Global stats are kept as metadata; actual normalization is per-mini-batch
        # in the training loop, matching per-county normalization at inference time.
        y_np = y_train.values.astype(np.float32)
        self._y_mean = float(y_np.mean())  # metadata only
        self._y_std = float(y_np.std())    # metadata only
        y_standardized = y_np              # raw; normalized per-batch in training loop

        # --- Optional target transform ---
        if config.target_transform == "power":
            self._target_transform = Pipeline([
                ("power", PowerTransformer()),
                ("standard", StandardScaler()),
            ]).fit(y_standardized.reshape(-1, 1))
            y_standardized = self._target_transform.transform(
                y_standardized.reshape(-1, 1)
            ).astype(np.float32).squeeze()
        elif config.target_transform == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            self._target_transform = QuantileTransformer(
                output_distribution="normal", random_state=config.random_state
            ).fit(y_standardized.reshape(-1, 1))
            y_standardized = self._target_transform.transform(
                y_standardized.reshape(-1, 1)
            ).astype(np.float32).squeeze()
        else:
            self._target_transform = FunctionTransformer(func=None)

        # --- Move data to device ---
        Y_train = torch.tensor(y_standardized, dtype=torch.float32, device=device)
        if x_num_train is not None:
            x_num_train = x_num_train.to(device)
        if x_cat_train is not None:
            x_cat_train = x_cat_train.to(device)

        # Store training data for prediction-time context
        self._train_data = {
            "x_num": x_num_train,
            "x_cat": x_cat_train,
            "y": Y_train,
        }

        # --- Prepare validation data ---
        x_num_val = x_cat_val = Y_val = None
        if X_val is not None and y_val is not None:
            x_num_val_t, x_cat_val_t, _ = self._prepare_features(X_val, self._continuous_cols)
            y_val_np = y_val.values.astype(np.float32)
            # Store raw val targets; _evaluate_val normalizes using Y_train stats
            Y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
            if x_num_val_t is not None:
                x_num_val = x_num_val_t.to(device)
            if x_cat_val_t is not None:
                x_cat_val = x_cat_val_t.to(device)

        # --- Find checkpoint ---
        checkpoint_path = self._find_checkpoint(config.checkpoint_path)
        logger.info(f"  Checkpoint: {checkpoint_path}")

        # --- Build model ---
        self.model = TabPFN2(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=5000,
            is_regression=True,
            checkpoint_path=checkpoint_path,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Total parameters: {n_params:,}")

        # --- Set up loss ---
        borders = torch.load(checkpoint_path, weights_only=True)[
            "state_dict"
        ]["criterion.borders"].to(device)
        loss_fn = FullSupportBarDistribution(borders)
        self._criterion = loss_fn
        renormalized_criterion = FullSupportBarDistribution(
            loss_fn.borders * self._y_std + self._y_mean,
        ).float()
        self._pred_transform = regression_output_transform(
            self._target_transform,
            loss_fn,
            renormalized_criterion,
            softmax_temperature=config.softmax_temperature,
            device=device,
        )

        # --- Set up optimizer ---
        if config.finetune_mode == "full":
            params = list(self.model.parameters())
        elif config.finetune_mode == "embeds":
            params = [
                p for n, p in self.model.named_parameters()
                if "m_num" in n or "m_cat" in n or "y_embedding" in n or "pos_embs" in n
            ]
            for n, p in self.model.named_parameters():
                if not any(k in n for k in ("m_num", "m_cat", "y_embedding", "pos_embs")):
                    p.requires_grad = False
        elif config.finetune_mode == "head":
            params = [
                p for n, p in self.model.named_parameters()
                if "decoder.2" in n
            ]
            for n, p in self.model.named_parameters():
                if "decoder.2" not in n:
                    p.requires_grad = False
        elif config.finetune_mode == "lora":
            apply_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Default: full
            params = list(self.model.parameters())

        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        logger.info(f"  Trainable parameters: {n_trainable:,}")

        optimizer = torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # --- LR scheduler ---
        lr_scheduler = None
        if config.n_lr_warmup_epochs > 0:
            n_warmup_steps = min(
                10000, config.n_lr_warmup_epochs * config.epoch_size
            )
            n_warmup_steps = max(
                1, math.trunc(n_warmup_steps / config.epoch_size)
            ) * config.epoch_size
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=n_warmup_steps
            )
            logger.info(f"  LR warmup steps: {n_warmup_steps}")

        # --- AMP setup ---
        amp_enabled = (
            config.use_amp
            and device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        logger.info(f"  AMP enabled: {amp_enabled}")

        # --- Zero-shot evaluation ---
        if Y_val is not None:
            zs_metrics = self._evaluate_val(
                x_num_train, x_cat_train, Y_train,
                x_num_val, x_cat_val, Y_val,
                loss_fn, amp_enabled, device,
            )
            logger.info(f"  Zero-shot val R²: {zs_metrics.get('r2', 'N/A'):.4f}")

        # --- Training loop ---
        train_size = len(Y_train)
        best_val_score = -math.inf
        patience_counter = 0
        step = 0

        # Cap seq_len_pred so that n_candidates < train_size (CandidateQueue requirement)
        effective_seq_len_pred = min(config.seq_len_pred, train_size - 1)
        if effective_seq_len_pred < config.seq_len_pred:
            logger.info(f"  seq_len_pred capped: {config.seq_len_pred} -> {effective_seq_len_pred} (train_size={train_size})")

        logger.info(f"  Training: {config.max_epochs} max epochs, "
                     f"{config.epoch_size} steps/epoch, "
                     f"seq_len_pred={effective_seq_len_pred}, "
                     f"max_context_size={config.max_context_size or 'unlimited'}")

        for epoch in range(config.max_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_losses = []

            idx_queue = CandidateQueue(
                train_size,
                n_candidates=config.batch_size * effective_seq_len_pred,
                device=device,
            )

            for _ in range(config.epoch_size):
                # Sample query indices
                idx = next(idx_queue)
                idx = idx.view(config.batch_size, -1)

                # Compute context indices (all training samples except queries)
                mask = idx.new_ones(
                    (config.batch_size, train_size), dtype=torch.bool
                )
                mask[
                    torch.arange(config.batch_size, device=device).unsqueeze(-1),
                    idx,
                ] = False
                idx_train = (
                    torch.arange(train_size, device=device)
                    .expand(config.batch_size, train_size)[mask]
                    .view(config.batch_size, -1)
                )

                # Context size diversity: sample size log-uniformly each step
                if config.max_context_size is not None:
                    max_ctx = min(config.max_context_size, idx_train.shape[1])
                    min_ctx = min(config.min_context_size, max_ctx)
                    log_min = math.log(max(min_ctx, 1))
                    log_max = math.log(max(max_ctx, 1))
                    ctx_size = int(math.exp(random.uniform(log_min, log_max)))
                    ctx_size = max(min_ctx, min(ctx_size, idx_train.shape[1]))
                    if ctx_size < idx_train.shape[1]:
                        perm = torch.randperm(idx_train.shape[1], device=device)[:ctx_size]
                        perm, _ = perm.sort()
                        idx_train = idx_train[:, perm]

                # Build context + query features
                x_num_batch = None
                if x_num_train is not None:
                    x_num_batch = torch.cat([
                        x_num_train[idx_train],
                        x_num_train[idx],
                    ], dim=1)

                x_cat_batch = None
                if x_cat_train is not None:
                    x_cat_batch = torch.cat([
                        x_cat_train[idx_train],
                        x_cat_train[idx],
                    ], dim=1)

                # Forward pass
                optimizer.zero_grad()
                with torch.autocast(
                    device.type,
                    enabled=amp_enabled,
                    dtype=torch.bfloat16 if amp_enabled else None,
                ):
                    with torch.nn.attention.sdpa_kernel([
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.MATH,  # fallback for older GPUs (sm<80)
                    ]):
                        # Per-batch y normalization (matches per-county inference behavior)
                        Y_ctx_raw = Y_train[idx_train]  # (batch, ctx_size)
                        y_ctx_mean = Y_ctx_raw.mean(dim=1, keepdim=True)
                        y_ctx_std = Y_ctx_raw.std(dim=1, keepdim=True).clamp(min=1e-8)
                        Y_ctx_norm = (Y_ctx_raw - y_ctx_mean) / y_ctx_std
                        Y_tgt_norm = (Y_train[idx] - y_ctx_mean) / y_ctx_std

                        logits = self.model(
                            x_num=x_num_batch,
                            x_cat=x_cat_batch,
                            y_train=Y_ctx_norm,
                        ).float()

                    # Bar distribution loss on per-batch normalized targets
                    loss = loss_fn(
                        logits.permute(1, 0, 2),  # (seq_len_pred, batch, 5000)
                        Y_tgt_norm.transpose(0, 1),  # (seq_len_pred, batch)
                    ).mean()

                loss.backward()

                if config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.gradient_clip
                    )

                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                step += 1
                epoch_losses.append(loss.detach().item())

            # --- Epoch stats ---
            mean_loss = float(np.mean(epoch_losses))
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            self.history.train_losses.append(mean_loss)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)

            # --- Validation ---
            val_score = -math.inf
            if Y_val is not None:
                val_metrics = self._evaluate_val(
                    x_num_train, x_cat_train, Y_train,
                    x_num_val, x_cat_val, Y_val,
                    loss_fn, amp_enabled, device,
                )
                val_r2 = val_metrics.get("r2", -math.inf)
                val_score = val_r2
                self.history.val_metrics.append(val_metrics)

                logger.info(
                    f"  Epoch {epoch + 1}/{config.max_epochs}: "
                    f"train_loss={mean_loss:.4f}, val_r2={val_r2:.4f}, "
                    f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
                )
            else:
                logger.info(
                    f"  Epoch {epoch + 1}/{config.max_epochs}: "
                    f"train_loss={mean_loss:.4f}, "
                    f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
                )

            # --- Early stopping ---
            if val_score > best_val_score:
                best_val_score = val_score
                self.history.best_epoch = epoch + 1
                self.history.best_val_loss = mean_loss
                self._best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                logger.info(f"    New best epoch!")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

            # --- Check parameter updates ---
            if epoch == 0:
                n_changed = sum(
                    1 for n, p in self.model.named_parameters()
                    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
                )
                n_total = sum(
                    1 for _, p in self.model.named_parameters() if p.requires_grad
                )
                logger.info(f"  Params with non-zero grad: {n_changed}/{n_total}")

        # --- Restore best model ---
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            self.model.to(device)
            logger.info(f"  Restored best model from epoch {self.history.best_epoch}")

        # --- Rebuild inference context from FULL training set (train + val) ---
        # The val split was only needed for early stopping. At inference time we
        # pass all available training data as context, matching zero-shot TabPFN.
        x_num_full, x_cat_full, _ = self._prepare_features(X_full, self._continuous_cols)
        y_full_np = y_full.values.astype(np.float32)
        # Store raw y; predict() normalizes per-context using local y stats
        self._train_data = {
            "x_num": x_num_full.to(device) if x_num_full is not None else None,
            "x_cat": x_cat_full.to(device) if x_cat_full is not None else None,
            "y": torch.tensor(y_full_np, dtype=torch.float32, device=device),
        }
        logger.info(f"  Inference context: {len(y_full)} samples (full train+val set)")

        self.is_fitted = True
        logger.info("Finetuning complete!")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def _evaluate_val(
        self,
        x_num_train: Optional[Tensor],
        x_cat_train: Optional[Tensor],
        Y_train: Tensor,
        x_num_val: Optional[Tensor],
        x_cat_val: Optional[Tensor],
        Y_val: Tensor,
        loss_fn: FullSupportBarDistribution,
        amp_enabled: bool,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate on validation set using entire training set as context."""
        self.model.eval()
        eval_batch_size = self.config.eval_batch_size
        train_size = len(Y_train)
        val_size = len(Y_val)

        # Normalize Y_train (raw) using its own stats (matches per-county inference)
        y_train_mean = float(Y_train.mean().item())
        y_train_std = float(Y_train.std().item())
        if y_train_std < 1e-8:
            y_train_std = 1.0
        Y_ctx_norm = (Y_train - y_train_mean) / y_train_std

        # Build eval pred_transform for this context's y stats
        renorm_eval = FullSupportBarDistribution(
            self._criterion.borders * y_train_std + y_train_mean
        ).float()
        eval_pred_transform = regression_output_transform(
            self._target_transform, self._criterion, renorm_eval,
            softmax_temperature=self.config.softmax_temperature, device=device,
        )

        all_logits = []
        for start in range(0, val_size, eval_batch_size):
            end = min(start + eval_batch_size, val_size)

            x_num_batch = None
            if x_num_train is not None and x_num_val is not None:
                x_num_batch = torch.cat([
                    x_num_train.unsqueeze(0),
                    x_num_val[start:end].unsqueeze(0),
                ], dim=1)

            x_cat_batch = None
            if x_cat_train is not None and x_cat_val is not None:
                x_cat_batch = torch.cat([
                    x_cat_train.unsqueeze(0),
                    x_cat_val[start:end].unsqueeze(0),
                ], dim=1)

            with torch.autocast(
                device.type,
                enabled=amp_enabled,
                dtype=torch.bfloat16 if amp_enabled else None,
            ):
                with torch.nn.attention.sdpa_kernel([
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,  # fallback for older GPUs (sm<80)
                ]):
                    logits = self.model(
                        x_num=x_num_batch,
                        x_cat=x_cat_batch,
                        y_train=Y_ctx_norm.unsqueeze(0),
                    ).float().squeeze(0)

            all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits, dim=0)

        # Convert logits to predictions using per-context pred_transform
        predictions = eval_pred_transform(all_logits).numpy()

        # Val targets are raw (no inverse transform needed)
        y_val_original = Y_val.cpu().numpy()

        # Compute metrics
        from sklearn.metrics import r2_score, mean_absolute_error
        metrics = {
            "r2": float(r2_score(y_val_original, predictions)),
            "mae": float(mean_absolute_error(y_val_original, predictions)),
            "rmse": float(np.sqrt(np.mean((y_val_original - predictions) ** 2))),
        }

        return metrics

    @torch.inference_mode()
    def _evaluate_val_per_county(
        self,
        x_num_all: Optional[Tensor],
        x_cat_all: Optional[Tensor],
        Y_all: Tensor,
        county_idx_tensors: Dict[str, Tensor],
        val_fips_list: List[str],
        loss_fn: FullSupportBarDistribution,
        amp_enabled: bool,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Evaluate per-county: each val county gets a 50/50 context/test split.

        Returns dict with mean_r2, mean_mae, mean_loss, n_val_counties, and
        per_county detail {fips: {r2, mae, loss, size}}.
        """
        self.model.eval()
        from sklearn.metrics import r2_score, mean_absolute_error

        per_county = {}
        all_r2, all_mae, all_loss = [], [], []

        for fips in val_fips_list:
            county_idx = county_idx_tensors[fips]
            county_size = len(county_idx)

            # Deterministic 50/50 split seeded by FIPS
            fips_seed = hash(fips) % (2**31)
            gen = torch.Generator(device=device)
            gen.manual_seed(fips_seed)
            perm = torch.randperm(county_size, generator=gen, device=device)
            n_ctx = county_size // 2
            if n_ctx < 2:
                continue  # skip tiny counties where split is meaningless

            ctx_idx = county_idx[perm[:n_ctx]]
            test_idx = county_idx[perm[n_ctx:]]
            n_test = len(test_idx)

            # Per-county y normalization from context stats
            Y_ctx_raw = Y_all[ctx_idx]
            y_ctx_mean = Y_ctx_raw.mean()
            y_ctx_std = Y_ctx_raw.std().clamp(min=1e-8)
            Y_ctx_norm = (Y_ctx_raw - y_ctx_mean) / y_ctx_std
            Y_test_raw = Y_all[test_idx]
            Y_test_norm = (Y_test_raw - y_ctx_mean) / y_ctx_std

            # Build input tensors: (1, ctx+test, features)
            # Per-county x normalization: same as training — use context mean/std
            # to normalize numerical features, matching per-county Phase 2 scaling.
            # Clamp to [-10, 10] for consistency with training.
            x_num_batch = None
            if x_num_all is not None:
                x_num_ctx_raw = x_num_all[ctx_idx]
                x_num_test_raw = x_num_all[test_idx]
                x_ctx_mean = x_num_ctx_raw.mean(dim=0, keepdim=True)
                x_ctx_std = x_num_ctx_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
                x_num_ctx_norm = ((x_num_ctx_raw - x_ctx_mean) / x_ctx_std).clamp(-10, 10)
                x_num_test_norm = ((x_num_test_raw - x_ctx_mean) / x_ctx_std).clamp(-10, 10)
                x_num_batch = torch.cat([x_num_ctx_norm, x_num_test_norm], dim=0).unsqueeze(0)

            x_cat_batch = None
            if x_cat_all is not None:
                x_cat_batch = torch.cat([
                    x_cat_all[ctx_idx], x_cat_all[test_idx]
                ], dim=0).unsqueeze(0)

            with torch.autocast(
                device.type,
                enabled=amp_enabled,
                dtype=torch.bfloat16 if amp_enabled else None,
            ):
                with torch.nn.attention.sdpa_kernel([
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]):
                    logits = self.model(
                        x_num=x_num_batch,
                        x_cat=x_cat_batch,
                        y_train=Y_ctx_norm.unsqueeze(0),
                    ).float().squeeze(0)  # (n_test, 5000)

            # Bar distribution loss on normalized targets
            county_loss = loss_fn(
                logits.unsqueeze(1),  # (n_test, 1, 5000)
                Y_test_norm.unsqueeze(1),  # (n_test, 1)
            ).mean().item()

            # Predictions in original scale
            renorm_county = FullSupportBarDistribution(
                loss_fn.borders * y_ctx_std.item() + y_ctx_mean.item()
            ).float()
            county_pred_transform = regression_output_transform(
                self._target_transform, loss_fn, renorm_county,
                softmax_temperature=self.config.softmax_temperature,
                device=device,
            )
            preds = county_pred_transform(logits.cpu()).numpy()
            y_true = Y_test_raw.cpu().numpy()

            r2 = float(r2_score(y_true, preds)) if n_test >= 2 else 0.0
            mae = float(mean_absolute_error(y_true, preds))

            per_county[fips] = {'r2': r2, 'mae': mae, 'loss': county_loss, 'size': county_size}
            all_r2.append(r2)
            all_mae.append(mae)
            all_loss.append(county_loss)

        mean_r2 = float(np.mean(all_r2)) if all_r2 else 0.0
        mean_mae = float(np.mean(all_mae)) if all_mae else 0.0
        mean_loss = float(np.mean(all_loss)) if all_loss else 0.0

        return {
            'mean_r2': mean_r2,
            'mean_mae': mean_mae,
            'mean_loss': mean_loss,
            'n_val_counties': len(all_r2),
            'per_county': per_county,
        }

    # -------------------------------------------------------------------------
    # Per-County Fit
    # -------------------------------------------------------------------------

    def _fit_per_county(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        county_ids: pd.Series,
        continuous_cols: List[str],
    ) -> None:
        """Per-county task batching: each training step = one county.

        Context and query are both drawn from the same county, matching the
        actual inference scenario. Validation evaluates per-county with a
        deterministic 50/50 context/test split per county.
        """
        config = self.config
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # --- Store column info ---
        self._all_columns = list(X_train.columns)
        self._continuous_cols = continuous_cols

        # --- Group samples by county ---
        county_ids_arr = county_ids.values.astype(str)
        unique_fips = np.unique(county_ids_arr)

        # Build per-county row indices, filtering by min_county_size
        county_indices: Dict[str, np.ndarray] = {}
        for fips in unique_fips:
            rows = np.where(county_ids_arr == fips)[0]
            if len(rows) >= config.min_county_size:
                county_indices[fips] = rows

        n_counties = len(county_indices)
        n_samples = sum(len(v) for v in county_indices.values())
        logger.info(f"  Per-county task batching: {n_counties} counties "
                     f"({n_samples:,} samples, min_size={config.min_county_size})")

        # --- Split counties into train/val (80/20) ---
        all_fips = sorted(county_indices.keys())
        county_sizes = [len(county_indices[f]) for f in all_fips]

        # Stratify by size bucket (small/medium/large)
        size_arr = np.array(county_sizes)
        size_terciles = np.percentile(size_arr, [33, 66])
        strata = np.digitize(size_arr, size_terciles)

        from sklearn.model_selection import train_test_split as tts
        train_fips, val_fips = tts(
            all_fips, test_size=config.val_fraction,
            random_state=config.random_state, stratify=strata,
        )
        logger.info(f"  County split: {len(train_fips)} train, {len(val_fips)} val")

        # --- Prepare features ---
        x_num, x_cat, cat_cardinalities = self._prepare_features(X_train, continuous_cols)
        self._cat_cardinalities = cat_cardinalities
        n_num_features = x_num.shape[1] if x_num is not None else 0
        logger.info(f"  Numerical features: {n_num_features}")
        logger.info(f"  Categorical features: {len(cat_cardinalities)}")

        # --- Per-batch y normalization: store raw targets ---
        y_np = y_train.values.astype(np.float32)
        self._y_mean = float(y_np.mean())  # metadata only
        self._y_std = float(y_np.std())    # metadata only

        # --- Move data to device ---
        Y_all = torch.tensor(y_np, dtype=torch.float32, device=device)
        if x_num is not None:
            x_num = x_num.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)

        # Build per-county index tensors on device
        county_idx_tensors: Dict[str, Tensor] = {
            fips: torch.tensor(county_indices[fips], dtype=torch.long, device=device)
            for fips in all_fips
        }

        # Store full training data as inference context (same as global mode)
        self._train_data = {
            "x_num": x_num,
            "x_cat": x_cat,
            "y": Y_all,
        }

        # --- Find checkpoint and build model ---
        checkpoint_path = self._find_checkpoint(config.checkpoint_path)
        logger.info(f"  Checkpoint: {checkpoint_path}")

        self.model = TabPFN2(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=5000,
            is_regression=True,
            checkpoint_path=checkpoint_path,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Total parameters: {n_params:,}")

        # --- Set up loss ---
        borders = torch.load(checkpoint_path, weights_only=True)[
            "state_dict"
        ]["criterion.borders"].to(device)
        loss_fn = FullSupportBarDistribution(borders)
        self._criterion = loss_fn
        renormalized_criterion = FullSupportBarDistribution(
            loss_fn.borders * self._y_std + self._y_mean,
        ).float()
        # No global target transform in per_county mode — set before calling regression_output_transform
        self._target_transform = FunctionTransformer(func=None)
        self._pred_transform = regression_output_transform(
            self._target_transform,
            loss_fn,
            renormalized_criterion,
            softmax_temperature=config.softmax_temperature,
            device=device,
        )

        # --- Set up optimizer ---
        if config.finetune_mode == "full":
            params = list(self.model.parameters())
        elif config.finetune_mode == "lora":
            apply_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = list(self.model.parameters())

        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        logger.info(f"  Trainable parameters: {n_trainable:,}")

        optimizer = torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # --- AMP setup ---
        amp_enabled = (
            config.use_amp
            and device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        logger.info(f"  AMP enabled: {amp_enabled}")

        # --- Zero-shot per-county evaluation ---
        zs_metrics = self._evaluate_val_per_county(
            x_num, x_cat, Y_all, county_idx_tensors, val_fips,
            loss_fn, amp_enabled, device,
        )
        logger.info(f"  Zero-shot per-county val: R²={zs_metrics['mean_r2']:.4f}, "
                     f"MAE={zs_metrics['mean_mae']:.4f}, "
                     f"loss={zs_metrics['mean_loss']:.4f}, "
                     f"n_counties={zs_metrics['n_val_counties']}")
        self.history.zeroshot_val_loss = zs_metrics['mean_loss']

        # --- Zero-shot on ALL counties for spike diagnostics ---
        self._zs_all_county_losses = None
        if config.spike_diagnostics:
            logger.info("  [spike_diag] Running zero-shot eval on ALL training counties...")
            zs_train_metrics = self._evaluate_val_per_county(
                x_num, x_cat, Y_all, county_idx_tensors, train_fips,
                loss_fn, amp_enabled, device,
            )
            zs_all = {}
            for fp, data in zs_train_metrics['per_county'].items():
                zs_all[fp] = data['loss']
            for fp, data in zs_metrics['per_county'].items():
                zs_all[fp] = data['loss']
            self._zs_all_county_losses = zs_all
            logger.info(
                f"  [spike_diag] Zero-shot all counties: n={len(zs_all)}, "
                f"mean_loss={np.mean(list(zs_all.values())):.4f}, "
                f"max_loss={np.max(list(zs_all.values())):.4f}"
            )

        # --- Training loop ---
        best_val_r2 = -math.inf
        patience_counter = 0
        step = 0
        frac_lo, frac_hi = config.context_fraction_range
        do_spike_diag = config.spike_diagnostics
        spike_records: List[Dict[str, Any]] = []

        logger.info(f"  Training: {config.max_epochs} max epochs, "
                     f"{config.epoch_size} steps/epoch, "
                     f"context_fraction=[{frac_lo}, {frac_hi}]")
        if do_spike_diag:
            logger.info(f"  Spike diagnostics ENABLED (threshold={config.spike_threshold})")

        for epoch in range(config.max_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_losses = []

            for _ in range(config.epoch_size):
                # 1. Sample a random training county (uniform over counties)
                fips = random.choice(train_fips)
                county_idx = county_idx_tensors[fips]
                county_size = len(county_idx)

                # 2. Random context/query split
                frac = random.uniform(frac_lo, frac_hi)
                n_ctx = max(2, int(frac * county_size))
                n_ctx = min(n_ctx, county_size - 1)  # at least 1 query sample

                perm = torch.randperm(county_size, device=device)
                ctx_idx = county_idx[perm[:n_ctx]]
                qry_idx = county_idx[perm[n_ctx:]]

                # 3. Per-county y normalization
                Y_ctx_raw = Y_all[ctx_idx]
                y_ctx_mean = Y_ctx_raw.mean()
                y_ctx_std = Y_ctx_raw.std().clamp(min=1e-8)
                # Clamp to [-10, 10] (same range as features) to prevent gradient
                # explosion when a tiny context has near-zero std — e.g. n_ctx=2
                # with two nearly-identical log prices causes y_ctx_std ≈ 1e-8,
                # making Y_qry_norm blow up to ±millions and producing catastrophic
                # loss spikes (confirmed by spike diagnostic on fips 31007).
                Y_ctx_norm = ((Y_ctx_raw - y_ctx_mean) / y_ctx_std).clamp(-10, 10)
                Y_qry_raw = Y_all[qry_idx]
                Y_qry_norm = ((Y_qry_raw - y_ctx_mean) / y_ctx_std).clamp(-10, 10)

                # 4. Build input tensors: (1, ctx+qry, features)
                # Per-county x normalization: normalize numerical features by context
                # mean/std so training matches per-county Phase 2 scaling at inference.
                # Clamp to [-10, 10] to prevent gradient explosion when a feature has
                # near-zero variance in a small context (e.g., all samples same BEDS).
                x_num_batch = None
                if x_num is not None:
                    x_num_ctx_raw = x_num[ctx_idx]
                    x_num_qry_raw = x_num[qry_idx]
                    x_ctx_mean = x_num_ctx_raw.mean(dim=0, keepdim=True)
                    x_ctx_std = x_num_ctx_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
                    x_num_ctx_norm = ((x_num_ctx_raw - x_ctx_mean) / x_ctx_std).clamp(-10, 10)
                    x_num_qry_norm = ((x_num_qry_raw - x_ctx_mean) / x_ctx_std).clamp(-10, 10)
                    x_num_batch = torch.cat([x_num_ctx_norm, x_num_qry_norm], dim=0).unsqueeze(0)

                x_cat_batch = None
                if x_cat is not None:
                    x_cat_batch = torch.cat([
                        x_cat[ctx_idx], x_cat[qry_idx]
                    ], dim=0).unsqueeze(0)

                # 5. Forward pass
                optimizer.zero_grad()
                with torch.autocast(
                    device.type,
                    enabled=amp_enabled,
                    dtype=torch.bfloat16 if amp_enabled else None,
                ):
                    with torch.nn.attention.sdpa_kernel([
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.MATH,
                    ]):
                        logits = self.model(
                            x_num=x_num_batch,
                            x_cat=x_cat_batch,
                            y_train=Y_ctx_norm.unsqueeze(0),
                        ).float()

                    # 6. Loss on query samples
                    loss = loss_fn(
                        logits.permute(1, 0, 2),  # (n_qry, 1, 5000)
                        Y_qry_norm.unsqueeze(1),  # (n_qry, 1)
                    ).mean()

                step_loss_val = loss.detach().item()

                # Detect spike before backward (snapshot params if needed)
                is_spike = do_spike_diag and step_loss_val > config.spike_threshold
                param_snapshot = None
                if is_spike:
                    param_snapshot = {
                        name: p.detach().clone()
                        for name, p in self.model.named_parameters()
                        if p.requires_grad
                    }

                loss.backward()

                if config.gradient_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.gradient_clip
                    ).item()
                else:
                    grad_norm = 0.0

                optimizer.step()
                step += 1
                epoch_losses.append(step_loss_val)

                # Record spike diagnostic
                if is_spike:
                    # Compute optimizer step norm
                    opt_step_norm = sum(
                        (p.detach() - param_snapshot[name]).pow(2).sum().item()
                        for name, p in self.model.named_parameters()
                        if p.requires_grad and name in param_snapshot
                    ) ** 0.5
                    # Compute param norm
                    param_norm = sum(
                        p.detach().pow(2).sum().item()
                        for p in self.model.parameters() if p.requires_grad
                    ) ** 0.5

                    spike_record = {
                        'epoch': epoch,
                        'step': step,
                        'fips': fips,
                        'county_size': county_size,
                        'n_ctx': n_ctx,
                        'n_qry': len(qry_idx),
                        'y_ctx_mean': y_ctx_mean.item(),
                        'y_ctx_std': y_ctx_std.item(),
                        'y_qry_mean': Y_qry_raw.mean().item(),
                        'y_qry_std': Y_qry_raw.std().item() if len(Y_qry_raw) > 1 else 0.0,
                        'y_ctx_raw_min': Y_ctx_raw.min().item(),
                        'y_ctx_raw_max': Y_ctx_raw.max().item(),
                        'y_qry_raw_min': Y_qry_raw.min().item(),
                        'y_qry_raw_max': Y_qry_raw.max().item(),
                        'y_qry_norm_min': Y_qry_norm.min().item(),
                        'y_qry_norm_max': Y_qry_norm.max().item(),
                        'has_nan_features': bool(x_num_batch is not None and x_num_batch.isnan().any()),
                        'has_inf_features': bool(x_num_batch is not None and x_num_batch.isinf().any()),
                        'has_nan_targets': bool(Y_qry_norm.isnan().any()),
                        'has_inf_targets': bool(Y_qry_norm.isinf().any()),
                        'has_nan_logits': bool(logits.isnan().any()),
                        'has_inf_logits': bool(logits.isinf().any()),
                        'step_loss': step_loss_val,
                        'grad_norm_before_clip': grad_norm,
                        'param_norm': param_norm,
                        'optimizer_step_norm': opt_step_norm,
                    }
                    spike_records.append(spike_record)
                    self.history.spike_count += 1
                    if epoch not in self.history.spike_epochs:
                        self.history.spike_epochs.append(epoch)
                    logger.warning(
                        f"    SPIKE ep={epoch+1} step={step}: "
                        f"loss={step_loss_val:.2f}, fips={fips}, "
                        f"size={county_size}, n_ctx={n_ctx}, "
                        f"grad_norm={grad_norm:.4f}, "
                        f"nan_logits={spike_record['has_nan_logits']}, "
                        f"inf_logits={spike_record['has_inf_logits']}"
                    )

            # --- Epoch stats ---
            mean_loss = float(np.mean(epoch_losses))
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            self.history.train_losses.append(mean_loss)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)

            # --- Per-county validation ---
            val_result = self._evaluate_val_per_county(
                x_num, x_cat, Y_all, county_idx_tensors, val_fips,
                loss_fn, amp_enabled, device,
            )
            val_r2 = val_result['mean_r2']
            val_mae = val_result['mean_mae']
            val_loss = val_result['mean_loss']

            self.history.val_r2.append(val_r2)
            self.history.val_mae.append(val_mae)
            self.history.val_losses.append(val_loss)
            self.history.val_metrics.append({
                'r2': val_r2, 'mae': val_mae, 'loss': val_loss,
                'n_val_counties': val_result['n_val_counties'],
            })

            # Store per-county detail every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.history.val_county_metrics.append({
                    'epoch': epoch + 1,
                    'per_county': val_result['per_county'],
                })

            logger.info(
                f"  Epoch {epoch + 1}/{config.max_epochs}: "
                f"train_loss={mean_loss:.4f}, val_r2={val_r2:.4f}, "
                f"val_mae={val_mae:.4f}, val_loss={val_loss:.4f}, "
                f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
            )

            # --- Early stopping on val R² ---
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                self.history.best_epoch = epoch + 1
                self.history.best_val_loss = mean_loss
                self._best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                logger.info(f"    New best epoch! (val_r2={val_r2:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

            # --- Check parameter updates (first epoch only) ---
            if epoch == 0:
                n_changed = sum(
                    1 for n, p in self.model.named_parameters()
                    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
                )
                n_total = sum(
                    1 for _, p in self.model.named_parameters() if p.requires_grad
                )
                logger.info(f"  Params with non-zero grad: {n_changed}/{n_total}")

        # --- Store spike records ---
        self._spike_records = spike_records
        if spike_records:
            logger.info(f"  Spike diagnostics: {len(spike_records)} spikes recorded "
                         f"across {len(self.history.spike_epochs)} epochs")

        # --- Restore best model ---
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            self.model.to(device)
            logger.info(f"  Restored best model from epoch {self.history.best_epoch}")

        # --- Inference context: full training data (all counties) ---
        # predict() normalizes per-context using local y stats, so raw y is fine
        logger.info(f"  Inference context: {len(y_train)} samples (full training set)")

        self.is_fitted = True
        logger.info("Per-county finetuning complete!")

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    def predict(
        self,
        X_test: pd.DataFrame,
        X_context: Optional[pd.DataFrame] = None,
        y_context: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Make predictions using finetuned TabPFN v2.

        Args:
            X_test: Test features
            X_context: Context features (defaults to training data)
            y_context: Context targets (defaults to training targets)
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        device = next(self.model.parameters()).device

        # Prepare context data
        if X_context is not None and y_context is not None:
            # Per-county y normalization: use local stats from the provided
            # context instead of the global training stats.  This matches
            # what TabPFNRegressor does internally and avoids systematic bias
            # when the county's y distribution differs from the global mean.
            x_num_ctx, x_cat_ctx, _ = self._prepare_features(
                X_context, self._continuous_cols
            )
            y_ctx_np = y_context.values.astype(np.float32)
            y_mean_local = float(y_ctx_np.mean())
            y_std_local = float(y_ctx_np.std())
            if y_std_local < 1e-8:
                y_std_local = 1.0
            y_ctx_std = np.clip((y_ctx_np - y_mean_local) / y_std_local, -10, 10)
            if self.config.target_transform is not None:
                y_ctx_std = self._target_transform.transform(
                    y_ctx_std.reshape(-1, 1)
                ).astype(np.float32).squeeze()
            Y_ctx = torch.tensor(y_ctx_std, dtype=torch.float32, device=device)
            if x_num_ctx is not None:
                x_num_ctx = x_num_ctx.to(device)
            if x_cat_ctx is not None:
                x_cat_ctx = x_cat_ctx.to(device)

            # Build per-county pred_transform with local y stats
            renorm_local = FullSupportBarDistribution(
                self._criterion.borders * y_std_local + y_mean_local,
            ).float()
            pred_transform = regression_output_transform(
                self._target_transform,
                self._criterion,
                renorm_local,
                softmax_temperature=self.config.softmax_temperature,
                device=device,
            )
        else:
            # Stored training data — normalize per-context (same as explicit-context path)
            x_num_ctx = self._train_data["x_num"]
            x_cat_ctx = self._train_data["x_cat"]
            y_stored = self._train_data["y"]  # raw
            y_mean_local = float(y_stored.mean().item())
            y_std_local = float(y_stored.std().item())
            if y_std_local < 1e-8:
                y_std_local = 1.0
            Y_ctx = (y_stored - y_mean_local) / y_std_local
            renorm_local = FullSupportBarDistribution(
                self._criterion.borders * y_std_local + y_mean_local,
            ).float()
            pred_transform = regression_output_transform(
                self._target_transform,
                self._criterion,
                renorm_local,
                softmax_temperature=self.config.softmax_temperature,
                device=device,
            )

        # Prepare test features
        x_num_test, x_cat_test, _ = self._prepare_features(
            X_test, self._continuous_cols
        )
        if x_num_test is not None:
            x_num_test = x_num_test.to(device)
        if x_cat_test is not None:
            x_cat_test = x_cat_test.to(device)

        # Batch prediction
        self.model.eval()
        eval_batch_size = self.config.eval_batch_size
        all_predictions = []

        amp_enabled = (
            self.config.use_amp
            and device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        with torch.inference_mode():
            for start in range(0, len(X_test), eval_batch_size):
                end = min(start + eval_batch_size, len(X_test))

                x_num_batch = None
                if x_num_ctx is not None and x_num_test is not None:
                    x_num_batch = torch.cat([
                        x_num_ctx.unsqueeze(0),
                        x_num_test[start:end].unsqueeze(0),
                    ], dim=1)

                x_cat_batch = None
                if x_cat_ctx is not None and x_cat_test is not None:
                    x_cat_batch = torch.cat([
                        x_cat_ctx.unsqueeze(0),
                        x_cat_test[start:end].unsqueeze(0),
                    ], dim=1)

                with torch.autocast(
                    device.type,
                    enabled=amp_enabled,
                    dtype=torch.bfloat16 if amp_enabled else None,
                ):
                    with torch.nn.attention.sdpa_kernel([
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.MATH,  # fallback for older GPUs (sm<80)
                    ]):
                        logits = self.model(
                            x_num=x_num_batch,
                            x_cat=x_cat_batch,
                            y_train=Y_ctx.unsqueeze(0),
                        ).float().squeeze(0)

                predictions = pred_transform(logits.cpu()).numpy()
                all_predictions.append(predictions)

        return np.concatenate(all_predictions, axis=0)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _find_checkpoint(self, configured_path: Optional[str] = None) -> str:
        """Find or download the TabPFN v2 regressor checkpoint."""
        if configured_path and Path(configured_path).exists():
            return configured_path

        # Check common locations
        candidates = [
            Path("/tmp/tabpfn/tabpfn-v2-regressor.ckpt"),
            Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-regressor.ckpt",
        ]
        for path in candidates:
            if path.exists():
                return str(path)

        # Try downloading via tabpfn package
        try:
            from tabpfn.model_loading import download_model
            path = download_model("tabpfn-v2-regressor.ckpt")
            return str(path)
        except (ImportError, Exception):
            pass

        # Try huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="Prior-Labs/TabPFN-v2-reg",
                filename="tabpfn-v2-regressor.ckpt",
                cache_dir="/tmp/tabpfn",
            )
            return path
        except (ImportError, Exception):
            pass

        raise FileNotFoundError(
            "Could not find TabPFN v2 regressor checkpoint. "
            "Please provide checkpoint_path in config, or install tabpfn package."
        )

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._train_data = None
        self._best_model_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # -------------------------------------------------------------------------
    # Checkpoint Save / Load (for global finetuning)
    # -------------------------------------------------------------------------

    def save_to_disk(self, save_dir: str) -> None:
        """Save finetuned model checkpoint and metadata to disk.

        Saves:
        - model.pt: model state dict
        - metadata.json: config, column info, target stats
        - transforms.pkl: target_transform and pred_transform (pickled)
        - history.json: full training history for loss curve plotting
        """
        import json
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        assert self.is_fitted, "Model must be fitted before saving"

        # Save model state dict
        state_dict = self._best_model_state or {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
        torch.save(state_dict, save_path / "model.pt")
        logger.info(f"  Saved model state dict to {save_path / 'model.pt'}")

        # Save metadata
        metadata = {
            'config': {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'max_epochs': self.config.max_epochs,
                'patience': self.config.patience,
                'epoch_size': self.config.epoch_size,
                'seq_len_pred': self.config.seq_len_pred,
                'max_context_size': self.config.max_context_size,
                'batch_size': self.config.batch_size,
                'gradient_clip': self.config.gradient_clip,
                'use_amp': self.config.use_amp,
                'finetune_mode': self.config.finetune_mode,
                'lora_rank': self.config.lora_rank,
                'lora_alpha': self.config.lora_alpha,
                'target_transform': self.config.target_transform,
                'softmax_temperature': self.config.softmax_temperature,
                'val_fraction': self.config.val_fraction,
                'eval_batch_size': self.config.eval_batch_size,
                'random_state': self.config.random_state,
                'training_mode': self.config.training_mode,
                'min_county_size': self.config.min_county_size,
                'context_fraction_range': list(self.config.context_fraction_range),
                'spike_diagnostics': self.config.spike_diagnostics,
                'spike_threshold': self.config.spike_threshold,
            },
            'continuous_cols': self._continuous_cols,
            'all_columns': self._all_columns,
            'cat_cardinalities': self._cat_cardinalities,
            'y_mean': self._y_mean,
            'y_std': self._y_std,
        }
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save transforms (these aren't easily JSON-serializable)
        import pickle
        transforms = {
            'target_transform': self._target_transform,
        }
        with open(save_path / "transforms.pkl", 'wb') as f:
            pickle.dump(transforms, f)

        # Save training history
        with open(save_path / "history.json", 'w') as f:
            json.dump(self.history.to_dict(), f, indent=2)

        # Save spike diagnostics if any
        spike_records = getattr(self, '_spike_records', [])
        if spike_records:
            with open(save_path / "spike_diagnostics.jsonl", 'w') as f:
                for record in spike_records:
                    f.write(json.dumps(record) + '\n')
            logger.info(f"  Saved {len(spike_records)} spike records to spike_diagnostics.jsonl")

        # Save zero-shot per-county losses
        zs_losses = getattr(self, '_zs_all_county_losses', None)
        if zs_losses is not None:
            with open(save_path / "zeroshot_per_county.json", 'w') as f:
                json.dump(zs_losses, f, indent=2)
            logger.info(f"  Saved zero-shot per-county losses ({len(zs_losses)} counties)")

        logger.info(f"  Saved checkpoint to {save_path}")

    @classmethod
    def load_from_disk(cls, save_dir: str, device: str = "cuda") -> "DirectFineTunedTabPFNModel":
        """Load a finetuned model from a saved checkpoint.

        Args:
            save_dir: Directory containing model.pt, metadata.json, etc.
            device: Device to load model to

        Returns:
            Ready-to-predict DirectFineTunedTabPFNModel instance
        """
        import json
        import pickle
        save_path = Path(save_dir)

        # Load metadata
        with open(save_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Reconstruct config
        cfg_dict = metadata['config']
        config = FinetuningConfigV2(
            learning_rate=cfg_dict['learning_rate'],
            weight_decay=cfg_dict['weight_decay'],
            max_epochs=cfg_dict['max_epochs'],
            patience=cfg_dict['patience'],
            epoch_size=cfg_dict['epoch_size'],
            seq_len_pred=cfg_dict['seq_len_pred'],
            max_context_size=cfg_dict.get('max_context_size'),
            batch_size=cfg_dict['batch_size'],
            gradient_clip=cfg_dict['gradient_clip'],
            use_amp=cfg_dict['use_amp'],
            finetune_mode=cfg_dict['finetune_mode'],
            lora_rank=int(cfg_dict.get('lora_rank', 0)),
            lora_alpha=float(cfg_dict.get('lora_alpha', 16.0)),
            target_transform=cfg_dict.get('target_transform'),
            softmax_temperature=cfg_dict['softmax_temperature'],
            val_fraction=cfg_dict['val_fraction'],
            eval_batch_size=cfg_dict['eval_batch_size'],
            device=device,
            random_state=cfg_dict['random_state'],
            training_mode=cfg_dict.get('training_mode', 'global'),
            min_county_size=int(cfg_dict.get('min_county_size', 5)),
            context_fraction_range=tuple(cfg_dict.get('context_fraction_range', [0.3, 0.7])),
            spike_diagnostics=bool(cfg_dict.get('spike_diagnostics', False)),
            spike_threshold=float(cfg_dict.get('spike_threshold', 100.0)),
        )

        # Create instance
        instance = cls(config)
        instance._continuous_cols = metadata['continuous_cols']
        instance._all_columns = metadata['all_columns']
        instance._cat_cardinalities = metadata['cat_cardinalities']
        instance._y_mean = metadata['y_mean']
        instance._y_std = metadata['y_std']

        # Load transforms
        with open(save_path / "transforms.pkl", 'rb') as f:
            transforms = pickle.load(f)
        instance._target_transform = transforms['target_transform']

        # Find the original TabPFN checkpoint for model architecture
        checkpoint_path = instance._find_checkpoint(config.checkpoint_path)

        # Build model architecture
        n_num_features = len([c for c in instance._continuous_cols if c in instance._all_columns])
        cat_cardinalities = instance._cat_cardinalities or []

        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        instance.model = TabPFN2(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=5000,
            is_regression=True,
            checkpoint_path=checkpoint_path,
        ).to(device_obj)

        # Apply LoRA wrappers if needed (must match architecture before loading state dict)
        if config.lora_rank > 0:
            apply_lora(instance.model, rank=config.lora_rank, alpha=config.lora_alpha)

        # Load finetuned weights
        state_dict = torch.load(save_path / "model.pt", map_location=device_obj, weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        # Set up loss/prediction transform
        borders = torch.load(checkpoint_path, weights_only=True)[
            "state_dict"
        ]["criterion.borders"].to(device_obj)
        loss_fn = FullSupportBarDistribution(borders)
        instance._criterion = loss_fn
        renormalized_criterion = FullSupportBarDistribution(
            loss_fn.borders * instance._y_std + instance._y_mean,
        ).float()
        instance._pred_transform = regression_output_transform(
            instance._target_transform,
            loss_fn,
            renormalized_criterion,
            softmax_temperature=config.softmax_temperature,
            device=device_obj,
        )

        # No training data stored — caller must provide context via predict(X_test, X_context, y_context)
        instance._train_data = None
        instance.is_fitted = True

        # Load history if available
        history_path = save_path / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                hist_dict = json.load(f)
            instance.history = TrainingHistory(
                train_losses=hist_dict.get('train_losses', []),
                val_losses=hist_dict.get('val_losses', []),
                val_metrics=hist_dict.get('val_metrics', []),
                learning_rates=hist_dict.get('learning_rates', []),
                epoch_times=hist_dict.get('epoch_times', []),
                best_epoch=hist_dict.get('best_epoch', 0),
                best_val_loss=hist_dict.get('best_val_loss', float('inf')),
                train_r2=hist_dict.get('train_r2', []),
                val_r2=hist_dict.get('val_r2', []),
                val_mae=hist_dict.get('val_mae', []),
                val_county_metrics=hist_dict.get('val_county_metrics', []),
            )

        logger.info(f"  Loaded finetuned model from {save_path}")
        logger.info(f"  n_num_features={n_num_features}, cat_cardinalities={len(cat_cardinalities)}")
        logger.info(f"  y_mean={instance._y_mean:.4f}, y_std={instance._y_std:.4f}")

        return instance
