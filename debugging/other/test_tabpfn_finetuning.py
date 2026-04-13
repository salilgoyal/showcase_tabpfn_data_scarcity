"""
Test TabPFN fine-tuning with validation to see if val loss changes across epochs.
"""
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# change to cuda if you have one
device = "cpu"

# for illustration, we want it to be fast
n_samples = 500

X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X[:n_samples], y[:n_samples], test_size=0.2, random_state=42
)

print(f"Train samples: {len(X_train_raw)}, Val samples: {len(X_val_raw)}")

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

regressor_args = {
    "device": device,
    "n_estimators": 1
}

reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, **regressor_args, fit_mode="batched", ignore_pretraining_limits=True)

from tabpfn.preprocessing import DatasetCollectionWithPreprocessing

datasets_collection = reg.get_preprocessed_datasets(
    X_train_raw, y_train_raw, train_test_split, max_data_size=150
)

from torch.utils.data import DataLoader
from tabpfn.utils import meta_dataset_collator

data_loader = DataLoader(
    datasets_collection, batch_size=1, collate_fn=meta_dataset_collator, shuffle=True
)

from torch.optim import AdamW
import torch

# CRITICAL BUG FOUND: We need to create optimizer AFTER fit_from_preprocessed initializes model_
# The optimizer was being created BEFORE the model exists!
print("\n" + "="*60)
print("DEBUGGING: When is model_ initialized?")
print("="*60)
print(f"Before get_preprocessed_datasets: hasattr(reg, 'model_') = {hasattr(reg, 'model_')}")
if hasattr(reg, 'model_'):
    print(f"  reg.model_ is None: {reg.model_ is None}")

optimizer = None  # Will be created after first fit_from_preprocessed

print("\n" + "="*60)
print("TESTING TABPFN FINE-TUNING WITH VALIDATION")
print("="*60)

def validate(reg, X_val, y_val):
    """Validate the model - testing different approaches."""

    # Approach 1: Save weights, call fit, restore weights, then predict
    print("\n  Approach 1: Save/restore weights around fit()")

    # Save fine-tuned weights
    saved_state = {k: v.cpu().clone() for k, v in reg.model_.state_dict().items()}
    weight_sum_before = sum(p.sum().item() for p in reg.model_.parameters())

    # Switch to fit_preprocessors mode
    original_mode = reg.fit_mode
    reg.fit_mode = "fit_preprocessors"

    # Call fit (will overwrite weights)
    reg.fit(X_val[:50], y_val[:50])

    weight_sum_after_fit = sum(p.sum().item() for p in reg.model_.parameters())

    # Restore fine-tuned weights
    reg.model_.load_state_dict(saved_state)
    reg.model_.to(device)

    weight_sum_after_restore = sum(p.sum().item() for p in reg.model_.parameters())

    print(f"    Weight sum before: {weight_sum_before:.6f}")
    print(f"    Weight sum after fit: {weight_sum_after_fit:.6f}")
    print(f"    Weight sum after restore: {weight_sum_after_restore:.6f}")
    print(f"    Weights changed during fit: {abs(weight_sum_after_fit - weight_sum_before) > 1e-3}")
    print(f"    Weights correctly restored: {abs(weight_sum_after_restore - weight_sum_before) < 1e-6}")

    # Now predict
    y_pred = reg.predict(X_val)

    # Switch back
    reg.fit_mode = original_mode

    mse = np.mean((y_val - y_pred) ** 2)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    return mse, r2, mae


# Before training - baseline
print("\n--- BEFORE TRAINING (Pretrained model) ---")
baseline_mse, baseline_r2, baseline_mae = validate(reg, X_val_raw, y_val_raw)
print(f"Baseline - MSE: {baseline_mse:.6f}, R2: {baseline_r2:.4f}, MAE: {baseline_mae:.4f}")

# Training loop
do_epochs = 5
for epoch in range(do_epochs):
    print(f"\n--- EPOCH {epoch + 1}/{do_epochs} ---")

    epoch_losses = []
    reg.model_.train()

    for batch_idx, data_batch in enumerate(data_loader):
        # Check weights BEFORE any operation
        weight_sum_start = sum(p.sum().item() for p in reg.model_.parameters())

        if optimizer is not None:
            optimizer.zero_grad()

        # extract data and config from the batch
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            normalized_bardist_,
            bardist_,
            batch_x_test_raw,
            batch_y_test_raw,
        ) = data_batch

        # set the criterion (bar dist.) on the regressor
        reg.normalized_bardist_ = normalized_bardist_[0]

        # Check weights BEFORE fit_from_preprocessed
        weight_sum_before_fit = sum(p.sum().item() for p in reg.model_.parameters())

        # fit the regressor from the preprocessed tensors
        is_first_batch = (batch_idx == 0 and epoch == 0)

        # Save current weights BEFORE fit_from_preprocessed (which may recreate model)
        if not is_first_batch:
            saved_weights = {k: v.clone() for k, v in reg.model_.state_dict().items()}

        # ALWAYS call fit_from_preprocessed with no_refit=True after first batch
        reg.fit_from_preprocessed(
            X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs,
            no_refit=(not is_first_batch)
        )

        # Restore weights after fit_from_preprocessed (if not first batch)
        if not is_first_batch:
            reg.model_.load_state_dict(saved_weights)

        # Check weights AFTER fit_from_preprocessed
        weight_sum_after_fit = sum(p.sum().item() for p in reg.model_.parameters())

        if is_first_batch:
            print("  Initialized model with fit_from_preprocessed()")
            print(f"  Weight sum after initialization: {weight_sum_after_fit:.6f}")

            # NOW create the optimizer - after model_ is initialized!
            print("\n  Creating optimizer AFTER model initialization...")
            optimizer = AdamW(reg.model_.parameters(), lr=1e-4)
            print(f"  Optimizer created with {len(list(reg.model_.parameters()))} parameter groups")

            # Check if model_ and models_[0] are the same object
            print(f"  reg.model_ is reg.models_[0]: {reg.model_ is reg.models_[0]}")

            # Verify optimizer has correct params
            optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
            model_param_ids = {id(p) for p in reg.model_.parameters()}
            print(f"  Optimizer tracking same params as model: {optimizer_param_ids == model_param_ids}")

            # Check param ids
            first_param = list(reg.model_.parameters())[0]
            print(f"  First param id in model: {id(first_param)}")
            print(f"  First param id in optimizer: {id(list(optimizer.param_groups[0]['params'])[0])}")
        else:
            # CRITICAL: fit_from_preprocessed creates new parameter tensors!
            # We need to update the optimizer to track the new parameters
            optimizer = AdamW(reg.model_.parameters(), lr=1e-4)

        # Track if optimizer params match after fit_from_preprocessed
        if batch_idx == 0 and epoch == 1:
            optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
            model_param_ids = {id(p) for p in reg.model_.parameters()}
            print(f"  After optimizer refresh: optimizer tracking model params: {optimizer_param_ids == model_param_ids}")

        # forward pass through the regressor
        averaged_pred_logits, _, _ = reg.forward(
            X_tests_preprocessed, use_inference_mode=False
        )

        # despite naming bardist is the normalized bar distribution
        lossfn = bardist_[0]

        # compute the loss
        nll_loss_per_sample = lossfn(averaged_pred_logits, y_test_standardized.to(device))

        # compute mean loss across all test samples in single forward pass
        loss = nll_loss_per_sample.mean()

        epoch_losses.append(loss.item())

        # Check weights BEFORE backward
        weight_sum_before_backward = sum(p.sum().item() for p in reg.model_.parameters())

        loss.backward()

        # Check weights AFTER backward (should be same)
        weight_sum_after_backward = sum(p.sum().item() for p in reg.model_.parameters())

        # Check gradient
        total_grad_norm = sum(p.grad.norm().item() for p in reg.model_.parameters() if p.grad is not None)

        optimizer.step()

        # Check weights AFTER optimizer step
        weight_sum_after_step = sum(p.sum().item() for p in reg.model_.parameters())

        if batch_idx == 0:
            print(f"  Batch {batch_idx}: grad_norm={total_grad_norm:.6f}")
            print(f"    Before backward: {weight_sum_before_backward:.6f}")
            print(f"    After backward: {weight_sum_after_backward:.6f} (should be same)")
            print(f"    After optimizer.step(): {weight_sum_after_step:.6f}")
            print(f"    Weight changed by optimizer.step(): {abs(weight_sum_after_step - weight_sum_before_backward) > 1e-6}")

    train_loss = np.mean(epoch_losses)
    print(f"  Train loss: {train_loss:.6f}")

    # Validation
    reg.model_.eval()
    with torch.no_grad():
        val_mse, val_r2, val_mae = validate(reg, X_val_raw, y_val_raw)

    print(f"  Val - MSE: {val_mse:.6f}, R2: {val_r2:.4f}, MAE: {val_mae:.4f}")
    print(f"  Val MSE change from baseline: {val_mse - baseline_mse:.6f}")

print("\n" + "="*60)
print("EXPECTED: Val MSE should DECREASE and R2 should INCREASE across epochs")
print("If val metrics are stuck, then the issue is with the validation approach")
print("="*60)
