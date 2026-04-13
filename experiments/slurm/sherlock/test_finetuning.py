# test_finetuning.py
import pandas as pd
import numpy as np
from src.models.tabpfn_finetuning import FineTunedTabPFNModel, FinetuningConfig

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
y_train = pd.Series(np.random.randn(n_samples))
X_val = pd.DataFrame(np.random.randn(200, n_features))
y_val = pd.Series(np.random.randn(200))
X_test = pd.DataFrame(np.random.randn(100, n_features))

# Create minimal config
config = FinetuningConfig(
    learning_rate=1e-5,
    max_epochs=3,  # Just 3 epochs
    batch_size=32,
    patience=2,
    device='cuda',  # or 'cpu' if no GPU
)

# Test initialization
print("Testing model initialization...")
model = FineTunedTabPFNModel(config=config)

# Test fitting
print("Testing fit...")
model.fit(X_train, y_train, X_val, y_val)

# Test prediction
print("Testing predict...")
y_pred = model.predict(X_test)
print(f"Predictions shape: {y_pred.shape}")

# Test context prediction
print("Testing predict_with_context...")
y_pred_context = model.predict_with_context(
    X_test, X_train[:50], y_train[:50]
)
print(f"Context predictions shape: {y_pred_context.shape}")

print("✓ All basic tests passed!")
