"""
Test to verify step_best_lag works with per_group_prep=True
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Create test time series data with groups
np.random.seed(42)
n_groups = 2
n_per_group = 100

data_list = []
for group_name in ['A', 'B']:
    # Create time series with lagged relationships
    t = np.arange(n_per_group)

    # x1 has different optimal lags for different groups
    if group_name == 'A':
        # Group A: y depends on x1_lag3
        x1 = np.sin(t / 10) + np.random.randn(n_per_group) * 0.1
        x2 = np.cos(t / 15) + np.random.randn(n_per_group) * 0.1
        y = np.zeros(n_per_group)
        y[3:] = 2 * x1[:-3] + 1.5 * x2[3:] + np.random.randn(n_per_group-3) * 0.2
    else:  # B
        # Group B: y depends on x1_lag1
        x1 = np.sin(t / 8) + np.random.randn(n_per_group) * 0.1
        x2 = np.cos(t / 12) + np.random.randn(n_per_group) * 0.1
        y = np.zeros(n_per_group)
        y[1:] = 2 * x1[:-1] + 1.5 * x2[1:] + np.random.randn(n_per_group-1) * 0.2

    group_data = pd.DataFrame({
        'group': group_name,
        'x1': x1,
        'x2': x2,
        'y': y
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)

# Split train/test
train_idx = int(len(data) * 0.8)
train_data = data.iloc[:train_idx].copy()
test_data = data.iloc[train_idx:].copy()

print("Testing step_best_lag with per_group_prep=True...")
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Groups: {train_data['group'].unique()}")
print()

# Test step_best_lag with per_group_prep=True
try:
    rec_best_lag = (
        recipe()
        .step_best_lag(
            outcome='y',
            max_lag=5,
            alpha=0.1  # More lenient for small test data
        )
    )

    wf_best_lag = (
        workflow()
        .add_recipe(rec_best_lag)
        .add_model(linear_reg().set_engine("sklearn"))
        .add_model_name("best_lag_test")
    )

    print("Fitting with per_group_prep=True...")
    fit_best_lag = wf_best_lag.fit_nested(train_data, group_col='group', per_group_prep=True)

    print("✓ Fit successful!")

    # Get predictions
    print("\nMaking predictions on test data...")
    predictions = fit_best_lag.predict(test_data)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Get feature comparison
    print("\nGetting feature comparison across groups...")
    feature_comparison = fit_best_lag.get_feature_comparison()
    print("✓ Feature comparison retrieved!")
    print(feature_comparison)

    # Extract outputs
    print("\nExtracting outputs...")
    outputs, coefs, stats = fit_best_lag.extract_outputs()
    print(f"✓ Outputs shape: {outputs.shape}")
    print(f"✓ Coefficients shape: {coefs.shape}")
    print(f"✓ Stats shape: {stats.shape}")

    print("\n" + "="*60)
    print("✅ SUCCESS! step_best_lag works with per_group_prep=True")
    print("="*60)
    print("\nLag features created per group:")
    print(feature_comparison)
    print("\nNote: Different groups may have different optimal lags")
    print("based on their group-specific Granger causality tests.")

except Exception as e:
    print("\n" + "="*60)
    print("❌ FAILED!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
