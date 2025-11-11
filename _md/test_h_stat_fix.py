"""
Test to verify step_h_stat works with per_group_prep=True
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Create test data with groups and interactions
np.random.seed(42)
n_groups = 3
n_per_group = 100

data_list = []
for group_name in ['A', 'B', 'C']:
    # Create features with interactions
    x1 = np.random.randn(n_per_group)
    x2 = np.random.randn(n_per_group)
    x3 = np.random.randn(n_per_group)

    # Outcome depends on features AND their interaction
    # Different groups have different interaction strengths
    if group_name == 'A':
        y = 2*x1 + 1.5*x2 + 3*x1*x2 + 0.5*x3 + np.random.randn(n_per_group)*0.5
    elif group_name == 'B':
        y = 2*x1 + 1.5*x2 + 5*x1*x3 + np.random.randn(n_per_group)*0.5
    else:  # C
        y = 2*x1 + 1.5*x2 + 4*x2*x3 + np.random.randn(n_per_group)*0.5

    group_data = pd.DataFrame({
        'group': group_name,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)

# Split train/test
train_idx = int(len(data) * 0.8)
train_data = data.iloc[:train_idx].copy()
test_data = data.iloc[train_idx:].copy()

print("Testing step_h_stat with per_group_prep=True...")
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Groups: {train_data['group'].unique()}")
print()

# Test step_h_stat with per_group_prep=True
try:
    rec_h_stat = (
        recipe()
        .step_h_stat(
            outcome='y',
            top_n=2,  # Keep top 2 interactions
            n_estimators=50,  # Fewer trees for speed
            random_state=42
        )
    )

    wf_h_stat = (
        workflow()
        .add_recipe(rec_h_stat)
        .add_model(linear_reg().set_engine("sklearn"))
        .add_model_name("h_stat_test")
    )

    print("Fitting with per_group_prep=True...")
    fit_h_stat = wf_h_stat.fit_nested(train_data, group_col='group', per_group_prep=True)

    print("✓ Fit successful!")

    # Get predictions
    print("\nMaking predictions on test data...")
    predictions = fit_h_stat.predict(test_data)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Get feature comparison
    print("\nGetting feature comparison across groups...")
    feature_comparison = fit_h_stat.get_feature_comparison()
    print("✓ Feature comparison retrieved!")
    print(feature_comparison)

    # Extract outputs
    print("\nExtracting outputs...")
    outputs, coefs, stats = fit_h_stat.extract_outputs()
    print(f"✓ Outputs shape: {outputs.shape}")
    print(f"✓ Coefficients shape: {coefs.shape}")
    print(f"✓ Stats shape: {stats.shape}")

    print("\n" + "="*60)
    print("✅ SUCCESS! step_h_stat works with per_group_prep=True")
    print("="*60)
    print("\nInteraction features created per group:")
    print(feature_comparison)
    print("\nNote: Different groups may have different interaction features")
    print("based on their group-specific H-statistics.")

except Exception as e:
    print("\n" + "="*60)
    print("❌ FAILED!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
