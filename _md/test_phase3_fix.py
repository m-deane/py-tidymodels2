"""
Quick test to verify Phase 3 supervised selection steps work with per_group_prep=True
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Create test data with groups
np.random.seed(42)
n_groups = 3
n_per_group = 50

data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], n_per_group),
    'x1': np.random.randn(n_groups * n_per_group),
    'x2': np.random.randn(n_groups * n_per_group),
    'x3': np.random.randn(n_groups * n_per_group),
    'x4': np.random.randn(n_groups * n_per_group),
    'x5': np.random.randn(n_groups * n_per_group),
})

# Create target with some correlation to features
data['y'] = (
    2.0 * data['x1'] +
    1.5 * data['x2'] -
    0.8 * data['x3'] +
    np.random.randn(len(data)) * 0.5
)

# Split train/test
train_idx = int(len(data) * 0.8)
train_data = data.iloc[:train_idx].copy()
test_data = data.iloc[train_idx:].copy()

print("Testing Phase 3 step with per_group_prep=True...")
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Groups: {train_data['group'].unique()}")

# Test step_pvalue with per_group_prep=True
try:
    rec_pvalue = (
        recipe()
        .step_normalize()
        .step_pvalue(
            outcome='y',
            threshold=0.1  # More lenient for small test data
        )
    )

    wf_pvalue = (
        workflow()
        .add_recipe(rec_pvalue)
        .add_model(linear_reg().set_engine("sklearn"))
        .add_model_name("pvalue_test")
    )

    print("\nFitting with per_group_prep=True...")
    fit_pvalue = wf_pvalue.fit_nested(train_data, group_col='group', per_group_prep=True)

    print("✓ Fit successful!")

    # Get predictions
    print("\nMaking predictions on test data...")
    predictions = fit_pvalue.predict(test_data)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Get feature comparison
    print("\nGetting feature comparison across groups...")
    feature_comparison = fit_pvalue.get_feature_comparison()
    print("✓ Feature comparison retrieved!")
    print(feature_comparison)

    # Extract outputs
    print("\nExtracting outputs...")
    outputs, coefs, stats = fit_pvalue.extract_outputs()
    print(f"✓ Outputs shape: {outputs.shape}")
    print(f"✓ Coefficients shape: {coefs.shape}")
    print(f"✓ Stats shape: {stats.shape}")

    print("\n" + "="*60)
    print("✅ SUCCESS! Phase 3 step works with per_group_prep=True")
    print("="*60)
    print("\nFeatures selected per group may differ (as expected):")
    print(feature_comparison)

except Exception as e:
    print("\n" + "="*60)
    print("❌ FAILED!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
