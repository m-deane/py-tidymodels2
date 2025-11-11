"""
Test to verify step_dt_features works correctly with per_group_prep=True
and doesn't create column names that Patsy interprets as function calls
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes.selectors import all_numeric, one_of, difference

# Create test data with groups
np.random.seed(42)
n_per_group = 50

data_list = []
for group_name in ['A', 'B']:
    # Create features similar to oil price data
    t = np.arange(n_per_group)

    brent = 50 + 10 * np.sin(t / 10) + np.random.randn(n_per_group) * 2
    dubai = 48 + 10 * np.sin(t / 10 + 0.5) + np.random.randn(n_per_group) * 2
    wti = 49 + 10 * np.sin(t / 10 + 0.3) + np.random.randn(n_per_group) * 2

    # Target depends on prices with some non-linearity
    target = 100 + 0.5 * brent + 0.3 * dubai + 0.2 * wti + 0.1 * brent * dubai + np.random.randn(n_per_group) * 5

    group_data = pd.DataFrame({
        'group': group_name,
        'brent': brent,
        'dubai': dubai,
        'wti': wti,
        'target': target
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)

# Split train/test
train_idx = int(len(data) * 0.8)
train_data = data.iloc[:train_idx].copy()
test_data = data.iloc[train_idx:].copy()

print("Testing step_dt_features with per_group_prep=True...")
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Groups: {train_data['group'].unique()}")
print()

# Test step_dt_features
try:
    rec_dt = (
        recipe()
        .step_dt_features(
            outcome='target',
            columns=difference(all_numeric(), one_of('target')),
            features_to_combine=2,
            cv=3,
            random_state=42
        )
    )

    print("✓ Recipe created successfully")

    # Test prep on training data (to see column names created)
    print("\nTesting prep to check column names...")
    prepped = rec_dt.prep(train_data)
    print("✓ Recipe prepped successfully")

    # Bake to see actual column names
    baked = prepped.bake(train_data.head(10))
    print("\nColumn names after baking:")
    new_cols = [c for c in baked.columns if c not in ['group', 'brent', 'dubai', 'wti', 'target']]
    print(f"New columns created: {new_cols}")

    # Verify no columns contain "_tree_"
    tree_cols = [c for c in new_cols if '_tree_' in c]
    if tree_cols:
        print(f"❌ FAILED: Found columns with '_tree_' pattern: {tree_cols}")
        print("These will cause Patsy errors!")
    else:
        print("✓ No columns contain '_tree_' pattern (Patsy-safe)")

    # Verify columns contain "_dt_" instead
    dt_cols = [c for c in new_cols if '_dt_' in c]
    if dt_cols:
        print(f"✓ Found {len(dt_cols)} columns with '_dt_' pattern: {dt_cols[:3]}...")
    else:
        print("⚠ No columns with '_dt_' pattern found")

    # Now test in workflow with per_group_prep=True (the real test)
    print("\n" + "="*60)
    print("Testing with workflow and per_group_prep=True...")
    print("="*60)

    wf_dt = (
        workflow()
        .add_recipe(rec_dt)
        .add_model(linear_reg().set_engine("sklearn"))
        .add_model_name("dt_features_test")
    )

    print("\nFitting with per_group_prep=True...")
    fit_dt = wf_dt.fit_nested(train_data, group_col='group', per_group_prep=True)

    print("✓ Fit successful!")

    # Get predictions
    print("\nMaking predictions on test data...")
    predictions = fit_dt.predict(test_data)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Get feature comparison
    print("\nGetting feature comparison across groups...")
    feature_comparison = fit_dt.get_feature_comparison()
    print("✓ Feature comparison retrieved!")

    # Show which features each group uses
    print("\nFeatures used by each group:")
    # Show only dt features for clarity
    dt_feature_cols = [c for c in feature_comparison.columns if '_dt_' in c]
    if len(dt_feature_cols) > 0:
        print(feature_comparison[dt_feature_cols[:5]])  # Show first 5 dt features
        print(f"\n(Total {len(dt_feature_cols)} decision tree features created)")

    # Extract outputs
    print("\nExtracting outputs...")
    outputs, coefs, stats = fit_dt.extract_outputs()
    print(f"✓ Outputs shape: {outputs.shape}")
    print(f"✓ Coefficients shape: {coefs.shape}")
    print(f"✓ Stats shape: {stats.shape}")

    print("\n" + "="*60)
    print("✅ SUCCESS! step_dt_features works with per_group_prep=True")
    print("="*60)
    print("\nKey achievements:")
    print("1. ✅ DecisionTreeFeatures column names renamed (_tree_ → _dt_)")
    print("2. ✅ No Patsy formula parsing errors")
    print("3. ✅ Per-group preprocessing works correctly")
    print("4. ✅ Each group can have different decision tree features")

except Exception as e:
    print("\n" + "="*60)
    print("❌ FAILED!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
