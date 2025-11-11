"""
Test that step_safe_v2 and step_splitwise work correctly with per-group preprocessing.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg
from sklearn.ensemble import RandomForestRegressor

print("=" * 80)
print("TESTING step_safe_v2 AND step_splitwise PER-GROUP FIX")
print("=" * 80)

# Create test data with multiple groups
np.random.seed(42)
n_per_group = 150

data_list = []
for group in ['Algeria', 'Denmark']:
    group_data = pd.DataFrame({
        'date': pd.date_range('2018-01-01', periods=n_per_group, freq='ME'),
        'country': [group] * n_per_group,
        'x1': np.random.randn(n_per_group),
        'x2': np.random.randn(n_per_group),
        'x3': np.random.randn(n_per_group),
        'x4': np.random.randn(n_per_group),
        'refinery_kbd': np.random.randn(n_per_group) * 10 + 50
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)
train_data = data[data['date'] < '2022-01-01'].copy()
test_data = data[data['date'] >= '2022-01-01'].copy()

print(f"\nData: {len(train_data)} train, {len(test_data)} test")
print(f"Groups: {sorted(data['country'].unique())}")

# Test 1: step_safe_v2
print("\n" + "-" * 80)
print("TEST 1: step_safe_v2 with per_group_prep=True")
print("-" * 80)

try:
    # Create unfitted surrogate model
    surrogate = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    
    rec_safe = (
        recipe()
        .step_safe_v2(
            surrogate_model=surrogate,
            outcome='refinery_kbd',
            top_n=3,
            grid_resolution=10
        )
        .step_normalize()
    )
    
    wf_safe = workflow().add_recipe(rec_safe).add_model(linear_reg())
    
    print("  Fitting nested model...")
    fit_safe = wf_safe.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("  ✓ Fit completed")
    
    print("  Evaluating on test data...")
    fit_safe = fit_safe.evaluate(test_data)
    print("  ✓ Evaluation completed")
    
    outputs, coeffs, stats = fit_safe.extract_outputs()
    print(f"  ✓ Outputs shape: {outputs.shape}")
    
    print("\n✅ TEST 1 PASSED - step_safe_v2 works with per-group preprocessing!")
    
except Exception as e:
    print(f"\n❌ TEST 1 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: step_splitwise
print("\n" + "-" * 80)
print("TEST 2: step_splitwise with per_group_prep=True")
print("-" * 80)

try:
    rec_split = (
        recipe()
        .step_splitwise(
            outcome='refinery_kbd',
            transformation_mode='univariate',
            min_support=0.1,
            feature_type='dummies'
        )
        .step_normalize()
    )
    
    wf_split = workflow().add_recipe(rec_split).add_model(linear_reg())
    
    print("  Fitting nested model...")
    fit_split = wf_split.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("  ✓ Fit completed")
    
    print("  Evaluating on test data...")
    fit_split = fit_split.evaluate(test_data)
    print("  ✓ Evaluation completed")
    
    outputs, coeffs, stats = fit_split.extract_outputs()
    print(f"  ✓ Outputs shape: {outputs.shape}")
    
    print("\n✅ TEST 2 PASSED - step_splitwise works with per-group preprocessing!")
    
except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
