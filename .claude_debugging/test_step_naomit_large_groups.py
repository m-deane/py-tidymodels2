"""
Test step_naomit alignment with LARGE groups to trigger actual per-group preprocessing.

The previous test had groups smaller than min_group_size=30, so they fell back
to global recipe. This test uses 50+ samples per group.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create sample data with LARGE groups (50 per group)
np.random.seed(42)
n_per_group = 50
n_groups = 2

data_list = []
for i, country in enumerate(['USA', 'UK']):
    group_data = pd.DataFrame({
        'country': [country] * n_per_group,
        'date': pd.date_range(f'2020-01-01', periods=n_per_group),
        'x1': np.random.randn(n_per_group) * 10 + 50 + i*5,  # Different means
        'x2': np.random.randn(n_per_group) * 5 + 20 + i*2,
        'refinery_kbd': np.random.randn(n_per_group) * 15 + 100 + i*10
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)
train_data = data.iloc[:80]  # 40 per group in training
test_data = data.iloc[80:]   # 10 per group in test

print("=" * 70)
print("Testing step_naomit alignment with LARGE groups (per-group prep)")
print("=" * 70)
print(f"Total data: {len(data)} rows ({n_per_group} per group)")
print(f"Train: {len(train_data)} rows (40 per group)")
print(f"Test: {len(test_data)} rows (10 per group)")
print()

# Test 1: step_lag with step_naomit (per-group prep)
print("1. Testing step_lag with step_naomit (per_group_prep=True)...")
try:
    rec = (
        recipe()
        .step_lag(['x1', 'x2'], lags=[1, 2])
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train_data, group_col='country', per_group_prep=True)

    print("   ✓ fit_nested() succeeded")

    # Try evaluate
    fit = fit.evaluate(test_data)
    print("   ✓ evaluate() succeeded")

    outputs, coeffs, stats = fit.extract_outputs()
    print(f"   ✓ extract_outputs() succeeded")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Training rows preserved after naomit: {len(outputs[outputs['split']=='train'])}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

    print("   ✓ SUCCESS: Per-group step_lag with step_naomit works!")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 2: step_diff with step_naomit (per-group prep)
print("\n2. Testing step_diff with step_naomit (per_group_prep=True)...")
try:
    rec = (
        recipe()
        .step_diff(['x1', 'x2'], lag=1)
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train_data, group_col='country', per_group_prep=True)

    print("   ✓ fit_nested() succeeded")

    fit = fit.evaluate(test_data)
    print("   ✓ evaluate() succeeded")

    outputs, coeffs, stats = fit.extract_outputs()
    print(f"   ✓ extract_outputs() succeeded")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

    print("   ✓ SUCCESS: Per-group step_diff with step_naomit works!")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 3: Compare with global recipe (should still work)
print("\n3. Testing step_lag with step_naomit (global recipe)...")
try:
    rec = (
        recipe()
        .step_lag(['x1', 'x2'], lags=[1, 2])
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train_data, group_col='country', per_group_prep=False)
    fit = fit.evaluate(test_data)

    outputs, coeffs, stats = fit.extract_outputs()
    print("   ✓ SUCCESS: Global recipe works!")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
