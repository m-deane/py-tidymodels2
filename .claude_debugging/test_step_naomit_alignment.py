"""
Test that step_naomit correctly aligns outcome with baked data.

This tests the fix for the issue where step_naomit() removes rows, but the
outcome column wasn't being aligned correctly during recombination.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create sample data
np.random.seed(42)
n = 50

data = pd.DataFrame({
    'country': ['USA'] * 25 + ['UK'] * 25,
    'date': pd.date_range('2020-01-01', periods=50),
    'x1': np.random.randn(n) * 10 + 50,
    'x2': np.random.randn(n) * 5 + 20,
    'refinery_kbd': np.random.randn(n) * 15 + 100
})

train_data = data.iloc[:40]
test_data = data.iloc[40:]

print("=" * 70)
print("Testing step_naomit alignment with outcome column")
print("=" * 70)

# Test 1: step_lag with step_naomit
print("\n1. Testing step_lag with step_naomit...")
try:
    rec = (
        recipe()
        .step_lag(['x1', 'x2'], lags=[1, 2])
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit = fit.evaluate(test_data)

    outputs, coeffs, stats = fit.extract_outputs()
    print("   ✓ SUCCESS: step_lag with step_naomit works!")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Training rows preserved after naomit: {len(outputs[outputs['split']=='train'])}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")

# Test 2: step_diff with step_naomit
print("\n2. Testing step_diff with step_naomit...")
try:
    rec = (
        recipe()
        .step_diff(['x1', 'x2'], lag=1)
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit = fit.evaluate(test_data)

    outputs, coeffs, stats = fit.extract_outputs()
    print("   ✓ SUCCESS: step_diff with step_naomit works!")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")

# Test 3: Without per_group_prep (global recipe)
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
    print("   ✓ SUCCESS: step_lag with step_naomit works (global recipe)!")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN in outputs
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals column")
    else:
        print(f"   ✗ Found {nan_count} NaN values in actuals!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")

# Test 4: Regular fit() should still work
print("\n4. Testing regular fit() with step_lag and step_naomit...")
try:
    rec = (
        recipe()
        .step_lag(['x1', 'x2'], lags=[1, 2])
        .step_naomit()
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit(train_data)
    predictions = fit.predict(test_data)

    print("   ✓ SUCCESS: Regular fit() still works!")
    print(f"   Predictions shape: {predictions.shape}")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
