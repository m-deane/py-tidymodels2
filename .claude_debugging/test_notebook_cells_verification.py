"""
Verify that the key problematic notebook cells now work after all fixes.

This tests the specific patterns from _md/forecasting_recipes_grouped.ipynb
that were failing before the fixes.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe, all_numeric_predictors
from py_parsnip import linear_reg

# Create sample grouped data similar to notebook
np.random.seed(42)
n_per_group = 50

data_list = []
for country in ['USA', 'UK']:
    group_data = pd.DataFrame({
        'country': [country] * n_per_group,
        'date': pd.date_range('2020-01-01', periods=n_per_group),
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'x2': np.random.randn(n_per_group) * 5 + 20,
        'x3': np.random.randn(n_per_group) * 8 + 30,
        'refinery_kbd': np.random.randn(n_per_group) * 15 + 100
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)
train_data = data.iloc[:80]
test_data = data.iloc[80:]

print("=" * 70)
print("Verifying Notebook Cell Patterns")
print("=" * 70)

# Test 1: Cell 32 Pattern - step_select_corr with correct API
print("\n1. Testing Cell 32 pattern (step_select_corr)...")
try:
    rec_corr = (
        recipe()
        .step_select_corr(outcome='refinery_kbd', threshold=0.4, method='multicollinearity')
        .step_normalize(all_numeric_predictors())
    )
    wf_corr = (
        workflow()
        .add_recipe(rec_corr)
        .add_model(linear_reg())
    )
    fit_corr = wf_corr.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit_corr = fit_corr.evaluate(test_data)

    outputs, coeffs, stats = fit_corr.extract_outputs()
    print(f"   ✓ SUCCESS: step_select_corr works")
    print(f"   Outputs shape: {outputs.shape}")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)[:100]}")
    traceback.print_exc()

# Test 2: Cell 49 Pattern - step_lag with step_naomit
print("\n2. Testing Cell 49 pattern (step_lag + step_naomit)...")
try:
    rec_lag = (
        recipe()
        .step_lag(['x1', 'x2', 'x3'], lags=[1])
        .step_naomit()
    )
    wf_lag = workflow().add_recipe(rec_lag).add_model(linear_reg())
    fit_lag = wf_lag.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit_lag = fit_lag.evaluate(test_data)

    outputs, coeffs, stats = fit_lag.extract_outputs()
    print(f"   ✓ SUCCESS: step_lag + step_naomit works")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals")
    else:
        print(f"   ✗ Found {nan_count} NaN values!")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)[:100]}")
    traceback.print_exc()

# Test 3: Cell 50 Pattern - step_diff with step_naomit
print("\n3. Testing Cell 50 pattern (step_diff + step_naomit)...")
try:
    rec_diff = (
        recipe()
        .step_diff(['x1', 'x2'], lag=1)
        .step_naomit()
    )
    wf_diff = workflow().add_recipe(rec_diff).add_model(linear_reg())
    fit_diff = wf_diff.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit_diff = fit_diff.evaluate(test_data)

    outputs, coeffs, stats = fit_diff.extract_outputs()
    print(f"   ✓ SUCCESS: step_diff + step_naomit works")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals")
    else:
        print(f"   ✗ Found {nan_count} NaN values!")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)[:100]}")
    traceback.print_exc()

# Test 4: Cell 69 Pattern - step_sqrt with step_naomit
print("\n4. Testing Cell 69 pattern (step_naomit + step_sqrt)...")
try:
    # Make some values negative to test sqrt NaN handling
    train_data_copy = train_data.copy()
    train_data_copy.loc[0:5, 'x1'] = -10  # Some negative values
    test_data_copy = test_data.copy()

    rec_sqrt = (
        recipe()
        .step_naomit()  # Remove existing NaN first
        .step_sqrt(all_numeric_predictors())  # May create more NaN from negatives
    )
    wf_sqrt = workflow().add_recipe(rec_sqrt).add_model(linear_reg())
    fit_sqrt = wf_sqrt.fit_nested(train_data_copy, group_col='country', per_group_prep=True)
    fit_sqrt = fit_sqrt.evaluate(test_data_copy)

    outputs, coeffs, stats = fit_sqrt.extract_outputs()
    print(f"   ✓ SUCCESS: step_sqrt with step_naomit works")
    print(f"   Outputs shape: {outputs.shape}")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)[:100]}")
    traceback.print_exc()

# Test 5: Complex recipe with multiple steps
print("\n5. Testing complex recipe (multiple steps + naomit)...")
try:
    rec_complex = (
        recipe()
        .step_lag(['x1', 'x2'], lags=[1, 2])
        .step_diff(['x3'], lag=1)
        .step_naomit()  # Remove all NaN created by lag and diff
        .step_normalize(all_numeric_predictors())
    )
    wf_complex = workflow().add_recipe(rec_complex).add_model(linear_reg())
    fit_complex = wf_complex.fit_nested(train_data, group_col='country', per_group_prep=True)
    fit_complex = fit_complex.evaluate(test_data)

    outputs, coeffs, stats = fit_complex.extract_outputs()
    print(f"   ✓ SUCCESS: Complex recipe works")
    print(f"   Outputs shape: {outputs.shape}")

    # Verify no NaN
    nan_count = outputs['actuals'].isna().sum()
    if nan_count == 0:
        print(f"   ✓ No NaN values in actuals")
    else:
        print(f"   ✗ Found {nan_count} NaN values!")

except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {str(e)[:100]}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Verification Complete!")
print("=" * 70)
print("\nAll patterns from the notebook should now work correctly.")
print("The notebook can be re-run from beginning with kernel restart.")
