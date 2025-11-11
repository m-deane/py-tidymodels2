"""
Test that supervised feature selection works correctly during evaluate() with test data.

This tests the fix for the issue where supervised steps need outcome during bake,
but evaluate() was separating outcome from predictors before baking.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create grouped data
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
        'x4': np.random.randn(n_per_group) * 12 + 40,
        'refinery_kbd': np.random.randn(n_per_group) * 15 + 100
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)
train_data = data.iloc[:80]
test_data = data.iloc[80:]

print("=" * 70)
print("Testing Supervised Feature Selection with evaluate()")
print("=" * 70)

# Test 1: step_filter_anova + step_normalize
print("\n1. Testing step_filter_anova + step_normalize...")
try:
    from py_recipes.selectors import all_numeric_predictors

    rec_anova = (
        recipe()
        .step_filter_anova(outcome="refinery_kbd", top_p=0.5, use_pvalue=True)
        .step_normalize(all_numeric_predictors())
    )

    wf_anova = workflow().add_recipe(rec_anova).add_model(linear_reg())

    # Fit nested
    fit_anova = wf_anova.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("   ✓ fit_nested() succeeded")

    # Evaluate (this was failing before the fix)
    fit_anova = fit_anova.evaluate(test_data)
    print("   ✓ evaluate() succeeded")

    # Extract outputs
    outputs, coeffs, stats = fit_anova.extract_outputs()
    print(f"   ✓ extract_outputs() succeeded")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   ✓ SUCCESS: step_filter_anova + evaluate() works!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

# Test 2: step_filter_rf_importance + step_normalize
print("\n2. Testing step_filter_rf_importance + step_normalize...")
try:
    from py_recipes.selectors import all_numeric_predictors

    rec_rf = (
        recipe()
        .step_filter_rf_importance(outcome="refinery_kbd", top_n=3)
        .step_normalize(all_numeric_predictors())
    )

    wf_rf = workflow().add_recipe(rec_rf).add_model(linear_reg())

    # Fit nested
    fit_rf = wf_rf.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("   ✓ fit_nested() succeeded")

    # Evaluate
    fit_rf = fit_rf.evaluate(test_data)
    print("   ✓ evaluate() succeeded")

    # Extract outputs
    outputs, coeffs, stats = fit_rf.extract_outputs()
    print(f"   ✓ extract_outputs() succeeded")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   ✓ SUCCESS: step_filter_rf_importance + evaluate() works!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

# Test 3: Global recipe (per_group_prep=False)
print("\n3. Testing supervised steps with global recipe...")
try:
    from py_recipes.selectors import all_numeric_predictors

    rec_global = (
        recipe()
        .step_filter_anova(outcome="refinery_kbd", top_p=0.5)
        .step_normalize(all_numeric_predictors())
    )

    wf_global = workflow().add_recipe(rec_global).add_model(linear_reg())

    # Fit nested with global recipe
    fit_global = wf_global.fit_nested(train_data, group_col='country', per_group_prep=False)
    print("   ✓ fit_nested() succeeded")

    # Evaluate
    fit_global = fit_global.evaluate(test_data)
    print("   ✓ evaluate() succeeded")

    # Extract outputs
    outputs, coeffs, stats = fit_global.extract_outputs()
    print(f"   ✓ extract_outputs() succeeded")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   ✓ SUCCESS: Global recipe + evaluate() works!")

except Exception as e:
    print(f"   ✗ FAILED: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
