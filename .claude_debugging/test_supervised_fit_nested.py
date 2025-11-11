"""
Test that fit_nested() works with supervised feature selection steps.

This tests the fix for the issue where supervised steps like step_select_permutation
need the outcome during prep().
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe, step_select_permutation
from py_parsnip import linear_reg
from sklearn.ensemble import RandomForestRegressor

# Create sample data
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'country': ['USA'] * 50 + ['UK'] * 50,
    'date': pd.date_range('2020-01-01', periods=100),
    'x1': np.random.randn(n) * 10 + 50,
    'x2': np.random.randn(n) * 5 + 20,
    'x3': np.random.randn(n) * 3 + 10,
    'x4': np.random.randn(n) * 2 + 5,
    'x5': np.random.randn(n) * 7 + 30,
    'refinery_kbd': np.random.randn(n) * 15 + 100
})

print("=" * 70)
print("Testing fit_nested() with supervised feature selection")
print("=" * 70)

# Test 1: step_select_permutation with per_group_prep=True
print("\n1. Testing step_select_permutation with per_group_prep=True...")
try:
    rec = (
        recipe()
        .step_normalize()
        .step_select_permutation(
            outcome='refinery_kbd',
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            top_n=3
        )
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(data, group_col='country', per_group_prep=True)
    print("   ✓ SUCCESS: fit_nested with step_select_permutation works!")
    print(f"   Groups fitted: {list(fit.group_fits.keys())}")
except Exception as e:
    print(f"   ✗ FAILED: {str(e)}")

# Test 2: step_select_permutation with per_group_prep=False (global recipe)
print("\n2. Testing step_select_permutation with per_group_prep=False...")
try:
    rec = (
        recipe()
        .step_normalize()
        .step_select_permutation(
            outcome='refinery_kbd',
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            top_n=3
        )
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(data, group_col='country', per_group_prep=False)
    print("   ✓ SUCCESS: fit_nested with global recipe works!")
    print(f"   Groups fitted: {list(fit.group_fits.keys())}")
except Exception as e:
    print(f"   ✗ FAILED: {str(e)}")

# Test 3: Regular fit() should still work
print("\n3. Testing regular fit() with step_select_permutation...")
try:
    rec = (
        recipe()
        .step_normalize()
        .step_select_permutation(
            outcome='refinery_kbd',
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            top_n=3
        )
    )
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit(data)
    print("   ✓ SUCCESS: Regular fit() still works!")
except Exception as e:
    print(f"   ✗ FAILED: {str(e)}")

# Test 4: Non-supervised steps should still work (outcome excluded)
print("\n4. Testing non-supervised step (step_normalize only)...")
try:
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(data, group_col='country', per_group_prep=True)
    print("   ✓ SUCCESS: Non-supervised steps still work!")
    print(f"   Groups fitted: {list(fit.group_fits.keys())}")
except Exception as e:
    print(f"   ✗ FAILED: {str(e)}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
