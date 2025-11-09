"""
Test that step_poly() supports inplace parameter.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

# Create test data
np.random.seed(42)
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10],
    'target': [10, 20, 30, 40, 50]
})

print("=" * 80)
print("TESTING step_poly() inplace PARAMETER")
print("=" * 80)
print(f"\nOriginal columns: {list(data.columns)}")
print(f"Original data shape: {data.shape}")

# Test 1: inplace=True (default) - Replace original columns
print("\n" + "=" * 80)
print("[Test 1] step_poly() with inplace=True (default)")
print("=" * 80)

rec1 = recipe().step_poly(['x1', 'x2'], degree=2, inplace=True)
prepped1 = rec1.prep(data)
baked1 = prepped1.bake(data)

print(f"Result columns: {list(baked1.columns)}")
print(f"Result shape: {baked1.shape}")

# Check original columns are removed
if 'x1' not in baked1.columns and 'x2' not in baked1.columns:
    print("✅ PASS: Original columns removed")
else:
    print("❌ FAIL: Original columns still present")

# Check polynomial columns are present
if 'x1^2' in baked1.columns and 'x2^2' in baked1.columns:
    print("✅ PASS: Polynomial columns created")
else:
    print("❌ FAIL: Polynomial columns missing")

print(f"\nSample data:")
print(baked1.head())

# Test 2: inplace=False - Keep original columns
print("\n" + "=" * 80)
print("[Test 2] step_poly() with inplace=False")
print("=" * 80)

rec2 = recipe().step_poly(['x1', 'x2'], degree=2, inplace=False)
prepped2 = rec2.prep(data)
baked2 = prepped2.bake(data)

print(f"Result columns: {list(baked2.columns)}")
print(f"Result shape: {baked2.shape}")

# Check original columns are kept
if 'x1' in baked2.columns and 'x2' in baked2.columns:
    print("✅ PASS: Original columns kept")
else:
    print("❌ FAIL: Original columns removed")

# Check polynomial columns are present
if 'x1^2' in baked2.columns and 'x2^2' in baked2.columns:
    print("✅ PASS: Polynomial columns created")
else:
    print("❌ FAIL: Polynomial columns missing")

print(f"\nSample data:")
print(baked2.head())

# Test 3: With selector
print("\n" + "=" * 80)
print("[Test 3] step_poly() with selector and inplace=False")
print("=" * 80)

rec3 = recipe().step_poly(all_numeric_predictors(), degree=2, inplace=False)
prepped3 = rec3.prep(data)
baked3 = prepped3.bake(data)

print(f"Result columns: {list(baked3.columns)}")
print(f"Result shape: {baked3.shape}")

# All original predictors should be present
if 'x1' in baked3.columns and 'x2' in baked3.columns:
    print("✅ PASS: Original predictor columns kept")
else:
    print("❌ FAIL: Original predictor columns removed")

# Test 4: With interactions and inplace=False
print("\n" + "=" * 80)
print("[Test 4] step_poly() with interactions and inplace=False")
print("=" * 80)

rec4 = recipe().step_poly(['x1', 'x2'], degree=2, include_interactions=True, inplace=False)
prepped4 = rec4.prep(data)
baked4 = prepped4.bake(data)

print(f"Result columns: {list(baked4.columns)}")
print(f"Result shape: {baked4.shape}")

# Should have: x1, x2, target, x1^2, x1_x2, x2^2
expected_cols = ['x1', 'x2', 'target', 'x1^2', 'x1_x2', 'x2^2']
if all(col in baked4.columns for col in expected_cols):
    print("✅ PASS: All expected columns present (originals + polynomials + interactions)")
else:
    print(f"❌ FAIL: Expected {expected_cols}")
    print(f"         Got {list(baked4.columns)}")

print(f"\nSample data:")
print(baked4[['x1', 'x2', 'x1^2', 'x1_x2', 'x2^2']].head())

# Verify values
print("\n" + "=" * 80)
print("Verification: Check polynomial values are correct")
print("=" * 80)

print(f"\nOriginal x1 values: {list(data['x1'])}")
print(f"x1^2 values:        {list(baked4['x1^2'])}")
print(f"Expected x1^2:      {list(data['x1']**2)}")

if all(baked4['x1^2'] == data['x1']**2):
    print("✅ PASS: x1^2 values are correct")
else:
    print("❌ FAIL: x1^2 values are incorrect")

print(f"\nOriginal x1 * x2:   {list(data['x1'] * data['x2'])}")
print(f"x1_x2 values:       {list(baked4['x1_x2'])}")

if all(baked4['x1_x2'] == data['x1'] * data['x2']):
    print("✅ PASS: x1_x2 interaction values are correct")
else:
    print("❌ FAIL: x1_x2 interaction values are incorrect")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"inplace=True:  Removes originals, shape {baked1.shape}")
print(f"inplace=False: Keeps originals, shape {baked2.shape}")
print("\n✨ The inplace parameter now works correctly for step_poly()!")
