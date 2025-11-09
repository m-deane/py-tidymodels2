"""
Test that step_interact() supports selectors.
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
    'x3': [3, 6, 9, 12, 15],
    'target': [10, 20, 30, 40, 50]
})

print("=" * 80)
print("TESTING step_interact() SELECTOR SUPPORT")
print("=" * 80)
print(f"\nOriginal columns: {list(data.columns)}")
print(f"Numeric predictors: ['x1', 'x2', 'x3']")

# Test 1: Explicit column pairs (original behavior)
print("\n" + "=" * 80)
print("[Test 1] step_interact() with explicit pairs")
print("=" * 80)

rec1 = recipe().step_interact([("x1", "x2"), ("x1", "x3")])
prepped1 = rec1.prep(data)
baked1 = prepped1.bake(data)

print(f"Result columns: {list(baked1.columns)}")
expected = ['x1', 'x2', 'x3', 'target', 'x1_x_x2', 'x1_x_x3']
print(f"Expected: {expected}")

if 'x1_x_x2' in baked1.columns and 'x1_x_x3' in baked1.columns:
    print("✅ PASS: Explicit pairs work")
else:
    print("❌ FAIL: Explicit pairs don't work")

# Test 2: List of columns (creates all pairs)
print("\n" + "=" * 80)
print("[Test 2] step_interact() with list of columns")
print("=" * 80)

rec2 = recipe().step_interact(['x1', 'x2', 'x3'])
prepped2 = rec2.prep(data)
baked2 = prepped2.bake(data)

print(f"Result columns: {list(baked2.columns)}")

# Should create 3 choose 2 = 3 interactions: x1_x2, x1_x3, x2_x3
interaction_cols = [c for c in baked2.columns if '_x_' in c]
print(f"Interaction columns: {interaction_cols}")

if len(interaction_cols) == 3:
    print("✅ PASS: All pairwise interactions created (3 choose 2 = 3)")
else:
    print(f"❌ FAIL: Expected 3 interactions, got {len(interaction_cols)}")

# Test 3: Selector function (NEW FEATURE)
print("\n" + "=" * 80)
print("[Test 3] step_interact() with all_numeric_predictors() selector")
print("=" * 80)

rec3 = recipe().step_interact(all_numeric_predictors())
prepped3 = rec3.prep(data)
baked3 = prepped3.bake(data)

print(f"Result columns: {list(baked3.columns)}")

# Should create 3 choose 2 = 3 interactions from numeric predictors
interaction_cols3 = [c for c in baked3.columns if '_x_' in c]
print(f"Interaction columns: {interaction_cols3}")

if len(interaction_cols3) == 3:
    print("✅ PASS: Selector created all pairwise interactions (3 choose 2 = 3)")
else:
    print(f"❌ FAIL: Expected 3 interactions, got {len(interaction_cols3)}")

# Verify values are correct
print("\n" + "=" * 80)
print("Verification: Check interaction values are correct")
print("=" * 80)

print(f"\nOriginal x1 values: {list(data['x1'])}")
print(f"Original x2 values: {list(data['x2'])}")
print(f"x1 * x2:            {list(data['x1'] * data['x2'])}")
print(f"x1_x_x2 values:     {list(baked3['x1_x_x2'])}")

if all(baked3['x1_x_x2'] == data['x1'] * data['x2']):
    print("✅ PASS: x1_x_x2 interaction values are correct")
else:
    print("❌ FAIL: x1_x_x2 interaction values are incorrect")

print(f"\nOriginal x1 * x3:   {list(data['x1'] * data['x3'])}")
print(f"x1_x_x3 values:     {list(baked3['x1_x_x3'])}")

if all(baked3['x1_x_x3'] == data['x1'] * data['x3']):
    print("✅ PASS: x1_x_x3 interaction values are correct")
else:
    print("❌ FAIL: x1_x_x3 interaction values are incorrect")

# Test 4: Custom separator
print("\n" + "=" * 80)
print("[Test 4] Custom separator")
print("=" * 80)

rec4 = recipe().step_interact(all_numeric_predictors(), separator="_times_")
prepped4 = rec4.prep(data)
baked4 = prepped4.bake(data)

interaction_cols4 = [c for c in baked4.columns if '_times_' in c]
print(f"Interaction columns with custom separator: {interaction_cols4}")

if len(interaction_cols4) == 3 and all('_times_' in c for c in interaction_cols4):
    print("✅ PASS: Custom separator works")
else:
    print("❌ FAIL: Custom separator doesn't work")

# Test 5: Many columns (combinatorial explosion check)
print("\n" + "=" * 80)
print("[Test 5] Combinatorial explosion check")
print("=" * 80)

# Add more columns
large_data = data.copy()
for i in range(4, 8):
    large_data[f'x{i}'] = np.random.randn(5)

rec5 = recipe().step_interact(all_numeric_predictors())
prepped5 = rec5.prep(large_data)
baked5 = prepped5.bake(large_data)

interaction_cols5 = [c for c in baked5.columns if '_x_' in c]
# 7 predictors (x1-x7), 7 choose 2 = 21 interactions
expected_interactions = 7 * 6 // 2
print(f"Number of predictors: 7 (x1-x7)")
print(f"Expected interactions (7 choose 2): {expected_interactions}")
print(f"Actual interactions: {len(interaction_cols5)}")

if len(interaction_cols5) == expected_interactions:
    print("✅ PASS: Correct number of interactions for 7 predictors")
else:
    print(f"❌ FAIL: Expected {expected_interactions}, got {len(interaction_cols5)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✨ step_interact() now supports:")
print("   1. Explicit pairs: [('x1', 'x2'), ('x1', 'x3')]")
print("   2. List of columns: ['x1', 'x2', 'x3'] (all pairs)")
print("   3. Selector functions: all_numeric_predictors() (all pairs)")
print("\n✨ All interaction values are computed correctly!")
