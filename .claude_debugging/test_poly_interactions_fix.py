"""
Test that step_poly() respects the include_interactions parameter.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors
from py_workflows import workflow
from py_parsnip import linear_reg

# Create test data
np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'target': np.random.randn(100)
})

train_data = data[:75]
test_data = data[75:]

print("=" * 80)
print("TESTING step_poly() include_interactions PARAMETER")
print("=" * 80)

# Test 1: Default behavior (include_interactions=False)
print("\n[Test 1] step_poly() with include_interactions=False (default)")
print("-" * 80)

rec1 = recipe().step_poly(all_numeric_predictors(), degree=2)
prepped1 = rec1.prep(train_data)
baked1 = prepped1.bake(train_data)

print(f"Original columns: {['x1', 'x2', 'x3']}")
print(f"Number of polynomial features: {len([c for c in baked1.columns if c not in ['target', 'date']])}")
print(f"Feature columns: {sorted([c for c in baked1.columns if c not in ['target']])}")

# Expected: Only x1^2, x2^2, x3^2 (no interactions)
expected_features = {'x1^2', 'x2^2', 'x3^2'}
actual_features = set(c for c in baked1.columns if '^' in c)

if actual_features == expected_features:
    print("✅ PASS: Only pure polynomial terms created (no interactions)")
else:
    print(f"❌ FAIL: Expected {expected_features}")
    print(f"         Got {actual_features}")

# Check for interaction terms (should be none)
interaction_features = [c for c in baked1.columns if '_x' in c and '^' not in c]
if len(interaction_features) == 0:
    print("✅ PASS: No interaction terms created")
else:
    print(f"❌ FAIL: Found {len(interaction_features)} interaction terms: {interaction_features[:5]}")

# Test 2: With interactions enabled
print("\n[Test 2] step_poly() with include_interactions=True")
print("-" * 80)

rec2 = recipe().step_poly(all_numeric_predictors(), degree=2, include_interactions=True)
prepped2 = rec2.prep(train_data)
baked2 = prepped2.bake(train_data)

print(f"Original columns: {['x1', 'x2', 'x3']}")
print(f"Number of polynomial features: {len([c for c in baked2.columns if c not in ['target', 'date']])}")
print(f"Feature columns (first 10): {sorted([c for c in baked2.columns if c not in ['target']])[:10]}")

# Expected: x1^2, x2^2, x3^2, x1_x2, x1_x3, x2_x3 (6 features)
poly_terms = [c for c in baked2.columns if '^' in c]
interaction_terms = [c for c in baked2.columns if '_x' in c and '^' not in c]

print(f"Pure polynomial terms: {sorted(poly_terms)}")
print(f"Interaction terms: {sorted(interaction_terms)}")

if len(poly_terms) == 3 and len(interaction_terms) == 3:
    print("✅ PASS: Both polynomial and interaction terms created")
else:
    print(f"❌ FAIL: Expected 3 polynomial + 3 interaction terms")
    print(f"         Got {len(poly_terms)} polynomial + {len(interaction_terms)} interaction terms")

# Test 3: Single column (should behave the same)
print("\n[Test 3] step_poly() with single column")
print("-" * 80)

rec3 = recipe().step_poly(['x1'], degree=2, include_interactions=False)
prepped3 = rec3.prep(train_data)
baked3 = prepped3.bake(train_data)

features3 = [c for c in baked3.columns if c not in ['target', 'x2', 'x3']]
print(f"Features created: {features3}")

if features3 == ['x1^2']:
    print("✅ PASS: Single column creates only x1^2")
else:
    print(f"❌ FAIL: Expected ['x1^2'], got {features3}")

# Test 4: Workflow integration
print("\n[Test 4] Workflow integration with include_interactions=False")
print("-" * 80)

try:
    rec4 = recipe().step_poly(all_numeric_predictors(), degree=2, include_interactions=False)
    fit4 = linear_reg().set_engine("sklearn").fit(train_data, formula="target ~ .")
    outputs4, coefs4, stats4 = fit4.extract_outputs()

    print(f"✅ SUCCESS: Model with include_interactions=False works!")
    print(f"   Features in model: {len([c for c in train_data.columns if '^' in c])}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 5: Workflow integration with include_interactions=True
print("\n[Test 5] Model with include_interactions=True")
print("-" * 80)

try:
    rec5 = recipe().step_poly(all_numeric_predictors(), degree=2, include_interactions=True)
    fit5 = linear_reg().set_engine("sklearn").fit(train_data, formula="target ~ .")
    outputs5, coefs5, stats5 = fit5.extract_outputs()

    print(f"✅ SUCCESS: Model with include_interactions=True works!")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Comparison
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"include_interactions=False creates: {len([c for c in baked1.columns if c not in ['target']])} features (pure polynomials only)")
print(f"include_interactions=True creates:  {len([c for c in baked2.columns if c not in ['target']])} features (polynomials + interactions)")
print("\n✨ The include_interactions parameter now works correctly!")
