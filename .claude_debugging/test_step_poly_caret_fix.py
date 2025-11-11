"""
Test that step_poly no longer creates column names with ^ character
that cause patsy XOR errors.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

print("=" * 70)
print("TEST: step_poly ^ character fix")
print("=" * 70)

# Create test data
np.random.seed(42)
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'country': ['USA'] * 50 + ['UK'] * 50,
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'target': np.random.randn(100)
})

train_data = data[:80]
test_data = data[80:]

print(f"\n1. Create recipe with step_poly(degree=2)")
rec = recipe().step_poly(['x1', 'x2'], degree=2, inplace=False)

print(f"\n2. Prep and bake recipe to check column names")
prepped = rec.prep(train_data)
baked = prepped.bake(train_data)

# Check for ^ character in column names
poly_cols = [col for col in baked.columns if 'pow' in col or '^' in col]
has_caret = any('^' in col for col in baked.columns)

print(f"\nPolynomial columns created: {poly_cols}")
print(f"Contains ^ character: {has_caret}")

if has_caret:
    print("❌ FAILURE: Column names still contain ^ character")
else:
    print("✓ SUCCESS: No ^ character in column names")

print(f"\n3. Test with workflow.fit_nested() (the failing pattern)")
wf = workflow().add_recipe(rec).add_model(linear_reg())

try:
    fit = wf.fit_nested(train_data, group_col='country')
    fit = fit.evaluate(test_data)
    print("✓ SUCCESS: fit_nested() completed without patsy XOR errors")

    # Extract outputs to verify everything works end-to-end
    outputs, _, _ = fit.extract_outputs()
    print(f"✓ Outputs extracted: {outputs.shape[0]} rows")
    print(f"✓ First column: '{outputs.columns[0]}'")
    print(f"✓ Second column: '{outputs.columns[1]}'")

except Exception as e:
    print(f"❌ FAILURE: {type(e).__name__}: {str(e)[:100]}")

print(f"\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
