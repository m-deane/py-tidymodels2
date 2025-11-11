"""
Test that add_model_name() and add_model_group_name() methods work correctly.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

print("=" * 70)
print("TEST: add_model_name() and add_model_group_name() methods")
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

print(f"\n1. Test with standard workflow (fit)")

# Create workflow with custom names
wf_poly = (
    workflow()
    .add_recipe(recipe().step_poly(['x1', 'x2'], degree=2, inplace=False))
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("poly")
    .add_model_group_name("polynomial_models")
)

# Fit and evaluate
fit_poly = wf_poly.fit(train_data)
fit_poly = fit_poly.evaluate(test_data)

# Extract outputs
outputs, _, _ = fit_poly.extract_outputs()

print(f"   Model names in outputs: {outputs['model'].unique()}")
print(f"   Model group names in outputs: {outputs['model_group_name'].unique()}")

# Verify
if 'poly' in outputs['model'].values:
    print("   ✓ SUCCESS: model_name 'poly' found in outputs")
else:
    print("   ✗ FAILURE: model_name 'poly' not found")

if 'polynomial_models' in outputs['model_group_name'].values:
    print("   ✓ SUCCESS: model_group_name 'polynomial_models' found in outputs")
else:
    print("   ✗ FAILURE: model_group_name 'polynomial_models' not found")

print(f"\n2. Test with nested workflow (fit_nested)")

# Create workflow with custom names
wf_nested = (
    workflow()
    .add_recipe(recipe().step_normalize())
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("baseline")
    .add_model_group_name("linear_models")
)

# Fit nested
fit_nested = wf_nested.fit_nested(train_data, group_col='country')
fit_nested = fit_nested.evaluate(test_data)

# Extract outputs
outputs_nested, _, _ = fit_nested.extract_outputs()

print(f"   Model names in outputs: {outputs_nested['model'].unique()}")
print(f"   Model group names in outputs: {outputs_nested['model_group_name'].unique()}")

# Verify
if 'baseline' in outputs_nested['model'].values:
    print("   ✓ SUCCESS: model_name 'baseline' found in nested outputs")
else:
    print("   ✗ FAILURE: model_name 'baseline' not found in nested outputs")

if 'linear_models' in outputs_nested['model_group_name'].values:
    print("   ✓ SUCCESS: model_group_name 'linear_models' found in nested outputs")
else:
    print("   ✗ FAILURE: model_group_name 'linear_models' not found in nested outputs")

print(f"\n3. Test method chaining order")

# Verify that method chaining works in any order
wf_chain = (
    workflow()
    .add_model_name("test_model")  # Before add_model
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_group_name("test_group")  # After add_model
    .add_recipe(recipe().step_normalize())  # After everything
)

fit_chain = wf_chain.fit(train_data)
outputs_chain, _, _ = fit_chain.extract_outputs()

print(f"   Model name: {outputs_chain['model'].unique()[0]}")
print(f"   Model group name: {outputs_chain['model_group_name'].unique()[0]}")

if outputs_chain['model'].unique()[0] == 'test_model':
    print("   ✓ SUCCESS: Method chaining works correctly")
else:
    print("   ✗ FAILURE: Method chaining failed")

print(f"\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
