"""
Run workflow with debug prints enabled to trace the exact flow.
"""

import pandas as pd
import numpy as np
import sys

# Enable debug mode
sys._workflow_debug = True

from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'group': ['A'] * 100,
    'feat1': np.random.randn(100),
    'feat2': np.random.randn(100),
    'feat3': np.random.randn(100),
    'feat4': np.random.randn(100),
    'feat5': np.random.randn(100),
    'target': np.random.randn(100)
})

print("=" * 80)
print("WORKFLOW FIT_NESTED WITH DEBUG PRINTS")
print("=" * 80)

# Create recipe with supervised filter
rec = (
    recipe()
    .step_filter_mutual_info(outcome='target', top_n=3)
    .step_normalize()
)

# Create workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

print("\nCalling fit_nested...")
nested_fit = wf.fit_nested(data, group_col='group', per_group_prep=True)

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)

group_a_fit = nested_fit.group_fits['A']
print(f"\nFormula: {group_a_fit.formula}")

prep_recipe = group_a_fit.pre
for i, step in enumerate(prep_recipe.prepared_steps):
    print(f"Step {i+1}: {type(step).__name__}")
    if hasattr(step, '_selected_features'):
        print(f"   Selected: {step._selected_features}")
    if hasattr(step, 'scaler') and hasattr(step.scaler, 'feature_names_in_'):
        print(f"   Scaler fitted on: {list(step.scaler.feature_names_in_)}")

print("\n" + "=" * 80)
