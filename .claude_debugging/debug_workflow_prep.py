"""
Debug the EXACT workflow fit_nested flow to find where the bug occurs.
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
    'group': ['A'] * 100 + ['B'] * 100,
    'feat1': np.random.randn(200),
    'feat2': np.random.randn(200),
    'feat3': np.random.randn(200),
    'feat4': np.random.randn(200),
    'feat5': np.random.randn(200),
    'target': np.random.randn(200)
})

print("=" * 80)
print("DEBUGGING WORKFLOW FIT_NESTED FLOW")
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

print("\n1. Calling fit_nested...")
nested_fit = wf.fit_nested(data, group_col='group', per_group_prep=True)

print("\n2. Checking Group A's workflow fit...")
group_a_fit = nested_fit.group_fits['A']
print(f"   Formula: {group_a_fit.formula}")

# Check the prepared recipe
prep_recipe = group_a_fit.pre
print(f"\n3. Group A's prepared recipe steps:")
for i, step in enumerate(prep_recipe.prepared_steps):
    print(f"   Step {i+1}: {type(step).__name__}")
    if hasattr(step, '_selected_features'):
        print(f"      Selected features: {step._selected_features}")
    if hasattr(step, 'scaler') and hasattr(step.scaler, 'feature_names_in_'):
        print(f"      Scaler fitted on: {list(step.scaler.feature_names_in_)}")

# Check the model's sklearn model
model_fit = group_a_fit.fit
if hasattr(model_fit, 'fit_data') and 'model' in model_fit.fit_data:
    sklearn_model = model_fit.fit_data['model']
    if hasattr(sklearn_model, 'feature_names_in_'):
        print(f"\n4. Model's sklearn feature names:")
        print(f"   {list(sklearn_model.feature_names_in_)}")

print("\n5. Manual bake test on Group A training data...")
group_a_train = data[data['group'] == 'A'].drop(columns=['group']).copy()
print(f"   Input columns: {list(group_a_train.columns)}")

try:
    baked = prep_recipe.bake(group_a_train)
    print(f"   ✓ Baked successfully")
    print(f"   Output columns: {list(baked.columns)}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "=" * 80)
