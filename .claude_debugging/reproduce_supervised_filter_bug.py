"""
Script to reproduce supervised filter bug with per-group preprocessing.

Run this to see the exact error and understand the root cause.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg
from sklearn.ensemble import RandomForestRegressor

# Create synthetic data with groups
np.random.seed(42)
n_samples = 200
n_groups = 2

data = pd.DataFrame({
    'group': np.repeat(['A', 'B'], n_samples // 2),
    'feat1': np.random.randn(n_samples),
    'feat2': np.random.randn(n_samples),
    'feat3': np.random.randn(n_samples),
    'feat4': np.random.randn(n_samples),
    'feat5': np.random.randn(n_samples),
    'target': np.random.randn(n_samples)
})

# Split into train/test
train = data.iloc[:160].copy()
test = data.iloc[160:].copy()

print("=" * 80)
print("REPRODUCING SUPERVISED FILTER BUG")
print("=" * 80)

# Create recipe with supervised filter
rec = (
    recipe()
    .step_filter_mutual_info(
        outcome='target',
        top_n=3  # Select only 3 features
    )
    .step_normalize()
)

# Create workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

print("\n1. Fitting nested models with per_group_prep=True...")
nested_fit = wf.fit_nested(train, group_col='group', per_group_prep=True)
print("   ✓ Fit completed")

print("\n2. Checking what features each group selected...")
for group_name, group_fit in nested_fit.group_fits.items():
    # Get the prepared recipe
    prep_recipe = group_fit.pre

    # Find the filter step
    for step in prep_recipe.prepared_steps:
        if hasattr(step, '_selected_features'):
            print(f"   Group {group_name} selected: {step._selected_features}")
            break

print("\n3. Making predictions on test data...")
try:
    predictions = nested_fit.predict(test)
    print("   ✓ Predictions successful")
    print(f"   Shape: {predictions.shape}")
except Exception as e:
    print(f"   ✗ ERROR during predict: {e}")
    import traceback
    print("\n" + traceback.format_exc())

print("\n4. Running evaluate() on test data...")
try:
    eval_fit = nested_fit.evaluate(test)
    print("   ✓ Evaluate successful")
except Exception as e:
    print(f"   ✗ ERROR during evaluate: {e}")
    import traceback
    print("\n" + traceback.format_exc())

print("\n" + "=" * 80)
print("DEBUGGING: Let's manually trace what happens...")
print("=" * 80)

# Manually trace for Group A
group_a_test = test[test['group'] == 'A'].drop(columns=['group'])
group_a_fit = nested_fit.group_fits['A']
prep_recipe = group_a_fit.pre

print("\n5. Group A test data columns BEFORE baking:")
print(f"   {list(group_a_test.columns)}")

print("\n6. Baking Group A test data through recipe...")
try:
    baked = prep_recipe.bake(group_a_test)
    print(f"   ✓ Baked successfully")
    print(f"   Columns AFTER baking: {list(baked.columns)}")
except Exception as e:
    print(f"   ✗ ERROR during bake: {e}")
    import traceback
    print("\n" + traceback.format_exc())

print("\n7. Checking model expectations...")
model_fit = group_a_fit.fit
print(f"   Model type: {type(model_fit)}")
print(f"   Formula: {group_a_fit.formula}")

# Check sklearn model expectations
if hasattr(model_fit, 'fit_data') and 'model' in model_fit.fit_data:
    sklearn_model = model_fit.fit_data['model']
    if hasattr(sklearn_model, 'feature_names_in_'):
        print(f"   Sklearn expects features: {list(sklearn_model.feature_names_in_)}")

print("\n" + "=" * 80)
