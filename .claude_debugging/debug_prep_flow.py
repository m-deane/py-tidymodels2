"""
Debug the exact flow during recipe prep to see what each step sees.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe

# Create simple data
np.random.seed(42)
data = pd.DataFrame({
    'feat1': np.random.randn(100),
    'feat2': np.random.randn(100),
    'feat3': np.random.randn(100),
    'feat4': np.random.randn(100),
    'feat5': np.random.randn(100),
    'target': np.random.randn(100)
})

print("=" * 80)
print("DEBUGGING RECIPE PREP FLOW")
print("=" * 80)

print("\n1. Original data columns:")
print(f"   {list(data.columns)}")

# Create recipe
rec = (
    recipe()
    .step_filter_mutual_info(outcome='target', top_n=3)
    .step_normalize()
)

print("\n2. Prepping recipe...")
print("   (Manually stepping through each step)")

# Manually prep to see what each step receives
current_data = data.copy()
prepared_steps = []

for i, step in enumerate(rec.steps):
    print(f"\n3.{i+1}. Step {i+1}: {type(step).__name__}")
    print(f"     Input columns: {list(current_data.columns)}")

    # Prep the step
    prepared_step = step.prep(current_data, training=True)
    prepared_steps.append(prepared_step)

    # Bake to get data for next step
    baked_data = prepared_step.bake(current_data)
    print(f"     Output columns: {list(baked_data.columns)}")

    # Check what the step learned
    if hasattr(prepared_step, '_selected_features'):
        print(f"     Selected features: {prepared_step._selected_features}")
    if hasattr(prepared_step, 'scaler') and hasattr(prepared_step.scaler, 'feature_names_in_'):
        print(f"     Scaler fitted on: {list(prepared_step.scaler.feature_names_in_)}")

    current_data = baked_data

print("\n4. Final processed data columns:")
print(f"   {list(current_data.columns)}")

print("\n5. Testing bake on new data...")
test_data = pd.DataFrame({
    'feat1': np.random.randn(20),
    'feat2': np.random.randn(20),
    'feat3': np.random.randn(20),
    'feat4': np.random.randn(20),
    'feat5': np.random.randn(20),
    'target': np.random.randn(20)
})

print(f"   Test data columns: {list(test_data.columns)}")

# Now bake step by step
current_test_data = test_data.copy()
for i, prepared_step in enumerate(prepared_steps):
    print(f"\n6.{i+1}. Baking step {i+1}: {type(prepared_step).__name__}")
    print(f"     Input columns: {list(current_test_data.columns)}")
    try:
        baked_test = prepared_step.bake(current_test_data)
        print(f"     Output columns: {list(baked_test.columns)}")
        current_test_data = baked_test
    except Exception as e:
        print(f"     ✗ ERROR: {e}")
        break

print("\n" + "=" * 80)
