"""
Debug test for step_naomit to understand why it's not removing NaN rows.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe

# Create simple test data
np.random.seed(42)
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'x2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'y': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
})

print("=" * 70)
print("Debug: step_naomit with step_lag")
print("=" * 70)
print(f"\nOriginal data shape: {data.shape}")
print(f"Original columns: {list(data.columns)}")

# Create recipe with lag and naomit
rec = (
    recipe()
    .step_lag(['x1', 'x2'], lags=[1, 2])
    .step_naomit()
)

print("\n1. Prepping recipe...")
rec_prepped = rec.prep(data)
print(f"   ✓ Recipe prepped")

# Check prepared steps
print(f"\n2. Prepared steps: {len(rec_prepped.prepared_steps)}")
for i, step in enumerate(rec_prepped.prepared_steps):
    print(f"   Step {i}: {type(step).__name__}")
    if hasattr(step, 'columns'):
        print(f"      Columns: {step.columns if hasattr(step, 'columns') else 'N/A'}")

print(f"\n3. Baking data...")
baked_data = rec_prepped.bake(data)
print(f"   ✓ Data baked")
print(f"   Baked data shape: {baked_data.shape}")
print(f"   Baked columns: {list(baked_data.columns)}")
print(f"   NaN count per column:")
for col in baked_data.columns:
    nan_count = baked_data[col].isna().sum()
    print(f"      {col}: {nan_count} NaN")

print(f"\n4. Checking if NaN rows were removed...")
total_nan = baked_data.isna().any(axis=1).sum()
if total_nan == 0:
    print(f"   ✓ SUCCESS: No rows with NaN values")
else:
    print(f"   ✗ FAILURE: {total_nan} rows still have NaN values")
    print(f"\n   Rows with NaN:")
    nan_rows = baked_data[baked_data.isna().any(axis=1)]
    print(nan_rows)

print("\n" + "=" * 70)
print("Debug complete")
print("=" * 70)
