"""
Test if recipe steps are being reused across multiple prep() calls.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe

# Create two different datasets
np.random.seed(42)
data_a = pd.DataFrame({
    'feat1': np.random.randn(100),
    'feat2': np.random.randn(100),
    'feat3': np.random.randn(100),
    'feat4': np.random.randn(100),
    'feat5': np.random.randn(100),
    'target': np.random.randn(100)
})

np.random.seed(123)  # Different seed = different MI rankings
data_b = pd.DataFrame({
    'feat1': np.random.randn(100),
    'feat2': np.random.randn(100),
    'feat3': np.random.randn(100),
    'feat4': np.random.randn(100),
    'feat5': np.random.randn(100),
    'target': np.random.randn(100)
})

print("=" * 80)
print("TESTING RECIPE REUSE ACROSS MULTIPLE PREP() CALLS")
print("=" * 80)

# Create recipe ONCE
rec = (
    recipe()
    .step_filter_mutual_info(outcome='target', top_n=3)
    .step_normalize()
)

print("\n1. Original recipe steps:")
print(f"   Step 1: {type(rec.steps[0]).__name__} (id={id(rec.steps[0])})")
print(f"   Step 2: {type(rec.steps[1]).__name__} (id={id(rec.steps[1])})")

print("\n2. Prep on data_a:")
prep_a = rec.prep(data_a)
filter_step_a = prep_a.prepared_steps[0]
print(f"   Selected features: {filter_step_a._selected_features}")
print(f"   PreparedStep id: {id(filter_step_a)}")
print(f"   Original step id: {id(rec.steps[0])}")
print(f"   Are they the same object? {filter_step_a is rec.steps[0]}")

print("\n3. Prep on data_b (SAME recipe object):")
prep_b = rec.prep(data_b)
filter_step_b = prep_b.prepared_steps[0]
print(f"   Selected features: {filter_step_b._selected_features}")
print(f"   PreparedStep id: {id(filter_step_b)}")
print(f"   Original step id: {id(rec.steps[0])}")
print(f"   Are they the same object? {filter_step_b is rec.steps[0]}")

print("\n4. Check if prep_a's filter step was modified:")
print(f"   prep_a filter selected: {filter_step_a._selected_features}")
print(f"   prep_b filter selected: {filter_step_b._selected_features}")
print(f"   Are they the same object? {filter_step_a is filter_step_b}")

if filter_step_a is filter_step_b:
    print("\n   ⚠️  BUG CONFIRMED: Both preps share the SAME step object!")
    print("   ⚠️  The second prep() OVERWRITES the first prep's state!")

print("\n" + "=" * 80)
