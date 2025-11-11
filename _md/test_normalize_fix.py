"""
Test to verify normalization excludes outcome column correctly
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric, one_of, difference, all_numeric_predictors
from py_workflows import workflow
from py_parsnip import linear_reg

# Create test data
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'x1': np.random.randn(n) * 10 + 50,  # Mean ~50, std ~10
    'x2': np.random.randn(n) * 5 + 20,   # Mean ~20, std ~5
    'x3': np.random.randn(n) * 2 + 10,   # Mean ~10, std ~2
    'refinery_kbd': np.random.randn(n) * 100 + 500,  # Outcome: Mean ~500, std ~100
})

print("Original data statistics:")
print(data.describe())
print()

# Test 1: What all_numeric_predictors() does (WRONG - includes outcome)
print("="*60)
print("Test 1: all_numeric_predictors() behavior")
print("="*60)

selector = all_numeric_predictors()
selected = selector(data)
print(f"Columns selected: {selected}")
print(f"Includes 'refinery_kbd'? {'refinery_kbd' in selected}")
print()

# Test 2: What difference(all_numeric(), one_of('refinery_kbd')) does (CORRECT)
print("="*60)
print("Test 2: difference(all_numeric(), one_of('refinery_kbd'))")
print("="*60)

selector = difference(all_numeric(), one_of('refinery_kbd'))
selected = selector(data)
print(f"Columns selected: {selected}")
print(f"Includes 'refinery_kbd'? {'refinery_kbd' in selected}")
print()

# Test 3: Verify normalization with corrected selector
print("="*60)
print("Test 3: Normalization with difference() selector")
print("="*60)

rec = (
    recipe()
    .step_normalize(difference(all_numeric(), one_of('refinery_kbd')))
)

prep_rec = rec.prep(data)
normalized = prep_rec.bake(data)

print("Normalized data statistics:")
print(normalized.describe())
print()

# Verify outcome is NOT normalized
print("Verification:")
print(f"x1 mean after normalization: {normalized['x1'].mean():.4f} (should be ~0)")
print(f"x1 std after normalization: {normalized['x1'].std():.4f} (should be ~1)")
print()
print(f"refinery_kbd mean after: {normalized['refinery_kbd'].mean():.2f} (should be ~500, NOT ~0)")
print(f"refinery_kbd std after: {normalized['refinery_kbd'].std():.2f} (should be ~100, NOT ~1)")
print()

if abs(normalized['refinery_kbd'].mean() - 500) < 50:
    print("✅ SUCCESS! Outcome column NOT normalized (mean preserved)")
else:
    print("❌ FAILED! Outcome column was normalized (mean changed)")

if normalized['refinery_kbd'].std() > 50:
    print("✅ SUCCESS! Outcome column NOT normalized (std preserved)")
else:
    print("❌ FAILED! Outcome column was normalized (std changed)")

# Test 4: With old selector (WRONG - normalizes outcome)
print()
print("="*60)
print("Test 4: Normalization with all_numeric_predictors() (WRONG)")
print("="*60)

rec_wrong = (
    recipe()
    .step_normalize(all_numeric_predictors())
)

prep_rec_wrong = rec_wrong.prep(data)
normalized_wrong = prep_rec_wrong.bake(data)

print("Normalized data with all_numeric_predictors():")
print(f"refinery_kbd mean: {normalized_wrong['refinery_kbd'].mean():.4f}")
print(f"refinery_kbd std: {normalized_wrong['refinery_kbd'].std():.4f}")
print()

if abs(normalized_wrong['refinery_kbd'].mean()) < 1:
    print("❌ Outcome WAS normalized (this is the BUG)")
else:
    print("✅ Outcome NOT normalized")
