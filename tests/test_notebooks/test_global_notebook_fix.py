"""
Test that the global notebook fix resolves the normalization issue.

Verifies that all columns (including 'brent') are normalized when using
the 'target' column name instead of 'refinery_kbd'.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

# Load data
raw_data = pd.read_csv('_md/__data/refinery_margins.csv')
df = raw_data
df['date'] = pd.to_datetime(df['date'])

# Split data
from py_rsample import initial_split, training, testing
split = initial_split(df, prop=0.75, seed=123)
train_data = training(split)
test_data = testing(split)

# Rename outcome for auto-detection (THE FIX)
train_data = train_data.rename(columns={"refinery_kbd": "target"})
test_data = test_data.rename(columns={"refinery_kbd": "target"})

# Create recipe with normalization
rec_normalize = recipe().step_normalize(all_numeric_predictors())

# Workflow
wf_normalize = workflow().add_recipe(rec_normalize).add_model(linear_reg().set_engine("sklearn"))

# Fit
fit_normalize = wf_normalize.fit_global(train_data, group_col='country')
fit_normalize = fit_normalize.evaluate(test_data)

# Extract preprocessed data
processed_train = fit_normalize.extract_preprocessed_data(train_data)

# Check normalization
print("\n" + "="*70)
print("VERIFICATION: Column Normalization After Fix")
print("="*70)

# Check key columns
test_cols = ['brent', 'dubai', 'wti', 'target']
for col in test_cols:
    if col in processed_train.columns:
        mean = processed_train[col].mean()
        std = processed_train[col].std()
        print(f"{col:20s}: mean={mean:7.2f}, std={std:6.2f}", end="")

        if col == 'target':
            # target should NOT be normalized (it's the outcome)
            if abs(mean) > 0.1 or abs(std - 1.0) > 0.1:
                print("  ✓ Not normalized (outcome)")
            else:
                print("  ⚠ Unexpectedly normalized?")
        else:
            # Predictors should be normalized (mean≈0, std≈1)
            if abs(mean) < 0.01 and abs(std - 1.0) < 0.01:
                print("  ✓ NORMALIZED")
            else:
                print("  ✗ NOT NORMALIZED")

print("\n" + "="*70)
print("TEST RESULT")
print("="*70)

# Verify brent is normalized
brent_mean = processed_train['brent'].mean()
brent_std = processed_train['brent'].std()

if abs(brent_mean) < 0.01 and abs(brent_std - 1.0) < 0.01:
    print("✅ SUCCESS: 'brent' column is now NORMALIZED")
    print(f"   brent mean: {brent_mean:.6f} (expected: 0)")
    print(f"   brent std:  {brent_std:.6f} (expected: 1)")
else:
    print("❌ FAILED: 'brent' column is NOT normalized")
    print(f"   brent mean: {brent_mean:.2f}")
    print(f"   brent std:  {brent_std:.2f}")
    exit(1)
