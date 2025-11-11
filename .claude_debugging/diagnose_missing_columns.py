"""
Diagnostic script to identify why test data is missing columns.

Run this in your Jupyter notebook to understand the data flow.
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("DIAGNOSTIC: Missing Columns in Test Data")
print("=" * 70)

# Instructions for user
print("""
INSTRUCTIONS:
Copy and paste this code into a NEW cell in your notebook BEFORE Cell 57.

Then modify it to use your actual train_data and test_data variables.
""")

print("""
# === PASTE THIS INTO YOUR NOTEBOOK ===

import pandas as pd

print("\\n1. Checking train_data columns...")
print(f"Train data shape: {train_data.shape}")
print(f"Train columns: {len(train_data.columns)} columns")
print(f"Columns: {sorted(train_data.columns)}")

print("\\n2. Checking test_data columns...")
print(f"Test data shape: {test_data.shape}")
print(f"Test columns: {len(test_data.columns)} columns
print(f"Columns: {sorted(test_data.columns)}")

print("\\n3. Comparing train vs test columns...")
train_cols = set(train_data.columns)
test_cols = set(test_data.columns)

missing_in_test = train_cols - test_cols
missing_in_train = test_cols - train_cols

if missing_in_test:
    print(f"\\n⚠️  ISSUE: Test data is MISSING these columns:")
    for col in sorted(missing_in_test):
        print(f"   - {col}")
else:
    print("\\n✓ Test data has all training columns")

if missing_in_train:
    print(f"\\n⚠️  Test data has EXTRA columns not in training:")
    for col in sorted(missing_in_train):
        print(f"   - {col}")

print("\\n4. Checking per-group data...")
for country in ['USA', 'UK']:
    train_group = train_data[train_data['country'] == country]
    test_group = test_data[test_data['country'] == country]
    print(f"\\n{country}:")
    print(f"  Train: {train_group.shape}")
    print(f"  Test: {test_group.shape}")

    # Check for NaN
    train_nan = train_group.isna().sum().sum()
    test_nan = test_group.isna().sum().sum()
    print(f"  Train NaN count: {train_nan}")
    print(f"  Test NaN count: {test_nan}")

# === END PASTE ===
""")

print("\n" + "=" * 70)
print("After running the above, look for:")
print("1. Are train and test columns identical?")
print("2. Does test data have NaN values in some columns?")
print("3. Are there group-specific differences?")
print("=" * 70)
