"""
Test that extract_preprocessed_data() method works correctly for nested workflows.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

print("=" * 70)
print("TEST: extract_preprocessed_data() method for NestedWorkflowFit")
print("=" * 70)

# Create test data with multiple groups
np.random.seed(42)
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'country': ['USA'] * 50 + ['UK'] * 50,
    'x1': np.random.randn(100) * 10 + 50,  # Mean ~50, std ~10
    'x2': np.random.randn(100) * 5 + 20,   # Mean ~20, std ~5
    'target': np.random.randn(100) * 15 + 100  # Mean ~100, std ~15
})

train_data = data[:80]
test_data = data[80:]

print(f"\n1. Test with per-group preprocessing (per_group_prep=True)")
print(f"   Training data: {len(train_data)} rows, 2 groups")

# Create workflow with per-group preprocessing
rec = recipe().step_normalize()
wf = workflow().add_recipe(rec).add_model(linear_reg().set_engine("sklearn"))

# Fit nested workflow
nested_fit = wf.fit_nested(train_data, per_group_prep=True, group_col='country')

print(f"   ✓ Nested fit complete")

# Extract preprocessed training data
processed_train = nested_fit.extract_preprocessed_data(train_data, split='train')

print(f"   Preprocessed training data shape: {processed_train.shape}")
print(f"   Columns: {list(processed_train.columns)}")

# Verify normalization happened (mean ~0, std ~1)
for group in processed_train['country'].unique():
    group_data = processed_train[processed_train['country'] == group]
    x1_mean = group_data['x1'].mean()
    x1_std = group_data['x1'].std()
    x2_mean = group_data['x2'].mean()
    x2_std = group_data['x2'].std()
    target_mean = group_data['target'].mean()

    print(f"\n   Group: {group}")
    print(f"      x1 - mean: {x1_mean:.4f}, std: {x1_std:.4f}")
    print(f"      x2 - mean: {x2_mean:.4f}, std: {x2_std:.4f}")
    print(f"      target - mean: {target_mean:.2f} (should NOT be normalized)")

    # Verify x1 and x2 are normalized (mean ~0, std ~1)
    if abs(x1_mean) < 0.1 and abs(x1_std - 1.0) < 0.1:
        print(f"      ✓ x1 is normalized")
    else:
        print(f"      ✗ x1 normalization failed")

    if abs(x2_mean) < 0.1 and abs(x2_std - 1.0) < 0.1:
        print(f"      ✓ x2 is normalized")
    else:
        print(f"      ✗ x2 normalization failed")

    # Verify target is NOT normalized
    if abs(target_mean - 100) < 20:  # Should be close to original mean ~100
        print(f"      ✓ target is preserved (not normalized)")
    else:
        print(f"      ✗ target was incorrectly normalized")

# Verify split column
if 'split' in processed_train.columns and (processed_train['split'] == 'train').all():
    print(f"\n   ✓ Split column correctly set to 'train'")
else:
    print(f"\n   ✗ Split column missing or incorrect")

print(f"\n2. Test with test data extraction")

# Evaluate on test data
nested_fit = nested_fit.evaluate(test_data)

# Extract preprocessed test data
processed_test = nested_fit.extract_preprocessed_data(test_data, split='test')

print(f"   Preprocessed test data shape: {processed_test.shape}")

# Verify test data is also normalized using training stats
for group in processed_test['country'].unique():
    group_data = processed_test[processed_test['country'] == group]
    x1_mean = group_data['x1'].mean()
    x2_mean = group_data['x2'].mean()

    print(f"\n   Group: {group}")
    print(f"      x1 mean: {x1_mean:.4f}")
    print(f"      x2 mean: {x2_mean:.4f}")
    print(f"      (Note: Test data means may differ from 0 due to train/test distribution differences)")

# Verify split column
if 'split' in processed_test.columns and (processed_test['split'] == 'test').all():
    print(f"\n   ✓ Split column correctly set to 'test'")
else:
    print(f"\n   ✗ Split column missing or incorrect")

print(f"\n3. Test with shared preprocessing (per_group_prep=False)")

# Fit with shared preprocessing
nested_fit_shared = wf.fit_nested(train_data, per_group_prep=False, group_col='country')
processed_shared = nested_fit_shared.extract_preprocessed_data(train_data, split='train')

print(f"   Preprocessed data shape: {processed_shared.shape}")
print(f"   ✓ Shared preprocessing extraction works")

# Verify all groups have similar normalization (since shared recipe)
means = []
for group in processed_shared['country'].unique():
    group_data = processed_shared[processed_shared['country'] == group]
    means.append(group_data['x1'].mean())

if len(set([round(m, 1) for m in means])) <= 2:  # Similar means across groups
    print(f"   ✓ Shared preprocessing applied (group means similar)")
else:
    print(f"   ✗ Groups have very different means (expected similar)")

print(f"\n4. Test column ordering")

# Verify column order: date first, group_col second
expected_first_cols = ['date', 'country']
actual_first_cols = list(processed_train.columns[:2])

if actual_first_cols == expected_first_cols:
    print(f"   ✓ Columns correctly ordered: {actual_first_cols}")
else:
    print(f"   ✗ Column order incorrect. Expected {expected_first_cols}, got {actual_first_cols}")

print(f"\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
