"""
Simple test that supervised feature selection works with per-group preprocessing.
Tests the exact scenario from the user's notebook.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create test data matching the user's scenario
np.random.seed(42)
n_per_group = 200  # Enough to pass min_group_size

data_list = []
for group in ['Algeria', 'Denmark']:
    group_data = pd.DataFrame({
        'date': pd.date_range('2018-01-01', periods=n_per_group, freq='M'),
        'country': [group] * n_per_group,
        'x1': np.random.randn(n_per_group),
        'x2': np.random.randn(n_per_group),
        'x3': np.random.randn(n_per_group),
        'x4': np.random.randn(n_per_group),
        'x5': np.random.randn(n_per_group),
        'x6': np.random.randn(n_per_group),
        'refinery_kbd': np.random.randn(n_per_group) * 10 + 50
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)

# Split train/test
train_data = data[data['date'] < '2022-01-01'].copy()
test_data = data[data['date'] >= '2022-01-01'].copy()

print("=" * 80)
print("SUPERVISED FEATURE SELECTION - PER-GROUP PREPROCESSING FIX TEST")
print("=" * 80)

print(f"\nData:")
print(f"  Train rows: {len(train_data)} ({len(train_data)//2} per group)")
print(f"  Test rows: {len(test_data)} ({len(test_data)//2} per group)")
print(f"  Features: 6 (x1-x6)")
print(f"  Groups: {sorted(data['country'].unique())}")

# Test: step_filter_anova with per-group preprocessing
print("\n" + "-" * 80)
print("TEST: step_filter_anova + step_normalize with per_group_prep=True")
print("-" * 80)

try:
    rec_anova = (
        recipe()
        .step_filter_anova(outcome="refinery_kbd", top_p=0.5, use_pvalue=True)
        .step_normalize()
    )
    
    wf_anova = workflow().add_recipe(rec_anova).add_model(linear_reg())
    
    print("\n  1. Fitting nested model...")
    fit_anova = wf_anova.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("  ✓ Fit completed")
    
    print("\n  2. Evaluating on test data...")
    fit_anova = fit_anova.evaluate(test_data)
    print("  ✓ Evaluation completed")
    
    print("\n  3. Extracting outputs...")
    outputs, coeffs, stats = fit_anova.extract_outputs()
    print(f"  ✓ Outputs shape: {outputs.shape}")
    print(f"  ✓ Stats shape: {stats.shape}")
    
    # Verify results
    print("\n  4. Verifying results...")
    test_outputs = outputs[outputs['split'] == 'test']
    print(f"  Test predictions: {len(test_outputs)} rows")
    print(f"  Groups: {sorted(test_outputs['country'].unique())}")
    
    # Check for errors (NaN predictions, etc.)
    nan_preds = test_outputs['fitted'].isna().sum()
    if nan_preds > 0:
        print(f"  ❌ WARNING: {nan_preds} NaN predictions")
    else:
        print(f"  ✓ No NaN predictions")
    
    print("\n✅ TEST PASSED - Per-group preprocessing with supervised feature selection works!")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {type(e).__name__}")
    print(f"  {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETED")
print("=" * 80)
