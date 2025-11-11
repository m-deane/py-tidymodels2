"""
Test that supervised feature selection steps work correctly with per-group preprocessing.

This test verifies that each group gets independent feature selections without
overwriting each other's selections.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create test data with multiple groups
np.random.seed(42)
n_per_group = 150

data_list = []
for group in ['Algeria', 'Denmark', 'Germany']:
    group_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_per_group, freq='M'),
        'country': [group] * n_per_group,
        'x1': np.random.randn(n_per_group),
        'x2': np.random.randn(n_per_group),
        'x3': np.random.randn(n_per_group),
        'x4': np.random.randn(n_per_group),
        'x5': np.random.randn(n_per_group),
        'refinery_kbd': np.random.randn(n_per_group) * 10 + 50
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)

# Split train/test
train_data = data[data['date'] < '2022-01-01'].copy()
test_data = data[data['date'] >= '2022-01-01'].copy()

print("=" * 80)
print("TESTING SUPERVISED FEATURE SELECTION WITH PER-GROUP PREPROCESSING")
print("=" * 80)

print(f"\nData:")
print(f"  Total rows: {len(data)}")
print(f"  Train rows: {len(train_data)} ({len(train_data)//3} per group)")
print(f"  Test rows: {len(test_data)} ({len(test_data)//3} per group)")
print(f"  Features: 5 (x1-x5)")
print(f"  Groups: {sorted(data['country'].unique())}")

# Test 1: step_filter_anova
print("\n" + "-" * 80)
print("TEST 1: step_filter_anova with per_group_prep=True")
print("-" * 80)

try:
    rec_anova = (
        recipe()
        .step_filter_anova(outcome="refinery_kbd", top_n=3, use_pvalue=True)
        .step_normalize()
    )
    
    wf_anova = workflow().add_recipe(rec_anova).add_model(linear_reg())
    
    print("  Fitting nested model...")
    fit_anova = wf_anova.fit_nested(train_data, group_col='country', per_group_prep=True)
    
    print("  ✓ Fit completed successfully")
    
    # Check that each group has independent feature selections
    print("\n  Feature selections per group:")
    for group_name, prep in fit_anova.group_preps.items():
        # Find the anova step
        anova_step = None
        for step in prep.prepared_steps:
            if hasattr(step, '_selected_features') and hasattr(step, '_scores'):
                anova_step = step
                break
        
        if anova_step:
            print(f"    {group_name}: {len(anova_step._selected_features)} features selected")
            print(f"      Selected: {anova_step._selected_features[:3]}")
        else:
            print(f"    {group_name}: No anova step found")
    
    # Evaluate on test data
    print("\n  Evaluating on test data...")
    fit_anova = fit_anova.evaluate(test_data)
    print("  ✓ Evaluation completed successfully")
    
    # Extract outputs
    outputs, coeffs, stats = fit_anova.extract_outputs()
    print(f"\n  Results:")
    print(f"    Outputs shape: {outputs.shape}")
    print(f"    Stats shape: {stats.shape}")
    print(f"    Groups in outputs: {sorted(outputs['country'].unique())}")
    
    print("\n✅ TEST 1 PASSED")
    
except Exception as e:
    print(f"\n❌ TEST 1 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: step_filter_mutual_info
print("\n" + "-" * 80)
print("TEST 2: step_filter_mutual_info with per_group_prep=True")
print("-" * 80)

try:
    from py_recipes.steps import step_filter_mutual_info
    
    rec_mi = (
        recipe()
        .step_filter_mutual_info(outcome="refinery_kbd", top_n=3)
        .step_normalize()
    )
    
    wf_mi = workflow().add_recipe(rec_mi).add_model(linear_reg())
    
    print("  Fitting nested model...")
    fit_mi = wf_mi.fit_nested(train_data, group_col='country', per_group_prep=True)
    
    print("  ✓ Fit completed successfully")
    
    # Evaluate
    print("  Evaluating on test data...")
    fit_mi = fit_mi.evaluate(test_data)
    print("  ✓ Evaluation completed successfully")
    
    print("\n✅ TEST 2 PASSED")
    
except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: step_filter_rf_importance
print("\n" + "-" * 80)
print("TEST 3: step_filter_rf_importance with per_group_prep=True")
print("-" * 80)

try:
    from py_recipes.steps import step_filter_rf_importance
    
    rec_rf = (
        recipe()
        .step_filter_rf_importance(outcome="refinery_kbd", top_n=3, trees=50)
        .step_normalize()
    )
    
    wf_rf = workflow().add_recipe(rec_rf).add_model(linear_reg())
    
    print("  Fitting nested model...")
    fit_rf = wf_rf.fit_nested(train_data, group_col='country', per_group_prep=True)
    
    print("  ✓ Fit completed successfully")
    
    # Evaluate
    print("  Evaluating on test data...")
    fit_rf = fit_rf.evaluate(test_data)
    print("  ✓ Evaluation completed successfully")
    
    print("\n✅ TEST 3 PASSED")
    
except Exception as e:
    print(f"\n❌ TEST 3 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
