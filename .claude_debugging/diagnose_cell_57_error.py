"""
CELL 57 ERROR DIAGNOSTIC SCRIPT
================================

This script diagnoses the missing columns error in Cell 57 by checking:
1. Whether the updated evaluate() code is loaded in kernel memory
2. Whether train/test data have matching columns
3. Which specific group (USA/UK) is causing the error
4. What the actual data looks like after preprocessing

USAGE: Copy this entire cell and run it IMMEDIATELY AFTER the error in Cell 57.
"""

import pandas as pd
import numpy as np
import inspect
from py_workflows.workflow import WorkflowFit

print("=" * 80)
print("CELL 57 ERROR DIAGNOSTICS")
print("=" * 80)

# ============================================================================
# CHECK 1: Is the updated code loaded in memory?
# ============================================================================
print("\n[CHECK 1] Verifying updated evaluate() code is loaded...")

eval_source = inspect.getsource(WorkflowFit.evaluate)

# Check for the specific fix
has_fix_1 = 'needs_outcome = self.workflow._recipe_requires_outcome' in eval_source
has_fix_2 = 'outcome_col = self.workflow._get_outcome_from_recipe' in eval_source

if has_fix_1 and has_fix_2:
    print(" PASS: Updated evaluate() code IS loaded")
    print("   - Found needs_outcome check")
    print("   - Found outcome_col from recipe")
else:
    print("L FAIL: Old evaluate() code still cached!")
    print(f"   - needs_outcome check: {'FOUND' if has_fix_1 else 'MISSING'}")
    print(f"   - outcome_col from recipe: {'FOUND' if has_fix_2 else 'MISSING'}")
    print("\n      ACTION REQUIRED - KERNEL STILL HAS OLD CODE:")
    print("   1. Close this notebook tab completely")
    print("   2. In Jupyter home, click 'Running' tab ’ Shut down this kernel")
    print("   3. Open terminal and run:")
    print("      cd '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels'")
    print("      find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null")
    print("      source py-tidymodels2/bin/activate")
    print("      pip install -e . --force-reinstall --no-deps")
    print("   4. Restart Jupyter, reopen notebook, re-run from Cell 1")

# ============================================================================
# CHECK 2: Do train_data and test_data have matching columns?
# ============================================================================
print("\n[CHECK 2] Checking column consistency between train and test data...")

# Exclude metadata columns
metadata_cols = {'date', 'country', 'refinery_kbd'}
train_features = set(train_data.columns) - metadata_cols
test_features = set(test_data.columns) - metadata_cols

missing_in_test = train_features - test_features
extra_in_test = test_features - train_features

print(f"   Train shape: {train_data.shape}")
print(f"   Test shape: {test_data.shape}")
print(f"   Train features: {len(train_features)} columns")
print(f"   Test features: {len(test_features)} columns")

if len(missing_in_test) == 0 and len(extra_in_test) == 0:
    print(" PASS: Train and test data have matching features")
else:
    print("L FAIL: Train and test data have different features!")
    if missing_in_test:
        print(f"\n   Missing in test (but in train): {len(missing_in_test)} columns")
        print(f"   First 10: {sorted(missing_in_test)[:10]}")
    if extra_in_test:
        print(f"\n   Extra in test (not in train): {len(extra_in_test)} columns")
        print(f"   First 10: {sorted(extra_in_test)[:10]}")

# ============================================================================
# CHECK 3: Per-group column analysis
# ============================================================================
print("\n[CHECK 3] Checking per-group data consistency...")

for group in sorted(train_data['country'].unique()):
    print(f"\n  Group: {group}")

    train_group = train_data[train_data['country'] == group]
    test_group = test_data[test_data['country'] == group]

    train_group_features = set(train_group.columns) - metadata_cols
    test_group_features = set(test_group.columns) - metadata_cols

    missing = train_group_features - test_group_features
    extra = test_group_features - train_group_features

    print(f"    Train rows: {len(train_group)}, Test rows: {len(test_group)}")
    print(f"    Train features: {len(train_group_features)}, Test features: {len(test_group_features)}")

    if missing:
        print(f"    L Missing in test: {sorted(missing)[:5]}")
    if extra:
        print(f"       Extra in test: {sorted(extra)[:5]}")
    if not missing and not extra:
        print(f"     Matching features")

# ============================================================================
# CHECK 4: Check for NaN values that might cause issues
# ============================================================================
print("\n[CHECK 4] Checking for NaN/Inf values...")

train_nan_cols = train_data.columns[train_data.isna().any()].tolist()
test_nan_cols = test_data.columns[test_data.isna().any()].tolist()

train_inf_cols = []
test_inf_cols = []
for col in train_data.select_dtypes(include=[np.number]).columns:
    if np.isinf(train_data[col]).any():
        train_inf_cols.append(col)
for col in test_data.select_dtypes(include=[np.number]).columns:
    if np.isinf(test_data[col]).any():
        test_inf_cols.append(col)

if train_nan_cols:
    print(f"     Train data has NaN in {len(train_nan_cols)} columns: {train_nan_cols[:5]}")
else:
    print(f"   No NaN in train data")

if test_nan_cols:
    print(f"     Test data has NaN in {len(test_nan_cols)} columns: {test_nan_cols[:5]}")
else:
    print(f"   No NaN in test data")

if train_inf_cols:
    print(f"     Train data has Inf in {len(train_inf_cols)} columns: {train_inf_cols[:5]}")
if test_inf_cols:
    print(f"     Test data has Inf in {len(test_inf_cols)} columns: {test_inf_cols[:5]}")

# ============================================================================
# CHECK 5: Specific error columns diagnosis
# ============================================================================
print("\n[CHECK 5] Diagnosing specific error columns...")

error_cols = ['bakken_coking_usmc', 'brent_cracking_nw_europe',
              'es_sider_cracking_med', 'x30_70_wcs_bakken_cracking_usmc']

print(f"\nColumns from error message: {error_cols}")

for col in error_cols:
    in_train = col in train_data.columns
    in_test = col in test_data.columns

    print(f"\n  '{col}':")
    print(f"    In train_data: {in_train}")
    print(f"    In test_data: {in_test}")

    if in_train:
        print(f"    Train NaN count: {train_data[col].isna().sum()}")
        print(f"    Train dtype: {train_data[col].dtype}")
        print(f"    Train sample: {train_data[col].head(3).tolist()}")

    if in_test:
        print(f"    Test NaN count: {test_data[col].isna().sum()}")
        print(f"    Test dtype: {test_data[col].dtype}")
        print(f"    Test sample: {test_data[col].head(3).tolist()}")

# ============================================================================
# CHECK 6: Recipe step analysis
# ============================================================================
print("\n[CHECK 6] Analyzing recipe steps...")

# This assumes fit_anova is the failed workflow
if 'fit_anova' in dir():
    print("\n  Analyzing fit_anova workflow...")

    # Check if per_group_prep was used
    if hasattr(fit_anova, 'group_preps'):
        print(f"   Per-group preprocessing IS enabled")
        print(f"  Number of groups: {len(fit_anova.group_preps)}")

        for group_name, prep in fit_anova.group_preps.items():
            print(f"\n    Group: {group_name}")
            print(f"    Recipe steps: {len(prep.steps)}")

            # Show which columns each group's recipe selected
            if hasattr(prep, 'retained_predictors'):
                print(f"    Retained predictors: {len(prep.retained_predictors)} columns")
                print(f"    First 10: {prep.retained_predictors[:10]}")
    else:
        print(f"     Per-group preprocessing NOT enabled")
else:
    print("     fit_anova not found in workspace - run Cell 57 first")

# ============================================================================
# CHECK 7: Test if we can manually bake test data
# ============================================================================
print("\n[CHECK 7] Testing manual bake of test data...")

if 'fit_anova' in dir() and hasattr(fit_anova, 'group_preps'):
    try:
        # Try baking test data for USA group
        usa_test = test_data[test_data['country'] == 'USA'].copy()
        usa_prep = fit_anova.group_preps['USA']

        print(f"  Attempting to bake USA test data ({len(usa_test)} rows)...")
        usa_baked = usa_prep.bake(usa_test)
        print(f"   SUCCESS: Baked to shape {usa_baked.shape}")
        print(f"  Columns after bake: {usa_baked.columns.tolist()[:10]}")

    except Exception as e:
        print(f"  L FAILED to bake test data: {type(e).__name__}: {e}")
        print(f"  This indicates the error is in the bake() step")
else:
    print("     Cannot test baking - fit_anova not available")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

issues_found = []

if not (has_fix_1 and has_fix_2):
    issues_found.append("L OLD CODE CACHED - Must restart Jupyter kernel properly")

if len(missing_in_test) > 0:
    issues_found.append(f"L DATA QUALITY - Test missing {len(missing_in_test)} columns from train")

if len(train_nan_cols) > 0 or len(test_nan_cols) > 0:
    issues_found.append("   NaN VALUES - May cause issues with some steps")

if len(issues_found) == 0:
    print("\n All checks passed!")
    print("\nPossible reasons for error:")
    print("  1. The error was transient and may not recur")
    print("  2. The error occurs during group-specific processing")
    print("  3. Check the output of CHECK 7 above for baking issues")
else:
    print("\nIssues detected:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")

    print("\nRECOMMENDED ACTIONS:")
    if not (has_fix_1 and has_fix_2):
        print("\n  Priority 1: RESTART KERNEL")
        print("  - The updated code is NOT loaded in memory")
        print("  - Follow the restart instructions in CHECK 1 above")

    if len(missing_in_test) > 0:
        print("\n  Priority 2: INVESTIGATE DATA")
        print(f"  - Test data is missing {len(missing_in_test)} columns that train has")
        print("  - Check if test data was processed differently than train")
        print("  - Verify data loading code in earlier cells")

print("\n" + "=" * 80)
print("END OF DIAGNOSTICS")
print("=" * 80)
