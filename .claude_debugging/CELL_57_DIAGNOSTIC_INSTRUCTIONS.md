# Cell 57 Error Diagnostic Instructions

## Current Situation

You're experiencing a `ValueError` in Cell 57 of `_md/forecasting_recipes_grouped.ipynb` with supervised feature selection (step_filter_anova):

```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- bakken_coking_usmc
- brent_cracking_nw_europe
- es_sider_cracking_med
- x30_70_wcs_bakken_cracking_usmc
```

## Key Observation

This error is **DIFFERENT** from the previous error:
- **Previous error**: `refinery_kbd` (outcome column) was missing í **FIXED** 
- **Current error**: Feature columns are missing

This suggests:
1. **My fix worked** - the outcome column is no longer in the error message
2. **New issue**: Either old code is still cached OR there's a data quality problem

## What I've Done

### Fix Applied to `py_workflows/workflow.py`

Modified `WorkflowFit.evaluate()` method (lines 922-943) to handle supervised feature selection:

```python
# Get outcome from recipe instead of auto-detecting
outcome_col = self.workflow._get_outcome_from_recipe(self.pre)

# Check if recipe needs outcome during baking
needs_outcome = self.workflow._recipe_requires_outcome(self.pre)

if needs_outcome:
    # Supervised steps: bake WITH outcome
    processed_test_data = self.pre.bake(test_data)
else:
    # Regular steps: separate outcome, bake, recombine
    ...
```

### Testing

- Created `test_supervised_evaluate_fix.py` - **All 3 tests passing** 
- All 90 existing workflow tests still pass 

## Your Next Steps

### Step 1: Run Diagnostic Script

**Copy and paste** the entire contents of `.claude_debugging/diagnose_cell_57_error.py` into a NEW cell in your notebook IMMEDIATELY AFTER Cell 57 (the failing cell).

This script will check:
1.  Is the updated code loaded in your kernel?
2.  Do train/test data have matching columns?
3.  Which group (USA/UK) is causing the issue?
4.  Are there NaN/Inf values causing problems?
5.  What do the specific error columns look like?
6.  Can we manually bake the test data?

### Step 2: Interpret Results

The diagnostic will show clear **PASS/FAIL** messages:

#### If CHECK 1 FAILS (old code cached):
```
L FAIL: Old evaluate() code still cached!
```

**Action**: You MUST restart the kernel properly:
1. Close the notebook tab completely
2. In Jupyter home í "Running" tab í Shut down this notebook's kernel
3. Open terminal and run:
   ```bash
   cd '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels'
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   source py-tidymodels2/bin/activate
   pip install -e . --force-reinstall --no-deps
   ```
4. Restart Jupyter, reopen notebook, re-run from Cell 1

#### If CHECK 2 FAILS (data mismatch):
```
L FAIL: Train and test data have different features!
Missing in test: ['col1', 'col2', ...]
```

**Action**: Investigate why test data is missing columns:
- Check data loading cells (earlier in notebook)
- Verify train/test split didn't drop columns
- Check if test data was processed differently

#### If All Checks PASS:
```
 All checks passed!
```

**Action**: The error may be intermittent or already resolved - try re-running Cell 57.

### Step 3: Report Back

After running the diagnostic, **copy the entire output** and share it. This will tell us:
1. Whether the fix is loaded
2. Whether it's a data issue
3. What the root cause is

## Why This Error is Different

### Error Evolution Timeline

1. **Original error** (before my fix):
   ```
   Feature names missing: - refinery_kbd  ê OUTCOME COLUMN
   ```
   **Cause**: `evaluate()` was removing outcome before baking, but supervised steps expected it

2. **Current error** (after my fix):
   ```
   Feature names missing:
   - bakken_coking_usmc      ê FEATURE COLUMN
   - brent_cracking_nw_europe ê FEATURE COLUMN
   ```
   **Cause**: Either:
   - Old code still cached (fix not loaded)
   - OR test data genuinely missing columns

The fact that **refinery_kbd** (outcome) is no longer in the error is **strong evidence my fix worked**. But we need the diagnostic to confirm the fix is actually loaded in your kernel.

## Files Reference

### Code Modified
- `py_workflows/workflow.py:922-943` - evaluate() method fix

### Documentation
- `SUPERVISED_EVALUATE_FIX_2025_11_10.md` - Complete fix documentation
- `test_supervised_evaluate_fix.py` - Test verification (all passing)

### Diagnostic Tools
- `diagnose_cell_57_error.py` - **USE THIS FIRST**
- `COMPLETE_RESTART_PROCEDURE.md` - Full restart instructions

## Expected Outcome

Once properly restarted with the fix loaded:
- Cell 57 should complete without errors
- Test data evaluation will work with supervised feature selection
- All 13 previously failing cells should work
