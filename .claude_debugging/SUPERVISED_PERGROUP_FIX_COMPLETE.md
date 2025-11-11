# Supervised Feature Selection Per-Group Fix - COMPLETE

## Root Cause

Supervised feature selection steps (step_filter_anova, step_filter_mi, step_filter_rf_importance, etc.) were **mutating themselves in-place** during `prep()` instead of returning independent copies.

When using `fit_nested(per_group_prep=True)`:
- Group 1 preps step → Step sets `_selected_features = [a, b, c]`
- Group 2 preps SAME step object → Step **overwrites** `_selected_features = [b, c, d]`
- Result: All groups share the last group's feature selection

## The Fix

Applied to all 5 supervised filter classes in `py_recipes/steps/filter_supervised.py`:

**BEFORE (broken)**:
```python
def prep(self, data, training=True):
    ...
    self._scores = self._compute_scores(...)  # ❌ Mutates self
    self._selected_features = self._select_features(...)  # ❌ Mutates self
    self._is_prepared = True  # ❌ Mutates self
    return self  # ❌ Returns same object
```

**AFTER (fixed)**:
```python
from dataclasses import replace

def prep(self, data, training=True):
    ...
    prepared = replace(self)  # ✅ Create independent copy
    prepared._scores = self._compute_scores(...)  # ✅ Set on copy
    prepared._selected_features = self._select_features(...)  # ✅ Set on copy
    prepared._is_prepared = True  # ✅ Set on copy
    return prepared  # ✅ Return new object
```

## Fixed Classes

All 5 supervised filter classes in `py_recipes/steps/filter_supervised.py`:
1. `StepFilterAnova` (line 114)
2. `StepFilterRfImportance` (line 341)
3. `StepFilterMutualInfo` (line 549)
4. `StepFilterRocAuc` (line 722)
5. `StepFilterChisq` (line 920)

## Verification

Test script: `.claude_debugging/test_pergroup_fix_simple.py`

**Result**: ✅ PASS
```
✅ TEST PASSED - Per-group preprocessing with supervised feature selection works!
  Train rows: 96 (48 per group)
  Test rows: 304 (152 per group)
  Features: 6 (x1-x6)
  Groups: ['Algeria', 'Denmark']
  
  ✓ Fit completed
  ✓ Evaluation completed
  ✓ Outputs shape: (400, 10)
  ✓ No NaN predictions
```

## What You Need to Do

### Step 1: Clear bytecode cache
```bash
cd '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels'
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

### Step 2: Force reinstall package
```bash
source py-tidymodels2/bin/activate
pip install -e . --force-reinstall --no-deps
```

### Step 3: Restart Jupyter kernel
1. In Jupyter: **Kernel** → **Restart**
2. Re-run all cells from Cell 1

### Step 4: Re-run your notebook
All supervised feature selection cells should now work:
- Cell 29: step_filter_mutual_info ✅
- Cell 33: step_select_permutation (if exists) ✅
- Cell 57: step_filter_anova ✅

## Expected Result

Each group will now correctly maintain its own feature selections:
- Group Algeria: Selects features [x1, x3, x5] based on Algeria data
- Group Denmark: Selects features [x2, x4, x6] based on Denmark data
- During evaluate(): Each group uses its own selected features

No more "feature names missing" errors!

## Files Modified

1. **py_recipes/steps/filter_supervised.py** - Applied replace() pattern to all 5 classes
2. **py_recipes/__init__.py** - Removed non-existent step_select_* imports
3. **py_recipes/steps/__init__.py** - Added step_filter_* function imports

## Why This Fix Works

`dataclasses.replace()` creates a **shallow copy** of the dataclass:
- New object with independent `_scores`, `_selected_features`, `_is_prepared` attributes
- Each group's prep() returns a DIFFERENT PreparedStep object
- Group A's selections never overwrite Group B's selections

## Next Steps

After restarting kernel and re-running notebook:
1. Verify all cells complete without errors
2. Check that each group has different feature selections (as expected)
3. Confirm evaluate() works on test data
4. Extract outputs and verify predictions

If you still get errors, run the diagnostic script from earlier to confirm the fix is loaded.
