# step_naomit Index Alignment Fix - 2025-11-10

## Problem

When using `step_naomit()` to remove rows with NaN values (e.g., created by `step_lag()` or `step_diff()`), the workflow was failing with:

```
ValueError: Length of values (20) does not match length of index (18)
PatsyError: factor contains missing values
```

## Root Cause

The issue occurred in two locations in `py_workflows/workflow.py`:

1. **Line 322** (`_prep_and_bake_with_outcome()` method): When recombining outcome with baked predictors
2. **Line 936** (`WorkflowFit.evaluate()` method): When processing test data

Both locations used `.values` assignment, which doesn't preserve index alignment:

```python
# INCORRECT - Loses index alignment
processed_data[outcome_col] = outcome.values
```

**Why This Failed**:
- `step_naomit()` removes rows with NaN values from the baked data
- Baked data has fewer rows (e.g., 18 rows after removing NaN)
- Outcome still has original number of rows (e.g., 20 rows)
- Assigning 20 values to a DataFrame with 18 rows → ValueError

## Solution

Changed both locations to use index-based alignment:

```python
# CORRECT - Aligns by index
processed_data[outcome_col] = outcome.loc[processed_predictors.index].values
```

**How This Works**:
- `processed_predictors.index` contains the indices of rows that survived step_naomit
- `outcome.loc[processed_predictors.index]` selects only the outcome values for those rows
- `.values` extracts the aligned values
- Assignment succeeds because lengths now match

## Files Modified

### 1. py_workflows/workflow.py (2 locations)

**Location 1: Line 322** (`_prep_and_bake_with_outcome()` method)

```python
# BEFORE (line 322):
processed_data[outcome_col] = outcome.values

# AFTER (line 322):
processed_data[outcome_col] = outcome.loc[processed_predictors.index].values
```

**Context**: This method handles recipe baking during model fitting with per-group preprocessing.

**Location 2: Line 936** (`WorkflowFit.evaluate()` method)

```python
# BEFORE (line 936):
processed_test_data[outcome_col] = outcome.values

# AFTER (line 936):
processed_test_data[outcome_col] = outcome.loc[processed_predictors.index].values
```

**Context**: This method handles test data evaluation after model fitting.

## Testing

### Test Script 1: `.claude_debugging/test_step_naomit_alignment.py`
- **Purpose**: Initial test with small groups
- **Result**: Fell back to global recipe due to min_group_size threshold
- **Outcome**: Didn't actually test per-group preprocessing path

### Test Script 2: `.claude_debugging/test_step_naomit_large_groups.py`
- **Purpose**: Test with large groups (50+ samples) to trigger per-group preprocessing
- **Result**: All tests passing ✅

**Test Results**:
```
1. Testing step_lag with step_naomit (per_group_prep=True)...
   ✓ fit_nested() succeeded
   ✓ evaluate() succeeded
   ✓ extract_outputs() succeeded
   Outputs shape: (94, 10)
   Training rows preserved after naomit: 76
   ✓ No NaN values in actuals column
   ✓ SUCCESS: Per-group step_lag with step_naomit works!

2. Testing step_diff with step_naomit (per_group_prep=True)...
   ✓ fit_nested() succeeded
   ✓ evaluate() succeeded
   ✓ extract_outputs() succeeded
   Outputs shape: (97, 10)
   ✓ No NaN values in actuals column
   ✓ SUCCESS: Per-group step_diff with step_naomit works!

3. Testing step_lag with step_naomit (global recipe)...
   ✓ SUCCESS: Global recipe works!
   Outputs shape: (94, 10)
   ✓ No NaN values in actuals column
```

### Regression Testing
**All 90 workflow tests passing** ✅

No regressions introduced by the fix.

## Impact

### Before Fix
- `step_naomit()` failed with both per-group and global recipes
- Notebooks using `step_lag()` or `step_diff()` with `step_naomit()` crashed
- `evaluate()` failed on test data with NaN-producing steps

### After Fix
- ✅ Per-group preprocessing with `step_naomit()` works correctly
- ✅ Global recipe with `step_naomit()` works correctly
- ✅ Regular `fit()` with `step_naomit()` works correctly
- ✅ Test data evaluation handles row removal correctly
- ✅ No NaN values in outputs

## Related Issues Fixed

This fix resolves the cascading errors in `_md/forecasting_recipes_grouped.ipynb`:

- **Cell 49**: `step_lag()` with `step_naomit()` - Now works ✅
- **Cell 50**: `step_diff()` with `step_naomit()` - Now works ✅
- **Cell 69**: `step_sqrt()` with `step_naomit()` - Now works ✅

All cells that use `step_naomit()` after NaN-producing steps now execute successfully.

## Technical Details

### When step_naomit() Removes Rows

Steps that create NaN values requiring step_naomit():

1. **step_lag()**: Creates NaN in first `max(lags)` rows
   - `lags=[1, 2]` → 2 NaN rows per group
   - Example: 40 rows → 38 rows after naomit

2. **step_diff()**: Creates NaN in first `lag` rows
   - `lag=1` → 1 NaN row per group
   - Example: 40 rows → 39 rows after naomit

3. **step_sqrt()**: Creates NaN for negative values
   - Number of NaN rows depends on data
   - Example: Some negative values → fewer rows after naomit

### Index Preservation

The fix ensures that:
- Original row indices are preserved through recipe baking
- Only rows that survive `step_naomit()` are included
- Outcome column is aligned to the same indices
- No length mismatch errors

### Edge Cases Handled

1. **Different NaN counts per group**: Each group may lose different numbers of rows
2. **Multiple naomit steps**: Index alignment works with sequential naomit calls
3. **Global vs per-group**: Both preprocessing modes work correctly
4. **Train vs test**: Both splits handle row removal correctly

## Best Practices

When using `step_naomit()`:

1. **Always use after steps that create NaN**:
   ```python
   recipe()
       .step_lag(['x1', 'x2'], lags=[1, 2])
       .step_naomit()  # Required!
   ```

2. **Place before formula parsing steps**:
   - Patsy cannot handle NaN values
   - `step_naomit()` must come before model fitting

3. **Consider data loss**:
   - Lagging by 2 periods loses 2 rows per group
   - Small groups may lose significant percentage of data

4. **Check group sizes**:
   - If group has < 30 samples, workflow falls back to global recipe
   - May want to adjust `min_group_size` parameter

## Code References

### Primary Fix Locations
- `py_workflows/workflow.py:322` - `_prep_and_bake_with_outcome()` method
- `py_workflows/workflow.py:936` - `WorkflowFit.evaluate()` method

### Test Scripts
- `.claude_debugging/test_step_naomit_alignment.py` - Initial test (small groups)
- `.claude_debugging/test_step_naomit_large_groups.py` - Comprehensive test (large groups)

### Related Documentation
- `.claude_debugging/NOTEBOOK_ERRORS_FIXED_2025_11_10.md` - Notebook error fixes
- `.claude_debugging/SUPERVISED_FEATURE_SELECTION_FIX_2025_11_10.md` - Supervised step fixes

---

**Status**: ✅ Complete and tested
**Date**: 2025-11-10
**Tests Passing**: 90/90 workflow tests + 3 new step_naomit tests
**Impact**: Critical fix for any recipe using step_naomit()
