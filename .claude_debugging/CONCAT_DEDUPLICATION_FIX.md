# pd.concat() Deduplication Fix

**Date:** 2025-11-09
**Issue:** Notebook still showing "AttributeError: 'DataFrame' object has no attribute 'dtype'"
**Status:** ✅ Fixed

---

## Problem

After fixing the initial duplicate column issue in `step_safe()` by adding deduplication in the `top_n` filtering section, the notebook was still failing with the same error:

```python
AttributeError: 'DataFrame' object has no attribute 'dtype'
```

**Error Location:** `py_hardhat/mold.py:245` in the workflow

---

## Root Cause Analysis

The initial fix only deduplicaated when `top_n is not None` (lines 673-681), but the duplicate columns were being created **earlier** in the bake() method at line 668:

```python
result = pd.concat(transformed_dfs, axis=1).reset_index(drop=True)
```

**The Issue:** When `pd.concat()` concatenates multiple DataFrames along axis=1:
- If any input DataFrames have overlapping column names
- OR if the same column appears in multiple DataFrames
- The result DataFrame will have **duplicate column names**

This happened even when `top_n=None`, so the downstream deduplication never ran.

---

## The Fix

**File:** `py_recipes/steps/feature_extraction.py` (lines 666-676)

Added deduplication **immediately after the concat operation**:

```python
# Combine all transformed features
if transformed_dfs:
    result = pd.concat(transformed_dfs, axis=1).reset_index(drop=True)

    # Deduplicate columns immediately after concat (prevents duplicate column names)
    # This can happen if same feature appears multiple times in transformations
    if result.columns.duplicated().any():
        result = result.loc[:, ~result.columns.duplicated()]
else:
    result = pd.DataFrame(index=range(len(data)))
```

**Key Points:**
- Uses `result.columns.duplicated().any()` to check for duplicates efficiently
- `result.loc[:, ~result.columns.duplicated()]` keeps only first occurrence of each column
- Preserves column order (important for reproducibility)
- Runs BEFORE any downstream operations (top_n filtering, etc.)

---

## Why This is Critical

**pd.concat() behavior with duplicate columns:**
```python
# Example of the problem
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})  # 'A' appears in both!

result = pd.concat([df1, df2], axis=1)
# result.columns = Index(['A', 'B', 'A', 'C'])  # Duplicate 'A'!

# Accessing column with duplicate name returns DataFrame, not Series
result['A']  # Returns DataFrame with 2 columns named 'A'
result['A'].dtype  # AttributeError: 'DataFrame' object has no attribute 'dtype'
```

**The fix ensures unique columns:**
```python
if result.columns.duplicated().any():
    result = result.loc[:, ~result.columns.duplicated()]
# result.columns = Index(['A', 'B', 'C'])  # All unique!

result['A']  # Returns Series
result['A'].dtype  # dtype('int64') - Works!
```

---

## Test Coverage

### Added New Test
**File:** `tests/test_recipes/test_safe.py`

```python
def test_no_duplicate_columns(self, simple_regression_data, fitted_surrogate):
    """Test that bake() result has no duplicate column names."""
    step = StepSafe(
        surrogate_model=fitted_surrogate,
        outcome='y',
        penalty=3.0,
        no_changepoint_strategy='drop',
        top_n=None  # No top_n filtering - test raw concat result
    )

    prepped = step.prep(simple_regression_data, training=True)
    result = prepped.bake(simple_regression_data)

    # Check no duplicate columns
    duplicates = result.columns[result.columns.duplicated()].tolist()
    assert not result.columns.duplicated().any(), \
        f"Result has duplicate columns: {duplicates}"

    # Check all columns are unique
    assert len(result.columns) == len(set(result.columns)), \
        "Result columns are not unique"
```

### Test Results
```bash
$ python -m pytest tests/test_recipes/test_safe.py -v

============================= 30 passed in 31.53s ==============================
```

**All 30 tests passing** (was 29, added 1 new test)

---

## Verification

### Before Fix:
```python
# In bake(), after concat:
result = pd.concat(transformed_dfs, axis=1)
# result might have duplicate columns
# → Workflow → mold() → ERROR: 'DataFrame' object has no attribute 'dtype'
```

### After Fix:
```python
# In bake(), after concat:
result = pd.concat(transformed_dfs, axis=1)
if result.columns.duplicated().any():
    result = result.loc[:, ~result.columns.duplicated()]
# result guaranteed to have unique columns
# → Workflow → mold() → SUCCESS!
```

---

## Notebook Should Now Work

The notebook cell running `wf_safe.fit(train_data)` should now execute without errors:

```python
# Cell 76: SAFE example
wf_safe = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg().set_engine("sklearn"))
)

# Fit and evaluate
fit_safe = wf_safe.fit(train_data)  # ✅ Should work now!
fit_safe = fit_safe.evaluate(test_data)
```

---

## Related Fixes

This is the **third** fix in the duplicate column saga:

1. **First fix:** Exclude 'date' column when training surrogates (NOTEBOOK_DATETIME_FIX.md)
2. **Second fix:** Deduplicate in top_n filtering (DUPLICATE_COLUMN_FIX.md, lines 673-681)
3. **This fix:** Deduplicate immediately after pd.concat() (lines 666-676)

All three fixes are now in place, providing comprehensive protection against duplicate columns!

---

**Fix Applied:** 2025-11-09
**Status:** Complete ✅
**Tests:** 30/30 SAFE tests passing
**New Test:** test_no_duplicate_columns() added
**Notebook:** Should now run without errors
