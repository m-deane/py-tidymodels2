# Duplicate Column Names Fix

**Date:** 2025-11-09 (Updated with additional fix)
**Issue:** AttributeError when using step_safe() and step_eix() in workflows
**Status:** ✅ Fixed (with comprehensive solution)

---

## Problem

When running step_safe() and step_eix() in the forecasting recipes notebook, encountered this error:

```python
AttributeError: 'DataFrame' object has no attribute 'dtype'
```

**Traceback:**
```python
File ~/py_hardhat/mold.py, line 245, in mold
    ptypes = {col: str(data[col].dtype) for col in data.columns}
```

**Root Cause:**
This error occurs when a DataFrame has duplicate column names. When you access `df[col_name]` and there are multiple columns with that name, pandas returns a DataFrame instead of a Series (which has no `.dtype` attribute).

The issue was in both `StepSafe.bake()` and `StepEIX.bake()` methods:
- If `_selected_features` contained duplicates, the same column would be added multiple times
- **Additional issue**: `pd.concat()` can create duplicate columns if input DataFrames have overlapping column names
- This created a DataFrame with duplicate column names
- When the workflow passed this to `mold()`, it failed

---

## Solution

### Fix 1: StepSafe.bake() - Deduplicate after concat (CRITICAL FIX)

**File:** `py_recipes/steps/feature_extraction.py` (lines 666-676)

**Added deduplication immediately after pd.concat():**
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

**Why this is critical:** The `pd.concat(transformed_dfs, axis=1)` operation can create duplicate columns if any of the input DataFrames share column names. This must be caught immediately after concat, not just during top_n filtering.

### Fix 2: StepSafe.bake() - Deduplicate selected features

**File:** `py_recipes/steps/feature_extraction.py` (lines 678-686)

**Changed from:**
```python
# Filter to selected features if top_n specified
if self.top_n is not None and self._selected_features:
    available_features = [f for f in self._selected_features if f in result.columns]
    result = result[available_features]
```

**Changed to:**
```python
# Filter to selected features if top_n specified
if self.top_n is not None and self._selected_features:
    # Deduplicate while preserving order (prevents duplicate column names)
    available_features = []
    seen = set()
    for f in self._selected_features:
        if f in result.columns and f not in seen:
            available_features.append(f)
            seen.add(f)
    result = result[available_features]
```

### Fix 2: StepEIX.bake() - Deduplicate features and interactions

**File:** `py_recipes/steps/interaction_detection.py` (lines 441-455)

**Changed from:**
```python
result = pd.DataFrame(index=data.index)

# Add selected features
for feature in self._selected_features:
    if feature in data.columns:
        result[feature] = data[feature]

# Create interaction features
for interaction in self._interactions_to_create:
    parent = interaction['parent']
    child = interaction['child']
    name = interaction['name']

    if parent in data.columns and child in data.columns:
        result[name] = data[parent] * data[child]
```

**Changed to:**
```python
result = pd.DataFrame(index=data.index)

# Add selected features (deduplicate to prevent duplicate columns)
added_features = set()
for feature in self._selected_features:
    if feature in data.columns and feature not in added_features:
        result[feature] = data[feature]
        added_features.add(feature)

# Create interaction features (deduplicate to prevent duplicate columns)
for interaction in self._interactions_to_create:
    parent = interaction['parent']
    child = interaction['child']
    name = interaction['name']

    if parent in data.columns and child in data.columns and name not in result.columns:
        result[name] = data[parent] * data[child]
```

---

## Why Duplicates Occurred

In `StepSafe.prep()` and `StepEIX.prep()`, when building `_selected_features`, the same feature name could potentially be added multiple times if:

1. **StepSafe:** A transformed feature appears multiple times in the importance table
2. **StepEIX:** A variable appears in both the variables list and as part of an interaction

While the prep logic should ideally prevent this, the deduplication in bake() provides a defensive safeguard.

---

## Impact

**Files Modified:**
1. `py_recipes/steps/feature_extraction.py` - StepSafe.bake() deduplication
2. `py_recipes/steps/interaction_detection.py` - StepEIX.bake() deduplication

**Tests:**
- All 64 tests passing (30 SAFE + 34 EIX)
- Added new test: `test_no_duplicate_columns()` to explicitly verify deduplication
- No test failures or regressions

**New Test Added:**
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

---

## Verification

### Before Fix:
```python
# If _selected_features = ['x1_safe', 'x2_safe', 'x1_safe']  # duplicate!
result = result[['x1_safe', 'x2_safe', 'x1_safe']]  # Creates duplicate columns!
# result.columns = Index(['x1_safe', 'x2_safe', 'x1_safe'])
# result['x1_safe'] returns DataFrame (not Series) → error in mold()
```

### After Fix:
```python
# Deduplication ensures unique columns
available_features = ['x1_safe', 'x2_safe']  # No duplicates
result = result[['x1_safe', 'x2_safe']]  # All unique columns
# result.columns = Index(['x1_safe', 'x2_safe'])
# result['x1_safe'] returns Series → works correctly in mold()
```

---

## Key Takeaway

When building DataFrames programmatically by adding columns in a loop:
- **Always check if column already exists** before adding
- **Use sets to track added columns** for efficient deduplication
- **Preserve order** when deduplicating (use list + set pattern)

Pattern:
```python
result = pd.DataFrame()
added = set()
for col in column_list:
    if col not in added:
        result[col] = data[col]
        added.add(col)
```

This prevents the subtle but critical issue of duplicate column names that cause pandas to return DataFrames instead of Series.

---

**Fix Applied:** 2025-11-09
**Status:** Complete ✅
**Tests:** 64/64 passing (30 SAFE + 34 EIX)
**Additional Test:** test_no_duplicate_columns() added to verify fix

---

## Summary of All Fixes

1. **pd.concat() deduplication** (CRITICAL) - Prevents duplicates at source
2. **top_n filtering deduplication** - Additional safety when selecting features
3. **StepEIX loop deduplication** - Already safe with one-by-one column addition
4. **New test added** - Explicit verification of deduplication behavior

All potential sources of duplicate columns now handled!
