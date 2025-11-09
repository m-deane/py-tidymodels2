# step_naomit() - Infinity Value Handling

**Date:** 2025-11-09
**Enhancement:** Added infinity value detection and removal to step_naomit()
**Status:** ✅ COMPLETE

---

## Enhancement

Updated `step_naomit()` to automatically remove rows containing **both** NaN/NA values AND infinite values (±Inf).

### Previous Behavior
- Only removed rows with NaN/NA values
- Infinite values were kept in the data
- Could cause issues in downstream models that don't handle infinity

### New Behavior
- Removes rows with NaN/NA values (as before)
- **Also removes rows with +Inf or -Inf values**
- Only checks numeric columns for infinity (non-numeric columns ignored)
- Works seamlessly with column selection

---

## Implementation

**Modified File:** `py_recipes/steps/naomit.py`

### Changes Made

1. **Added numpy import** (line 11):
   ```python
   import numpy as np
   ```

2. **Updated docstrings** to mention infinity handling:
   - Module docstring: "NA/NaN or infinite values"
   - Class docstring: Examples updated
   - Method docstring: "NAs or infinities"

3. **Enhanced bake() method** (lines 85-91):
   ```python
   # Remove rows with any infinite values in the specified columns
   # Check only numeric columns (infinity doesn't apply to non-numeric)
   numeric_cols = [col for col in cols_to_check if pd.api.types.is_numeric_dtype(result[col])]
   if numeric_cols:
       # Create mask for rows with any infinity
       inf_mask = result[numeric_cols].apply(lambda col: np.isinf(col)).any(axis=1)
       result = result[~inf_mask]
   ```

---

## Usage

### Basic Usage (Check All Columns)
```python
from py_recipes import recipe
import numpy as np

# Data with NaN and Inf values
data = pd.DataFrame({
    'x1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'x2': [10.0, 20.0, 30.0, np.inf, 50.0],
    'target': [1, 2, 3, 4, 5]
})

# Remove all rows with NaN or Inf
rec = recipe().step_naomit()
prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: Rows 2 and 3 removed (NaN in x1, +Inf in x2)
```

### Selective Column Checking
```python
# Only check specific columns for NaN/Inf
rec = recipe().step_naomit(columns=['x2'])
prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: Only row 3 removed (+Inf in x2)
# Row 2 kept (NaN in x1, but x1 not in check list)
```

### Common Use Case: After Log Transformation
```python
# Log transformation can create -Inf for zero values
rec = (
    recipe()
    .step_log(all_numeric_predictors())  # May create -Inf for zeros
    .step_naomit()  # Remove rows with -Inf
)
```

### Time Series with Lags
```python
# Lag features create NaN at the start
rec = (
    recipe()
    .step_lag(['value'], lags=[1, 7, 14])  # Creates NaN at start
    .step_naomit()  # Remove rows with NaN (including lag NaNs)
)
```

---

## Test Results

All tests passing:

```
✅ Test 1: Remove all NaN and Inf (check all columns)
   Original: 8 rows (3 with NaN/Inf)
   Result: 5 rows (3 removed correctly)

✅ Test 2: Remove only from specific columns
   Original: 8 rows
   Check: ['x2'] only
   Result: 6 rows (2 removed from x2, NaN in x1 kept)

✅ Test 3: Non-numeric columns in check list
   No errors when non-numeric columns included
   Only numeric columns checked for infinity
```

---

## Behavior Details

### What Gets Removed

**NaN/NA Values:**
- `np.nan`
- `None` (in numeric columns)
- `pd.NA`

**Infinite Values:**
- `np.inf` (positive infinity)
- `-np.inf` (negative infinity)
- Only checked in **numeric columns**

### What Doesn't Get Removed

**Non-numeric columns:**
- String values are never checked for infinity
- Categorical columns are not checked for infinity
- Only NaN is checked in non-numeric columns

**Valid numeric values:**
- Very large numbers (e.g., 1e308) are kept
- Very small numbers (e.g., 1e-308) are kept
- Only actual infinity (±Inf) is removed

---

## Why This Enhancement?

### Common Sources of Infinity
1. **Division by zero**: `1 / 0` → `inf`
2. **Log of zero**: `np.log(0)` → `-inf`
3. **Overflow**: Very large calculations
4. **Mathematical operations**: `np.tan(np.pi/2)` → `inf`

### Problems Infinity Causes
- Many models don't handle infinity gracefully
- Patsy formula parsing can fail with infinity
- Statistical functions may return incorrect results
- Downstream transformations may propagate infinity

### Solution
`step_naomit()` now acts as a **data cleaning step** that removes both:
- **Missing data** (NaN/NA)
- **Invalid data** (±Inf)

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code works exactly as before
- Only **additional** filtering for infinity
- If no infinity values present, behavior unchanged
- API unchanged (no new parameters)

---

## Related Steps

**Data Cleaning Steps:**
- `step_naomit()` - Remove NaN and Inf ✓
- `step_impute_*()` - Replace NaN with values
- `step_filter_missing()` - Remove columns with too many NaN

**Steps That May Create Infinity:**
- `step_log()` - log(0) = -Inf
- `step_sqrt()` - sqrt(negative) = NaN
- `step_mutate()` - Custom transformations may create Inf
- Division operations in custom steps

---

## Summary

✅ **ENHANCED** - step_naomit() now removes both NaN and infinite values

- Automatically detects and removes ±Inf values
- Only checks numeric columns for infinity
- Works with selective column checking
- Fully backward compatible
- Prevents downstream model failures
- Clean data = better models

**Usage:** Use step_naomit() after any transformation that might create NaN or Inf:
```python
rec = (
    recipe()
    .step_log(all_numeric_predictors())  # May create -Inf
    .step_naomit()  # ✓ Removes both NaN and ±Inf
)
```
