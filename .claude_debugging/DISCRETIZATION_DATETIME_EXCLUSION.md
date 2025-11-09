# Discretization and Dummy Steps - Datetime Column Exclusion

**Date:** 2025-11-09
**Issue:** ValueError in formula parsing when datetime columns are processed by discretization and dummy encoding steps
**Status:** ✅ FIXED

---

## Problem

When using discretization steps (`step_discretize()`, `step_cut()`, `step_percentile()`) or dummy encoding (`step_dummy()`) with data containing datetime columns, the steps would attempt to process the datetime columns, creating invalid column names that break patsy's formula parser:

```
ValueError: Failed to parse formula 'target ~ Q("date_2020-04-01T00:00:00.000000000") + Q("date_2020-05-01T00:00:00.000000000") + ...'
```

### Root Cause

**Two-stage problem:**

1. **Discretization steps** were processing ALL columns returned by selectors (including datetime columns), creating categorical bins
2. **step_dummy()** was treating datetime columns as "nominal" (because datetime64 is not `np.number`), and creating dummy variables with datetime-based column names like:
   - `date_2020-04-01T00:00:00.000000000`
   - `date_2020-05-01T00:00:00.000000000`
   - etc.

These column names cannot be parsed by patsy's formula parser, causing `ValueError`.

### Error Location

```python
# In forecasting_recipes.ipynb:
rec_discretize = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4, method="quantile")
    .step_dummy(all_nominal_predictors())
)

wf_discretize = workflow().add_recipe(rec_discretize).add_model(linear_reg())
fit_discretize = wf_discretize.fit(train_data)  # ← ERROR: ValueError in formula parsing

# train_data has columns: ['date', 'x1', 'x2', ..., 'target']
# step_discretize was trying to discretize the 'date' column
# This created columns like 'date_2020-04-01T00:00:00.000000000'
# Formula parser cannot handle these datetime-based column names
```

---

## Solution

Modified all discretization steps AND step_dummy() to **automatically exclude datetime columns** before processing.

### Implementation

In `py_recipes/steps/discretization.py`, added datetime column filtering in `prep()` methods for all 3 discretization steps:

#### 1. StepDiscretize (Lines 51-55)

```python
# OLD CODE (lines 47-52):
# Resolve selector to column list
selector = self.columns if self.columns is not None else all_numeric()
cols = resolve_selector(selector, data)

# Calculate bin edges for each column
bin_edges = {}
```

```python
# NEW CODE:
# Resolve selector to column list
selector = self.columns if self.columns is not None else all_numeric()
cols = resolve_selector(selector, data)

# Exclude datetime columns (cannot be discretized)
cols = [
    c for c in cols
    if not pd.api.types.is_datetime64_any_dtype(data[c])
]

# Calculate bin edges for each column
bin_edges = {}
```

#### 2. StepCut (Lines 168-172)

```python
# OLD CODE (lines 160-163):
cols = [col for col in self.columns if col in data.columns]

# Validate breaks
valid_breaks = {}
valid_labels = {}
```

```python
# NEW CODE:
cols = [col for col in self.columns if col in data.columns]

# Exclude datetime columns (cannot be discretized)
cols = [
    c for c in cols
    if not pd.api.types.is_datetime64_any_dtype(data[c])
]

# Validate breaks
valid_breaks = {}
valid_labels = {}
```

#### 3. StepPercentile (Lines 281-285)

```python
# OLD CODE (lines 266-270):
# Resolve selector to column list
selector = self.columns if self.columns is not None else all_numeric()
cols = resolve_selector(selector, data)

# Calculate percentile breakpoints for each column
percentile_breaks = {}
```

```python
# NEW CODE:
# Resolve selector to column list
selector = self.columns if self.columns is not None else all_numeric()
cols = resolve_selector(selector, data)

# Exclude datetime columns (cannot be converted to percentiles)
cols = [
    c for c in cols
    if not pd.api.types.is_datetime64_any_dtype(data[c])
]

# Calculate percentile breakpoints for each column
percentile_breaks = {}
```

---

## Behavior

### Before Fix

```python
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

data = pd.DataFrame({
    'date': pd.date_range('2020-04-01', periods=50, freq='MS'),
    'x1': np.random.randn(50),
    'x2': np.random.randn(50),
    'target': np.random.randn(50)
})

rec = recipe().step_discretize(all_numeric_predictors(), num_breaks=4)
prepped = rec.prep(data)
baked = prepped.bake(data)

# ❌ ERROR: Creates columns like 'date_2020-04-01T00:00:00.000000000'
# These column names break formula parsing
```

### After Fix

```python
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

data = pd.DataFrame({
    'date': pd.date_range('2020-04-01', periods=50, freq='MS'),
    'x1': np.random.randn(50),
    'x2': np.random.randn(50),
    'target': np.random.randn(50)
})

rec = recipe().step_discretize(all_numeric_predictors(), num_breaks=4)
prepped = rec.prep(data)
baked = prepped.bake(data)

# ✅ Works! Only x1 and x2 are discretized
# Date column is preserved as-is (not discretized)
# baked columns: ['date', 'x1', 'x2', 'target']
# x1 and x2 are now categorical with bin labels
```

---

## Important Notes

### 1. Datetime Columns Not Discretized

Datetime columns are **completely excluded** from discretization. They remain as datetime columns in the output.

If you want to use datetime information, extract features first:
```python
rec = (
    recipe()
    .step_date(date_col='date', features=['year', 'month', 'day', 'dow'])  # Extract features
    .step_rm(['date'])  # Remove original date column
    .step_discretize(all_numeric_predictors(), num_breaks=4)  # Discretize numeric features
)
```

### 2. Multiple Datetime Columns

The fix handles multiple datetime columns correctly:

```python
data = pd.DataFrame({
    'date1': pd.date_range('2020-01-01', periods=50),
    'date2': pd.date_range('2021-01-01', periods=50),
    'x1': np.random.randn(50),
    'x2': np.random.randn(50),
    'target': np.random.randn(50)
})

rec = recipe().step_discretize(all_numeric_predictors(), num_breaks=4)
prepped = rec.prep(data)  # ✅ Works, both date columns excluded from discretization
```

### 3. Workflow Integration

Works seamlessly in workflow context:

```python
from py_workflows import workflow
from py_parsnip import linear_reg

wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_discretize(all_numeric_predictors(), num_breaks=4)
        .step_dummy(all_nominal_predictors())
    )
    .add_model(linear_reg())
)

fit = wf.fit(train_data)  # ✅ No ValueError
fit = fit.evaluate(test_data)
```

---

## Test Results

All tests passing:

```
✅ Test 1: step_discretize() with datetime
   Works, excludes date column from discretization

✅ Test 2: Workflow Integration (discretize → dummy)
   Fits and evaluates without errors
   No formula parsing errors

✅ Test 3: Multiple Datetime Columns
   Works with 2+ datetime columns

✅ Test 4: All Discretization Steps
   ✅ step_discretize()
   ✅ step_cut()
   ✅ step_percentile()
```

#### 4. StepDummy (Lines 50-54)

```python
# OLD CODE (lines 47-48):
# Filter to existing columns
existing_cols = [col for col in cols if col in data.columns]
```

```python
# NEW CODE:
# Filter to existing columns
existing_cols = [col for col in cols if col in data.columns]

# Exclude datetime columns (cannot be dummy encoded)
existing_cols = [
    c for c in existing_cols
    if not pd.api.types.is_datetime64_any_dtype(data[c])
]
```

---

## Files Modified

**Modified:**
- `py_recipes/steps/discretization.py`
  - StepDiscretize.prep() - Lines 51-55
  - StepCut.prep() - Lines 168-172
  - StepPercentile.prep() - Lines 281-285
- `py_recipes/steps/dummy.py`
  - StepDummy.prep() - Lines 50-54

**Changes:** Added datetime exclusion filter in all 3 discretization steps AND step_dummy()

---

## Examples

### Basic Usage

```python
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors
import pandas as pd
import numpy as np

# Data with datetime column
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100, freq='MS'),
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'target': np.random.randn(100)
})

# Discretize numeric predictors
rec = recipe().step_discretize(all_numeric_predictors(), num_breaks=4)
prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: x1, x2, x3 discretized into 4 bins each, date excluded
# baked columns: ['date', 'x1', 'x2', 'x3', 'target']
# x1, x2, x3 are now categorical: 'bin_1', 'bin_2', 'bin_3', 'bin_4'
```

### With Dummy Encoding

```python
# This was the failing pattern - now works!
rec = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4, method="quantile")
    .step_dummy(all_nominal_predictors())
)
prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: Discretized columns are one-hot encoded
# No formula parsing errors!
```

### In Forecasting Pipeline

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Time series forecasting with discretization
wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_lag(['target'], lags=[1, 2, 3])  # Create lag features
        .step_discretize(all_numeric_predictors(), num_breaks=5)  # Discretize lags
        .step_dummy(all_nominal_predictors())  # One-hot encode bins
    )
    .add_model(linear_reg())
)

fit = wf.fit(train_data)  # ✅ Works even with date column
predictions = fit.predict(test_data)
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code without datetime columns: No change in behavior
- Existing code with datetime columns: Now works instead of erroring
- No API changes, only internal logic enhancement

---

## Related Issues

This fix resolves:
- ValueError in formula parsing when discretization steps process datetime columns
- Any use of discretization steps with datetime-indexed data
- Workflow integration issues with time series data containing discretization

---

## Related Documentation

- `py_recipes/steps/discretization.py` - Implementation
- `.claude_debugging/SUPERVISED_FILTERS_DATETIME_FIX.md` - Similar fix for supervised filters
- `.claude_debugging/STEP_DUMMY_SELECTOR_SUPPORT.md` - Related selector fix

---

## Summary

✅ **FIXED** - All discretization steps AND step_dummy() now automatically exclude datetime columns

- No more ValueError with datetime data
- Datetime columns excluded from discretization and dummy encoding
- Works in all contexts: recipes, workflows, forecasting pipelines
- All 3 discretization steps updated
- step_dummy() updated
- Fully backward compatible
- All tests passing

**Usage:** Just use discretization and dummy steps normally - datetime exclusion is automatic:
```python
# Both steps automatically exclude datetime columns
rec = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4)
    .step_dummy(all_nominal_predictors())  # ✓ Excludes datetime columns
)
# Datetime columns automatically excluded from both steps ✓
```

**Root Cause Analysis:**
The issue had TWO sources:
1. Discretization steps processing datetime columns (FIXED in first attempt)
2. step_dummy() treating datetime as "nominal" and creating dummy variables (FIXED in second iteration)

Both fixes were needed to fully resolve the ValueError in formula parsing.
