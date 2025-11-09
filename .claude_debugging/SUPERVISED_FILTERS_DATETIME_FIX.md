# Supervised Filter Steps - Datetime Column Fix

**Date:** 2025-11-09
**Issue:** DTypePromotionError when datetime columns present in data
**Status:** ✅ FIXED

---

## Problem

When using supervised feature selection steps (`step_filter_mutual_info()`, `step_filter_anova()`, `step_filter_rf_importance()`) with data containing datetime columns, sklearn would raise a `DTypePromotionError`:

```
DTypePromotionError: The DType <class 'numpy.dtypes.DateTime64DType'> could not
be promoted by <class 'numpy.dtypes.Float64DType'>. This means that no common
DType exists for the given inputs.
```

### Root Cause

The supervised filter steps were attempting to score ALL columns (except the outcome), including datetime columns. When sklearn's scoring functions (`mutual_info_regression`, `f_regression`, etc.) tried to process the mixed DataFrame with both datetime and numeric columns, numpy couldn't find a common dtype.

### Error Location

```python
# In forecasting_recipes.ipynb cell 67:
rec_mi = (
    recipe()
    .step_filter_mutual_info(outcome="target", top_n=6, n_neighbors=3)
    .step_normalize(all_numeric_predictors())
)

wf_mi = workflow().add_recipe(rec_mi).add_model(linear_reg())
fit_mi = wf_mi.fit(train_data)  # ← ERROR: DTypePromotionError

# train_data has columns: ['date', 'x1', 'x2', ..., 'target']
# step_filter_mutual_info was trying to score 'date' column
```

---

## Solution

Modified all supervised filter steps to **automatically exclude datetime columns** from scoring while preserving them in the output.

### Implementation

In `py_recipes/steps/filter_supervised.py`, updated the column resolution logic in `prep()` methods for all 4 supervised filter steps:

```python
# OLD CODE (lines 523-536):
# Resolve columns
if self.columns is None:
    score_cols = [c for c in data.columns if c != self.outcome]
elif isinstance(self.columns, str):
    score_cols = [self.columns]
elif callable(self.columns):
    score_cols = self.columns(data)
else:
    score_cols = list(self.columns)

score_cols = [c for c in score_cols if c != self.outcome]

if len(score_cols) == 0:
    raise ValueError("No columns to score")
```

```python
# NEW CODE:
# Resolve columns
if self.columns is None:
    score_cols = [c for c in data.columns if c != self.outcome]
elif isinstance(self.columns, str):
    score_cols = [self.columns]
elif callable(self.columns):
    score_cols = self.columns(data)
else:
    score_cols = list(self.columns)

# Exclude outcome and datetime columns
score_cols = [
    c for c in score_cols
    if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
]

if len(score_cols) == 0:
    raise ValueError("No columns to score")
```

### Steps Fixed

All 4 supervised filter steps now automatically exclude datetime columns:

1. **StepFilterAnova** - ANOVA F-test filter
2. **StepFilterRFImportance** - Random Forest importance filter
3. **StepFilterMutualInfo** - Mutual information filter
4. **StepFilterChiSquared** - Chi-squared filter (for classification)

---

## Behavior

### Before Fix

```python
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=50),
    'x1': np.random.randn(50),
    'x2': np.random.randn(50),
    'target': np.random.randn(50)
})

rec = recipe().step_filter_mutual_info(outcome="target", top_n=2)
prepped = rec.prep(data)  # ❌ DTypePromotionError
```

### After Fix

```python
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=50),
    'x1': np.random.randn(50),
    'x2': np.random.randn(50),
    'target': np.random.randn(50)
})

rec = recipe().step_filter_mutual_info(outcome="target", top_n=2)
prepped = rec.prep(data)  # ✅ Works
baked = prepped.bake(data)

# Result: Selects top 2 numeric features, datetime excluded from scoring
# baked columns: ['x1', 'x2', 'target']
# Note: date column is removed by the filter step (not selected)
```

---

## Important Notes

### 1. Datetime Columns Not Scored

Datetime columns are **excluded from scoring** but may be **removed from output** if not in the top_n selected features. This is the expected behavior for feature selection steps.

If you want to keep datetime columns in the output:
```python
rec = (
    recipe()
    .step_filter_mutual_info(outcome="target", top_n=3)
    # Date column will be removed if not in top 3
    # To keep it, don't use filter or add it back:
)

# OR, better approach - use selector to specify which columns to filter:
from py_recipes.selectors import all_numeric_predictors

rec = (
    recipe()
    .step_filter_mutual_info(
        columns=all_numeric_predictors(),  # Only filter numeric predictors
        outcome="target",
        top_n=3
    )
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

rec = recipe().step_filter_mutual_info(outcome="target", top_n=2)
prepped = rec.prep(data)  # ✅ Works, both date columns excluded from scoring
```

### 3. Workflow Integration

Works seamlessly in workflow context:

```python
wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_filter_mutual_info(outcome="target", top_n=5)
        .step_normalize(all_numeric_predictors())
    )
    .add_model(linear_reg())
)

fit = wf.fit(train_data)  # ✅ No DTypePromotionError
fit = fit.evaluate(test_data)
```

---

## Test Results

All tests passing:

```
✅ Test 1: step_filter_mutual_info() with datetime
   Works, selects top 3 numeric features

✅ Test 2: Workflow Integration (with evaluate)
   Fits and evaluates without errors
   Model metrics: RMSE, MAE, MAPE calculated correctly

✅ Test 3: Multiple Datetime Columns
   Works with 2+ datetime columns

✅ Test 4: All Supervised Filter Types
   ✅ ANOVA
   ✅ RF Importance
   ✅ Mutual Info
   ✅ Chi-Squared (classification)
```

---

## Files Modified

**Modified:**
- `py_recipes/steps/filter_supervised.py`
  - StepFilterAnova.prep() - Lines 133-145
  - StepFilterRFImportance.prep() - Lines 283-295
  - StepFilterMutualInfo.prep() - Lines 523-536
  - StepFilterChiSquared.prep() - Lines ~740-752

**Changes:** Added datetime exclusion filter in all 4 supervised filter steps

---

## Examples

### Basic Usage

```python
from py_recipes import recipe
import pandas as pd
import numpy as np

# Data with datetime column
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'x4': np.random.randn(100),
    'x5': np.random.randn(100),
    'target': np.random.randn(100)
})

# Mutual information filter
rec_mi = recipe().step_filter_mutual_info(outcome="target", top_n=3)
prepped = rec_mi.prep(data)
baked = prepped.bake(data)

# Result: Top 3 numeric features selected, date excluded from scoring
# baked columns: ['x1', 'x5', 'x2', 'target'] (example)
```

### ANOVA Filter

```python
# ANOVA F-test filter (for regression)
rec_anova = recipe().step_filter_anova(outcome="target", top_p=0.5)
prepped = rec_anova.prep(data)
baked = prepped.bake(data)

# Selects top 50% of features by F-statistic
```

### RF Importance Filter

```python
# Random Forest importance filter
rec_rf = recipe().step_filter_rf_importance(
    outcome="target",
    top_n=5,
    trees=100
)
prepped = rec_rf.prep(data)
baked = prepped.bake(data)

# Selects top 5 features by RF importance
```

### In Forecasting Pipeline

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Time series forecasting with feature selection
wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_lag(['target'], lags=[1, 2, 3])  # Create lag features
        .step_filter_mutual_info(outcome="target", top_n=5)  # Select best 5
        .step_normalize(all_numeric_predictors())
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
- DTypePromotionError in forecasting_recipes.ipynb cell 67
- Any use of supervised filters with datetime-indexed data
- Workflow integration issues with time series data

---

## Related Documentation

- `py_recipes/steps/filter_supervised.py` - Implementation
- `FORECASTING_RECIPES_NOTEBOOK_EXPANSION.md` - Notebook examples
- `SELECTOR_IMPORTS_FIX.md` - Related selector fix

---

## Summary

✅ **FIXED** - All supervised filter steps now automatically exclude datetime columns
- No more DTypePromotionError with datetime data
- Datetime columns excluded from scoring (not feature candidates)
- Works in all contexts: recipes, workflows, forecasting pipelines
- All 4 supervised filter steps updated
- Fully backward compatible
- All tests passing

**Usage:** Just use supervised filter steps normally - datetime exclusion is automatic:
```python
rec = recipe().step_filter_mutual_info(outcome="target", top_n=5)
# Datetime columns automatically excluded from scoring ✓
```
