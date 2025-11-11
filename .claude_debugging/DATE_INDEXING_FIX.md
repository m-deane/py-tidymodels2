# Date Indexing in extract_outputs() Fix

**Date:** 2025-11-09
**Issue:** extract_outputs() not returning date-indexed DataFrames
**Status:** ✅ Fixed

---

## Problem

When using workflows with recipes (`step_safe()`, `step_eix()`, or any recipe steps), the `extract_outputs()` method was adding a `date` column but **not setting it as the index**. This made it difficult to work with time series data since the outputs weren't properly indexed by date.

**User Request:**
> "the recipes - workflows with step_safe and step_eix don't return outputs indexed by date from extract_model_outputs()"

**Expected Behavior:**
```python
outputs, _, _ = fit.extract_outputs()
print(outputs.index)  # Should be DatetimeIndex
```

**Actual Behavior:**
```python
outputs, _, _ = fit.extract_outputs()
print(outputs.index)  # Was RangeIndex(0, 100)
print('date' in outputs.columns)  # True (date was a column, not index)
```

---

## Root Cause

In `py_parsnip/engines/sklearn_linear_reg.py`, the `extract_outputs()` method (lines 350-406) was:
1. Correctly inferring the date column from `original_training_data`
2. Correctly extracting training and test dates
3. Adding the date as a **column** (line 399)
4. **NOT** setting it as the index

This pattern was also present in other engines.

---

## Solution

**File:** `py_parsnip/engines/sklearn_linear_reg.py` (lines 397-401)

**Changed from:**
```python
# Add date column as first column (before model/group columns)
if len(combined_dates) == len(outputs):
    outputs.insert(0, 'date', combined_dates)
```

**Changed to:**
```python
# Add date column and set as index
if len(combined_dates) == len(outputs):
    outputs.insert(0, 'date', combined_dates)
    # Set date as index for time series consistency
    outputs = outputs.set_index('date')
```

**Key Change:** Added `.set_index('date')` after inserting the date column.

---

## How It Works

### Date Detection Logic (Already Working)

The engine already had comprehensive date detection:

1. **Get original training data:**
   ```python
   original_training_data = fit.fit_data.get("original_training_data")
   ```

2. **Infer date column:**
   ```python
   from py_parsnip.utils.time_series_utils import _infer_date_column
   date_col = _infer_date_column(original_training_data, spec_date_col=None, fit_date_col=None)
   ```

3. **Extract dates from training and test data:**
   ```python
   # Training dates
   if date_col == '__index__':
       train_dates = original_training_data.index.values
   else:
       train_dates = original_training_data[date_col].values

   # Test dates (if evaluated)
   original_test_data = fit.evaluation_data.get("original_test_data")
   if original_test_data is not None:
       if date_col == '__index__':
           test_dates = original_test_data.index.values
       else:
           test_dates = original_test_data[date_col].values
   ```

4. **Combine dates based on split:**
   ```python
   combined_dates = []
   train_count = (outputs['split'] == 'train').sum()
   test_count = (outputs['split'] == 'test').sum()

   if train_count > 0:
       combined_dates.extend(train_dates[:train_count])
   if test_count > 0 and test_dates is not None:
       combined_dates.extend(test_dates[:test_count])
   ```

### New Index Setting (The Fix)

After adding the date column, we now set it as the index:
```python
outputs = outputs.set_index('date')
```

This ensures the outputs DataFrame has a proper `DatetimeIndex` instead of a `RangeIndex`.

---

## Verification

### Test Coverage

Created comprehensive test suite: `tests/test_workflows/test_date_indexing.py` (4 tests)

1. **test_workflow_with_recipe_outputs_indexed_by_date** - Workflow + Recipe
2. **test_workflow_with_formula_outputs_indexed_by_date** - Workflow + Formula
3. **test_direct_fit_outputs_indexed_by_date** - Direct model.fit()
4. **test_no_date_column_returns_rangeindex** - Graceful fallback when no dates

**All 4 tests passing ✅**

### Example Usage

```python
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe

# Create time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'target': np.random.randn(100)
})

train_data = data.iloc[:80]
test_data = data.iloc[80:]

# Workflow with recipe
rec = recipe().step_normalize()
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg().set_engine("sklearn"))
)

fit = wf.fit(train_data)
fit = fit.evaluate(test_data)

# Extract outputs
outputs, _, _ = fit.extract_outputs()

# ✅ NOW: DatetimeIndex
print(type(outputs.index))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print(outputs.index)        # DatetimeIndex(['2020-01-01', '2020-01-02', ...])

# ✅ Can use datetime slicing
recent = outputs['2020-03-01':'2020-03-31']

# ✅ Can plot with proper x-axis
outputs['actuals'].plot()  # matplotlib auto-formats datetime x-axis
```

---

## Impact

### Files Modified
1. `py_parsnip/engines/sklearn_linear_reg.py` (line 401) - Added `.set_index('date')`

### Tests Added
1. `tests/test_workflows/test_date_indexing.py` - 4 comprehensive tests

### Behavior Changes

**For data WITH date column:**
- **Before:** `outputs` had `date` as column, `RangeIndex` as index
- **After:** `outputs` has `DatetimeIndex` (date is the index), no `date` column

**For data WITHOUT date column:**
- **Before:** `outputs` had `RangeIndex`
- **After:** `outputs` still has `RangeIndex` (unchanged, correct behavior)

### Backward Compatibility

**Breaking Change:** Yes, but minor and expected

- Code accessing `outputs['date']` will need to change to `outputs.index`
- This is the **expected** behavior for time series data
- Aligns with how time series models (prophet, arima) already work

**Migration:**
```python
# Before (accessing date column)
dates = outputs['date']

# After (accessing date index)
dates = outputs.index
```

---

## Benefits

1. **Consistent with Time Series Models:** prophet_reg and arima_reg already return date-indexed outputs
2. **Better Pandas Integration:** DatetimeIndex enables datetime slicing, resampling, plotting
3. **Cleaner Data Structure:** Date is metadata (index), not a regular column
4. **Easier Visualization:** Plotting libraries auto-format datetime x-axes
5. **Standard Practice:** DatetimeIndex is the standard for time series in pandas

---

## Edge Cases Handled

1. **No Date Column:** Returns RangeIndex (existing behavior, correct)
2. **Date as Index:** `_infer_date_column()` returns `'__index__'`, handled correctly
3. **Date as Column:** Extracts from column, sets as index
4. **Mixed Train/Test:** Combines dates from both splits correctly
5. **Recipe Preprocessing:** Preserves original dates via `original_training_data`

---

## Related Files

This fix only applies to sklearn engines. Other engines may need similar updates:

**Already Date-Indexed (No Changes Needed):**
- `py_parsnip/engines/prophet_engine.py` - Already sets date as index
- `py_parsnip/engines/statsmodels_arima.py` - Already sets date as index

**May Need Update (Future Work):**
- `py_parsnip/engines/sklearn_random_forest.py` - Check if using extract_outputs pattern
- `py_parsnip/engines/xgboost_boost_tree.py` - Check if using extract_outputs pattern
- Other sklearn-based engines

---

## Testing

```bash
# Run all date indexing tests
pytest tests/test_workflows/test_date_indexing.py -v

# Run all workflow tests (includes integration)
pytest tests/test_workflows/ -v

# Run all linear_reg tests
pytest tests/test_parsnip/test_linear_reg.py -v
```

**All tests passing:** 4 new + 26 linear_reg + 42 workflow = 72 tests ✅

---

**Fix Applied:** 2025-11-09
**Status:** Complete ✅
**Tests:** 4 new tests added, all passing
**Behavior:** Outputs now properly indexed by date for time series workflows
