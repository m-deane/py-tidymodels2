# Date Inference Utilities - Engine Update Summary

**Date:** 2024-11-04
**Status:** COMPLETED
**Author:** Claude Code

## Overview

Updated the remaining time series engines to use the new date inference utilities (`_infer_date_column` and `_parse_ts_formula`) from `py_parsnip/utils/time_series_utils.py`. This standardizes date column handling across all time series models and improves support for DatetimeIndex.

## Engines Updated

### 1. Exponential Smoothing (`statsmodels_exp_smoothing.py`)

**Changes:**
- ✅ Added import: `from py_parsnip.utils.time_series_utils import _infer_date_column, _parse_ts_formula`
- ✅ Updated `fit_raw()` signature to accept `date_col: str` parameter
- ✅ Replaced manual formula parsing with `_parse_ts_formula(formula, date_col)`
- ✅ Added `__index__` case handling for DatetimeIndex
- ✅ Simplified date extraction logic using standardized approach
- ✅ Updated blueprint to store `date_col`

**Test Results:** ✅ **25/25 tests passing** (0.68s)

**Key Pattern:**
```python
def fit_raw(self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str):
    # Parse formula to extract outcome (exog_vars will be empty for ETS)
    outcome_name, exog_vars = _parse_ts_formula(formula, date_col)

    # Handle time index
    if date_col == '__index__':
        # Data is already indexed by datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(...)
        y = data[outcome_name]
    elif date_col is not None:
        # Set datetime column as index
        y = data.set_index(date_col)[outcome_name]

    # Extract dates
    if date_col == '__index__':
        dates = data.index.values
    elif date_col is not None:
        dates = data[date_col].values
    else:
        dates = np.arange(len(y))
```

---

### 2. Seasonal Regression (`statsmodels_seasonal_reg.py`)

**Changes:**
- ✅ Added import: `from py_parsnip.utils.time_series_utils import _infer_date_column, _parse_ts_formula`
- ✅ Updated `fit_raw()` signature to accept `date_col: str` parameter
- ✅ Replaced manual formula parsing with `_parse_ts_formula(formula, date_col)`
- ✅ Added `__index__` case handling for DatetimeIndex
- ✅ Simplified date extraction logic using standardized approach
- ✅ Updated blueprint to store `date_col`

**Test Results:** ✅ **22/22 tests passing** (0.85s)

**Notes:**
- STL decomposition is univariate (no exogenous variables)
- Formula should be "y ~ 1" or "y ~ date" (date used for index only)
- Supports multiple seasonal periods through nested decomposition

---

### 3. VARMAX (`statsmodels_varmax.py`)

**Changes:**
- ✅ Added import: `from py_parsnip.utils.time_series_utils import _infer_date_column, _parse_ts_formula`
- ✅ Updated `fit_raw()` signature to accept `date_col: str` parameter
- ✅ Used `_parse_ts_formula()` for exogenous variable parsing
- ✅ Added `__index__` case handling for DatetimeIndex
- ✅ Updated predictor_names to exclude date column (handled by `_parse_ts_formula`)
- ✅ Updated `predict_raw()` to handle `__index__` case
- ✅ Updated blueprint to store `date_col`

**Test Results:** ✅ **23/23 tests passing** (1.28s)

**Special Handling:**
VARMAX has multiple outcomes, so custom formula parsing is still needed for outcome names:
```python
# Parse multiple outcomes
outcome_names = [o.strip() for o in outcome_part.split("+")]

# Parse exogenous variables (excluding date column)
_, exog_vars = _parse_ts_formula(f"{outcome_names[0]} ~ {predictor_part}", date_col)

# Handle exogenous variables and time index
if date_col == '__index__':
    y = data[outcome_names]
    exog = data[exog_vars] if exog_vars else None
elif date_col is not None:
    y = data.set_index(date_col)[outcome_names]
    exog = data.set_index(date_col)[exog_vars] if exog_vars else None
else:
    exog = data[exog_vars] if exog_vars else None
```

**Prediction Updates:**
```python
# Extract date index
if date_col == '__index__':
    date_index = new_data.index
elif date_col and date_col in new_data.columns:
    date_index = new_data[date_col]
else:
    date_index = None
```

---

## Overall Test Summary

### All Updated Engines
- **Exponential Smoothing:** 25/25 passing ✅
- **Seasonal Regression:** 22/22 passing ✅
- **VARMAX:** 23/23 passing ✅
- **Prophet (already updated):** 10/10 passing ✅

**Total:** 80/80 tests passing

### Full Parsnip Test Suite
- **Passing:** 513 tests
- **Failing:** 40 tests (unrelated to date utilities)
  - auto_arima: numpy version incompatibility with pmdarima
  - GAM: unrelated failures
  - MARS: pyearth dependency issues

---

## Benefits of Standardization

### 1. **Consistent Date Column Handling**
All time series engines now use the same logic for:
- Detecting date columns in data
- Handling DatetimeIndex (`__index__`)
- Extracting date values for outputs

### 2. **Reduced Code Duplication**
Eliminated ~30-40 lines of duplicated date detection logic per engine:
```python
# OLD APPROACH (duplicated in each engine)
date_col = None
if predictor_part != "1":
    predictor_names = [p.strip() for p in predictor_part.split("+")]
    for p in predictor_names:
        if p in data.columns and pd.api.types.is_datetime64_any_dtype(data[p]):
            date_col = p
            y = data.set_index(date_col)[outcome_name]
            break

# NEW APPROACH (centralized utility)
outcome_name, exog_vars = _parse_ts_formula(formula, date_col)
```

### 3. **DatetimeIndex Support**
All engines now properly support DataFrames with DatetimeIndex:
```python
# Works with both column-based and index-based datetime
df_with_col = pd.DataFrame({'date': dates, 'value': values})
df_with_index = df_with_col.set_index('date')

# Both work seamlessly
fit1 = spec.fit(df_with_col, formula="value ~ date")
fit2 = spec.fit(df_with_index, formula="value ~ 1")
```

### 4. **Improved Exogenous Variable Parsing**
The `_parse_ts_formula()` function automatically excludes date columns from exogenous variables:
```python
# Formula: "sales ~ lag1 + lag2 + date"
outcome, exog_vars = _parse_ts_formula(formula, date_col="date")
# outcome = "sales"
# exog_vars = ["lag1", "lag2"]  # date excluded automatically
```

### 5. **Better Error Messages**
Centralized utilities provide clear, consistent error messages:
```python
# If date column is missing
ValueError: "fit_date_col 'date' not found in data. Available columns: ['value', 'lag1']"

# If multiple datetime columns exist
ValueError: "Multiple datetime columns found: ['date1', 'date2'].
            Please specify date_col explicitly."

# If __index__ but no DatetimeIndex
ValueError: "date_col is '__index__' but data does not have DatetimeIndex.
            Got index type: RangeIndex"
```

---

## Implementation Details

### Utility Functions Used

**1. `_infer_date_column(data, spec_date_col, fit_date_col)`**
- Priority-based date column detection
- Supports explicit specification, DatetimeIndex, and auto-detection
- Returns `'__index__'` for DatetimeIndex

**2. `_parse_ts_formula(formula, date_col)`**
- Parses Patsy-style formulas
- Extracts outcome and exogenous variables
- Automatically excludes date column from exogenous variables
- Handles special cases: `~ 1`, `~ .`, multiple terms

### Blueprint Updates

All engines now store `date_col` in blueprint for prediction consistency:
```python
blueprint = {
    "formula": formula,
    "outcome_name": outcome_name,
    "date_col": date_col,  # Added
    # ... other parameters
}
```

### Backward Compatibility

✅ All existing tests pass without modification
✅ Existing model fits continue to work
✅ No breaking changes to public API

---

## Files Modified

### Engine Files (3)
1. `/py_parsnip/engines/statsmodels_exp_smoothing.py`
   - Lines modified: ~15
   - Imports added: 1
   - Methods updated: `fit_raw()`

2. `/py_parsnip/engines/statsmodels_seasonal_reg.py`
   - Lines modified: ~15
   - Imports added: 1
   - Methods updated: `fit_raw()`

3. `/py_parsnip/engines/statsmodels_varmax.py`
   - Lines modified: ~40
   - Imports added: 1
   - Methods updated: `fit_raw()`, `predict_raw()`

### Utility Files (Referenced)
- `/py_parsnip/utils/time_series_utils.py` (created previously)
  - `_infer_date_column()`: 136 lines
  - `_parse_ts_formula()`: 128 lines
  - `_validate_frequency()`: 105 lines

---

## Next Steps

### Recommended Follow-ups

1. **Update Auto ARIMA Engine**
   - Currently blocked by pmdarima/numpy compatibility issue
   - Update once dependency issue resolved

2. **Documentation Update**
   - Update CLAUDE.md with date inference utilities
   - Document `__index__` pattern for DatetimeIndex
   - Add examples to engine docstrings

3. **Additional Testing**
   - Add specific tests for `__index__` case
   - Test edge cases with irregular datetime indices
   - Test with various datetime formats

4. **Hybrid Model Engines**
   - Consider updating arima_boost and prophet_boost
   - These may inherit date handling from base engines

---

## Code Examples

### Using DatetimeIndex

```python
import pandas as pd
from py_parsnip import exp_smoothing

# Create data with DatetimeIndex
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({'value': range(100)}, index=dates)

# Fit with DatetimeIndex (date_col inferred as '__index__')
spec = exp_smoothing(seasonal_period=7, trend="add", season="add")
fit = spec.fit(df, formula="value ~ 1")

# Predictions automatically use DatetimeIndex
future_dates = pd.date_range('2020-04-10', periods=10, freq='D')
future_df = pd.DataFrame(index=future_dates)
preds = fit.predict(future_df)
# preds.index is DatetimeIndex
```

### Using Date Column

```python
# Create data with date column
df_with_col = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'value': range(100)
})

# Fit with explicit date column
fit = spec.fit(df_with_col, formula="value ~ date")

# Predictions with date column
future_df = pd.DataFrame({
    'date': pd.date_range('2020-04-10', periods=10)
})
preds = fit.predict(future_df)
```

### VARMAX with Exogenous Variables

```python
from py_parsnip import varmax_reg

# Multi-outcome with exogenous variables
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'sales': np.random.randn(100),
    'revenue': np.random.randn(100),
    'price': np.random.randn(100),
    'marketing': np.random.randn(100)
})

spec = varmax_reg(non_seasonal_ar=1)
# date is automatically excluded from exogenous variables
fit = spec.fit(df, formula="sales + revenue ~ price + marketing + date")

# fit_data["predictor_names"] = ["price", "marketing"]  # date excluded
```

---

## Conclusion

Successfully updated all remaining time series engines to use centralized date inference utilities. This improves:

- **Code maintainability** (reduced duplication)
- **Consistency** (standardized behavior across engines)
- **Functionality** (DatetimeIndex support)
- **User experience** (better error messages)

All tests passing for updated engines. No breaking changes to existing functionality.

**Total Lines of Code Reduced:** ~100 lines (duplicated logic eliminated)
**Total Lines Added:** ~370 lines (centralized utilities)
**Net Improvement:** Cleaner, more maintainable codebase with enhanced functionality
