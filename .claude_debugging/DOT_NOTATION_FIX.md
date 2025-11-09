# Dot Notation Support for Time Series Models

**Date:** 2025-11-09
**Issue:** Time series models using `fit_raw()` didn't support patsy's `.` notation for "all variables"
**Solution:** Added `_expand_dot_notation()` utility function and applied to all 9 time series engines

---

## Problem

User attempted to use `"target ~ ."` formula with Prophet in forecasting notebook:

```python
spec = prophet_reg()
fit = spec.fit(data, "target ~ .")  # ERROR: Exogenous variable '.' not found in data
```

**Root Cause:**
- Time series models use `fit_raw()` which bypasses patsy's formula parsing
- Manual formula parser `_parse_ts_formula()` returned `['.']` as literal string
- Engines tried to find a column named `.` in the data, which doesn't exist

**Expected Behavior:**
The `.` in patsy formulas means "all columns except the outcome", so `"target ~ ."` should automatically use all columns except `target` and `date` as exogenous variables.

---

## Solution

### 1. Created Helper Function

Added `_expand_dot_notation()` to `py_parsnip/utils/time_series_utils.py`:

```python
def _expand_dot_notation(exog_vars: List[str], data: pd.DataFrame, outcome_name: str, date_col: str) -> List[str]:
    """
    Expand patsy's "." notation to all columns except outcome and date.

    If exog_vars == ['.'], returns all columns except outcome_name and date_col.
    Otherwise returns exog_vars unchanged.
    """
    if exog_vars == ['.']:
        # Expand to all columns except outcome and date
        return [col for col in data.columns
                if col != outcome_name and col != date_col and col != '__index__']
    return exog_vars
```

### 2. Updated All Time Series Engines

Applied fix to 9 engines that use `fit_raw()`:

**Pattern Applied:**
```python
# Before
outcome_name, exog_vars = _parse_ts_formula(formula, date_col)

# After
outcome_name, exog_vars = _parse_ts_formula(formula, date_col)
exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, date_col)
```

**Files Updated:**
1. `py_parsnip/engines/prophet_engine.py` - Prophet models
2. `py_parsnip/engines/statsmodels_arima.py` - ARIMA models
3. `py_parsnip/engines/statsforecast_auto_arima.py` - Auto ARIMA
4. `py_parsnip/engines/statsmodels_varmax.py` - Multivariate VARMAX
5. `py_parsnip/engines/statsmodels_seasonal_reg.py` - STL decomposition
6. `py_parsnip/engines/pmdarima_auto_arima.py` - pmdarima Auto ARIMA
7. `py_parsnip/engines/statsmodels_exp_smoothing.py` - Exponential smoothing
8. `py_parsnip/engines/hybrid_prophet_boost.py` - Prophet + XGBoost hybrid
9. `py_parsnip/engines/hybrid_arima_boost.py` - ARIMA + XGBoost hybrid

**Also Updated:**
- `py_parsnip/engines/skforecast_recursive.py` - Removed old manual expansion code in favor of utility function

### 3. Special Case: VARMAX

VARMAX handles multiple outcomes (e.g., `"y1 + y2 ~ ."`), so the expansion excludes ALL outcome variables:

```python
# For VARMAX with formula "y1 + y2 ~ ."
outcome_names = ['y1', 'y2']
exog_vars = [col for col in data.columns
             if col not in outcome_names and col != date_col]
```

---

## Usage Examples

### Before (Manual Listing)
```python
from py_parsnip import prophet_reg

# Had to manually list all exogenous variables
spec = prophet_reg()
fit = spec.fit(data, "target ~ x1 + x2 + x3 + x4 + date")
```

### After (Dot Notation)
```python
from py_parsnip import prophet_reg

# Use "." to automatically include all columns
spec = prophet_reg()
fit = spec.fit(data, "target ~ .")  # ✅ Automatically uses x1, x2, x3, x4 (excludes target and date)
```

### Works with All Time Series Models
```python
from py_parsnip import arima_reg, prophet_reg, exp_smoothing, seasonal_reg

# ARIMA with all exogenous variables
spec = arima_reg(non_seasonal_ar=1, non_seasonal_differences=1, non_seasonal_ma=1)
fit = spec.fit(data, "sales ~ .")

# Prophet with all exogenous variables
spec = prophet_reg()
fit = spec.fit(data, "revenue ~ .")

# ETS with all exogenous variables
spec = exp_smoothing(method="holt-winters")
fit = spec.fit(data, "demand ~ .")

# STL with all exogenous variables
spec = seasonal_reg(seasonal_period_1=12)
fit = spec.fit(data, "temperature ~ .")
```

### Combining with Transformations
```python
# Use all variables plus custom interaction
spec = prophet_reg()
fit = spec.fit(data, "sales ~ . + I(price*advertising)")  # All vars + interaction term
```

### VARMAX with Multiple Outcomes
```python
from py_parsnip import varmax_reg

# Use all variables as exogenous for multiple outcomes
spec = varmax_reg()
fit = spec.fit(data, "y1 + y2 ~ .")  # Excludes y1, y2, and date from exogenous
```

---

## Testing

### Test 1: Prophet
```python
import pandas as pd
import numpy as np
from py_parsnip import prophet_reg

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'target': np.random.randn(100).cumsum() + 100
})

spec = prophet_reg()
fit = spec.fit(data, "target ~ .")
# ✅ SUCCESS: Automatically uses x1, x2, x3 as exogenous variables
```

### Test 2: ARIMA
```python
from py_parsnip import arima_reg

spec = arima_reg(non_seasonal_ar=1, non_seasonal_differences=1, non_seasonal_ma=1)
fit = spec.fit(data, "target ~ .")
# ✅ SUCCESS: Automatically uses x1, x2 as exogenous variables
```

**Both tests passed!**

---

## Code References

### Modified Files
- `py_parsnip/utils/time_series_utils.py` - Added `_expand_dot_notation()` function (lines 266-299)
- `py_parsnip/utils/__init__.py` - Exported `_expand_dot_notation`
- 9 engine files - Added expansion call after formula parsing

### Key Functions
1. **`_parse_ts_formula(formula, date_col)`** - Parses formula, returns `['.']` if found
2. **`_expand_dot_notation(exog_vars, data, outcome_name, date_col)`** - Expands `['.']` to actual column names

---

## Impact

### User Experience
- **Before:** Had to manually list all exogenous variables
- **After:** Can use convenient `.` notation like in R tidymodels

### Consistency
- **Before:** Standard models (via mold/forge) supported `.`, but time series models didn't
- **After:** All models support `.` notation consistently

### Code Quality
- Centralized expansion logic in utility function
- All 9 engines use same pattern
- Easy to maintain and test

---

## Related Documentation

- **Formula Parsing:** `py_parsnip/utils/time_series_utils.py:_parse_ts_formula()`
- **Dot Expansion:** `py_parsnip/utils/time_series_utils.py:_expand_dot_notation()`
- **Engine Support:** All engines using `fit_raw()` method

---

## Conclusion

All time series models in py-tidymodels now support the convenient `"target ~ ."` notation for automatically including all available columns as exogenous variables. This brings parity with R's tidymodels and improves user experience when working with datasets with many predictor variables.

**Total Impact:**
- 9 engines updated
- 1 utility function added
- Consistent behavior across all time series models
- Improved user experience and code readability
