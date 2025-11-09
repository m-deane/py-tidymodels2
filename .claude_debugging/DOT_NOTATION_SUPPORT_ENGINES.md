# Dot Notation Support Added to Time Series Engines

**Date:** 2025-11-09  
**Status:** COMPLETE  
**Files Modified:** 9 engine files

## Summary

Added support for "target ~ ." formula notation to all time series engines that use `_parse_ts_formula()`. This allows users to include all available columns as exogenous variables without explicitly naming them.

## Pattern Applied

For each engine, added two changes:

### Step 1: Import `_expand_dot_notation`
```python
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _expand_dot_notation
```

### Step 2: Expand dot notation after parsing formula
```python
# Parse formula to extract outcome and exogenous variables
outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

# Expand "." notation to all columns except outcome and date
exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, inferred_date_col)
```

## Files Updated

1. **statsmodels_arima.py** ✓
   - Import added on line 15
   - Expansion added on lines 80-81

2. **statsforecast_auto_arima.py** ✓
   - Import added on line 17
   - Expansion added on lines 93-94

3. **statsmodels_varmax.py** ✓
   - Import added on line 15
   - SPECIAL HANDLING: Manual expansion for multiple outcomes (lines 80-90)
   - Uses custom logic to exclude ALL outcome variables

4. **statsmodels_seasonal_reg.py** ✓
   - Import added on line 20
   - Expansion added on lines 69-70

5. **pmdarima_auto_arima.py** ✓
   - Import added on line 15
   - Expansion added on lines 92-93

6. **statsmodels_exp_smoothing.py** ✓
   - Import added on line 19
   - Expansion added on lines 72-73

7. **hybrid_prophet_boost.py** ✓
   - Import added on line 20
   - Expansion added on lines 166-167

8. **hybrid_arima_boost.py** ✓
   - Import added on line 20
   - Expansion added on lines 93-94
   - Removed old manual expansion code (lines 98-103)

9. **skforecast_recursive.py** ✓
   - Import added on line 12
   - Expansion added on lines 60-61
   - Removed old manual expansion code (lines 66-73)

## Special Cases

### VARMAX Engine
The VARMAX engine has special handling because it supports **multiple outcome variables**:

```python
# For formula: "y1 + y2 ~ ."
# Must exclude ALL outcomes (y1 AND y2), not just the first one

all_outcome_names = outcome_names  # List of all outcome variables
exclude_cols = set(all_outcome_names)
if date_col and date_col != '__index__':
    exclude_cols.add(date_col)

if exog_vars == ['.']:
    exog_vars = [col for col in data.columns if col not in exclude_cols]
```

### Engines with Manual Expansion (Now Removed)
Two engines previously had manual "." expansion:
- **hybrid_arima_boost.py** - Replaced with `_expand_dot_notation()`
- **skforecast_recursive.py** - Replaced with `_expand_dot_notation()`

## Testing

All engines now support formulas like:
- `"sales ~ ."` - Use all columns except sales and date
- `"y ~ . + I(x1*x2)"` - All columns plus interaction term
- `"y1 + y2 ~ ."` - For VARMAX: all columns except y1, y2, and date

## Benefits

1. **Consistency:** All time series engines now handle "." the same way
2. **Code Reuse:** Single utility function instead of duplicated logic
3. **Maintainability:** Changes to expansion logic happen in one place
4. **User Experience:** Simple syntax for including all predictors

## Related Files

- **py_parsnip/utils/time_series_utils.py** - Contains `_expand_dot_notation()` utility
- **py_parsnip/engines/prophet_engine.py** - Already supported via earlier fix

## Total Changes

- **9 files modified**
- **18 lines added** (2 per file: import + expansion call)
- **16 lines removed** (manual expansion code in 2 files)
- **Net change:** +2 lines

