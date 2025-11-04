# ARIMA Engine Update: Date Inference Utilities Integration

## Summary

Updated the statsmodels ARIMA engine (`py_parsnip/engines/statsmodels_arima.py`) to use the new shared date inference utilities from `py_parsnip/utils/time_series_utils.py`. This provides consistent date column handling across all time series models and improves support for exogenous variables.

## Changes Made

### 1. Import New Utilities

```python
from py_parsnip.utils import _infer_date_column, _parse_ts_formula
```

### 2. Updated `fit_raw()` Method

**Before:**
- Manual formula parsing with `split("~")` and `split("+")`
- Manual detection of datetime columns in predictors
- Manual separation of date column from exogenous variables

**After:**
- Uses `_infer_date_column()` for consistent date column detection
  - Supports priority-based detection: fit_date_col > spec_date_col > DatetimeIndex > auto-detect
  - Returns `'__index__'` for DatetimeIndex case
- Uses `_parse_ts_formula()` to extract outcome and exogenous variables
  - Automatically excludes date column from exogenous variables
  - Handles `~ 1` (intercept-only) and `~ .` (all predictors) cases
  - Returns empty list for exog_vars when only date is specified

**Key Logic:**
```python
# Use provided date_col if given (already inferred by ModelSpec),
# otherwise infer it here
if date_col is None:
    inferred_date_col = _infer_date_column(data, spec_date_col=None, fit_date_col=None)
else:
    # date_col was already inferred by ModelSpec, use it directly
    inferred_date_col = date_col

# Parse formula to extract outcome and exogenous variables
outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)
```

**Blueprint Storage:**
- Changed from `predictor_names` to `exog_vars` (excludes date column)
- Stores `date_col` as either column name or `'__index__'`

### 3. Updated `predict_raw()` Method

**Before:**
- Extracted `predictor_names` from fit_data
- Manually filtered out date column to get exogenous variables
- Date index extraction was ad-hoc

**After:**
- Extracts `exog_vars` directly from fit_data (already excludes date)
- Uses `_infer_date_column()` with `fit_date_col` priority for consistency
- Proper date index extraction for both regular columns and DatetimeIndex

**Key Logic:**
```python
exog_vars = fit.fit_data.get("exog_vars", [])
fit_date_col = fit.fit_data.get("date_col")

# Infer date column from new_data (prioritize fit_date_col for consistency)
inferred_date_col = _infer_date_column(
    new_data,
    spec_date_col=None,
    fit_date_col=fit_date_col
)

# Extract date index
if inferred_date_col == '__index__':
    date_index = new_data.index
else:
    date_index = new_data[inferred_date_col]
```

### 4. Updated `extract_outputs()` Method

**Before:**
- Manual search for date columns in test data

**After:**
- Uses `date_col` from `fit_data` for consistent date extraction
- Handles `'__index__'` case properly

**Key Logic:**
```python
date_col = fit.fit_data.get("date_col")
if date_col == '__index__':
    test_dates = test_data.index.values
elif date_col and date_col in test_data.columns:
    test_dates = test_data[date_col].values
else:
    test_dates = np.arange(len(test_actuals))  # Fallback
```

## Benefits

1. **Consistency**: Uses same date detection logic as other time series models (Prophet, seasonal_reg, etc.)
2. **Robustness**: Handles DatetimeIndex case properly with `'__index__'` sentinel value
3. **Clarity**: Separates date column from exogenous variables explicitly
4. **Maintainability**: Shared utilities reduce code duplication and bugs
5. **Error Handling**: Better validation of exogenous variables during prediction

## Supported Use Cases

### 1. Date Column with No Exogenous Variables
```python
data = pd.DataFrame({'date': dates, 'value': values})
fit = arima_reg().fit(data, formula='value ~ date')
```
- `date_col = 'date'`
- `exog_vars = []`

### 2. Date Column with Exogenous Variables
```python
data = pd.DataFrame({'date': dates, 'value': values, 'x1': x1, 'x2': x2})
fit = arima_reg().fit(data, formula='value ~ date + x1 + x2')
```
- `date_col = 'date'`
- `exog_vars = ['x1', 'x2']`

### 3. DatetimeIndex (No Date Column)
```python
data = pd.DataFrame({'value': values}, index=dates)
fit = arima_reg().fit(data, formula='value ~ 1')
```
- `date_col = '__index__'`
- `exog_vars = []`

### 4. DatetimeIndex with Exogenous Variables
```python
data = pd.DataFrame({'value': values, 'x1': x1, 'x2': x2}, index=dates)
fit = arima_reg().fit(data, formula='value ~ x1 + x2')
```
- `date_col = '__index__'`
- `exog_vars = ['x1', 'x2']`

## Backward Compatibility

✅ **All existing tests pass** - no breaking changes to API
✅ **ARIMA boost tests pass** - hybrid models continue to work
✅ **Prophet tests pass** - other time series models unaffected
✅ **Linear regression tests pass** - standard models unaffected

## Testing

Comprehensive testing confirms:
- ✅ Date column detection (regular column)
- ✅ DatetimeIndex detection (`'__index__'`)
- ✅ Exogenous variable handling
- ✅ Prediction with proper date indexing
- ✅ Error handling for missing exogenous variables
- ✅ Confidence interval predictions
- ✅ Extract outputs with proper metadata
- ✅ All existing test suites pass

## Files Modified

- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/engines/statsmodels_arima.py`

## Related Files (No Changes)

- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/utils/time_series_utils.py` (used by this engine)
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/utils/__init__.py` (exports utilities)

## Future Work

Consider updating other engines to use these utilities:
- `pmdarima_auto_arima.py` - Auto ARIMA engine (currently has numpy compatibility issue)
- Any future time series engines

## Notes

- The `date_col` parameter in `fit_raw()` is already inferred by `ModelSpec.fit()` before being passed to the engine
- The engine must handle both cases: when `date_col` is provided (already inferred) and when it's None (must infer)
- The `'__index__'` sentinel value is used consistently to indicate DatetimeIndex usage
- Exogenous variables are now explicitly separated from the date column in fit_data storage
