# Time Series Utilities Implementation

**Created:** 2025-11-04
**Status:** Complete - All 41 tests passing
**Module:** `py_parsnip/utils/time_series_utils.py`

## Overview

Implemented a comprehensive utility module for consistent date column handling and formula parsing across all time series model engines. This eliminates code duplication and provides a standardized interface for time series operations.

## Implementation Summary

### Files Created

1. **`py_parsnip/utils/time_series_utils.py`** - Core utility functions (370 lines)
2. **`py_parsnip/utils/__init__.py`** - Module exports
3. **`tests/test_parsnip/test_time_series_utils.py`** - Comprehensive test suite (351 lines, 41 tests)

### Functions Implemented

#### 1. `_infer_date_column(data, spec_date_col=None, fit_date_col=None) -> str`

**Purpose:** Intelligently infer the date column from a DataFrame using priority-based detection.

**Priority Order:**
1. `fit_date_col` - Date column from fitted model (used during prediction for consistency)
2. `spec_date_col` - Date column from model specification
3. DatetimeIndex - Auto-detect if data has DatetimeIndex
4. Auto-detect - Find single datetime column in DataFrame

**Key Features:**
- Returns `'__index__'` if using DatetimeIndex
- Clear error messages when multiple datetime columns exist
- Validates column existence and dtype
- Ensures consistency between fit and predict phases

**Example Usage:**
```python
import pandas as pd
from py_parsnip.utils import _infer_date_column

# Explicit specification
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=10),
    'value': range(10)
})
date_col = _infer_date_column(df, spec_date_col='date')  # Returns 'date'

# Auto-detect DatetimeIndex
df_indexed = df.set_index('date')
date_col = _infer_date_column(df_indexed)  # Returns '__index__'

# Auto-detect single datetime column
date_col = _infer_date_column(df)  # Returns 'date'

# Prediction consistency
train_date_col = _infer_date_column(train_df)
pred_date_col = _infer_date_column(pred_df, fit_date_col=train_date_col)
```

**Test Coverage:** 11 tests covering all priority paths and error cases

#### 2. `_parse_ts_formula(formula: str, date_col: str) -> Tuple[str, List[str]]`

**Purpose:** Parse time series formula to extract outcome and exogenous variables, automatically excluding the date column.

**Key Features:**
- Extracts outcome variable (left side of ~)
- Extracts exogenous variables (right side of ~), excluding date column
- Handles special cases:
  - `"target ~ 1"` - Intercept only (no regressors)
  - `"target ~ ."` - All columns except outcome
  - `"y1 + y2 ~ x1 + x2"` - Multiple outcomes (VARMAX)
- Date column automatically excluded from exogenous variables
- Comprehensive error handling with clear messages

**Example Usage:**
```python
from py_parsnip.utils import _parse_ts_formula

# Standard formula
outcome, exog = _parse_ts_formula("sales ~ lag1 + lag2 + date", "date")
# Returns: ('sales', ['lag1', 'lag2'])

# Intercept only
outcome, exog = _parse_ts_formula("sales ~ 1", "date")
# Returns: ('sales', [])

# Date only (becomes no exogenous)
outcome, exog = _parse_ts_formula("sales ~ date", "date")
# Returns: ('sales', [])

# Multiple outcomes
outcome, exog = _parse_ts_formula("y1 + y2 ~ x1 + x2 + date", "date")
# Returns: ('y1 + y2', ['x1', 'x2'])

# DatetimeIndex case
outcome, exog = _parse_ts_formula("y ~ lag1 + lag2", "__index__")
# Returns: ('y', ['lag1', 'lag2'])
```

**Test Coverage:** 15 tests covering all formula patterns and error cases

#### 3. `_validate_frequency(series: pd.Series, require_freq=True, infer_freq=True) -> pd.Series`

**Purpose:** Validate and optionally infer frequency for time series with DatetimeIndex.

**Key Features:**
- Returns series as-is if frequency already set
- Infers frequency using `pd.infer_freq()` if missing
- Fallback to most common difference (only if all differences are equal)
- Configurable requirement and inference behavior
- Preserves series name and values

**Example Usage:**
```python
import pandas as pd
from py_parsnip.utils import _validate_frequency

# Series with explicit frequency
dates = pd.date_range('2020-01-01', periods=10, freq='D')
s = pd.Series(range(10), index=dates)
validated = _validate_frequency(s)  # Returns series with freq='D'

# Series without frequency - will infer
dates_no_freq = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10))
dates_no_freq.freq = None
s_no_freq = pd.Series(range(10), index=dates_no_freq)
validated = _validate_frequency(s_no_freq)  # Infers and sets frequency

# Optional frequency requirement
irregular_dates = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-08'])
s_irregular = pd.Series([1, 2, 3], index=irregular_dates)
validated = _validate_frequency(s_irregular, require_freq=False)  # Returns as-is
```

**Test Coverage:** 15 tests covering frequency inference and validation scenarios

## Test Suite Details

### Test Classes

1. **TestInferDateColumn** (11 tests)
   - Explicit date column specification
   - Priority handling (fit_date_col vs spec_date_col)
   - DatetimeIndex detection
   - Auto-detection of single datetime column
   - Error handling (multiple columns, no columns, invalid columns)

2. **TestParseTsFormula** (15 tests)
   - Standard formulas with date columns
   - Intercept-only formulas
   - Date-only formulas
   - All predictors (.) syntax
   - Multiple outcomes (VARMAX)
   - Whitespace handling
   - Complex expressions and interactions
   - Error cases (missing ~, empty sides)

3. **TestValidateFrequency** (11 tests)
   - Explicit frequency preservation
   - Frequency inference (daily, monthly, hourly)
   - Fallback to common difference
   - Optional frequency requirement
   - Error handling (irregular frequencies)
   - Series name and value preservation

4. **TestIntegrationScenarios** (4 tests)
   - DatetimeIndex workflow
   - Explicit date column workflow
   - Multiple datetime columns handling
   - Fit/predict consistency

### Test Results

```bash
$ python -m pytest tests/test_parsnip/test_time_series_utils.py -v

============================== 41 passed in 0.22s ==============================
```

**All 41 tests passing** with comprehensive coverage of:
- Normal operation paths
- Edge cases
- Error conditions
- Integration scenarios

## Design Rationale

### 1. Priority-Based Date Column Detection

The `_infer_date_column()` function uses a priority system to handle different scenarios:

- **fit_date_col priority**: Ensures consistency between fit and predict phases
- **spec_date_col**: Allows explicit user control when needed
- **DatetimeIndex detection**: Automatically handles index-based time series
- **Auto-detection**: Convenient default for single datetime column

This design eliminates ambiguity and provides clear, predictable behavior.

### 2. Automatic Date Column Exclusion

The `_parse_ts_formula()` function automatically excludes the date column from exogenous variables because:

- Date serves as the time index, not as a predictor
- Including date as predictor causes issues in most time series models
- Reduces user error and code repetition across engines
- Maintains consistency with R tidymodels conventions

### 3. Flexible Frequency Validation

The `_validate_frequency()` function provides three modes:

- **Strict mode** (`require_freq=True`): Raises error if frequency cannot be determined
- **Lenient mode** (`require_freq=False`): Returns series as-is if inference fails
- **No inference** (`infer_freq=False`): Only validates existing frequency

This flexibility accommodates different engine requirements (e.g., skforecast requires frequency, others don't).

### 4. Consistent Error Messages

All functions provide clear, actionable error messages:

```python
# Example error from _infer_date_column
ValueError: Multiple datetime columns found: ['date1', 'date2'].
            Please specify date_col explicitly.

# Example error from _parse_ts_formula
ValueError: Invalid formula: 'sales'. Formula must contain '~' separator.

# Example error from _validate_frequency
ValueError: Could not infer frequency for DatetimeIndex.
            Please ensure the index has regular intervals or specify frequency explicitly.
            Index range: 2020-01-01 to 2020-12-31, Length: 365
```

## Usage in Time Series Engines

These utilities are designed to be used across all time series engines:

### Example: ARIMA Engine

```python
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _validate_frequency

def fit_raw(self, spec, data, formula):
    # Infer date column
    date_col = _infer_date_column(
        data,
        spec_date_col=spec.args.get('date_col'),
        fit_date_col=None  # None during fit
    )

    # Parse formula
    outcome, exog_vars = _parse_ts_formula(formula, date_col)

    # Prepare time series
    if date_col == '__index__':
        y = data[outcome]
    else:
        y = data.set_index(date_col)[outcome]

    # Validate frequency
    y = _validate_frequency(y, require_freq=True)

    # Fit ARIMA model
    # ...
```

### Example: Recursive Forecasting Engine

```python
from py_parsnip.utils import _infer_date_column, _validate_frequency

def fit_raw(self, spec, data, formula):
    # Infer date column
    date_col = _infer_date_column(data, spec_date_col=spec.args.get('date_col'))

    # Set as index
    if date_col != '__index__':
        data = data.set_index(date_col)

    # Validate frequency (required by skforecast)
    y = data[outcome]
    y = _validate_frequency(y, require_freq=True)

    # Fit recursive model
    # ...
```

## Integration with Existing Code

The utilities are backward compatible and can be gradually adopted:

1. **Existing engines continue to work** - No breaking changes
2. **New engines use utilities** - Standardized implementation
3. **Gradual migration** - Existing engines can be updated incrementally

## Documentation

All functions include comprehensive docstrings with:

- Purpose and behavior description
- Parameter descriptions with types
- Return value descriptions
- Raises section for exceptions
- Multiple usage examples
- Integration with other utilities

## Future Enhancements

Potential additions to the utilities module:

1. **`_prepare_time_series_data()`** - Combined function for common workflow
2. **`_validate_forecast_horizon()`** - Validate forecast horizon parameters
3. **`_create_lagged_features()`** - Generate lagged features for ML models
4. **`_split_train_test_ts()`** - Time series-aware train/test splitting

## Summary

The time series utilities module provides:

- **Consistent interface** across all time series engines
- **Comprehensive testing** with 41 passing tests
- **Clear error messages** for debugging
- **Flexible configuration** for different engine requirements
- **Production-ready code** with full type hints and documentation

**Files:**
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/utils/time_series_utils.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/utils/__init__.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/tests/test_parsnip/test_time_series_utils.py`

**Test Command:**
```bash
cd /Users/matthewdeane/Documents/Data\ Science/python/_projects/py-tidymodels
source py-tidymodels2/bin/activate
python -m pytest tests/test_parsnip/test_time_series_utils.py -v
```

**Import:**
```python
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _validate_frequency
```
