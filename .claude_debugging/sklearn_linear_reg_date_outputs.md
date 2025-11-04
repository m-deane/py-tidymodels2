# sklearn Linear Regression Date Column Support

**Date:** 2025-11-04
**File:** `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/engines/sklearn_linear_reg.py`
**Tests:** `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/tests/test_parsnip/test_linear_reg_date_outputs.py`

## Summary

Enhanced the sklearn linear regression engine to automatically add a `date` column to the outputs DataFrame when working with time series data. This brings sklearn linear regression in line with time series models (Prophet, ARIMA) that already include date columns in their outputs.

## Changes Made

### 1. Updated `fit()` Method Signature

Added optional `original_training_data` parameter to store the unpreprocessed training data:

```python
def fit(
    self,
    spec: ModelSpec,
    molded: MoldedData,
    original_training_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
```

The `original_training_data` is stored in the `fit_data` dict for later use in `extract_outputs()`.

### 2. Enhanced `extract_outputs()` Method

Added date column extraction logic after creating the outputs DataFrame:

```python
# Add date column if available in original data
if not outputs.empty:
    try:
        from py_parsnip.utils.time_series_utils import _infer_date_column

        # For training data
        original_training_data = fit.fit_data.get("original_training_data")
        if original_training_data is not None:
            try:
                date_col = _infer_date_column(
                    original_training_data,
                    spec_date_col=None,
                    fit_date_col=None
                )

                # Extract training dates
                if date_col == '__index__':
                    train_dates = original_training_data.index.values
                else:
                    train_dates = original_training_data[date_col].values

                # For test data (if evaluated)
                test_dates = None
                original_test_data = fit.evaluation_data.get("original_test_data")
                if original_test_data is not None:
                    # Extract test dates using same date_col
                    ...

                # Combine dates based on split
                combined_dates = []
                train_count = (outputs['split'] == 'train').sum()
                test_count = (outputs['split'] == 'test').sum()

                if train_count > 0:
                    combined_dates.extend(train_dates[:train_count])
                if test_count > 0 and test_dates is not None:
                    combined_dates.extend(test_dates[:test_count])

                # Add date column as first column
                if len(combined_dates) == len(outputs):
                    outputs.insert(0, 'date', combined_dates)

            except ValueError:
                # No datetime columns - skip date column
                pass
    except ImportError:
        # time_series_utils not available - skip date column
        pass
```

### 3. Added Import

```python
from typing import Dict, Any, Literal, Optional  # Added Optional
```

## Features

### Automatic Date Detection

The engine uses `_infer_date_column()` from `py_parsnip.utils.time_series_utils` to automatically detect date columns with the following priority:

1. **Explicit date column** in DataFrame columns
2. **DatetimeIndex** (returns `'__index__'`)
3. **Auto-detect** single datetime column

### Supports Multiple Date Formats

- **Date column**: `pd.DataFrame({'date': pd.date_range(...), ...})`
- **DatetimeIndex**: `pd.DataFrame({...}, index=pd.date_range(...))`

### Backward Compatibility

The implementation maintains full backward compatibility:

- **No datetime data**: Date column is not added (existing behavior)
- **No original_data provided**: Date column is not added (existing behavior)
- **All existing tests pass**: 26/26 original tests pass without modification

### Output Structure

When date column is present, the outputs DataFrame has this structure:

```python
outputs.columns:
[
    'date',              # NEW - First column (when available)
    'actuals',
    'fitted',
    'forecast',
    'residuals',
    'split',
    'model',
    'model_group_name',
    'group'
]
```

## Usage Examples

### Basic Usage with Date Column

```python
import pandas as pd
import numpy as np
from py_parsnip.models.linear_reg import linear_reg

# Create time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'date': dates,
    'x1': np.random.randn(100),
    'y': np.random.randn(100)
})

# Split train/test
train = df.iloc[:80]
test = df.iloc[80:]

# Fit with original_training_data
spec = linear_reg()
fit = spec.fit(train, 'y ~ x1', original_training_data=train)

# Evaluate with original_test_data
fit = fit.evaluate(test, original_test_data=test)

# Extract outputs with date column
outputs, coefs, stats = fit.extract_outputs()
print(outputs.columns)
# ['date', 'actuals', 'fitted', 'forecast', 'residuals', 'split', ...]
```

### Usage with DatetimeIndex

```python
# Create data with DatetimeIndex
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'x1': np.random.randn(100),
    'y': np.random.randn(100)
}, index=dates)

train = df.iloc[:80]
test = df.iloc[80:]

# Fit and evaluate
spec = linear_reg()
fit = spec.fit(train, 'y ~ x1', original_training_data=train)
fit = fit.evaluate(test, original_test_data=test)

# Extract outputs with date column (from index)
outputs, _, _ = fit.extract_outputs()
assert 'date' in outputs.columns
```

### Backward Compatibility - No Date Column

```python
# Non-time series data (no datetime columns)
df = pd.DataFrame({
    'x1': np.random.randn(100),
    'y': np.random.randn(100)
})

train = df.iloc[:80]
test = df.iloc[80:]

# Fit without original_training_data
spec = linear_reg()
fit = spec.fit(train, 'y ~ x1')
fit = fit.evaluate(test)

# Extract outputs WITHOUT date column (backward compatible)
outputs, _, _ = fit.extract_outputs()
assert 'date' not in outputs.columns
```

## Integration with ModelSpec

The `ModelSpec.fit()` method already supports passing `original_training_data` to engine's `fit()` method:

```python
# In ModelSpec.fit() (lines 172-178)
import inspect
fit_signature = inspect.signature(engine.fit)
accepts_original_data = 'original_training_data' in fit_signature.parameters

if accepts_original_data and original_training_data is not None:
    fit_data = engine.fit(self, molded, original_training_data=original_training_data)
else:
    fit_data = engine.fit(self, molded)
```

Similarly, `ModelFit.evaluate()` already stores `original_test_data` in `evaluation_data`:

```python
# In ModelFit.evaluate() (line 342)
if original_test_data is not None:
    self.evaluation_data["original_test_data"] = original_test_data
```

## Tests

Created comprehensive test suite in `tests/test_parsnip/test_linear_reg_date_outputs.py`:

- `test_date_column_with_explicit_date_col` - Date column in DataFrame
- `test_date_column_with_datetime_index` - DatetimeIndex support
- `test_date_ranges_match_splits` - Train/test date alignment
- `test_backward_compatibility_no_datetime` - No date for non-TS data
- `test_backward_compatibility_no_original_data` - No date without original_data
- `test_date_column_train_only` - Train-only (no evaluate)
- `test_date_column_values_align_with_actuals` - Date/actual alignment
- `test_outputs_structure_with_date` - Column order verification
- `test_multiple_datetime_columns_raises_error` - Multiple datetime handling

**All 35 tests pass** (26 original + 9 new).

## Benefits

1. **Consistency**: sklearn linear regression now matches time series models (Prophet, ARIMA) in output format
2. **Time series visualization**: Date column enables plotting forecasts with matplotlib/plotly
3. **Backward compatibility**: Existing code continues to work without modification
4. **Flexible**: Works with both date columns and DatetimeIndex
5. **Robust**: Graceful error handling for edge cases (multiple datetime columns, missing data)

## Future Work

This pattern can be applied to other sklearn-based engines:

- `sklearn_random_forest.py`
- `sklearn_decision_tree.py`
- `sklearn_svm_rbf.py`
- `sklearn_svm_linear.py`
- `sklearn_nearest_neighbor.py`
- `sklearn_mlp.py`
- `xgboost_boost_tree.py`
- `lightgbm_boost_tree.py`
- `catboost_boost_tree.py`

The same pattern (add `original_training_data` parameter, use `_infer_date_column()`, add date to outputs) can be replicated across all engines.
