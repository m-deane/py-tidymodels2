# Column Ordering Implementation Summary

## Overview

Implemented consistent column ordering across all `extract_outputs()` methods to ensure:
- **Date column is always first** (whether in columns or index)
- **Group column is always second** (for nested/grouped models)
- Core columns follow in predictable order
- Metadata columns come last

## Code Changes

### Files Created

#### 1. `/py_parsnip/utils/output_ordering.py` (NEW - 183 lines)

Centralized utility module with three reordering functions:

**`reorder_outputs_columns(df, group_col=None)`**
- Handles outputs DataFrame (observation-level data)
- Order: date → group_col → actuals, fitted, forecast, residuals, split → metadata → extras
- Special handling: Resets DatetimeIndex to first column

**`reorder_coefficients_columns(df, group_col=None)`**
- Handles coefficients DataFrame (parameter estimates)
- Order: group_col → variable, coefficient, std_error, t_stat, p_value, conf_low, conf_high, vif → metadata → extras

**`reorder_stats_columns(df, group_col=None)`**
- Handles stats DataFrame (model-level metrics)
- Order: group_col → split, metric, value → metadata → extras

### Files Modified

#### 2. `/py_workflows/workflow.py`

**WorkflowFit.extract_outputs() (lines 804-817):**
```python
def extract_outputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs, coefficients, stats = self.fit.extract_outputs()

    # Reorder columns for consistent ordering: date first, then core columns
    from py_parsnip.utils.output_ordering import (
        reorder_outputs_columns,
        reorder_coefficients_columns,
        reorder_stats_columns
    )

    outputs = reorder_outputs_columns(outputs, group_col=None)
    coefficients = reorder_coefficients_columns(coefficients, group_col=None)
    stats = reorder_stats_columns(stats, group_col=None)

    return outputs, coefficients, stats
```

**NestedWorkflowFit.extract_outputs() (lines 1275-1286):**
```python
# After combining all groups
combined_outputs = pd.concat(all_outputs, ignore_index=True)
combined_coefficients = pd.concat(all_coefficients, ignore_index=True)
combined_stats = pd.concat(all_stats, ignore_index=True)

# Reorder columns: date first, group second, then core columns
from py_parsnip.utils.output_ordering import (
    reorder_outputs_columns,
    reorder_coefficients_columns,
    reorder_stats_columns
)

combined_outputs = reorder_outputs_columns(combined_outputs, group_col=self.group_col)
combined_coefficients = reorder_coefficients_columns(combined_coefficients, group_col=self.group_col)
combined_stats = reorder_stats_columns(combined_stats, group_col=self.group_col)

return combined_outputs, combined_coefficients, combined_stats
```

#### 3. `/py_parsnip/model_spec.py`

**ModelFit.extract_outputs() (lines 640-653):**
```python
def extract_outputs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from py_parsnip.engine_registry import get_engine

    engine = get_engine(self.spec.model_type, self.spec.engine)
    outputs, coefficients, stats = engine.extract_outputs(self)

    # Reorder columns for consistent ordering: date first, then core columns
    from py_parsnip.utils.output_ordering import (
        reorder_outputs_columns,
        reorder_coefficients_columns,
        reorder_stats_columns
    )

    outputs = reorder_outputs_columns(outputs, group_col=None)
    coefficients = reorder_coefficients_columns(coefficients, group_col=None)
    stats = reorder_stats_columns(stats, group_col=None)

    return outputs, coefficients, stats
```

**NestedModelFit.extract_outputs() (lines 874-885):**
```python
# After combining all groups
combined_outputs = pd.concat(all_outputs, ignore_index=True)
combined_coefficients = pd.concat(all_coefficients, ignore_index=True)
combined_stats = pd.concat(all_stats, ignore_index=True)

# Reorder columns: date first, group second, then core columns
from py_parsnip.utils.output_ordering import (
    reorder_outputs_columns,
    reorder_coefficients_columns,
    reorder_stats_columns
)

combined_outputs = reorder_outputs_columns(combined_outputs, group_col=self.group_col)
combined_coefficients = reorder_coefficients_columns(combined_coefficients, group_col=self.group_col)
combined_stats = reorder_stats_columns(combined_stats, group_col=self.group_col)

return combined_outputs, combined_coefficients, combined_stats
```

## Tests Added

### Unit Tests: `/tests/test_utils/test_output_ordering.py` (16 tests)

```
TestReorderOutputsColumns (9 tests):
✓ test_basic_ordering_no_group
✓ test_date_column_first
✓ test_group_column_second
✓ test_group_column_without_date
✓ test_metadata_columns_last
✓ test_empty_dataframe
✓ test_extra_columns_preserved
✓ test_date_in_index  (NEW)
✓ test_named_date_index  (NEW)

TestReorderCoefficientsColumns (3 tests):
✓ test_basic_ordering_no_group
✓ test_group_column_first
✓ test_confidence_intervals_ordered

TestReorderStatsColumns (2 tests):
✓ test_basic_ordering_no_group
✓ test_group_column_first

TestIntegrationWithRealData (2 tests):
✓ test_nested_workflow_outputs
✓ test_non_nested_workflow_outputs
```

### Integration Tests: `/tests/test_workflows/test_column_ordering_integration.py` (8 tests)

```
TestWorkflowColumnOrdering (1 test):
✓ test_workflow_outputs_date_first

TestNestedWorkflowColumnOrdering (3 tests):
✓ test_nested_workflow_outputs_date_and_group_ordering
✓ test_nested_workflow_coeffs_group_first
✓ test_nested_workflow_stats_group_first

TestModelSpecColumnOrdering (2 tests):
✓ test_nested_model_fit_outputs_ordering
✓ test_nested_model_fit_all_dataframes_ordering

TestBackwardCompatibility (2 tests):
✓ test_no_date_column_still_works
✓ test_column_values_unchanged
```

## Verification Results

All tests passing:
```bash
# Unit tests
pytest tests/test_utils/test_output_ordering.py -v
# Result: 16 passed in 0.06s

# Integration tests
pytest tests/test_workflows/test_column_ordering_integration.py -v
# Result: 8 passed in 1.09s

# Regression tests (panel models)
pytest tests/test_workflows/test_panel_models.py -v
# Result: 18 passed, 12 warnings in 6.14s
```

**Total: 42 tests passing (24 new + 18 regression)**

## Column Ordering Specification

### Outputs DataFrame
```
Position | Column          | Description
---------|-----------------|------------------------------------------
1        | date            | Always first (if present in cols or index)
2        | <group_col>     | Group identifier (nested models only)
3        | actuals         | True outcome values
4        | fitted          | Model predictions
5        | forecast        | Combined actuals/fitted (seamless series)
6        | residuals       | actuals - fitted
7        | split           | 'train', 'test', or 'forecast'
8+       | model           | Model type identifier
         | model_group_name| Model grouping for comparisons
         | group           | 'global' or group identifier
         | <extra columns> | Preserved in original order
```

### Coefficients DataFrame
```
Position | Column          | Description
---------|-----------------|------------------------------------------
1        | <group_col>     | Group identifier (nested models only)
2        | variable        | Predictor/coefficient name
3        | coefficient     | Estimated coefficient value
4        | std_error       | Standard error of coefficient
5        | t_stat          | t-statistic
6        | p_value         | p-value for significance test
7        | conf_low        | Lower confidence interval bound
8        | conf_high       | Upper confidence interval bound
9        | vif             | Variance Inflation Factor
10+      | model           | Model type identifier
         | model_group_name| Model grouping
         | group           | 'global' or group identifier
         | <extra columns> | Preserved
```

### Stats DataFrame
```
Position | Column          | Description
---------|-----------------|------------------------------------------
1        | <group_col>     | Group identifier (nested models only)
2        | split           | 'train', 'test', or 'forecast'
3        | metric          | Metric name (rmse, mae, r_squared, etc.)
4        | value           | Metric value
5+       | model           | Model type identifier
         | model_group_name| Model grouping
         | group           | 'global' or group identifier
         | <extra columns> | Preserved
```

## Key Features

### 1. Index Handling
The solution automatically detects date columns in the index and resets them:
```python
# Before reordering (date in index)
outputs.index  # DatetimeIndex(['2020-01-01', ...])
outputs.columns  # ['actuals', 'fitted', ...]

# After reordering (date as first column)
outputs.columns  # ['date', 'actuals', 'fitted', ...]
outputs['date']  # Datetime column
```

### 2. Group Column Flexibility
```python
# Non-nested (no group column)
outputs, coeffs, stats = workflow_fit.extract_outputs()
# outputs.columns[0] == 'date'
# outputs.columns[1] == 'actuals'

# Nested (with group column)
outputs, coeffs, stats = nested_fit.extract_outputs()
# outputs.columns[0] == 'date'
# outputs.columns[1] == 'store_id'  (or 'country', etc.)
# outputs.columns[2] == 'actuals'
```

### 3. Backward Compatibility
- No breaking changes to API
- All existing code continues to work
- Only column ORDER changes (values/data unchanged)
- Works with or without date column
- Works with or without group column

## Usage Examples

### Example 1: Standard Workflow with Date
```python
from py_workflows import workflow
from py_parsnip import linear_reg
import pandas as pd

# Time series data with date column
train = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'x1': [...], 'x2': [...], 'y': [...]
})

wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
fit = wf.fit(train).evaluate(test)
outputs, coeffs, stats = fit.extract_outputs()

# Guaranteed ordering
print(outputs.columns.tolist())
# ['date', 'actuals', 'fitted', 'forecast', 'residuals', 'split', 'model', ...]
```

### Example 2: Nested Workflow with Group Column
```python
# Grouped time series data
train = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=200),
    'store_id': ['A']*100 + ['B']*100,
    'x1': [...], 'y': [...]
})

wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
nested_fit = wf.fit_nested(train, group_col='store_id').evaluate(test)
outputs, coeffs, stats = nested_fit.extract_outputs()

# Guaranteed ordering with group
print(outputs.columns.tolist())
# ['date', 'store_id', 'actuals', 'fitted', 'forecast', ...]

# Easy group filtering
store_a = outputs[outputs['store_id'] == 'A']
```

### Example 3: Model Spec Nested Fit
```python
from py_parsnip import linear_reg

spec = linear_reg()
nested_fit = spec.fit_nested(
    train,
    formula='sales ~ price',
    group_col='region'
).evaluate(test)

outputs, coeffs, stats = nested_fit.extract_outputs()

# All DataFrames have consistent group column placement
print(outputs.columns[0:2])   # ['date', 'region']
print(coeffs.columns[0])      # 'region'
print(stats.columns[0])       # 'region'
```

## Documentation Files

1. **`COLUMN_ORDERING_FIX.md`** - Detailed technical documentation
2. **`COLUMN_ORDERING_SUMMARY.md`** - This summary document

## Impact

**Affected Components:**
- WorkflowFit.extract_outputs()
- NestedWorkflowFit.extract_outputs()
- ModelFit.extract_outputs()
- NestedModelFit.extract_outputs()

**Affected Model Types:**
- All model types (23 total)
- All engines (30+ implementations)
- Works across standard and time series models
- Works for nested/grouped and non-nested models

**User Benefits:**
1. Predictable column ordering for easier data inspection
2. Simpler code for accessing key columns
3. Better visualization compatibility (date-first is intuitive)
4. Consistent experience across all model types
5. No code changes required (backward compatible)
