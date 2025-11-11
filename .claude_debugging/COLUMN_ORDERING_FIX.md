# Column Ordering Fix for extract_outputs()

**Date:** 2025-11-10
**Status:** Complete
**Tests Added:** 24 tests (16 unit + 8 integration)

## Problem Statement

The `extract_outputs()` methods across workflows and model specs did not guarantee consistent column ordering. Users reported that:
- Date column was sometimes last instead of first
- Group column (for nested/grouped models) was not consistently positioned
- Core columns (actuals, fitted, forecast, residuals) had varying order

This made it harder to:
- Visually inspect outputs in notebooks
- Write consistent code that references columns by position
- Compare outputs across different model types

## Solution

Created a centralized column ordering utility (`py_parsnip/utils/output_ordering.py`) that enforces consistent ordering:

### Outputs DataFrame Ordering
1. **`date`** - Always first (whether column or index)
2. **Group column** - Always second (e.g., `store_id`, `country`)
3. **Core columns** - Fixed order: `actuals`, `fitted`, `forecast`, `residuals`, `split`
4. **Metadata columns** - `model`, `model_group_name`, `group`
5. **Extra columns** - Preserved in original order

### Coefficients DataFrame Ordering
1. **Group column** - First if present
2. **Core coefficient columns** - `variable`, `coefficient`, `std_error`, `t_stat`, `p_value`, `conf_low`, `conf_high`, `vif`
3. **Metadata columns** - `model`, `model_group_name`, `group`
4. **Extra columns** - Preserved

### Stats DataFrame Ordering
1. **Group column** - First if present
2. **Core stats columns** - `split`, `metric`, `value`
3. **Metadata columns** - `model`, `model_group_name`, `group`
4. **Extra columns** - Preserved

## Implementation Details

### Files Modified

1. **`py_parsnip/utils/output_ordering.py`** (NEW)
   - `reorder_outputs_columns()` - Handles outputs DataFrame
   - `reorder_coefficients_columns()` - Handles coefficients DataFrame
   - `reorder_stats_columns()` - Handles stats DataFrame
   - Special handling: Resets date from index to first column

2. **`py_workflows/workflow.py`**
   - Updated `WorkflowFit.extract_outputs()` to apply reordering
   - Updated `NestedWorkflowFit.extract_outputs()` to apply reordering with group column

3. **`py_parsnip/model_spec.py`**
   - Updated `ModelFit.extract_outputs()` to apply reordering
   - Updated `NestedModelFit.extract_outputs()` to apply reordering with group column

### Key Features

**Index Handling:**
The solution automatically detects and handles date columns in the index:
```python
# Engines may set date as index (e.g., sklearn_linear_reg)
if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()  # Move to column
    # Date becomes first column
```

**Group Column Flexibility:**
```python
# Without group column (standard workflow)
outputs = reorder_outputs_columns(outputs, group_col=None)
# Result: date, actuals, fitted, ...

# With group column (nested workflow)
outputs = reorder_outputs_columns(outputs, group_col='store_id')
# Result: date, store_id, actuals, fitted, ...
```

**Backward Compatibility:**
- All existing functionality preserved
- No breaking changes to API
- Only column order changes (values unchanged)

## Tests Added

### Unit Tests (`tests/test_utils/test_output_ordering.py`) - 16 tests

**TestReorderOutputsColumns (9 tests):**
- Basic column ordering without group
- Date column always first
- Group column always second (after date)
- Group column first when no date
- Metadata columns last
- Empty DataFrame handling
- Extra columns preserved
- Date in index handling
- Named date index handling

**TestReorderCoefficientsColumns (3 tests):**
- Basic coefficient ordering
- Group column first
- Confidence intervals ordered

**TestReorderStatsColumns (2 tests):**
- Basic stats ordering
- Group column first

**TestIntegrationWithRealData (2 tests):**
- Realistic nested workflow outputs
- Realistic non-nested workflow outputs

### Integration Tests (`tests/test_workflows/test_column_ordering_integration.py`) - 8 tests

**TestWorkflowColumnOrdering (1 test):**
- Standard workflow date ordering

**TestNestedWorkflowColumnOrdering (3 tests):**
- Nested workflow date and group ordering
- Nested workflow coefficients group first
- Nested workflow stats group first

**TestModelSpecColumnOrdering (2 tests):**
- Nested model fit outputs ordering
- All three DataFrames consistent ordering

**TestBackwardCompatibility (2 tests):**
- No date column still works
- Column values unchanged

## Usage Examples

### Standard Workflow
```python
from py_workflows import workflow
from py_parsnip import linear_reg

wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
fit = wf.fit(train).evaluate(test)
outputs, coeffs, stats = fit.extract_outputs()

# Guaranteed ordering:
# outputs.columns[0] == 'date'  (if present)
# outputs.columns[1] == 'actuals'
# outputs.columns[2] == 'fitted'
# outputs.columns[3] == 'forecast'
# outputs.columns[4] == 'residuals'
# outputs.columns[5] == 'split'
```

### Nested Workflow (Grouped Models)
```python
wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
nested_fit = wf.fit_nested(train, group_col='store_id').evaluate(test)
outputs, coeffs, stats = nested_fit.extract_outputs()

# Guaranteed ordering:
# outputs.columns[0] == 'date'      (if present)
# outputs.columns[1] == 'store_id'  (group column)
# outputs.columns[2] == 'actuals'
# outputs.columns[3] == 'fitted'
# ... etc
```

### Model Spec Nested Fit
```python
spec = linear_reg()
nested_fit = spec.fit_nested(train, 'y ~ x', group_col='country').evaluate(test)
outputs, coeffs, stats = nested_fit.extract_outputs()

# All three DataFrames have consistent group column placement:
# outputs.columns[0:2] == ['date', 'country']
# coeffs.columns[0] == 'country'
# stats.columns[0] == 'country'
```

## Verification

All tests pass:
```bash
# Unit tests (16 tests)
pytest tests/test_utils/test_output_ordering.py -v
# 16 passed

# Integration tests (8 tests)
pytest tests/test_workflows/test_column_ordering_integration.py -v
# 8 passed

# Panel model regression tests (18 tests)
pytest tests/test_workflows/test_panel_models.py -v
# 18 passed, 12 warnings
```

Total: **24 new tests + 18 regression tests = 42 tests passing**

## Benefits

1. **Predictable Structure:** Users can rely on consistent column ordering
2. **Visual Clarity:** Date first, group second makes outputs easier to read
3. **Code Simplicity:** Can reference columns by position if needed
4. **Centralized Logic:** Single source of truth for column ordering
5. **Backward Compatible:** No breaking changes to existing code

## Related Issues

- User request: "Date should be first column, group should be second"
- Visualization compatibility: plot_forecast() works better with consistent ordering
- Data export: CSV/Excel outputs more intuitive with date first

## Future Considerations

- Could extend to other DataFrames (predictions, metrics)
- Could add user-configurable ordering preferences
- Could add validation to ensure required columns present
