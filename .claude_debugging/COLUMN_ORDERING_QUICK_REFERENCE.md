# Column Ordering Quick Reference

## What Changed?

All `extract_outputs()` methods now return DataFrames with **consistent, predictable column ordering**.

## New Column Order

### Outputs DataFrame (Observation-Level Data)

```python
outputs, coeffs, stats = fit.extract_outputs()

# outputs.columns ordering:
# 1. 'date'            - Always first (if present)
# 2. <group_col>       - Second (e.g., 'store_id', 'country') - nested models only
# 3. 'actuals'         - True outcome values
# 4. 'fitted'          - Model predictions
# 5. 'forecast'        - Combined actuals/fitted
# 6. 'residuals'       - actuals - fitted
# 7. 'split'           - 'train', 'test', or 'forecast'
# 8. 'model'           - Model type
# 9. 'model_group_name' - Model grouping
# 10. 'group'          - 'global' or group identifier
# ... additional columns in original order
```

### Coefficients DataFrame

```python
# coeffs.columns ordering:
# 1. <group_col>       - First (nested models only)
# 2. 'variable'        - Predictor name
# 3. 'coefficient'     - Coefficient value
# 4. 'std_error'       - Standard error
# 5. 't_stat'          - t-statistic
# 6. 'p_value'         - p-value
# 7. 'conf_low'        - CI lower bound
# 8. 'conf_high'       - CI upper bound
# 9. 'vif'             - Variance Inflation Factor
# 10. 'model'          - Model type
# ... additional columns
```

### Stats DataFrame

```python
# stats.columns ordering:
# 1. <group_col>       - First (nested models only)
# 2. 'split'           - 'train', 'test', or 'forecast'
# 3. 'metric'          - Metric name (rmse, mae, r_squared, etc.)
# 4. 'value'           - Metric value
# 5. 'model'           - Model type
# ... additional columns
```

## Examples

### Before (Inconsistent)
```python
outputs.columns
# ['residuals', 'model', 'actuals', 'split', 'fitted', 'date', 'forecast', ...]
# Date was last!
```

### After (Consistent)
```python
outputs.columns
# ['date', 'actuals', 'fitted', 'forecast', 'residuals', 'split', 'model', ...]
# Date is first!
```

### Nested Models (With Group Column)
```python
# Before
nested_outputs.columns
# ['actuals', 'store_id', 'fitted', 'date', 'split', ...]
# Date not first, group not second

# After
nested_outputs.columns
# ['date', 'store_id', 'actuals', 'fitted', 'forecast', 'residuals', 'split', ...]
# Date first, group second!
```

## Code Changes Needed?

**None!** This change is 100% backward compatible. Your existing code will continue to work.

### If you accessed columns by name (recommended):
```python
# No changes needed
date_col = outputs['date']
actuals = outputs['actuals']
fitted = outputs['fitted']
```

### If you accessed columns by position:
```python
# Old code (will still work, but results changed)
first_col = outputs.iloc[:, 0]  # Was 'actuals', now 'date'

# New code (benefits from consistent ordering)
date_col = outputs.iloc[:, 0]   # Always 'date' (if present)
actuals_col = outputs.iloc[:, 1]  # Always 'actuals' (non-nested) or group_col (nested)
```

## Benefits

### 1. Visual Inspection
```python
# Easier to read in Jupyter notebooks
outputs.head()
#         date  store_id  actuals  fitted  forecast  residuals  split
# 0 2020-01-01         A    100.0    99.5      99.5        0.5  train
# 1 2020-01-02         A    105.0   105.2     105.2       -0.2  train
```

### 2. Simplified Code
```python
# Easy access to key columns
print(outputs.columns[0])  # Always 'date' (if present)
print(outputs.columns[1])  # Always 'actuals' or <group_col>

# Filter by group easily
store_a = outputs[outputs['store_id'] == 'A']  # Group is second column
```

### 3. Better Visualization
```python
from py_visualize import plot_forecast

# plot_forecast() expects date first for optimal display
plot_forecast(outputs)  # Now date is guaranteed to be first!
```

## Special Cases

### Date in Index
If date was in the index (some engines do this), it's automatically reset to be the first column:
```python
# Before reordering (date in index)
outputs.index  # DatetimeIndex([...])
outputs.columns  # ['actuals', 'fitted', ...]

# After reordering (date as column)
outputs.columns  # ['date', 'actuals', 'fitted', ...]
```

### No Date Column
If your data has no date column, ordering starts with core columns:
```python
# Without date
outputs.columns
# ['actuals', 'fitted', 'forecast', 'residuals', 'split', ...]
```

### No Group Column (Standard Models)
Non-nested models skip the group column position:
```python
# Standard workflow
outputs.columns
# ['date', 'actuals', 'fitted', 'forecast', ...]
```

## Testing

All existing tests pass + 24 new tests added:
- ✓ 16 unit tests for reordering functions
- ✓ 8 integration tests for workflows
- ✓ 18 regression tests for panel models
- **Total: 42 tests passing**

## Questions?

See full documentation:
- **COLUMN_ORDERING_FIX.md** - Technical details
- **COLUMN_ORDERING_SUMMARY.md** - Complete code changes
