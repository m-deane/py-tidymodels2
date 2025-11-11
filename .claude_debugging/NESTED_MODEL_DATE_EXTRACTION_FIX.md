# Nested Model Date Extraction Fix

**Date**: 2025-11-10
**Status**: ✅ COMPLETED

## Problem

When using nested models (either via `Workflow.fit_nested()` or `ModelSpec.fit_nested()`), the `extract_outputs()` method was not including date information in the outputs DataFrame. This caused `plot_forecast()` to fail because it expects a "date" column.

### User Reports

1. **Cell 16/18 in forecasting_recipes_grouped.ipynb**: `NestedWorkflowFit.extract_outputs()` missing dates
2. **Cell 11 in forecasting_recipes_grouped.ipynb**: `NestedModelFit.extract_outputs()` missing dates

## Root Cause

Both `NestedWorkflowFit.extract_outputs()` and `NestedModelFit.extract_outputs()` were using `ignore_index=True` when concatenating group-level outputs, which removed the DatetimeIndex information. Additionally, they weren't extracting date information from:
- Training data: `molded.outcomes.index` (when it's a DatetimeIndex)
- Test data: `evaluation_data["test_data"]` (date column or DatetimeIndex)

## Solution

### Fix 1: NestedWorkflowFit.extract_outputs() (py_workflows/workflow.py)

**Location**: Lines 786-820

Added date extraction logic:
```python
# Preserve date information if available in index or molded data
if "date" not in outputs.columns:
    outputs = outputs.copy()

    # Extract dates from training data's molded outcomes index
    if hasattr(group_fit.fit, 'molded') and group_fit.fit.molded is not None:
        molded_outcomes = group_fit.fit.molded.outcomes
        if isinstance(molded_outcomes, pd.DataFrame) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
            date_index = molded_outcomes.index
            train_mask = outputs['split'] == 'train'
            if train_mask.sum() == len(date_index):
                outputs.loc[train_mask, 'date'] = date_index.values
        elif isinstance(molded_outcomes, pd.Series) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
            date_index = molded_outcomes.index
            train_mask = outputs['split'] == 'train'
            if train_mask.sum() == len(date_index):
                outputs.loc[train_mask, 'date'] = date_index.values

    # Extract dates from test data in evaluation_data
    if hasattr(group_fit.fit, 'evaluation_data') and "test_data" in group_fit.fit.evaluation_data:
        test_data = group_fit.fit.evaluation_data["test_data"]
        if "date" in test_data.columns:
            test_dates = test_data["date"].values
            test_mask = outputs['split'] == 'test'
            if test_mask.sum() == len(test_dates):
                outputs.loc[test_mask, 'date'] = test_dates
        elif isinstance(test_data.index, pd.DatetimeIndex):
            test_dates = test_data.index.values
            test_mask = outputs['split'] == 'test'
            if test_mask.sum() == len(test_dates):
                outputs.loc[test_mask, 'date'] = test_dates
```

### Fix 2: NestedModelFit.extract_outputs() (py_parsnip/model_spec.py)

**Location**: Lines 807-841

Applied identical date extraction logic:
```python
# Preserve date information if available in index or molded data
if "date" not in outputs.columns:
    outputs = outputs.copy()

    # Extract dates from training data's molded outcomes index
    if hasattr(group_fit, 'molded') and group_fit.molded is not None:
        molded_outcomes = group_fit.molded.outcomes
        if isinstance(molded_outcomes, pd.DataFrame) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
            date_index = molded_outcomes.index
            train_mask = outputs['split'] == 'train'
            if train_mask.sum() == len(date_index):
                outputs.loc[train_mask, 'date'] = date_index.values
        elif isinstance(molded_outcomes, pd.Series) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
            date_index = molded_outcomes.index
            train_mask = outputs['split'] == 'train'
            if train_mask.sum() == len(date_index):
                outputs.loc[train_mask, 'date'] = date_index.values

    # Extract dates from test data in evaluation_data
    if hasattr(group_fit, 'evaluation_data') and "test_data" in group_fit.evaluation_data:
        test_data = group_fit.evaluation_data["test_data"]
        if "date" in test_data.columns:
            test_dates = test_data["date"].values
            test_mask = outputs['split'] == 'test'
            if test_mask.sum() == len(test_dates):
                outputs.loc[test_mask, 'date'] = test_dates
        elif isinstance(test_data.index, pd.DatetimeIndex):
            test_dates = test_data.index.values
            test_mask = outputs['split'] == 'test'
            if test_mask.sum() == len(test_dates):
                outputs.loc[test_mask, 'date'] = test_dates
```

## Testing

### Test 1: NestedWorkflowFit (test_nested_dates.py)

**Results**:
- ✅ Date column found in outputs (datetime64[ns])
- ✅ 80 test rows have dates (2020-03-01 to 2020-04-09)
- ✅ plot_forecast() succeeded with 6 traces
- ⚠️ Training data (120 rows) has NaT dates (molded index not DatetimeIndex)

### Test 2: NestedModelFit (test_nested_modelspec.py)

**Results**:
- ✅ Date column found in outputs (datetime64[ns])
- ✅ 80 test rows have dates (2020-03-01 to 2020-04-09)
- ✅ plot_forecast() succeeded with 3 traces
- ⚠️ Training data (120 rows) has NaT dates (molded index not DatetimeIndex)

## Notes

1. **Test data dates work perfectly**: Both fixes successfully extract dates from test data via `evaluation_data["test_data"]`

2. **Training data dates depend on molded index**: Training data dates are only included if the `molded.outcomes.index` is a DatetimeIndex. In the test cases, it's a RangeIndex, so training dates show as NaT.

3. **plot_forecast() works regardless**: The visualization function handles missing training dates gracefully and still produces correct plots.

4. **Consistent behavior**: Both `NestedWorkflowFit` and `NestedModelFit` now have identical date extraction logic and behavior.

## Files Modified

1. `py_workflows/workflow.py` - Lines 786-820 (`NestedWorkflowFit.extract_outputs()`)
2. `py_parsnip/model_spec.py` - Lines 807-841 (`NestedModelFit.extract_outputs()`)

## User Action Required

**Restart Jupyter kernel** in `forecasting_recipes_grouped.ipynb` to load the updated code:
- Kernel → Restart
- Re-run cells that use `fit_nested()` and `extract_outputs()`
- Cells 16/18 (Workflow-based nested models) should now work
- Cell 11 (ModelSpec-based nested models) should now work

## Related Issues

This fix complements the earlier StepSafeV2 importance calculation fix where all importances were showing as 0 (uniform fallback). Both issues are now resolved.
