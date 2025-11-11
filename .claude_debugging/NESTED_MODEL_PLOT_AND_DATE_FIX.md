# Nested Model Plot and Date Extraction Fix

**Date**: 2025-11-10
**Status**: ✅ COMPLETED

## Issues Reported

User reported two issues with cell 7 in `forecasting_recipes_grouped.ipynb`:

1. **plot_forecast() not plotting separate groups**: Plot showed single model instead of separate subplots for each country
2. **Some dates showing NaT**: `outputs_baseline` had NaT values in date column

## Root Causes

### Issue 1: plot_forecast() Not Detecting NestedModelFit

**Problem**: The `plot_forecast()` function only checked for `NestedWorkflowFit`, but not `NestedModelFit`.

```python
# OLD CODE (line 64 in forecast.py)
from py_workflows.workflow import NestedWorkflowFit
is_nested = isinstance(fit, NestedWorkflowFit)  # Missing NestedModelFit!
```

When user called `ModelSpec.fit_nested()`, it returned a `NestedModelFit`, which wasn't detected as nested, so it was plotted as a single model without group separation.

### Issue 2: Training Data Dates Not Preserved

**Problem**: The `NestedModelFit.extract_outputs()` method wasn't preserving training data dates because:

1. `fit_nested()` dropped the group column before fitting, but didn't store the original training data
2. Formula `"refinery_kbd ~ ."` excluded the date column from predictors
3. Molded data had RangeIndex, not DatetimeIndex (since date wasn't used as index)
4. Only test data dates were extracted (from `evaluation_data`)

Result: Training data had 0% date coverage (all NaT), test data had 100% coverage.

## Solutions Implemented

### Fix 1: Update plot_forecast() to Detect NestedModelFit

**File**: `py_visualize/forecast.py` (lines 64-66)

```python
# NEW CODE
from py_workflows.workflow import NestedWorkflowFit
from py_parsnip.model_spec import NestedModelFit
is_nested = isinstance(fit, (NestedWorkflowFit, NestedModelFit))
```

Now detects both `NestedWorkflowFit` (from `Workflow.fit_nested()`) and `NestedModelFit` (from `ModelSpec.fit_nested()`).

### Fix 2: Store and Extract Training Data Dates

**Changes made**:

1. **Store original training data** (`py_parsnip/model_spec.py`, lines 352-359)
   ```python
   group_fits = {}
   group_train_data = {}  # NEW: Store original training data per group

   for group in groups:
       group_data = data[data[group_col] == group].copy()

       # Store original group training data (before dropping group column)
       group_train_data[group] = group_data.copy()  # NEW
   ```

2. **Pass to NestedModelFit** (line 384)
   ```python
   return NestedModelFit(
       spec=self,
       group_col=group_col,
       group_fits=group_fits,
       formula=formula,
       group_train_data=group_train_data  # NEW
   )
   ```

3. **Update NestedModelFit attributes** (line 674)
   ```python
   @dataclass
   class NestedModelFit:
       spec: ModelSpec
       group_col: str
       group_fits: Dict[Any, ModelFit]
       formula: str
       group_train_data: Dict[Any, pd.DataFrame]  # NEW
   ```

4. **Extract dates from stored training data** (`extract_outputs()`, lines 822-829)
   ```python
   # Try to get training dates from stored original training data
   if hasattr(self, 'group_train_data') and group in self.group_train_data:
       train_data_orig = self.group_train_data[group]
       if "date" in train_data_orig.columns:
           train_dates = train_data_orig["date"].values
           train_mask = outputs['split'] == 'train'
           if train_mask.sum() == len(train_dates):
               outputs.loc[train_mask, 'date'] = train_dates
   ```

## Test Results

### Before Fixes
- ❌ Training data: 0% date coverage (all NaT)
- ✅ Test data: 100% date coverage
- ❌ plot_forecast(): Single plot without group separation

### After Fixes
- ✅ Training data: 100% date coverage (120/120 rows)
- ✅ Test data: 100% date coverage (80/80 rows)
- ✅ plot_forecast(): 2 subplots (one per country) with 6 traces total

**Test verification**:
```
TRAIN split:
  Total rows: 120
  Rows with dates: 120 (100.0%)
  Rows with NaT: 0 (0.0%)

TEST split:
  Total rows: 80
  Rows with dates: 80 (100.0%)
  Rows with NaT: 0 (0.0%)

By GROUP:
  USA:
    Train: 60/60 with dates
    Test: 40/40 with dates
  UK:
    Train: 60/60 with dates
    Test: 40/40 with dates

plot_forecast() succeeded!
   Plot has 6 traces
   Plot has subplots (nested model detected)
   Number of subplot rows: 2
```

## Files Modified

1. **`py_visualize/forecast.py`** (lines 64-66)
   - Added import for `NestedModelFit`
   - Updated `isinstance()` check to include both nested types

2. **`py_parsnip/model_spec.py`** (multiple locations)
   - Lines 352-384: Store original training data in `fit_nested()`
   - Line 674: Add `group_train_data` attribute to `NestedModelFit`
   - Lines 822-829: Extract training dates from stored data in `extract_outputs()`

## Test Coverage

- ✅ All 18 panel model tests passed (`tests/test_workflows/test_panel_models.py`)
- ✅ All 26 linear_reg tests passed (`tests/test_parsnip/test_linear_reg.py`)
- ✅ Manual test verified 100% date coverage and subplot creation

## Usage Notes

### For Workflow-based Nested Models
```python
wf = workflow().add_formula("y ~ .").add_model(linear_reg())
fit = wf.fit_nested(train_data, group_col='country')
fit = fit.evaluate(test_data)

# Both work now
outputs, _, _ = fit.extract_outputs()  # 100% date coverage
fig = plot_forecast(fit)  # Separate subplots per group
```

### For ModelSpec-based Nested Models
```python
spec = linear_reg().set_engine("sklearn")
fit = spec.fit_nested(train_data, "y ~ .", group_col='country')
fit = fit.evaluate(test_data)

# Both work now
outputs, _, _ = fit.extract_outputs()  # 100% date coverage
fig = plot_forecast(fit)  # Separate subplots per group
```

## Memory Considerations

**Storage overhead**: The fix stores the original training data per group in `group_train_data` dict. For datasets with many groups or large training sets, this increases memory usage.

**Example**:
- 10 groups × 1000 rows × 10 columns = ~80KB additional memory (negligible)
- 100 groups × 10000 rows × 100 columns = ~80MB additional memory (moderate)

**Trade-off**: This is acceptable because:
1. Enables proper date extraction for visualization and analysis
2. Memory is freed when `NestedModelFit` object is deleted
3. Most use cases have <100 groups

## Related Fixes

This fix complements:
1. Previous `NestedWorkflowFit.extract_outputs()` date extraction (completed earlier)
2. StepSafeV2 importance calculation fix (separate issue)

## User Action Required

**Restart your Jupyter kernel** in `forecasting_recipes_grouped.ipynb`:
1. Kernel → Restart
2. Re-run all cells
3. Cell 7 should now show:
   - No NaT dates in `outputs_baseline`
   - Separate subplots for each country in `plot_forecast()`
