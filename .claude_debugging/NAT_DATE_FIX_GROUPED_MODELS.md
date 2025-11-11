# NaT Date Issue Fix for Grouped Models with Recipes

**Date:** 2025-11-10
**Issue:** plot_forecast() drops train data when extract_outputs() returns NaT values in date column for grouped/nested models using recipes

## Problem Statement

When using grouped models (`fit_nested()`) with recipes that preprocess data:
1. `extract_outputs()` returned NaT (Not a Time) values in the date column
2. `plot_forecast()` dropped all rows with NaT dates
3. Result: Entire train set disappeared from forecast visualizations

### User-Visible Symptom
```python
# Cell 9 in forecasting_recipes_grouped.ipynb
wf = workflow().add_recipe(recipe().step_normalize(...)).add_model(linear_reg())
fit = wf.fit_nested(train_data, group_col='country')
fit = fit.evaluate(test_data)

# Visualization fails
fig = plot_forecast(fit)  # Train data not visible!
```

## Root Cause Analysis

### Complete Issue Chain

1. **Recipe preprocessing excludes datetime columns** from auto-generated formulas (line 278-280 of workflow.py)
   - This is correct behavior to prevent patsy from treating dates as categorical

2. **fit_nested() did NOT store original training data** with date columns
   - Unlike `NestedModelFit` which stores `group_train_data`
   - `NestedWorkflowFit` had no way to access original unprocessed training data

3. **evaluate() did NOT store original test data** with date columns
   - Processed test data through recipe, which excludes date
   - Passed processed data (without date) to `group_fit.evaluate()`
   - `evaluation_data["test_data"]` had NO date column

4. **extract_outputs() failed to find dates**
   - Engines return outputs DataFrame WITHOUT date column (by design)
   - `NestedWorkflowFit.extract_outputs()` tried to add dates from stored data
   - Train dates: Failed because no `group_train_data` was stored
   - Test dates: Failed because `evaluation_data["test_data"]` had no date column
   - Result: Both train and test rows kept NaT values

5. **plot_forecast() drops NaT rows**
   - Plotly automatically drops NaT values during rendering
   - All train/test data disappeared from visualization

### Why This Only Affected Grouped Models with Recipes

- **Formula-only workflows:** Date column present in raw data, forge() handles it
- **Ungrouped workflows:** Single model, simpler data flow
- **Grouped + recipes:** Combination of:
  - Recipe preprocessing that excludes dates
  - Multiple group processing that loses track of original data
  - No storage mechanism for original unprocessed data

## The Fix

### Three-Part Solution

**Part 1: Store Original Training Data (fit_nested)**
- **File:** `py_workflows/workflow.py` lines 401-411
- **Change:** Added `group_train_data = {}` dict to store original training data per group
- **Why:** Enables extraction of train dates with all original columns (including date)

```python
# NEW: Store original training data per group
group_train_data = {}
for group in groups:
    group_data = data[data[group_col] == group].copy()
    group_train_data[group] = group_data.copy()  # BEFORE preprocessing!
```

**Part 2: Store Original Test Data (evaluate)**
- **File:** `py_workflows/workflow.py` lines 1031-1048
- **Change:** Added `group_test_data = {}` dict to store original test data per group
- **Why:** Enables extraction of test dates from unprocessed test data

```python
# NEW: Store original test data per group
if not hasattr(self, 'group_test_data'):
    self.group_test_data = {}

for group, group_fit in self.group_fits.items():
    group_data = test_data[test_data[self.group_col] == group].copy()
    self.group_test_data[group] = group_data.copy()  # BEFORE preprocessing!
```

**Part 3: Use Stored Data in extract_outputs**
- **File:** `py_workflows/workflow.py` lines 1203-1245
- **Change:** Modified date extraction logic to use stored original data as PRIMARY source

```python
# TRAIN dates: Use group_train_data as PRIMARY source
if hasattr(self, 'group_train_data') and group in self.group_train_data:
    train_data_orig = self.group_train_data[group]
    if "date" in train_data_orig.columns:
        train_dates = train_data_orig["date"].values
        outputs.loc[train_mask, 'date'] = train_dates

# TEST dates: Use group_test_data as PRIMARY source
if hasattr(self, 'group_test_data') and group in self.group_test_data:
    test_data_orig = self.group_test_data[group]
    if "date" in test_data_orig.columns:
        test_dates = test_data_orig["date"].values
        outputs.loc[test_mask, 'date'] = test_dates
```

**Part 4: Update NestedWorkflowFit Dataclass**
- **File:** `py_workflows/workflow.py` lines 880-931
- **Change:** Added `group_train_data` field to dataclass
- **Why:** Makes the storage explicit in the class structure

```python
@dataclass
class NestedWorkflowFit:
    workflow: Workflow
    group_col: str
    group_fits: dict
    group_recipes: Optional[dict]
    group_train_data: dict  # NEW: Original training data with dates
```

## Verification

### Test Results
```
Total rows: 200
Train rows: 160
Test rows: 40

Train dates - NaT: 0 / 160  ✓
Test dates  - NaT: 0 / 40   ✓

SUCCESS: All dates populated correctly!
```

### Existing Tests
- All 18 panel model tests continue to pass
- No regressions introduced

## Impact

### Before Fix
- Train data invisible in forecast plots for grouped models with recipes
- Test data also invisible if recipe preprocessing excluded date
- User confusion about missing visualization data

### After Fix
- All dates properly populated in extract_outputs()
- plot_forecast() shows complete train and test data
- Consistent behavior across all model types and preprocessing strategies

## Files Changed

1. `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py`
   - Lines 401-411: Store `group_train_data` in `fit_nested()`
   - Lines 545-550: Pass `group_train_data` to `NestedWorkflowFit` constructor
   - Lines 880-931: Add `group_train_data` field to `NestedWorkflowFit` dataclass
   - Lines 1031-1048: Store `group_test_data` in `evaluate()`
   - Lines 1203-1245: Use stored data in `extract_outputs()`

## Design Principles Applied

1. **Store Early, Process Later**: Save original data before any transformations
2. **Parallel Structure**: `NestedWorkflowFit` now matches `NestedModelFit` pattern
3. **Fallback Hierarchy**: Primary source → secondary source → tertiary source
4. **Backward Compatibility**: All existing tests pass without modification

## Prevention

This class of issues (losing metadata during preprocessing) can be prevented by:

1. **Always store original unprocessed data** alongside processed versions
2. **Document what gets excluded** during preprocessing (e.g., datetime columns)
3. **Test visualization pipelines** end-to-end, not just data transformations
4. **Check for NaT/NaN values** in critical columns before passing to visualization

## Related Code Patterns

The same pattern is used in:
- `NestedModelFit` (py_parsnip/model_spec.py): Stores `group_train_data` for date extraction
- `ModelFit.evaluate()`: Stores `original_test_data` for raw value access
- `Workflow.fit()`: Stores `original_data` for engines needing raw values

## Future Improvements

Consider:
1. Centralized metadata preservation strategy across all nested/grouped operations
2. Explicit "metadata columns" concept (date, ID, etc.) that are always preserved
3. Validation that required columns (like date) exist before visualization
4. Warning when NaT values detected in visualization inputs
