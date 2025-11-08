# Datetime Column Formula Exclusion Fix

**Date**: 2025-11-07
**Issue**: User reported error when using `.add_recipe()` workflows with time series data containing datetime columns.

## Problem

When using a workflow with a recipe but NO explicit formula, the workflow auto-generates a formula like `"target ~ ."` which expands to include ALL columns, including datetime columns like `date`.

**Error Encountered**:
```python
# User's code
rec = recipe().step_impute_median().step_boxcox()
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)
fit = fit.evaluate(test_data)  # ERROR!

# Error message
ValueError: Failed to apply blueprint to new data: Error converting data to categorical:
observation with value Timestamp('2023-10-01 00:00:00') does not match any of the
expected levels (expected: [Timestamp('2020-04-01'), ..., Timestamp('2023-09-01')])
```

**Root Cause**:
1. Recipe processes data but keeps `date` column
2. Workflow auto-generates formula: `"target ~ date + x1 + x2 + ..."`
3. Patsy treats `date` (datetime) as categorical variable
4. Test data has NEW dates not seen in training → categorical encoding fails

**Why This Is Wrong**:
For time series regression, datetime columns should be the **index**, not exogenous variables. Including them as predictors causes errors when forecasting future dates that weren't in the training set.

## Solution

Modified `py_workflows/workflow.py` (lines 216-225) to automatically **exclude datetime columns** from auto-generated formulas:

```python
# Build explicit formula (patsy doesn't support "y ~ ." notation)
# Exclude datetime columns - they should be indices, not predictors
predictor_cols = [
    col for col in processed_data.columns
    if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
]
if len(predictor_cols) == 0:
    raise ValueError("No predictor columns found after recipe preprocessing")
formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"
```

**Key Change**: Added `and not pd.api.types.is_datetime64_any_dtype(processed_data[col])` filter.

## Results

### ✅ User's Scenario Now Works

```python
# Create recipe without removing date
rec = recipe().step_impute_median().step_boxcox()

# Workflow auto-generates formula excluding date
wf = workflow().add_recipe(rec).add_model(linear_reg())

# ✅ Fit on training data (dates 2020-04 to 2023-09)
fit = wf.fit(train_data)

# ✅ Evaluate on test data with NEW dates (2023-10+)
fit = fit.evaluate(test_data)  # No error!
```

The auto-generated formula is now: `"target ~ x1 + x2 + x3"` (date excluded)

### ✅ Multiple Datetime Columns Handled

```python
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'timestamp': pd.to_datetime(...),
    'created_at': pd.to_datetime(...),
    'x1': [...],
    'target': [...]
})

# All datetime columns excluded from formula
wf = workflow().add_recipe(recipe()).add_model(linear_reg())
fit = wf.fit(data)
# Auto-formula: "target ~ x1" (all datetime columns excluded)
```

### ✅ Works When No Datetime Columns

```python
# Regular tabular data (no time series)
data = pd.DataFrame({
    'x1': [...],
    'x2': [...],
    'target': [...]
})

wf = workflow().add_recipe(recipe()).add_model(linear_reg())
fit = wf.fit(data)
# Auto-formula: "target ~ x1 + x2" (normal behavior)
```

## Test Coverage

**File**: `tests/test_workflows/test_datetime_exclusion.py`

5 comprehensive tests, all passing:
1. ✅ `test_datetime_excluded_from_recipe_formula` - Single datetime column excluded
2. ✅ `test_datetime_exclusion_with_new_dates` - New date ranges work
3. ✅ `test_multiple_datetime_columns_excluded` - Multiple datetime columns all excluded
4. ✅ `test_no_datetime_columns_still_works` - Non-time-series data unaffected
5. ✅ `test_explicit_formula_overrides_datetime_exclusion` - Explicit formulas respected

## Files Modified

1. **`py_workflows/workflow.py`** (lines 216-225)
   - Added datetime column filter to auto-generated formula logic
   - Only affects recipes without explicit formulas

2. **`tests/test_workflows/test_datetime_exclusion.py`** (NEW)
   - 5 tests covering datetime exclusion behavior
   - Tests edge cases: multiple datetime columns, no datetime columns, new dates

## Important Notes

### Only Affects Auto-Generated Formulas

This exclusion ONLY applies when:
- Workflow has a recipe via `.add_recipe()`
- NO explicit formula provided via `.add_formula()`

If user explicitly provides a formula, it's used as-is:
```python
# Explicit formula - user controls what's included
wf = workflow()
    .add_formula("target ~ date + x1")  # User explicitly wants date
    .add_model(linear_reg())
# Uses user's formula unchanged
```

### Alternative User Workarounds

Before this fix, users had to manually remove date columns:
```python
# BEFORE (workaround)
rec = recipe().step_rm("date")  # Manually remove date

# AFTER (automatic)
rec = recipe()  # Date automatically excluded from formula
```

## Comparison with R tidymodels

In R's tidymodels, date columns are typically handled via:
1. `step_rm()` to explicitly remove them
2. Or using time series-specific recipes that handle dates properly

Our Python implementation now matches this behavior by automatically excluding datetime columns from predictor lists.

## Test Results

- **Workflow tests**: 55/57 passing (96.5%)
  - 2 pre-existing failures unrelated to this fix
- **Datetime exclusion tests**: 5/5 passing (100%)

## Key Takeaway

**Smart formula generation**: Auto-generated formulas should only include valid predictor variables. Datetime columns are indices/timestamps, not predictors, so they should be automatically excluded to prevent categorical encoding errors during forecasting.
