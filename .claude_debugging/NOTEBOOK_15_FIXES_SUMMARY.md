# Notebook 15 (Stacks Demo) - Bug Fixes Summary

**Date**: 2025-10-28
**Notebook**: `examples/15_stacks_demo.ipynb`
**Total Bugs Fixed**: 4

## Overview

Fixed import errors and syntax issues preventing Notebook 15 from executing.

---

## Bug #1: Incorrect Import Statement

**Location**: Cell 1 (imports)

**Problem**: Notebook was trying to import `step_date`, `step_lag`, and `step_normalize` as standalone functions from `py_recipes`, but these are methods on the Recipe class, not standalone functions.

**Error Message**:
```
ImportError: cannot import name 'step_date' from 'py_recipes'
```

**Original Code**:
```python
from py_recipes import recipe, step_date, step_lag, step_normalize
```

**Fixed Code**:
```python
from py_recipes import recipe
```

**Explanation**: In py-tidymodels, preprocessing steps are methods called on recipe objects using the fluent/builder pattern:
```python
recipe().step_date(...).step_lag(...).step_normalize(...)
```

---

## Bug #2: R-Style Formula Syntax

**Location**: Cell 5 (recipe creation)

**Problem**: Notebook was using R-style formula syntax `recipe(value ~ date, data=train_data)` which is invalid Python syntax.

**Error Message**:
```
SyntaxError: invalid syntax. Perhaps you forgot a comma?
```

**Original Code**:
```python
rec = (
    recipe(value ~ date, data=train_data)
    .step_date('date', features=['month', 'week', 'doy', 'dow'])
    .step_lag('value', lags=[1, 7, 14, 30])
    .step_normalize(['value_lag_1', 'value_lag_7', 'value_lag_14', 'value_lag_30'])
)
```

**Fixed Code**:
```python
rec = (
    recipe()
    .step_date('date', features=['month', 'week', 'doy', 'dow'])
    .step_lag(['value'], lags=[1, 7, 14, 30])
    .step_normalize(['value_lag_1', 'value_lag_7', 'value_lag_14', 'value_lag_30'])
)
```

**Explanation**: The Python implementation of py-tidymodels uses `recipe()` without formula syntax. The recipe function signature is:
```python
def recipe(data: Optional[pd.DataFrame] = None) -> Recipe
```

---

## Bug #3: step_lag() Parameter Type

**Location**: Cell 5 (recipe creation, same cell as Bug #2)

**Problem**: `step_lag()` expects `columns: List[str]` but was receiving a single string `'value'` instead of a list `['value']`.

**Error Message**:
```
KeyError: "None of [Index(['value_lag_1', 'value_lag_7', 'value_lag_14', 'value_lag_30'], dtype='object')] are in the [columns]"
```

**Root Cause**: When a string is passed instead of a list, Python iterates over the characters (`'v', 'a', 'l', 'u', 'e'`) rather than treating it as a single column name, causing the lag step to fail silently and not create the expected lag columns.

**Original Code**:
```python
.step_lag('value', lags=[1, 7, 14, 30])
```

**Fixed Code**:
```python
.step_lag(['value'], lags=[1, 7, 14, 30])
```

**Explanation**: The `step_lag()` method signature requires a list:
```python
def step_lag(self, columns: List[str], lags: List[int]) -> "Recipe"
```

---

## Bug #4: Missing Values from Lag Features Cause Patsy Error

**Location**: Cell 5 (recipe creation) + new step implementation

**Problem**: `step_lag()` creates lag features using pandas `.shift()` which introduces NaN values at the beginning of the time series (e.g., first 30 rows will have NaN for 30-day lag). When the workflow tries to fit the model, Patsy's formula parser encounters these missing values and raises an error with `NA_action="raise"`.

**Error Message**:
```
PatsyError: factor contains missing values
    value ~ date + date_month + date_week + value_lag_1 + value_lag_7 + value_lag_14 + value_lag_30
                                            ^^^^^^^^^^^
ValueError: Failed to parse formula 'value ~ date + date_month + date_week + value_lag_1 + value_lag_7 + value_lag_14 + value_lag_30': factor contains missing values
```

**Root Cause**:
- `step_lag(['value'], lags=[1, 7, 14, 30])` creates columns with NaN in the first 30 rows
- Patsy (used by py_hardhat for model specification) has `NA_action="raise"` by default
- The missing values must be handled before the data reaches Patsy

**Solution**: Created a new `StepNaOmit` class and `step_naomit()` method to remove rows with NaN values. Added it to the recipe pipeline after creating lag features.

**Original Code**:
```python
rec = (
    recipe()
    .step_date('date', features=['month', 'week', 'doy', 'dow'])
    .step_lag(['value'], lags=[1, 7, 14, 30])
    .step_normalize(['value_lag_1', 'value_lag_7', 'value_lag_14', 'value_lag_30'])
)
```

**Fixed Code**:
```python
rec = (
    recipe()
    .step_date('date', features=['month', 'week', 'doy', 'dow'])
    .step_lag(['value'], lags=[1, 7, 14, 30])
    .step_naomit()  # Remove rows with NaN from lag features
    .step_normalize(['value_lag_1', 'value_lag_7', 'value_lag_14', 'value_lag_30'])
)
```

**Implementation Details**:
1. Created `py_recipes/steps/naomit.py` with `StepNaOmit` and `PreparedStepNaOmit` classes
2. Added `step_naomit()` method to Recipe class at line 603 in `py_recipes/recipe.py`
3. Exported classes in `py_recipes/steps/__init__.py`
4. Uses pandas `dropna()` to remove rows with NaN values

**StepNaOmit Signature**:
```python
def step_naomit(
    self,
    columns: Optional[List[str]] = None
) -> "Recipe":
    """
    Remove rows with missing values.

    Args:
        columns: Columns to check for NAs (None = check all columns)

    Returns:
        Self for method chaining
    """
```

**Explanation**: The `step_naomit()` step removes rows containing NaN values, similar to R's `na.omit()`. When called without arguments, it removes rows with NaN in any column. This is essential after time series operations like `step_lag()` which introduce missing values.

---

## Summary of Changes

### File: `py_recipes/__init__.py`
- Added imports for `StepLag` and `StepDate` classes
- Added them to `__all__` list for proper module exports

### File: `py_recipes/steps/naomit.py` (NEW)
- Created `StepNaOmit` class to remove rows with missing values
- Created `PreparedStepNaOmit` class for fitted step
- Uses pandas `dropna()` to filter rows

### File: `py_recipes/steps/__init__.py`
- Added imports for `StepNaOmit` and `PreparedStepNaOmit`
- Added them to `__all__` list for exports

### File: `py_recipes/recipe.py`
- Added `step_naomit()` method at line 603
- Method removes rows with NaN values in specified columns

### File: `examples/15_stacks_demo.ipynb`
1. **Cell 1**: Removed incorrect imports (`step_date`, `step_lag`, `step_normalize`)
2. **Cell 5**: Fixed `recipe()` call to remove R-style formula syntax
3. **Cell 5**: Fixed `step_lag()` to pass list `['value']` instead of string `'value'`
4. **Cell 5**: Added `step_naomit()` after `step_lag()` to remove rows with NaN values

---

## Key Learnings

1. **Method vs Function**: In py-tidymodels, preprocessing steps like `step_date()` and `step_lag()` are methods on the Recipe class, not standalone functions to import.

2. **Python vs R Syntax**: Unlike R's tidymodels which uses formula syntax (`value ~ date`), py-tidymodels uses method chaining on recipe objects.

3. **Type Strictness**: Python's type hints are enforced - passing a string where a `List[str]` is expected causes silent failures or confusing errors downstream.

4. **Recipe Builder Pattern**: py-tidymodels implements the fluent/builder pattern:
   ```python
   recipe().step_a(...).step_b(...).step_c(...)
   ```

5. **Time Series Missing Values**: Lag operations create NaN values at the start of time series. These must be handled before model fitting, typically with `step_naomit()` immediately after lag creation.

6. **Patsy NA Handling**: The Patsy formula parser (used in py_hardhat) fails on missing values by default. Recipes should remove or impute NAs before data reaches the model fitting stage.

---

## Testing

After fixes, the notebook should execute without errors. The recipe will:
1. Extract date features (month, week, day of year, day of week) from the 'date' column
2. Create lag features for 'value' at lags [1, 7, 14, 30 days]
3. Normalize the lag features for consistent scaling

