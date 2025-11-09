# Selector Imports Fix

**Date:** 2025-11-09
**Issue:** Missing pattern-matching selector imports
**Status:** ✅ FIXED

---

## Problem

User encountered `NameError: name 'contains' is not defined` when running cells in `_md/forecasting_recipes.ipynb`.

The notebook's new recipe step cells (added in previous expansion) used pattern-matching selectors like:
- `contains()`
- `starts_with()`
- `ends_with()`
- `matches()`

But these were not imported in the notebook's import cell.

---

## Root Cause

The import cell only included basic selectors:
```python
from py_recipes.selectors import (
    all_numeric, all_nominal, all_predictors, all_outcomes,
    all_numeric_predictors, all_nominal_predictors
)
```

Pattern-matching selectors exist in `py_recipes/selectors.py` but were not imported:
- `contains(substring)` - Line 164
- `starts_with(prefix)` - Line 114
- `ends_with(suffix)` - Line 139
- `matches(pattern)` - Line 189

---

## Solution

Updated the import cell to include pattern-matching selectors:

```python
from py_recipes.selectors import (
    all_numeric, all_nominal, all_predictors, all_outcomes,
    all_numeric_predictors, all_nominal_predictors,
    contains, starts_with, ends_with, matches  # ADDED
)
```

---

## Pattern-Matching Selector Usage

### contains()
```python
# Select columns containing "lag"
rec = recipe().step_rm(contains("lag"))
```

### starts_with()
```python
# Select columns starting with "temp_"
rec = recipe().step_normalize(starts_with("temp_"))
```

### ends_with()
```python
# Select columns ending with "_1"
rec = recipe().step_log(ends_with("_1"))
```

### matches()
```python
# Select columns matching regex pattern
rec = recipe().step_scale(matches(r"^x\d+$"))
```

---

## Selector Functions Available

**Type Selectors:**
- `all_numeric()` - All numeric columns
- `all_nominal()` - All categorical columns
- `all_integer()` - Integer columns
- `all_float()` - Float columns
- `all_string()` - String columns
- `all_datetime()` - Datetime columns

**Pattern Selectors:**
- `contains(substring)` - Contains substring
- `starts_with(prefix)` - Starts with prefix
- `ends_with(suffix)` - Ends with suffix
- `matches(pattern)` - Regex pattern match

**Role Selectors:**
- `all_predictors()` - All predictor columns
- `all_outcomes()` - All outcome columns
- `all_numeric_predictors()` - Numeric predictors
- `all_nominal_predictors()` - Categorical predictors

**Utility Selectors:**
- `everything()` - All columns
- `one_of(*columns)` - Specific column list
- `none_of(*columns)` - Exclude columns
- `where(predicate)` - Custom predicate

**Combination Selectors:**
- `union(*selectors)` - OR combination
- `intersection(*selectors)` - AND combination
- `difference(include, exclude)` - Set difference

---

## Examples in Notebook

The new cells use pattern-matching selectors for:

**Cell 30 (Lag Features):**
```python
rec = (
    recipe()
    .step_rm(contains("lag"))  # Remove existing lag columns
    .step_lag(all_numeric_predictors(), lags=[1, 2, 3])
)
```

**Other Potential Uses:**
```python
# Remove all rolling window features
rec = recipe().step_rm(contains("rolling"))

# Select all crack-related features
rec = recipe().step_normalize(contains("crack"))

# Select lag features for specific columns
rec = recipe().step_select(starts_with("target_lag"))
```

---

## Files Modified

**Modified:**
- `_md/forecasting_recipes.ipynb` - Updated import cell (index 1)

**Selectors Added to Imports:**
- `contains`
- `starts_with`
- `ends_with`
- `matches`

---

## Verification

```python
# Test in notebook
from py_recipes.selectors import contains, starts_with, ends_with, matches

# These should now work without errors
rec = recipe().step_rm(contains("lag"))
rec = recipe().step_normalize(starts_with("temp_"))
rec = recipe().step_log(ends_with("_1"))
rec = recipe().step_scale(matches(r"^x\d+$"))
```

---

## Status

✅ **FIXED** - All pattern-matching selectors now available in notebook
- Import cell updated with 4 additional selectors
- All new recipe step cells will now execute without NameError
- Users can leverage full selector API for flexible column selection

---

## Related Documentation

- `py_recipes/selectors.py` - All selector implementations
- `STEP_INTERACT_SELECTOR_SUPPORT.md` - Example of selector usage with step_interact()
- `FORECASTING_RECIPES_NOTEBOOK_EXPANSION.md` - Notebook expansion that introduced these selectors
