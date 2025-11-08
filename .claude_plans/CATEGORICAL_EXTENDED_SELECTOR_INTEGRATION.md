# Categorical Extended Selector Integration

**Date:** 2025-11-07
**Status:** COMPLETE
**Tests:** 39/39 passing

## Summary

Successfully integrated `resolve_selector()` into all four categorical step classes in `py_recipes/steps/categorical_extended.py`, enabling flexible column selection via selectors (e.g., `all_nominal()`, `all_numeric()`) or explicit column lists.

## Changes Made

### 1. Import Additions
```python
from typing import List, Optional, Dict, Any, Union, Callable
from py_recipes.selectors import resolve_selector, all_nominal
```

### 2. Class Updates

Updated all four categorical step classes:

#### StepOther (Lines 16-63)
- **Purpose:** Pool infrequent categorical levels into "other"
- **Dataclass Change:** `columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None`
- **prep() Change:**
  ```python
  selector = self.columns if self.columns is not None else all_nominal()
  cols = resolve_selector(selector, data)
  ```

#### StepNovel (Lines 106-144)
- **Purpose:** Handle novel categorical levels in new data
- **Dataclass Change:** `columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None`
- **prep() Change:**
  ```python
  selector = self.columns if self.columns is not None else all_nominal()
  cols = resolve_selector(selector, data)
  ```

#### StepUnknown (Lines 187-220)
- **Purpose:** Assign missing categorical values to "unknown" level
- **Dataclass Change:** `columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None`
- **prep() Change:**
  ```python
  selector = self.columns if self.columns is not None else all_nominal()
  cols = resolve_selector(selector, data)
  ```

#### StepIndicateNa (Lines 257-292)
- **Purpose:** Create indicator columns for missing values
- **Dataclass Change:** `columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None`
- **prep() Change:**
  ```python
  if self.columns is None:
      cols = [col for col in data.columns if data[col].isna().any()]
  else:
      selector = self.columns
      cols = resolve_selector(selector, data)
      cols = [col for col in cols if data[col].isna().any()]
  ```
  *Note: StepIndicateNa has special logic to filter for columns with missing values*

### 3. StepInteger (Lines 329-418)
- **Status:** NOT updated (kept original implementation)
- **Reason:** Uses manual categorical detection, can be updated later if needed

## Test Results

All 39 tests passing in `tests/test_recipes/test_categorical_extended.py`:

**StepOther Tests (5):**
- test_other_basic
- test_other_specific_columns
- test_other_threshold
- test_other_new_data
- test_other_preserves_shape

**StepNovel Tests (5):**
- test_novel_basic
- test_novel_specific_columns
- test_novel_no_novel_values
- test_novel_all_novel
- test_novel_preserves_known

**StepIndicateNa Tests (6):**
- test_indicate_na_basic
- test_indicate_na_values
- test_indicate_na_specific_columns
- test_indicate_na_no_missing
- test_indicate_na_new_data
- test_indicate_na_preserves_original

**StepInteger Tests (6):**
- test_integer_basic
- test_integer_specific_columns
- test_integer_consistent_encoding
- test_integer_new_data
- test_integer_preserves_shape
- test_integer_zero_based

**StepUnknown Tests (9):**
- test_unknown_basic
- test_unknown_specific_columns
- test_unknown_custom_label
- test_unknown_no_missing
- test_unknown_all_missing
- test_unknown_new_data
- test_unknown_preserves_known_values
- test_unknown_preserves_shape
- test_unknown_with_numeric

**Pipeline Tests (4):**
- test_other_then_integer
- test_novel_then_integer
- test_unknown_then_integer
- test_indicate_na_with_imputation

**Edge Case Tests (4):**
- test_other_all_frequent
- test_integer_single_category
- test_indicate_na_all_missing
- test_novel_with_numeric

## Usage Examples

### Using Selectors
```python
from py_recipes.recipe import recipe
from py_recipes.steps.categorical_extended import StepOther, StepNovel, StepUnknown
from py_recipes.selectors import all_nominal, has_type

# Use all_nominal() selector
rec = recipe(data, "y ~ .") \
    .step_other(all_nominal(), threshold=0.05) \
    .step_novel(all_nominal()) \
    .step_unknown(all_nominal())

# Use has_type() selector
rec = recipe(data, "y ~ .") \
    .step_other(has_type("object"))

# Use explicit columns
rec = recipe(data, "y ~ .") \
    .step_other(["color", "region"])
```

### Integration with Recipe
```python
# Default behavior (all categorical columns)
rec = recipe(iris_data, "Species ~ .") \
    .step_other(threshold=0.01) \
    .step_novel() \
    .step_unknown()

# Selective application
rec = recipe(data, "y ~ .") \
    .step_other(["category", "region"], threshold=0.05) \
    .step_novel(has_type("object"))
```

## Benefits

1. **Consistent API:** All categorical steps now use the same selector pattern as other recipe steps
2. **Flexibility:** Users can specify columns via:
   - Selectors: `all_nominal()`, `has_type("object")`, `has_role("predictor")`
   - Explicit lists: `["col1", "col2"]`
   - Single string: `"column_name"`
   - Callables: Custom selector functions
3. **Backward Compatibility:** Default behavior (None → all categorical) preserved
4. **Type Safety:** Union type hints provide clear API documentation

## Files Modified

- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps/categorical_extended.py` (419 lines)

## Integration Points

These steps integrate with:
- `py_recipes.recipe.Recipe` - Main recipe pipeline
- `py_recipes.selectors` - Column selection utilities
- `py_recipes.steps.__init__` - Recipe step factory methods

## Next Steps

1. Optional: Update `StepInteger` to use `resolve_selector()` for consistency
2. Update recipe factory methods if needed (e.g., `recipe.step_other()`)
3. Add selector usage examples to documentation/notebooks
4. Consider adding selector-specific tests (already covered by existing tests)

## Verification

```bash
# Verify all tests pass
cd /Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels
source py-tidymodels2/bin/activate
python -m pytest tests/test_recipes/test_categorical_extended.py -v

# Expected: 39 passed in ~0.5s
```

## Technical Notes

### Special Handling: StepIndicateNa

`StepIndicateNa` has unique logic because it needs to filter for columns with missing values:

```python
# When columns=None, find all columns with NA
if self.columns is None:
    cols = [col for col in data.columns if data[col].isna().any()]
else:
    # When columns specified, resolve selector then filter for NA
    selector = self.columns
    cols = resolve_selector(selector, data)
    cols = [col for col in cols if data[col].isna().any()]
```

This preserves the step's semantic: only create indicators for columns that actually have missing values.

### Type Signature Pattern

Consistent type signature across all four updated classes:
```python
columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
```

This matches the pattern used in other recipe steps like:
- `py_recipes/steps/impute.py`
- `py_recipes/steps/transformations.py`
- `py_recipes/steps/discretization.py`

## Implementation Quality

- Zero test failures
- No breaking changes to existing API
- Consistent with project's selector infrastructure
- Proper type hints for IDE support
- Clear docstring updates
- Production-ready code with no placeholders
