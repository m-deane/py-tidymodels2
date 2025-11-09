# step_dummy() Selector Support

**Date:** 2025-11-09
**Feature:** Added selector function support to `step_dummy()`
**Status:** ✅ COMPLETE

---

## Problem

The `step_dummy()` function only accepted explicit column lists, causing a `TypeError` when selector functions were passed:

```python
rec = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4)
    .step_dummy(all_nominal_predictors())  # ❌ TypeError: 'function' object is not iterable
)
```

### Error Details

```
TypeError: 'function' object is not iterable
File py_recipes/steps/dummy.py:41, in StepDummy.prep
    existing_cols = [col for col in self.columns if col in data.columns]
```

The code was trying to iterate over `self.columns`, which was a selector function, not a list.

---

## Solution

Updated `step_dummy()` to support selector functions using the same pattern as other recipe steps:

1. Changed type hint from `List[str]` to `Union[List[str], Callable]`
2. Added selector resolution in `prep()` method
3. Updated documentation with selector examples

---

## Implementation

### Files Modified

**1. `py_recipes/steps/dummy.py`**

```python
# OLD
from typing import List, Any
import pandas as pd

@dataclass
class StepDummy:
    columns: List[str]
    one_hot: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True):
        # Filter to existing columns
        existing_cols = [col for col in self.columns if col in data.columns]
```

```python
# NEW
from typing import List, Any, Union, Callable
import pandas as pd

@dataclass
class StepDummy:
    columns: Union[List[str], Callable]  # ← Now accepts selectors
    one_hot: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True):
        from py_recipes.selectors import resolve_selector

        # Resolve selector to column list
        if callable(self.columns):
            cols = resolve_selector(self.columns, data)
        else:
            cols = self.columns

        # Filter to existing columns
        existing_cols = [col for col in cols if col in data.columns]
```

**2. `py_recipes/recipe.py`**

Updated method signature and docstring:

```python
def step_dummy(
    self,
    columns: Union[List[str], Callable],  # ← Updated type
    one_hot: bool = True
) -> "Recipe":
    """
    Create dummy variables from categorical columns.

    Args:
        columns: Categorical columns to encode (list or selector function)  # ← Updated
        one_hot: Use one-hot encoding (True) or integer encoding (False)

    Examples:
        >>> rec = Recipe().step_dummy(["category", "group"])
        >>> rec = Recipe().step_dummy(all_nominal_predictors())  # ← NEW
    """
```

---

## Usage

### Before (Explicit List Only)

```python
from py_recipes import recipe

# Only way to use step_dummy
rec = recipe().step_dummy(['category1', 'category2'])
```

### After (Selectors Supported)

```python
from py_recipes import recipe
from py_recipes.selectors import all_nominal_predictors, all_nominal, contains

# Option 1: Explicit list (still works)
rec = recipe().step_dummy(['category1', 'category2'])

# Option 2: Selector function (NEW)
rec = recipe().step_dummy(all_nominal_predictors())

# Option 3: Pattern matching (NEW)
rec = recipe().step_dummy(contains('category'))

# Option 4: All nominal columns (NEW)
rec = recipe().step_dummy(all_nominal())
```

---

## Examples

### Example 1: Basic Selector Usage

```python
import pandas as pd
from py_recipes import recipe
from py_recipes.selectors import all_nominal_predictors

data = pd.DataFrame({
    'x1': [1, 2, 3],
    'x2': [4, 5, 6],
    'category1': ['A', 'B', 'A'],
    'category2': ['X', 'Y', 'X'],
    'target': [10, 20, 30]
})

# Automatically encodes all nominal predictor columns
rec = recipe().step_dummy(all_nominal_predictors())
prepped = rec.prep(data)
baked = prepped.bake(data)

print(baked.columns)
# Output: ['x1', 'x2', 'target',
#          'category1_A', 'category1_B',
#          'category2_X', 'category2_Y']
```

### Example 2: Discretize → Dummy Pattern

This is the pattern that was failing in forecasting_recipes.ipynb:

```python
# Discretize numeric → encode bins as dummy variables
rec = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4, method="quantile")
    .step_dummy(all_nominal_predictors())  # ✅ Now works!
)

# Before: numeric columns (x1, x2, x3)
# After step_discretize: categorical bins (x1_bin_1, x1_bin_2, etc.)
# After step_dummy: one-hot encoded (x1_bin_1, x1_bin_2, etc. as 0/1 columns)
```

### Example 3: Pattern-Based Selection

```python
from py_recipes.selectors import contains, starts_with

# Encode all columns containing 'cat'
rec = recipe().step_dummy(contains('cat'))

# Encode all columns starting with 'factor_'
rec = recipe().step_dummy(starts_with('factor_'))
```

### Example 4: In Workflows

```python
from py_workflows import workflow
from py_parsnip import linear_reg

wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_dummy(all_nominal_predictors())
        .step_normalize(all_numeric_predictors())
    )
    .add_model(linear_reg())
)

fit = wf.fit(train_data)
```

### Example 5: Combined with Other Steps

```python
rec = (
    recipe()
    # 1. Create categorical bins from continuous
    .step_discretize(['age', 'income'], num_breaks=5)

    # 2. Encode all categorical columns (original + bins)
    .step_dummy(all_nominal_predictors())

    # 3. Normalize numeric columns
    .step_normalize(all_numeric_predictors())
)
```

---

## Available Selectors

All selectors work with `step_dummy()`:

**Type Selectors:**
- `all_nominal()` - All categorical columns
- `all_nominal_predictors()` - All categorical predictors (excludes outcome)
- `all_string()` - All string columns

**Pattern Selectors:**
- `contains(substring)` - Columns containing substring
- `starts_with(prefix)` - Columns starting with prefix
- `ends_with(suffix)` - Columns ending with suffix
- `matches(pattern)` - Regex pattern matching

**Utility Selectors:**
- `one_of(*columns)` - Specific column list
- `where(predicate)` - Custom predicate function

---

## Test Results

All tests passing:

```
✅ Test 1: Explicit column list
   Works as before (backward compatible)

✅ Test 2: all_nominal_predictors() selector
   Selects and encodes all categorical predictors

✅ Test 3: all_nominal() selector
   Selects and encodes all categorical columns

✅ Test 4: Workflow Integration
   step_discretize → step_dummy pattern works

✅ Test 5: Full Workflow with Model
   Workflow with step_dummy selector fits successfully
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code with explicit lists: No changes needed
- All existing tests pass without modification
- New selector functionality is additive, not breaking

```python
# Old code still works exactly the same
rec = recipe().step_dummy(['category1', 'category2'])

# New code adds functionality
rec = recipe().step_dummy(all_nominal_predictors())
```

---

## Resolves

This fix resolves the `TypeError` in forecasting_recipes.ipynb cell 88:

```python
# This now works without errors:
rec_discretize = (
    recipe()
    .step_discretize(all_numeric_predictors(), num_breaks=4, method="quantile")
    .step_dummy(all_nominal_predictors())  # ✅ Fixed!
)
```

---

## Related Features

Works seamlessly with:
- ✅ All selector functions
- ✅ step_discretize() (categorical binning)
- ✅ Workflow integration
- ✅ Recipe chaining
- ✅ Both one-hot and label encoding modes

---

## Summary

`step_dummy()` now accepts:

1. **Explicit lists**: `["category1", "category2"]`
2. **Selectors**: `all_nominal_predictors()`
3. **Pattern matching**: `contains("category")`

All three modes create dummy variables from categorical columns using sklearn's OneHotEncoder.

**Status:** COMPLETE - Feature implemented, tested, and documented.

---

## Documentation Files

- `.claude_debugging/STEP_DUMMY_SELECTOR_SUPPORT.md` (this file)
- `py_recipes/steps/dummy.py` - Implementation
- `py_recipes/recipe.py` - Recipe method
