# Reduction Steps - Selector Function Support

**Date:** 2025-11-09
**Enhancement:** Added selector function support to reduction steps (ICA, kernel PCA, PLS)
**Status:** ✅ COMPLETE

---

## Problem

The dimensionality reduction steps (`step_ica()`, `step_kpca()`, `step_pls()`) only accepted explicit column lists, causing a `TypeError` when selector functions were passed:

```python
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

# This failed with TypeError: 'function' object is not iterable
rec = recipe().step_ica(all_numeric_predictors(), num_comp=3)
```

### Error Details

```
TypeError: 'function' object is not iterable
File py_recipes/steps/reduction.py:49, in StepIca.prep
    cols = [col for col in self.columns if col in data.columns]
```

The code was trying to iterate over `self.columns`, which was a selector function, not a list.

---

## Solution

Updated all three reduction steps to support selector functions using the same pattern as other recipe steps:

1. Changed type hints to accept `Union[None, str, List[str], Callable]`
2. Added `resolve_selector()` import and usage
3. Added numeric-only filtering after selector resolution
4. Updated docstrings with selector examples

---

## Implementation

### Files Modified

**`py_recipes/steps/reduction.py`** - Updated all 3 reduction step classes

#### Changes Applied to All Three Steps

**1. Added Imports (lines 8, 11):**
```python
from typing import List, Optional, Any, Union, Callable
from py_recipes.selectors import resolve_selector
```

**2. Updated Type Hints:**
```python
# OLD
columns: Optional[List[str]] = None

# NEW
columns: Union[None, str, List[str], Callable] = None
```

**3. Updated Selector Resolution:**

**StepIca (lines 47-52):**
```python
# OLD
if self.columns is None:
    cols = data.select_dtypes(include=[np.number]).columns.tolist()
else:
    cols = [col for col in self.columns if col in data.columns]

# NEW
if self.columns is None:
    cols = data.select_dtypes(include=[np.number]).columns.tolist()
else:
    cols = resolve_selector(self.columns, data)
    # Filter to numeric only
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col])]
```

**StepKpca (lines 159-164):**
```python
# Same pattern as StepIca
```

**StepPls (lines 278-285):**
```python
# OLD
if self.columns is None:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols = [col for col in numeric_cols if col != self.outcome]
else:
    cols = [col for col in self.columns if col in data.columns and col != self.outcome]

# NEW
if self.columns is None:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols = [col for col in numeric_cols if col != self.outcome]
else:
    cols = resolve_selector(self.columns, data)
    # Filter to numeric only and exclude outcome
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col]) and col != self.outcome]
```

**4. Updated Docstrings:**
```python
# OLD
columns: Columns to apply ICA to (None = all numeric)

# NEW
columns: Columns to apply ICA to (None = all numeric, supports selectors)
```

---

## Usage

### Before Fix (Explicit Lists Only)

```python
from py_recipes import recipe

# Only way to use reduction steps
rec = recipe().step_ica(['x1', 'x2', 'x3'], num_comp=2)
rec = recipe().step_kpca(['x1', 'x2', 'x3'], num_comp=2)
rec = recipe().step_pls(['x1', 'x2', 'x3'], outcome='target', num_comp=2)
```

### After Fix (Selectors Supported)

```python
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors, all_numeric, contains

# Option 1: Explicit list (still works)
rec = recipe().step_ica(['x1', 'x2', 'x3'], num_comp=2)

# Option 2: Selector function (NEW)
rec = recipe().step_ica(all_numeric_predictors(), num_comp=3)
rec = recipe().step_kpca(all_numeric_predictors(), num_comp=3)
rec = recipe().step_pls(all_numeric_predictors(), outcome='target', num_comp=3)

# Option 3: Pattern matching (NEW)
rec = recipe().step_ica(contains('feature'), num_comp=2)

# Option 4: All numeric columns (NEW)
rec = recipe().step_ica(all_numeric(), num_comp=3)
```

---

## Examples

### Example 1: ICA with Selector

```python
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'x4': np.random.randn(100),
    'category': ['A', 'B'] * 50,
    'target': np.random.randn(100)
})

# Automatically applies ICA to all numeric predictor columns
rec = recipe().step_ica(all_numeric_predictors(), num_comp=3)
prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: x1, x2, x3, x4 replaced with IC1, IC2, IC3
print(baked.columns)
# Output: ['category', 'target', 'IC1', 'IC2', 'IC3']
```

### Example 2: Kernel PCA in Workflow

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Non-linear dimensionality reduction
rec = recipe().step_kpca(all_numeric_predictors(), num_comp=2, kernel='rbf')

wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg().set_engine("sklearn"))
)

fit = wf.fit(data)
```

### Example 3: PLS Regression

```python
# Supervised dimensionality reduction with PLS
rec = recipe().step_pls(
    all_numeric_predictors(),
    outcome='target',
    num_comp=3
)

prepped = rec.prep(data)
baked = prepped.bake(data)

# Result: Predictors replaced with PLS components maximizing correlation with target
```

### Example 4: Combined with Other Steps

```python
rec = (
    recipe()
    # 1. Impute missing values
    .step_impute_median(all_numeric_predictors())

    # 2. Normalize features
    .step_normalize(all_numeric_predictors())

    # 3. Apply ICA for feature extraction
    .step_ica(all_numeric_predictors(), num_comp=5)
)
```

---

## Available Selectors

All selectors work with reduction steps:

**Type Selectors:**
- `all_numeric()` - All numeric columns
- `all_numeric_predictors()` - All numeric predictors (excludes outcome)

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
✅ Test 1: step_ica() with all_numeric_predictors()
   Creates IC1, IC2, IC3 components from x1-x4

✅ Test 2: step_kpca() with all_numeric_predictors()
   Creates KPC1, KPC2, KPC3 components

✅ Test 3: step_pls() with all_numeric_predictors()
   Creates PLS1, PLS2, PLS3 components

✅ Test 4: Full workflow with step_ica()
   Workflow fits and makes predictions successfully
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code with explicit lists: No changes needed
- All existing tests pass without modification
- New selector functionality is additive, not breaking

```python
# Old code still works exactly the same
rec = recipe().step_ica(['x1', 'x2', 'x3'], num_comp=2)

# New code adds functionality
rec = recipe().step_ica(all_numeric_predictors(), num_comp=2)
```

---

## Resolves

This fix resolves the `TypeError` when using selector functions with:
- `step_ica()` - Independent Component Analysis
- `step_kpca()` - Kernel Principal Component Analysis
- `step_pls()` - Partial Least Squares

```python
# All of these now work without errors:
rec = (
    recipe()
    .step_ica(all_numeric_predictors(), num_comp=3)
    .step_kpca(all_numeric_predictors(), num_comp=3)
    .step_pls(all_numeric_predictors(), outcome='target', num_comp=3)
)
```

---

## Related Features

Works seamlessly with:
- ✅ All selector functions
- ✅ Workflow integration
- ✅ Recipe chaining
- ✅ Both supervised (PLS) and unsupervised (ICA, KPCA) reduction

---

## Summary

All three reduction steps now accept:

1. **Explicit lists**: `["x1", "x2", "x3"]`
2. **Selectors**: `all_numeric_predictors()`
3. **Pattern matching**: `contains("feature")`

All three steps perform dimensionality reduction on numeric columns:
- **ICA**: Blind source separation, independent components
- **Kernel PCA**: Non-linear dimensionality reduction
- **PLS**: Supervised reduction maximizing outcome correlation

**Status:** COMPLETE - Feature implemented, tested, and documented.

---

## Documentation Files

- `.claude_debugging/REDUCTION_STEPS_SELECTOR_SUPPORT.md` (this file)
- `py_recipes/steps/reduction.py` - Implementation
- `py_recipes/recipe.py` - Recipe methods

**Restart Jupyter Kernel:** After reinstalling package, restart kernel to pick up changes.
