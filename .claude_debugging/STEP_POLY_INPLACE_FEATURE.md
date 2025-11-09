# step_poly() inplace Parameter

**Date:** 2025-11-09
**Feature:** Added `inplace` parameter to `step_poly()`
**Status:** ✅ COMPLETE

---

## Overview

`step_poly()` now supports an `inplace` parameter, matching the behavior of transformation steps like `step_log()`, `step_sqrt()`, etc.

---

## Usage

### Default Behavior (inplace=True)

**Replaces original columns with polynomial features:**

```python
from py_recipes import recipe

rec = recipe().step_poly(['x1', 'x2'], degree=2)
# or explicitly:
rec = recipe().step_poly(['x1', 'x2'], degree=2, inplace=True)

prepped = rec.prep(data)
baked = prepped.bake(data)

# Before: ['x1', 'x2', 'target']
# After:  ['target', 'x1^2', 'x2^2']  (original columns removed)
```

### New Behavior (inplace=False)

**Keeps original columns AND adds polynomial features:**

```python
rec = recipe().step_poly(['x1', 'x2'], degree=2, inplace=False)

prepped = rec.prep(data)
baked = prepped.bake(data)

# Before: ['x1', 'x2', 'target']
# After:  ['x1', 'x2', 'target', 'x1^2', 'x2^2']  (originals kept)
```

---

## Examples

### Example 1: Feature Comparison

Keep originals to compare with polynomial features:

```python
rec = (recipe()
    .step_poly(all_numeric_predictors(), degree=2, inplace=False)
)

# Result has both x1, x2 AND x1^2, x2^2
# Useful for model comparison or feature importance analysis
```

### Example 2: Feature Engineering Pipeline

Combine originals with polynomials for richer feature set:

```python
rec = (recipe()
    .step_normalize(all_numeric_predictors())           # Normalize first
    .step_poly(all_numeric_predictors(), degree=2, inplace=False)  # Add polynomials
    .step_corr(threshold=0.9)                           # Remove multicollinearity
)

# Creates a rich feature set: normalized originals + polynomials
# Then removes highly correlated features
```

### Example 3: With Interactions

```python
rec = recipe().step_poly(
    ['x1', 'x2', 'x3'],
    degree=2,
    include_interactions=True,
    inplace=False
)

# Before: ['x1', 'x2', 'x3', 'target']
# After:  ['x1', 'x2', 'x3', 'target',  (originals)
#          'x1^2', 'x2^2', 'x3^2',      (polynomials)
#          'x1_x2', 'x1_x3', 'x2_x3']   (interactions)
```

### Example 4: Memory Efficient (Default)

When you don't need originals, use default `inplace=True`:

```python
rec = recipe().step_poly(
    all_numeric_predictors(),
    degree=3,
    inplace=True  # Default, can omit
)

# Replaces x1, x2, x3 with x1^2, x1^3, x2^2, x2^3, x3^2, x3^3
# More memory efficient for large datasets
```

---

## Implementation Details

### Files Modified

1. **`py_recipes/steps/basis.py`**
   - Added `inplace: bool = True` to `StepPoly` dataclass (line 305)
   - Added `inplace: bool = True` to `PreparedStepPoly` dataclass (line 388)
   - Updated `prep()` to pass `inplace` parameter (line 369)
   - Updated `bake()` to conditionally drop columns (lines 420-422)

2. **`py_recipes/recipe.py`**
   - Added `inplace: bool = True` parameter to `step_poly()` method (line 1107)
   - Updated docstring (line 1116)
   - Pass parameter to `StepPoly()` (line 1122)

### Code Changes

**In `bake()` method:**

```python
# Before:
result = result.drop(columns=self.columns)  # Always dropped

# After:
if self.inplace:
    result = result.drop(columns=self.columns)  # Conditional
```

---

## Test Results

All tests passing:

```
[Test 1] inplace=True (default)
✅ PASS: Original columns removed
✅ PASS: Polynomial columns created
   Shape: (5, 3) - Replaced

[Test 2] inplace=False
✅ PASS: Original columns kept
✅ PASS: Polynomial columns created
   Shape: (5, 5) - Added

[Test 3] With selector and inplace=False
✅ PASS: Original predictor columns kept

[Test 4] With interactions and inplace=False
✅ PASS: All expected columns present
   Originals + polynomials + interactions

Existing test suite:
tests/test_recipes/test_basis.py::TestStepPoly  6/6 PASSING ✅
```

---

## Comparison with Other Steps

`step_poly()` now matches the behavior of transformation steps:

| Step | inplace=True | inplace=False |
|------|--------------|---------------|
| `step_log()` | Replaces columns | Adds `_log` suffix |
| `step_sqrt()` | Replaces columns | Adds `_sqrt` suffix |
| `step_boxcox()` | Replaces columns | Adds `_boxcox` suffix |
| `step_poly()` | Replaces columns | **Adds `^2`, `_x2` etc.** |

---

## Backward Compatibility

✅ **Fully backward compatible**

- Default is `inplace=True` (replaces columns), matching previous behavior
- Existing code without `inplace` parameter works identically
- All existing tests pass without modification

---

## Use Cases

### When to use `inplace=False`:

1. **Feature Comparison**: Compare model performance with/without polynomials
2. **Feature Importance**: Analyze which features (original vs polynomial) are most important
3. **Hybrid Models**: Some algorithms benefit from both linear and polynomial terms
4. **Debugging**: Keep originals to verify polynomial calculations
5. **Interpretability**: Maintain original features for easier interpretation

### When to use `inplace=True` (default):

1. **Memory Efficiency**: Large datasets with many features
2. **Feature Replacement**: You only want polynomial terms, not originals
3. **Dimensionality Reduction**: Replace features to keep feature count manageable
4. **Standard Pipeline**: When originals aren't needed downstream

---

## Related Features

Works seamlessly with:
- ✅ `include_interactions` parameter (create both polynomials + interactions)
- ✅ Selectors (`all_numeric_predictors()`, etc.)
- ✅ All degrees (degree=2, 3, 4, etc.)
- ✅ Workflow integration
- ✅ Recipe chaining

---

## Summary

```python
# Default (backward compatible):
step_poly(columns, degree=2)
# Replaces: x → x^2

# New feature:
step_poly(columns, degree=2, inplace=False)
# Keeps both: x, x^2

# With interactions:
step_poly(columns, degree=2, include_interactions=True, inplace=False)
# Keeps all: x, y, x^2, y^2, x_y
```

**Status:** COMPLETE - Feature implemented, tested, and documented.
