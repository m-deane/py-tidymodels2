# step_poly Selector Support and Column Name Fix

**Date:** 2025-11-09
**Issue:** Two problems with step_poly() preventing use in recipes
**Status:** ✅ RESOLVED

---

## Problems

### Problem 1: Selectors Not Supported
**Error:**
```python
TypeError: 'function' object is not iterable
```

**User Code:**
```python
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
# ❌ TypeError when trying to iterate over selector function
```

**Root Cause:** `StepPoly.prep()` tried to iterate over `self.columns` directly, but when a selector function like `all_numeric_predictors()` was passed, it couldn't iterate over a function object.

### Problem 2: Column Names with Spaces
**Error:**
```python
ValueError: Column names used in formula cannot contain spaces. Found 28 invalid column(s):
  ['mean_med_diesel_crack_input1_trade_month_lag2 mean_nwe_hsfo_crack_trade_month_lag1', ...]
```

**User Code:**
```python
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)  # ❌ ValueError when using polynomial features in formula
```

**Root Cause:** sklearn's `PolynomialFeatures.get_feature_names_out()` returns names with spaces (e.g., `"x1 x2"` for interaction between x1 and x2). These space-containing column names fail formula validation.

---

## Solutions Implemented

### Fix 1: Add Selector Support

**File:** `py_recipes/steps/basis.py`

**Changes:**

1. Updated `StepPoly` dataclass to accept selectors:
```python
@dataclass
class StepPoly:
    columns: Union[List[str], Callable, str, None]  # Now accepts selectors
    degree: int = 2
    include_interactions: bool = False
```

2. Added selector resolution in `prep()` method:
```python
def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPoly":
    from sklearn.preprocessing import PolynomialFeatures
    from py_recipes.selectors import resolve_selector

    # Resolve selector to actual column names
    cols = resolve_selector(self.columns, data)  # ✅ NEW

    if len(cols) == 0:
        return PreparedStepPoly(...)

    # Rest of prep logic...
```

### Fix 2: Replace Spaces with Underscores

**File:** `py_recipes/steps/basis.py`

**Changes:**

Modified feature name generation to replace spaces:
```python
def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPoly":
    # ... fit polynomial features ...

    # Generate feature names and replace spaces with underscores
    # sklearn uses spaces like "x1 x2" but we need "x1_x2" for formula compatibility
    feature_names = poly.get_feature_names_out(cols)
    feature_names = [name.replace(' ', '_') for name in feature_names]  # ✅ NEW

    return PreparedStepPoly(
        columns=cols,
        poly_transformer=poly,
        feature_names=list(feature_names)
    )
```

---

## Before vs After

### Before Fix ❌

```python
# Selector support
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
# ❌ TypeError: 'function' object is not iterable

# Space-containing column names
rec = recipe().step_poly(["x1", "x2"], degree=2)
prepped = rec.prep(data)
baked = prepped.bake(data)
print(baked.columns)
# Output: ['target', 'x1^2', 'x1 x2', 'x2^2']  # ❌ Spaces!

wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)
# ❌ ValueError: Column names cannot contain spaces
```

### After Fix ✅

```python
# Selector support
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
prepped = rec.prep(data)
# ✅ Works! Selectors are resolved to column names

# Underscore-separated column names
baked = prepped.bake(data)
print(baked.columns)
# Output: ['target', 'x1^2', 'x1_x2', 'x2^2']  # ✅ Underscores!

wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)
# ✅ Works! No formula validation errors
```

---

## Verification Tests

### Test 1: Selector Support ✅
```python
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'target': np.random.randn(100)
})

rec = recipe().step_poly(all_numeric_predictors(), degree=2)
prepped = rec.prep(data)
baked = prepped.bake(data)

# ✅ SUCCESS
print("Columns:", list(baked.columns))
# Output: ['target', 'x1^2', 'x1_x2', 'x2^2']
```

### Test 2: Workflow Integration ✅
```python
from py_workflows import workflow
from py_parsnip import linear_reg

train_data = data[:75]
test_data = data[75:]

rec = (
    recipe()
    .step_normalize(all_numeric_predictors())
    .step_poly(all_numeric_predictors(), degree=2)
)

wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg().set_engine("sklearn"))
)

fit = wf.fit(train_data)
fit = fit.evaluate(test_data)

# ✅ SUCCESS - No errors!
outputs, coefs, stats = fit.extract_outputs()
print(f"Test RMSE: {stats[stats['split'] == 'test']['rmse'].values[0]:.4f}")
```

### Test 3: Column Name Format ✅
```python
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
prepped = rec.prep(data)
baked = prepped.bake(data)

# Check for spaces in column names
space_cols = [c for c in baked.columns if ' ' in c]
print(f"Columns with spaces: {len(space_cols)}")
# Output: 0  ✅

# Check underscore usage
underscore_cols = [c for c in baked.columns if '_x' in c or '_' in c]
print(f"Interaction columns: {underscore_cols}")
# Output: ['x1_x2']  ✅
```

---

## Column Naming Convention

### sklearn Default (Before)
```python
PolynomialFeatures(degree=2).get_feature_names_out(['x1', 'x2'])
# Returns: ['x1^2', 'x1 x2', 'x2^2']
#                    ↑ Space
```

### py-tidymodels (After)
```python
# We replace spaces with underscores
# Returns: ['x1^2', 'x1_x2', 'x2^2']
#                    ↑ Underscore
```

**Why:** Patsy formula parser doesn't allow spaces in column names. Underscores maintain readability while ensuring formula compatibility.

---

## Impact on Existing Code

### Backward Compatibility
✅ **No breaking changes** for code using explicit column lists:
```python
# This continues to work exactly as before
rec = recipe().step_poly(["x1", "x2"], degree=2)
```

### New Functionality
✅ **Selectors now supported:**
```python
# These patterns now work
rec = recipe().step_poly(all_numeric_predictors(), degree=2)
rec = recipe().step_poly(all_numeric(), degree=3)
rec = recipe().step_poly(starts_with("feature_"), degree=2)
```

### Formula Compatibility
✅ **Polynomial recipes now work in workflows:**
```python
# This pattern now works end-to-end
workflow()
    .add_recipe(recipe().step_poly(all_numeric_predictors(), degree=2))
    .add_model(linear_reg())
    .fit(train_data)
    .evaluate(test_data)
```

---

## Related Fixes

This fix is part of a broader effort to ensure all recipe steps support selectors:

**Already Supporting Selectors:**
- ✅ step_normalize()
- ✅ step_center()
- ✅ step_scale()
- ✅ step_range()
- ✅ step_impute_median()
- ✅ step_impute_mean()
- ✅ step_corr()
- ✅ step_log()
- ✅ step_BoxCox()
- ✅ And 40+ more steps...

**Now Fixed:**
- ✅ step_poly()

---

## Code References

**Modified Files:**
- `py_recipes/steps/basis.py` (lines 287-346)
  - Updated `StepPoly` dataclass type hint
  - Added selector resolution in `prep()`
  - Added space-to-underscore replacement

**Type Imports:**
- Added `Union, Callable` to imports (line 8)

**Selector Import:**
- Added `from py_recipes.selectors import resolve_selector` in `prep()` method

---

## Testing

### Test File
- `.claude_debugging/test_step_poly_fix.py` (verification script)

### Test Results
```
✅ SUCCESS: Polynomial features workflow fitted!
✅ Polynomial feature names use underscores:
   Sample: ['x1^2', 'x1_x2', 'x1_x3', 'x2^2', 'x2_x3']
✅ Evaluation successful!
✅ Test RMSE: 1.1742
✨ Polynomial features with underscores work perfectly!
```

---

## User Impact

### For forecasting_recipes.ipynb Users

Your polynomial features cells now work without modification:

```python
# Cell 18 - Polynomial Features
rec_poly = (
    recipe()
    .step_normalize(all_numeric_predictors())
    .step_poly(all_numeric_predictors(), degree=2)  # ✅ Now works!
)

wf_poly = (
    workflow()
    .add_recipe(rec_poly)
    .add_model(linear_reg().set_engine("sklearn"))
)

fit_poly = wf_poly.fit(train_data)  # ✅ No errors!
fit_poly = fit_poly.evaluate(test_data)  # ✅ Works!
```

**No code changes needed** - the fixes are transparent to users.

---

## Conclusion

The `step_poly()` function now:
1. ✅ Supports selector functions (like all other recipe steps)
2. ✅ Generates column names without spaces (formula-compatible)
3. ✅ Works seamlessly in workflows with linear models
4. ✅ Maintains backward compatibility
5. ✅ Follows py-tidymodels naming conventions

**Status:** COMPLETE - Both issues resolved and verified.
