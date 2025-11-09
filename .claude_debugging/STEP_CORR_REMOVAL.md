# step_corr() Removal - Code Cleanup

**Date:** 2025-11-09
**Action:** Removed redundant `step_corr()` function from library
**Reason:** `step_select_corr()` covers the same functionality with more features
**Status:** ✅ COMPLETE

---

## Background

The library had two correlation-based feature selection steps:

1. **`step_corr()`** - Basic multicollinearity removal
   - Only removed highly correlated predictors
   - Single method: remove one from each correlated pair

2. **`step_select_corr()`** - Advanced correlation-based selection
   - **Same functionality as step_corr()** via `method='multicollinearity'`
   - **Additional functionality**: outcome-based correlation filtering via `method='outcome'`
   - More flexible and comprehensive

### Why Remove step_corr()?

**Redundancy:** `step_select_corr()` does everything `step_corr()` does, plus more:

```python
# OLD: step_corr() for multicollinearity
rec = recipe().step_corr(threshold=0.9)

# NEW: step_select_corr() does the same thing
rec = recipe().step_select_corr(
    outcome='target',
    threshold=0.9,
    method='multicollinearity'  # Same as step_corr()
)

# PLUS: step_select_corr() can also do outcome-based filtering
rec = recipe().step_select_corr(
    outcome='target',
    threshold=0.3,
    method='outcome'  # Keep features correlated with outcome
)
```

**Code Maintenance:** Having two similar functions:
- Confuses users about which to use
- Increases maintenance burden
- Duplicates testing effort
- Creates documentation overhead

---

## Files Removed/Modified

### Removed Files
- `tests/test_recipes/test_step_corr.py` - Test file for step_corr()

### Modified Files
1. **`py_recipes/steps/feature_selection.py`**
   - Removed `StepCorr` class (lines 233-329)
   - Removed `PreparedStepCorr` class (lines 332-359)

2. **`py_recipes/recipe.py`**
   - Removed `step_corr()` method (lines 441-478)

3. **`py_recipes/steps/__init__.py`**
   - Removed `StepCorr` from imports (line 34)
   - Removed `PreparedStepCorr` from imports (line 35)
   - Removed `"StepCorr"` from `__all__` list (line 181)
   - Removed `"PreparedStepCorr"` from `__all__` list (line 182)

---

## Migration Guide

If you were using `step_corr()`, migrate to `step_select_corr()`:

### Before (Removed)
```python
from py_recipes import recipe

# Remove multicollinear features
rec = recipe().step_corr(threshold=0.9)

# With custom correlation method
rec = recipe().step_corr(threshold=0.85, method='spearman')

# With specific columns
rec = recipe().step_corr(columns=['x1', 'x2', 'x3'], threshold=0.9)
```

### After (Current)
```python
from py_recipes import recipe

# Same functionality with step_select_corr()
rec = recipe().step_select_corr(
    outcome='target',
    threshold=0.9,
    method='multicollinearity',
    corr_method='pearson'
)

# With Spearman correlation
rec = recipe().step_select_corr(
    outcome='target',
    threshold=0.85,
    method='multicollinearity',
    corr_method='spearman'
)

# Note: step_select_corr() processes all numeric predictors by default
# (columns parameter not needed for typical use)
```

### Migration Notes

**Key Differences:**
1. `step_select_corr()` requires `outcome` parameter
2. Use `method='multicollinearity'` for step_corr() behavior
3. Use `corr_method` parameter instead of `method` for correlation type
4. No `columns` parameter needed (processes all numeric predictors)

**Additional Benefits:**
```python
# NEW: Can also filter by correlation with outcome
rec = recipe().step_select_corr(
    outcome='target',
    threshold=0.3,
    method='outcome'  # Keep features with |corr| > 0.3 with outcome
)
```

---

## Verification Tests

All verification tests passed:

```
✅ Test 1: step_corr() properly removed
   AttributeError: 'Recipe' object has no attribute 'step_corr'

✅ Test 2: step_select_corr() still works correctly
   Successfully filters multicollinear features

✅ Test 3: StepCorr not importable
   ImportError when trying to import StepCorr
```

---

## Impact Analysis

### Breaking Changes
- ❌ `recipe().step_corr()` will raise `AttributeError`
- ❌ `from py_recipes.steps import StepCorr` will raise `ImportError`

### What Still Works
- ✅ `step_select_corr()` with `method='multicollinearity'` provides same functionality
- ✅ All other recipe steps unchanged
- ✅ No impact on existing code using `step_select_corr()`

### Recommended Action
**Search and replace** in your code:
```bash
# Find usages
grep -r "step_corr" your_notebooks/

# Replace with step_select_corr
# Add outcome parameter and method='multicollinearity'
```

---

## Benefits of This Cleanup

1. **Reduced Complexity**
   - One clear function for correlation-based selection
   - Easier for users to find the right tool

2. **Better Functionality**
   - `step_select_corr()` has both multicollinearity AND outcome filtering
   - More powerful and flexible

3. **Easier Maintenance**
   - Less code to maintain
   - Fewer tests to update
   - Simpler documentation

4. **Clearer API**
   - No confusion about which correlation function to use
   - Single source of truth for correlation-based selection

---

## step_select_corr() Feature Comparison

| Feature | step_corr() (REMOVED) | step_select_corr() (KEEP) |
|---------|----------------------|---------------------------|
| **Multicollinearity removal** | ✅ | ✅ (method='multicollinearity') |
| **Outcome correlation filtering** | ❌ | ✅ (method='outcome') |
| **Custom correlation method** | ✅ | ✅ (corr_method parameter) |
| **Threshold control** | ✅ | ✅ |
| **Column selection** | ✅ | ✅ (automatic for predictors) |

---

## Related Documentation

- **step_select_corr() usage**: See CLAUDE.md section on Feature Selection
- **Recipe step catalog**: See COMPLETE_RECIPE_REFERENCE.md
- **Migration examples**: See examples/10_recipes_comprehensive_demo.ipynb

---

## Summary

✅ **CLEANUP COMPLETE** - step_corr() removed from library

- Redundant functionality eliminated
- Users should use `step_select_corr()` instead
- Migration is straightforward (add outcome parameter, method='multicollinearity')
- Library is now simpler and more maintainable
- `step_select_corr()` provides same functionality plus more

**Action Required:** Update any code using `step_corr()` to use `step_select_corr()` with `method='multicollinearity'`

**Restart Jupyter Kernel:** After reinstalling package, restart kernel to pick up changes
