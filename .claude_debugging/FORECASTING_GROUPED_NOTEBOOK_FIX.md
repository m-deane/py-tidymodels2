# Forecasting Grouped Notebook Fixes

**Date:** 2025-11-10
**Notebook:** `_md/forecasting_grouped.ipynb`
**Issue:** Cells 78 and 80 were using incorrect workflow fitting methods for grouped/panel data

## Problems Fixed

### Cell 78 - Formula Approach

**Before:**
```python
wf_formula = (
    workflow()
    .add_formula(FORMULA_STR)  # Wrong formula
    .add_model(linear_reg())
)

fit_formula = wf_formula.fit(train_processed)  # Wrong: using .fit() and train_processed
fit_formula = fit_formula.evaluate(test_processed)
```

**After:**
```python
wf_formula = (
    workflow()
    .add_formula(FORMULA_NESTED)  # Correct: uses FORMULA_NESTED
    .add_model(linear_reg())
)

# For grouped data, use fit_nested
fit_formula = wf_formula.fit_nested(train_mix, group_col='country')  # Correct: fit_nested with train_mix
fit_formula = fit_formula.evaluate(test_mix)
```

**Changes:**
1. Changed `.fit()` to `.fit_nested(train_mix, group_col='country')`
2. Changed `FORMULA_STR` to `FORMULA_NESTED`
3. Changed from `train_processed`/`test_processed` to `train_mix`/`test_mix`

### Cell 80 - Recipe Approach

**Before:**
```python
wf_recipe = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

fit_recipe = wf_recipe.fit(train_mix)  # Wrong: using .fit() instead of fit_nested
fit_recipe = fit_recipe.evaluate(test_mix)
```

**After:**
```python
wf_recipe = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

# For grouped data, use fit_nested
fit_recipe = wf_recipe.fit_nested(train_mix, group_col='country')  # Correct: fit_nested
fit_recipe = fit_recipe.evaluate(test_mix)
```

**Changes:**
1. Changed `.fit(train_mix)` to `.fit_nested(train_mix, group_col='country')`

## Why These Changes Matter

### Grouped/Panel Data Modeling
This notebook works with panel data (multiple countries over time). For panel data, there are two approaches:

1. **Nested/Per-Group** (`fit_nested()`): Fits independent model for each group
   - Best when groups have different patterns
   - Each country gets its own model
   - Returns `NestedWorkflowFit` with unified interface

2. **Global** (`fit_global()`): Fits single model with group as feature
   - Best when groups share similar patterns
   - More efficient with limited data per group

The cells were incorrectly using standard `.fit()` which doesn't handle grouped data properly.

### Data Variables
- `train_mix` / `test_mix`: Correct variables (aliases for `train_data` / `test_data`)
- `train_processed` / `test_processed`: Recipe-processed data from Cell 76, NOT appropriate for grouped modeling

### Formula Variables
- `FORMULA_NESTED = "refinery_kbd ~ ."`: Correct formula for nested/grouped modeling
- `FORMULA_STR = "refinery_kbd ~ ."`: Alternative formula definition (same result but wrong context)

## Verification

After fixes, both cells now:
- ✓ Use `fit_nested()` for grouped modeling
- ✓ Use correct data variables (`train_mix`, `test_mix`)
- ✓ Use appropriate formulas (`FORMULA_NESTED`)
- ✓ Include `group_col='country'` parameter
- ✓ Maintain all diagnostic and visualization code

## Related Code References

**Grouped Modeling Implementation:**
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py:fit_nested()` - Nested fitting method
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py:NestedWorkflowFit` - Grouped workflow class
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/tests/test_workflows/test_panel_models.py` - 13 tests for grouped models

**Example Notebooks:**
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/examples/13_panel_models_demo.ipynb` - Demo of nested and global approaches

## Testing

To verify the fixes work correctly:

```bash
cd "/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels"
source py-tidymodels2/bin/activate

# Clear outputs
jupyter nbconvert --clear-output --inplace _md/forecasting_grouped.ipynb

# Test execution of fixed cells
jupyter nbconvert --to notebook --execute _md/forecasting_grouped.ipynb \
  --output /tmp/forecasting_grouped_test.ipynb \
  --ExecutePreprocessor.timeout=1200
```

## Summary

Successfully fixed 2 cells in the forecasting grouped notebook to properly use grouped/panel modeling methods. No additional issues found in other cells. The notebook should now execute correctly with per-country model fitting and evaluation.
