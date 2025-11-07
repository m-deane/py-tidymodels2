# sklearn Linear Regression Residuals Bug Fix

**Date:** 2025-11-04
**Fixed By:** Claude Code
**Issue:** Residuals column in outputs showed actual values instead of calculated residuals
**Affected Component:** `py_parsnip/engines/sklearn_linear_reg.py`

## Problem Description

When using sklearn linear regression engine, the `extract_outputs()` method was returning incorrect residuals for **training data only**. Instead of calculating residuals as `(actuals - fitted)`, it was returning the actual values themselves.

### Symptoms

**Training Data (INCORRECT - before fix):**
```python
# outputs_sk for training data showed:
   actuals    fitted    residuals
0   137.65   103.031      137.65   # Wrong! Should be 34.62
1   113.53   100.681      113.53   # Wrong! Should be 12.85
2    43.31    99.273       43.31   # Wrong! Should be -55.96
```

**Test Data (CORRECT - even before fix):**
```python
# outputs_sk for test data was correct:
   actuals    fitted    residuals
42  107.65   102.147      5.50     # Correct!
43  118.37   105.257     13.11     # Correct!
```

The statsmodels engine (`outputs_sm`) had correct residuals for both training and test data.

## Root Cause

**Location:** `py_parsnip/engines/sklearn_linear_reg.py`, line 96 in the `fit()` method

**Buggy Code:**
```python
# Calculate fitted values and residuals
fitted = model.predict(X)
residuals = y.values if isinstance(y, pd.Series) else y - fitted
```

**Issue:** The ternary expression incorrectly returned `y.values` (the actual values) when `y` was a pandas Series, instead of calculating the residuals. This logic was meant to convert `y` to a numpy array before subtraction, but it short-circuited the calculation entirely.

## Solution

**Fixed Code:**
```python
# Calculate fitted values and residuals
fitted = model.predict(X)
y_array = y.values if isinstance(y, pd.Series) else y
residuals = y_array - fitted
```

**Change:** Split the logic into two steps:
1. First convert `y` to numpy array if it's a Series
2. Then calculate residuals as `y_array - fitted`

## Testing

### Unit Tests
All 26 existing linear_reg tests pass:
```bash
pytest tests/test_parsnip/test_linear_reg.py -v
# ============================== 26 passed ==============================
```

### Manual Verification
Created test script `_md/test_residuals_fix.py` that demonstrates:
- Residuals are now correctly calculated as `(actuals - fitted)`
- Both Series and array inputs work correctly
- Results match manual calculation

### Re-running Forecasting Notebook
To see the fix in your `_md/forecasting.ipynb`:
1. Restart the Jupyter kernel (to reload the updated module)
2. Re-run the cells that fit sklearn models (spec_sk, fit_sk)
3. Re-run the cells that extract outputs (outputs_sk)

You should now see correct residuals like:
```python
# After fix:
   actuals    fitted    residuals
0   137.65   103.031     34.62     # Correct!
1   113.53   100.681     12.85     # Correct!
2    43.31    99.273    -55.96     # Correct!
```

## Impact

- **Training residuals:** Now correctly calculated as `(actuals - fitted)`
- **Test residuals:** No change (were already correct)
- **statsmodels engine:** No change (was already correct)
- **Backward compatibility:** Maintained (all tests pass)

## Related Files

- **Fixed:** `py_parsnip/engines/sklearn_linear_reg.py` (line 96-97)
- **Tests:** `tests/test_parsnip/test_linear_reg.py` (26 tests passing)
- **Verification:** `_md/test_residuals_fix.py`
- **User Report:** `_md/forecasting.ipynb`

## Prevention

This bug highlights the importance of:
1. Testing both training and test residuals separately
2. Avoiding complex ternary expressions for calculations
3. Breaking logic into clear, separate steps
4. Adding explicit test cases for Series vs array inputs

## Notes

- The statsmodels engine was not affected because it uses statsmodels' built-in residuals calculation: `fitted_model.resid.values`
- The test residuals were correct because they go through a different code path in `extract_outputs()` that always calculated `test_actuals - test_predictions` directly
- The training residuals used the pre-calculated `residuals` from `fit_data`, which contained the bug
