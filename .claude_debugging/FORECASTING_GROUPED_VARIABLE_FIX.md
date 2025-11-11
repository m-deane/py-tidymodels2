# Forecasting Grouped Notebook Variable Overwrite Fix

## Date
2025-11-09

## Problem Description

Critical bug in `_md/forecasting_grouped.ipynb` where visualization cells were overwriting the original training/test data variables.

### Root Cause
Visualization cells contained code that reassigned `train_mix` and `test_mix` variables:
```python
train_mix = country_data[country_data['split'] == 'train']
test_mix = country_data[country_data['split'] == 'test']
```

This overwrote the original training/test DataFrames created in cell 10:
```python
train_mix = split_mixed.training()
test_mix = split_mixed.testing()
```

### Impact
- Subsequent cells failed with "KeyError" or wrong column errors
- Training data became filtered output data with different schema
- Made notebook unusable after running visualization cells

## Solution Implemented

### Changes Applied
1. **Renamed visualization variables** (44 cells modified):
   - `train_mix = country_data[...]` → `train_viz = country_data[...]`
   - `test_mix = country_data[...]` → `test_viz = country_data[...]`

2. **Updated all references** (528 total replacements):
   - `len(train_mix)` → `len(train_viz)`
   - `train_mix.index` → `train_viz.index`
   - `train_mix['column']` → `train_viz['column']`
   - Same for test_mix → test_viz

### Verification Results
```
✅ Cells modified: 44
✅ Total replacements: 528
✅ train_mix assignment overwrites: 0 (was 44)
✅ test_mix assignment overwrites: 0 (was 44)
✅ train_viz assignments: 44
✅ test_viz assignments: 44
✅ train_viz.index references: 88
✅ train_viz[...] references: 176
✅ test_viz[...] references: 176
✅ Original train_mix/test_mix definitions: Preserved in cell 10
```

## Pattern Applied

### Before (Problematic)
```python
# Cell 10 - Original definition
train_mix = split_mixed.training()
test_mix = split_mixed.testing()

# Cell 19 - Visualization (OVERWRITES!)
country_data, _, _ = fit_usa.extract_outputs()
train_mix = country_data[country_data['split'] == 'train']  # ❌ OVERWRITES!
test_mix = country_data[country_data['split'] == 'test']    # ❌ OVERWRITES!

if len(train_mix) > 0:
    ax.plot(train_mix.index, train_mix['actuals'], ...)
```

### After (Fixed)
```python
# Cell 10 - Original definition (UNCHANGED)
train_mix = split_mixed.training()
test_mix = split_mixed.testing()

# Cell 19 - Visualization (SAFE)
country_data, _, _ = fit_usa.extract_outputs()
train_viz = country_data[country_data['split'] == 'train']  # ✅ NEW VARIABLE
test_viz = country_data[country_data['split'] == 'test']    # ✅ NEW VARIABLE

if len(train_viz) > 0:
    ax.plot(train_viz.index, train_viz['actuals'], ...)
```

## Files Modified
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_grouped.ipynb`

## Testing Recommendation

Run the notebook end-to-end to verify:
1. Cell 10 creates train_mix/test_mix correctly
2. All visualization cells use train_viz/test_viz
3. No variable overwrites occur
4. All plots render correctly
5. Downstream cells can still access original train_mix/test_mix

## Related Issues
- Variable shadowing in data pipelines
- Importance of immutable data patterns in notebooks
- Need for linting/validation of notebook variable usage

## Prevention
Consider adding notebook validation rules:
- Warn on variable reassignment that changes type/schema
- Flag when visualization code modifies analysis variables
- Enforce naming conventions (e.g., _viz suffix for plotting data)
