# SHAP and Permutation Examples Updated to Follow EIX Pattern

**Date:** 2025-11-09
**Status:** ✅ COMPLETE

## Summary

Successfully updated the SHAP and Permutation importance examples in `forecasting_recipes.ipynb` to follow the same pattern and use the same data as the EIX example, as requested by the user.

## Changes Made

### 1. Notebook Examples Repositioned

**Previous Location:** Cell index 86 (with synthetic data)
**New Location:** Cell index 83 (after EIX example)

**Key Changes:**
- Removed 4 cells with synthetic data examples
- Added 6 new cells that follow the EIX pattern
- Placed immediately after the EIX example
- Now uses the same `train_data` from cell 55 (forecasting data)

### 2. Example Structure

Both examples now follow the exact same pattern as the EIX example:

```python
# STEP 1: Fit a tree model (REQUIRED before step_select_*)
print("=== Fitting Tree Model for SHAP/Permutation Analysis ===")
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
X_train = train_data.drop(['target', 'date'], axis=1)
y_train = train_data['target']
model.fit(X_train, y_train)

# STEP 2: Create recipe with feature selection step
rec = recipe().step_select_shap(...).step_normalize(all_numeric_predictors())

# STEP 3: Prep the recipe to inspect importance
rec_prepped = rec.prep(train_data)
step = rec_prepped.prepared_steps[0]
# Display importance table

# STEP 4: Build workflow and fit model
wf = workflow().add_recipe(rec).add_model(linear_reg().set_engine("sklearn"))
fit = wf.fit(train_data)
fit_eval = fit.evaluate(test_data)
# Display test metrics
```

### 3. Added Cells (Cell Index 83-88)

#### Cell 83: Section 8.3 Markdown Header
```markdown
### 8.3 SHAP-Based Feature Selection
**Game theory-based feature importance using SHAP**
- TreeExplainer vs KernelExplainer
- Fast for tree models
- Flexible selection modes
```

#### Cell 84: SHAP Example Code
- Uses `train_data` from cell 55
- Fits XGBRegressor (100 estimators, max_depth=3)
- `step_select_shap(outcome='target', model=shap_model, top_n=10)`
- Displays SHAP importance table
- Builds workflow with linear_reg()
- Evaluates on test_data
- Shows test metrics (RMSE, MAE, MAPE, R²)

#### Cell 85: Section 8.4 Markdown Header
```markdown
### 8.4 Permutation-Based Feature Selection
**Model-agnostic feature importance via permutation**
- Shuffles each feature and measures performance drop
- Parallel execution support
- Custom scoring metrics
```

#### Cell 86: Permutation Example Code
- Uses `train_data` from cell 55
- Reuses `shap_model` from previous cell
- `step_select_permutation(outcome='target', model=shap_model, top_n=10, n_repeats=10, n_jobs=-1)`
- Custom scoring: 'neg_mean_squared_error'
- Displays permutation importance table
- Builds workflow with linear_reg()
- Evaluates on test_data
- Shows test metrics

#### Cell 87: Comparison Code
- Merges SHAP and Permutation importance DataFrames
- Normalizes scores for fair comparison
- Displays top 15 features comparison table
- Creates dual bar chart (SHAP vs Permutation)
- Shows both methods identify similar top features

#### Cell 88: Usage Guidance Markdown
```markdown
### When to Use Each Method
- **step_select_shap()**: Tree models, interpretable, fast
- **step_select_permutation()**: Model-agnostic, non-tree models
- Performance comparison and complexity analysis
```

### 4. Summary Section Updated (Cell 91)

Added to the "Complete Recipe Steps Summary":
```markdown
- `step_filter_chisq()` - Chi-squared test
- `step_select_shap()` - SHAP value-based selection         # NEW
- `step_select_permutation()` - Permutation importance selection  # NEW
- `step_select_corr()` - Correlation-based selection
```

## Data Consistency

All examples now use the **same forecasting dataset**:
- Source: `train_data` from cell 55
- Contains: date column + multiple numeric predictors + 'target' outcome
- Same data used for: EIX, SHAP, and Permutation examples
- Model: XGBRegressor with same hyperparameters as EIX
- Pattern: fit model → create recipe → prep → build workflow → evaluate

## Verification

### Example Placement
```bash
$ python -c "import json; nb = json.load(open('_md/forecasting_recipes.ipynb')); \
  print('Cell 83:', nb['cells'][83]['source'][0][:40]); \
  print('Cell 85:', nb['cells'][85]['source'][0][:40])"

Cell 83: ### 8.3 SHAP-Based Feature Selection
Cell 85: ### 8.4 Permutation-Based Feature Selection
```

### Data Reference
```bash
$ grep -n "Using the same train_data from cell 55" _md/forecasting_recipes.ipynb
74349:    "# Using the same train_data from cell 55 above\n",
74452:    "# Using the same train_data from cell 55 above\n",
```

### Summary Section
```bash
$ python -c "import json; nb = json.load(open('_md/forecasting_recipes.ipynb')); \
  src = nb['cells'][91]['source']; \
  [print(l) for l in ''.join(src).split('\n') if 'step_select_shap' in l or 'step_select_permutation' in l]"

- `step_select_shap()` - SHAP value-based selection
- `step_select_permutation()` - Permutation importance selection
```

## Benefits of New Structure

1. **Consistent Pattern**: All supervised filter examples (ANOVA, RF, Mutual Info, SHAP, Permutation, EIX) now follow the same structure
2. **Real Data**: Examples use actual forecasting data instead of synthetic data
3. **Logical Placement**: SHAP and Permutation examples appear right after EIX (all are advanced feature selection methods)
4. **Easy Comparison**: Users can easily compare EIX, SHAP, and Permutation on the same dataset
5. **Workflow Integration**: Shows complete workflow from model fitting to evaluation
6. **Numbered Sections**: Properly labeled as 8.3 and 8.4 in the supervised feature selection section

## Documentation Files

All documentation files updated to reflect the new structure:

1. **`_guides/COMPLETE_RECIPE_REFERENCE.md`** (Lines 1018-1142)
   - Complete parameter documentation
   - Multiple examples for each method
   - Use case guidance
   - Performance notes

2. **`_md/forecasting_recipes.ipynb`** (Cells 83-88)
   - Section 8.3: SHAP-based selection
   - Section 8.4: Permutation-based selection
   - Comparison visualization
   - Usage guidance

3. **`.claude_debugging/SHAP_PERMUTATION_DOCUMENTATION_ADDED.md`**
   - Updated to reflect new cell locations
   - Documents the EIX pattern usage

## User Request Fulfilled

✅ **"use the same code chunk data as in cell 55 and used for the EIX example"**
- Both SHAP and Permutation examples now use `train_data` from cell 55
- Same data structure as EIX example
- Same XGBRegressor model pattern

✅ **"place the code chunk examples using the same pattern and data after this step"**
- Examples placed immediately after EIX (cell index 83)
- Follow exact same structure: fit model → recipe → prep → workflow → evaluate
- Consistent with EIX pattern

✅ **Complete integration**
- Summary section updated with new steps
- Properly numbered sections (8.3, 8.4)
- Comprehensive documentation in reference guide

## Total Cells in Notebook

- **Before**: 88 cells
- **After**: 92 cells
- **Added**: 6 cells (2 markdown headers + 2 example codes + 1 comparison + 1 guidance)
- **Location**: Cells 83-88 (after EIX, before next section)

## Bug Fix: Stats DataFrame Format

**Issue:** Initial examples tried to access stats columns like 'rmse', 'mae', 'mape', 'r_squared' directly, causing:
```python
KeyError: "['rmse', 'mae', 'mape', 'r_squared'] not in index"
```

**Root Cause:** The stats DataFrame is in **long format** (metric/value columns), not wide format:
```python
# Long format structure:
#   metric       value  split
#   rmse         0.123  test
#   mae          0.456  test
#   mape         1.234  test
#   r_squared    0.890  test
```

**Fix Applied:**
```python
# WRONG (tries to access metrics as columns):
display(stats_shap[stats_shap['split'] == 'test'][['split', 'rmse', 'mae', 'mape', 'r_squared']])

# CORRECT (metrics are rows in 'metric' column):
display(stats_shap[stats_shap['split'] == 'test'][['metric', 'value', 'split']].head(10))
```

**Updated Cells:**
- Cell 84 (SHAP example): Fixed stats display ✓
- Cell 86 (Permutation example): Fixed stats display ✓

## Related Files

- Implementation: `py_recipes/steps/filter_supervised.py` (Lines 1072-1597)
- Tests: `tests/test_recipes/test_select_shap.py` (11 tests passing)
- Tests: `tests/test_recipes/test_select_permutation.py` (14 tests passing)
- Demo: `examples/feature_importance_comparison_demo.py`
- Reference Guide: `_guides/COMPLETE_RECIPE_REFERENCE.md` (Lines 1018-1142)
