# Date Column Extraction - Implementation Summary

## Overview

Added date column extraction to `gen_additive_mod()` and `boost_tree()` models (all engines: XGBoost, LightGBM, CatBoost) so that time series data outputs include a `date` column.

## Files Modified

### Engine Files (4 files)
1. **`py_parsnip/engines/pygam_gam.py`** - GAM engine
2. **`py_parsnip/engines/xgboost_boost_tree.py`** - XGBoost engine
3. **`py_parsnip/engines/lightgbm_boost_tree.py`** - LightGBM engine
4. **`py_parsnip/engines/catboost_boost_tree.py`** - CatBoost engine

### Changes Made to Each Engine

1. **Added `Optional` import:**
   ```python
   from typing import Dict, Any, Literal, Optional
   ```

2. **Updated `fit()` signature to accept `original_training_data`:**
   ```python
   def fit(
       self,
       spec: ModelSpec,
       molded: MoldedData,
       original_training_data: Optional[pd.DataFrame] = None
   ) -> Dict[str, Any]:
   ```

3. **Stored `original_training_data` in fit_data dict:**
   ```python
   return {
       "model": model,
       # ... other metadata ...
       "original_training_data": original_training_data,
   }
   ```

4. **Added date column extraction in `extract_outputs()` method:**
   - Uses `_infer_date_column()` from `py_parsnip.utils.time_series_utils`
   - Extracts dates from both training and test data
   - Inserts date column as first column in outputs DataFrame
   - Gracefully handles cases where date column doesn't exist

## Compatibility with Workflows and WorkflowSets

### ✅ Fully Compatible

The implementation is **fully compatible** with all workflow patterns:

1. **Direct Model Fit:**
   ```python
   fit = spec_gam.fit(train_data, formula, original_training_data=train_data)
   fit = fit.evaluate(test_data, original_test_data=test_data)
   ```

2. **Workflow:**
   ```python
   wf = Workflow().add_formula('y ~ x').add_model(gen_additive_mod())
   wf_fit = wf.fit(train_data)  # Automatically passes original_training_data
   wf_fit = wf_fit.evaluate(test_data)  # Automatically passes original_test_data
   ```

3. **WorkflowSet with Cross-Validation:**
   ```python
   wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
   results = wf_set.fit_resamples(folds, metrics)  # Works correctly
   ```

4. **fit_nested() for Grouped Data:**
   ```python
   nested_fit = wf.fit_nested(data, group_col='store_id')
   nested_fit = nested_fit.evaluate(test)
   outputs, _, _ = nested_fit.extract_outputs()  # Includes date column
   ```

5. **fit_global() for Global Models:**
   ```python
   global_fit = wf.fit_global(data, group_col='store_id')
   # Date column extraction works correctly
   ```

### How It Works

The workflow infrastructure already passes `original_training_data` through the call chain:

1. **`Workflow.fit()`** (line 230 of workflow.py):
   ```python
   model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)
   ```

2. **`WorkflowFit.evaluate()`** (line 469 of workflow.py):
   ```python
   self.fit = self.fit.evaluate(processed_test_data, outcome_col, original_test_data=test_data)
   ```

3. **`ModelSpec.fit()`** (line 176-180 of model_spec.py):
   - Checks if engine accepts `original_training_data` parameter
   - Passes it if explicitly provided (not defaulting to data itself for backwards compatibility)

4. **`ModelFit.evaluate()`** (line 343-344 of model_spec.py):
   - Only passes `original_test_data` if explicitly provided

## Usage in Notebooks

### Option 1: Direct Model Fit (Explicit Parameters)
```python
from py_parsnip import gen_additive_mod, boost_tree

# Fit with original data
fit_gam = spec_gam.fit(train_data, FORMULA_STR, original_training_data=train_data)
fit_gam = fit_gam.evaluate(test_data, original_test_data=test_data)

# Extract outputs with date column
outputs_gam, coefs_gam, stats_gam = fit_gam.extract_outputs()
display(outputs_gam)  # Now includes 'date' column!
```

### Option 2: Workflow (Automatic - Recommended)
```python
from py_workflows import Workflow

# Create workflow
wf = (Workflow()
    .add_formula('y ~ x1 + x2')
    .add_model(gen_additive_mod()))

# Fit and evaluate - original data passed automatically
wf_fit = wf.fit(train_data)
wf_fit = wf_fit.evaluate(test_data)

# Extract outputs with date column
outputs, coefs, stats = wf_fit.extract_outputs()
display(outputs)  # Includes 'date' column automatically!
```

### Option 3: WorkflowSet (For Multi-Model Comparison)
```python
from py_workflowsets import WorkflowSet

# Create multiple workflows
formulas = ['y ~ x1 + x2', 'y ~ x1']
models = [gen_additive_mod(), boost_tree().set_engine('xgboost')]
wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

# Fit with cross-validation
folds = vfold_cv(train_data, v=5)
results = wf_set.fit_resamples(folds, metrics)

# Date columns preserved in all evaluations
```

## Test Results

### ✅ All Tests Passing

1. **Engine Tests:**
   - `test_gen_additive_mod.py`: 27/27 passing
   - `test_boost_tree.py`: 37/37 passing

2. **Date Column Extraction Tests:**
   - ✅ GAM engine
   - ✅ XGBoost engine
   - ✅ LightGBM engine
   - ✅ CatBoost engine

3. **Integration Tests:**
   - ✅ Direct model fit with date column
   - ✅ Workflow with date column
   - ✅ WorkflowSet with cross-validation
   - ✅ fit_nested() with grouped data
   - ✅ fit_global() with global model

## Backwards Compatibility

The implementation maintains **full backwards compatibility**:

1. **Optional Parameter:** `original_training_data` defaults to `None`
2. **No Breaking Changes:** Existing code works without modification
3. **Opt-In Behavior:** Date column only added when original data is provided
4. **Workflow Integration:** Workflows automatically pass original data, so date columns work "out of the box"

## Summary

The date column extraction feature is now:
- ✅ Implemented for GAM and all boost_tree engines
- ✅ Fully compatible with Workflows
- ✅ Fully compatible with WorkflowSets
- ✅ Fully compatible with fit_nested() and fit_global()
- ✅ Backwards compatible with existing code
- ✅ All tests passing

Users can now use `gen_additive_mod()` and `boost_tree()` models with time series data and automatically get date columns in their outputs, making visualization and analysis much easier.
