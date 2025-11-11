# Unfitted Model Support for Recipe Steps - Implementation Summary

**Date**: 2025-11-10
**Files Modified**: 
- `/py_recipes/steps/filter_supervised.py`

## Overview

Enhanced `StepSelectShap` and `StepSelectPermutation` recipe steps to accept **unfitted** models and automatically fit them during `prep()`. This provides a more user-friendly API where users don't need to pre-fit models before using these feature selection steps.

## Changes Made

### 1. StepSelectShap (Lines 1073-1400)

**Added Attributes**:
```python
_fitted_model: Any = field(default=None, init=False, repr=False)
```

**Added Helper Method**:
```python
def _is_model_fitted(self, model: Any) -> bool:
    """Check if model is already fitted."""
    fitted_attrs = ['n_features_in_', 'feature_names_in_', 'coef_', 'estimators_']
    return any(hasattr(model, attr) for attr in fitted_attrs)
```

**Modified `prep()` Method**:
Added model fitting logic before computing SHAP importance:
```python
# Fit model if not already fitted
if not self._is_model_fitted(self.model):
    # Prepare data for fitting
    X_fit = data[score_cols].copy()
    y_fit = y.copy()
    
    # Handle categorical columns (one-hot encode)
    cat_cols = X_fit.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        X_fit = pd.get_dummies(X_fit, columns=cat_cols, drop_first=True)
    
    # Drop rows with missing values
    mask = ~(X_fit.isna().any(axis=1) | y_fit.isna())
    X_fit = X_fit[mask]
    y_fit = y_fit[mask]
    
    # Handle model classes vs instances
    if isinstance(self.model, type):
        # Model class - instantiate and fit
        self._fitted_model = self.model().fit(X_fit, y_fit)
    else:
        # Model instance - clone and fit
        from sklearn.base import clone
        self._fitted_model = clone(self.model).fit(X_fit, y_fit)
else:
    # Already fitted - use as is
    self._fitted_model = self.model
```

**Updated `_compute_shap_importance()` Method**:
Changed all references from `self.model` to `self._fitted_model`:
- Line 1281: `hasattr(self._fitted_model, 'feature_names_in_')`
- Line 1304: `type(self._fitted_model).__name__.lower()`
- Line 1313: `shap.TreeExplainer(self._fitted_model)`
- Line 1319: `shap.KernelExplainer(self._fitted_model.predict, background)`
- Line 1361: Error message uses `self._fitted_model`

**Updated Documentation**:
```python
model : object
    Scikit-learn compatible model (fitted or unfitted). Can be tree-based
    (XGBoost, LightGBM, RandomForest) for fast TreeExplainer, or any model
    for slower KernelExplainer. If unfitted, will be fitted automatically
    during prep() using the recipe data.
```

### 2. StepSelectPermutation (Lines 1366-1660)

**Added Attributes**:
```python
_fitted_model: Any = field(default=None, init=False, repr=False)
```

**Added Helper Method**:
```python
def _is_model_fitted(self, model: Any) -> bool:
    """Check if model is already fitted."""
    fitted_attrs = ['n_features_in_', 'feature_names_in_', 'coef_', 'estimators_']
    return any(hasattr(model, attr) for attr in fitted_attrs)
```

**Modified `prep()` Method**:
Added identical model fitting logic as StepSelectShap (lines 1542-1568).

**Updated `_compute_permutation_importance()` Method**:
Changed reference from `self.model` to `self._fitted_model`:
- Line 1603: `permutation_importance(self._fitted_model, ...)`

**Updated Documentation**:
```python
model : object
    Scikit-learn compatible model (fitted or unfitted). If unfitted, will be
    fitted automatically during prep() using the recipe data.
```

### 3. StepEIX - No Changes Required

**Important**: `StepEIX` in `/py_recipes/steps/interaction_detection.py` was **NOT modified** because:
- It **requires** a pre-fitted tree model by design
- It analyzes tree structure (nodes, gains) rather than just predictions
- It validates that the model is fitted in `__post_init__()` (line 121-125)
- Documentation explicitly states: "Pre-fitted tree-based model (XGBoost or LightGBM). REQUIRED."

This is fundamentally different from SHAP/permutation importance which only need the model's `.predict()` method.

## Implementation Details

### Model Fitting Logic

The implementation handles three scenarios:

1. **Model Class** (e.g., `RandomForestRegressor`):
   - Instantiates with default parameters: `self.model().fit(X_fit, y_fit)`
   - Creates a new instance internally

2. **Unfitted Model Instance** (e.g., `RandomForestRegressor(n_estimators=100)`):
   - Clones the instance to preserve original: `clone(self.model).fit(X_fit, y_fit)`
   - Original model remains unmodified

3. **Fitted Model Instance** (already has `n_features_in_` attribute):
   - Uses model as-is: `self._fitted_model = self.model`
   - No cloning or re-fitting

### Data Preprocessing for Model Fitting

Before fitting, the data is cleaned:
1. One-hot encode categorical columns (`pd.get_dummies()`)
2. Drop rows with missing values (both features and outcome)
3. Fit on cleaned data

This ensures the model can be fitted successfully during `prep()`.

### Fitted Model Detection

Uses heuristic to detect if model is fitted by checking for common sklearn attributes:
- `n_features_in_` - Number of features seen during fit
- `feature_names_in_` - Feature names seen during fit
- `coef_` - Coefficients (linear models)
- `estimators_` - Ensemble of estimators (RF, GBM)

Returns `True` if **any** of these attributes exist.

## Testing Results

### Existing Tests
All 38 existing tests in `test_filter_supervised.py` continue to pass:
```
======================= 38 passed, 153 warnings in 1.16s =======================
```

### New Comprehensive Tests
Created 10 comprehensive tests covering:

1. ✓ StepSelectPermutation with unfitted model class (regression)
2. ✓ StepSelectPermutation with unfitted model instance (regression)
3. ✓ StepSelectPermutation with fitted model (regression)
4. ✓ StepSelectShap with unfitted model class (regression)
5. ✓ StepSelectShap with unfitted model instance (regression)
6. ✓ StepSelectShap with fitted model (regression)
7. ✓ StepSelectPermutation with classification (unfitted)
8. ✓ StepSelectShap with classification (unfitted)
9. ✓ Full prep/bake workflow with unfitted model
10. ✓ StepSelectPermutation with LinearRegression (unfitted)

**Result**: 10/10 tests passed

## Usage Examples

### Before (Required Pre-fitted Model)
```python
from sklearn.ensemble import RandomForestRegressor
from py_recipes import recipe
from py_recipes.steps import step_select_shap

# Had to fit model first
X_train = data[['x1', 'x2', 'x3', 'x4', 'x5']]
y_train = data['y']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Extra step!

# Then use in recipe
rec = recipe(data, "y ~ .").step_select_shap(
    outcome='y',
    model=model,  # Must be fitted
    top_n=10
)
```

### After (Works with Unfitted Model)
```python
from sklearn.ensemble import RandomForestRegressor
from py_recipes import recipe
from py_recipes.steps import step_select_shap

# Just pass the model - fitting happens automatically
rec = recipe(data, "y ~ .").step_select_shap(
    outcome='y',
    model=RandomForestRegressor(n_estimators=100, random_state=42),  # Unfitted!
    top_n=10
)

# Or even simpler - pass the class
rec = recipe(data, "y ~ .").step_select_shap(
    outcome='y',
    model=RandomForestRegressor,  # Model class
    top_n=10
)
```

## Benefits

1. **Simplified API**: Users don't need to manually fit models before using feature selection steps
2. **Consistent with tidymodels**: Matches R's tidymodels philosophy where models are specified but fit later
3. **Flexible**: Still supports pre-fitted models for users who want control over training data
4. **Safe**: Uses `sklearn.base.clone()` to avoid modifying user's original model instances
5. **Robust**: Handles model classes, unfitted instances, and fitted instances seamlessly

## Backward Compatibility

✅ **Fully backward compatible**. All existing code using pre-fitted models continues to work without modification.

## Files Modified

- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps/filter_supervised.py`
  - Modified: `StepSelectShap` class (lines 1073-1400)
  - Modified: `StepSelectPermutation` class (lines 1366-1660)
  - Added: `_fitted_model` attribute to both classes
  - Added: `_is_model_fitted()` helper method to both classes
  - Updated: `prep()` methods to fit models if needed
  - Updated: Importance calculation methods to use `self._fitted_model`
  - Updated: Docstrings to reflect unfitted model support

## Related Documentation

- sklearn.base.clone: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
- SHAP values: https://github.com/slundberg/shap
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
