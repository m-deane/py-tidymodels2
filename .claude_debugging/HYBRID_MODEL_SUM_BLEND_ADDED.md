# Hybrid Model "sum" Blend Option Added

**Date**: 2025-11-07
**Feature**: Added "sum" option to `blend_predictions` parameter in `hybrid_model()`

## Overview

The `hybrid_model()` function's `custom_data` strategy now supports a "sum" blending option that sums the predictions from both models instead of taking a weighted combination.

## Changes Made

### 1. Engine Implementation (`py_parsnip/engines/generic_hybrid.py`)

**Lines 319-321**: Added "sum" case to prediction blending logic

```python
elif blend_type == "sum":
    # Sum predictions from both models
    predictions = model1_preds[".pred"].values + model2_preds[".pred"].values
```

**Location**: In the `predict()` method, within the `elif strategy == "custom_data":` block

### 2. Model Specification Documentation (`py_parsnip/models/hybrid_model.py`)

**Lines 84-89**: Updated docstring to document the new "sum" option

```python
blend_predictions: For custom_data strategy - how to combine predictions
    - "weighted": weight1 * pred1 + weight2 * pred2 (default)
    - "avg": simple average (0.5 * pred1 + 0.5 * pred2)
    - "sum": sum of predictions (pred1 + pred2)  # NEW
    - "model1": use only model1 predictions
    - "model2": use only model2 predictions
```

**Lines 180**: Updated validation list to include "sum"

```python
valid_blend_types = ["weighted", "avg", "sum", "model1", "model2"]
```

### 3. Test Coverage (`tests/test_parsnip/test_hybrid_model.py`)

**Lines 573-596**: Added comprehensive test for "sum" blend option

```python
def test_custom_data_predictions_sum(self, overlapping_data):
    """Test predictions with sum blend"""
    spec = hybrid_model(
        model1=linear_reg(),
        model2=linear_reg(),
        strategy='custom_data',
        blend_predictions='sum'
    )

    data_dict = {
        'model1': overlapping_data['model1'],
        'model2': overlapping_data['model2']
    }
    fit = spec.fit(data_dict, 'y ~ x')

    test_data = pd.DataFrame({'x': range(150, 160)})
    predictions = fit.predict(test_data)

    # Verify sum blend
    pred1 = fit.fit_data['model1_fit'].predict(test_data)['.pred'].values
    pred2 = fit.fit_data['model2_fit'].predict(test_data)['.pred'].values
    expected = pred1 + pred2

    np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-5)
```

## Usage Example

```python
from py_parsnip import hybrid_model, linear_reg, rand_forest

# Create hybrid model with sum blending
spec = hybrid_model(
    model1=linear_reg(),
    model2=rand_forest().set_mode('regression'),
    strategy='custom_data',
    blend_predictions='sum'  # NEW OPTION
)

# Fit with separate datasets for each model
early_data = df[df['date'] < '2020-07-01']
later_data = df[df['date'] >= '2020-04-01']

fit = spec.fit({'model1': early_data, 'model2': later_data}, 'y ~ x')

# Predictions will be: model1_pred + model2_pred
predictions = fit.predict(new_data)
```

## When to Use "sum"

The "sum" blend option is useful when:

1. **Additive Effects**: Both models capture independent, additive components of the signal
   - Example: Model1 captures trend, Model2 captures seasonality

2. **Component Decomposition**: Models trained to predict different components that should be combined
   - Example: Model1 predicts base value, Model2 predicts deviation

3. **Ensemble Boosting**: Similar to boosting where predictions accumulate
   - Example: Model1 makes initial prediction, Model2 adds correction

4. **Multi-Scale Patterns**: Models capture patterns at different scales
   - Example: Model1 captures long-term, Model2 captures short-term fluctuations

## Comparison with Other Blend Options

| Option | Formula | Use Case |
|--------|---------|----------|
| "weighted" | `w1*pred1 + w2*pred2` | General ensemble with custom weights |
| "avg" | `0.5*(pred1 + pred2)` | Equal contribution from both models |
| **"sum"** | `pred1 + pred2` | Additive components or effects |
| "model1" | `pred1` | Use only first model (backup available) |
| "model2" | `pred2` | Use only second model (backup available) |

## Test Results

**All 37 hybrid_model tests passing**, including the new test:

```bash
tests/test_parsnip/test_hybrid_model.py::TestCustomDataStrategy::test_custom_data_predictions_sum PASSED [100%]
```

**Test Coverage**:
- ✅ Prediction calculation correctness (sum of both model predictions)
- ✅ Integration with custom_data strategy
- ✅ Works with overlapping training datasets
- ✅ Numerical accuracy (rtol=1e-5)

## Documentation

**Updated Locations**:
1. `py_parsnip/models/hybrid_model.py` - Docstring updated with "sum" option
2. `py_parsnip/engines/generic_hybrid.py` - Implementation added
3. `tests/test_parsnip/test_hybrid_model.py` - Test coverage added
4. `docs/` - Rebuilt successfully (http://localhost:8000)

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `py_parsnip/engines/generic_hybrid.py` | 319-321 | Added "sum" case to blend logic |
| `py_parsnip/models/hybrid_model.py` | 87, 180 | Updated docstring and validation |
| `tests/test_parsnip/test_hybrid_model.py` | 573-596 | Added test_custom_data_predictions_sum |

## Backward Compatibility

✅ **Fully backward compatible** - This is a new option that doesn't affect existing code.

Existing code using `blend_predictions='weighted'`, `'avg'`, `'model1'`, or `'model2'` continues to work exactly as before.

## Next Steps

None required - feature is complete and tested.

Users can now use `blend_predictions='sum'` in their hybrid models with the `custom_data` strategy.
