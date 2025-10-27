# Hybrid Time Series Models Implementation Summary

## Overview

Successfully implemented two hybrid time series forecasting models that combine classical forecasting methods with gradient boosting to capture both linear and non-linear patterns.

## Models Implemented

### 1. arima_boost (ARIMA + XGBoost Hybrid)

**Strategy:**
1. Fit ARIMA model to capture linear patterns, trend, and autocorrelation
2. Calculate ARIMA residuals
3. Fit XGBoost on residuals to capture non-linear patterns
4. Final prediction = ARIMA prediction + XGBoost prediction

**Files Created:**
- Model specification: `/py_parsnip/models/arima_boost.py`
- Engine implementation: `/py_parsnip/engines/hybrid_arima_boost.py`
- Tests: `/tests/test_parsnip/test_arima_boost.py`

**Parameters:**

ARIMA parameters:
- `seasonal_period`: Seasonality period (e.g., 12 for monthly data)
- `non_seasonal_ar`: Number of non-seasonal AR terms (p)
- `non_seasonal_differences`: Number of non-seasonal differences (d)
- `non_seasonal_ma`: Number of non-seasonal MA terms (q)
- `seasonal_ar`: Number of seasonal AR terms (P)
- `seasonal_differences`: Number of seasonal differences (D)
- `seasonal_ma`: Number of seasonal MA terms (Q)

XGBoost parameters:
- `trees`: Number of boosting iterations (default: 100)
- `tree_depth`: Maximum tree depth (default: 6)
- `learn_rate`: Learning rate (default: 0.1)
- `min_n`: Minimum samples in leaf (default: 1)
- `loss_reduction`: Minimum loss reduction (default: 0.0)
- `sample_size`: Subsample ratio (default: 1.0)
- `mtry`: Feature sampling ratio (default: 1.0)

**Example Usage:**
```python
from py_parsnip import arima_boost

# Create model specification
spec = arima_boost(
    # ARIMA parameters
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    seasonal_period=12,
    # XGBoost parameters
    trees=100,
    tree_depth=5,
    learn_rate=0.1
)

# Fit model
fit = spec.fit(train_data, "sales ~ date")

# Make predictions
preds = fit.predict(test_data, type="numeric")

# Extract outputs
outputs, coefficients, stats = fit.extract_outputs()
```

### 2. prophet_boost (Prophet + XGBoost Hybrid)

**Strategy:**
1. Fit Prophet model to capture trend, seasonality, and holiday effects
2. Calculate Prophet residuals
3. Fit XGBoost on residuals to capture non-linear patterns
4. Final prediction = Prophet prediction + XGBoost prediction

**Files Created:**
- Model specification: `/py_parsnip/models/prophet_boost.py`
- Engine implementation: `/py_parsnip/engines/hybrid_prophet_boost.py`
- Tests: `/tests/test_parsnip/test_prophet_boost.py`

**Parameters:**

Prophet parameters:
- `growth`: Trend type ('linear' or 'logistic', default: 'linear')
- `changepoint_prior_scale`: Flexibility of trend changes (default: 0.05)
- `seasonality_prior_scale`: Flexibility of seasonality (default: 10.0)
- `seasonality_mode`: How components combine ('additive' or 'multiplicative', default: 'additive')
- `n_changepoints`: Number of potential changepoints (default: 25)
- `changepoint_range`: Proportion of history for changepoints (default: 0.8)

XGBoost parameters:
- Same as arima_boost (trees, tree_depth, learn_rate, etc.)

**Example Usage:**
```python
from py_parsnip import prophet_boost

# Create model specification
spec = prophet_boost(
    # Prophet parameters
    growth="linear",
    changepoint_prior_scale=0.1,
    seasonality_mode="additive",
    # XGBoost parameters
    trees=100,
    tree_depth=5,
    learn_rate=0.1
)

# Fit model
fit = spec.fit(train_data, "sales ~ date")

# Make predictions
preds = fit.predict(test_data, type="numeric")

# Extract outputs
outputs, coefficients, stats = fit.extract_outputs()
```

## Implementation Details

### Architecture

Both hybrid models follow a consistent two-stage architecture:

1. **Stage 1: Base Model**
   - Fit classical forecasting model (ARIMA or Prophet)
   - Capture linear patterns, trends, and seasonality
   - Calculate fitted values and residuals

2. **Stage 2: Residual Modeling**
   - Fit XGBoost on base model residuals
   - Capture non-linear patterns missed by base model
   - Use time-based features as predictors

3. **Prediction**
   - Get base model predictions
   - Get XGBoost predictions on residuals
   - Combine: final_prediction = base_prediction + xgb_prediction

### Engine Registration

Both engines are registered in `/py_parsnip/engines/__init__.py`:
- `hybrid_arima_boost.py`: Registered as "hybrid_arima_xgboost" engine
- `hybrid_prophet_boost.py`: Registered as "hybrid_prophet_xgboost" engine

### Package Integration

Models are exported from `/py_parsnip/__init__.py`:
```python
from py_parsnip.models.arima_boost import arima_boost
from py_parsnip.models.prophet_boost import prophet_boost

__all__ = [
    # ... other models
    "arima_boost",
    "prophet_boost",
]
```

## Testing

### Test Coverage

**arima_boost tests (11 tests, all passing):**
- Model specification creation
- Parameter validation
- Fitting with various ARIMA configurations
- Seasonal ARIMA support
- Prediction functionality
- Component separation (ARIMA + XGBoost)
- Extract outputs structure
- Metrics calculation
- Hybrid predictions sum correctly

**prophet_boost tests (15 tests, all passing):**
- Model specification creation
- Parameter validation
- Fitting with various Prophet configurations
- Multiplicative seasonality support
- Prediction functionality (near and far future)
- Component separation (Prophet + XGBoost)
- Extract outputs structure
- Metrics calculation
- Different XGBoost configurations
- Hybrid predictions sum correctly

### Running Tests

```bash
# Test arima_boost
pytest tests/test_parsnip/test_arima_boost.py -v

# Test prophet_boost
pytest tests/test_parsnip/test_prophet_boost.py -v

# Test both
pytest tests/test_parsnip/test_arima_boost.py tests/test_parsnip/test_prophet_boost.py -v
```

## Demo Script

A comprehensive demonstration script is available at:
`/examples/hybrid_models_demo.py`

The demo:
1. Creates synthetic time series data with linear and non-linear patterns
2. Fits both hybrid models
3. Makes predictions on test data
4. Compares performance metrics
5. Extracts and displays model components
6. Creates visualizations showing:
   - Training fit (base model vs hybrid)
   - Test predictions
   - Component decomposition

Run the demo:
```bash
cd /path/to/py-tidymodels
PYTHONPATH=.:$PYTHONPATH python examples/hybrid_models_demo.py
```

## Performance Characteristics

### When to Use Hybrid Models

**arima_boost is effective when:**
- Data has both linear temporal patterns and non-linear relationships
- ARIMA alone leaves structured residuals
- You need to capture complex interactions
- Time series has autocorrelation plus non-linear trends

**prophet_boost is effective when:**
- Data has strong seasonality but also non-linear patterns
- Prophet alone leaves structured residuals
- You want to leverage Prophet's trend detection plus XGBoost's flexibility
- Data has multiple seasonal patterns plus non-linear components

### Advantages

1. **Best of Both Worlds**: Combines interpretability of classical models with flexibility of gradient boosting
2. **Residual Modeling**: XGBoost captures patterns missed by base model
3. **Robust**: Base model provides stable foundation, boosting adds adaptability
4. **Comprehensive Outputs**: Extract both base and boosting components

### Considerations

1. **Computational Cost**: Fitting two models is more expensive than one
2. **Hyperparameter Tuning**: More parameters to tune (base model + XGBoost)
3. **Overfitting Risk**: XGBoost on residuals can overfit if not regularized
4. **Interpretability**: Less interpretable than base model alone

## Output Structure

Both models provide comprehensive outputs via `extract_outputs()`:

### 1. Outputs DataFrame
- `date`: Date/time index
- `actuals`: Actual values
- `<base>_fitted`: Base model fitted values (arima_fitted or prophet_fitted)
- `xgb_fitted`: XGBoost fitted values on residuals
- `fitted`: Combined fitted values (base + xgb)
- `forecast`: Forecast values
- `residuals`: Final residuals
- `split`: Data split indicator (train/test)
- `model`, `model_group_name`, `group`: Model metadata

### 2. Coefficients DataFrame
- Base model parameters with statistical inference (ARIMA) or hyperparameters (Prophet)
- XGBoost hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- All prefixed with `arima_` or `prophet_` and `xgb_` for clarity

### 3. Stats DataFrame
- Performance metrics (RMSE, MAE, MAPE, SMAPE, R-squared, MDA) by split
- Model information (formula, model_type, orders, parameters)
- Information criteria (AIC, BIC for ARIMA)
- Training dates and observation counts

## File Paths Summary

### Model Specifications
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/models/arima_boost.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/models/prophet_boost.py`

### Engine Implementations
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/engines/hybrid_arima_boost.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/engines/hybrid_prophet_boost.py`

### Tests
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/tests/test_parsnip/test_arima_boost.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/tests/test_parsnip/test_prophet_boost.py`

### Demo
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/examples/hybrid_models_demo.py`

### Updated Package Files
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/__init__.py`
- `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/engines/__init__.py`

## Dependencies

Both models require:
- `pandas`
- `numpy`
- `xgboost`
- `statsmodels` (for arima_boost)
- `prophet` (for prophet_boost)
- `scipy` (for statistical tests)

## Next Steps

Potential enhancements:
1. Add confidence intervals for hybrid predictions
2. Support for exogenous variables in hybrid models
3. Automated hyperparameter tuning
4. Cross-validation support
5. Feature importance from XGBoost component
6. Alternative boosting algorithms (LightGBM, CatBoost)
7. Multi-step ahead forecasting optimization

## Test Results

All tests passing:
- **arima_boost**: 11/11 tests passed
- **prophet_boost**: 15/15 tests passed
- **Total**: 26/26 tests passed

Demo script successfully executed with:
- ARIMA + XGBoost test RMSE: 30.67
- Prophet + XGBoost test RMSE: 28.11
