# ML Models for Time Series Regression - Documentation Added

**Date**: 2025-11-07
**File Modified**: `docs/user_guide/time_series.rst`

## New Section Added

Added comprehensive documentation section: **"Using ML Models for Time Series Regression"**

This section explains how to use machine learning models (Random Forest, XGBoost, SVM, k-NN, Neural Networks, etc.) for time series regression with date-indexed outputs.

## Key Topics Covered

### 1. When to Use ML vs Traditional Time Series Models

**Use ML Models When:**
- You have rich feature sets (exogenous variables)
- Relationships are non-linear
- You need to capture complex interactions
- Multiple seasonalities or irregular patterns exist
- You're doing supervised regression (not pure forecasting)

**Use Traditional Models (ARIMA/Prophet) When:**
- You have univariate time series with limited features
- You need probabilistic forecasts with prediction intervals
- The data shows clear trend/seasonality patterns
- You need interpretable decomposition

### 2. Date-Indexed Outputs Explanation

**Key Point**: All models in py-tidymodels return date-indexed outputs from `extract_outputs()` when the data contains a date column.

**Output Structure**:
```python
outputs, coefs, stats = fit.extract_outputs()

# outputs DataFrame (date-indexed):
#   - date: Original date from input data
#   - actuals: True values
#   - fitted: Model predictions
#   - forecast: Combined series (actuals + fitted)
#   - residuals: actuals - fitted
#   - split: 'train', 'test', or 'forecast'
```

### 3. Comprehensive Examples for Each ML Model

#### Random Forest
- Full example with time-based split
- Shows date-indexed output structure
- Includes visualization code
- Demonstrates filtering test period

#### Gradient Boosting (All 3 Engines)
- XGBoost for time series
- LightGBM for time series (faster training)
- CatBoost for time series (handles categoricals)

#### Support Vector Machines
- SVM with RBF kernel
- Accessing metrics from date-indexed outputs

#### k-Nearest Neighbors
- Example with distance weighting
- k=10 configuration

#### Neural Networks (MLP)
- Multi-layer perceptron for time series
- Hidden units, epochs, learning rate configuration

### 4. Feature Engineering for ML Time Series

Comprehensive recipe showing 8 essential steps:

1. **Lagged features** - Autoregressive patterns
2. **Rolling statistics** - Moving averages, standard deviations
3. **Time-based features** - Year, month, day of week extraction
4. **Differencing** - For stationarity
5. **Encoding categorical features** - Month, day of week
6. **Normalization** - Scaling features
7. **Imputation** - Handle missing values from lagging
8. **Correlation filtering** - Remove redundant features

**Important Note**: Outputs remain date-indexed even after all preprocessing!

### 5. Understanding the Three-DataFrame Output

Detailed explanation of all three DataFrames:

**outputs**: Observation-level results
- Shows exact structure with column descriptions
- Explains date indexing

**coefficients**: Model parameters/importances
- For tree models: feature importances
- Shows example output format

**stats**: Model-level metrics by split
- Metrics by train/test split
- Example of filtering test metrics

### 6. Comparing Multiple ML Models

Complete example using WorkflowSet:
- Define multiple models (RF, XGBoost, SVM, k-NN)
- Create workflow set
- Evaluate with cross-validation
- Rank models by performance
- Select and fit best model
- Extract date-indexed outputs from winner

## Code Pattern Demonstrated

All examples follow this consistent pattern:

```python
from py_parsnip import model_name

# Create specification
spec = model_name(param1=value1, param2=value2).set_mode('regression')

# Fit model - date column automatically handled
fit = spec.fit(train, "target ~ feature1 + feature2")

# Evaluate on test set
fit = fit.evaluate(test)

# Extract date-indexed outputs
outputs, coefs, stats = fit.extract_outputs()

# outputs includes 'date' column for visualization
print(outputs[outputs['split']=='test'][['date', 'actuals', 'fitted']])
```

## Models Covered in Examples

1. ✅ **Random Forest** (`rand_forest`)
2. ✅ **XGBoost** (`boost_tree` with 'xgboost' engine)
3. ✅ **LightGBM** (`boost_tree` with 'lightgbm' engine)
4. ✅ **CatBoost** (`boost_tree` with 'catboost' engine)
5. ✅ **SVM RBF** (`svm_rbf`)
6. ✅ **SVM Linear** (`svm_linear`)
7. ✅ **k-Nearest Neighbors** (`nearest_neighbor`)
8. ✅ **Neural Networks** (`mlp`)

## Key Benefits Highlighted

### Date Indexing Advantage
- **Visualization**: Easy to plot predictions vs actuals over time
- **Analysis**: Filter by date ranges, aggregate by time periods
- **Comparison**: Align multiple model outputs by date
- **Debugging**: Identify when model performs poorly

### Unified Interface
- Same `extract_outputs()` method for all models
- Consistent three-DataFrame structure
- Works with both traditional TS models and ML models
- Seamless integration with visualization tools

### Workflow Integration
- Works with recipes for feature engineering
- Compatible with workflow composition
- Supports hyperparameter tuning
- Enables model comparison via WorkflowSet

## Location in Documentation

**File**: `docs/user_guide/time_series.rst`
**Section**: "Using ML Models for Time Series Regression"
**Position**: After "Recursive ML Forecasting", before "Panel/Grouped Time Series"
**Lines**: ~315-658 (approximately 343 new lines)

## Build Status

✅ Documentation built successfully
- HTML pages generated in `_build/html`
- 97 warnings (pre-existing, not from this addition)
- No errors

## Access Documentation

```bash
cd /Users/matthewdeane/Documents/Data\ Science/python/_projects/py-tidymodels/docs
source ../py-tidymodels2/bin/activate
make serve
# Open browser to http://localhost:8000
# Navigate to: User Guide > Time Series Modeling
```

## Related Documentation

This section complements:
- **Native Time Series Models** (ARIMA, Prophet, ETS, STL)
- **Hybrid Models** (ARIMA+Boost, Prophet+Boost)
- **Recursive Forecasting** (skforecast integration)
- **Panel/Grouped Time Series** (nested models)
- **Feature Engineering** (recipes for time series)
- **Hyperparameter Tuning** (time series CV)

## Key Messages for Users

1. **ML models work great for time series** when you have rich features
2. **Date indexing is automatic** - no special configuration needed
3. **Same interface as traditional TS models** - unified API
4. **Feature engineering is crucial** - lag features, rolling stats, time features
5. **Outputs enable visualization** - date column preserved throughout
6. **Model comparison is easy** - use WorkflowSet to evaluate multiple models
7. **Works with workflows and recipes** - full preprocessing support

## Example Use Cases Shown

- **Sales forecasting** with price, promotion, temperature features
- **Multi-model comparison** to find best performer
- **Feature engineering** with comprehensive recipe
- **Visualization** of forecasts with matplotlib
- **Metric extraction** from test set
- **Cross-validation** with time series folds

This documentation provides users with everything they need to successfully use ML models for time series regression with date-indexed outputs!
