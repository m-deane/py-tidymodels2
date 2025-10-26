# py-tidymodels: Python Port of R Tidymodels Ecosystem

## Project Vision
Create a unified Python modeling framework for time series regression and forecasting, based on R's tidymodels ecosystem but avoiding its limitations (specifically the modeltime_table/calibrate pattern).

## ⚠️ CRITICAL ARCHITECTURAL DECISIONS

### 1. DO NOT Implement modeltime_table/calibrate Pattern
**Explicitly Avoid:**
- ❌ `modeltime_table()` - table-based model organization
- ❌ `modeltime_calibrate()` - separate calibration phase
- ❌ `modeltime_refit()` - table-based refitting
- ❌ Table-centric workflow pattern

**Reason:** These patterns are clunky, inefficient, don't scale to 100+ models, and conflict with the workflows + workflowsets architecture.

**Alternative:** Use **workflows + workflowsets** for all multi-model comparison and organization.

### 2. Integrate Time Series into parsnip (NOT Separate Package)
**Strategy:**
- Extend parsnip with time series model specifications: `arima_reg()`, `prophet_reg()`, `exp_smoothing()`, `seasonal_reg()`
- Time series models are first-class parsnip models, not a separate package
- All models work seamlessly with workflows and workflowsets
- Recursive forecasting is a wrapper: `recursive()` that converts any ML model to autoregressive forecaster

### 3. Leverage Existing Python Packages
**pytimetk (v2.2.0) - CRITICAL DISCOVERY:**
- ✅ **USE AS-IS** - Production-ready, 66 test files, GPU acceleration
- Saves 2-3 months development time
- Wrap pytimetk functions in recipe steps: `step_lag()` wraps `pytimetk.augment_lags()`
- DO NOT rebuild time series feature engineering from scratch

**skforecast Integration:**
- Use skforecast forecasters as backends for recursive forecasting
- `ForecasterRecursive` → backend for `recursive()` wrapper
- `ForecasterMultiSeries` → backend for panel/grouped models
- Wrap in tidymodels API, don't replicate their API

## Core Functionality Requirements

In the @reference/ folder, there is an R ecosystem called tidymodels. I want to convert these libraries into a unified Python package with time series focus. The unified interface should support:

### Model Fitting Modes

1. **Single Model on Ungrouped Data:**
   ```python
   wf = workflow().add_recipe(recipe).add_model(arima_reg())
   fitted = wf.fit(train_data)
   predictions = fitted.predict(test_data)
   ```

2. **Nested Models (Separate Model Per Group):**
   ```python
   # Fit independent model to each store
   nested_results = (
       workflow_set(workflows=[wf1, wf2])
       .fit_nested(data, group_by="store_id")
   )
   # Output includes group_id column
   ```

3. **Global/Panel Models (Single Model Across Groups):**
   ```python
   # Single model trained on all groups simultaneously
   # Group ID as exogenous variable or feature
   global_model = (
       rand_forest()
       .set_mode("regression")
       .fit("sales ~ date + price | store_id", data)
   )
   ```

### Panel/Grouped Time Series Specifications

**Data Structure Requirements:**
- Required columns: `date_column`, `group_id`, `target`, features
- Data must be sorted by `group_id`, `date_column` ascending
- No missing dates within groups (or handle explicitly)

**Cross-Validation:**
- CV must respect group boundaries (no leakage across groups)
- Nested: Independent train/test splits per group
- Global: Single split, but groups remain together

**Output Format:**
- All DataFrames include `group_id` column for grouped models
- `group_id = NULL` for ungrouped models
- Results stack across all groups with consistent schema

## Standardized Output DataFrames

All fitted models MUST return three standardized DataFrames in tidy/long format. Schema is consistent across ALL model types so multiple models can be stacked without schema changes.

### 1. model_outputs DataFrame

**Columns (Time Series):**
- `model_id`: str - unique model identifier
- `group_id`: str - group identifier for panel data (NULL if ungrouped)
- `date`: datetime - observation timestamp (index for time series)
- `actual`: float - true target value (NULL for future forecasts)
- `fitted`: float - in-sample predictions from training period
- `predicted`: float - out-of-sample predictions (test/forecast periods)
- `residual`: float - actual - (fitted or predicted)
- `split`: str - 'train', 'test', or 'forecast'
- `.pred_lower`: float - lower prediction interval (optional, if available)
- `.pred_upper`: float - upper prediction interval (optional, if available)

**Index:** `[model_id, group_id, date]`
**Sort:** model_id, group_id, date ascending

**Example:**
```
model_id  group_id  date        actual  fitted  predicted  residual  split     .pred_lower  .pred_upper
ARIMA_01  store_A   2024-01-01  100.0   98.5    NULL       1.5       train     NULL         NULL
ARIMA_01  store_A   2024-06-01  NULL    NULL    105.2      NULL      forecast  98.1         112.3
RF_01     store_A   2024-01-01  100.0   99.1    NULL       0.9       train     NULL         NULL
```

### 2. coefficients DataFrame

**Columns:**
- `model_id`: str
- `group_id`: str - for nested models (NULL if global)
- `parameter`: str - parameter name (coefficient name or hyperparameter)
- `value`: float - parameter value
- `std_error`: float - standard error (if statistical model, NULL otherwise)
- `t_statistic`: float - t-statistic (if available, NULL otherwise)
- `p_value`: float - significance level (if available, NULL otherwise)
- `term`: str - original feature/variable name
- `parameter_type`: str - 'coefficient', 'hyperparameter', or 'feature_importance'

**For Models Without Coefficients (ML Models):**
- Return hyperparameters: trees, learning_rate, max_depth, etc.
- Return feature importances if available
- Use `parameter_type = 'hyperparameter'` or `'feature_importance'`

**Example:**
```
model_id  parameter      value   std_error  p_value  term        parameter_type
ARIMA_01  ar.L1          0.85    0.05       0.000    ar.L1       coefficient
ARIMA_01  ma.L1          0.45    0.06       0.000    ma.L1       coefficient
RF_01     n_estimators   500     NULL       NULL     trees       hyperparameter
RF_01     max_depth      10      NULL       NULL     max_depth   hyperparameter
RF_01     importance     0.42    NULL       NULL     price       feature_importance
```

### 3. stats DataFrame

**Columns:**
- `model_id`: str
- `group_id`: str - for nested models (NULL if global)
- `metric`: str - metric name
- `value`: float - metric value
- `split`: str - 'train', 'test', or 'overall'

**Required Metrics:**
- **Fit Statistics:** rmse, mae, mape, smape, mase, r_squared
- **Model Info:** aic, bic, log_likelihood (if available, NULL otherwise)
- **Residual Tests:** durbin_watson, ljung_box_p, shapiro_wilk_p (time series)
- **Sample Sizes:** n_train, n_test, n_forecast
- **Timing:** fit_time_seconds

**Example:**
```
model_id  metric          value    split
ARIMA_01  rmse            5.23     test
ARIMA_01  mae             4.12     test
ARIMA_01  mase            0.89     test
ARIMA_01  aic             1523.4   overall
ARIMA_01  durbin_watson   1.98     overall
ARIMA_01  n_train         365      overall
```

### Stacking Rules
- Multiple models: concatenate with different `model_id` values
- Schema never changes - consistent across all model types
- Missing values: `NULL/NaN` for unavailable metrics (e.g., p_values for ML models)
- All DataFrames are "tidy" - one observation per row

## Model Specifications to Implement

### General ML Models (parsnip core)
As a starting example, implement **linear regression** from scikit-learn and statsmodels as separate engines:
```python
linear_reg().set_engine("sklearn")   # Ridge regression from sklearn
linear_reg().set_engine("statsmodels")  # OLS from statsmodels
```

**Other ML Models (with multiple engines):**
1. **Random Forest** - engines: sklearn, ranger (via rpy2 optional)
2. **Gradient Boosting** - engines: sklearn, xgboost, lightgbm, catboost
3. **XGBoost** - engine: xgboost
4. **LightGBM** - engine: lightgbm
5. **CatBoost** - engine: catboost
6. **LASSO/Ridge/Elastic Net** - engines: sklearn, glmnet (via rpy2 optional)
7. **MARS** - engine: py-earth

### Time Series Models (parsnip extensions)
**Critical:** These are parsnip model specifications, NOT a separate modeltime package!

1. **ARIMA/SARIMAX** - `arima_reg()`
   - Engines: statsmodels, pmdarima (auto_arima)

2. **Prophet** - `prophet_reg()`
   - Engine: prophet

3. **Exponential Smoothing** - `exp_smoothing()`
   - Engine: statsmodels

4. **Seasonal Naive** - `seasonal_reg()`
   - Engine: custom (benchmark models)

5. **Naive/Mean/Drift** - `naive_reg()`, `mean_reg()`, `drift_reg()`
   - Engines: custom (baseline benchmarks)

### Recursive Forecasting Wrapper

**Critical for ML Models on Time Series:**

ML models (Random Forest, XGBoost, etc.) don't natively forecast multiple steps ahead. The `recursive()` wrapper enables multi-step forecasting by converting them to autoregressive models.

**API:**
```python
from py_parsnip import rand_forest, recursive

# Create recursive forecaster
model = (
    rand_forest(trees=500, mtry=10)
    .set_engine("sklearn")
    .set_mode("regression")
    .recursive(
        id="RF_recursive",
        lags=14,           # Use last 14 observations as features
        h=30,              # Forecast 30 steps ahead
        exogenous=["price", "promotion"]  # External regressors
    )
)

# Fit on historical data
fitted = model.fit("sales ~ date", data=train)

# Forecast future
forecast = fitted.predict(new_data=future_data)  # 30 steps ahead
```

**Behavior:**
1. During fit: Create lag features automatically (lag_1, lag_2, ..., lag_14)
2. During predict for step 1: Use actual historical values for lags
3. For step 2+: Use previous predictions as lag inputs (recursive)
4. Return: Multi-step forecasts with date index
5. Exogenous variables: User must provide future values in `new_data`

**Backend:** Use skforecast's `ForecasterRecursive` as implementation, wrap in tidymodels API.

### Time Series Output Requirements

**All models must return predictions indexed by date:**
- model_outputs DataFrame has `date` column as part of index
- Predictions are chronologically ordered
- Future forecasts have `actual = NULL`, `split = 'forecast'`
- Date column type: `datetime64[ns]` (pandas) or `datetime` (Python) 

## Core Infrastructure Components

### 1. Engine Registration System (parsnip)

**Purpose:** Translate unified tidymodels parameters to engine-specific parameters.

**Architecture:**
- Engines are backends: sklearn, statsmodels, xgboost, prophet, etc.
- Each model type (e.g., `rand_forest`) supports multiple engines
- Parameter translation layer: `trees` → sklearn's `n_estimators`

**Registration API:**
```python
@register_engine(model_type="rand_forest", engine="sklearn")
class SklearnRandForestEngine:
    param_mapping = {
        "trees": "n_estimators",
        "mtry": "max_features",
        "min_n": "min_samples_split"
    }

    def fit(self, model_spec, formula, data):
        # Translation and fitting logic
        pass

    def predict(self, model_fit, new_data, type="numeric"):
        # Prediction logic with type handling
        pass
```

**Key Engines:**
- **Time Series:** statsmodels, prophet, pmdarima, skforecast
- **ML:** sklearn, xgboost, lightgbm, catboost

### 2. Model Specifications (parsnip)

**Classes:**
- `ModelSpec`: Stores model type, engine, mode, and hyperparameters
- `ModelFit`: Stores fitted model object, preprocessing info, metadata

**Example:**
```python
spec = rand_forest(trees=500, mtry=10).set_engine("sklearn").set_mode("regression")
fitted = spec.fit("y ~ x1 + x2", data=train)
predictions = fitted.predict(test)
```

### 3. Formula Interface (patsy integration)

**R-style Formulas for Time Series:**
```python
# Basic time series
"sales ~ date + price + promotion"

# All predictors except date
"sales ~ . - date - id"

# Interactions
"sales ~ price * promotion"  # price + promotion + price:promotion

# Grouped/nested models
"sales ~ date | store_id"  # Fit separate model per store
"sales ~ date + price | region + category"  # Nested by region×category
```

**Time Series Special Handling:**
- Date column should NOT be numeric predictor
- Use `recipe().update_role("date", new_role="time_index")`
- Extract date features with `step_date()` instead

### 4. Prediction Intervals (Uncertainty Quantification)

**All time series models must support:**
1. **Point forecasts** (default, `type="numeric"`)
2. **Prediction intervals** (forecast uncertainty, `type="pred_int"`)

**API:**
```python
forecast = fitted_model.predict(
    new_data=future_data,
    type="pred_int",
    level=0.95  # 95% prediction intervals
)
```

**Output Columns:**
- `.pred`: point forecast
- `.pred_lower`: lower bound
- `.pred_upper`: upper bound

**Implementation by Model Type:**
- **ARIMA:** Analytical intervals from statsmodels
- **Prophet:** Built-in uncertainty estimates
- **ML models:** Conformal prediction or quantile regression

### 5. Exogenous Variables (External Regressors)

**Supported in:**
- ARIMA with regressors (ARIMAX/SARIMAX)
- Prophet with additional regressors
- ML models (via recipe features - native support)

**Recipe API:**
```python
recipe_spec = (
    recipe("sales ~ date + price + promotion", data=train)
    .update_role("date", new_role="time_index")
    .update_role("price", "promotion", new_role="exogenous")
    .step_date("date", features=["dow", "month"])
    .step_lag("sales", lags=[1, 2, 7])
)
```

**Forecasting Requirement:**
- User MUST provide future values of exogenous variables in `new_data`
- Raise clear error if exogenous vars missing during prediction
- Document which variables require future values

### 6. Workflows Architecture (PRIMARY ORGANIZING PRINCIPLE)

**Workflows are the core composition layer - NOT modeltime_table!**

**Single Model Workflow:**
```python
wf = (
    workflow()
    .add_recipe(preprocessing_recipe)
    .add_model(model_spec)
)

fitted_wf = wf.fit(train_data)
predictions = fitted_wf.predict(test_data)
```

**Benefits:**
- Automatic preprocessing at prediction time
- Prevents data leakage (preprocessing fit only on train)
- Consistent API for all models
- Integration with tune for hyperparameter optimization

**Multiple Models → Use workflowsets:**
```python
# Create multiple workflows
wf1 = workflow().add_recipe(recipe1).add_model(arima_reg())
wf2 = workflow().add_recipe(recipe1).add_model(prophet_reg())
wf3 = workflow().add_recipe(recipe2).add_model(rand_forest().recursive(lags=7))

# Organize into workflow set
wf_set = workflow_set(
    workflows=[wf1, wf2, wf3],
    ids=["ARIMA", "Prophet", "RF_Recursive"]
)

# Fit to time series CV folds
results = wf_set.fit_resamples(
    resamples=time_series_cv(data, initial="1 year", assess="3 months")
)

# Extract results - all in standard DataFrame format
model_outputs = results.collect_predictions()
stats = results.collect_metrics()
```

**DO NOT implement:**
- ❌ modeltime_table() - replaced by workflow_set()
- ❌ modeltime_calibrate() - happens automatically during fit_resamples()
- ❌ Custom model registry - workflows handle this

## Core Tidymodels Packages to Port

### 1. recipes - Feature Engineering & Preprocessing

**Purpose:** Step-by-step transformations applied consistently to train/test/forecast data.

**Critical Strategy:** **Wrap existing pytimetk package** - DO NOT rebuild from scratch!

**pytimetk Integration:**
```python
from pytimetk import augment_lags, augment_timeseries_signature
from py_recipes import recipe, step_lag, step_date

# Recipe steps wrap pytimetk functions
recipe_spec = (
    recipe("sales ~ date", data=train)
    .step_lag("sales", lags=[1, 2, 7, 14])  # Wraps pytimetk.augment_lags()
    .step_date("date", features=["dow", "month", "quarter"])  # Wraps pytimetk.augment_timeseries_signature()
    .step_holiday("date", country="US")  # Wraps pytimetk.augment_holiday_signature()
    .step_fourier("date", period=365, K=5)  # Wraps pytimetk.augment_fourier()
    .step_rolling("sales", window=7, fn="mean")  # Wraps pytimetk.augment_rolling()
)
```

**pytimetk Capabilities (already built!):**
- Lags and leads (29 timeseries signature features)
- Rolling windows (mean, sum, std, etc.)
- Date/time features (dow, month, quarter, year, etc.)
- Holiday features (US, UK, custom calendars)
- Fourier terms for seasonality
- Differences and pct_change
- GPU acceleration via cuDF (optional)

**Additional Recipe Steps (non-pytimetk):**
- `step_normalize()` - center and scale (sklearn StandardScaler)
- `step_dummy()` - one-hot encoding (sklearn OneHotEncoder)
- `step_impute_*()` - missing value imputation
- `step_filter()` - row filtering
- `step_mutate()` - feature creation (dplyr-style)

**Key Methods:**
- `prep()`: Fit preprocessing on training data
- `bake()`: Apply preprocessing to new data
- `juice()`: Extract preprocessed training data

### 2. parsnip - Unified Model Interface + Time Series Extensions

**Purpose:** Consistent API across all models and engines.

**Core Components:**
- Model specifications: `rand_forest()`, `linear_reg()`, `boost_tree()`
- **Time series extensions:** `arima_reg()`, `prophet_reg()`, `exp_smoothing()`
- Engine registration and parameter translation
- Prediction type handling: `"numeric"`, `"pred_int"`, `"conf_int"`

**Critical:** Time series models are parsnip extensions, NOT a separate package!

### 3. workflows - Recipe + Model Composition

**Purpose:** Bundle preprocessing and modeling into single object.

**API:**
```python
wf = workflow().add_recipe(recipe).add_model(model_spec)
fitted_wf = wf.fit(train)
predictions = fitted_wf.predict(test)
```

**Benefits:**
- Preprocessing automatically applied during prediction
- Prevents data leakage
- Works with tune for hyperparameter optimization

### 4. workflowsets - Multi-Model Comparison (CRITICAL!)

**Purpose:** Run and compare dozens or hundreds of model configurations.

**This replaces modeltime_table/calibrate pattern!**

**API:**
```python
# Option 1: Provide workflows directly
wf_set = workflow_set(
    workflows=[wf1, wf2, wf3],
    ids=["ARIMA", "Prophet", "RF"]
)

# Option 2: Cross all combinations
wf_set = workflow_set(
    preproc=[recipe1, recipe2],
    models=[model1, model2, model3],
    cross=True  # 2×3 = 6 workflows
)

# Fit all workflows to CV folds
results = wf_set.fit_resamples(cv_folds)

# Collect results in standard DataFrames
model_outputs = results.collect_predictions()
stats = results.collect_metrics()
best = results.rank_results(metric="rmse")
```

### 5. tune - Hyperparameter Optimization

**Purpose:** Grid search, random search, Bayesian optimization.

**API:**
```python
from py_tune import tune_grid, tune_bayes, tune

# Mark parameters for tuning
model = rand_forest(trees=tune(), mtry=tune())

# Grid search
grid_results = tune_grid(
    wf,
    resamples=cv_folds,
    grid=10  # 10 combinations
)

# Extract best parameters
best_params = grid_results.select_best(metric="rmse")
final_model = finalize_workflow(wf, best_params)
```

### 6. rsample - Cross-Validation & Resampling

**Purpose:** Time series CV respecting temporal ordering.

**Critical:** Enhance existing `py-modeltime-resample` package!

**Key Functions:**
```python
# Time series CV (rolling/expanding window)
cv_splits = time_series_cv(
    data,
    initial="1 year",    # Initial training period
    assess="3 months",   # Test period size
    skip="1 month",      # Gap between folds
    cumulative=True      # Expanding window (vs. rolling)
)

# Initial split
split = initial_time_split(data, prop=0.8)
train = training(split)
test = testing(split)
```

**Nested CV for Grouped Data:**
```python
nested_cv = nested_time_series_cv(
    data,
    group_by="store_id",
    initial="6 months",
    assess="1 month"
)
```

### 7. yardstick - Performance Metrics

**Purpose:** Standardized evaluation metrics.

**Time Series Metrics (Priority):**
- `rmse()`, `mae()`, `mape()`, `smape()`, `mase()`
- `r_squared()`, `rsq_trad()`
- Custom: `durbin_watson()`, `ljung_box()`, `shapiro_wilk()`

**General Metrics:**
- Regression: `rmse`, `mae`, `r_squared`
- Classification: `accuracy`, `roc_auc`, `f_meas`

### 8. stacks - Model Ensembling

**Purpose:** Combine multiple models via stacking.

**This replaces modeltime.ensemble!**

**Implementation:** Defer to Phase 3 (months 9-11).

### 9. filtro - Feature Selection

**Purpose:** Automated feature selection as recipe steps.

**Implementation Strategy:** Wrap sklearn.feature_selection methods in recipe step API.

**Key Recipe Steps:**
```python
from py_recipes import recipe, step_select_vip, step_select_corr, step_select_boruta

# Variable importance (VIP) selection
recipe_spec = (
    recipe("sales ~ .", data=train)
    .step_select_vip(
        estimator="random_forest",
        threshold=0.01,  # Keep features with importance > 0.01
        top_n=20  # Or keep top N features
    )
)

# Correlation-based filtering
recipe_spec = (
    recipe("sales ~ .", data=train)
    .step_select_corr(
        threshold=0.9,  # Remove features correlated > 0.9
        method="pearson"
    )
)

# Boruta algorithm (all-relevant features)
recipe_spec = (
    recipe("sales ~ .", data=train)
    .step_select_boruta(
        estimator="random_forest",
        max_iter=100,
        alpha=0.05
    )
)
```

**Selection Methods (from R filtro):**
1. **Variable Importance (VIP):**
   - Random Forest importance
   - Permutation importance
   - SHAP values
   - Backend: `sklearn.feature_selection.SelectFromModel`

2. **Statistical Tests:**
   - ANOVA F-test (continuous target)
   - Chi-square test (categorical target)
   - Mutual information
   - Backend: `sklearn.feature_selection.SelectKBest`

3. **Correlation Filtering:**
   - Remove highly correlated features
   - Pearson, Spearman, Kendall
   - Backend: Custom implementation

4. **Boruta Algorithm:**
   - All-relevant feature selection
   - Wrapper around Random Forest
   - Backend: `boruta` package or custom implementation

5. **Recursive Feature Elimination (RFE):**
   - Backward elimination with model refitting
   - Backend: `sklearn.feature_selection.RFE`

**Integration with Workflows:**
```python
# Feature selection in recipe
recipe_with_selection = (
    recipe("sales ~ .", data=train)
    .step_date("date", features=["dow", "month"])
    .step_lag("sales", lags=[1, 7, 14])
    .step_select_vip(estimator="xgboost", top_n=20)  # Select top 20
    .step_normalize()
)

wf = workflow().add_recipe(recipe_with_selection).add_model(xgb_spec)
```

**Output Integration:**
- Selected features appear in coefficients DataFrame
- Dropped features tracked in metadata
- Selection criteria logged in stats DataFrame

### 10. mlflow - Model Tracking and Registry

**Purpose:** Save models with metadata, track experiments.

**Implementation:**
```python
import mlflow

# Track workflow run
with mlflow.start_run():
    fitted_wf = wf.fit(train)
    mlflow.log_params({"model": "ARIMA", "engine": "statsmodels"})
    mlflow.log_metrics({"rmse": 5.2, "mae": 4.1})
    mlflow.sklearn.log_model(fitted_wf, "model")
```

**Defer to Phase 4** (month 12+).

## Visualization Helpers (Plotly Interactive)

Create visualization functions that return interactive Plotly figures. All plots should work with both single models and workflowsets.

### Required Plot Types:

#### 1. Time Series Forecast Plot
```python
from py_tidymodels.viz import plot_forecast

fig = plot_forecast(
    fitted_wf,
    actual_data=train,
    forecast_data=test,
    show_intervals=True,
    title="Sales Forecast - ARIMA vs Prophet"
)
fig.show()
```

**Elements:**
- Actual values (historical) - line chart
- Fitted values (in-sample) - line chart
- Predictions (out-of-sample test) - line chart
- Forecast (future) - line chart
- Prediction intervals - shaded region
- Train/test split - vertical line
- Multiple models overlaid - different colors
- Interactive hover: date, value, model_id

#### 2. Residual Diagnostics
```python
from py_tidymodels.viz import plot_residuals

fig = plot_residuals(fitted_wf, layout="grid")
```

**Subplots:**
- Residuals vs Fitted (scatter)
- Residuals vs Time (time series line)
- QQ Plot (normality check)
- ACF/PACF Plots (autocorrelation)
- Histogram of Residuals (distribution)

#### 3. Model Comparison
```python
from py_tidymodels.viz import plot_model_comparison

fig = plot_model_comparison(
    workflowset_results,
    metric="rmse",
    models=["ARIMA", "Prophet", "RF"]
)
```

**Visualizations:**
- Metric comparison - horizontal bar chart (lower is better)
- Forecast accuracy by horizon - line chart
- Metric heatmap - models × metrics
- Ranking table with metrics

#### 4. Workflow/Tune Results
```python
from py_tidymodels.viz import plot_tune_results

fig = plot_tune_results(tune_results, metric="rmse")
```

**Visualizations:**
- Metric distribution across CV folds - box plots
- Parameter vs Metric scatter plots
- Parallel coordinates - hyperparameters
- Best parameters table

### API Design:
- All functions return `plotly.graph_objects.Figure`
- Compatible with Dash dashboard integration
- Export to HTML: `fig.write_html("forecast.html")`
- Consistent color schemes across plots
- Responsive sizing for mobile/desktop

## Interactive Dashboard (Phase 4 - Month 12+)

**Vision:** Fully interactive web application using Dash + Plotly.

**Features:**
1. **Data Upload:** CSV/Excel file upload with preview
2. **Data Configuration:**
   - Select date column, target variable, exogenous variables
   - Specify group column for panel data
   - Handle missing values and date gaps
3. **Train/Test Split:**
   - Interactive slider for initial/assess periods
   - Visualize split on timeline
4. **Recipe Builder:**
   - Drag-and-drop preprocessing steps
   - Configure step parameters (lag windows, Fourier terms, etc.)
   - Preview transformed data
5. **Model Selection:**
   - Checkboxes for model types
   - Engine selection dropdowns
   - Hyperparameter configuration
6. **Workflow Execution:**
   - Run button to fit all workflows
   - Progress bar with live updates
   - Cancel option for long-running jobs
7. **Results Tabs:**
   - Forecast plot (interactive)
   - Model comparison table and charts
   - Coefficients/hyperparameters table
   - Residual diagnostics
   - Download results as CSV/Excel

**Technology Stack:**
- Dash for web framework
- Plotly for interactive charts
- Pandas for data manipulation
- Background jobs: Celery or Dask (for long-running workflows)

## Implementation Phases

### Phase 1: CRITICAL Foundation (Months 1-4)

**Goal:** Core infrastructure enabling single model workflows with preprocessing.

**Packages to Implement:**
1. **py-hardhat** - mold/forge data preprocessing abstractions
2. **py-rsample** - Enhance existing `py-modeltime-resample` package
   - Add nested CV for grouped data
   - Improve period parsing
3. **py-parsnip** - Core model specs + engines
   - Basic ML: `linear_reg()`, `rand_forest()`, `boost_tree()`
   - Engines: sklearn, statsmodels
   - Time series extensions: `arima_reg()`, `prophet_reg()`
   - Engines: statsmodels, prophet
4. **py-workflows** - Recipe + model composition

**Deliverable:** Can fit single models with preprocessing and time series CV.

**Success Criteria:**
- Fit ARIMA and Random Forest on ungrouped time series data
- Apply recipe with lags (via pytimetk wrapper)
- Evaluate on time series CV folds
- Return standardized model_outputs, coefficients, stats DataFrames

**Documentation Deliverables:**
- API reference documentation for py-parsnip, py-workflows, py-rsample
- Tutorial notebook: `01_getting_started.ipynb`
  - **Use py-tidymodels2 kernel**
  - Load time series data
  - Create train/test split
  - Fit ARIMA and Random Forest models
  - Compare predictions
- Demo script: `examples/basic_workflow_demo.py` (include env verification)
- Demo script: `examples/time_series_cv_demo.py` (include env verification)
- README with installation instructions and quick start example
  - Include py-tidymodels2 environment setup instructions
  - Add "Verify your environment" section
- User guide: "Understanding Model Outputs" (explaining the 3 DataFrames)

### Phase 2: Scale and Evaluate (Months 5-8)

**Goal:** Multi-model comparison and hyperparameter tuning at scale.

**Packages to Implement:**
1. **py-recipes** - Recipe steps wrapping pytimetk
   - `step_lag()`, `step_date()`, `step_holiday()`, `step_fourier()`, `step_rolling()`
   - General steps: `step_normalize()`, `step_dummy()`, `step_impute_*()`
   - **Feature selection (filtro):** `step_select_vip()`, `step_select_corr()`, `step_select_boruta()`
2. **py-tune** - Hyperparameter optimization
   - Grid search, random search
   - Integration with workflows
3. **py-yardstick** - Metrics
   - Time series: rmse, mae, mape, smape, mase
   - Residual tests: durbin_watson, ljung_box, shapiro_wilk
4. **py-workflowsets** - Multi-model comparison (replaces modeltime_table!)
   - Run 10-100 model configurations
   - Parallel processing
   - Results collection in standard DataFrames

**Deliverable:** Can run 100+ model configurations with tuning and compare results.

**Success Criteria:**
- Fit 20 workflows (5 models × 4 recipes) on time series CV
- Tune hyperparameters for XGBoost
- Use feature selection to reduce 50 features to top 20
- Compare all models by RMSE, MAPE, MASE
- Rank workflows and select best

**Documentation Deliverables:**
- API reference documentation for all Phase 2 packages
- Tutorial notebook: `02_recipes_and_feature_engineering.ipynb` (**py-tidymodels2 kernel**)
- Tutorial notebook: `03_hyperparameter_tuning.ipynb` (**py-tidymodels2 kernel**)
- Tutorial notebook: `04_multi_model_comparison.ipynb` (**py-tidymodels2 kernel**)
- Demo script: `examples/feature_selection_demo.py` (include env verification)
- Demo script: `examples/workflowsets_demo.py` (include env verification)
- Update README with Phase 2 capabilities
- **Update requirements.txt** with Phase 2 dependencies (tune, yardstick, workflowsets)

### Phase 3: Advanced Features (Months 9-11)

**Goal:** Recursive forecasting, ensembles, and grouped/panel models.

**Packages to Implement:**
1. **Recursive forecasting wrapper** - `recursive()` for ML models
   - Backend: skforecast's `ForecasterRecursive`
   - API: `rand_forest().recursive(lags=7, h=30)`
2. **Panel/grouped model support**
   - Nested models: fit per group
   - Global models: single model across groups
   - Backend: skforecast's `ForecasterMultiSeries`
3. **py-stacks** - Model ensembling (replaces modeltime.ensemble)
   - Stacking multiple models
   - Weight optimization
4. **py-dials** - Parameter grid generation
   - Smart grids for common models
   - Latin hypercube sampling
5. **Visualization helpers** - Plotly plots
   - `plot_forecast()`, `plot_residuals()`, `plot_model_comparison()`, `plot_tune_results()`

**Deliverable:** Recursive forecasting for ML models, ensembles, panel data support.

**Success Criteria:**
- Forecast 30 days ahead with Random Forest using `recursive()`
- Fit nested models to 50 stores in panel data
- Ensemble ARIMA + Prophet + XGBoost
- Visualize forecast with prediction intervals

**Documentation Deliverables:**
- API reference documentation for recursive forecasting, stacks, dials, viz
- Tutorial notebook: `05_recursive_forecasting.ipynb` (**py-tidymodels2 kernel**)
  - ML models for time series
  - Comparing direct vs recursive forecasting
  - Multi-step ahead predictions
- Tutorial notebook: `06_panel_data_modeling.ipynb` (**py-tidymodels2 kernel**)
  - Nested models (fit per group)
  - Global models (single model across groups)
  - Comparing approaches
- Tutorial notebook: `07_model_ensembling.ipynb` (**py-tidymodels2 kernel**)
  - Stacking multiple models
  - Weight optimization
- Tutorial notebook: `08_visualization.ipynb` (**py-tidymodels2 kernel**)
  - All plot types with examples
  - Customization options
- Demo script: `examples/recursive_ml_demo.py` (include env verification)
- Demo script: `examples/panel_data_demo.py` (include env verification)
- Demo script: `examples/ensemble_demo.py` (include env verification)
- Update README with Phase 3 capabilities
- **Update requirements.txt** with Phase 3 dependencies (stacks, dials, visualization libs)

### Phase 4: Polish and Extend (Month 12+)

**Goal:** Production-ready with dashboard and MLflow integration.

**Features to Add:**
1. **Additional engines:**
   - LightGBM, CatBoost
   - pmdarima (auto_arima)
2. **Interactive Dashboard** (Dash + Plotly)
   - Upload data, configure models, view results
   - 7 tabs: data, split, recipe, models, run, results, export
3. **MLflow integration**
   - Track experiments
   - Save models with metadata
   - Model versioning and deployment
4. **Performance optimizations**
   - Parallel processing for workflowsets
   - Caching for expensive operations
   - GPU acceleration via pytimetk

**Deliverable:** Production-ready framework with dashboard.

**Success Criteria:**
- Dashboard deployed and functional
- All models tracked in MLflow
- Can run 100+ workflows in parallel
- Complete documentation and tutorials

**Documentation Deliverables:**
- Tutorial notebook: `09_dashboard_usage.ipynb` (**py-tidymodels2 kernel**)
  - How to use the interactive dashboard
  - Uploading data and configuring models
  - Interpreting results
- Tutorial notebook: `10_mlflow_integration.ipynb` (**py-tidymodels2 kernel**)
  - Tracking experiments
  - Model versioning
  - Deployment workflows
- Demo script: `examples/dashboard_demo.py` (include env verification)
- Demo script: `examples/mlflow_demo.py` (include env verification)
- **Comprehensive User Guide:**
  - **Environment setup section** (py-tidymodels2 installation and activation)
  - Installation and setup
  - Core concepts (workflows, recipes, parsnip, etc.)
  - Time series modeling best practices
  - Panel/grouped data guide
  - Feature selection guide
  - Troubleshooting and FAQ
    - "Common environment issues" section
- **Complete API Reference:**
  - All packages documented
  - All functions with examples
  - Search functionality
- **Comparison Guide:**
  - py-tidymodels vs R tidymodels
  - py-tidymodels vs scikit-learn
  - py-tidymodels vs skforecast
  - When to use each approach
- Video tutorials (optional):
  - Quick start (10 min) - include environment setup
  - Deep dive series (5-10 videos)
- **Final requirements.txt, requirements-dev.txt, requirements-optional.txt**
  - All dependencies documented with versions
  - Installation instructions tested
  - Optional dependencies clearly marked

**Packages/Features to NEVER Implement:**
- ❌ modeltime_table/calibrate infrastructure
- ❌ Separate py-timetk (use pytimetk instead)
- ❌ modeltime.ensemble (use stacks)

---

## Environment Setup Requirements

### Python Virtual Environment - MANDATORY

**CRITICAL:** Create and use a dedicated Python virtual environment for this project.

**Environment Name:** `py-tidymodels2`

**Setup Steps (Execute Once at Project Start):**

1. **Create the virtual environment:**
   ```bash
   python -m venv py-tidymodels2
   ```

2. **Activate the environment:**
   ```bash
   # macOS/Linux
   source py-tidymodels2/bin/activate

   # Windows
   py-tidymodels2\Scripts\activate
   ```

3. **Upgrade pip and install build tools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install initial dependencies:**
   ```bash
   # Core dependencies
   pip install pandas numpy scipy scikit-learn
   pip install statsmodels prophet pmdarima
   pip install pytimetk skforecast
   pip install plotly dash

   # Development dependencies
   pip install pytest pytest-cov black flake8 mypy
   pip install jupyter ipykernel
   pip install sphinx sphinx-rtd-theme
   ```

5. **Create requirements.txt:**
   ```bash
   pip freeze > requirements.txt
   ```

6. **Register Jupyter kernel (for notebooks):**
   ```bash
   python -m ipykernel install --user --name=py-tidymodels2 --display-name "Python (py-tidymodels2)"
   ```

### Environment Usage Rules - MANDATORY

**ALWAYS activate the environment before any work:**
- ✅ **DO:** Activate `py-tidymodels2` before running ANY script, test, or command
- ✅ **DO:** Verify active environment: `which python` should show path to py-tidymodels2
- ✅ **DO:** Update requirements.txt after installing new packages: `pip freeze > requirements.txt`
- ❌ **NEVER:** Run scripts or tests in system Python or other environments
- ❌ **NEVER:** Install packages globally - always use the virtual environment

**Dependency Management:**
- Track all dependencies in `requirements.txt`
- Use `requirements-dev.txt` for development-only dependencies (testing, docs, linting)
- Document optional dependencies in `requirements-optional.txt` (GPU support, etc.)
- Update dependencies incrementally as new packages are added per phase

**Before Running Any Script:**
```bash
# 1. Verify environment is active
echo $VIRTUAL_ENV  # Should show path to py-tidymodels2

# 2. If not active, activate it
source py-tidymodels2/bin/activate

# 3. Then run your script
python examples/basic_workflow_demo.py
```

**Before Running Tests:**
```bash
# 1. Ensure environment is active
source py-tidymodels2/bin/activate

# 2. Run tests
pytest tests/ -v --cov=py_parsnip
```

**Jupyter Notebooks:**
- **Always** select the "Python (py-tidymodels2)" kernel
- Verify kernel: Check top-right corner of notebook shows "py-tidymodels2"
- If kernel not available, re-run: `python -m ipykernel install --user --name=py-tidymodels2`

### Dependency Files Structure

```
py-tidymodels/
├── requirements.txt              # Core runtime dependencies
├── requirements-dev.txt          # Development dependencies (pytest, black, etc.)
├── requirements-optional.txt     # Optional dependencies (GPU, additional engines)
├── setup.py or pyproject.toml   # Package metadata and dependencies
└── py-tidymodels2/              # Virtual environment (in .gitignore)
```

**requirements.txt (Core):**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
prophet>=1.1.0
pytimetk>=2.2.0
skforecast>=0.12.0
plotly>=5.0.0
patsy>=0.5.0
```

**requirements-dev.txt:**
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
sphinx>=6.0.0
sphinx-rtd-theme>=1.0.0
```

**requirements-optional.txt:**
```
# GPU acceleration
cudf-cu11>=23.0.0  # RAPIDS for pytimetk GPU support
xgboost[gpu]>=2.0.0

# Additional engines
lightgbm>=4.0.0
catboost>=1.2.0
rpy2>=3.5.0  # For ranger engine (optional)
```

### .gitignore Configuration

**CRITICAL:** Add the following to `.gitignore` to prevent committing the virtual environment:

```gitignore
# Virtual environment
py-tidymodels2/
venv/
env/

# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local
```

### Environment Verification Checklist

**Before each implementation session:**
- [ ] Virtual environment activated: `echo $VIRTUAL_ENV`
- [ ] Correct Python: `python --version` shows expected version
- [ ] Dependencies installed: `pip list` shows required packages
- [ ] Tests can run: `pytest --collect-only` shows test collection
- [ ] .gitignore includes `py-tidymodels2/`

**Add to every demo script:**
```python
#!/usr/bin/env python
"""
Demo script for [feature name]

Environment: py-tidymodels2
Run: python examples/demo_script.py
"""
import sys
import os

# Verify virtual environment
if 'py-tidymodels2' not in sys.prefix:
    print("ERROR: Must run in py-tidymodels2 virtual environment!")
    print(f"Current environment: {sys.prefix}")
    print("Activate with: source py-tidymodels2/bin/activate")
    sys.exit(1)

# Your script code here...
```

---

## Project Planning Instructions

### Coding Rules and Claude Code Resources

**CRITICAL:** Code according to the guidelines set out in @.claude/CLAUDE.md

This includes:
- **Direct implementation only** - No mocks, stubs, TODOs, or placeholders
- **Multi-dimensional analysis** for complex requirements (4 observers + synthesis)
- **Anti-pattern elimination** - No social validation, hedging language, or token-wasting patterns
- **Dynamic mode adaptation** - Switch between exploration, implementation, debugging, and optimization modes
- **Quality assurance** - Production-ready error handling, comprehensive validation
- **Test continuously** - Write tests to `tests/` folder after every checkpoint
- **File organization** - No files in root directory, use `.claude_plans/` for project plans

### Use Available Claude Code Resources

The `.claude/` folder contains specialized tools and resources - **USE THEM PROACTIVELY**:

#### 1. **Agents** (`.claude/agents/`)
Use specialized agents for specific tasks:
- Launch agents with the Task tool when tasks match their expertise
- Agents handle complex research, analysis, and specialized workflows
- Run agents in parallel when tasks are independent

#### 2. **Slash Commands** (`.claude/commands/`)
Use custom slash commands via the SlashCommand tool:
- `/create-architecture-documentation` - Generate comprehensive architecture docs
- `/architecture-review` - Review architecture and design patterns
- `/code-review` - Comprehensive code quality review
- `/ultra-think` - Deep analysis and problem solving
- `/generate-tests` - Generate comprehensive test suites
- `/todo` - Manage project todos in todos.md
- `/update-docs` - Update project documentation systematically
- `/workflow-orchestrator` - Orchestrate complex workflows
- `/generate-api-documentation` - Auto-generate API docs

**Example Usage:**
```
When creating architecture documentation for Phase 1, use:
/create-architecture-documentation --full-suite

When reviewing code after major checkpoints, use:
/code-review --full

When generating tests for a new package, use:
/generate-tests py_parsnip/
```

#### 3. **Scripts** (`.claude/scripts/`)
Available utility scripts:
- `context-monitor.py` - Real-time context usage monitoring (already active in status line)

#### 4. **Skills** (`.claude/skills/`)
Available specialized skills via Skill tool:
- `mcp-builder` - For creating MCP servers (if needed for integrations)
- `webapp-testing` - For testing web applications (useful for Phase 4 dashboard)

**When to Use:**
- Dashboard development (Phase 4): Use `webapp-testing` skill
- External integrations: Use `mcp-builder` skill if creating MCP servers

### Reference Research Documentation

**IMPORTANT:** Comprehensive research has already been completed. Reference these documents during implementation:

1. **@TIDYMODELS_RESEARCH_SUMMARY.md** - CRITICAL REFERENCE
   - Complete API references for ALL R tidymodels packages
   - 20+ packages documented with full function signatures
   - skforecast integration strategy
   - pytimetk integration details
   - Implementation priorities and architectural decisions
   - **USE THIS** when implementing any tidymodels functionality

2. **@PROMPT_ANALYSIS_AND_RECOMMENDATIONS.md**
   - Analysis of original requirements
   - Conflicts identified and resolved
   - Missing context added to this prompt
   - **USE THIS** to understand why certain decisions were made

3. **@TIDYMODELS_ECOSYSTEM_RESEARCH.json** (if needed)
   - Technical specifications in JSON format
   - Structured data for automated processing

**When implementing a feature:**
1. Check TIDYMODELS_RESEARCH_SUMMARY.md for R API reference
2. Adapt to Python conventions (snake_case, type hints, etc.)
3. Integrate with existing Python packages (pytimetk, skforecast)
4. Follow standardized output DataFrames (model_outputs, coefficients, stats)

### Planning Workflow

Before we start building this out, I want to do comprehensive planning:

1. **Create `.claude_plans/projectplan.md`** with:
   - High-level checkpoints for each phase (1-4 above)
   - Broken down list of small tasks for each checkpoint
   - Dependencies between tasks
   - Success criteria for each checkpoint
   - Estimated complexity/time for each task
   - **USE:** `/ultra-think` to analyze complex planning decisions

2. **Research Tasks** (if needed - most research already complete):
   - Analyze user needs for specific features
   - Compare implementation approaches
   - Investigate edge cases or technical challenges
   - **USE:** Technical researcher agent for additional research

3. **Feature Planning** (if needed):
   - Prioritize features within each phase
   - Define MVP vs nice-to-have features
   - Plan API design for consistency
   - Reference TIDYMODELS_RESEARCH_SUMMARY.md for R API patterns

4. **Review Process:**
   - Present plan for review
   - Incorporate feedback
   - Finalize before implementation

### Implementation Principles

1. **Simplicity First:** Every change should be as simple as possible, impacting minimal code
2. **Incremental Progress:** Complete one small task at a time, mark as done
3. **Test Continuously:** Write and run tests after each checkpoint
   - **USE:** `/generate-tests` command to create comprehensive test suites
   - **USE:** Test engineer agent for test strategy and coverage analysis
4. **Document Continuously:** Create documentation and demo files after each major checkpoint and feature addition:
   - **After each checkpoint:** Update API reference documentation
     - **USE:** `/generate-api-documentation` command for automated API doc generation
   - **After each major feature:** Create tutorial notebook (.ipynb) demonstrating the feature
   - **After each package:** Create demo script (.py) with working examples
   - **Location:** Notebooks in `docs/tutorials/`, demos in `examples/`, API docs in `docs/api/`
   - **Content requirements:**
     - All code must be runnable without errors
     - Use realistic example data (not just toy examples)
     - Include explanatory markdown cells in notebooks
     - Show expected outputs
     - Explain parameters and options
5. **Code Review After Major Changes:**
   - **USE:** `/code-review --full` after completing each phase
   - **USE:** Code reviewer agent for quality, security, and maintainability checks
6. **Architecture Documentation:**
   - **USE:** `/create-architecture-documentation --full-suite` after Phase 1 foundation
   - **USE:** `/architecture-review` before major architectural decisions
   - **USE:** `/ultra-think` for complex design problems requiring deep analysis
7. **Task Management:**
   - **USE:** `/todo` command to manage project tasks in todos.md
   - Track progress on each checkpoint and feature
   - Mark tasks complete as you finish them
8. **Standardized Outputs:** ALWAYS return model_outputs, coefficients, stats DataFrames
9. **Avoid modeltime_table:** Use workflows + workflowsets architecture instead

### Recommended Workflow Integration

**Phase 1 Start:**
1. **FIRST:** Create and activate `py-tidymodels2` virtual environment (see Environment Setup above)
2. Install initial dependencies and create requirements.txt
3. `/ultra-think` - Analyze Phase 1 architecture decisions
4. `/create-architecture-documentation --arc42` - Create initial architecture docs
5. Create `.claude_plans/phase1_plan.md` with detailed tasks
6. `/todo add` - Add Phase 1 tasks to todos.md

**During Implementation:**
1. **Verify environment:** `echo $VIRTUAL_ENV` (must show py-tidymodels2)
2. Implement feature following CLAUDE.md principles
3. Write tests immediately after implementation
4. **Run tests in py-tidymodels2:** `pytest tests/ -v`
5. `/generate-tests` - Ensure comprehensive test coverage
6. `/code-review` - Review code quality after significant changes
7. `/todo complete` - Mark completed tasks
8. **Update dependencies:** `pip freeze > requirements.txt` (if new packages added)

**After Each Checkpoint:**
1. **Activate py-tidymodels2** if not already active
2. Run all tests and verify passing: `pytest tests/ -v --cov`
3. `/generate-api-documentation` - Update API docs
4. Create tutorial notebook (select py-tidymodels2 kernel)
5. Create demo script (include environment verification code)
6. **Test demo scripts in py-tidymodels2:** `python examples/demo.py`
7. `/update-docs --sync` - Synchronize all documentation
8. `/architecture-review` - Verify architectural consistency

**Phase Completion:**
1. **Final test run in py-tidymodels2:** `pytest tests/ -v --cov --cov-report=html`
2. `/code-review --full` - Comprehensive code review
3. `/architecture-review --full` - Full architecture review
4. Update requirements.txt with all phase dependencies
5. Update `.claude_plans/phase{N}_plan.md` with completion notes
6. Create summary of phase deliverables
7. **Verify all demos run:** Test each demo script in py-tidymodels2

### Documentation Standards

**For Tutorial Notebooks (.ipynb):**
- Start with clear learning objectives
- Use realistic example dataset (time series sales, weather, stock prices, etc.)
- Include markdown explanations between code cells
- Show outputs inline
- End with "Next Steps" section pointing to related tutorials
- Include troubleshooting tips

**For Demo Scripts (.py):**
- Include docstring at top explaining what the script demonstrates
- Use argparse for configurable parameters
- Print progress and results clearly
- Save outputs to files (plots, CSVs) with descriptive names
- Include comments explaining each major step

**For API Documentation:**
- Follow NumPy docstring format
- Include Parameters, Returns, Raises, Examples sections
- Provide at least 2 examples per function: basic and advanced
- Link to related functions and tutorials
- Include mathematical formulas where relevant (using LaTeX)

### Python-Specific Coding Standards

**CRITICAL:** Follow @.claude/CLAUDE.md principles but adapt naming conventions for Python:

**Naming Conventions (Python):**
- **Variables:** `snake_case` (not camelCase)
- **Functions:** `snake_case` with descriptive verbs (e.g., `fit_model`, `create_recipe`)
- **Classes:** `PascalCase` (e.g., `ModelSpec`, `RecipeStep`, `WorkflowSet`)
- **Constants:** `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_ENGINE`, `MAX_ITERATIONS`)
- **Modules/Files:** `snake_case.py` (e.g., `model_spec.py`, `time_series_cv.py`)
- **Packages:** `snake_case` (e.g., `py_parsnip`, `py_workflows`)

**Python Best Practices:**
- Use type hints for function signatures: `def fit(self, data: pd.DataFrame) -> ModelFit:`
- Follow PEP 8 style guide
- Use dataclasses or attrs for data structures
- Prefer composition over inheritance
- Use context managers (`with` statements) for resource management
- Use pathlib.Path for file paths, not strings

**Tidymodels-Specific Python Conventions:**
- Model specification functions return instances: `linear_reg()` returns `ModelSpec`
- Method chaining for fluent API: `model.set_engine("sklearn").set_mode("regression")`
- Recipe steps use `step_*` naming: `step_lag()`, `step_normalize()`
- All fit methods return new instances (immutability where possible)

**Error Handling:**
- Raise informative exceptions with actionable messages
- Use custom exception classes for framework-specific errors
- Validate inputs at function entry points
- Provide helpful error messages for common mistakes

**Testing Requirements:**
- Use pytest framework
- Test files mirror source structure: `tests/py_parsnip/test_model_spec.py`
- Parametrize tests for multiple scenarios
- Include doctests in docstrings for simple examples
- Aim for >90% code coverage

**Performance Considerations:**
- Use NumPy/Pandas operations (vectorized) instead of Python loops
- Cache expensive computations when appropriate
- Use generators for large datasets
- Profile code for bottlenecks before optimizing

We will review the plan together before beginning implementation.