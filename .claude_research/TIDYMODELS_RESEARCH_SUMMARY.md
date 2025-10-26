# Comprehensive Tidymodels Ecosystem Research Report
## Python Conversion Strategy with Time Series Focus

**Research Date:** October 26, 2025
**Research Objective:** Analyze R tidymodels ecosystem for Python conversion with emphasis on time series regression and forecasting

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Tidymodels Packages](#core-tidymodels-packages)
3. [Time Series Extensions](#time-series-extensions)
4. [Package Interdependencies](#package-interdependencies)
5. [Time Series Workflows](#time-series-workflows)
6. [Python Port Strategy](#python-port-strategy)
7. [Gap Analysis](#gap-analysis)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Recommendations](#recommendations)

---

## Executive Summary

### Overview

Tidymodels is a comprehensive R ecosystem for statistical modeling and machine learning that follows tidy data principles. The ecosystem consists of **20+ packages** organized into:

- **Core infrastructure**: recipes, parsnip, workflows, tune, yardstick, rsample, broom, dials
- **Specialized extensions**: workflowsets, stacks, finetune
- **Time series**: modeltime, modeltime.resample, modeltime.ensemble, timetk

### Key Design Principles

1. **Unified interface** across diverse modeling packages
2. **Tidy data structures** (tibbles) throughout
3. **Composable workflow components** via piping (`|>` or `%>%`)
4. **Separation of model specification from implementation**
5. **Consistent API patterns** across all packages
6. **Support for both traditional statistics and machine learning**

### Time Series Capabilities

The **modeltime ecosystem** integrates:
- Classical methods: ARIMA, ETS, Prophet, Seasonal Decomposition
- Machine learning: XGBoost, Random Forest, Neural Networks for forecasting
- Specialized resampling: rolling origin, sliding windows
- Time series preprocessing: lags, rolling stats, date features
- Forecasting workflows: calibration, ensemble, recursive forecasting

### Current Python Port Status

**py-modeltime-resample** (24 Python files) implements:
- ✅ Time series cross-validation (rsample + modeltime.resample functionality)
- ✅ Rolling/expanding window CV
- ✅ Model fitting to resamples
- ✅ Accuracy calculation
- ✅ Visualization (static and interactive)
- ✅ Parallel processing

**Missing components** for complete ecosystem:
- ❌ recipes (preprocessing pipelines)
- ❌ parsnip (unified model interface)
- ❌ workflows (composition layer)
- ❌ tune (hyperparameter optimization)
- ❌ yardstick (comprehensive metrics)
- ❌ workflowsets (experimentation framework)
- ❌ stacks (ensembling)
- ❌ modeltime time series models (as parsnip extensions)

### ⚠️ Critical Architectural Decisions

**1. Reject modeltime_table/calibrate Pattern**
- **Decision**: Do NOT port `modeltime_table()` or `modeltime_calibrate()` functions
- **Reason**: Clunky, inefficient, doesn't scale to workflowsets, incompatible with native tidymodels workflow
- **Alternative**: Use workflows + workflowsets directly for model organization and comparison

**2. Integrate Time Series Models into Parsnip**
- **Decision**: Add time series model specs (arima_reg, prophet_reg, etc.) as parsnip extensions
- **Reason**: Maintains unified interface, works seamlessly with workflows, no separate model table layer needed
- **Benefit**: Time series models work exactly like any other parsnip model

**3. Leverage Existing pytimetk Package**
- **Decision**: Use production-ready pytimetk (v2.2.0) instead of building py-timetk from scratch
- **Reason**: Saves 2-3 months development time, GPU acceleration, 66 test files, professional quality
- **Strategy**: Wrap pytimetk functions in recipe steps (step_lag → pytimetk.augment_lags)

**4. Use Workflowsets Not Model Tables**
- **Decision**: workflowsets is the primary tool for multi-model comparison
- **Reason**: Native tidymodels integration, handles 100+ configs efficiently, parallel processing built-in
- **Benefit**: No custom table/calibrate infrastructure needed

**5. Stacks Replaces modeltime.ensemble**
- **Decision**: Use stacks package for ensembling instead of modeltime.ensemble
- **Reason**: More flexible, works with all models, native integration with tune

See [CRITICAL ARCHITECTURAL DECISION section](#%EF%B8%8F-critical-architectural-decision-avoid-modeltime_tablecalibrate-pattern) for detailed rationale and alternative patterns.

---

## Core Tidymodels Packages

### 1. recipes - Feature Engineering & Preprocessing

**Version:** 1.3.1
**Status:** Stable
**Priority:** HIGH

#### Key Concepts

- **Recipe**: Specification of preprocessing steps
- **Step**: Individual transformation (`step_*()` functions)
- **Role**: Variable designation (predictor, outcome, ID)
- **prep**: Estimate parameters from training data
- **bake**: Apply transformations to new data

#### Time Series Steps (Critical for Port)

```r
# Date/Time Features
step_date(date, features = c('dow', 'month', 'year'))
step_time(datetime, features = c('hour', 'minute'))
step_holiday(date, holidays = timeDate::listHolidays())

# Lag Features
step_lag(target, lag = 1:7)

# Rolling Window Features
step_window(target, window_fn = 'mean', window_size = 7)

# Ordering and Filtering
step_arrange(date)
step_filter(condition)
step_slice(1:100)
```

#### All Step Categories

**Imputation**: bag, knn, linear, mean, median, mode, lower, roll, unknown

**Transformations**: log, sqrt, BoxCox, YeoJohnson, logit, inverse, relu, hyperbolic

**Basis Functions**: bs (B-splines), ns (natural splines), poly (polynomial), harmonic

**Encoding**: dummy, bin2factor, num2factor, string2factor, other, novel, indicate_na

**Normalization**: center, scale, normalize, range

**Filters**: zv (zero variance), nzv (near-zero variance), corr (correlation), lincomb

**Dimensionality Reduction**: pca, pls, ica, kpca, isomap, nnmf

#### Python Gaps

- No native lag/window functions (manual feature engineering needed)
- Limited time series preprocessing
- No built-in holiday features
- Less flexible composition syntax than recipes

---

### 2. parsnip - Unified Model Interface

**Version:** 1.3.3
**Status:** Stable
**Priority:** CRITICAL

#### Key Concepts

- **Model Spec**: Separates model definition from fitting
- **Engine**: Underlying package (ranger, xgboost, glmnet, etc.)
- **Mode**: regression, classification, or censored regression
- **Parameter Harmonization**: Consistent names across engines

#### Model Types

**Linear Models**: `linear_reg()`, `logistic_reg()`, `multinom_reg()`, `poisson_reg()`

**Tree-Based**: `decision_tree()`, `rand_forest()`, `boost_tree()`, `bart()`, `bag_tree()`

**Rules**: `C5_rules()`, `cubist_rules()`, `rule_fit()`

**SVM**: `svm_linear()`, `svm_poly()`, `svm_rbf()`

**Neural Networks**: `mlp()`, `bag_mlp()`

**Other**: `mars()`, `nearest_neighbor()`, `gen_additive_mod()`, `naive_Bayes()`, `pls()`

#### Example: Unified API

```r
# Same interface, different engines
rand_forest(mtry = 10, trees = 2000) |>
  set_engine("ranger") |>  # or "randomForest", "spark"
  set_mode("regression") |>
  fit(y ~ ., data = train_data)
```

#### Argument Harmonization

| Tidymodels | randomForest | ranger | xgboost |
|------------|--------------|--------|---------|
| `trees` | `ntree` | `num.trees` | `nrounds` |
| `mtry` | `mtry` | `mtry` | `colsample_bytree` |
| `penalty` | N/A | N/A | `reg_lambda` |

#### Python Gap

sklearn has no unified parameter interface across packages - each has different APIs.

---

### 3. workflows - Composition Layer

**Version:** 1.1.4
**Status:** Stable
**Priority:** CRITICAL

#### Key Concepts

Bundle preprocessor + model + postprocessor into single object.

```r
# Create workflow
car_wflow <- workflow() |>
  add_recipe(spline_recipe) |>
  add_model(bayes_lm)

# Fit entire pipeline
car_wflow_fit <- fit(car_wflow, data = mtcars)

# Predict (automatic preprocessing)
predict(car_wflow_fit, new_data = test_data)
```

#### Advantages

1. Single object encapsulates entire pipeline
2. No need to track recipe and model separately
3. Automatic preprocessing at prediction time
4. Simplified tuning interface
5. Better production organization

#### Python Equivalent

`sklearn.pipeline.Pipeline` is similar but:
- More restrictive (transformers must have fit/transform)
- No explicit postprocessor support
- Less flexible preprocessor types

---

### 4. tune - Hyperparameter Optimization

**Version:** 1.3.0
**Status:** Stable
**Priority:** HIGH

#### Key Functions

```r
# Grid search
tune_grid(workflow, resamples = cv_folds, grid = param_grid)

# Bayesian optimization
tune_bayes(workflow, resamples = cv_folds, initial = 5, iter = 50)

# Evaluate without tuning
fit_resamples(workflow, resamples = cv_folds)

# Final fit
last_fit(workflow, split = train_test_split)
```

#### Parameter Marking

```r
# Mark parameters for tuning with tune()
recipe(y ~ .) |>
  step_pca(all_predictors(), num_comp = tune()) |>
  step_normalize(all_predictors())

rand_forest(mtry = tune(), trees = tune()) |>
  set_engine("ranger")
```

#### Result Inspection

```r
show_best(tune_results, metric = "rmse", n = 5)
select_best(tune_results, metric = "rmse")
select_by_one_std_err(tune_results, metric = "rmse")
finalize_workflow(workflow, best_params)
```

---

### 5. yardstick - Performance Metrics

**Version:** 1.3.2
**Status:** Stable
**Priority:** HIGH

#### Regression Metrics (Time Series Relevant)

```r
# Error metrics
rmse(data, truth = actual, estimate = predicted)
mae(data, truth = actual, estimate = predicted)
mape(data, truth = actual, estimate = predicted)
mase(data, truth = actual, estimate = predicted)  # Time series specific
smape(data, truth = actual, estimate = predicted)

# Goodness of fit
rsq(data, truth = actual, estimate = predicted)
ccc(data, truth = actual, estimate = predicted)  # Concordance correlation
```

#### Metric Sets

```r
# Create custom metric collection
ts_metrics <- metric_set(rmse, mae, mape, mase, rsq)

# Apply to data
ts_metrics(data, truth = actual, estimate = predicted)
```

#### Classification Metrics

accuracy, precision, recall, f_meas, roc_auc, pr_auc, kap, mcc, sens, spec

#### Python Gap

sklearn.metrics lacks:
- MASE (mean absolute scaled error)
- Easy metric_set concept
- Consistent tidy output format

---

### 6. rsample - Resampling Infrastructure

**Version:** 1.2.1
**Status:** Stable
**Priority:** CRITICAL (partially implemented)

#### Time Series Methods

```r
# Single chronological split
initial_time_split(data, prop = 0.8)

# Rolling origin (SUPERSEDED but still used)
rolling_origin(data, initial = 365, assess = 30, skip = 30, cumulative = FALSE)

# Sliding period (RECOMMENDED)
sliding_period(
  data,
  index = date_column,
  period = "month",
  lookback = "2 years",
  assess_start = 1,
  assess_stop = 3,
  step = 1
)

# Sliding window/index
sliding_window(data, lookback = 365, assess = 30, step = 7)
sliding_index(data, index = date_column, lookback = 365, assess = 30)
```

#### General CV Methods

```r
vfold_cv(data, v = 10)
mc_cv(data, prop = 0.75, times = 25)
bootstraps(data, times = 25)
nested_cv(data, outside = vfold_cv(v = 5), inside = bootstraps(times = 10))
```

#### Python Status

**py-modeltime-resample** implements:
- ✅ `time_series_split()` - Single split
- ✅ `time_series_cv()` - Rolling/expanding window
- ✅ Period-based specifications
- ❌ `sliding_period()` exact equivalent
- ❌ Full rsample API

---

### 7. Additional Core Packages

#### broom - Tidy Model Outputs

```r
# Extract coefficients
tidy(model_fit)

# Model-level statistics
glance(model_fit)

# Add predictions/residuals to data
augment(model_fit, data = test_data)
```

#### dials - Parameter Definitions

```r
# Create parameter grids
grid_regular(mtry(), trees(), levels = 5)
grid_random(penalty(), mixture(), size = 50)
grid_latin_hypercube(cost_complexity(), min_n(), size = 20)
```

#### workflowsets - Grid of Workflows

```r
# Create all combinations of preprocessors × models
workflow_set(
  preproc = list(basic = basic_recipe, pca = pca_recipe),
  models = list(rf = rand_forest_spec, xgb = boost_tree_spec),
  cross = TRUE
)

# Tune all workflows
workflow_map(wf_set, "tune_grid", resamples = cv_folds)

# Rank results
rank_results(wf_set, rank_metric = "rmse")
```

#### stacks - Model Ensembling

```r
# Stack workflow results
stacks() |>
  add_candidates(tune_results_rf) |>
  add_candidates(tune_results_xgb) |>
  blend_predictions() |>  # Determine stacking coefficients
  fit_members()           # Refit on full data
```

#### finetune - Advanced Tuning

```r
# Simulated annealing
tune_sim_anneal(workflow, resamples = cv_folds, iter = 50)

# Racing methods
tune_race_anova(workflow, resamples = cv_folds, grid = param_grid)
tune_race_win_loss(workflow, resamples = cv_folds, grid = param_grid)
```

---

## Time Series Extensions

### 1. modeltime - Core Time Series Package

**Version:** 1.3.1
**Status:** Stable
**Priority:** CRITICAL (with architectural modifications)

---

## ⚠️ CRITICAL ARCHITECTURAL DECISION: Avoid modeltime_table/calibrate Pattern

### What NOT to Implement

**DO NOT port these modeltime functions:**
- ❌ `modeltime_table()` - Clunky table-based model organization
- ❌ `modeltime_calibrate()` - Inefficient calibration pattern
- ❌ `modeltime_refit()` - Table-based refitting
- ❌ Table-centric workflow pattern

### Why This Decision

The R modeltime package's table-based approach has significant limitations:

1. **Performance Issues**: The table pattern adds unnecessary overhead when working with multiple models
2. **Not Scalable**: Difficult to scale to workflowsets with 100+ model configurations
3. **Workflow Incompatibility**: Doesn't integrate smoothly with tidymodels' workflow/workflowsets paradigm
4. **Clunky API**: Extra steps (table creation, calibration) that should be transparent
5. **Limited Flexibility**: Hard to extend or customize model evaluation pipelines

### What TO Implement from modeltime

**✅ Port these valuable components:**

1. **Time Series Model Specifications** (as parsnip extensions)
   - `arima_reg()`, `arima_boost()` - ARIMA models
   - `prophet_reg()`, `prophet_boost()` - Facebook Prophet
   - `exp_smoothing()` - Exponential smoothing (ETS)
   - `seasonal_reg()` - Seasonal decomposition models
   - `nnetar_reg()` - Neural network autoregression
   - `naive_reg()` - Naive forecasting benchmarks

2. **Recursive Forecasting Infrastructure**
   - `recursive()` wrapper for converting ML models to autoregressive forecasters
   - `panel_tail()` for multi-series recursive forecasting
   - Lag transformation utilities

3. **Time Series-Specific Engines**
   - statsmodels engines (ARIMA, SARIMAX, ETS)
   - prophet engine
   - pmdarima (auto_arima) engine
   - Integration with skforecast

4. **Forecasting Utilities**
   - Horizon specification (`h = "3 months"`)
   - Prediction interval generation
   - Multi-step ahead forecasting
   - Exogenous variable handling

### Recommended Alternative Architecture

Instead of the table/calibrate pattern, use **workflows + workflowsets** directly:

```python
# ❌ AVOID: modeltime table pattern
models_tbl = modeltime_table(model1, model2, model3)
calibrated = models_tbl.modeltime_calibrate(test)

# ✅ USE: workflows + workflowsets pattern
from py_workflows import workflow
from py_workflowsets import workflow_set
from py_tune import fit_resamples

# Create workflows
wf1 = workflow().add_recipe(recipe).add_model(arima_reg())
wf2 = workflow().add_recipe(recipe).add_model(prophet_reg())
wf3 = workflow().add_recipe(recipe).add_model(rand_forest())

# Option 1: Workflow set for multiple models
wf_set = workflow_set(
    workflows=[wf1, wf2, wf3],
    ids=["ARIMA", "Prophet", "RandomForest"]
)

# Fit to time series CV
results = wf_set.fit_resamples(
    resamples=time_series_cv(data, initial="1 year", assess="3 months")
)

# Option 2: Single workflow with tuning
tuned_wf = tune_grid(
    wf1,
    resamples=cv_splits,
    grid=param_grid
)

# Generate forecasts (workflow handles preprocessing automatically)
forecast = fitted_wf.predict(future_data)
```

### Benefits of Workflow-Based Approach

1. **Native tidymodels Integration**: Works seamlessly with recipes, tune, yardstick
2. **Scalability**: workflowsets handle 100+ model configs efficiently
3. **Automatic Preprocessing**: Recipes applied automatically during prediction
4. **Parallel Processing**: Built-in parallel support via tune
5. **Consistent API**: Same pattern for all models (TS and non-TS)
6. **Flexible Evaluation**: Use tune::fit_resamples or py-modeltime-resample functions
7. **No Redundant Steps**: No separate "calibration" phase - just fit and predict

### Implementation Strategy

1. **Extend parsnip**: Add time series model types (arima_reg, prophet_reg, etc.)
2. **Create engines**: Implement statsmodels, prophet, pmdarima backends
3. **Recursive wrapper**: Implement as workflow postprocessor or model wrapper
4. **Integrate with workflows**: Ensure time series models work in workflow()
5. **Use workflowsets**: Leverage for multi-model comparison (no custom table needed)
6. **Extend py-modeltime-resample**: Make it work with workflows instead of custom tables

---

#### Available Models

**Classical Time Series:**
```r
arima_reg() |> set_engine("auto_arima")
arima_boost() |> set_engine("auto_arima_xgboost")
exp_smoothing() |> set_engine("ets")
seasonal_reg() |> set_engine("stlm_ets")
prophet_reg() |> set_engine("prophet")
prophet_boost() |> set_engine("prophet_xgboost")
nnetar_reg() |> set_engine("nnetar")
naive_reg() |> set_engine("naive")
```

**Machine Learning for Forecasting:**
```r
# Any parsnip model
linear_reg() |> set_engine("lm")
rand_forest() |> set_engine("ranger")
boost_tree() |> set_engine("xgboost")
mars() |> set_engine("earth")
svm_rbf() |> set_engine("kernlab")
```

#### Modeltime Workflow (R - FOR REFERENCE ONLY)

> ⚠️ **DO NOT IMPLEMENT THIS PATTERN IN PYTHON**
>
> This R workflow uses the table/calibrate pattern which has been **deliberately excluded** from the Python implementation. See the "CRITICAL ARCHITECTURAL DECISION" section above for details and alternatives.
>
> This is shown for reference to understand what R users are familiar with, NOT as a pattern to replicate.

```r
# ❌ This pattern should NOT be ported to Python
# 1. Create models
model_arima <- arima_reg() |> set_engine("auto_arima")
model_prophet <- prophet_reg() |> set_engine("prophet")
model_xgb <- boost_tree() |> set_engine("xgboost")

# 2. Organize in table (AVOID IN PYTHON)
models_tbl <- modeltime_table(
  model_arima,
  model_prophet,
  model_xgb
)

# 3. Calibrate on holdout set (AVOID IN PYTHON)
calibrated_tbl <- models_tbl |>
  modeltime_calibrate(new_data = test_data)

# 4. Calculate accuracy
calibrated_tbl |> modeltime_accuracy()

# 5. Generate forecasts
forecasts <- calibrated_tbl |>
  modeltime_forecast(
    new_data = test_data,
    actual_data = train_data,
    h = "3 months"
  )

# 6. Visualize
plot_modeltime_forecast(forecasts)

# 7. Refit on full data (AVOID IN PYTHON)
refitted_tbl <- calibrated_tbl |>
  modeltime_refit(data = full_data)

# 8. Future forecast
future_forecast <- refitted_tbl |>
  modeltime_forecast(h = "6 months")
```

**Python Alternative (Recommended):**

```python
# ✅ Use workflows + workflowsets instead
from py_workflows import workflow
from py_parsnip import arima_reg, prophet_reg, rand_forest
from py_recipes import recipe, step_date, step_lag
from py_workflowsets import workflow_set
from py_rsample import time_series_cv

# Create preprocessing recipe
ts_recipe = (
    recipe("value ~ date", data=train)
    .step_date("date", features=["dow", "month", "year"])
    .step_lag("value", lags=[1, 7, 30])
)

# Create model specifications
arima_spec = arima_reg().set_engine("statsmodels")
prophet_spec = prophet_reg().set_engine("prophet")
rf_spec = rand_forest(trees=1000).set_engine("sklearn")

# Create workflows
wf_arima = workflow().add_recipe(ts_recipe).add_model(arima_spec)
wf_prophet = workflow().add_recipe(ts_recipe).add_model(prophet_spec)
wf_rf = workflow().add_recipe(ts_recipe).add_model(rf_spec)

# Combine into workflow set
wf_set = workflow_set(
    workflows=[wf_arima, wf_prophet, wf_rf],
    ids=["ARIMA", "Prophet", "RandomForest"]
)

# Evaluate on time series CV (replaces calibrate)
cv_results = wf_set.fit_resamples(
    resamples=time_series_cv(train, initial="1 year", assess="3 months"),
    metrics=metric_set(rmse, mae, mape)
)

# Select best model and fit on full data
best_wf_id = cv_results.rank_results(metric="rmse").iloc[0]["wflow_id"]
final_wf = wf_set.workflows[best_wf_id].fit(train)

# Generate forecasts (automatic preprocessing)
forecasts = final_wf.predict(future_data)

# Plot
plot_forecast(forecasts, train)
```

#### Recursive Forecasting

Convert ML models to recursive multi-step forecasters:

```r
# Make recursive (auto-regressive)
model_recursive <- linear_reg() |>
  set_engine("lm") |>
  recursive(
    transform = lag_transformer,
    train_tail = panel_tail(data, id_column, date_column, tail_length)
  )
```

#### Model Engines by Type

| Model | Available Engines |
|-------|-------------------|
| arima_reg | auto_arima, arima |
| arima_boost | auto_arima_xgboost, arima_xgboost |
| exp_smoothing | ets, smooth_es, croston, theta |
| seasonal_reg | stlm_ets, stlm_arima, tbats |
| prophet_reg | prophet |
| prophet_boost | prophet_xgboost |
| nnetar_reg | nnetar |

---

### 2. modeltime.resample - Time Series CV

**Version:** 0.2.3
**Status:** Stable (Python: Partial)

#### R Functions

```r
# Fit models to time series CV folds
modeltime_fit_resamples(
  models_tbl,
  resamples = cv_splits,
  control = control_resamples()
)

# Visualize resample performance
plot_modeltime_resamples(resample_results)

# Summarize accuracy across folds
summarize_accuracy(resample_results)
```

#### Python Implementation Status

**py-modeltime-resample** includes:

✅ **Implemented:**
- `time_series_split()` - Single train/test split
- `time_series_cv()` - Rolling/expanding window CV
- `fit_resamples()` - Fit models to CV folds
- `resample_accuracy()` - Calculate metrics across folds
- `plot_time_series_cv_plan()` - Visualize CV plan
- `plot_resamples()` - Plot fitted/predicted values
- `evaluate_model()` - High-level convenience function
- `compare_models()` - Multi-model comparison
- `fit_resamples_parallel()` - Parallel processing
- `create_interactive_dashboard()` - Interactive Dash app
- `plot_model_comparison_matrix()` - Heatmap/radar charts

❌ **Missing:**
- Integration with workflows (doesn't exist yet)
- Direct modeltime model support
- Some specialized resampling methods

---

### 3. timetk - Time Series Feature Engineering

**Version:** 2.9.0
**Status:** Stable
**Priority:** HIGH

#### Key Capabilities

**Date/Time Features:**
```r
tk_augment_timeseries_signature(data, date_column)
# Adds: year, half, quarter, month, day, hour, minute, second, wday, mday, qday, yday, etc.
```

**Lag Features:**
```r
tk_augment_lags(data, target, .lags = 1:7)
```

**Differences:**
```r
tk_augment_differences(data, target, .differences = 1:2)
```

**Rolling Window Features:**
```r
tk_augment_slidify(
  data,
  target,
  .period = 7,
  .f = mean,
  .align = "center"
)
```

**Fourier Features:**
```r
tk_augment_fourier(data, date_column, .periods = c(7, 14, 30, 365))
```

**Holidays:**
```r
tk_augment_holiday_signature(data, date_column)
```

**Visualization:**
```r
plot_time_series(data, date_column, value)
plot_seasonal_diagnostics(data, date_column, value)
plot_acf_diagnostics(data, date_column, value)
```

#### Python Gaps

- pandas has basic date features but not as comprehensive
- No built-in holiday calendar
- Manual rolling window creation
- No Fourier feature generation
- Consider integrating into py-recipes as time series steps

---

### 4. Other Modeltime Ecosystem Packages

**modeltime.ensemble** - Ensemble methods
- `ensemble_average()` - Simple averaging
- `ensemble_weighted()` - Weighted by accuracy
- `ensemble_model_spec()` - Metalearner stacking

**modeltime.h2o** - H2O AutoML integration

**modeltime.gluonts** - Deep learning time series (GluonTS)

---

## Package Interdependencies

### Dependency Graph

```
hardhat (foundation)
    ↓
├── rsample (resampling) ─────────┐
├── recipes (preprocessing) ──────┤
├── parsnip (models) ─────────────┤
│                                 ↓
└────→ workflows (composition) ←──┤
           ↓                      │
    ┌──────┴───────┐              │
    ↓              ↓              │
  tune         workflowsets       │
    ↓              ↓              │
    └──────┬───────┘              │
           ↓                      │
       stacks (ensembling)        │
                                  │
dials (parameters) ───────────────┘
yardstick (metrics) ──────────────┘
broom (tidying) ──────────────────┘

modeltime (time series)
    ↓
├── Uses: parsnip, workflows, rsample, yardstick
├── modeltime.resample
└── timetk (feature engineering)
```

### Typical Workflow Sequence

**General ML Workflow:**
1. **rsample** - Create train/test split or CV folds
2. **recipes** - Define preprocessing steps
3. **parsnip** - Specify model type and engine
4. **workflows** - Combine recipe + model
5. **dials** - Define parameter grid (if tuning)
6. **tune** - Find optimal parameters via CV
7. **yardstick** - Evaluate performance
8. **workflows** - Finalize with best parameters
9. **parsnip** - Fit final model
10. **yardstick** - Assess on test set

**Time Series Workflow:**
1. **timetk** - Feature engineering (lags, rolling, date features)
2. **rsample** - Time series CV (sliding_period, rolling_origin)
3. **recipes** - Additional preprocessing (normalization, etc.)
4. **modeltime** - Create time series models (arima_reg, prophet_reg, ML models)
5. **modeltime_table** - Organize models
6. **modeltime.resample** - Fit to CV folds OR **tune** for hyperparameters
7. **modeltime_calibrate** - Calibrate on holdout set
8. **modeltime_accuracy** - Calculate accuracy metrics
9. **modeltime.ensemble** - Create ensemble (optional)
10. **modeltime_forecast** - Generate forecasts
11. **plot_modeltime_forecast** - Visualize

---

## Time Series Workflows

> ⚠️ **Note**: These R workflows show modeltime_table/calibrate patterns. For Python implementation, use the workflow/workflowset approach described in the "CRITICAL ARCHITECTURAL DECISION" section above.

### 1. Classical Forecasting Workflow (R - Adapt to Workflow Pattern for Python)

```r
library(tidymodels)
library(modeltime)
library(timetk)

# Data prep
splits <- initial_time_split(data, prop = 0.8)
train <- training(splits)
test <- testing(splits)

# Create models
model_arima <- arima_reg() |>
  set_engine("auto_arima") |>
  fit(value ~ date, data = train)

model_prophet <- prophet_reg() |>
  set_engine("prophet") |>
  fit(value ~ date, data = train)

model_ets <- exp_smoothing() |>
  set_engine("ets") |>
  fit(value ~ date, data = train)

# Modeltime workflow
models_tbl <- modeltime_table(
  model_arima,
  model_prophet,
  model_ets
)

# Calibrate and evaluate
calibrated <- models_tbl |>
  modeltime_calibrate(new_data = test)

accuracy <- calibrated |> modeltime_accuracy()

# Forecast
forecast <- calibrated |>
  modeltime_forecast(
    new_data = test,
    actual_data = train
  )

plot_modeltime_forecast(forecast)
```

### 2. ML-Based Forecasting Workflow

```r
# Feature engineering
recipe_spec <- recipe(value ~ date, data = train) |>
  # Date features
  step_date(date, features = c("dow", "month", "year")) |>
  step_holiday(date, holidays = timeDate::listHolidays()) |>
  # Lag features
  step_lag(value, lag = 1:7) |>
  # Rolling features
  step_window(value, window_fn = "mean", window_size = 7) |>
  # Preprocessing
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# Model
model_spec <- rand_forest(mtry = tune(), trees = tune()) |>
  set_engine("ranger") |>
  set_mode("regression")

# Workflow
wf <- workflow() |>
  add_recipe(recipe_spec) |>
  add_model(model_spec)

# Time series CV
cv_splits <- sliding_period(
  train,
  index = date,
  period = "month",
  lookback = "6 months",
  assess_start = 1,
  assess_stop = 1,
  step = 1
)

# Tune
tune_results <- tune_grid(
  wf,
  resamples = cv_splits,
  grid = 20,
  metrics = metric_set(rmse, mae, mape)
)

# Finalize and fit
best_params <- select_best(tune_results, "rmse")
final_wf <- finalize_workflow(wf, best_params)
final_fit <- fit(final_wf, data = train)

# Make recursive for multi-step forecasting
final_recursive <- final_fit |>
  recursive(
    transform = lag_transformer,
    train_tail = tail(train, 50)
  )

# Forecast
forecast <- modeltime_table(final_recursive) |>
  modeltime_forecast(h = "3 months")
```

### 3. Ensemble Workflow

```r
# Create multiple diverse models
model_list <- list(
  arima = arima_reg() |> set_engine("auto_arima"),
  prophet = prophet_reg() |> set_engine("prophet"),
  ets = exp_smoothing() |> set_engine("ets"),
  rf = rand_forest() |> set_engine("ranger"),
  xgb = boost_tree() |> set_engine("xgboost")
)

# Fit all models
models_fitted <- model_list |>
  map(~ fit(.x, value ~ date, data = train))

# Organize
models_tbl <- do.call(modeltime_table, models_fitted)

# Calibrate
calibrated <- models_tbl |>
  modeltime_calibrate(test)

# Ensemble approach 1: Simple average
ensemble_avg <- calibrated |>
  ensemble_average(type = "mean")

# Ensemble approach 2: Weighted by accuracy
ensemble_weighted <- calibrated |>
  ensemble_weighted(
    loadings = c(0.3, 0.3, 0.1, 0.15, 0.15)
  )

# Ensemble approach 3: Stacking
ensemble_stack <- calibrated |>
  ensemble_model_spec(
    model_spec = linear_reg() |> set_engine("lm"),
    kfolds = 5
  )

# Compare
comparison <- modeltime_table(
  ensemble_avg,
  ensemble_weighted,
  ensemble_stack
) |>
  modeltime_accuracy()
```

---

## Python Port Strategy

### Priority Tiers

#### Tier 1: CRITICAL (Months 1-4)
**Packages:** hardhat, rsample, parsnip (with time series extensions), workflows

**Rationale:** Core infrastructure for any modeling workflow, especially time series

**⚠️ Architectural Note:** Do NOT implement modeltime as a separate package with table/calibrate pattern. Instead, integrate time series model specifications directly into parsnip as extensions.

**Approach:**
1. Start with **hardhat** abstractions (mold/forge pattern)
2. Build **rsample** (enhance existing py-modeltime-resample)
3. Develop **parsnip** (unified model interface)
4. **Extend parsnip with time series models** (arima_reg, prophet_reg, etc.) - NO separate modeltime_table package
5. Create **workflows** (composition layer that works seamlessly with all models including time series)

#### Tier 2: HIGH PRIORITY (Months 5-8)
**Packages:** recipes (wrapping pytimetk), tune, yardstick, workflowsets

**Rationale:** Essential for preprocessing, optimization, and evaluation at scale

**⚠️ Key Decision:** Use existing **pytimetk** package instead of building py-timetk from scratch (saves 2-3 months!)

**Approach:**
- **recipes**: Create step_* wrappers around pytimetk functions (step_lag → pytimetk.augment_lags, etc.)
- **tune**: Integrate grid/Bayesian search with workflows
- **yardstick**: Comprehensive metrics including time series (MASE, etc.)
- **workflowsets**: CRITICAL for running multiple models - replaces modeltime_table pattern

#### Tier 3: MEDIUM PRIORITY (Months 9-11)
**Packages:** stacks, dials, broom

**Rationale:** Ensembling and parameter grids enhance modeling capabilities

**Approach:**
- **stacks**: Model ensembling - replaces modeltime.ensemble functionality
- **dials**: Parameter grid generation
- **broom**: Tidy model outputs (lower priority)

#### Tier 4: LOW PRIORITY (Month 12+)
**Packages:** finetune, advanced integrations

**Rationale:** Advanced features that can be deferred

**⚠️ Do NOT Implement:**
- ❌ modeltime.ensemble (use stacks instead)
- ❌ modeltime.h2o (niche use case)
- ❌ modeltime.gluonts (deep learning - consider separate project)

**Consider for Later:**
- ✅ finetune (simulated annealing, racing methods)
- ✅ Additional time series engines (tbats, croston, theta)
- ✅ Advanced recursive forecasting strategies

---

### Architecture Recommendations

#### Base Classes

```python
# hardhat foundation
class Preprocessor(ABC):
    """Base class for recipes, formulas, variable specs"""
    @abstractmethod
    def mold(self, data): pass

    @abstractmethod
    def forge(self, new_data, blueprint): pass

class ModelSpec(ABC):
    """Parsnip-like model specification"""
    def __init__(self, mode=None, engine=None):
        self.mode = mode
        self.engine = engine
        self.args = {}

    def set_mode(self, mode):
        return self.__class__(mode=mode, engine=self.engine)

    def set_engine(self, engine, **kwargs):
        return self.__class__(mode=self.mode, engine=engine)

    @abstractmethod
    def fit(self, formula, data): pass

    @abstractmethod
    def predict(self, new_data): pass

class Workflow:
    """Container for preprocessor + model + postprocessor"""
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.postprocessor = None

    def add_recipe(self, recipe):
        self.preprocessor = recipe
        return self

    def add_model(self, model):
        self.model = model
        return self

    def fit(self, data):
        # mold data with preprocessor
        # fit model
        # return fitted workflow
        pass

    def predict(self, new_data):
        # forge new data
        # predict with model
        # apply postprocessor
        pass
```

#### Design Patterns

- **Builder**: For model/recipe specification
- **Strategy**: For different engines
- **Template Method**: For fit/predict workflows
- **Composite**: For workflows
- **Factory**: For model creation

#### API Principles

1. **Method chaining** for fluent interface (like pandas)
2. **Consistent naming** (fit, predict, transform)
3. **Type hints** throughout
4. **Immutable specifications** (create new on update)
5. **Integration with pandas** DataFrames

---

### Package Structure

```
py-tidymodels/
├── py-hardhat/          # Foundation
│   ├── hardhat/
│   │   ├── __init__.py
│   │   ├── mold.py
│   │   ├── forge.py
│   │   ├── blueprint.py
│   │   └── validate.py
│   └── setup.py
│
├── py-rsample/          # Resampling (enhance existing)
│   ├── rsample/
│   │   ├── __init__.py
│   │   ├── splits.py
│   │   ├── rset.py
│   │   └── time_series.py
│   └── setup.py
│
├── py-parsnip/          # Models
│   ├── parsnip/
│   │   ├── __init__.py
│   │   ├── model_spec.py
│   │   ├── engines/
│   │   │   ├── sklearn_engine.py
│   │   │   ├── statsmodels_engine.py
│   │   │   └── xgboost_engine.py
│   │   ├── models/
│   │   │   ├── linear_reg.py
│   │   │   ├── rand_forest.py
│   │   │   └── boost_tree.py
│   │   ├── fit.py
│   │   └── predict.py
│   └── setup.py
│
├── py-recipes/          # Preprocessing
│   ├── recipes/
│   │   ├── __init__.py
│   │   ├── recipe.py
│   │   ├── step.py
│   │   ├── steps/
│   │   │   ├── datetime.py     # step_date, step_time, step_holiday
│   │   │   ├── lag.py          # step_lag
│   │   │   ├── window.py       # step_window
│   │   │   ├── normalization.py
│   │   │   ├── encoding.py
│   │   │   └── filters.py
│   │   ├── prep.py
│   │   ├── bake.py
│   │   └── roles.py
│   └── setup.py
│
├── py-workflows/        # Composition
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── workflow.py
│   │   ├── fit.py
│   │   └── predict.py
│   └── setup.py
│
├── py-tune/             # Optimization
│   ├── tune/
│   │   ├── __init__.py
│   │   ├── grid.py
│   │   ├── bayes.py
│   │   ├── fit_resamples.py
│   │   └── control.py
│   └── setup.py
│
├── py-yardstick/        # Metrics
│   ├── yardstick/
│   │   ├── __init__.py
│   │   ├── regression.py
│   │   ├── classification.py
│   │   ├── time_series.py
│   │   └── metric_set.py
│   └── setup.py
│
├── py-modeltime/        # Time series
│   ├── modeltime/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── arima_reg.py
│   │   │   ├── prophet_reg.py
│   │   │   └── exp_smoothing.py
│   │   ├── table.py
│   │   ├── calibrate.py
│   │   ├── forecast.py
│   │   ├── accuracy.py
│   │   ├── recursive.py
│   │   └── plot.py
│   └── setup.py
│
└── py-timetk/           # Time series features
    ├── timetk/
    │   ├── __init__.py
    │   ├── augment.py
    │   ├── features.py
    │   └── plot.py
    └── setup.py
```

---

## Gap Analysis

### Python Ecosystem Current State

#### Preprocessing
**Available:** sklearn.preprocessing, category_encoders, feature-engine

**Gaps:**
- No recipe-style pipeline builder with roles
- Limited time series preprocessing
- No step_lag, step_window equivalents
- No built-in holiday features
- Less composable than recipes

#### Modeling
**Available:** sklearn models, statsmodels, xgboost, lightgbm, prophet

**Gaps:**
- No unified interface (parsnip-equivalent)
- Different APIs across packages
- No separation of specification from fitting
- No consistent parameter naming

#### Workflows
**Available:** sklearn.pipeline

**Gaps:**
- Less flexible than workflows
- No explicit preprocessor types
- Limited postprocessing support
- Harder to update components

#### Tuning
**Available:** sklearn GridSearchCV/RandomizedSearchCV, optuna, hyperopt

**Gaps:**
- Not integrated with pipeline builders
- Different APIs for different strategies
- Limited recipe parameter tuning
- No workflowsets concept

#### Resampling
**Available:** sklearn.model_selection, sktime

**Gaps:**
- TimeSeriesSplit less flexible than rsample
- No period-based specifications
- Limited support for irregular time series
- **Partially addressed by py-modeltime-resample**

#### Metrics
**Available:** sklearn.metrics

**Gaps:**
- Limited time series metrics (no MASE)
- No metric_set concept
- Different API patterns

#### Time Series Forecasting
**Available:** statsmodels.tsa, prophet, sktime, darts

**Gaps:**
- No unified interface across packages
- No calibration concept
- Limited ensemble capabilities
- No recursive ML wrapper
- Different workflows for ARIMA vs Prophet vs ML
- **No modeltime-equivalent orchestration**

---

### py-modeltime-resample Status

#### ✅ Implemented
- Time series splitting (single and CV)
- Rolling and expanding window CV
- Period-based specifications
- Model fitting to resamples
- Accuracy calculation
- Visualization (static and interactive)
- Parallel processing
- Interactive dashboard
- Model comparison matrix

#### ❌ Missing from R Version
- Integration with workflows (doesn't exist yet in Python)
- Direct modeltime model support (would need py-modeltime)
- Some specialized resampling methods
- Automatic feature engineering integration

#### Architecture Assessment

**Strengths:**
- Clean functional API
- Pandas-centric design
- Scikit-learn compatible
- Good documentation
- Interactive features exceed R version

**Areas for Improvement:**
- Better integration with future workflow system
- More resampling strategies
- Enhanced time series specific features

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-4)
**Goal:** Basic modeling pipeline for time series

**Packages:**
1. **py-hardhat** (core abstractions)
   - mold/forge pattern
   - blueprint system
   - validation utilities

2. **py-rsample** (enhance existing)
   - Time series CV methods
   - split/rset abstractions
   - integration with hardhat

3. **py-parsnip** (basic models)
   - Model specification API
   - Engine system
   - Parameter harmonization
   - fit/predict methods
   - Initial engines: sklearn, statsmodels

**Deliverable:** Can specify model, create CV splits, fit, predict

**Example Usage:**
```python
from py_parsnip import linear_reg, rand_forest
from py_rsample import time_series_cv

# Specify model
model = (
    rand_forest(mtry=10, trees=100)
    .set_engine("sklearn")
    .set_mode("regression")
)

# Create CV splits
cv_splits = time_series_cv(
    data,
    initial="1 year",
    assess="3 months",
    skip="1 month",
    cumulative=False
)

# Fit
fitted = model.fit("value ~ .", data=train)

# Predict
predictions = fitted.predict(test)
```

---

### Phase 2: Workflows & Preprocessing (Months 5-8)
**Goal:** Preprocessing and composition

**Packages:**
1. **py-recipes** (core steps)
   - Recipe class
   - step_lag, step_window
   - step_date, step_holiday
   - step_normalize, step_dummy
   - prep/bake methods

2. **py-workflows**
   - Workflow container
   - add_recipe/model
   - fit/predict pipeline

3. **py-yardstick** (basic metrics)
   - Regression metrics (rmse, mae, mape, mase)
   - Classification metrics
   - metric_set

**Deliverable:** Can preprocess, combine with model, evaluate

**Example Usage:**
```python
from py_recipes import recipe, step_lag, step_date, step_normalize
from py_workflows import workflow
from py_yardstick import metric_set, rmse, mae, mase

# Create recipe
rec = (
    recipe("value ~ date", data=train)
    .step_date("date", features=["dow", "month", "year"])
    .step_lag("value", lags=[1, 2, 3, 7])
    .step_normalize(all_numeric_predictors())
)

# Create workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(model)
)

# Fit
fitted_wf = wf.fit(train)

# Predict and evaluate
preds = fitted_wf.predict(test)

metrics = metric_set(rmse, mae, mase)
results = metrics(test, truth="value", estimate=preds)
```

---

### Phase 3: Optimization (Months 9-11)
**Goal:** Hyperparameter tuning

**Packages:**
1. **py-tune**
   - tune_grid
   - tune_bayes (optional, or use optuna)
   - fit_resamples
   - collect_metrics
   - select_best

2. **py-dials**
   - Parameter definitions
   - Grid creation

**Deliverable:** Can tune workflows with grid/random search

**Example Usage:**
```python
from py_tune import tune_grid, show_best, select_best
from py_dials import grid_regular, mtry, trees

# Mark parameters for tuning
model = (
    rand_forest(mtry=tune(), trees=tune())
    .set_engine("sklearn")
)

wf = workflow().add_recipe(rec).add_model(model)

# Create parameter grid
param_grid = grid_regular(
    mtry(range=(5, 20)),
    trees(range=(50, 500)),
    levels=5
)

# Tune
tune_results = tune_grid(
    wf,
    resamples=cv_splits,
    grid=param_grid,
    metrics=metric_set(rmse, mae)
)

# Select best
best = select_best(tune_results, metric="rmse")
final_wf = finalize_workflow(wf, best)
```

---

### Phase 4: Time Series (Months 12-15)
**Goal:** Full time series functionality

**Packages:**
1. **py-modeltime**
   - arima_reg, prophet_reg, exp_smoothing
   - modeltime_table
   - modeltime_calibrate
   - modeltime_forecast
   - modeltime_accuracy
   - recursive() wrapper

2. **py-timetk**
   - Lag/rolling operations
   - Date feature extraction
   - Fourier features
   - Visualization

**Deliverable:** Complete time series workflow with forecasting

**Example Usage:**
```python
from py_modeltime import (
    arima_reg, prophet_reg,
    modeltime_table, modeltime_calibrate,
    modeltime_forecast, plot_modeltime_forecast
)

# Create models
model_arima = arima_reg().set_engine("statsmodels")
model_prophet = prophet_reg().set_engine("prophet")
model_ml = rand_forest().set_engine("sklearn")

# Fit
models = [
    model_arima.fit("value ~ date", train),
    model_prophet.fit("value ~ date", train),
    model_ml.fit(wf, train)  # with recipe
]

# Modeltime workflow
models_tbl = modeltime_table(*models)

calibrated = models_tbl.modeltime_calibrate(test)

accuracy = calibrated.modeltime_accuracy()

forecast = calibrated.modeltime_forecast(
    new_data=test,
    actual_data=train,
    h="3 months"
)

plot_modeltime_forecast(forecast)
```

---

### Phase 5: Advanced (Months 16-18)
**Goal:** Advanced features

**Packages:**
1. **py-workflowsets** - Experimentation framework
2. **py-stacks** - Ensembling
3. **py-finetune** - Advanced tuning (optional)

**Deliverable:** Experimentation and ensembling capabilities

---

## Recommendations

### Immediate Actions

1. **Start with py-hardhat**
   - Foundation for all other packages
   - Define core abstractions (mold/forge, blueprint)
   - Establish design patterns

2. **Enhance py-modeltime-resample**
   - Integrate with future workflow system
   - Add more resampling strategies
   - Improve documentation

3. **Prototype py-parsnip**
   - Critical for unified model interface
   - Start with sklearn, statsmodels backends
   - Prove out engine pattern

### Critical Success Factors

1. **API Consistency**
   - Mirror R tidymodels where appropriate
   - Use Pythonic conventions (snake_case, etc.)
   - Provide clear migration guide from R

2. **Integration Quality**
   - Packages must work seamlessly together
   - Consistent data structures (pandas DataFrames)
   - Clear error messages

3. **Documentation**
   - Comprehensive API docs
   - Tutorials for each package
   - End-to-end workflow examples
   - R to Python translation guide

4. **Testing**
   - Unit tests for each component
   - Integration tests for workflows
   - Comparison with R outputs
   - Performance benchmarks

### Long-Term Vision

**Goal:** Create a comprehensive, well-integrated Python ecosystem for statistical modeling and time series forecasting that brings the elegance and consistency of R tidymodels to the Python world.

**Success Metrics:**
- Adoption by data science community
- Contributions from developers
- Integration with broader Python ML ecosystem
- Performance comparable to sklearn
- Comprehensive time series capabilities

**Potential Impact:**
- Unified interface for diverse modeling approaches
- Better reproducibility in time series projects
- Easier experimentation with multiple models
- Smoother transition from traditional stats to ML
- Stronger Python time series forecasting ecosystem

---

## File Paths Reference

All research files are located in:
- Main directory: `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/`
- Reference packages: `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/reference/`
- Existing Python implementation: `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/reference/py-modeltime-resample/`

---

# PART 2: COMPREHENSIVE API REFERENCES & SKFORECAST ANALYSIS

## Complete Tidymodels API Reference

This section provides a comprehensive function-by-function reference for ALL tidymodels packages to serve as a complete implementation guide.

### recipes Package - Complete Function Reference

**Version:** 1.3.1 | **Priority:** HIGH

#### Core Recipe Functions

| Function | Description | Key Parameters | Return Type |
|----------|-------------|----------------|-------------|
| `recipe()` | Create a recipe for data preprocessing | `x` (formula/data), `data`, `formula`, `vars`, `roles` | recipe object |
| `prep()` | Estimate recipe parameters from training data | `x` (recipe), `training`, `fresh`, `verbose`, `retain`, `strings_as_factors` | prepared recipe |
| `bake()` | Apply preprocessing to new data | `object` (prepped recipe), `new_data`, `composition` | tibble |
| `juice()` | Extract transformed training data (superseded) | `object` (prepped recipe), `composition` | tibble |

#### Variable Selection Helpers

| Helper | Selects |
|--------|---------|
| `all_outcomes()` | Variables with "outcome" role |
| `all_predictors()` | Variables with "predictor" role |
| `all_numeric()` | Numeric variables |
| `all_numeric_predictors()` | Numeric predictors |
| `all_nominal()` | Factor/character variables |
| `all_nominal_predictors()` | Nominal predictors |
| `all_date()` | Date/datetime variables |
| `all_date_predictors()` | Date predictors |
| `has_role(role)` | Variables with specific role |
| `has_type(type)` | Variables of specific type |

#### Role Management

| Function | Purpose |
|----------|---------|
| `add_role()` | Add role to variable (non-destructive) |
| `update_role()` | Replace existing role |
| `remove_role()` | Remove specific role from variable |
| `update_role_requirements()` | Modify role behavior |

#### Imputation Steps (Missing Value Handling)

| Step | Method | Key Parameters |
|------|--------|----------------|
| `step_impute_bag()` | Bagged tree imputation | `impute_with`, `trees`, `seed_val` |
| `step_impute_knn()` | K-nearest neighbors | `neighbors`, `impute_with` |
| `step_impute_linear()` | Linear regression | `impute_with` |
| `step_impute_lower()` | Below-minimum substitution | `threshold` |
| `step_impute_mean()` | Mean replacement | None |
| `step_impute_median()` | Median replacement | None |
| `step_impute_mode()` | Mode replacement | None |
| `step_impute_roll()` | Rolling window statistics | `statistic`, `window` |
| `step_unknown()` | Create "unknown" factor level | None |

#### Mathematical Transformations

| Step | Transformation | Parameters |
|------|----------------|------------|
| `step_log()` | Natural logarithm | `base`, `offset`, `signed` |
| `step_sqrt()` | Square root | None |
| `step_inverse()` | Inverse (1/x) | `offset` |
| `step_logit()` | Logit transformation | None |
| `step_invlogit()` | Inverse logit | None |
| `step_BoxCox()` | Box-Cox transformation | `lambdas`, `limits` |
| `step_YeoJohnson()` | Yeo-Johnson transformation | `lambdas`, `limits` |
| `step_hyperbolic()` | Hyperbolic transformations | `func` (sin, cos, tan) |
| `step_relu()` | Rectified linear unit | `reverse`, `smooth` |

#### Basis Functions & Polynomials

| Step | Type | Parameters |
|------|------|------------|
| `step_poly()` | Polynomial features | `degree`, `options` |
| `step_poly_bernstein()` | Bernstein polynomials | `degree` |
| `step_bs()` | B-splines | `deg_free`, `degree` |
| `step_ns()` | Natural splines | `deg_free` |
| `step_harmonic()` | Harmonic/Fourier features | `frequency`, `cycle_size` |

#### Categorical Encoding

| Step | Purpose | Key Parameters |
|------|---------|----------------|
| `step_dummy()` | One-hot encoding | `one_hot`, `naming`, `keep_original_cols` |
| `step_dummy_extract()` | Extract/encode from text | `pattern`, `sep` |
| `step_dummy_multi_choice()` | Multi-label encoding | `sep`, `threshold` |
| `step_bin2factor()` | Binary to factor conversion | `levels` |
| `step_num2factor()` | Numeric to factor | `levels`, `transform`, `ordered` |
| `step_ordinalscore()` | Replace ordinal with score | None |
| `step_factor2string()` | Factor to character | None |
| `step_string2factor()` | Character to factor | `levels`, `ordered` |
| `step_relevel()` | Reorder factor levels | `ref_level` |
| `step_unorder()` | Remove ordering from factor | None |
| `step_other()` | Pool infrequent levels | `threshold`, `other` |
| `step_novel()` | Handle novel factor levels | `new_level` |
| `step_integer()` | Encode as integer | `zero_based` |
| `step_count()` | Count occurrences | `pattern` |
| `step_regex()` | Regular expression indicator | `pattern` |
| `step_percentile()` | Percentile binning | `outside`, `options` |
| `step_indicate_na()` | Missing indicator variables | None |

#### Date/Time Feature Engineering (Critical for Time Series)

| Step | Features Extracted | Parameters |
|------|-------------------|------------|
| `step_date()` | Date components: dow, doy, week, month, decimal, semester, quarter, year | `features`, `abbr`, `label`, `ordinal`, `locale` |
| `step_time()` | Time components: hour, minute, second, am, decimal_day | `features`, `keep_original_cols` |
| `step_holiday()` | Holiday indicators | `holidays` (timeDate list), `locale` |

#### Time Series-Specific Steps

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_lag()` | Create lagged predictors | `lag` (vector of lags), `prefix`, `default` |
| `step_window()` | Rolling window statistics | `size`, `statistic`, `names`, `na_rm` |

#### Normalization & Scaling

| Step | Method | Parameters |
|------|--------|------------|
| `step_center()` | Center to mean 0 | None |
| `step_scale()` | Scale to SD 1 | None |
| `step_normalize()` | Center + scale | None |
| `step_range()` | Scale to custom range | `min`, `max` |

#### Discretization

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_discretize()` | Bin numeric into categories | `num_breaks`, `min_unique` |
| `step_cut()` | Cut into factor bins | `breaks`, `include_outside_range` |

#### Interaction Terms

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_interact()` | Create interaction features | `terms`, `sep` |

#### Dimensionality Reduction

| Step | Method | Key Parameters |
|------|--------|----------------|
| `step_pca()` | Principal component analysis | `num_comp`, `threshold`, `options` |
| `step_kpca()` | Kernel PCA | `num_comp`, `options` |
| `step_kpca_poly()` | Polynomial kernel PCA | `num_comp`, `degree`, `scale_factor`, `offset` |
| `step_kpca_rbf()` | RBF kernel PCA | `num_comp`, `sigma` |
| `step_ica()` | Independent component analysis | `num_comp`, `options` |
| `step_isomap()` | Isometric mapping | `num_terms`, `neighbors`, `options` |
| `step_nnmf()` | Non-negative matrix factorization | `num_comp`, `options` |
| `step_nnmf_sparse()` | Sparse NNMF | `num_comp`, `penalty`, `options` |
| `step_pls()` | Partial least squares | `num_comp`, `outcome` |

#### Distance & Classification-Based

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_classdist()` | Class centroids distance | None |
| `step_classdist_shrunken()` | Shrunken centroids | `threshold`, `log_base` |
| `step_depth()` | Data depth features | `metric` |
| `step_geodist()` | Geographic distance | `lat`, `lon`, `log`, `name` |
| `step_spatialsign()` | Spatial sign transformation | None |

#### Feature Filtering

| Step | Removes | Parameters |
|------|---------|------------|
| `step_zv()` | Zero variance predictors | `freq_cut`, `unique_cut` |
| `step_nzv()` | Near-zero variance predictors | `freq_cut`, `unique_cut` |
| `step_corr()` | Highly correlated features | `threshold`, `use`, `method` |
| `step_lincomb()` | Linear combinations | `max_steps` |
| `step_filter_missing()` | High-missingness variables | `threshold` |
| `step_rm()` | Explicitly remove variables | None |

#### Row Operations

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_arrange()` | Sort rows | `...` (dplyr syntax) |
| `step_filter()` | Filter rows | `...` (dplyr syntax), `skip` |
| `step_slice()` | Select row positions | `...` (dplyr syntax) |
| `step_sample()` | Random row sampling | `size`, `replace` |
| `step_shuffle()` | Randomize row order | None |
| `step_naomit()` | Remove rows with NA | None |

#### Other Transformations

| Step | Purpose | Parameters |
|------|---------|------------|
| `step_mutate()` | Custom mutations (dplyr-style) | `...` (expressions) |
| `step_ratio()` | Create ratio features | `denom` |
| `step_intercept()` | Add intercept column | None |
| `step_profile()` | Profiling datasets | `profile`, `ptype`, `index`, `id`, `sep` |
| `step_rename()` | Rename variables | `...` (new_name = old_name) |
| `step_rename_at()` | Rename with function | `fn` |

#### Check Operations (Validation)

| Check | Validates | Parameters |
|-------|-----------|------------|
| `check_class()` | Variable types | `class_nm`, `class_fns`, `allow_additional` |
| `check_cols()` | Column existence | None |
| `check_missing()` | No missing values | None |
| `check_new_values()` | No new factor levels | `ignore_NA` |
| `check_range()` | Values within range | `min`, `max`, `slack_prop` |

#### Developer & Internal Functions

| Function | Purpose |
|----------|---------|
| `add_step()` | Register custom step |
| `add_check()` | Register custom check |
| `detect_step()` | Find steps in recipe |
| `fully_trained()` | Check if all steps prepped |
| `prepper()` | Wrapper for resampling |
| `tidy()` | Extract step results |
| `update()` | Modify step parameters |
| `names0()` | Generate sequential names |
| `dummy_names()` | Create dummy variable names |

---

### parsnip Package - Complete Function Reference

**Version:** 1.3.3 | **Priority:** CRITICAL

#### Model Specification Functions (30+ Models)

**Tree-Based Models:**

| Function | Model Type | Modes | Key Parameters |
|----------|------------|-------|----------------|
| `decision_tree()` | Single decision tree | regression, classification, censored regression | `cost_complexity`, `tree_depth`, `min_n` |
| `rand_forest()` | Random forest | regression, classification, censored regression | `mtry`, `trees`, `min_n` |
| `boost_tree()` | Gradient boosting | regression, classification, censored regression | `mtry`, `trees`, `min_n`, `tree_depth`, `learn_rate`, `loss_reduction`, `sample_size`, `stop_iter` |
| `bart()` | Bayesian additive regression trees | regression, classification | `trees`, `prior_terminal_node_coef`, `prior_terminal_node_expo`, `prior_outcome_range` |
| `bag_tree()` | Bagged decision trees | regression, classification | `cost_complexity`, `tree_depth`, `min_n`, `class_cost` |

**Linear Models:**

| Function | Model Type | Modes | Key Parameters |
|----------|------------|-------|----------------|
| `linear_reg()` | Linear regression | regression, censored regression | `penalty`, `mixture` |
| `logistic_reg()` | Logistic regression | classification | `penalty`, `mixture` |
| `multinom_reg()` | Multinomial regression | classification | `penalty`, `mixture` |
| `poisson_reg()` | Poisson regression | regression | `penalty`, `mixture` |

**Rule-Based Models:**

| Function | Model Type | Modes | Parameters |
|----------|------------|-------|------------|
| `C5_rules()` | C5.0 rules | classification | `trees`, `min_n` |
| `cubist_rules()` | Cubist rules | regression | `committees`, `neighbors`, `max_rules` |
| `rule_fit()` | RuleFit | regression, classification | `mtry`, `trees`, `min_n`, `tree_depth`, `learn_rate`, `loss_reduction`, `sample_size`, `penalty` |

**Support Vector Machines:**

| Function | Kernel | Modes | Parameters |
|----------|--------|-------|------------|
| `svm_linear()` | Linear | regression, classification | `cost`, `margin` |
| `svm_poly()` | Polynomial | regression, classification | `cost`, `degree`, `scale_factor`, `margin` |
| `svm_rbf()` | Radial basis function | regression, classification | `cost`, `rbf_sigma`, `margin` |

**Neural Networks:**

| Function | Type | Modes | Parameters |
|----------|------|-------|------------|
| `mlp()` | Multi-layer perceptron | regression, classification, censored regression | `hidden_units`, `penalty`, `dropout`, `epochs`, `activation`, `learn_rate` |
| `bag_mlp()` | Bagged neural networks | regression, classification | Same as mlp() plus `class_cost` |

**Other Models:**

| Function | Model Type | Modes | Key Parameters |
|----------|------------|-------|----------------|
| `mars()` | Multivariate adaptive regression splines | regression, classification | `num_terms`, `prod_degree`, `prune_method` |
| `bag_mars()` | Bagged MARS | regression, classification | Same as mars() |
| `nearest_neighbor()` | K-nearest neighbors | regression, classification, censored regression | `neighbors`, `weight_func`, `dist_power` |
| `gen_additive_mod()` | Generalized additive models | regression, classification | `select_features`, `adjust_deg_free` |
| `naive_Bayes()` | Naive Bayes | classification | `smoothness`, `Laplace` |
| `pls()` | Partial least squares | regression, classification | `num_comp`, `predictor_prop` |
| `null_model()` | Null/baseline model | regression, classification, censored regression | None |
| `auto_ml()` | Automated ML (H2O) | regression, classification | `max_runtime_secs`, `max_models`, `seed` |

**Discriminant Analysis:**

| Function | Type | Modes | Parameters |
|----------|------|-------|------------|
| `discrim_linear()` | Linear discriminant | classification | `penalty` |
| `discrim_quad()` | Quadratic discriminant | classification | `regularization_method` |
| `discrim_flexible()` | Flexible discriminant | classification | `num_terms`, `prod_degree`, `prune_method` |
| `discrim_regularized()` | Regularized discriminant | classification | `frac_common_cov`, `frac_identity` |

**Survival/Censored Regression:**

| Function | Type | Modes | Parameters |
|----------|------|-------|------------|
| `survival_reg()` | Parametric survival | censored regression | `dist` |
| `proportional_hazards()` | Cox proportional hazards | censored regression | `penalty`, `mixture` |

#### Model Configuration Functions

| Function | Purpose | Parameters |
|----------|---------|------------|
| `set_engine()` | Specify computational engine | `engine` (e.g., "lm", "glmnet", "ranger", "xgboost"), `...` (engine-specific args) |
| `set_mode()` | Set prediction mode | `mode` ("regression", "classification", "censored regression") |
| `set_args()` | Update main arguments | `...` (named arguments) |
| `update()` | Modify model specification | `...` (parameters to update) |
| `translate()` | Show actual call to engine | `x` (model spec) |
| `show_engines()` | List available engines for model | `x` (model type) |

#### Fitting & Prediction Functions

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `fit()` | Fit model with formula | `object`, `formula`, `data`, `control`, `...` | model_fit object |
| `fit_xy()` | Fit model with x/y | `object`, `x`, `y`, `control`, `...` | model_fit object |
| `predict()` | Generate predictions | `object`, `new_data`, `type`, `...` | tibble |
| `predict_raw()` | Engine-native predictions | `object`, `new_data`, `...` | varies |
| `multi_predict()` | Multiple prediction types | `object`, `new_data`, `...` | nested tibble |
| `augment()` | Add predictions to data | `x`, `new_data`, `...` | tibble |

**Prediction Types:**
- `type = "numeric"` - Numeric predictions (regression)
- `type = "class"` - Class predictions (classification)
- `type = "prob"` - Class probabilities
- `type = "conf_int"` - Confidence intervals
- `type = "pred_int"` - Prediction intervals
- `type = "quantile"` - Quantile predictions
- `type = "time"` - Survival time
- `type = "survival"` - Survival probability
- `type = "hazard"` - Hazard
- `type = "raw"` - Engine-specific

#### Extraction & Inspection Functions

| Function | Extracts | Returns |
|----------|----------|---------|
| `extract_spec_parsnip()` | Model specification | parsnip model spec |
| `extract_fit_engine()` | Underlying fitted model | engine-specific object |
| `extract_fit_time()` | Training duration | named vector |
| `extract_parameter_set_dials()` | All tunable parameters | dials parameter set |
| `extract_parameter_dials()` | Single parameter | dials parameter |
| `tidy()` | Model coefficients/results | tibble |
| `glance()` | Model-level statistics | tibble |
| `autoplot()` | Visualize model | ggplot |

#### Utility Functions

| Function | Purpose |
|----------|---------|
| `add_rowindex()` | Add row index to data |
| `case_weights()` | Create case weights object |
| `case_weights_allowed()` | Check if engine supports weights |
| `control_parsnip()` | Control fitting process |
| `repair_call()` | Fix model call object |
| `required_pkgs()` | List required packages |
| `sparse_data` | Indicator for sparse matrix support |

#### Developer Tools

**Model Registration:**
- `set_new_model()` - Register new model type
- `set_model_mode()` - Define allowed modes
- `set_model_engine()` - Register engine
- `set_model_arg()` - Define arguments
- `set_dependency()` - Specify package dependencies

**Data Utilities:**
- `maybe_matrix()`, `maybe_data_frame()` - Convert data formats
- `min_cols()`, `min_rows()` - Check data dimensions
- `max_mtry_formula()` - Calculate max mtry from formula

---

### workflows Package - Complete Function Reference

**Version:** 1.1.4 | **Priority:** CRITICAL

#### Core Workflow Functions

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `workflow()` | Create empty workflow | None | workflow object |
| `is_trained_workflow()` | Check if fitted | `x` | logical |
| `control_workflow()` | Control fitting process | `control_parsnip`, `control_recipe`, `control_tailor` | control object |

#### Adding Components to Workflows

**Preprocessor Addition:**

| Function | Adds | Parameters |
|----------|------|------------|
| `add_formula()` | Formula preprocessor | `formula`, `blueprint` |
| `add_recipe()` | Recipe preprocessor | `recipe`, `blueprint` |
| `add_variables()` | Variable specification | `outcomes`, `predictors`, `blueprint` |
| `workflow_variables()` | Helper to specify variables | `outcomes`, `predictors` |

**Model Addition:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `add_model()` | Add parsnip model | `spec`, `formula` |

**Postprocessor Addition:**

| Function | Adds | Parameters |
|----------|------|------------|
| `add_tailor()` | Postprocessing step | `tailor` |

**Case Weights:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `add_case_weights()` | Add case weights | `case_weights` |

#### Updating Components

| Function | Updates | Parameters |
|----------|---------|------------|
| `update_formula()` | Replace formula | `formula`, `blueprint` |
| `update_recipe()` | Replace recipe | `recipe`, `blueprint` |
| `update_variables()` | Replace variables | `outcomes`, `predictors`, `blueprint` |
| `update_model()` | Replace model | `spec`, `formula` |
| `update_tailor()` | Replace postprocessor | `tailor` |
| `update_case_weights()` | Replace weights | `case_weights` |

#### Removing Components

| Function | Removes |
|----------|---------|
| `remove_formula()` | Formula preprocessor |
| `remove_recipe()` | Recipe preprocessor |
| `remove_variables()` | Variable specification |
| `remove_model()` | Model specification |
| `remove_tailor()` | Postprocessor |
| `remove_case_weights()` | Case weights |

#### Fitting & Prediction

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `fit()` | Fit entire workflow | `object`, `data`, `control`, `...` | fitted workflow |
| `predict()` | Generate predictions | `object`, `new_data`, `type`, `...` | tibble |
| `augment()` | Add predictions to data | `x`, `new_data`, `...` | tibble |

#### Extraction Functions

| Function | Extracts | Returns |
|----------|----------|---------|
| `extract_spec_parsnip()` | Model specification | parsnip spec |
| `extract_recipe()` | Recipe | recipe object |
| `extract_fit_parsnip()` | Fitted parsnip model | model_fit |
| `extract_fit_engine()` | Underlying engine model | engine object |
| `extract_mold()` | Preprocessed data | mold object |
| `extract_preprocessor()` | Any preprocessor | formula/recipe/variables |
| `extract_postprocessor()` | Tailor postprocessor | tailor object |
| `extract_parameter_set_dials()` | All parameters | dials parameters |
| `extract_parameter_dials()` | Single parameter | dials parameter |
| `extract_fit_time()` | Training time | named vector |

#### Model Inspection

| Function | Purpose | Returns |
|----------|---------|---------|
| `tidy()` | Extract model results | tibble |
| `glance()` | Model-level statistics | tibble |

#### Optimization (Butcher)

Functions to reduce model size for deployment:
- `axe_call()` - Remove call object
- `axe_ctrl()` - Remove control
- `axe_data()` - Remove training data
- `axe_env()` - Remove environments
- `axe_fitted()` - Remove fitted values

---

### tune Package - Complete Function Reference

**Version:** 1.3.0 | **Priority:** HIGH

#### Tuning Many Models

| Function | Strategy | Parameters | Returns |
|----------|----------|------------|---------|
| `tune_grid()` | Grid search | `object`, `resamples`, `grid`, `metrics`, `control`, `...` | tune_results |
| `tune_bayes()` | Bayesian optimization | `object`, `resamples`, `initial`, `iter`, `param_info`, `metrics`, `objective`, `control`, `...` | tune_results |
| `fit_resamples()` | No tuning, just resampling | `object`, `resamples`, `metrics`, `control`, `...` | resample_results |

**Acquisition Functions (Bayesian Optimization):**

| Function | Strategy | Purpose |
|----------|----------|---------|
| `exp_improve()` | Expected improvement | Balances exploration/exploitation |
| `prob_improve()` | Probability of improvement | Focuses on beating current best |
| `conf_bound()` | Confidence bound | Upper/lower confidence bound |

**Grid Control:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `control_grid()` | Control grid search | `verbose`, `allow_par`, `extract`, `save_pred`, `pkgs`, `save_workflow`, `event_level`, `parallel_over` |
| `control_bayes()` | Control Bayesian search | Similar to control_grid() plus `no_improve`, `uncertain`, `seed`, `verbose_iter`, `time_limit` |
| `control_resamples()` | Control fit_resamples | Similar to control_grid() |

#### Fitting Single Final Model

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `fit_best()` | Fit best configuration | `x`, `metric`, `...` | fitted workflow |
| `last_fit()` | Fit and evaluate test set | `object`, `split`, `metrics`, `control`, `...` | last_fit object |
| `control_last_fit()` | Control last_fit | `verbose`, `allow_par`, `extract`, `save_pred`, `pkgs`, `save_workflow`, `event_level` |

#### Finalizing Objects

| Function | Finalizes | Parameters |
|----------|-----------|------------|
| `finalize_model()` | Model with best params | `x`, `parameters` |
| `finalize_recipe()` | Recipe with best params | `x`, `parameters` |
| `finalize_workflow()` | Workflow with best params | `x`, `parameters` |
| `finalize_tailor()` | Tailor with best params | `x`, `parameters` |

#### Inspecting Results

**Collecting Results:**

| Function | Collects | Returns |
|----------|----------|---------|
| `collect_predictions()` | Hold-out predictions | tibble |
| `collect_metrics()` | Performance metrics | tibble |
| `collect_notes()` | Warnings/errors | tibble |
| `collect_extracts()` | Extracted objects | tibble |

**Selecting Best:**

| Function | Selection Criteria | Parameters |
|----------|-------------------|------------|
| `show_best()` | Top N configurations | `x`, `metric`, `n`, `...` |
| `select_best()` | Single best | `x`, `metric`, `...` |
| `select_by_pct_loss()` | Best within % of optimal | `x`, `metric`, `limit`, `...` |
| `select_by_one_std_err()` | Simplest within 1 SE | `x`, `metric`, `...` |

**Other Inspection:**

| Function | Purpose |
|----------|---------|
| `show_notes()` | Display warnings/errors |
| `filter_parameters()` | Remove parameter results |
| `autoplot()` | Visualize tuning results |
| `coord_obs_pred()` | Coord for obs vs pred plots |
| `conf_mat_resampled()` | Average confusion matrix |

#### Metrics & Utilities

| Function | Purpose |
|----------|---------|
| `compute_metrics()` | Calculate performance metrics |
| `augment()` | Add predictions to data |
| `int_pctl()` | Bootstrap confidence intervals |

#### Developer Functions

| Function | Purpose |
|----------|---------|
| `merge()` | Merge parameter values into objects |
| `message_wrap()` | Format messages |
| `.use_case_weights_with_yardstick()` | Determine weight handling |
| `.stash_last_result()` | Save results for debugging |
| `extract_*()` | Various extraction functions |

---

### rsample Package - Complete Function Reference

**Version:** 1.2.1 | **Priority:** CRITICAL

#### Initial Splits (Train/Test)

| Function | Type | Parameters | Returns |
|----------|------|------------|---------|
| `initial_split()` | Random split | `data`, `prop`, `strata`, `breaks`, `pool` | rsplit |
| `initial_time_split()` | Chronological split | `data`, `prop`, `lag` | rsplit |
| `group_initial_split()` | Group-aware split | `data`, `group`, `prop`, `strata`, `pool` | rsplit |

**Validation Splits:**

| Function | Creates | Parameters |
|----------|---------|------------|
| `initial_validation_split()` | Train/val/test | `data`, `prop`, `strata`, `breaks`, `pool` |
| `initial_validation_time_split()` | Chronological 3-way | `data`, `prop` |
| `group_initial_validation_split()` | Group-aware 3-way | `data`, `group`, `prop`, `strata`, `pool` |
| `validation_set()` | Manual validation | `split` |

**Extracting Split Data:**

| Function | Extracts |
|----------|----------|
| `training()` | Training set |
| `testing()` | Test set |
| `validation()` | Validation set |
| `analysis()` | Analysis set (CV training) |
| `assessment()` | Assessment set (CV holdout) |

#### Cross-Validation

**V-Fold CV:**

| Function | Type | Parameters |
|----------|------|------------|
| `vfold_cv()` | Standard V-fold | `data`, `v`, `repeats`, `strata`, `breaks`, `pool` |
| `group_vfold_cv()` | Group-aware V-fold | `data`, `group`, `v`, `balance`, `strata`, `pool` |

**Monte Carlo CV:**

| Function | Type | Parameters |
|----------|------|------------|
| `mc_cv()` | Random subsampling | `data`, `prop`, `times`, `strata`, `breaks`, `pool` |
| `group_mc_cv()` | Group-aware MC | `data`, `group`, `prop`, `times`, `balance`, `strata`, `pool` |

**Other CV Methods:**

| Function | Type | Parameters |
|----------|------|------------|
| `loo_cv()` | Leave-one-out | `data` |
| `clustering_cv()` | Cluster-based | `data`, `vars`, `v`, `repeats`, `distance_function`, `cluster_function` |
| `nested_cv()` | Nested resampling | `data`, `outside`, `inside` |

#### Bootstrap

| Function | Type | Parameters |
|----------|------|------------|
| `bootstraps()` | Bootstrap resampling | `data`, `times`, `strata`, `breaks`, `pool`, `apparent` |
| `group_bootstraps()` | Group-aware bootstrap | `data`, `group`, `times`, `balance`, `strata`, `pool` |

#### Time Series Resampling

**Modern Time Series CV (Recommended):**

| Function | Strategy | Key Parameters |
|----------|----------|----------------|
| `sliding_window()` | Fixed-size sliding window | `data`, `lookback`, `assess_start`, `assess_stop`, `step`, `complete` |
| `sliding_index()` | Index-based sliding | `data`, `index`, `lookback`, `assess_start`, `assess_stop`, `step`, `complete` |
| `sliding_period()` | Period-based sliding | `data`, `index`, `period`, `lookback`, `assess_start`, `assess_stop`, `step`, `skip`, `complete` |

**Legacy:**

| Function | Status | Purpose |
|----------|--------|---------|
| `rolling_origin()` | Superseded | Rolling forecast origin (use sliding_* instead) |

#### Manual & Specialized

| Function | Purpose | Parameters |
|----------|---------|------------|
| `manual_rset()` | Create custom rset | `splits`, `ids` |
| `permutations()` | Permutation sampling | `data`, `permute`, `times` |
| `apparent()` | Apparent error rate | `data` |

#### Analysis Functions

| Function | Purpose | Parameters |
|----------|---------|------------|
| `int_pctl()` | Bootstrap percentile intervals | `object`, `statistics`, `alpha` |
| `int_t()` | Bootstrap t intervals | `object`, `statistics`, `alpha` |
| `int_bca()` | BCa intervals | `object`, `statistics`, `alpha`, `.fn` |
| `reg_intervals()` | Parametric intervals | `object`, `...` |

#### Utility Functions

| Function | Purpose |
|----------|---------|
| `add_resample_id()` | Add resample IDs to data |
| `complement()` | Determine assessment samples |
| `form_pred()` | Extract predictor names from formula |
| `get_rsplit()` | Retrieve specific rsplit |
| `labels()` | Extract object labels |
| `make_splits()` | Constructor for splits |
| `make_strata()` | Create stratification variable |
| `populate()` | Populate rset |
| `reshuffle_rset()` | Regenerate resamples |
| `reverse_splits()` | Swap analysis/assessment |
| `rsample2caret()` | Convert to caret format |
| `caret2rsample()` | Convert from caret |
| `rset_reconstruct()` | Extend rset with subclass |
| `tidy()` | Tidy resampling object |
| `.get_fingerprint()` | Get resample identifier |

---

### yardstick Package - Complete Function Reference

**Version:** 1.3.2 | **Priority:** HIGH

#### Regression Metrics

**Error Metrics:**

| Metric | Full Name | Formula | Range |
|--------|-----------|---------|-------|
| `rmse()` | Root mean squared error | sqrt(mean((y - ŷ)²)) | [0, ∞) |
| `mae()` | Mean absolute error | mean(\|y - ŷ\|) | [0, ∞) |
| `mape()` | Mean absolute percentage error | mean(\|y - ŷ\| / \|y\|) × 100 | [0, ∞) |
| `smape()` | Symmetric MAPE | mean(2\|y - ŷ\| / (\|y\| + \|ŷ\|)) × 100 | [0, 200] |
| `mase()` | Mean absolute scaled error | MAE / MAE_naive | [0, ∞) |

**Goodness of Fit:**

| Metric | Full Name | Range |
|--------|-----------|-------|
| `rsq()` | R-squared | (-∞, 1] |
| `rsq_trad()` | Traditional R-squared | (-∞, 1] |
| `ccc()` | Concordance correlation coefficient | [-1, 1] |

**Performance Ratios:**

| Metric | Full Name |
|--------|-----------|
| `rpd()` | Ratio of performance to deviation |
| `rpiq()` | Ratio of performance to IQR |
| `iic()` | Index of ideality of correlation |

**Robust Metrics:**

| Metric | Purpose |
|--------|---------|
| `huber_loss()` | Robust loss (less sensitive to outliers) |
| `huber_loss_pseudo()` | Pseudo-Huber loss |

**Specialized:**

| Metric | Use Case |
|--------|----------|
| `poisson_log_loss()` | Count data/Poisson models |

#### Classification Metrics

**Binary Classification:**

| Metric | Full Name | Range |
|--------|-----------|-------|
| `accuracy()` | Overall accuracy | [0, 1] |
| `sens()` / `sensitivity()` | Sensitivity/Recall/TPR | [0, 1] |
| `spec()` / `specificity()` | Specificity/TNR | [0, 1] |
| `precision()` / `ppv()` | Positive predictive value | [0, 1] |
| `npv()` | Negative predictive value | [0, 1] |
| `f_meas()` | F1 score | [0, 1] |
| `mcc()` | Matthews correlation coefficient | [-1, 1] |
| `kap()` | Cohen's kappa | [-1, 1] |
| `j_index()` | Youden's J statistic | [-1, 1] |
| `bal_accuracy()` | Balanced accuracy | [0, 1] |

**Prevalence:**

| Metric | Measures |
|--------|----------|
| `detection_prevalence()` | Predicted positive rate |

**Probability-Based:**

| Metric | Purpose | Range |
|--------|---------|-------|
| `roc_auc()` | Area under ROC curve | [0, 1] |
| `pr_auc()` | Area under PR curve | [0, 1] |
| `average_precision()` | Average precision | [0, 1] |
| `gain_capture()` | Gain curve metric | [0, 1] |
| `mn_log_loss()` | Multinomial log loss | [0, ∞) |
| `brier_class()` | Brier score | [0, 1] |
| `classification_cost()` | Cost-weighted metric | varies |

#### Fairness Metrics

| Metric | Measures |
|--------|----------|
| `demographic_parity()` | Equal selection rates across groups |
| `equalized_odds()` | Equal TPR and FPR across groups |
| `equal_opportunity()` | Equal TPR across groups |

#### Survival Analysis Metrics

**Dynamic (Time-Dependent):**

| Metric | Purpose |
|--------|---------|
| `brier_survival()` | Time-dependent Brier score |
| `brier_survival_integrated()` | Integrated Brier score |
| `roc_auc_survival()` | Time-dependent ROC AUC |

**Static:**

| Metric | Purpose |
|--------|---------|
| `concordance_survival()` | Harrell's C-index |

#### Utility Functions

**Metric Sets:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `metric_set()` | Combine multiple metrics | `...` (metric functions) |
| `metrics()` | General metric computation | `data`, `truth`, `estimate`, `...` |

**Confusion Matrix:**

| Function | Purpose |
|----------|---------|
| `conf_mat()` | Create confusion matrix |
| `summary()` | Calculate all CM-based metrics |
| `autoplot()` | Visualize confusion matrix |

**Curve Data:**

| Function | Generates |
|----------|-----------|
| `roc_curve()` | ROC curve data |
| `pr_curve()` | Precision-recall curve |
| `gain_curve()` | Gain curve |
| `lift_curve()` | Lift curve |

**Vector Versions:**

All metrics have `*_vec()` versions that work on vectors instead of data frames (e.g., `rmse_vec()`, `accuracy_vec()`).

---

### modeltime Package - Complete Function Reference

**Version:** 1.3.1 | **Priority:** CRITICAL

#### Core Modeltime Workflow Functions

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `modeltime_table()` | Create model table | `...` (fitted models) | modeltime_table |
| `as_modeltime_table()` | Convert to modeltime table | `...` | modeltime_table |
| `modeltime_calibrate()` | Calibrate on holdout | `object`, `new_data`, `id`, `quiet` | calibrated table |
| `modeltime_forecast()` | Generate forecasts | `object`, `new_data`, `h`, `actual_data`, `conf_interval`, `keep_data`, `...` | tibble |
| `modeltime_accuracy()` | Calculate metrics | `object`, `metric_set`, `acc_by_id` | tibble |
| `modeltime_refit()` | Refit on new data | `object`, `data`, `control`, `refit_modeltime_model` | refitted table |

#### Forecasting Algorithms

**Classical Time Series:**

| Function | Model | Engine Options |
|----------|-------|----------------|
| `arima_reg()` | ARIMA | auto_arima, arima |
| `arima_boost()` | ARIMA + XGBoost | auto_arima_xgboost, arima_xgboost |
| `exp_smoothing()` | Exponential smoothing | ets, smooth_es, croston, theta |
| `seasonal_reg()` | Seasonal decomposition | stlm_ets, stlm_arima, tbats |
| `prophet_reg()` | Facebook Prophet | prophet |
| `prophet_boost()` | Prophet + XGBoost | prophet_xgboost |
| `nnetar_reg()` | Neural network AR | nnetar |
| `naive_reg()` | Naive forecasts | naive, snaive |

**Advanced:**

| Function | Model | Description |
|----------|-------|-------------|
| `adam_reg()` | ADAM | State-space models |
| `temporal_hierarchy()` | THIEF | Temporal hierarchies |

**Baseline:**

| Function | Purpose |
|----------|---------|
| `window_reg()` | Window-based regression baseline |

#### Recursive & Panel Functions

| Function | Purpose | Parameters |
|----------|---------|------------|
| `recursive()` | Convert ML to recursive forecaster | `object`, `transform`, `train_tail`, `id` |
| `panel_tail()` | Extract last N rows per group | `data`, `id`, `date`, `n` |

#### Visualization

| Function | Creates | Parameters |
|----------|---------|------------|
| `plot_modeltime_forecast()` | Interactive forecast plot | `object`, `...`, `.conf_interval_show`, `.plotly_slider`, `.facet_ncol`, `.facet_nrow`, `.facet_scales`, `.title`, `.interactive` |
| `plot_modeltime_residuals()` | Interactive residuals plot | `object`, `...`, `.type`, `.smooth`, `.legend_show`, `.title`, `.interactive` |
| `table_modeltime_accuracy()` | Interactive accuracy table | `object`, `...`, `.bordered`, `.resizable` |

#### Residual Analysis

| Function | Purpose |
|----------|---------|
| `modeltime_residuals()` | Extract residuals |
| `modeltime_residuals_test()` | Statistical tests on residuals |

#### Modeltime Table Utilities

| Function | Purpose |
|----------|---------|
| `combine_modeltime_tables()` | Merge multiple tables |
| `add_modeltime_model()` | Add model to table |
| `drop_modeltime_model()` | Remove model from table |
| `update_modeltime_model()` | Modify model in table |
| `pluck_modeltime_model()` | Extract model |
| `pull_modeltime_model()` | Extract model (deprecated) |

#### Nested Forecasting (Multiple Series)

**Main Functions:**

| Function | Purpose |
|----------|---------|
| `modeltime_nested_fit()` | Fit workflows to nested data |
| `modeltime_nested_select_best()` | Select best models per group |
| `modeltime_nested_refit()` | Refit nested models |
| `modeltime_nested_forecast()` | Generate nested forecasts |

**Extractors:**

| Function | Extracts |
|----------|----------|
| `extract_nested_best_model_report()` | Best model summary |
| `extract_nested_error_report()` | Fitting errors |
| `extract_nested_test_accuracy()` | Accuracy by group |
| `extract_nested_test_forecast()` | Test forecasts |
| `extract_nested_future_forecast()` | Future forecasts |
| `extract_nested_modeltime_table()` | Modeltime tables |

#### Workflowsets Integration

| Function | Purpose |
|----------|---------|
| `modeltime_fit_workflowset()` | Fit workflowset to time series |
| `control_fit_workflowset()` | Control fitting process |

#### Metrics

| Function | Purpose |
|----------|---------|
| `default_forecast_accuracy_metric_set()` | Standard metric set |
| `extended_forecast_accuracy_metric_set()` | Comprehensive metrics |
| `maape()` | Mean arctangent absolute percentage error |
| `summarize_accuracy_metrics()` | Aggregate metrics |

#### Parallel Processing

| Function | Purpose |
|----------|---------|
| `parallel_start()` | Start parallel cluster |
| `parallel_stop()` | Stop parallel cluster |

#### Control Functions

| Function | Controls |
|----------|----------|
| `control_refit()` | Refitting process |
| `control_nested_fit()` | Nested fitting |
| `create_model_grid()` | Parameter grid for models |

#### Developer Tools

| Function | Purpose |
|----------|---------|
| `new_modeltime_bridge()` | Create custom modeltime models |
| `create_xreg_recipe()` | Create exogenous variable recipe |
| `juice_xreg_recipe()` | Extract training data |
| `bake_xreg_recipe()` | Apply recipe to new data |
| `parse_index_from_data()` | Extract date/time index |

---

### timetk Package - Complete Function Reference

**Version:** 2.9.0 | **Priority:** HIGH

#### Plotting & Visualization

| Function | Creates | Key Parameters |
|----------|---------|----------------|
| `plot_time_series()` | Interactive time series plot | `.data`, `.date_var`, `.value`, `.color_var`, `.facet_vars`, `.smooth`, `.interactive` |
| `plot_time_series_boxplot()` | Boxplots by period | Similar plus `.period` |
| `plot_time_series_regression()` | Regression visualization | `.formula`, `.show_summary` |
| `plot_acf_diagnostics()` | ACF/PACF/CCF plots | `.date_var`, `.value`, `.lags` |
| `plot_anomaly_diagnostics()` | Anomaly detection plot | `.date_var`, `.value`, `.alpha`, `.max_anomalies` |
| `plot_seasonal_diagnostics()` | Seasonal patterns | `.date_var`, `.value`, `.feature_set` |
| `plot_stl_diagnostics()` | STL decomposition plot | `.date_var`, `.value`, `.frequency`, `.trend` |
| `plot_time_series_cv_plan()` | Resample visualization | `.data`, `.date_var`, `.value` |

#### Data Wrangling

**Time-Based Operations:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `summarise_by_time()` | Aggregate by time period | `.data`, `.date_var`, `.by`, `...` (summary expressions) |
| `mutate_by_time()` | Transform by time groups | Similar to summarise |
| `pad_by_time()` | Insert missing timestamps | `.data`, `.date_var`, `.by`, `.pad_value` |
| `filter_by_time()` | Filter by date range | `.data`, `.date_var`, `.start_date`, `.end_date` |
| `filter_period()` | Filter within periods | `.data`, `.date_var`, `.period`, `...` (expressions) |
| `slice_period()` | Slice rows within periods | `.data`, `.date_var`, `.period`, `...` (indices) |
| `condense_period()` | Reduce to lower frequency | `.data`, `.date_var`, `.period` |
| `future_frame()` | Create future timestamps | `.data`, `.date_var`, `.length_out`, `.bind_data` |

**Anomaly Detection:**

| Function | Purpose |
|----------|---------|
| `anomalize()` | Detect anomalies | `.data`, `.date_var`, `.value`, `.iqr_alpha`, `.clean_alpha`, `.max_anomalies` |

#### Feature Engineering & Augmentation

**Time Series Signature:**

| Function | Adds Features | Description |
|----------|---------------|-------------|
| `tk_augment_timeseries_signature()` | 29+ date features | year, half, quarter, month, day, hour, minute, second, wday, mday, qday, yday, etc. |
| `tk_get_timeseries_signature()` | Extract signature | Returns tibble of features |

**Holiday Features:**

| Function | Purpose |
|----------|---------|
| `tk_augment_holiday_signature()` | Holiday indicators |
| `tk_get_holiday_signature()` | Extract holidays |
| `tk_make_holiday_sequence()` | Generate holiday dates |
| `tk_make_weekend_sequence()` | Generate weekends |
| `tk_make_weekday_sequence()` | Generate weekdays |
| `tk_get_holidays_by_year()` | List holidays |

**Lags & Leads:**

| Function | Creates |
|----------|---------|
| `tk_augment_lags()` | Lagged features |
| `tk_augment_leads()` | Lead features |
| `lag_vec()` | Vectorized lag |
| `lead_vec()` | Vectorized lead |

**Differences:**

| Function | Purpose |
|----------|---------|
| `tk_augment_differences()` | Differenced features |
| `diff_vec()` | Vectorized difference |
| `diff_inv_vec()` | Inverse difference |

**Rolling Windows:**

| Function | Purpose | Parameters |
|----------|---------|------------|
| `tk_augment_slidify()` | Rolling window features | `.data`, `.value`, `.period`, `.f`, `.align`, `.partial` |
| `slidify()` | Create rolling function | `.f`, `.period`, `.align`, `.partial` |
| `slidify_vec()` | Vectorized rolling | `.x`, `.period`, `.f`, `.align`, `.partial` |

**Fourier Features:**

| Function | Creates |
|----------|---------|
| `tk_augment_fourier()` | Sine/cosine features |
| `fourier_vec()` | Vectorized Fourier |

#### Vectorized Transformations

**Normalization:**

| Function | Transform |
|----------|-----------|
| `normalize_vec()` | Min-max normalization |
| `normalize_inv_vec()` | Inverse normalize |
| `standardize_vec()` | Z-score standardization |
| `standardize_inv_vec()` | Inverse standardize |

**Box-Cox:**

| Function | Purpose |
|----------|---------|
| `box_cox_vec()` | Box-Cox transform |
| `box_cox_inv_vec()` | Inverse Box-Cox |
| `auto_lambda()` | Optimize lambda |

**Log Interval:**

| Function | Purpose |
|----------|---------|
| `log_interval_vec()` | Log transform with offset |
| `log_interval_inv_vec()` | Inverse log interval |

**Smoothing & Cleaning:**

| Function | Purpose |
|----------|---------|
| `smooth_vec()` | LOESS smoothing |
| `ts_clean_vec()` | Replace outliers |
| `ts_impute_vec()` | Impute missing values |

#### Recipe Steps (tidymodels Integration)

**Time Series Steps:**

| Step | Purpose |
|------|---------|
| `step_timeseries_signature()` | Add 29+ time features |
| `step_holiday_signature()` | Add holiday features |
| `step_fourier()` | Add Fourier terms |
| `step_diff()` | Add differences |
| `step_smooth()` | Smooth values |
| `step_slidify()` | Rolling window statistics |
| `step_slidify_augment()` | Augmented rolling features |

**Transformation Steps:**

| Step | Transform |
|------|-----------|
| `step_box_cox()` | Box-Cox transformation |
| `step_log_interval()` | Log interval transformation |

**Data Preparation:**

| Step | Purpose |
|------|---------|
| `step_ts_pad()` | Pad time series |
| `step_ts_impute()` | Impute missing |
| `step_ts_clean()` | Clean outliers |

#### Cross-Validation

| Function | Purpose | Parameters |
|----------|---------|------------|
| `time_series_split()` | Single train/test split | `data`, `date_var`, `initial`, `assess`, `lag`, `cumulative` |
| `time_series_cv()` | Rolling/expanding CV | `data`, `date_var`, `initial`, `assess`, `skip`, `lag`, `cumulative`, `slice_limit` |
| `tk_time_series_cv_plan()` | CV plan visualization data | `...` (rset object) |

#### Diagnostic & Summary

| Function | Generates |
|----------|-----------|
| `tk_summary_diagnostics()` | Summary statistics by group |
| `tk_anomaly_diagnostics()` | Anomaly detection results |
| `tk_acf_diagnostics()` | ACF/PACF/CCF data |
| `tk_seasonal_diagnostics()` | Seasonal feature analysis |
| `tk_stl_diagnostics()` | STL decomposition |
| `tk_tsfeatures()` | Time series feature matrix |

#### Index Operations

| Function | Purpose |
|----------|---------|
| `tk_index()` | Extract date/time index |
| `has_timetk_idx()` | Check for timetk index |
| `tk_get_frequency()` | Determine frequency |
| `tk_get_trend()` | Extract trend component |
| `tk_get_timeseries_unit_frequency()` | Unit frequency |
| `tk_get_timeseries_variables()` | Find time series columns |

#### Time Series Creation

| Function | Creates |
|----------|---------|
| `tk_make_timeseries()` | Intelligent sequence |
| `tk_make_future_timeseries()` | Future dates |

#### Type Conversion

**To Tibble:**

| Function | From |
|----------|------|
| `tk_tbl()` | ts, xts, zoo, zooreg objects |

**To Time Series:**

| Function | To |
|----------|---|
| `tk_ts()` / `tk_ts_()` | ts object |
| `tk_xts()` / `tk_xts_()` | xts object |
| `tk_zoo()` / `tk_zoo_()` | zoo object |
| `tk_zooreg()` / `tk_zooreg_()` | zooreg object |

#### Utilities

| Function | Purpose |
|----------|---------|
| `between_time()` | Check if in date range |
| `add_time()` / `%+time%` | Add time duration |
| `subtract_time()` / `%-time%` | Subtract duration |
| `parse_date2()` | Flexible date parsing |
| `parse_datetime2()` | Flexible datetime parsing |
| `is_date_class()` | Check if date/datetime |
| `set_tk_time_scale_template()` | Set time scale template |
| `get_tk_time_scale_template()` | Get time scale template |

---

### dials Package - Complete Function Reference

**Version:** Latest | **Priority:** MEDIUM

#### Parameter Sets

| Function | Purpose |
|----------|---------|
| `parameters()` | Create parameter set |
| `update()` | Update parameter in set |

**Range Functions:**

| Function | Purpose |
|----------|---------|
| `range_get()` | Get parameter range |
| `range_set()` | Set parameter range |
| `range_validate()` | Validate range |

**Value Functions:**

| Function | Purpose |
|----------|---------|
| `value_set()` | Set parameter value |
| `value_seq()` | Generate sequence |
| `value_sample()` | Random sample |
| `value_transform()` | Transform value |
| `value_inverse()` | Inverse transform |
| `value_validate()` | Validate value |

**Limit Functions:**

| Function | Gets |
|----------|------|
| `lower_limit()` | Lower bound |
| `upper_limit()` | Upper bound |

#### Grid Creation

| Function | Strategy | Parameters |
|----------|----------|------------|
| `grid_regular()` | Regular factorial grid | `...` (parameters), `levels`, `filter` |
| `grid_random()` | Random grid | `...` (parameters), `size`, `filter` |
| `grid_space_filling()` | Space-filling design | `...` (parameters), `size`, `type`, `variogram_range` |

#### Preprocessing Parameters

**Data Transformation:**
- `trim_amount()` - Trimming percentage
- `num_breaks()` - Number of bins
- `min_unique()` - Minimum unique values

**Dimension Reduction:**
- `num_comp()` - Number of components (PCA, PLS)
- `num_terms()` - Number of terms
- `predictor_prop()` - Proportion of predictors

**Nearest Neighbors:**
- `all_neighbors()` - All neighbors parameter
- `neighbors()` - Number of neighbors

**Text/Hashing:**
- `num_tokens()` - Number of tokens
- `num_hash()` - Hash size
- `signed_hash()` - Signed hash indicator
- `token()` - Token type
- `vocabulary_size()` - Vocabulary size
- `max_tokens()`, `min_tokens()` - Token limits
- `max_times()`, `min_times()` - Time limits

**Sampling:**
- `over_ratio()`, `under_ratio()` - Sampling ratios
- `validation_set_prop()` - Validation proportion

**UMAP:**
- `initial_umap()` - UMAP initialization
- `min_dist()` - Minimum distance

**Near-Zero Variance:**
- `freq_cut()` - Frequency cut ratio
- `unique_cut()` - Unique value cutoff

**Other:**
- `num_runs()` - Number of runs
- `weight()` - Weight parameter
- `weight_scheme()` - Weighting scheme
- `window_size()` - Window size
- `harmonic_frequency()` - Harmonic frequency

**Bayesian:**
- `prior_slab_dispersion()` - Slab dispersion
- `prior_mixture_threshold()` - Mixture threshold

**Formula:**
- `prop_terms()` - Proportion of terms

#### Modeling Parameters

**Tree Parameters:**
- `trees()` - Number of trees
- `min_n()` - Minimum node size
- `tree_depth()` - Maximum tree depth
- `mtry()` - Number of predictors sampled
- `mtry_long()` - Mtry for many predictors
- `mtry_prop()` - Mtry as proportion
- `sample_size()` - Sample size
- `sample_prop()` - Sample proportion
- `cost_complexity()` - Cost-complexity parameter
- `loss_reduction()` - Loss reduction
- `prune()` - Pruning indicator
- `prune_method()` - Pruning method

**Regularization:**
- `penalty()` - Regularization penalty (lambda)
- `mixture()` - Mixing proportion (alpha)

**Neural Networks:**
- `hidden_units()` - Hidden layer size
- `hidden_units_2()` - Second hidden layer
- `activation()` - Activation function
- `activation_2()` - Second activation
- `dropout()` - Dropout rate
- `epochs()` - Training epochs
- `batch_size()` - Batch size
- `learn_rate()` - Learning rate
- `momentum()` - Momentum

**Learning Rate Schedule:**
- `rate_initial()` - Initial rate
- `rate_largest()` - Largest rate
- `rate_reduction()` - Reduction factor
- `rate_steps()` - Steps for schedule
- `rate_step_size()` - Step size
- `rate_decay()` - Decay rate
- `rate_schedule()` - Schedule type

**SVM:**
- `cost()` - Cost parameter
- `svm_margin()` - Margin
- `rbf_sigma()` - RBF kernel sigma
- `scale_factor()` - Scale factor
- `kernel_offset()` - Kernel offset

**Polynomial:**
- `degree()` - Polynomial degree
- `degree_int()` - Integer degree
- `prod_degree()` - Product degree
- `spline_degree()` - Spline degree
- `num_knots()` - Number of knots

**Distance:**
- `dist_power()` - Distance power (Minkowski)

**Clustering:**
- `num_clusters()` - Number of clusters

**Degrees of Freedom:**
- `deg_free()` - Degrees of freedom
- `adjust_deg_free()` - Adjusted df

**Bayes:**
- `Laplace()` - Laplace correction
- `smoothness()` - Smoothness parameter

**Feature Selection:**
- `select_features()` - Feature selection indicator

**Survival:**
- `surv_dist()` - Survival distribution
- `survival_link()` - Survival link function

**BART:**
- Various BART-specific parameters

**Other:**
- `stop_iter()` - Stopping iterations
- `summary_stat()` - Summary statistic
- `threshold()` - Threshold value
- `weight_func()` - Weighting function
- `class_weights()` - Class weights
- `target_weight()` - Target weight
- `regularization_method()` - Regularization method

#### Post-Processing Parameters

- `buffer()` - Buffer for predictions
- `cal_method_class()` - Calibration method (classification)
- `cal_method_reg()` - Calibration method (regression)

#### Developer Tools

| Function | Purpose |
|----------|---------|
| `new_quant_param()` | Create quantitative parameter |
| `new_qual_param()` | Create qualitative parameter |
| `parameters_constr()` | Parameter constructor |
| `unknown()` | Create unknown placeholder |
| `is_unknown()` | Check if unknown |
| `has_unknowns()` | Check for unknowns in set |
| `encode_unit()` | Encode time unit |

---

### workflowsets Package - Complete Function Reference

**Version:** Latest | **Priority:** MEDIUM

#### Core Functions

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `workflow_set()` | Create workflow combinations | `preproc`, `models`, `cross` | workflow_set |
| `workflow_map()` | Apply function to workflows | `object`, `fn`, `...` | workflow_set |

#### Options Management

| Function | Purpose |
|----------|---------|
| `option_add()` | Add options to workflows |
| `option_remove()` | Remove options |
| `option_add_parameters()` | Add parameter specifications |
| `option_list()` | Create option list |

#### Comments/Annotations

| Function | Purpose |
|----------|---------|
| `comment_add()` | Add comments to workflows |
| `comment_get()` | Retrieve comments |
| `comment_reset()` | Remove comments |
| `comment_print()` | Print comments |

#### Workflow Updates

| Function | Updates |
|----------|---------|
| `update_workflow_model()` | Model in specific workflow |
| `update_workflow_recipe()` | Recipe in specific workflow |

#### Results Processing

| Function | Purpose |
|----------|---------|
| `rank_results()` | Rank by metric |
| `autoplot()` | Visualize results |
| `fit_best()` | Fit best workflow |

#### Collection Functions

| Function | Collects |
|----------|----------|
| `collect_metrics()` | Performance metrics |
| `collect_predictions()` | Predictions |
| `collect_notes()` | Warnings/errors |
| `collect_extracts()` | Extracted objects |

#### Extraction

| Function | Extracts |
|----------|----------|
| `extract_workflow_set_result()` | Results from specific workflow |
| `pull_workflow_set_result()` | (Deprecated) |
| `pull_workflow()` | (Deprecated) |

#### Utilities

| Function | Purpose |
|----------|---------|
| `as_workflow_set()` | Convert to workflow_set |
| `leave_var_out_formulas()` | Generate leave-one-out formulas |

---

## Skforecast Package - Comprehensive Analysis

### Package Overview

**Official Site:** https://skforecast.org/0.18.0/
**Language:** Python
**Core Purpose:** Time series forecasting using machine learning models
**License:** BSD-3-Clause
**Design Philosophy:** Wrap scikit-learn regressors to create autoregressive forecasters

### Architecture & Design

#### Core Concept
Skforecast transforms any scikit-learn-compatible regressor into a time series forecaster by:
1. Creating lagged features from the target variable
2. Optionally adding window-based features
3. Including exogenous variables
4. Training the regressor on the transformed data
5. Using recursive or direct strategies for multi-step forecasting

#### Key Design Principles
1. **Scikit-learn Compatibility** - Works with any regressor following sklearn API
2. **Flexible Feature Engineering** - Extensive lag and window feature creation
3. **Multiple Forecasting Strategies** - Recursive, direct, and multivariate approaches
4. **Production-Ready** - Built-in backtesting, hyperparameter tuning, drift detection
5. **Series Flexibility** - Single series, independent multi-series, dependent multivariate

### Complete Skforecast API Reference

#### Forecaster Classes

**1. ForecasterRecursive**

*Purpose:* Single-series recursive multi-step forecasting

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regressor` | sklearn regressor | Required | Base model (any sklearn-compatible regressor) |
| `lags` | int, list, array, range | None | Lag indices to use as predictors (1-indexed) |
| `window_features` | object, list | None | Window feature transformer instances |
| `transformer_y` | sklearn transformer | None | Preprocessing for target variable |
| `transformer_exog` | sklearn transformer | None | Preprocessing for exogenous variables |
| `weight_func` | callable | None | Custom sample weighting function |
| `differentiation` | int | None | Order of differencing (≥1) |
| `fit_kwargs` | dict | None | Additional arguments for regressor.fit() |
| `binner_kwargs` | dict | None | Configuration for QuantileBinner |
| `forecaster_id` | str, int | None | Identifier for the forecaster |

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `regressor` | object | Fitted regressor instance |
| `is_fitted` | bool | Whether forecaster is trained |
| `lags` | ndarray | Array of lag indices used |
| `lags_names` | list | Names of lag features |
| `max_lag` | int | Maximum lag value |
| `window_size` | int | Required historical window size |
| `window_features` | list | Feature transformer objects |
| `last_window_` | ndarray | Final training window for predictions |
| `training_range_` | pandas Index | Training data date range |
| `exog_in_` | bool | Whether exog variables were used |
| `exog_names_in_` | list | Exogenous variable names |
| `in_sample_residuals_` | ndarray | Training residuals (max 10,000) |
| `in_sample_residuals_by_bin_` | dict | Residuals grouped by prediction bins |
| `out_sample_residuals_` | ndarray | Validation residuals |
| `fit_date` | datetime | Timestamp of last training |

**Core Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `fit(y, exog, store_last_window, store_in_sample_residuals, random_state)` | Train forecaster | self |
| `predict(steps, last_window, exog, ...)` | Generate point forecasts | pandas Series/DataFrame |
| `predict_interval(steps, last_window, exog, interval, n_boot, ...)` | Prediction intervals | pandas DataFrame |
| `predict_bootstrapping(steps, last_window, exog, n_boot, ...)` | Bootstrap predictions | pandas DataFrame |
| `predict_quantiles(steps, last_window, exog, quantiles, ...)` | Quantile forecasts | pandas DataFrame |
| `predict_dist(steps, last_window, exog, distribution, ...)` | Probability distributions | pandas DataFrame |
| `create_train_X_y(y, exog)` | Generate training matrices | tuple (X, y) |
| `create_predict_X(steps, last_window, exog, ...)` | Build prediction inputs | pandas DataFrame |
| `create_sample_weights(X_train)` | Compute observation weights | numpy array |
| `set_params(**params)` | Update regressor parameters | self |
| `set_lags(lags)` | Modify lag structure | self |
| `set_window_features(window_features)` | Change feature engineering | self |
| `set_fit_kwargs(fit_kwargs)` | Update fit arguments | self |
| `set_in_sample_residuals(residuals)` | Provide stored residuals | self |
| `set_out_sample_residuals(residuals)` | Set validation residuals | self |
| `get_feature_importances(...)` | Extract feature importance | pandas DataFrame |

**2. ForecasterDirect**

*Purpose:* Direct multi-step forecasting (separate model per horizon)

**Key Differences from Recursive:**

| Aspect | ForecasterDirect |
|--------|------------------|
| Strategy | Trains separate model for each forecasting step |
| Models | `regressors_` dict with one model per step |
| `steps` parameter | Required - specifies max forecast horizon |
| Computational cost | Higher (multiple models) |
| Advantages | No error propagation between steps |

**Additional Parameters:**
- `steps` (int) - Required. Maximum number of future steps to forecast
- `n_jobs` (int, 'auto') - Parallel processing for training multiple models

**3. ForecasterRecursiveMultiSeries**

*Purpose:* Independent multi-series forecasting with global model

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoding` | str, None | 'ordinal' | Series encoding: 'ordinal', 'ordinal_category', 'onehot', None |
| `transformer_series` | object, dict | None | Single transformer or per-series dict |
| `series_weights` | dict | None | Series-level importance weights |
| `differentiation` | int, dict | None | Differencing order (single or per-series) |
| `dropna_from_series` | bool | False | Drop NaN rows from training matrices |

**Key Attributes:**

| Attribute | Description |
|-----------|-------------|
| `series_names_in_` | Input series names |
| `X_train_series_names_in_` | Series included in training |
| `encoding_mapping_` | Series encoding dictionary |
| `transformer_series_` | Per-series transformer dictionary |
| `differentiator_` | Per-series differentiator dictionary |

**Unique Methods:**

| Method | Purpose |
|--------|---------|
| `fit(series, exog, store_last_window, ...)` | Train on multiple series |
| `predict(steps, series, last_window, exog, ...)` | Forecast multiple series |
| `_create_lags(...)` | Internal lag creation |
| `_create_window_features(...)` | Internal window feature creation |

**4. ForecasterDirectMultiVariate**

*Purpose:* Dependent multivariate series forecasting

**Key Characteristics:**
- Models dependencies between multiple series
- Each series can be predicted jointly
- Captures cross-series relationships
- Similar API to ForecasterDirect but for multivariate data

**5. ForecasterSarimax**

*Purpose:* Wrapper for SARIMAX models within skforecast API

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `regressor` | Sarimax | Required. Sarimax model from skforecast |
| `transformer_y` | transformer | Optional target preprocessing |
| `transformer_exog` | transformer | Optional exog preprocessing |
| `fit_kwargs` | dict | Additional fit arguments |
| `forecaster_id` | str, int | Identifier |

**Unique Methods:**

| Method | Purpose |
|--------|---------|
| `get_info_criteria(criteria, method)` | Get AIC, BIC, or HQIC |

**6. ForecasterRNN**

*Purpose:* Recurrent neural networks for forecasting

**Key Features:**
- Deep learning implementation
- Supports LSTM, GRU architectures
- Keras/TensorFlow backend
- Handles sequential dependencies

---

#### Feature Engineering Module

**Window Features:**

Skforecast provides powerful window-based feature creation through custom transformers.

**RollingFeatures Class:**

```python
from skforecast.preprocessing import RollingFeatures

# Available statistics
statistics = [
    'mean', 'std', 'min', 'max', 'sum', 'median',
    'ratio_min_max', 'coef_variation', 'ewm'
]

# Example
window_features = RollingFeatures(
    stats=['mean', 'std', 'max'],
    window_sizes=[7, 14, 30]
)
```

**Custom Window Features:**

Users can create custom window feature transformers by implementing:
- `fit(X, y)` method
- `transform(X)` method
- `fit_transform(X, y)` method

---

#### Preprocessing Module

**TimeSeriesDifferentiator:**

| Method | Purpose |
|--------|---------|
| `fit(X)` | Store initial values |
| `transform(X)` | Apply differencing |
| `inverse_transform(X)` | Reverse differencing |
| `fit_transform(X)` | Fit and transform |

**QuantileBinner:**

| Method | Purpose |
|--------|---------|
| `fit(X)` | Learn quantile bins |
| `transform(X)` | Assign to bins |
| `fit_transform(X)` | Fit and transform |

**ConformalIntervalCalibrator:**

| Method | Purpose |
|--------|---------|
| `fit(residuals)` | Calibrate intervals |
| `predict_interval(...)` | Generate calibrated intervals |

**Reshaping Functions:**

| Function | Purpose |
|----------|---------|
| `reshape_series_wide_to_long()` | Wide DataFrame → long format with MultiIndex |
| `reshape_series_long_to_dict()` | Long DataFrame → dict of Series |
| `reshape_exog_long_to_dict()` | Long exog → dict of DataFrames |

---

#### Model Selection Module

**Backtesting Functions:**

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `backtesting_forecaster()` | Single-series backtesting | `forecaster`, `y`, `cv`, `metric`, `exog`, `interval`, `n_boot`, `n_jobs` |
| `backtesting_forecaster_multiseries()` | Multi-series backtesting | `forecaster`, `series`, `cv`, `metric`, `levels`, `add_aggregated_metric` |
| `backtesting_sarimax()` | SARIMAX backtesting | `forecaster`, `y`, `cv`, `metric` |

**Returns:**
- `metrics_*` DataFrame with performance by fold
- `backtest_predictions` DataFrame with predictions and actuals

**Hyperparameter Tuning Functions:**

| Function | Strategy | Key Parameters |
|----------|----------|----------------|
| `grid_search_forecaster()` | Exhaustive grid search | `forecaster`, `y`, `cv`, `param_grid`, `metric`, `lags_grid`, `return_best` |
| `random_search_forecaster()` | Random parameter sampling | Same plus `param_distributions`, `n_iter`, `random_state` |
| `bayesian_search_forecaster()` | Bayesian optimization (Optuna) | Same plus `search_space`, `n_trials`, `kwargs_create_study`, `kwargs_study_optimize` |
| `grid_search_forecaster_multiseries()` | Grid search multi-series | `forecaster`, `series`, `cv`, `param_grid`, `metric`, `aggregate_metric`, `levels` |
| `random_search_forecaster_multiseries()` | Random search multi-series | Similar parameters |
| `bayesian_search_forecaster_multiseries()` | Bayesian multi-series | Similar parameters |

**Returns:**
- `results` DataFrame with parameters and metrics
- `best_trial` object (Bayesian only)

**Cross-Validation Classes:**

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `TimeSeriesFold` | Time series CV splits | `steps`, `fold_stride`, `gap`, `skip_folds` |
| `OneStepAheadFold` | One-step-ahead CV | Similar parameters |
| `BaseFold` | Base class | - |

**Utility Functions:**

| Function | Purpose |
|----------|---------|
| `initialize_lags_grid()` | Create lag grid for search |
| `select_n_jobs_backtesting()` | Determine optimal parallelism |
| `check_backtesting_input()` | Validate backtesting parameters |

---

#### Metrics Module

| Metric | Function | Description |
|--------|----------|-------------|
| MASE | `mean_absolute_scaled_error()` | Scale-independent error metric |
| RMSSE | `root_mean_squared_scaled_error()` | RMSE-based scaled error |
| SMAPE | `symmetric_mean_absolute_percentage_error()` | Symmetric percentage error |

**Decorator:**
- `add_y_train_argument()` - Adds y_train parameter to functions

Note: Skforecast primarily uses sklearn.metrics for most metrics (MAE, MSE, RMSE, R², etc.)

---

### Skforecast Workflows

#### Basic Recursive Forecasting Workflow

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import TimeSeriesFold

# 1. Create forecaster
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(n_estimators=100),
    lags=12
)

# 2. Train
forecaster.fit(y=y_train)

# 3. Predict
predictions = forecaster.predict(steps=10)

# 4. Backtesting
cv = TimeSeriesFold(
    steps=10,
    fold_stride=10,
    gap=0
)

metrics, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric='mean_absolute_error',
    n_jobs=-1
)
```

#### Hyperparameter Tuning Workflow

```python
from skforecast.model_selection import grid_search_forecaster

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

lags_grid = [7, 14, [1, 2, 7, 14]]

# Grid search
results = grid_search_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    param_grid=param_grid,
    lags_grid=lags_grid,
    metric='mean_absolute_error',
    return_best=True,
    n_jobs=-1
)
```

#### Multi-Series Forecasting Workflow

```python
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

# Create multi-series forecaster
forecaster = ForecasterAutoregMultiSeries(
    regressor=RandomForestRegressor(),
    lags=14,
    encoding='ordinal',
    transformer_series=StandardScaler()
)

# Train on multiple series
forecaster.fit(series=series_df)

# Predict
predictions = forecaster.predict(
    steps=10,
    series=['series_1', 'series_2', 'series_3']
)

# Backtest
metrics, predictions = backtesting_forecaster_multiseries(
    forecaster=forecaster,
    series=series_df,
    cv=cv,
    metric='mean_absolute_error',
    levels=['series_1', 'series_2']
)
```

#### Feature Engineering Workflow

```python
from skforecast.preprocessing import RollingFeatures
from sklearn.preprocessing import StandardScaler

# Create window features
window_features = [
    RollingFeatures(
        stats=['mean', 'std'],
        window_sizes=[7, 14]
    )
]

# Create forecaster with features
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(),
    lags=7,
    window_features=window_features,
    transformer_y=StandardScaler(),
    transformer_exog=StandardScaler()
)

# Fit with exogenous variables
forecaster.fit(
    y=y_train,
    exog=exog_train
)

# Predict
predictions = forecaster.predict(
    steps=10,
    exog=exog_test
)
```

---

## Tidymodels vs Skforecast - Detailed Comparison

### Feature Comparison Matrix

| Feature | Tidymodels/Modeltime | Skforecast | Notes |
|---------|---------------------|------------|-------|
| **Core Architecture** |
| Design Philosophy | Unified API across packages | sklearn-compatible wrappers | Both emphasize consistency |
| Language | R | Python | - |
| Data Structure | tibbles (data.frames) | pandas DataFrame/Series | - |
| Syntax Style | Pipe-based (`%>%`, `|>`) | Object-oriented (`.fit()`, `.predict()`) | - |
| **Forecasting Strategies** |
| Recursive Forecasting | ✅ Via `recursive()` | ✅ ForecasterRecursive | Both support |
| Direct Forecasting | ❌ Not directly | ✅ ForecasterDirect | Skforecast advantage |
| Multi-Series (Global) | ✅ Via nested modeltime | ✅ ForecasterRecursiveMultiSeries | Both support |
| Multivariate Dependencies | ❌ Limited | ✅ ForecasterDirectMultiVariate | Skforecast advantage |
| **Model Types** |
| Classical TS Models | ✅ ARIMA, ETS, Prophet, TBATS, NNETAR | ✅ SARIMAX | Tidymodels has more |
| ML Models | ✅ Via parsnip (30+ types) | ✅ Any sklearn regressor | Both extensive |
| Boosted Hybrids | ✅ ARIMA+XGBoost, Prophet+XGBoost | ✅ Can be manually created | Tidymodels built-in |
| Deep Learning | ✅ Via modeltime.gluonts | ✅ ForecasterRNN | Both support |
| **Feature Engineering** |
| Lag Features | ✅ `step_lag()` | ✅ `lags` parameter | Both support |
| Rolling Windows | ✅ `step_window()`, `tk_augment_slidify()` | ✅ RollingFeatures, window_features | Both support |
| Date Features | ✅ `step_date()`, `tk_augment_timeseries_signature()` | ❌ Manual creation needed | Tidymodels advantage |
| Holiday Features | ✅ `step_holiday()`, `tk_augment_holiday_signature()` | ❌ Manual creation | Tidymodels advantage |
| Fourier Features | ✅ `step_harmonic()`, `tk_augment_fourier()` | ❌ Manual creation | Tidymodels advantage |
| Custom Window Features | ✅ Via custom steps | ✅ RollingFeatures + custom | Both support |
| Differencing | ✅ `step_diff()` | ✅ `differentiation` parameter | Both support |
| **Preprocessing** |
| Pipeline System | ✅ recipes | ✅ sklearn pipelines | recipes more flexible |
| Target Transformation | ✅ Via recipes | ✅ `transformer_y` | Both support |
| Exog Transformation | ✅ Via recipes | ✅ `transformer_exog` | Both support |
| Per-Series Transforms | ✅ In nested modeltime | ✅ `transformer_series` dict | Both support |
| **Cross-Validation** |
| Time Series CV | ✅ `sliding_period()`, `sliding_window()`, `time_series_cv()` | ✅ TimeSeriesFold | Both comprehensive |
| Rolling Origin | ✅ `rolling_origin()` (superseded) | ✅ Via TimeSeriesFold | Both support |
| Expanding Window | ✅ Via cumulative=TRUE | ✅ Via fold parameters | Both support |
| Fixed Window | ✅ Via cumulative=FALSE | ✅ Via fold parameters | Both support |
| Gap Support | ✅ Via lag parameter | ✅ `gap` parameter | Both support |
| Intermittent Refit | ✅ Manual control | ✅ Via fold_stride | Skforecast more explicit |
| **Hyperparameter Tuning** |
| Grid Search | ✅ `tune_grid()` | ✅ `grid_search_forecaster()` | Both support |
| Random Search | ✅ Via dials | ✅ `random_search_forecaster()` | Both support |
| Bayesian Optimization | ✅ `tune_bayes()` | ✅ `bayesian_search_forecaster()` (Optuna) | Both support |
| Lag Selection | ❌ Manual specification | ✅ `lags_grid` parameter | Skforecast advantage |
| Simultaneous Tuning | ✅ Via tune + workflows | ✅ Built-in | Both support |
| **Backtesting/Evaluation** |
| Backtesting Function | ✅ `modeltime_fit_resamples()` | ✅ `backtesting_forecaster()` | Both support |
| Multiple Metrics | ✅ `metric_set()` | ✅ Via metric parameter | Both support |
| Prediction Intervals | ✅ Via `conf_interval` | ✅ Via `interval` parameter | Both support |
| Residual Storage | ✅ Via `modeltime_residuals()` | ✅ `in_sample_residuals_`, `out_sample_residuals_` | Both support |
| **Probabilistic Forecasting** |
| Bootstrap Intervals | ✅ Limited support | ✅ `predict_bootstrapping()` | Skforecast better |
| Conformal Prediction | ❌ Not built-in | ✅ `predict_interval()` with conformal | Skforecast advantage |
| Quantile Forecasting | ✅ Via quantile regression | ✅ `predict_quantiles()` | Both support |
| Distribution Fitting | ❌ Limited | ✅ `predict_dist()` | Skforecast advantage |
| Binned Residuals | ❌ Not built-in | ✅ `binner_kwargs` | Skforecast advantage |
| **Metrics** |
| Standard Metrics | ✅ RMSE, MAE, MAPE, R² | ✅ Via sklearn.metrics | Both support |
| MASE | ✅ `mase()` | ✅ `mean_absolute_scaled_error()` | Both support |
| SMAPE | ✅ `smape()` | ✅ `symmetric_mean_absolute_percentage_error()` | Both support |
| Custom Metrics | ✅ Via yardstick | ✅ Via custom functions | Both support |
| Metric Sets | ✅ `metric_set()` | ❌ Pass list to functions | Tidymodels cleaner |
| **Workflow Management** |
| Workflow Objects | ✅ workflows package | ❌ Not formalized | Tidymodels advantage |
| Model Tables | ✅ `modeltime_table()` | ❌ Manual management | Tidymodels advantage |
| Workflowsets | ✅ workflowsets package | ❌ Not available | Tidymodels advantage |
| Model Comparison | ✅ `modeltime_accuracy()` | ✅ Via backtesting results | Both support |
| **Ensembling** |
| Simple Average | ✅ `ensemble_average()` | ❌ Manual | Tidymodels advantage |
| Weighted Average | ✅ `ensemble_weighted()` | ❌ Manual | Tidymodels advantage |
| Stacking | ✅ `ensemble_model_spec()`, stacks package | ❌ Manual | Tidymodels advantage |
| **Visualization** |
| Forecast Plots | ✅ `plot_modeltime_forecast()` | ❌ Manual (matplotlib) | Tidymodels advantage |
| Residual Plots | ✅ `plot_modeltime_residuals()` | ❌ Manual | Tidymodels advantage |
| CV Plan Visualization | ✅ `plot_time_series_cv_plan()` | ❌ Manual | Tidymodels advantage |
| Interactive Plots | ✅ Via plotly | ❌ Not built-in | Tidymodels advantage |
| **Production Features** |
| Model Serialization | ✅ saveRDS/readRDS | ✅ pickle/joblib | Both support |
| Drift Detection | ❌ Not built-in | ✅ drift detection module | Skforecast advantage |
| Model Registry | ❌ Manual | ❌ Manual | Neither |
| **Parallel Processing** |
| Parallel Tuning | ✅ Via foreach | ✅ `n_jobs` parameter | Both support |
| Parallel Backtesting | ✅ `parallel_start()` | ✅ `n_jobs` parameter | Both support |
| **Extensibility** |
| Custom Models | ✅ `new_modeltime_bridge()` | ✅ Custom sklearn classes | Both support |
| Custom Steps | ✅ recipes framework | ✅ Custom transformers | Both support |
| Engine System | ✅ Multiple engines per model | ✅ Any sklearn regressor | Tidymodels more structured |
| **Documentation** |
| Quality | ✅ Excellent (tidymodels.org) | ✅ Excellent (skforecast.org) | Both excellent |
| Examples | ✅ Extensive tutorials | ✅ Comprehensive guides | Both excellent |
| **Community** |
| Maturity | ✅ Mature (RStudio backed) | ✅ Active development | Both active |
| GitHub Stars | ~600 (modeltime) | ~1.1k (skforecast) | - |

### Architecture Comparison

#### Tidymodels Architecture

**Strengths:**
1. **Separation of Concerns:** Clear separation between preprocessing (recipes), modeling (parsnip), workflows (workflows), and tuning (tune)
2. **Composability:** Components can be mixed and matched flexibly
3. **Consistency:** Unified API across all packages
4. **Engine System:** Multiple backends for same model type with harmonized parameters
5. **Integration:** Deep integration with R statistical ecosystem
6. **Visualization:** Built-in plotting functions

**Weaknesses:**
1. **Complexity:** Steep learning curve due to many packages
2. **R Ecosystem:** Limited to R users
3. **Direct Forecasting:** No native direct multi-step strategy
4. **Probabilistic:** Limited conformal prediction support

#### Skforecast Architecture

**Strengths:**
1. **Simplicity:** Single package, focused API
2. **Sklearn Integration:** Leverages entire sklearn ecosystem
3. **Forecasting Strategies:** Both recursive and direct built-in
4. **Probabilistic:** Strong conformal prediction and bootstrapping
5. **Multi-Series:** Excellent global forecasting capabilities
6. **Lag Tuning:** Can tune lag structure automatically

**Weaknesses:**
1. **Feature Engineering:** Less comprehensive time series features (no holidays, limited date features)
2. **Visualization:** No built-in plotting functions
3. **Workflow Management:** No formal workflow/pipeline object
4. **Ensembling:** No built-in ensemble methods
5. **Model Types:** Fewer classical time series models

---

### Workflow Comparison (Side-by-Side)

#### Task: Forecast with Lagged Features and Hyperparameter Tuning

**Tidymodels:**

```r
library(tidymodels)
library(modeltime)
library(timetk)

# Feature engineering
rec <- recipe(value ~ date, data = train) %>%
  step_lag(value, lag = 1:7) %>%
  step_normalize(all_numeric_predictors())

# Model specification
rf_spec <- rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Workflow
wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

# Time series CV
cv_splits <- sliding_period(
  train,
  index = date,
  period = "month",
  lookback = "6 months",
  assess_start = 1,
  step = 1
)

# Hyperparameter tuning
tune_results <- tune_grid(
  wf,
  resamples = cv_splits,
  grid = 20,
  metrics = metric_set(rmse, mae)
)

# Finalize and forecast
best_params <- select_best(tune_results, "rmse")
final_wf <- finalize_workflow(wf, best_params)
final_fit <- fit(final_wf, train)

predictions <- predict(final_fit, test)
```

**Skforecast:**

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create forecaster with lags and transformation
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(),
    lags=range(1, 8),  # Lags 1-7
    transformer_y=StandardScaler()
)

# Define CV
cv = TimeSeriesFold(
    steps=30,  # Forecast horizon
    fold_stride=30,
    gap=0
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [3, 5, 7]
}

lags_grid = [range(1, 8)]

results = grid_search_forecaster(
    forecaster=forecaster,
    y=y_train,
    cv=cv,
    param_grid=param_grid,
    lags_grid=lags_grid,
    metric='mean_absolute_error',
    return_best=True,
    n_jobs=-1
)

# Forecaster is now fitted with best params
predictions = forecaster.predict(steps=len(y_test))
```

**Observations:**
- Tidymodels separates recipe, model, and workflow steps
- Skforecast is more compact, single-object approach
- Tidymodels has clearer pipeline visualization
- Skforecast `return_best=True` automatically refits
- Both support parallel processing

---

### Integration Strategy for py-tidymodels

#### Recommendation: Hybrid Approach

Based on this analysis, I recommend a **hybrid strategy** that combines the best of both:

**Use Skforecast for:**
1. **Backend forecaster implementation** - Leverage existing battle-tested recursive/direct forecasters
2. **Multi-series forecasting** - Use ForecasterRecursiveMultiSeries as backend
3. **Probabilistic forecasting** - Leverage conformal prediction and bootstrapping
4. **Lag selection** - Use built-in lag grid search

**Implement Tidymodels-style for:**
1. **Workflow management** - Build workflow objects (modeltime_table, workflows)
2. **Feature engineering** - Comprehensive recipes with time series steps
3. **Visualization** - Plot functions for forecasts, residuals, CV plans
4. **Ensembling** - Ensemble methods (average, weighted, stacking)
5. **Model types** - Wrap ARIMA, Prophet, TBATS as tidymodels-style

**Architecture Proposal:**

```python
# py-parsnip wraps skforecast
from py_parsnip import rand_forest
from py_recipes import recipe, step_lag, step_date
from py_workflows import workflow
from py_modeltime import modeltime_table, recursive

# Create recipe (extends skforecast feature engineering)
rec = (
    recipe("value ~ date + exog1", data=train)
    .step_date("date", features=["dow", "month"])
    .step_lag("value", lags=[1, 7, 14])
    .step_normalize(all_numeric_predictors())
)

# Model (uses skforecast backend)
rf_model = (
    rand_forest(mtry=10, trees=100)
    .set_engine("skforecast_sklearn")  # Wraps sklearn via skforecast
    .set_mode("regression")
)

# Workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(rf_model)
)

# Modeltime workflow
models_tbl = modeltime_table(
    wf.fit(train),
    arima_reg().set_engine("statsmodels").fit(train),
    prophet_reg().set_engine("prophet").fit(train)
)

# Use tidymodels-style workflow
calibrated = models_tbl.modeltime_calibrate(test)
accuracy = calibrated.modeltime_accuracy()
forecast = calibrated.modeltime_forecast(h="3 months")
plot_modeltime_forecast(forecast)
```

#### Integration Benefits

**Advantages of Hybrid Approach:**
1. ✅ Leverage skforecast's proven forecaster implementations
2. ✅ Avoid reimplementing complex recursive/direct strategies
3. ✅ Add tidymodels' superior feature engineering (holidays, date features)
4. ✅ Provide tidymodels' workflow management and visualization
5. ✅ Enable tidymodels' ensembling capabilities
6. ✅ Maintain consistent API for R users transitioning to Python

**Implementation Strategy:**

**Phase 1: Core Integration**
- Wrap skforecast forecasters in parsnip-style models
- Implement basic recipes with sklearn-compatible interface
- Create workflow objects that coordinate recipe + forecaster

**Phase 2: Enhanced Features**
- Add time series recipe steps (date, holiday, Fourier)
- Implement modeltime_table and calibration
- Build visualization functions

**Phase 3: Advanced Features**
- Implement ensembling on top of skforecast
- Add workflowsets for experimentation
- Create nested forecasting for multiple series

---

### Gaps & Opportunities

#### What Tidymodels Provides that Skforecast Doesn't

1. **Comprehensive Date Features** - Holiday calendars, automatic date component extraction
2. **Workflow Objects** - Formal pipeline objects for production
3. **Model Tables** - Organized comparison of multiple models
4. **Ensembling** - Built-in averaging, weighting, stacking
5. **Visualization** - Production-ready plotting functions
6. **Workflowsets** - Systematic experimentation framework
7. **Classical Models** - More ARIMA variants, ETS, Prophet, TBATS

#### What Skforecast Provides that Tidymodels Doesn't

1. **Direct Forecasting** - Native direct multi-step strategy
2. **Lag Tuning** - Automatic lag structure selection
3. **Conformal Prediction** - Rigorous prediction intervals
4. **Drift Detection** - Production model monitoring
5. **Binned Residuals** - Heteroscedastic prediction intervals
6. **Explicit Refit Control** - Fine-grained backtesting control

#### Priority Implementations for py-tidymodels

**High Priority (Months 1-6):**
1. ✅ Wrap skforecast forecasters in parsnip-style API
2. ✅ Implement time series recipes (lag, date, holiday steps)
3. ✅ Create modeltime_table and calibration workflow
4. ✅ Build visualization functions

**Medium Priority (Months 7-12):**
1. ✅ Implement ensembling methods
2. ✅ Add workflowsets for experimentation
3. ✅ Extend to nested/multi-series forecasting
4. ✅ Add ARIMA, Prophet wrappers

**Lower Priority (Year 2):**
1. ✅ Advanced probabilistic methods
2. ✅ Drift detection integration
3. ✅ Deep learning integration (GluonTS)
4. ✅ H2O AutoML integration

---

## Additional Package Analysis

### filtro - Filter-Based Feature Selection

**Version:** 0.2.0.9000 (Development)
**Status:** Experimental (Under Active Development)
**Priority:** MEDIUM
**Repository:** https://github.com/tidymodels/filtro
**CRAN:** Available

#### Overview

filtro is a tidymodels package that provides **supervised filter-based feature selection methods**. It scores and ranks feature relevance using statistical metrics, making it easier to identify important predictors before modeling. The package is designed to integrate seamlessly with the recipes ecosystem.

#### Core Concepts

- **Score Objects**: S7 objects that define feature scoring methods (e.g., `score_aov_pval`, `score_cor_pearson`)
- **fit()**: Compute scores for features in a dataset
- **Filtering**: Select top features by proportion (`show_best_score_prop()`) or count (`show_best_score_num()`)
- **Desirability Functions**: Multi-objective optimization using desirability2 package

#### Complete Feature Selection Methods

| Method | Function | Data Types | Description | Range |
|--------|----------|------------|-------------|-------|
| **ANOVA F-test** | `score_aov_fstat` | Numeric outcome + Factor predictors OR Factor outcome + Numeric predictors | F-statistic from linear model | [0, ∞) |
| **ANOVA p-value** | `score_aov_pval` | Same as above | -log10(p-value) from F-test | [0, ∞) |
| **Pearson Correlation** | `score_cor_pearson` | Numeric outcome + Numeric predictors | Linear correlation coefficient | [-1, 1] |
| **Spearman Correlation** | `score_cor_spearman` | Numeric outcome + Numeric predictors | Rank-based correlation | [-1, 1] |
| **Random Forest Importance** | `score_imp_rf` | Any outcome + Any predictors | Variable importance from RF | [0, 1] |
| **Information Gain** | `score_info_gain` | Any outcome + Any predictors | Entropy-based filter | [0, ∞) |
| **Gain Ratio** | `score_gain_ratio` | Any outcome + Any predictors | Normalized information gain | [0, 1] |
| **Symmetrical Uncertainty** | `score_sym_uncert` | Any outcome + Any predictors | Symmetric entropy measure | [0, 1] |
| **ROC AUC** | `score_roc_auc` | Binary factor outcome + Numeric predictors | Area under ROC curve | [0, 1] |
| **Chi-squared** | `score_chi_squared` | Factor outcome + Factor predictors | Chi-squared test statistic | [0, ∞) |
| **Fisher's Exact** | `score_fisher_exact` | Factor outcome + Factor predictors | Fisher's exact test p-value | [0, 1] |

#### Core Functions Reference

**Scoring Functions (fit method):**
```r
# Compute scores for all predictors
fitted_scores <- score_aov_pval |>
  fit(outcome ~ ., data = data, case_weights = NULL)

# Access results
fitted_scores@results  # tibble with columns: name, score, outcome, predictor
```

**Single Score Filtering Functions:**
```r
# Filter by proportion of features
show_best_score_prop(fitted_scores, prop_terms = 0.2)

# Filter by number of features
show_best_score_num(fitted_scores, num_terms = 5)

# Filter by score cutoff
show_best_score_cutoff(fitted_scores, cutoff = 10)

# Rank features (dense, min, or fractional)
rank_best_score_dense(fitted_scores)
rank_best_score_min(fitted_scores)
```

**Multiple Score Filtering with Desirability:**
```r
# Combine multiple scoring methods
scores_list <- list(
  score_cor_pearson |> fit(y ~ ., data),
  score_imp_rf |> fit(y ~ ., data),
  score_info_gain |> fit(y ~ ., data)
)

# Combine into wide format
scores_combined <- fill_safe_values(scores_list)

# Multi-objective optimization
show_best_desirability_prop(
  scores_combined,
  maximize(cor_pearson, low = 0, high = 1),
  maximize(imp_rf),
  maximize(infogain),
  prop_terms = 0.3
)

# Alternative desirability functions:
# - minimize() - smaller is better
# - target() - aim for specific value
# - constrain() - box constraint
```

**Utility Functions:**
```r
# Fill NA scores with safe fallback values
fill_safe_value(fitted_scores)
fill_safe_values(scores_list)

# Arrange scores by importance
arrange_score(fitted_scores)

# Bind multiple score objects
bind_scores(score1, score2, score3)

# Transform scores
filtro_abs_trans(scores)  # Absolute value transformation
```

#### Architecture & Design Patterns

1. **S7 Class System**: Uses modern S7 object system (not S3/S4)
   - `class_score`: Base class with properties for all scoring methods
   - Method-specific subclasses: `class_score_aov`, `class_score_cor`, etc.

2. **Functional Design**: Pipe-friendly interface
   ```r
   result <- score_aov_pval |>
     fit(y ~ ., data) |>
     fill_safe_value() |>
     show_best_score_prop(0.2)
   ```

3. **Fault Tolerance**: Handles missing data and failed computations gracefully
   - Returns NA for incomputable scores
   - Fallback values for filtering operations

4. **Integration Points**:
   - Works with standard R formulas and data frames
   - Compatible with recipes workflow
   - Uses desirability2 for multi-objective optimization
   - Supports case weights where applicable

#### Usage Pattern

```r
library(filtro)
library(desirability2)
library(dplyr)

# 1. Single method approach
scores <- score_cor_pearson |>
  fit(Sale_Price ~ ., data = ames) |>
  fill_safe_value()

# Select top 20% features
top_features <- scores |>
  show_best_score_prop(prop_terms = 0.2)

# 2. Multi-method with desirability
rf_scores <- score_imp_rf |> fit(class ~ ., data = cells)
ig_scores <- score_info_gain |> fit(class ~ ., data = cells)
cor_scores <- score_cor_pearson |> fit(class ~ ., data = cells)

combined <- list(rf_scores, ig_scores, cor_scores) |>
  fill_safe_values()

# Optimize multiple objectives
best <- combined |>
  show_best_desirability_num(
    maximize(imp_rf),
    maximize(infogain),
    maximize(cor_pearson, low = 0, high = 1),
    num_terms = 10
  )
```

#### Differences from recipes Built-in Steps

**filtro provides:**
- ✅ Statistical hypothesis tests (ANOVA, Chi-squared, Fisher)
- ✅ Entropy-based methods (information gain, gain ratio)
- ✅ Multi-objective feature selection via desirability functions
- ✅ Formal scoring objects with metadata
- ✅ Standalone evaluation before recipes workflow

**recipes provides:**
- ✅ Integration with preprocessing pipeline
- ✅ step_select_* functions for workflow integration
- ✅ Automatic handling of train/test splits
- ✅ Feature engineering combined with selection

**Relationship:**
- filtro is for **exploratory feature selection** and analysis
- recipes is for **production feature engineering** pipelines
- filtro results can inform which features to include in recipes

#### Python Port Strategy

**Priority Level: MEDIUM**

**Rationale:**
- Feature selection is important but not blocking core workflows
- Many alternatives exist in Python (scikit-learn, feature-engine, RFECV)
- Lower priority than recipes, parsnip, workflows

**Recommended Approach:**

**Phase 1 (Optional - Year 2):**
Create `py-filtro` with core functionality:
```python
# Python equivalent design
from py_filtro import score_correlation, score_mutual_info, score_f_test

# Single method
scores = score_correlation().fit(X, y)
top_features = scores.select_best(k=10)

# Multi-objective with desirability
from py_filtro import combine_scores, maximize

combined = combine_scores([
    score_correlation().fit(X, y),
    score_mutual_info().fit(X, y),
    score_f_test().fit(X, y)
])

best = combined.select_desirability(
    maximize('correlation'),
    maximize('mutual_info'),
    k=15
)
```

**Alternative: Use Existing Python Tools**
- `sklearn.feature_selection`: Chi2, f_classif, mutual_info_*
- `feature-engine`: Multiple feature selection transformers
- `mrmr`: Minimum redundancy maximum relevance
- `boruta`: All-relevant feature selection

**Recommendation:** Defer py-filtro implementation. Instead:
1. Document how to use scikit-learn's feature_selection with py-recipes
2. Create helper functions to wrap sklearn selectors in recipes-compatible format
3. Only build py-filtro if users specifically request filtro-style API

---

### pytimetk - Existing Python Time Series Toolkit

**Version:** 2.2.0
**Status:** PRODUCTION-READY ✅
**Priority:** CRITICAL - USE AS-IS
**Repository:** https://github.com/business-science/pytimetk
**PyPI:** Available
**Maintainer:** Business Science (same as R timetk)

#### Executive Summary

**🚨 CRITICAL FINDING: pytimetk is a mature, production-ready Python implementation of R's timetk package. We should USE IT, not reimplement it.**

#### Package Overview

pytimetk is the **official Python port** of R's timetk package, maintained by Business Science. It provides comprehensive time series preprocessing, feature engineering, and visualization capabilities optimized for both pandas and polars backends.

**Key Stats:**
- ✅ **66 test files** - comprehensive test coverage
- ✅ **Version 2.2.0** - stable and mature
- ✅ **Active development** - regular updates
- ✅ **Professional maintenance** - Business Science team
- ✅ **GPU acceleration** (Beta) - RAPIDS/cuDF support
- ✅ **Feature store** (Beta) - versioned feature persistence
- ✅ **Dual backend** - pandas and polars support

#### Complete Feature Set

**Module Structure:**
```
pytimetk/
├── core/               # Time series operations
├── feature_engineering/  # Feature creation
├── plot/               # Visualization
├── finance/            # Financial indicators
├── crossvalidation/    # Time series CV
├── feature_store/      # Feature versioning
└── utils/              # Helper functions
```

#### Core Functions (pytimetk.core)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `summarize_by_time()` | Aggregate time series by period | `freq`, `agg_func`, `engine` |
| `apply_by_time()` | Apply function by time period | `freq`, `func` |
| `pad_by_time()` | Fill gaps in irregular time series | `freq`, `fill_na_method` |
| `filter_by_time()` | Filter by date range | `start_date`, `end_date` |
| `future_frame()` | Create future time index | `length_out` |
| `ts_summary()` | Comprehensive time series summary | Returns frequency, gaps, etc. |
| `ts_features()` | Extract statistical features | Uses tsfeatures library |
| `anomalize()` | Detect and clean anomalies | `method`, `alpha`, `clean` |
| `correlate()` | Correlation funnel analysis | For feature selection |
| `binarize()` | Convert continuous to binary | For correlation funnel |

**Time Series Intelligence:**
```python
# Automatic frequency detection
get_frequency(data, date_column)
get_seasonal_frequency(data)
get_trend_frequency(data)

# Date/time summaries
get_date_summary(data, date_column)
get_diff_summary(data, date_column)
get_frequency_summary(data, date_column)
```

#### Feature Engineering Functions

**Time Series Signature (29 Features):**
```python
augment_timeseries_signature(
    data,
    date_column='date',
    engine='polars'  # or 'pandas'
)
```

**Generated Features:**
- Index: `_index_num`
- Year: `_year`, `_year_iso`, `_yearstart`, `_yearend`, `_leapyear`
- Quarter: `_quarter`, `_quarteryear`, `_quarterstart`, `_quarterend`
- Month: `_month`, `_month_lbl`, `_monthstart`, `_monthend`
- Week: `_yweek`, `_mweek`, `_half`
- Day: `_wday`, `_wday_lbl`, `_mday`, `_qday`, `_yday`, `_weekend`
- Time: `_hour`, `_minute`, `_second`, `_msecond`, `_nsecond`, `_am_pm`

**Lag & Lead Features:**
```python
augment_lags(data, date_column, value_column, lags=[1,7,14])
augment_leads(data, date_column, value_column, leads=[1,2,3])
augment_diffs(data, date_column, value_column, differences=[1,7])
augment_pct_change(data, date_column, value_column, periods=[1,7])
```

**Rolling Window Features:**
```python
# Standard rolling aggregations (10X to 3500X faster than pandas!)
augment_rolling(
    data,
    date_column='date',
    value_column='value',
    window=[7, 14, 28],
    window_func=['mean', 'std', 'min', 'max'],
    engine='polars'  # GPU acceleration with 'gpu'
)

# Custom rolling functions
augment_rolling_apply(
    data,
    date_column='date',
    value_column='value',
    window=7,
    func=custom_function
)
```

**Expanding Window Features:**
```python
augment_expanding(data, date_column, value_column,
                  window_func=['mean', 'std', 'sum'])

augment_expanding_apply(data, date_column, value_column, func=custom_func)
```

**Advanced Feature Engineering:**
```python
# Holiday features
augment_holiday_signature(data, date_column, country='US')

# Fourier terms for seasonality
augment_fourier(data, date_column, periods=[7, 365.25], max_order=5)

# Exponentially weighted features
augment_ewm(data, date_column, value_column,
            alpha=0.3, adjust=True)

# Spline transformations
augment_spline(data, date_column, value_column, n_knots=5)

# Hilbert transform (phase and amplitude)
augment_hilbert(data, value_column)

# Wavelet transform
augment_wavelet(data, value_column, scales=[1,2,4,8])
```

#### Financial Indicators (pytimetk.finance)

Complete technical analysis library:

| Function | Indicator | Description |
|----------|-----------|-------------|
| `augment_macd()` | MACD | Moving Average Convergence Divergence |
| `augment_rsi()` | RSI | Relative Strength Index |
| `augment_bbands()` | Bollinger Bands | Volatility bands |
| `augment_atr()` | ATR | Average True Range |
| `augment_roc()` | ROC | Rate of Change |
| `augment_ppo()` | PPO | Percentage Price Oscillator |
| `augment_cmo()` | CMO | Chande Momentum Oscillator |
| `augment_adx()` | ADX | Average Directional Index |
| `augment_stochastic_oscillator()` | Stochastic | Momentum indicator |
| `augment_qsmomentum()` | QS Momentum | Quantitative strategy momentum |
| `augment_drawdown()` | Drawdown | Peak-to-trough decline |
| `augment_rolling_risk_metrics()` | Risk Metrics | Sharpe, Sortino ratios |
| `augment_fip_momentum()` | FIP | Fractal Interpolation Polynomial |
| `augment_hurst_exponent()` | Hurst | Long-term memory |
| `augment_ewma_volatility()` | EWMA Vol | Exponential volatility |
| `augment_regime_detection()` | Regime | Market regime shifts |

#### Visualization Functions

**Core Plotting:**
```python
# Main time series plotting
plot_timeseries(
    data,
    date_column='date',
    value_column='value',
    color_column=None,  # For grouped series
    facet_ncol=1,
    smooth=False,
    engine='plotly'  # or 'plotnine', 'matplotlib'
)

# Anomaly visualization
plot_anomalies(data, date_column, observed_column, anomaly_column)
plot_anomalies_decomp(data, date_column, observed_column)
plot_anomalies_cleaned(data, date_column, observed_column)

# Feature selection visualization
plot_correlation_funnel(data, target_column)
```

**Themes:**
```python
theme_timetk()  # Consistent time series plot styling
palette_timetk()  # Color palette for time series
```

#### Time Series Cross-Validation

```python
from pytimetk import TimeSeriesCV, TimeSeriesCVSplitter

# Compatible with sklearn
cv = TimeSeriesCV(
    frequency="days",
    train_size=180,
    forecast_horizon=30,
    gap=0,
    stride=30,
    mode='forward'  # or 'backward', 'rolling'
)

# Use with sklearn models
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    model.fit(X_train, y.iloc[train_idx])
    predictions = model.predict(X_test)
```

#### Feature Store (Beta)

**Revolutionary Feature: Version and cache expensive features**

```python
from pytimetk import FeatureStore

store = FeatureStore(
    artifact_uri='./features',  # or 's3://...', 'gs://...'
    enable_locking=True
)

# Register feature transformation
store.register(
    'timeseries_signature',
    lambda data: tk.augment_timeseries_signature(
        data,
        date_column='date',
        engine='polars'
    ),
    default_key_columns=('id',),
    description='Calendar features for time series'
)

# Build once, cache forever
result = store.build('timeseries_signature', df)
print(result.from_cache)  # False first time, True subsequently

# Integration with MLflow (optional)
store.log_to_mlflow(run_id, feature_set_name)
```

#### Polars Support & GPU Acceleration

**Polars .tk Accessor:**
```python
import polars as pl
import pytimetk as tk

# Use directly on polars DataFrames
df = pl.read_csv('data.csv')

# All functions work with polars
result = df.tk.augment_timeseries_signature(
    date_column='date',
    engine='polars'
)

# Plot directly from polars
df.tk.plot_timeseries(date_column='date', value_column='sales')
```

**GPU Acceleration (Beta):**
```python
# Install GPU support
# pip install pytimetk[gpu] --extra-index-url=https://pypi.nvidia.com
# pip install "polars[gpu]" --extra-index-url=https://pypi.nvidia.com

# Automatic GPU acceleration
result = tk.augment_rolling(
    data,
    date_column='date',
    value_column='value',
    window=[7, 14, 28],
    window_func=['mean', 'std'],
    engine='polars'
)

# Use with polars GPU engine
lazy_df = pl.scan_csv('large_data.csv')
result = lazy_df.collect(engine='gpu')

# GPU utilities
from pytimetk.utils.gpu_support import (
    is_cudf_available,
    is_polars_gpu_available
)
```

#### Performance Characteristics

**Speed Comparisons (from pytimetk documentation):**

| Operation | pandas | pytimetk (polars) | Speedup |
|-----------|--------|-------------------|---------|
| `summarize_by_time()` | Baseline | 13.4X faster | 13.4X |
| `augment_rolling()` | Baseline | 10X-3500X faster | 10-3500X |
| `augment_timeseries_signature()` | 29 lines | 1 line | - |
| `plot_timeseries()` | 16 lines | 2 lines | - |

#### Code Quality Assessment

**✅ Production-Ready Indicators:**
- Professional package structure with proper modules
- Comprehensive documentation with examples
- 66 test files covering functionality
- Type hints throughout codebase
- Both pandas and polars backends
- Active development and maintenance
- Version 2.2.0 indicates maturity
- Feature store with MLflow integration
- GPU acceleration support

**Architecture Highlights:**
- Clean separation of concerns (core, feature_engineering, plot, etc.)
- Consistent API across all functions
- Pandas-flavor decorators for method chaining
- Polars namespace extension via `.tk` accessor
- Engine abstraction for backend switching
- Memory optimization utilities

#### Integration Recommendations

**CRITICAL RECOMMENDATION: Use pytimetk as py-timetk**

**DO NOT reimplement timetk. Instead:**

1. **✅ Use pytimetk directly** for all time series feature engineering
2. **✅ Integrate with py-recipes** by creating recipe steps that call pytimetk functions
3. **✅ Document integration patterns** for py-tidymodels users
4. **✅ Consider contributing** to pytimetk if features are missing

**Integration Strategy:**

```python
# In py-recipes, create steps that wrap pytimetk
from pytimetk import augment_timeseries_signature, augment_rolling
from py_recipes import step

@step
class step_timeseries_signature:
    def __init__(self, date_column, engine='polars'):
        self.date_column = date_column
        self.engine = engine

    def prep(self, data):
        return self

    def bake(self, data):
        return augment_timeseries_signature(
            data,
            date_column=self.date_column,
            engine=self.engine
        )

@step
class step_rolling:
    def __init__(self, date_column, value_column, window, window_func):
        self.date_column = date_column
        self.value_column = value_column
        self.window = window
        self.window_func = window_func

    def prep(self, data):
        return self

    def bake(self, data):
        return augment_rolling(
            data,
            date_column=self.date_column,
            value_column=self.value_column,
            window=self.window,
            window_func=self.window_func
        )
```

#### Gap Analysis: R timetk vs. pytimetk

**✅ Fully Implemented in pytimetk:**
- Time series signature (29 features)
- Lag, lead, diff, pct_change features
- Rolling and expanding window statistics
- Date/time filtering and padding
- Frequency detection and summaries
- Holiday features
- Fourier terms
- Anomaly detection
- Visualization (multiple engines)
- Cross-validation
- Financial indicators (16+ functions)

**✅ Python Enhancements (Not in R timetk):**
- Feature store with versioning and caching
- GPU acceleration via RAPIDS/cuDF
- Native polars support
- MLflow integration
- Dual backend (pandas/polars) architecture

**⚠️ Possible Gaps (Need Verification):**
- Some obscure date utilities from R's lubridate
- Specific R-only date formats
- Integration with R's tsibble objects (Python uses pandas/polars)

**Recommendation:** Any missing features are likely minor and can be added to pytimetk via contribution rather than building separate package.

#### Maintenance & Community

**Maintainer:** Business Science (Matt Dancho, Justin Kurland, Jeff Tackes, Samuel Macêdo, Lucas Okwudishu, Alex Riggio)

**Community Health:**
- ✅ Active GitHub repository
- ✅ Professional documentation site
- ✅ PyPI releases
- ✅ Responsive to issues
- ✅ Same team as R timetk (ensures consistency)

**Trust Factors:**
- Business Science is well-known in R/Python data science community
- Professional training courses and consulting services
- Long-term commitment to maintenance
- Enterprise-ready quality

#### Action Items for py-tidymodels Integration

**Immediate (Month 1):**
1. ✅ Add pytimetk as dependency in py-tidymodels
2. ✅ Create recipe steps that wrap pytimetk functions
3. ✅ Document integration patterns
4. ✅ Add examples showing pytimetk + py-recipes workflows

**Short-term (Months 2-3):**
1. ✅ Ensure pytimetk works with py-recipes prep/bake paradigm
2. ✅ Create helper functions for common feature engineering patterns
3. ✅ Add pytimetk to py-tidymodels ecosystem documentation
4. ✅ Test GPU acceleration with large datasets

**Long-term (Months 4-6):**
1. ✅ Contribute any needed features back to pytimetk
2. ✅ Collaborate with Business Science on integration
3. ✅ Showcase pytimetk in py-tidymodels tutorials
4. ✅ Consider joint documentation/tutorials

---

### modeltime.resample - Complete R Package Analysis

**Version:** 0.3.0.9000 (Development)
**Status:** Stable
**Priority:** CRITICAL - Already Partially Ported
**Repository:** https://github.com/business-science/modeltime.resample
**CRAN:** Available

#### Overview

modeltime.resample extends modeltime with **resampling-based model evaluation** for time series. It provides tools to assess model performance and stability across multiple time-based cross-validation folds.

**Note:** py-modeltime-resample already implements much of this functionality. This section documents the complete R API to identify any missing features.

#### Complete Function Reference

**Core Functions:**

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `modeltime_fit_resamples()` | Fit models to resamples | `object`, `resamples`, `control` | Modeltime table with `.resample_results` column |
| `modeltime_resample_accuracy()` | Calculate accuracy metrics | `object`, `summary_fns`, `metric_set` | Accuracy table by resample |
| `plot_modeltime_resamples()` | Visualize resample results | `object`, `type`, `summary_fns` | ggplot2 or plotly plot |

**Helper Functions:**

| Function | Description | Purpose |
|----------|-------------|---------|
| `unnest_modeltime_resamples()` | Extract predictions from resamples | Internal utility |
| `get_target_text_from_resamples()` | Identify target variable name | Internal utility |
| `mdl_time_fit_resamples()` | Low-level fitting function | S3 generic for workflows/models |

#### Detailed Function Specifications

**1. modeltime_fit_resamples()**

```r
modeltime_fit_resamples(
  object,       # Modeltime table (mdl_time_tbl)
  resamples,    # rset object (from rsample or timetk)
  control = control_resamples()  # tune::control_resamples()
)
```

**Purpose:** Fits each model in modeltime table to all resamples, generating out-of-sample predictions.

**Process:**
1. Iterates through each model in modeltime table
2. For each model, fits to every resample fold
3. Generates predictions on holdout sets
4. Returns table with `.resample_results` column containing nested predictions

**Control Parameters** (via `tune::control_resamples()`):
- `save_pred = TRUE` (always TRUE in modeltime.resample)
- `verbose = TRUE` - Show progress
- `extract = NULL` - Extract additional model components
- `save_workflow = FALSE` - Save fitted workflows

**Error Handling:**
- Uses `purrr::safely()` to catch errors
- Failed models return tibble with `.notes` column
- Sets seed deterministically per model (123 + model_id)

**Implementation Details:**
```r
# Internal workflow:
1. Check inputs (modeltime table, rset object)
2. Set control$save_pred = TRUE
3. Map over models with progress bar
4. For each model:
   - Call mdl_time_fit_resamples() S3 method
   - Handle workflows vs model_fit objects differently
   - Wrap with safely() for error handling
5. Return modeltime table with new column
```

**2. modeltime_resample_accuracy()**

```r
modeltime_resample_accuracy(
  object,                          # Modeltime table with resamples
  summary_fns = mean,              # Function(s) to aggregate
  metric_set = default_forecast_accuracy_metric_set(),
  ...                              # Additional args to summary_fns
)
```

**Purpose:** Calculate accuracy metrics for each resample, then optionally aggregate.

**Default Metrics:**
- MAE - Mean Absolute Error
- MAPE - Mean Absolute Percentage Error
- MASE - Mean Absolute Scaled Error
- SMAPE - Symmetric MAPE
- RMSE - Root Mean Squared Error
- RSQ - R-squared

**Summary Functions:**
- `NULL` - Returns unsummarized results (by .resample_id)
- `mean` (default) - Average across resamples
- `list(mean = mean, sd = sd)` - Multiple summary stats
- Custom lambda: `~ mean(.x, na.rm = TRUE)`

**Process:**
```r
1. Unnest .resample_results column
2. Identify target variable name
3. Group by model and resample
4. Apply metric_set to each group
5. If summary_fns provided:
   - Remove .resample_id
   - Group by model
   - Apply summary functions
6. Return accuracy table
```

**Output Formats:**

**Unsummarized** (`summary_fns = NULL`):
```r
# .model_id .model_desc .resample_id .type     mae mape  mase
# 1         ARIMA       Slice01      Resamples 100  5.2  0.89
# 1         ARIMA       Slice02      Resamples 95   4.8  0.85
```

**Summarized** (`summary_fns = mean`):
```r
# .model_id .model_desc .type     n   mae  mape  mase
# 1         ARIMA       Resamples 6   98   5.1   0.87
```

**Multi-summary** (`summary_fns = list(mean=mean, sd=sd)`):
```r
# .model_id .model_desc .type     n   mae_mean mae_sd mape_mean mape_sd
# 1         ARIMA       Resamples 6   98       12.3   5.1       0.8
```

**3. plot_modeltime_resamples()**

```r
plot_modeltime_resamples(
  .data,                    # Modeltime table with resamples
  .type = "boxplot",        # or "violin", "point"
  .summary_fns = NULL,      # Optional aggregation
  .smooth = FALSE,
  .facet_ncol = 1,
  .facet_scales = "free_y",
  .point_size = 2,
  .legend_show = TRUE,
  .title = "Resample Accuracy Plot",
  .x_lab = "",
  .y_lab = "",
  .color_lab = "Legend",
  .interactive = TRUE       # Plotly vs ggplot2
)
```

**Plot Types:**

**Boxplot** (`.type = "boxplot"`):
- Shows distribution of metrics across resamples
- One box per model per metric
- Identifies outlier resamples

**Violin** (`.type = "violin"`):
- Density plot of metric distributions
- Better for seeing shape of distribution

**Point** (`.type = "point"`):
- Individual resample points
- Optional smoothing via `.smooth = TRUE`
- Good for seeing trends across resamples

**Features:**
- Faceting by metric
- Color by model
- Interactive (plotly) or static (ggplot2)
- Consistent with modeltime visualization style

#### Advanced Features

**1. Panel Data / Nested Forecasting**

```r
# Works with grouped time series
nested_resamples <- nested_data %>%
  time_series_cv(
    assess = "8 weeks",
    initial = "1 year",
    skip = "4 weeks",
    slice_limit = 6
  )

# Fit models to all groups
models_resample <- models_tbl %>%
  modeltime_fit_resamples(nested_resamples)

# Accuracy by group and model
accuracy_tbl <- models_resample %>%
  modeltime_resample_accuracy(summary_fns = list(mean = mean, sd = sd))
```

**2. Integration with tune Package**

- Uses `tune::fit_resamples()` internally
- Compatible with `tune::control_resamples()`
- Supports parallel processing via future
- Works with rsample's rset objects

**3. Workflow Support**

```r
# Works with workflow objects
wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(model_spec)

models_tbl <- modeltime_table(wf)

# Fits the complete workflow to resamples
models_resample <- models_tbl %>%
  modeltime_fit_resamples(resamples)
```

**4. Progress Reporting**

```r
# Built-in progress reporting
models_resample <- models_tbl %>%
  modeltime_fit_resamples(
    resamples,
    control = control_resamples(verbose = TRUE)
  )

# Output:
# ═══ Fitting Resamples ════════════════════
#
# • Model ID: 1 ARIMA
# • Model ID: 2 Prophet
# • Model ID: 3 XGBoost
#
# 15.3 sec elapsed
```

#### Comparison: R modeltime.resample vs py-modeltime-resample

**✅ Implemented in py-modeltime-resample:**
- `modeltime_fit_resamples()` - Core resampling functionality
- `modeltime_resample_accuracy()` - Accuracy calculation
- `plot_modeltime_resamples()` - Visualization
- Summary statistics aggregation
- Interactive plotting (plotly)
- Static plotting (matplotlib/seaborn)
- Progress reporting
- Error handling

**✅ Python Enhancements (Not in R):**
- Parallel processing with joblib
- Pandas-native implementation
- More flexible visualization options
- Direct sklearn integration

**⚠️ Potential Gaps to Verify:**

1. **Workflow object support** - Does py version handle workflow-like objects?
2. **Panel data / nested forecasting** - Is grouped time series fully supported?
3. **Custom metric sets** - Can users define custom accuracy metrics?
4. **tune integration** - Is there equivalent to tune::control_resamples()?
5. **Extract parameter** - Can users extract model components during resampling?

**Recommendations for py-modeltime-resample:**

**Verify these features exist:**
```python
# 1. Custom metrics
custom_metrics = {
    'mae': mean_absolute_error,
    'rmse': root_mean_squared_error,
    'custom': my_custom_metric
}
accuracy = models_resample.modeltime_resample_accuracy(
    metric_set=custom_metrics
)

# 2. Workflow support (via py-workflows)
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import rand_forest

wf = workflow() \
    .add_recipe(my_recipe) \
    .add_model(rand_forest())

# Should work with resamples
models_tbl = modeltime_table(wf)
models_resample = models_tbl.modeltime_fit_resamples(resamples)

# 3. Panel data (grouped series)
# Should support group_by before resampling
models_resample = data \
    .groupby('store_id') \
    .modeltime_fit_resamples(resamples)
```

**Missing features to add:**
1. **Extract parameter** for pulling out model components
2. **Verbose control** for detailed progress reporting
3. **Save workflow** option for keeping fitted objects
4. **Professional accuracy tables** (gt/reactable equivalents in Python)

#### Python Integration Strategy

**Status:** py-modeltime-resample is largely complete.

**Action Items:**

**Priority 1 (Immediate):**
1. ✅ Verify custom metric support
2. ✅ Test with workflow objects (once py-workflows exists)
3. ✅ Ensure panel data / grouped series work correctly
4. ✅ Add any missing control parameters

**Priority 2 (Short-term):**
1. ✅ Create professional accuracy table formatting (pandas-styler or great-tables)
2. ✅ Add extract functionality for model inspection
3. ✅ Improve progress reporting with rich or tqdm enhancements
4. ✅ Document all comparison patterns with R version

**Priority 3 (Long-term):**
1. ✅ Contribute improvements back to R version if applicable
2. ✅ Create comprehensive tutorial comparing R and Python workflows
3. ✅ Add notebook examples for all use cases

---

## Updated Implementation Strategy

### Impact of pytimetk on Roadmap

**MAJOR CHANGE:** Since pytimetk exists and is production-ready, we can **accelerate** the py-tidymodels roadmap by:

1. **✅ Immediate Use** - Add pytimetk as core dependency
2. **✅ Skip Reimplementation** - Don't build py-timetk from scratch
3. **✅ Focus on Integration** - Build recipe steps that wrap pytimetk
4. **✅ Leverage Performance** - Use pytimetk's polars backend and GPU acceleration

### Revised Priority Recommendations

#### CRITICAL (Start Immediately):
1. **✅ py-recipes** - Recipe/step framework (wraps pytimetk, sklearn preprocessing)
2. **✅ pytimetk integration** - Recipe steps for time series features
3. **✅ py-parsnip** - Unified model interface
4. **✅ py-workflows** - Composition layer

#### HIGH (Months 1-3):
1. **✅ py-modeltime** - Core forecasting interface (wraps skforecast, statsmodels)
2. **✅ py-tune** - Hyperparameter optimization
3. **✅ py-yardstick** - Metrics library
4. **✅ py-rsample** - Resampling infrastructure (integrate with pytimetk CV)

#### MEDIUM (Months 4-6):
1. **✅ py-workflowsets** - Experimentation framework
2. **✅ py-stacks** - Ensembling
3. **✅ py-modeltime-ensemble** - Time series ensembles
4. **✅ py-dials** - Tuning parameters

#### LOW (Year 2):
1. **⚠️ py-filtro** - Feature selection (defer; use sklearn instead)
2. **✅ py-broom** - Model tidying (optional; pandas-friendly output)
3. **✅ Advanced integrations** - Deep learning, H2O AutoML

### Integration Architecture

```python
# Example: Complete py-tidymodels workflow with pytimetk
import pandas as pd
import pytimetk as tk
from py_recipes import recipe, step_timeseries_signature, step_lag, step_normalize
from py_parsnip import rand_forest
from py_workflows import workflow
from py_modeltime import modeltime_table, modeltime_calibrate, modeltime_forecast

# Load data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Create recipe using pytimetk functions
rec = recipe(df, formula='sales ~ .') \
    .step_timeseries_signature('date', engine='polars') \
    .step_lag('sales', lags=[1,7,14]) \
    .step_rolling('sales', window=[7,28], window_func=['mean','std']) \
    .step_holiday_signature('date', country='US') \
    .step_fourier('date', periods=[7, 365.25], max_order=5) \
    .step_normalize(all_predictors())

# Specify model
model = rand_forest(trees=500, mtry=10) \
    .set_engine('sklearn') \
    .set_mode('regression')

# Create workflow
wf = workflow() \
    .add_recipe(rec) \
    .add_model(model)

# Use modeltime interface
models_tbl = modeltime_table(
    wf,
    arima_reg().set_engine('statsmodels'),
    prophet_reg().set_engine('prophet')
)

# Calibrate and forecast
calibrated = models_tbl.modeltime_calibrate(test_data)
forecast = calibrated.modeltime_forecast(h=30)

# pytimetk visualization
tk.plot_timeseries(forecast, date_column='date', value_column='prediction')
```

### Key Design Decisions

**1. Use pytimetk directly, don't wrap it**
- pytimetk is well-designed and maintained
- Wrapping adds complexity without value
- Users can call pytimetk functions directly or via recipe steps

**2. Recipe steps delegate to pytimetk**
```python
# Recipe step wraps pytimetk function
class step_timeseries_signature:
    def bake(self, data):
        return tk.augment_timeseries_signature(
            data,
            date_column=self.date_column,
            engine='polars'
        )
```

**3. Feature store integration**
- Leverage pytimetk's feature store for caching
- Integrate with py-recipes prep/bake cycle
- Optional MLflow logging

**4. GPU acceleration pathway**
- Use pytimetk's GPU support where available
- Automatic fallback to CPU
- Document GPU setup for users

### Updated Recommendations

**For py-tidymodels Development:**

1. **✅ Add pytimetk dependency** immediately
2. **✅ Create recipe steps** that wrap pytimetk functions
3. **✅ Document integration patterns** extensively
4. **✅ Focus development** on workflow/parsnip/modeltime layers
5. **✅ Collaborate** with Business Science on integration
6. **✅ Contribute back** any needed pytimetk enhancements

**For filtro:**

1. **⚠️ Defer implementation** - Low priority
2. **✅ Document sklearn alternatives** for feature selection
3. **✅ Create helper functions** for sklearn.feature_selection in recipes context
4. **✅ Revisit** only if users specifically request filtro-style API

**For modeltime.resample:**

1. **✅ Verify completeness** of py-modeltime-resample
2. **✅ Add missing features** (custom metrics, extract, etc.)
3. **✅ Integrate** with pytimetk's TimeSeriesCV
4. **✅ Document** differences from R version

---

## Citations

[1] Business Science. "modeltime: The Tidymodels Extension for Time Series Modeling." R Package, Version 1.3.1. https://business-science.github.io/modeltime/

[2] Business Science. "timetk: A Tool Kit for Working with Time Series." R Package, Version 2.9.0. https://business-science.github.io/timetk/

[3] Kuhn, M. and Wickham, H. "tidymodels: Easily Install and Load the Tidymodels Packages." R Package. https://www.tidymodels.org/

[4] Kuhn, M. "recipes: Preprocessing and Feature Engineering Steps for Modeling." R Package, Version 1.3.1. https://recipes.tidymodels.org/

[5] Kuhn, M. and Vaughan, D. "parsnip: A Common API to Modeling and Analysis Functions." R Package, Version 1.3.3. https://parsnip.tidymodels.org/

[6] Kuhn, M. and Vaughan, D. "workflows: Modeling Workflows." R Package, Version 1.1.4. https://workflows.tidymodels.org/

[7] Kuhn, M. "tune: Tidy Tuning Tools." R Package, Version 1.3.0. https://tune.tidymodels.org/

[8] Frick, H. and Kuhn, M. "rsample: General Resampling Infrastructure." R Package, Version 1.2.1. https://rsample.tidymodels.org/

[9] Kuhn, M. and Vaughan, D. "yardstick: Tidy Characterizations of Model Performance." R Package, Version 1.3.2. https://yardstick.tidymodels.org/

[10] Kuhn, M. "dials: Tools for Creating Tuning Parameter Values." R Package. https://dials.tidymodels.org/

[11] Couch, S. and Kuhn, M. "workflowsets: Create a Collection of Tidymodels Workflows." R Package. https://workflowsets.tidymodels.org/

[12] Amat Rodrigo, J. and Escobar Ortiz, J. "skforecast: Time series forecasting with machine learning models." Python Package, Version 0.18.0, 2025. https://skforecast.org/0.18.0/

[13] Amat Rodrigo, J. "skforecast: Time series forecasting with machine learning models." GitHub Repository. https://github.com/skforecast/skforecast

[14] Lin, F., Kuhn, M., and Hvitfeldt, E. "filtro: Feature Selection Using Supervised Filter-Based Methods." R Package, Version 0.2.0.9000. https://github.com/tidymodels/filtro

[15] Business Science. "pytimetk: The Time Series Toolkit for Python." Python Package, Version 2.2.0. https://github.com/business-science/pytimetk

[16] Business Science. "modeltime.resample: Resampling Tools for Time Series Forecasting." R Package, Version 0.3.0.9000. https://github.com/business-science/modeltime.resample

---

**Report Updated:** October 26, 2025
**Research conducted by:** Claude (Anthropic AI Assistant)
**For:** py-tidymodels project development
