# Prompt Analysis and Recommendations
## Review of .claude/prompt.md Against Research Findings

**Date:** October 26, 2025
**Purpose:** Identify gaps, conflicts, and improvements needed in the project prompt

---

## 🚨 CRITICAL CONFLICTS IDENTIFIED

### 1. **modeltime Package Strategy CONFLICT**

**Current Prompt Says:**
> "modeltime - time series specific models and forecasting package - merge the functionality from this package into the others"

**Research Shows:**
- ❌ **DO NOT** implement `modeltime_table()` or `modeltime_calibrate()` patterns
- ❌ **DO NOT** create a separate modeltime package with table-based workflows
- ✅ **DO** integrate time series model specs into parsnip as extensions
- ✅ **DO** use workflows + workflowsets for model organization

**Required Prompt Update:**
```
4. parsnip with time series extensions - Add ARIMA, Prophet, ETS, etc.
   as parsnip model types (NOT a separate modeltime package)
   - arima_reg(), prophet_reg(), exp_smoothing(), seasonal_reg()
   - Engines: statsmodels, prophet, pmdarima, skforecast
   - Recursive forecasting wrapper: recursive()
   - DO NOT implement modeltime_table() or modeltime_calibrate() patterns
   - Use workflows + workflowsets for multi-model comparison instead
```

---

## 📋 MISSING CRITICAL CONTEXT

### 2. **pytimetk Package Discovery**

**Missing from Prompt:**
The research discovered that **pytimetk already exists** (v2.2.0, production-ready)!

**Impact:**
- Saves 2-3 months of development time
- 29 timeseries signature features already built
- GPU acceleration available
- 66 test files, professional quality

**Required Addition to Prompt:**
```
Time Series Feature Engineering:
- USE existing pytimetk package (v2.2.0) - DO NOT build from scratch
- Create recipe step wrappers around pytimetk functions:
  - step_lag() wraps pytimetk.augment_lags()
  - step_date() wraps pytimetk.augment_timeseries_signature()
  - step_holiday() wraps pytimetk.augment_holiday_signature()
  - step_fourier() wraps pytimetk.augment_fourier()
  - step_rolling() wraps pytimetk.augment_rolling()
- pytimetk provides: lags, rolling windows, date features, holidays,
  Fourier terms, differences, financial indicators
```

### 3. **skforecast Integration Strategy**

**Missing from Prompt:**
Research compared skforecast with tidymodels - hybrid approach recommended.

**Required Addition:**
```
skforecast Integration:
- USE skforecast forecasters as engines for parsnip models
- ForecasterRecursive → backend for recursive() wrapper
- ForecasterMultiSeries → backend for panel/grouped models
- Backtesting utilities → integrate with py-modeltime-resample
- DO NOT replicate skforecast's API - wrap it in tidymodels interface
- Leverage: lag optimization, hyperparameter tuning, multi-series support
```

### 4. **Grouped/Panel Time Series Details**

**Current Prompt Says:**
> "Fitting multiple models to each group of a grouped dataset"
> "Supporting global and panel models"

**Missing Specifics:**
- No details on how panel data should be structured
- No specification for nested vs. non-nested cross-validation
- No details on train/test splits preserving group structure

**Required Addition:**
```
Panel/Grouped Time Series Support:
1. Nested Forecasting (from modeltime.resample)
   - Fit separate model per group: group_split() → map models
   - Each group gets independent train/test split
   - Results include group_id column

2. Global Models (from skforecast)
   - Single model trained on all groups simultaneously
   - Group ID as exogenous variable or feature
   - Shared parameters across groups

3. Panel Data Structure:
   - Required columns: date_column, group_id, target, features
   - Ensure chronological ordering within groups
   - CV respects group boundaries (no leakage across groups)

4. Output Format for Grouped Models:
   - All DataFrames include 'group_id' column
   - model_outputs: stacked across all groups
   - coefficients: group_id if nested, global if single model
   - stats: one row per group (nested) or single row (global)
```

---

## 🎯 MISSING TECHNICAL SPECIFICATIONS

### 5. **Recursive Forecasting Infrastructure**

**Current Prompt Mentions:**
> "recursive models/forecasting"

**Missing Details:**
How recursive forecasting should work, what the API looks like.

**Required Addition:**
```
Recursive Forecasting (Critical for ML Models):
1. recursive() Wrapper Function:
   - Wraps any parsnip model to enable multi-step forecasting
   - Converts ML models (RF, XGBoost) to autoregressive forecasters

2. Implementation:
   from py_parsnip import rand_forest, recursive

   model = (
       rand_forest(trees=500)
       .set_engine("sklearn")
       .recursive(
           id="model_id",
           lags=7,  # Use last 7 observations
           h=30     # Forecast 30 steps ahead
       )
   )

3. Behavior:
   - Fit model on historical data with lag features
   - At prediction: use actual history for first step
   - For step 2+: use previous predictions as inputs
   - Return: multi-step forecasts with dates

4. Integration with skforecast:
   - Use ForecasterRecursive as backend implementation
   - Maintain tidymodels API on top
```

### 6. **Prediction Intervals and Uncertainty**

**Current Prompt:** No mention of prediction intervals.

**Research Shows:** modeltime supports conf_int and pred_int.

**Required Addition:**
```
Prediction Intervals (Uncertainty Quantification):
1. All time series models must support:
   - Point forecasts (default)
   - Confidence intervals (model uncertainty)
   - Prediction intervals (forecast uncertainty)

2. API:
   forecast = fitted_model.predict(
       new_data=future_data,
       type="pred_int",
       level=0.95  # 95% prediction intervals
   )

3. Output Columns:
   - .pred: point forecast
   - .pred_lower: lower bound
   - .pred_upper: upper bound

4. Methods by Model Type:
   - ARIMA: analytical intervals from statsmodels
   - Prophet: built-in uncertainty
   - ML models: conformal prediction or bootstrapping
```

### 7. **Exogenous Variables / External Regressors**

**Current Prompt:** No mention of exogenous variables.

**Research Shows:** Critical for time series models (ARIMA, Prophet).

**Required Addition:**
```
Exogenous Variables (External Regressors):
1. Support in Time Series Models:
   - ARIMA with regressors (ARIMAX/SARIMAX)
   - Prophet with additional regressors
   - ML models naturally support (via recipe features)

2. API in Recipe:
   recipe_spec = (
       recipe("sales ~ date + price + promotion", data=train)
       .update_role("date", new_role="time_index")
       .update_role("price", "promotion", new_role="exogenous")
       .step_date("date", features=["dow", "month"])
   )

3. Forecasting Requirement:
   - User must provide future values of exogenous variables
   - Error if exogenous vars not in new_data during prediction

4. Documentation:
   - Clear examples for each model type
   - Guidance on which variables can be exogenous
```

---

## 📊 OUTPUT FORMAT CLARIFICATIONS

### 8. **Output DataFrame Specifications - Need More Detail**

**Current Prompt Says:**
> "model_outputs, coefficients and stats as dataframes"

**Missing:**
- Exact column specifications
- How to handle models with different output types
- Nested model outputs (grouped data)
- Time series specific columns

**Required Enhancement:**
```
Standard Output DataFrames (ALL models must conform):

1. model_outputs DataFrame:
   Columns (time series):
   - model_id: str - unique identifier
   - group_id: str - for panel data (NULL if ungrouped)
   - date: datetime - observation timestamp
   - actual: float - true target value
   - fitted: float - in-sample predictions (train)
   - predicted: float - out-of-sample predictions (test/forecast)
   - residual: float - actual - (fitted or predicted)
   - split: str - 'train', 'test', or 'forecast'
   - .pred_lower: float - lower prediction interval (optional)
   - .pred_upper: float - upper prediction interval (optional)

   Index: [model_id, group_id, date]
   Sort: model_id, group_id, date ascending

2. coefficients DataFrame:
   Columns:
   - model_id: str
   - group_id: str (for nested models)
   - parameter: str - name (coef name or hyperparameter)
   - value: float - parameter value
   - std_error: float - standard error (if statistical model)
   - t_statistic: float - t-stat (if available)
   - p_value: float - significance (if available)
   - term: str - original feature name
   - parameter_type: str - 'coefficient', 'hyperparameter', 'feature_importance'

   Note: For ML models without coefficients:
   - Return hyperparameters (trees, learning_rate, etc.)
   - Return feature_importances if available

3. stats DataFrame:
   Columns:
   - model_id: str
   - group_id: str (for nested models)
   - metric: str - metric name
   - value: float - metric value
   - split: str - 'train', 'test', 'overall'

   Metrics to include:
   - Fit statistics: rmse, mae, mape, smape, mase, r_squared
   - Model info: aic, bic, log_likelihood (if available)
   - Residual tests: durbin_watson, ljung_box_p, shapiro_wilk_p
   - Sample sizes: n_train, n_test, n_forecast
   - Timing: fit_time_seconds

4. Stacking Rules:
   - Multiple models: concatenate with different model_id
   - Schema never changes - consistent across all model types
   - Missing values: NULL/NaN for unavailable metrics
   - All DataFrames are "tidy" - one observation per row
```

---

## 🏗️ ARCHITECTURE CLARIFICATIONS

### 9. **Workflow Priority Over Model Tables**

**Current Prompt:** Doesn't emphasize workflows enough.

**Research Shows:** Workflows + workflowsets are the core organizing principle.

**Required Addition:**
```
Core Architecture Principle:
workflows and workflowsets are the PRIMARY tools for model organization.

1. Single Model Workflow:
   wf = (
       workflow()
       .add_recipe(preprocessing_recipe)
       .add_model(model_spec)
   )
   fitted_wf = wf.fit(train_data)

2. Multiple Models (Use workflowsets):
   wf_set = workflow_set(
       preproc=[recipe1, recipe2],
       models=[model1, model2, model3],
       cross=True  # All combinations
   )
   results = wf_set.fit_resamples(cv_splits)

3. DO NOT implement these patterns:
   ❌ modeltime_table() - table-based model organization
   ❌ modeltime_calibrate() - separate calibration phase
   ❌ Custom model registry separate from workflows

4. Workflow Benefits:
   - Automatic preprocessing at prediction time
   - Tune integration for hyperparameter optimization
   - Parallel processing via tune::fit_resamples
   - Consistent API for all models
```

### 10. **filtro Package - Deprioritize**

**Current Prompt Says:**
> "filtro - feature selection recipes and preprocessors"

**Research Shows:** filtro is experimental, sklearn.feature_selection is sufficient.

**Required Update:**
```
5. Feature Selection (filtro) - DEFER TO LATER PHASE
   - Initial release: Use sklearn.feature_selection directly
   - filtro provides 11 methods (ANOVA, RF importance, etc.)
   - Modern S7 architecture (experimental in R)
   - Recommendation: Only implement if users specifically request
   - Priority: LOW (Tier 4)
   - Alternative: Wrap sklearn in recipe steps if needed
```

---

## 🔧 IMPLEMENTATION DETAILS NEEDED

### 11. **Engine Registration System**

**Current Prompt:** Mentions "Engine abstraction layer" but no details.

**Required Addition:**
```
Engine Registration System (from parsnip):

1. Architecture:
   - Engines are backends (sklearn, statsmodels, xgboost, etc.)
   - Each model type supports multiple engines
   - Engine translates unified params to engine-specific params

2. Example:
   # User code (engine-agnostic)
   model = rand_forest(trees=500, mtry=10).set_engine("sklearn")

   # Behind the scenes translation:
   sklearn: n_estimators=500, max_features=10
   ranger:  num.trees=500, mtry=10

3. Registration API:
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

4. Key Engines to Implement:
   Time Series:
   - statsmodels: ARIMA, SARIMAX, ETS
   - prophet: Prophet
   - pmdarima: auto_arima
   - skforecast: Recursive forecasters

   ML (General):
   - sklearn: All sklearn models
   - xgboost: XGBoost
   - lightgbm: LightGBM
   - catboost: CatBoost
```

### 12. **Formula Interface Details**

**Current Prompt Says:**
> "Formula interface with patsy"

**Research Shows:** Tidymodels uses formulas heavily, need more detail.

**Required Addition:**
```
Formula Interface (Patsy Integration):

1. R-style Formulas:
   # Basic
   "y ~ x1 + x2 + x3"

   # Time series
   "sales ~ date + price + promotion"

   # All predictors except
   "y ~ . - id - date"

   # Interactions
   "y ~ x1 * x2"  # x1 + x2 + x1:x2

2. Time Series Special Handling:
   - Date column should NOT be treated as numeric predictor
   - Update role: recipe(...).update_role("date", new_role="time_index")
   - Extract date features with step_date() instead

3. Grouped Models:
   "sales ~ date | store_id"  # Nested by store
   "sales ~ date + price | region + category"  # Nested by region×category

4. Integration:
   - recipe() accepts formula or DataFrame
   - Model fit() accepts formula
   - Consistent formula handling across all components
```

### 13. **Visualization Specifications**

**Current Prompt Says:**
> "Create visualisation helpers that create interactive plotly plots"

**Missing:** Specific plot types needed.

**Required Addition:**
```
Visualization Helpers (Plotly Interactive):

1. Required Plot Types:

   a) Time Series Forecast Plot:
      - Actual values (historical)
      - Fitted values (in-sample)
      - Predictions (out-of-sample)
      - Forecast (future)
      - Prediction intervals (shaded region)
      - Train/test split line
      - Multiple models overlaid (different colors)

   b) Residual Diagnostics:
      - Residuals vs fitted
      - Residuals vs time (for time series)
      - QQ plot (normality)
      - ACF/PACF plots (autocorrelation)
      - Histogram of residuals

   c) Model Comparison:
      - Metric comparison (bar chart, lower is better)
      - Forecast accuracy by horizon (line chart)
      - Metric heatmap (models × metrics)
      - Rank results table

   d) Workflow/Workflowset Results:
      - Autoplot for tune_results
      - Metric distribution across CV folds
      - Parameter vs metric scatter
      - Best parameters by model

2. API:
   from py_tidymodels.viz import (
       plot_forecast,
       plot_residuals,
       plot_model_comparison,
       plot_tune_results
   )

   fig = plot_forecast(
       fitted_wf,
       actual_data=train,
       forecast_data=test,
       show_intervals=True,
       title="Sales Forecast"
   )
   fig.show()

3. Dashboard Integration:
   - All plots should return plotly Figure objects
   - Compatible with Dash dashboard layout
   - Export to HTML for sharing
```

---

## 🎯 PRIORITY AND PHASING

### 14. **Implementation Phases - Update Needed**

**Current Prompt:** Asks for a project plan but doesn't specify phases.

**Research Provides:** Clear 5-phase roadmap.

**Required Addition:**
```
Implementation Phases (from Research):

Phase 1: CRITICAL Foundation (Months 1-4)
- py-hardhat: mold/forge abstractions
- py-rsample: Enhance existing py-modeltime-resample
- py-parsnip: Model specs + engines (sklearn, statsmodels)
- py-parsnip time series extensions: arima_reg, prophet_reg
- py-workflows: Recipe + model composition
Deliverable: Can fit single models with preprocessing and CV

Phase 2: Scale and Evaluate (Months 5-8)
- py-recipes: Step wrappers around pytimetk (use existing package!)
- py-tune: Hyperparameter optimization
- py-yardstick: Metrics including MASE
- py-workflowsets: Multi-model comparison (replaces modeltime_table)
Deliverable: Can run 100+ model configs with tuning

Phase 3: Advanced Features (Months 9-11)
- py-stacks: Ensembling (replaces modeltime.ensemble)
- py-dials: Parameter grids
- Recursive forecasting wrapper
- skforecast integration
Deliverable: Ensembles and recursive ML forecasting

Phase 4: Polish and Extend (Month 12+)
- Additional engines (lightgbm, catboost)
- Dashboard (Dash + Plotly)
- MLflow integration
- filtro wrapper (if requested)
Deliverable: Production-ready with dashboard

DO NOT IMPLEMENT:
- modeltime_table/calibrate infrastructure
- Separate py-timetk (use pytimetk)
- modeltime.ensemble (use stacks)
```

---

## 📝 SUMMARY OF REQUIRED PROMPT UPDATES

### Critical Changes (MUST Update):

1. ❌ Remove "modeltime as separate package" → ✅ "time series extensions to parsnip"
2. ❌ Remove "merge modeltime functionality" → ✅ "avoid modeltime_table pattern, use workflows"
3. ✅ Add "USE pytimetk package - DO NOT build from scratch"
4. ✅ Add "workflowsets is primary multi-model tool"
5. ✅ Add "skforecast integration strategy"

### Important Additions (SHOULD Add):

6. ✅ Detailed output DataFrame schemas with exact columns
7. ✅ Panel/grouped time series specifications
8. ✅ Recursive forecasting API and behavior
9. ✅ Prediction intervals requirements
10. ✅ Exogenous variables handling
11. ✅ Engine registration system details
12. ✅ Visualization specifications with plot types
13. ✅ Formula interface details for time series
14. ✅ Clear 4-phase implementation timeline

### Clarifications (NICE TO HAVE):

15. ✅ filtro priority → DEFER (use sklearn)
16. ✅ Workflow-first architecture principle
17. ✅ GPU acceleration via pytimetk
18. ✅ MLflow integration details
19. ✅ Dashboard specifications

---

## 🔄 RECOMMENDED NEW PROMPT STRUCTURE

```markdown
# py-tidymodels: Python Port of R Tidymodels Ecosystem

## Project Vision
Create a unified Python modeling framework for time series regression and
forecasting, based on R's tidymodels ecosystem but avoiding its limitations.

## Core Principles
1. workflows + workflowsets for model organization (NOT modeltime_table)
2. Integrate time series into parsnip (NOT separate package)
3. Leverage existing packages (pytimetk, skforecast)
4. Standardized tidy DataFrames for all outputs
5. Scale to 100+ model configurations efficiently

## [Detailed sections from recommendations above...]
```

---

## Next Steps

1. **Update .claude/prompt.md** with these recommendations
2. **Create projectplan.md** incorporating:
   - 4-phase timeline
   - Avoid modeltime_table tasks
   - pytimetk integration tasks
   - Standardized output format tasks
3. **Define model_outputs/ example schemas** with exact columns
4. **Review and approve** updated prompt before beginning implementation
