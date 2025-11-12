# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**py-tidymodels** is a Python port of R's tidymodels ecosystem focused on time series regression and forecasting. The project implements a unified, composable interface for machine learning models with emphasis on clean architecture patterns and extensibility.

## Development Environment

### Virtual Environment and Package Installation
The project uses a virtual environment at `py-tidymodels2/`:
```bash
source py-tidymodels2/bin/activate
```

**Important:** The package is installed in editable mode for development:
```bash
pip install -e .
```

This allows changes to source code to be immediately available in Jupyter notebooks without reinstalling. However, **you must restart the Jupyter kernel** after making changes to see updates.

For Jupyter to use the correct kernel:
```bash
python -m ipykernel install --user --name=py-tidymodels2
```

### Python Bytecode Cache Management
**CRITICAL:** When making code changes, clear bytecode cache to ensure updates load properly:

```bash
# Clear all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Force reinstall package
pip install -e . --force-reinstall --no-deps
```

**After package updates, ALWAYS restart Jupyter kernel** before running notebooks:
- In Jupyter: **Kernel** → **Restart & Clear Output**
- Re-run cells from beginning

### Running Tests
```bash
# Activate venv first
source py-tidymodels2/bin/activate

# All tests (762+ tests passing as of 2025-11-09)
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_hardhat/test_mold_forge.py -v
python -m pytest tests/test_parsnip/test_linear_reg.py -v
python -m pytest tests/test_recipes/test_recipe.py -v
python -m pytest tests/test_yardstick/test_metrics.py -v
python -m pytest tests/test_tune/test_tune.py -v

# With coverage
python -m pytest tests/ --cov=py_hardhat --cov=py_parsnip --cov=py_recipes --cov=py_yardstick --cov=py_tune --cov-report=html

# Get exact test count
python -m pytest tests/ --collect-only -q | tail -1
```

### Testing Notebooks
```bash
# Test individual notebook execution
cd "/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels"
source py-tidymodels2/bin/activate
jupyter nbconvert --to notebook --execute examples/16_baseline_models_demo.ipynb \
  --output /tmp/16_test.ipynb --ExecutePreprocessor.timeout=600

# Clear notebook outputs before testing (prevents cached execution issues)
jupyter nbconvert --clear-output --inplace examples/18_sklearn_regression_demo.ipynb

# Run all notebooks in sequence (useful for regression testing)
for nb in examples/{01..21}_*.ipynb; do
    echo "Testing $nb..."
    jupyter nbconvert --to notebook --execute "$nb" \
      --output "/tmp/$(basename $nb)" --ExecutePreprocessor.timeout=900 2>&1 | head -50
done
```

### Running Examples
```bash
# Activate venv first
source py-tidymodels2/bin/activate

# Launch Jupyter
jupyter notebook

# Then open notebooks in examples/
# - 01_hardhat_demo.ipynb - Data preprocessing with mold/forge
# - 02_parsnip_demo.ipynb - Linear regression with sklearn
# - 03_time_series_models.ipynb - Prophet and ARIMA models
# - 04_rand_forest_demo.ipynb - Random Forest (regression & classification)
# - 05_recipes_comprehensive_demo.ipynb - Feature engineering with recipes (51 steps)
# - 09_yardstick_demo.ipynb - Model evaluation metrics (17 metrics)
# - 10_tune_demo.ipynb - Hyperparameter tuning with grid search
```

## Architecture Overview

The project follows a layered architecture inspired by R's tidymodels:

### Layer 1: py-hardhat (Data Preprocessing)
**Purpose:** Low-level data preprocessing abstraction that ensures consistent transformations between training and prediction.

**Key Concepts:**
- **Blueprint**: Immutable preprocessing metadata (formula, factor levels, column order)
- **MoldedData**: Preprocessed data ready for modeling (predictors, outcomes, extras)
- **mold()**: Formula → model matrix conversion (training phase)
- **forge()**: Apply blueprint to new data (prediction phase)

**Critical Implementation Detail:**
- Uses patsy for R-style formula parsing (`y ~ x1 + x2`)
- Enforces categorical factor levels (errors on unseen levels)
- Blueprint is serializable for model persistence

**Files:**
- `py_hardhat/blueprint.py` - Blueprint dataclass
- `py_hardhat/mold_forge.py` - mold() and forge() functions
- `tests/test_hardhat/` - 14 tests, all passing

### Layer 2: py-parsnip (Model Interface)
**Purpose:** Unified model specification interface with pluggable engine backends.

**Key Design Patterns:**

1. **Immutable Specifications**: ModelSpec is frozen dataclass
   - Prevents side effects when reusing specs
   - Use `replace()` or `set_*()` methods to modify
   - Example: `spec.set_args(penalty=0.1)`

2. **Registry-Based Engines**: Decorator pattern for engine registration
   ```python
   @register_engine("linear_reg", "sklearn")
   class SklearnLinearEngine(Engine):
       ...
   ```
   - Allows multiple backends per model type
   - Runtime engine discovery via `get_engine(model_type, engine)`

3. **Dual-Path Preprocessing**: Standard vs Raw data handling
   - **Standard path**: mold() → fit() → forge() → predict()
     - Used by: linear_reg with sklearn
   - **Raw path**: fit_raw() → predict_raw()
     - Used by: prophet_reg, arima_reg (bypass hardhat due to datetime issues)
   - Engine indicates path via `hasattr(engine, "fit_raw")`

4. **Standardized Outputs**: Three-DataFrame pattern
   - `extract_outputs()` returns: `(outputs, coefficients, stats)`
   - **outputs**: Observation-level results (actuals, fitted, forecast, residuals, split)
   - **coefficients**: Model parameters with statistical inference (std_error, t_stat, p_value, CI, VIF)
     - For tree models: feature importances instead of coefficients
     - For Prophet/ARIMA: hyperparameters as "coefficients"
   - **stats**: Model-level metrics by split (RMSE, MAE, R², etc.) + residual diagnostics
   - All DataFrames include: `model`, `model_group_name`, `group` columns for multi-model tracking
   - Consistent across all model types
   - Inspired by R's broom package (`tidy()`, `glance()`, `augment()`)

**Implemented Models (23 Total):**

**Baseline Models (2):**
- `null_model()` - Mean/median baseline (strategy: mean, median, last)
- `naive_reg()` - Time series baselines (strategy: naive, seasonal_naive, drift, window)

**Linear & Generalized Models (3):**
- `linear_reg()` - Linear regression (sklearn, statsmodels engines)
- `poisson_reg()` - Poisson regression for count data
- `gen_additive_mod()` - Generalized Additive Models (pygam)

**Tree-Based Models (2):**
- `decision_tree()` - Single decision trees (sklearn)
- `rand_forest()` - Random forests (sklearn)

**Gradient Boosting (3 engines):**
- `boost_tree()` - XGBoost, LightGBM, CatBoost engines

**Support Vector Machines (2):**
- `svm_rbf()` - RBF kernel SVM
- `svm_linear()` - Linear kernel SVM

**Instance-Based & Adaptive (3):**
- `nearest_neighbor()` - k-NN regression
- `mars()` - Multivariate Adaptive Regression Splines
- `mlp()` - Multi-layer perceptron neural network

**Time Series Models (5):**
- `arima_reg()` - ARIMA/SARIMAX (statsmodels, auto_arima engines)
- `prophet_reg()` - Facebook Prophet
- `exp_smoothing()` - Exponential smoothing / ETS
- `seasonal_reg()` - STL decomposition models
- `varmax_reg()` - Multivariate VARMAX (statsmodels) - requires 2+ outcome variables

**Hybrid Time Series (2):**
- `arima_boost()` - ARIMA + XGBoost
- `prophet_boost()` - Prophet + XGBoost

**Recursive Forecasting (1):**
- `recursive_reg()` - ML models for multi-step forecasting (skforecast)

**Generic Hybrid Models (1):**
- `hybrid_model()` - Combines any two models with four strategies:
  - **residual** (default): Train model2 on residuals from model1
  - **sequential**: Different models for different time periods (regime changes)
  - **weighted**: Weighted combination of predictions (ensembling)
  - **custom_data** (NEW): Train models on different/overlapping datasets
    - Use dict input: `fit({'model1': early_data, 'model2': later_data}, formula)`
    - Supports overlapping or non-overlapping training periods
    - Flexible blending: weighted, avg, model1, or model2
    - Ideal for adaptive learning with changing data distributions

**Manual Coefficient Models (1):**
- `manual_reg()` - User-specified coefficients (no fitting)
  - Useful for comparing with external forecasts (Excel, R, SAS, etc.)
  - Incorporating domain expert knowledge
  - Creating baselines for benchmarking

**Parameter Translation:**
- Tidymodels naming → Engine-specific naming
- Example: `penalty` → `alpha` (sklearn), `non_seasonal_ar` → `p` (statsmodels)
- Handled via `param_map` dict in each engine

**Grouped/Panel Modeling on ModelSpec (NEW):**

ModelSpec now supports grouped/panel modeling directly without requiring workflow wrappers:

- **`fit_nested(data, formula, group_col)`**: Fit separate models per group
  ```python
  spec = linear_reg()
  nested_fit = spec.fit_nested(data, "sales ~ price", group_col="store_id")
  predictions = nested_fit.predict(test_data)
  outputs, coeffs, stats = nested_fit.extract_outputs()
  ```
  - Returns `NestedModelFit` with unified interface
  - Handles recursive models with date indexing automatically
  - Group column automatically added to all outputs
  - Simpler API: 2 lines instead of 3 (no workflow creation needed)

- **`fit_global(data, formula, group_col)`**: Fit single model using group as feature
  ```python
  spec = linear_reg()
  global_fit = spec.fit_global(data, "sales ~ price", group_col="store_id")
  ```
  - Automatically adds group column to formula
  - Returns standard `ModelFit` object
  - Best when groups share similar patterns

**When to Use:**
- Use `spec.fit_nested()` for formula-only grouped modeling (simplest)
- Use `workflow().fit_nested()` when using recipes for preprocessing
- Both approaches produce identical results

**Files:**
- `py_parsnip/model_spec.py` - ModelSpec and ModelFit classes, NestedModelFit class
- `py_parsnip/engine_registry.py` - Engine ABC and registry
- `py_parsnip/models/` - Model specification functions
- `py_parsnip/engines/` - Engine implementations
- `tests/test_parsnip/` - 22+ tests passing
- `tests/test_parsnip/test_nested_model_fit.py` - 21 tests for grouped modeling on ModelSpec

### Layer 3: py-rsample (Resampling)
**Purpose:** Train/test splitting and cross-validation for time series and general data.

**Key Functions:**
- **initial_time_split()**: Chronological train/test split with period parsing
- **time_series_cv()**: Rolling/expanding window cross-validation
- **time_series_nested_cv()**: Per-group CV for nested modeling (NEW - 2025-11-12)
  - Creates per-group CV splits automatically
  - Each group gets its own independent CV splits
  - Returns dict mapping group names → TimeSeriesCV objects
  - Designed for use with `WorkflowSet.fit_nested_resamples()`
  - Example:
    ```python
    cv_folds = time_series_nested_cv(
        data=train_data,
        group_col='country',
        date_column='date',
        initial='18 months',
        assess='3 months',
        skip='2 months',
        cumulative=False
    )
    # Returns: {'USA': cv_usa, 'Germany': cv_germany, ...}
    # Each group has different CV splits based on that group's data
    ```
- **time_series_global_cv()**: Global CV for global modeling (NEW - 2025-11-12)
  - Creates CV splits on FULL dataset (not per-group)
  - All groups share the same CV splits
  - Returns dict mapping group names → same TimeSeriesCV object
  - Designed for use with `WorkflowSet.fit_global_resamples()`
  - Example:
    ```python
    cv_folds = time_series_global_cv(
        data=train_data,
        group_col='country',
        date_column='date',
        initial='18 months',
        assess='3 months',
        skip='2 months',
        cumulative=False
    )
    # Returns: {'USA': cv_global, 'Germany': cv_global, ...}
    # All groups share the same CV object (same reference)
    ```
- **vfold_cv()**: Standard k-fold cross-validation with stratification support

**Features:**
- Period parsing: "2 years", "6 months", etc.
- Explicit date ranges (absolute, relative, mixed)
- Stratified sampling for classification
- Repeated CV support
- Group-aware CV for panel/grouped data

**Files:**
- `py_rsample/initial_split.py` - Initial time-based splits
- `py_rsample/time_series_cv.py` - Time series CV, nested CV, and global CV
- `py_rsample/vfold_cv.py` - K-fold CV
- `py_rsample/split.py` - Split and RSplit classes
- `tests/test_rsample/` - 45+ tests passing (35 existing + 6 nested CV + 4 global CV)

### Layer 4: py-workflows (Pipelines)
**Purpose:** Compose preprocessing + model + postprocessing into unified workflow.

**Key Classes:**
- **Workflow**: Immutable workflow specification
- **WorkflowFit**: Fitted workflow with predictions
- **NestedWorkflowFit**: Fitted workflow with per-group models for panel/grouped data

**Key Methods:**
- `add_formula()`, `add_model()` - Compose workflow
- `fit()` - Train workflow
- `fit_nested(data, group_col)` - Fit separate models for each group (panel/grouped modeling)
- `fit_global(data, group_col)` - Fit single model with group as a feature
- `predict()` - Make predictions with automatic preprocessing
- `evaluate()` - Train/test evaluation
- `extract_outputs()` - Get three-DataFrame outputs (includes group column for nested models)
- `extract_fit_parsnip()` - Get underlying ModelFit object
- `extract_preprocessor()` - Get fitted preprocessor (formula or PreparedRecipe)
- `extract_spec_parsnip()` - Get model specification
- `extract_formula()` - **NEW:** Get formula used for model fitting
- `extract_preprocessed_data(data)` - **NEW:** Apply preprocessing to data, return transformed DataFrame

**Panel/Grouped Modeling:**
- **Nested approach** (`fit_nested()`): Fit independent model per group
  - Best when groups have different patterns (e.g., premium vs budget stores)
  - Each group gets its own parameters
  - Returns `NestedWorkflowFit` with unified predict/evaluate interface
- **Global approach** (`fit_global()`): Fit single model with group as feature
  - Best when groups share similar patterns
  - More efficient with limited data per group
  - Returns standard `WorkflowFit`
- Works with any model type including `recursive_reg` for per-group forecasting

**Per-Group Preprocessing (NEW - 2025-11-10):**
- **Feature**: Each group can have its own recipe preprocessing
- **Use Case**: PCA, feature selection, or filtering where groups need different feature spaces
- **Usage**:
  ```python
  # Each group gets its own PreparedRecipe fitted on that group's data
  nested_fit = wf.fit_nested(data, group_col='country', per_group_prep=True)

  # Compare features across groups
  comparison = nested_fit.get_feature_comparison()
  # Returns DataFrame showing which features each group uses
  ```
- **Parameters**:
  - `per_group_prep=True`: Enable per-group recipe preparation (default: False)
  - `min_group_size=30`: Minimum samples for group-specific prep; smaller groups use global recipe
- **Key Features**:
  - Automatic outcome column preservation during recipe prep
  - Small group fallback to global recipe with warning
  - Error handling for new/unseen groups at prediction time
  - `get_feature_comparison()` utility shows feature differences across groups
- **Benefits**:
  - Groups with different data distributions get appropriate preprocessing
  - USA refineries: 5 PCA components; UK refineries: 3 components (example)
  - Feature selection selects different features per group based on group-specific importance
- **Code References**:
  - `py_workflows/workflow.py:121-179` - Outcome preservation helpers
  - `py_workflows/workflow.py:255-311` - fit_nested() with per_group_prep parameter
  - `py_workflows/workflow.py:392-543` - Per-group recipe prep and fitting logic
  - `py_workflows/workflow.py:1023-1113` - get_feature_comparison() method
  - `tests/test_workflows/test_per_group_prep.py` - Comprehensive tests (5 tests, all passing)
  - `.claude_debugging/PER_GROUP_PREPROCESSING_IMPLEMENTATION.md` - Full implementation documentation

**Files:**
- `py_workflows/workflow.py` - Workflow, WorkflowFit, and NestedWorkflowFit classes
- `tests/test_workflows/` - 64 tests passing (26 general + 13 panel models + 5 per-group prep + 20 other)

### Layer 5: py-recipes (Feature Engineering)
**Purpose:** Advanced feature preprocessing and engineering pipeline.

**Key Components:**
- **Recipe**: Immutable specification of preprocessing steps
- **51 preprocessing steps** across 8 categories:
  - Imputation (median, mean, mode, KNN, bag, linear)
  - Normalization (normalize, range, center, scale)
  - Encoding (dummy, one-hot, target, ordinal, bin, date)
  - Feature engineering (polynomial, interactions, splines, PCA, ICA, kernel PCA, PLS, log, sqrt, BoxCox, YeoJohnson)
  - Filtering (correlation, variance, missing, outliers, zero-variance)
  - Row operations (sample, filter, slice, arrange, shuffle)
  - Transformations (mutate, discretize)
  - Selectors (all_predictors, all_outcomes, all_numeric, all_nominal, has_role, has_type)

**Recent Recipe Enhancements (2025-11-09):**
- **Datetime exclusion**: `step_dummy()` and discretization steps (`step_discretize()`, `step_cut()`) automatically exclude datetime columns to prevent formula parsing errors
- **Infinity handling**: `step_naomit()` now removes rows with both NaN and ±Inf values
- **Selector support for reduction steps**: `step_ica()`, `step_kpca()`, `step_pls()` now support selector functions like `all_numeric_predictors()`
- **step_corr() removed**: Use `step_select_corr(method='multicollinearity')` instead for the same functionality plus more options

**Key Pattern:**
```python
recipe = (recipe(data, "y ~ x1 + x2")
    .step_impute_median(all_numeric())
    .step_normalize(all_numeric())
    .step_dummy(all_nominal()))
prepped = recipe.prep()
processed = prepped.bake(new_data)
```

**Files:**
- `py_recipes/recipe.py` - Recipe class and core methods
- `py_recipes/steps/` - 51 step implementations
- `tests/test_recipes/` - 265 tests passing

### Layer 6: py-yardstick (Model Metrics)
**Purpose:** Comprehensive model evaluation metrics.

**17 Metrics Implemented:**
- **Regression**: RMSE, MAE, MAPE, SMAPE, R², adjusted R², RSE
- **Classification**: Accuracy, Precision, Recall, F1, Specificity, Balanced Accuracy, MCC, AUC-ROC, Log Loss, Brier Score

**Key Functions:**
- Individual metrics: `rmse()`, `mae()`, `r_squared()`, etc.
- **metric_set()**: Combine multiple metrics
- Supports both 2-column (truth, estimate) and full DataFrame inputs

**Files:**
- `py_yardstick/metrics.py` - All 17 metrics + metric_set()
- `tests/test_yardstick/` - 59 tests passing

### Layer 7: py-tune (Hyperparameter Tuning)
**Purpose:** Grid search and hyperparameter optimization.

**Key Functions:**
- **tune()**: Mark parameters for tuning with `tune("param_name")`
- **grid_regular()**: Create evenly-spaced parameter grids
- **grid_random()**: Random parameter sampling
- **tune_grid()**: Grid search with cross-validation
- **fit_resamples()**: Evaluate without tuning
- **TuneResults**: Result analysis with `show_best()`, `select_best()`, `select_by_one_std_err()`
- **finalize_workflow()**: Apply best parameters to workflow

**Key Pattern:**
```python
spec = linear_reg(penalty=tune(), mixture=tune())
grid = grid_regular({"penalty": {"range": (0.001, 1.0), "trans": "log"},
                     "mixture": {"range": (0, 1)}}, levels=5)
results = tune_grid(workflow, resamples, grid=grid, metrics=metric_set(rmse, r_squared))
best = results.select_best("rmse", maximize=False)
final_wf = finalize_workflow(workflow, best)
```

**Data Format Handling:**
- TuneResults methods handle **both long-format** (metric/value columns) and **wide-format** (metrics as columns)
- Automatically detects format by checking for 'metric' column
- Long format: returned by `tune_grid()` and `fit_resamples()`
- Wide format: used in some test mocks

**Files:**
- `py_tune/tune.py` - All tuning functions and TuneResults class
- `tests/test_tune/` - 36 tests passing

### Layer 8: py-workflowsets (Multi-Model Comparison)
**Purpose:** Efficiently compare multiple workflows across different preprocessing strategies and models.

**Key Classes:**
- **WorkflowSet**: Collection of workflows for comparison
- **WorkflowSetResults**: Results from evaluating all workflows (via CV)
- **WorkflowSetNestedResults**: Results from grouped/panel modeling (NEW - 2025-11-11)

**Key Methods:**
- **WorkflowSet.from_cross()**: Create all combinations of preprocessors × models
  - Example: 5 formulas × 4 models = 20 workflows
- **WorkflowSet.from_workflows()**: Create from explicit workflow list
- **fit_resamples()**: Evaluate all workflows across CV folds
- **fit_nested()**: Fit all workflows across all groups (panel/grouped data)
- **fit_global()**: Fit all workflows globally with group as feature
- **collect_metrics()**: Aggregate metrics across resamples or groups
- **rank_results()**: Rank workflows by performance
- **autoplot()**: Automatic visualization of results

**Standard Workflow Comparison (CV-based):**
```python
# Define multiple preprocessing strategies
formulas = [
    "y ~ x1 + x2",
    "y ~ x1 + x2 + x3",
    "y ~ x1 + x2 + I(x1*x2)",  # Interaction
]

# Define multiple model specs
models = [
    linear_reg(),
    linear_reg(penalty=0.1, mixture=1.0),  # Lasso
]

# Create all combinations (3 × 2 = 6 workflows)
wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

# Evaluate all workflows across CV folds
folds = vfold_cv(train_data, v=5)
results = wf_set.fit_resamples(resamples=folds, metrics=metric_set(rmse, mae))

# Analyze results
top_models = results.rank_results("rmse", n=5)
results.autoplot("rmse")

# Select and finalize best workflow
best_wf_id = top_models.iloc[0]["wflow_id"]
best_wf = wf_set[best_wf_id].fit(train_data)
```

**Grouped/Panel Workflow Comparison (NEW - 2025-11-11):**
```python
# Same workflow set creation
wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

# Fit ALL workflows across ALL groups (e.g., 20 workflows × 10 countries = 200 models)
results = wf_set.fit_nested(train_data, group_col='country')

# Collect metrics per group or averaged
metrics_by_group = results.collect_metrics(by_group=True, split='test')
metrics_avg = results.collect_metrics(by_group=False, split='test')

# Rank workflows overall or per group
ranked = results.rank_results('rmse', by_group=False, n=5)  # Overall ranking
ranked_by_group = results.rank_results('rmse', by_group=True, n=3)  # Per-group ranking

# Extract best workflow(s)
best_wf_id = results.extract_best_workflow('rmse', by_group=False)  # Overall best
best_by_group = results.extract_best_workflow('rmse', by_group=True)  # Best per group

# Visualize comparison
fig = results.autoplot('rmse', by_group=False, top_n=10)  # Average with error bars
fig.show()

# Get all outputs for analysis
outputs_df = results.collect_outputs()  # Includes predictions, actuals, forecasts for all groups
```

**WorkflowSetNestedResults Methods:**
1. **`collect_metrics(by_group, split)`**: Get metrics per-group or averaged across groups
2. **`rank_results(metric, split, by_group, n)`**: Rank workflows by specified metric
3. **`extract_best_workflow(metric, split, by_group)`**: Get best workflow ID or per-group DataFrame
4. **`collect_outputs()`**: Collect all predictions, actuals, forecasts for all workflows and groups
5. **`autoplot(metric, split, by_group, top_n)`**: Visualize comparison with error bars or subplots

**WorkflowSetNestedResamples Helper Methods:**
1. **`compare_train_cv(train_stats, metrics)`**: **NEW (2025-11-12)** - Compare training vs CV performance to identify overfitting
   - Takes `train_stats` from `fit_nested().extract_outputs()[2]`
   - Compares with CV metrics from `fit_nested_resamples()`
   - Returns DataFrame with train/CV metrics side-by-side plus overfitting indicators
   - Automatic format detection (long or wide format)
   - Built-in status flags: 🟢 Good, 🟡 Moderate Overfit, 🔴 Severe Overfit
   - Sorted by CV performance (most reliable metric)

   ```python
   # Fit on full training data
   train_results = wf_set.fit_nested(train_data, group_col='country')
   outputs, coeffs, train_stats = train_results.extract_outputs()

   # Evaluate with CV
   cv_results = wf_set.fit_nested_resamples(cv_folds, group_col='country', metrics=metrics)

   # Compare (ONE LINE!)
   comparison = cv_results.compare_train_cv(train_stats)

   # Find overfitting workflows
   overfit = comparison[comparison['rmse_overfit_ratio'] > 1.2]

   # Best per group
   best = comparison.sort_values('rmse_cv').groupby('group').first()
   ```

   Documentation: `.claude_debugging/COMPARE_TRAIN_CV_HELPER.md`
   Tests: `tests/test_workflowsets/test_compare_train_cv.py` (5 tests, all passing)

**Group-Aware Cross-Validation (NEW - 2025-11-12):**

Two new methods for robust per-group CV evaluation:
- **`fit_nested_resamples()`**: Fit per-group models with CV (excludes group column)
- **`fit_global_resamples()`**: Fit global models with per-group CV evaluation (includes group as feature)

**Problem Solved:** Supervised feature selection steps (step_select_permutation, step_select_shap) fail when group column is present in data, causing errors like "could not convert string to float: 'Algeria'". These methods automatically exclude the group column from CV splits before evaluation, preventing supervised selection steps from receiving categorical group data.

**Technical Details of Fix:**
- `fit_nested_resamples()` drops the group column from RSplit objects before passing to `tune_fit_resamples()`
- Creates new Split objects with modified data (same train/test indices)
- Supervised selection steps now only see numeric features
- Group information preserved separately for results/metrics
- Code: `py_workflowsets/workflowset.py:561-584`

```python
from py_rsample import time_series_cv

# Create CV splits per group
cv_by_group = {}
for country in ['USA', 'Germany', 'Japan']:
    country_data = data[data['country'] == country]
    cv_by_group[country] = time_series_cv(
        country_data,
        date_column='date',
        initial='4 years',
        assess='1 year'
    )

# Evaluate all workflows on each group's CV splits (per-group models)
# verbose=False (default): Simple progress messages
results = wf_set.fit_nested_resamples(
    resamples=cv_by_group,
    group_col='country',
    metrics=metric_set(rmse, mae)
)

# verbose=True: Detailed progress with workflow, group, and fold counts
results = wf_set.fit_nested_resamples(
    resamples=cv_by_group,
    group_col='country',
    metrics=metric_set(rmse, mae),
    verbose=True  # Shows: [1/2] Workflow: prep_1_linear_reg_1
                  #         [1/3] Group: USA (2 folds) ✓
)

# OR: Evaluate global models with per-group CV
results = wf_set.fit_global_resamples(
    data=train_data,
    resamples=cv_by_group,
    group_col='country',
    metrics=metric_set(rmse, mae)
)

# Same analysis interface as fit_nested()
metrics_by_group = results.collect_metrics(by_group=True, summarize=True)
ranked = results.rank_results('rmse', by_group=False, n=5)
best_wf_id = results.extract_best_workflow('rmse', by_group=False)
```

**Verbose Output (verbose=True):**
```
Fitting 2 workflows across 3 groups with CV...
Total evaluations: 2 workflows × 3 groups × avg 2 folds

[1/2] Workflow: prep_1_linear_reg_1
  [1/3] Group: USA (2 folds) ✓
  [2/3] Group: Germany (2 folds) ✓
  [3/3] Group: Japan (2 folds) ✓

[2/2] Workflow: prep_2_linear_reg_1
  [1/3] Group: USA (2 folds) ✓
  [2/3] Group: Germany (2 folds) ✓
  [3/3] Group: Japan (2 folds) ✓

✓ CV evaluation complete
```

**Key Differences:**
- **fit_nested_resamples()**: Per-group models, group column excluded, separate training per group
- **fit_global_resamples()**: Global model, group column as feature, evaluated per-group
- Both return `WorkflowSetNestedResamples` with same methods
- **verbose parameter**: Control progress detail level (default: False)

**Code References:**
- `py_workflowsets/workflowset.py:488-716` - Method implementations
- `tests/test_workflowsets/test_fit_nested_resamples.py` - 10 comprehensive tests (all passing)
- `.claude_debugging/WORKFLOWSET_NESTED_RESAMPLES_IMPLEMENTATION.md` - Complete documentation
- `.claude_debugging/demo_verbose_output.py` - Verbose output demonstration

**Key Features:**
- Parallel workflow evaluation for speed
- Automatic ID generation (e.g., "minimal_linear_reg_1")
- Consistent result format with `collect_metrics()`
- Visual comparison with `autoplot()`
- Access individual workflows: `wf_set["workflow_id"]`
- **NEW**: Group-aware comparison (fit all workflows on all groups simultaneously)
- **NEW**: Per-group ranking and selection
- **NEW**: Identify heterogeneous patterns (different groups prefer different workflows)

**Files:**
- `py_workflowsets/workflowset.py` - WorkflowSet, WorkflowSetResults, and WorkflowSetNestedResamples classes
- `tests/test_workflowsets/` - 72 tests passing (20 general + 20 grouped + 10 nested resamples + 3 supervised selection + 19 other)
- `examples/11_workflowsets_demo.ipynb` - Standard CV-based demo
- `_md/forecasting_workflowsets_grouped.ipynb` - Grouped modeling demo (NEW)
- `_md/forecasting_workflowsets_cv_grouped.ipynb` - CV with grouped data demo (NEW)
- `_md/forecasting_advanced_workflow_grouped.ipynb` - Advanced grouped workflow demo (NEW)
- `.claude_debugging/SUPERVISED_SELECTION_FIX.md` - Supervised selection fix documentation (NEW)

## Critical Implementation Notes

### Formula I() Transformations and Column Validation
**Problem:** Formulas with patsy's `I()` function (e.g., `I(x1*x2)` for interactions, `I(x1**2)` for polynomials) were failing in `forge()` with "Required columns missing" errors.

**Root Cause:** The `_validate_columns()` function in `forge.py` was checking for transformed column names (like `"I(x1*x2)"`) in the raw input data, when these are columns created BY patsy during transformation. It should only check for base columns (like `"x1"` and `"x2"`).

**Solution:** Rewrote `_validate_columns()` to parse formulas with regex and extract only base variable names from inside `I()` functions:
```python
# Example: "y ~ x1 + x2 + I(x1*x2) + I(x1**2)"
# Validates only: x1, x2 (base columns)
# Ignores: I(x1*x2), I(x1**2) (transformed columns)
```

**Supported I() Patterns:**
- ✅ Interactions: `I(x1*x2)`, `I(x1*x2*x3)`
- ✅ Polynomials: `I(x1**2)`, `I(x2**3)`
- ✅ Arithmetic: `I(x1 + x2)`, `I(x1 / x2)`
- ✅ Multiple I() terms: `I(x1*x2) + I(x1**2)`

**Code References:**
- `py_hardhat/forge.py:139-211` - `_validate_columns()` with regex parsing
- `tests/test_hardhat/test_mold_forge.py` - 14/14 tests passing including I() tests
- `examples/11_workflowsets_demo.ipynb` - Demonstrates I() usage in multi-model comparison

### step_poly() and Patsy XOR Errors
**Problem:** Polynomial features created column names like `brent^2`, and when used in auto-generated formulas, patsy interpreted `^` as the XOR bitwise operator, causing `PatsyError: Cannot perform 'xor' with a dtyped [float64] array and scalar of type [bool]`.

**Root Cause:** sklearn's `PolynomialFeatures.get_feature_names_out()` returns column names with `^` for powers (e.g., `x^2`, `x^3`). The `StepPoly.prep()` method only replaced spaces with underscores, not `^` characters.

**Solution:** Replace both spaces and `^` characters in feature names:
```python
# In py_recipes/steps/basis.py:361-368
feature_names = [
    name.replace(' ', '_').replace('^', '_pow_')
    for name in feature_names
]
```

**Column Name Transformations:**
- `brent^2` → `brent_pow_2` (quadratic term)
- `dubai^3` → `dubai_pow_3` (cubic term)
- `x1 x2` → `x1_x2` (interaction term)

**Why This Matters:**
- In patsy: `^` is XOR operator, NOT exponentiation
- For exponentiation in formulas: Use `I(x**2)` syntax
- Column names with `^` cannot be used directly in formulas
- Replacing `^` with `_pow_` creates safe identifiers

**Impact:**
```python
# Before fix (ERROR)
rec = recipe().step_poly(['x1', 'x2'], degree=2)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit_nested(train, group_col='country')  # ❌ PatsyError: Cannot perform 'xor'

# After fix (WORKS)
rec = recipe().step_poly(['x1', 'x2'], degree=2)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit_nested(train, group_col='country')  # ✓ Works! Columns: x1_pow_2, x2_pow_2
```

**Code References:**
- `py_recipes/steps/basis.py:361-368` - Feature name sanitization
- `tests/test_recipes/test_basis.py::TestStepPoly` - 9/9 polynomial tests passing
- `.claude_debugging/STEP_POLY_CARET_FIX_2025_11_10.md` - Complete documentation

### Time Series Models and Datetime Handling
**Problem:** Patsy treats datetime columns as categorical, causing errors in forge() when prediction data has new dates.

**Solution:** Dual-path architecture
- Engines with special data requirements implement `fit_raw()` and `predict_raw()`
- These methods bypass hardhat molding entirely
- `ModelSpec.fit()` checks `hasattr(engine, "fit_raw")` to determine path

**Affected Models:** prophet_reg, arima_reg

**Special Handling for ARIMA:**
ARIMA treats date columns as the time series index, NOT as exogenous variables:
```python
# In fit_raw():
if pd.api.types.is_datetime64_any_dtype(data[predictor]):
    y = data.set_index(date_col)[outcome_name]
    # Date becomes index, not exog variable
```

**Code References:**
- `py_parsnip/model_spec.py:94` - fit() method with dual-path logic
- `py_parsnip/model_spec.py:177` - predict() method with dual-path logic
- `py_parsnip/engines/prophet_engine.py:44` - fit_raw() implementation
- `py_parsnip/engines/statsmodels_arima.py:44` - fit_raw() implementation with datetime detection

### Datetime Columns in Auto-Generated Formulas
**Problem:** When using workflows with recipes (no explicit formula), datetime columns were included in auto-generated formulas, causing categorical encoding errors during prediction.

**User Error Example:**
```python
# Data with date column
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'x1': [...],'x2': [...], 'target': [...]
})

# Recipe without explicit formula
rec = recipe().step_normalize()
wf = workflow().add_recipe(rec).add_model(linear_reg())

fit = wf.fit(train_data)  # Train dates: 2020-01 to 2023-09
fit = fit.evaluate(test_data)  # Test dates: 2023-10+

# ERROR: "observation with value Timestamp('2023-10-01') does not match
# any of the expected levels" - Patsy treated date as categorical!
```

**Root Cause:** Workflow auto-generated formula `"target ~ date + x1 + x2"` included the datetime column. Patsy treats datetime as categorical, failing when test data has new dates.

**Solution:** Auto-generated formulas now automatically exclude datetime columns:
```python
# In py_workflows/workflow.py:216-225
predictor_cols = [
    col for col in processed_data.columns
    if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
]
formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"
```

**Result:**
```python
# Same code now works!
rec = recipe().step_normalize()
wf = workflow().add_recipe(rec).add_model(linear_reg())

fit = wf.fit(train_data)  # Auto-formula: "target ~ x1 + x2" (date excluded)
fit = fit.evaluate(test_data)  # ✅ Works with new dates!
```

**Important Notes:**
- Only affects AUTO-generated formulas (when recipe used without explicit formula)
- If user provides explicit formula via `.add_formula()`, it's used as-is
- Multiple datetime columns are all excluded automatically
- Non-time-series data (no datetime columns) unaffected

**Code References:**
- `py_workflows/workflow.py:216-225` - Datetime column exclusion logic
- `tests/test_workflows/test_datetime_exclusion.py` - 5 comprehensive tests

### Dot Notation Formula Expansion
**Purpose:** Support R-style `"target ~ ."` formulas that automatically use all columns except outcome.

**Problem:** The dot notation in formulas like `"target ~ ."` needs special handling to:
1. Expand `.` to all columns except outcome
2. Exclude datetime columns (which cause patsy categorical errors on new dates)
3. Work consistently across all model types (time series and standard)

**Solution:** Two-tier expansion system based on model path:

#### For Time Series Models (fit_raw path):
**File:** `py_parsnip/utils/time_series_utils.py:266-299`
```python
def _expand_dot_notation(exog_vars: List[str], data: pd.DataFrame,
                        outcome_name: str, date_col: str) -> List[str]:
    """Expand patsy's "." notation to all columns except outcome and date."""
    if exog_vars == ['.']:
        return [col for col in data.columns
                if col != outcome_name and col != date_col and col != '__index__']
    return exog_vars
```

**Applied to 9 time series engines:**
- Prophet, ARIMA, Auto ARIMA, VARMAX, Seasonal Reg (STL), ETS
- Prophet Boost, ARIMA Boost, Recursive Forecasting

**Usage:**
```python
from py_parsnip import prophet_reg

# Training data with date + 3 features
spec = prophet_reg()
fit = spec.fit(data, "target ~ .")  # ✅ Expands to x1, x2, x3 (excludes target, date)
```

#### For Standard Models (mold/forge path):
**File:** `py_parsnip/model_spec.py:201-247`
```python
# In ModelSpec.fit(), BEFORE calling mold():
if ' . ' in formula or formula.endswith(' .') or ' ~ .' in formula:
    # Parse formula and expand dot notation
    predictor_cols = [
        col for col in data.columns
        if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(data[col])
    ]
    expanded_formula = f"{outcome_str} ~ {' + '.join(predictor_cols)}"
    formula = expanded_formula

molded = mold(formula, data)  # Receives expanded formula with date excluded
```

**Applied to all standard models:**
- linear_reg, rand_forest, decision_tree, svm_rbf, svm_linear, nearest_neighbor, etc.

**Usage:**
```python
from py_parsnip import linear_reg

# Training: Apr 2020 - Sep 2023
spec = linear_reg()
fit = spec.fit(train_data, "target ~ .")  # ✅ Expands to all except target, date

# Test: Oct 2023+ (NEW dates not in training)
fit = fit.evaluate(test_data)  # ✅ Works! Date was excluded from formula
```

**Supported Patterns:**
```python
# Pure dot notation
"target ~ ."                    # All columns except target and date

# Dot with additions
"target ~ . + I(x1*x2)"         # All columns + interaction term

# Terms before dot
"target ~ x1 + ."               # x1 + all other columns (no duplication)
```

**Why Datetime Exclusion is Critical:**
```python
# WITHOUT datetime exclusion:
fit = spec.fit(train_data, "target ~ .")
# Internally becomes: "target ~ date + x1 + x2 + x3"
# Patsy treats date as CATEGORICAL with training date levels
fit.evaluate(test_data)  # ❌ PatsyError: New dates don't match training levels

# WITH datetime exclusion (current behavior):
fit = spec.fit(train_data, "target ~ .")
# Internally becomes: "target ~ x1 + x2 + x3"  (date excluded)
# Patsy only sees numeric/categorical predictors
fit.evaluate(test_data)  # ✅ SUCCESS: No date-related categorical errors
```

**Code References:**
- Time series: `py_parsnip/utils/time_series_utils.py:266-299` - _expand_dot_notation()
- Standard models: `py_parsnip/model_spec.py:201-247` - Dot expansion in fit()
- Time series tests: `.claude_debugging/test_dot_notation_verification.py` - 4 tests
- Standard model tests: `.claude_debugging/test_standard_model_dot_notation.py` - 3 tests
- Documentation: `.claude_debugging/DOT_NOTATION_FIX.md` - Time series implementation
- Documentation: `.claude_debugging/STANDARD_MODEL_DOT_NOTATION_FIX.md` - Standard model implementation

### Recursive Forecasting with skforecast
**Purpose:** Multi-step time series forecasting using lagged features and recursive prediction.

**Key Implementation Details:**

1. **DatetimeIndex Frequency Requirement:**
   skforecast requires explicit frequency on DatetimeIndex. The engine automatically infers frequency:
   ```python
   if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
       freq = pd.infer_freq(y.index)
       if freq:
           y.index = pd.DatetimeIndex(y.index, freq=freq)
       else:
           # Fallback to most common difference
           diffs = y.index[1:] - y.index[:-1]
           most_common_diff = diffs.value_counts().idxmax()
           y = y.asfreq(most_common_diff)
   ```

2. **Base Model Mode Setting:**
   Base models (e.g., rand_forest) created without mode must be set to "regression":
   ```python
   if base_model_spec.mode == "unknown":
       base_model_spec = base_model_spec.set_mode("regression")
   ```

3. **In-Sample Residuals for Prediction Intervals:**
   Must store residuals during fit for prediction intervals:
   ```python
   forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
   ```

4. **Flexible Lag Specification:**
   - Integer: `lags=7` uses lags 1-7
   - List: `lags=[1, 7, 14]` uses specific lags
   - Supports differentiation parameter for non-stationary data

**Code References:**
- `py_parsnip/models/recursive_reg.py` - Model specification
- `py_parsnip/engines/skforecast_recursive.py` - Engine implementation
- `tests/test_parsnip/test_recursive.py` - 19 tests passing
- `examples/12_recursive_forecasting_demo.ipynb` - Demo notebook

### Panel/Grouped Models
**Purpose:** Fit models for datasets with multiple groups/entities (e.g., multiple stores, regions, customers).

**Two Approaches:**

1. **Nested/Per-Group (`fit_nested()`):**
   - Fits independent model for each group
   - Best when groups have different patterns
   - Returns `NestedWorkflowFit` with unified interface
   ```python
   nested_fit = workflow.fit_nested(data, group_col="store_id")
   predictions = nested_fit.predict(test)  # Routes to correct model
   ```

2. **Global (`fit_global()`):**
   - Fits single model with group as feature
   - Best when groups share similar patterns
   - Returns standard `WorkflowFit`
   ```python
   global_fit = workflow.fit_global(data, group_col="store_id")
   ```

**Key Implementation Details:**

1. **Date-Index Conversion Only for Recursive Models:**
   Panel methods only set date as index for `recursive_reg` models:
   ```python
   is_recursive = self.spec and self.spec.model_type == "recursive_reg"
   if is_recursive and "date" in group_data.columns:
       group_data = group_data.set_index("date")
   ```
   This ensures regular models can access date column in formulas (e.g., `sales ~ date`).

2. **Group Column in Outputs:**
   All three DataFrames from `extract_outputs()` include the group column:
   - Enables easy filtering by group
   - Supports group-wise metric comparison
   - Consistent across nested and global approaches

3. **Unified Prediction Interface:**
   `NestedWorkflowFit.predict()` automatically routes predictions to appropriate group model:
   ```python
   for group in test_data[self.group_col].unique():
       group_data = test_data[test_data[self.group_col] == group]
       preds = self.group_fits[group].predict(group_data)
   ```

**Code References:**
- `py_workflows/workflow.py:fit_nested()` - Nested fitting method
- `py_workflows/workflow.py:fit_global()` - Global fitting method
- `py_workflows/workflow.py:NestedWorkflowFit` - Grouped workflow class
- `tests/test_workflows/test_panel_models.py` - 13 tests passing
- `examples/13_panel_models_demo.ipynb` - Demo notebook

### Classification and One-Hot Encoding
**Problem:** Classification outcomes get one-hot encoded (e.g., "species" → "species[setosa]", "species[versicolor]"). The evaluate() method auto-detects outcome column but gets the encoded name.

**Solution:** Strip encoding suffixes when detecting outcome columns:
```python
# In ModelFit.evaluate()
if outcome_col and "[" in outcome_col:
    outcome_col = outcome_col.split("[")[0]  # "species[setosa]" → "species"
```

**Problem:** Test data may not have all categorical levels from training (e.g., test set has species A and B, but training had A, B, C).

**Solution:** forge() adds missing dummy columns with zeros:
```python
# In _align_columns()
missing = set(expected_cols) - set(actual_cols)
if missing:
    for col in missing:
        X_mat[col] = 0.0  # Missing categorical dummy
```

**Code References:**
- `py_parsnip/model_spec.py:268-271` - evaluate() strips one-hot encoding
- `py_hardhat/forge.py:212-218` - _align_columns() adds missing dummies

### Abstract Method Requirements
Engines must implement:
- `fit()` or `fit_raw()` - Fit model to data
- `predict()` or `predict_raw()` - Make predictions
- `extract_outputs()` - Return three-DataFrame output
- `translate_params()` - Map tidymodels params to engine params

If using raw path, standard methods should raise NotImplementedError.

### Multivariate Time Series (VARMAX)
**Purpose:** Model multiple correlated time series simultaneously.

**Key Requirements:**
- VARMAX requires **at least 2 outcome variables** in the formula:
  ```python
  # CORRECT - Multiple outcomes
  spec = varmax_reg()
  fit = spec.fit(data, formula="y1 + y2 ~ date")          # Bivariate
  fit = spec.fit(data, formula="y1 + y2 + y3 ~ date")     # Trivariate

  # ERROR - Single outcome
  fit = spec.fit(data, formula="y ~ date")  # Raises ValueError
  ```

**Multi-Outcome Predictions:**
- Predictions include separate columns for each outcome:
  ```python
  predictions = fit.predict(forecast_data, type="numeric")
  # Returns: .pred_y1, .pred_y2 columns

  predictions = fit.predict(forecast_data, type="conf_int")
  # Returns: .pred_y1, .pred_y1_lower, .pred_y1_upper
  #          .pred_y2, .pred_y2_lower, .pred_y2_upper
  ```

**Three-DataFrame Outputs:**
- **outputs**: Has `outcome_variable` column to distinguish between y1, y2, etc.
  - Each outcome has separate rows with its own actuals, fitted, residuals
  - Total rows = n_observations × n_outcomes
- **coefficients**: AR/MA parameters for all outcome variables
- **stats**: Includes `n_outcomes` metric showing number of outcome variables

**Code References:**
- `py_parsnip/models/varmax_reg.py` - Model specification
- `py_parsnip/engines/statsmodels_varmax.py` - VARMAX engine
- `tests/test_parsnip/test_varmax_reg.py` - 62 tests covering bivariate/trivariate models

### Auto ARIMA Engine
**Purpose:** Automatically select optimal ARIMA parameters via search.

**Parameter Interpretation:**
- In manual ARIMA: parameters are **exact** values
  ```python
  arima_reg(non_seasonal_ar=2).set_engine("statsmodels")
  # Fits ARIMA with p=2 (exactly)
  ```
- In auto_arima: parameters become **MAX constraints**
  ```python
  arima_reg(non_seasonal_ar=2).set_engine("auto_arima")
  # Searches for best p in range [0, 2] (max_p=2)
  ```

**Parameter Mapping:**
```python
# Non-seasonal
non_seasonal_ar → max_p (default: 5)
non_seasonal_differences → max_d (default: 2)
non_seasonal_ma → max_q (default: 5)

# Seasonal
seasonal_ar → max_P (default: 2)
seasonal_differences → max_D (default: 1)
seasonal_ma → max_Q (default: 2)
seasonal_period → m (exact value, not a maximum)
```

**Automatic Selection:**
- Uses AIC/BIC to compare models within constraints
- Returns optimal order in `fit.fit_data["order"]` as tuple `(p, d, q)`
- Returns seasonal order in `fit.fit_data["seasonal_order"]` as `(P, D, Q, m)`

**Code References:**
- `py_parsnip/engines/pmdarima_auto_arima.py` - auto_arima engine
- `tests/test_parsnip/test_auto_arima.py` - 57 tests covering search constraints

**Known Issue: numpy 2.x Compatibility**
- pmdarima 2.0.4 was compiled against numpy 1.x and has binary incompatibility with numpy 2.x
- Error: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
- **Solutions:**
  1. **Recommended:** Use statsmodels ARIMA engine instead:
     ```python
     spec = arima_reg().set_engine("statsmodels")
     ```
  2. Downgrade numpy to 1.26.x (if auto_arima is essential):
     ```bash
     pip install 'numpy<2.0'
     ```
  3. Wait for pmdarima to release numpy 2.x compatible wheels

**Workaround in Code:**
The auto_arima engine now catches this error and provides a helpful message:
```python
try:
    from pmdarima import auto_arima
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        raise ImportError("pmdarima has a numpy compatibility issue...")
```

### Notebook Index Mismatch Issues
**Problem:** When comparing pandas Series from test data with prediction DataFrames, index mismatch causes ValueError.

**Solution:** Always use `.values` to compare underlying numpy arrays:
```python
# Wrong
rmse = np.sqrt(np.mean((test['y'] - preds['.pred'])**2))

# Correct
rmse = np.sqrt(np.mean((test['y'].values - preds['.pred'].values)**2))
```

**Why:** Test data retains original DataFrame index, prediction DataFrames have RangeIndex(0, n).

### sklearn Model API Patterns (CRITICAL FOR NOTEBOOKS)

**Mode Setting Pattern:**
sklearn models (decision_tree, nearest_neighbor, svm_rbf, svm_linear, mlp) use `.set_mode()` method, NOT constructor parameter:

```python
# WRONG - Will cause TypeError
model = decision_tree(mode='regression', tree_depth=5, min_n=2)

# CORRECT - Mode set via method chaining
model = decision_tree(tree_depth=5, min_n=2).set_mode('regression')
```

**Affected Models:**
- decision_tree(), nearest_neighbor(), svm_rbf(), svm_linear(), mlp()

**Other models that DO accept mode parameter:**
- rand_forest(mode='regression') - WORKS
- boost_tree(mode='regression') - WORKS

**Formula Syntax Pattern:**
ALL sklearn models require Patsy formulas, not bare outcome names:

```python
# WRONG - Missing formula
fitted = model.fit(train_df, 'y')

# CORRECT - Proper Patsy formula
fitted = model.fit(train_df, 'y ~ X')      # Single predictor
fitted = model.fit(df_multi, 'target ~ .')  # All predictors
fitted = model.fit(df_circle, 'y ~ X1 + X2')  # Multiple predictors
```

**Prediction Column Name:**
Predictions always use `.pred` column, never `predictions`:

```python
# WRONG
rmse = np.sqrt(mean_squared_error(test['y'], test_preds['predictions']))

# CORRECT
rmse = np.sqrt(mean_squared_error(test['y'], test_preds['.pred']))
```

**Code References:**
- Fixed in examples/18_sklearn_regression_demo.ipynb (~77 total fixes applied)
- Pattern documented in PHASE_4A_NOTEBOOK_TESTING_REPORT.md

### statsmodels MSTL API Limitation (Multiple Seasonal-Trend decomposition using LOESS)
**Important API Behavior in statsmodels 0.14.5:**

In statsmodels 0.14.5, the `MSTL` class for decomposing time series with multiple seasonal periods has a key limitation:

**The `.seasonal` attribute returns a pandas Series (sum of all seasonal components), NOT a DataFrame with individual seasonal components.**

```python
from statsmodels.tsa.seasonal import MSTL

# Decompose with weekly (7) and yearly (365) patterns
mstl = MSTL(ts, periods=[7, 365], windows=[7, 365], iterate=2)
result = mstl.fit()

# result.seasonal is a Series, not DataFrame!
type(result.seasonal)  # <class 'pandas.core.series.Series'>
result.seasonal.shape  # (730,) - 1D array

# WRONG - This will fail with KeyError
seasonal_7 = result.seasonal["seasonal_7"]     # KeyError!
seasonal_365 = result.seasonal["seasonal_365"] # KeyError!

# CORRECT - Use the combined seasonal component
combined_seasonal = result.seasonal  # Sum of all seasonal effects
```

**Why This Matters:**
- MSTL combines multiple seasonal patterns (e.g., daily + weekly + yearly)
- You cannot access individual seasonal components separately
- The `.seasonal` Series contains the SUM of all seasonal effects
- This is the expected behavior in statsmodels 0.14.5, not a bug

**Workaround:**
When working with MSTL in notebooks or engines, always use the combined seasonal component:
```python
# In seasonal_reg engine's extract_components()
components = pd.DataFrame({
    'trend': fitted_model.trend,
    'seasonal': fitted_model.seasonal,  # Combined seasonal (Series)
    'residual': fitted_model.resid
})
```

**Code References:**
- `examples/19_time_series_ets_stl_demo.ipynb:cells-27-29,38` - Fixed MSTL usage (2025-10-28)
- Issue discovered during Notebook 19 testing: attempted `result.seasonal["seasonal_7"]` raised KeyError

### Jupyter Kernel Module Caching
**Problem:** After updating py_tune or other modules, Jupyter notebooks may still use the old cached version, causing errors like "KeyError: 'Column not found: rmse'" even though the code is fixed.

**Solution:** Restart the Jupyter kernel to reload updated modules:
1. In Jupyter: **Kernel** → **Restart** (or **Restart & Clear Output**)
2. Re-run all cells from the beginning

**Why:** Python modules are cached in memory when first imported. The package is installed in editable mode (`pip install -e .`), so code changes ARE reflected, but Jupyter's kernel caches the imported modules until restarted.

**Alternative:** Force module reload without restarting:
```python
import sys
if 'py_tune' in sys.modules:
    del sys.modules['py_tune']
# Then re-import
from py_tune import tune_grid
```

### Notebook Testing Cache Issues
**Problem:** `jupyter nbconvert --execute` may use cached notebook versions even after editing cells, causing old errors to persist.

**Solution:** Clear notebook outputs before testing:
```bash
jupyter nbconvert --clear-output --inplace examples/18_sklearn_regression_demo.ipynb
jupyter nbconvert --to notebook --execute examples/18_sklearn_regression_demo.ipynb \
  --output /tmp/18_test.ipynb --ExecutePreprocessor.timeout=900
```

**Why:** Jupyter nbconvert may cache intermediate results. Clearing outputs forces fresh execution of all cells.

### py_yardstick Metric Functions Return DataFrames
**Important:** All py_yardstick metric functions return DataFrames with columns `['.metric', 'value']`, NOT scalar values.

**Common Pattern:**
```python
from py_yardstick import rmse, mae, r_squared

# These return DataFrames
rmse_df = rmse(y_true, y_pred)  # Returns DataFrame
mae_df = mae(y_true, y_pred)    # Returns DataFrame

# Extract scalar values using .iloc[0]["value"]
rmse_val = rmse(y_true, y_pred).iloc[0]["value"]
mae_val = mae(y_true, y_pred).iloc[0]["value"]
r2_val = r_squared(y_true, y_pred).iloc[0]["value"]
```

**Why:** This design maintains consistency with tidymodels' yardstick package and allows metric_set() to combine multiple metrics into a single DataFrame.

**There are NO _vec versions:** Functions like `rmse_vec()`, `mae_vec()`, `rsq_vec()` do not exist. Always use the base functions and extract values as needed.

**Code References:**
- `py_yardstick/metrics.py` - All metric functions
- `examples/11_workflowsets_demo.ipynb:cell-31` - Correct usage pattern

## Project Status and Planning

**Current Status:** All Issues Complete (1-8), 782+ Tests Passing, WorkflowSet Grouped Modeling COMPLETE
**Last Updated:** 2025-11-11 (Latest: WorkflowSet grouped/panel modeling support)
**Total Tests Passing:** 782+ tests across all packages (762 base + 20 WorkflowSet grouped)
**Total Models:** 23 production-ready models (21 fitted + 1 hybrid + 1 manual)
**Total Engines:** 30+ engine implementations
**All Issues Completed:** ✅ Issues 1-8 from backlog

**Recent Enhancements (2025-11-11):**
- ✅ **WorkflowSet grouped modeling**: `fit_nested()` and `fit_global()` for multi-model comparison across groups
  - Fit ALL workflows across ALL groups with single method call
  - `WorkflowSetNestedResults` with 5 key methods (collect_metrics, rank_results, extract_best_workflow, collect_outputs, autoplot)
  - Group-aware ranking and visualization
  - Three demonstration notebooks updated
  - 20 comprehensive tests (all passing)
  - Code: `py_workflowsets/workflowset.py:313-1058`
  - Docs: `.claude_plans/WORKFLOWSET_GROUPED_IMPLEMENTATION_COMPLETE.md`

**Previous Enhancements (2025-11-09):**
- ✅ **WorkflowFit extract methods**: Added `extract_formula()` and `extract_preprocessed_data()` for debugging and inspection
- ✅ **Recipe datetime safety**: Discretization and dummy encoding now automatically exclude datetime columns
- ✅ **Recipe infinity handling**: `step_naomit()` removes both NaN and ±Inf values
- ✅ **Recipe selector support**: Reduction steps (ICA, KPCA, PLS) now support selector functions
- ✅ **Recipe cleanup**: Removed redundant `step_corr()` (use `step_select_corr()` instead)

**Phase 1 - COMPLETED (Foundation):**
- ✅ py-hardhat: 14 tests - Data preprocessing with mold/forge
- ✅ py-rsample: 35 tests - Time series CV, k-fold CV, period parsing
- ✅ py-parsnip: Core models (linear_reg, rand_forest, prophet_reg, arima_reg)
- ✅ py-workflows: 26 tests - Workflow composition and pipelines
- ✅ Integration tests: 11 tests
- ✅ Dual-path architecture for time series models
- ✅ Comprehensive three-DataFrame outputs
- ✅ evaluate() method for train/test comparison

**Phase 2 - COMPLETED (Scale & Evaluate):**
- ✅ py-recipes: 265 tests - 51 preprocessing steps across 8 categories
- ✅ py-yardstick: 59 tests - 17 evaluation metrics (regression + classification)
- ✅ py-tune: 36 tests - Hyperparameter tuning with grid search
- ✅ vfold_cv() added to py-rsample for standard k-fold cross-validation

**Phase 3 - COMPLETED (Advanced Features):**
- ✅ py-workflowsets: Multi-model comparison framework
  - ✅ WorkflowSet.from_cross() - Create workflow combinations
  - ✅ fit_resamples() - Evaluate across CV folds
  - ✅ collect_metrics() - Aggregate results
  - ✅ rank_results() - Performance ranking
  - ✅ autoplot() - Visualization
  - ✅ Fixed I() transformation support in forge.py
  - ✅ Example notebook: 11_workflowsets_demo.ipynb
- ✅ recursive_reg: Recursive/autoregressive forecasting (19 tests)
  - ✅ skforecast engine with ForecasterRecursive
  - ✅ Flexible lag specification (int or list)
  - ✅ Differentiation support for non-stationary data
  - ✅ Prediction intervals via in-sample residuals
  - ✅ Works with any sklearn-compatible base model
  - ✅ Example notebook: 12_recursive_forecasting_demo.ipynb
- ✅ Panel/Grouped Models: Per-group and global modeling (13 tests)
  - ✅ fit_nested() - Separate models per group
  - ✅ fit_global() - Single model with group as feature
  - ✅ NestedWorkflowFit - Unified interface for grouped predictions
  - ✅ Works with all model types including recursive_reg
  - ✅ Three-DataFrame outputs include group column
  - ✅ Example notebook: 13_panel_models_demo.ipynb
- ✅ py-visualize: Interactive Plotly visualizations (47+ test classes)
  - ✅ plot_forecast() - Time series forecasting plots
  - ✅ plot_residuals() - Diagnostic plots (4 types)
  - ✅ plot_model_comparison() - Multi-model comparison
  - ✅ plot_decomposition() - STL/ETS component visualization
  - ✅ Example notebook: 14_visualization_demo.ipynb
- ✅ py-stacks: Model ensembling via stacking (10 test classes)
  - ✅ Elastic net meta-learning
  - ✅ Non-negative weights option
  - ✅ Model weight visualization
  - ✅ Example notebook: 15_stacks_demo.ipynb

**Phase 4A - COMPLETED (Model Expansion - 300% Growth):**
- ✅ **15 New Models Added** (5 → 20 total models)
- ✅ **317+ New Tests** for Phase 4A models
- ✅ **26+ Engine Implementations** across all models
- ✅ **6 Demo Notebooks** (16-21) for new models

**New Models by Category:**
- ✅ Baseline Models (2): null_model, naive_reg
- ✅ Gradient Boosting (3 engines): XGBoost, LightGBM, CatBoost via boost_tree()
- ✅ sklearn Regression (5): decision_tree, nearest_neighbor, svm_rbf, svm_linear, mlp
- ✅ Time Series (3): exp_smoothing, seasonal_reg (STL), hybrid models
- ✅ Hybrid Time Series (2): arima_boost, prophet_boost
- ✅ Advanced Regression (3): mars, poisson_reg, gen_additive_mod

**Demo Notebooks:**
- ✅ 16_baseline_models_demo.ipynb - Null and naive forecasting (FULLY WORKING)
- ✅ 17_gradient_boosting_demo.ipynb - XGBoost, LightGBM, CatBoost
- ✅ 18_sklearn_regression_demo.ipynb - Decision trees, k-NN, SVM, MLP (77 API fixes applied)
- ✅ 19_time_series_ets_stl_demo.ipynb - ETS and STL decomposition
- ✅ 20_hybrid_models_demo.ipynb - ARIMA+XGBoost, Prophet+XGBoost
- ✅ 21_advanced_regression_demo.ipynb - MARS, Poisson, GAMs

**Known Issues (Phase 4A Notebooks):**
- ✅ Notebook 19: MSTL seasonal component access - FIXED (2025-10-28)
- Notebook 17: TuneResults.show_best() API mismatch - needs fix
- Notebook 21: pyearth dependency incompatible with Python 3.10 - blocking MARS model demos

See `PHASE_4A_NOTEBOOK_TESTING_REPORT.md` and `NOTEBOOK_TESTING_REPORT.md` for detailed testing results.

**Phase 5 - COMPLETED (Multivariate Time Series & Auto ARIMA):**
- ✅ **varmax_reg()**: Multivariate VARMAX for multiple outcome variables (62 tests)
  - Requires 2+ outcome variables (e.g., `y1 + y2 ~ date`)
  - Vector Autoregression with Moving Average and Exogenous variables
  - Handles cross-correlations between multiple time series
  - Supports exogenous predictors
  - Prediction intervals for all outcome variables
  - Three-DataFrame outputs include `outcome_variable` column
  - Tests: `tests/test_parsnip/test_varmax_reg.py` - 62 passing

- ✅ **auto_arima engine**: Automatic ARIMA order selection via pmdarima (57 tests)
  - Set via `arima_reg().set_engine("auto_arima")`
  - Automatically finds optimal (p,d,q) and seasonal (P,D,Q,m) parameters
  - Parameters become MAX constraints (e.g., `non_seasonal_ar=2` → `max_p=2`)
  - Supports seasonal and non-seasonal ARIMA
  - Works with exogenous variables
  - Tests: `tests/test_parsnip/test_auto_arima.py` - 57 passing

**Total Phase 5 Tests:** 119 (62 VARMAX + 57 auto_arima)

**Issues 7-8 - COMPLETED (Generic Hybrid & Manual Models):**
- ✅ **Issue 7: hybrid_model()**: Generic hybrid model combining any two models (24 tests)
  - Three strategies: residual, sequential, weighted
  - Flexible split points (int, float, date string)
  - Automatic mode setting for unknown modes
  - Works with all 23 model types
  - Tests: `tests/test_parsnip/test_hybrid_model.py` - 24 passing
  - Documentation: `_md/ISSUE_7_HYBRID_MODEL_SUMMARY.md`

- ✅ **Issue 8: manual_reg()**: Manual coefficient specification (24 tests)
  - User specifies coefficients directly (no fitting)
  - Compare with external forecasts (Excel, R, SAS)
  - Incorporate domain expert knowledge
  - Create baselines for benchmarking
  - Tests: `tests/test_parsnip/test_manual_reg.py` - 24 passing
  - Documentation: `_md/ISSUE_8_MANUAL_MODEL_SUMMARY.md`

**Total Issues 7-8 Tests:** 48 (24 hybrid + 24 manual)

**Example Notebooks:**
- 01_hardhat_demo.ipynb - Data preprocessing
- 02_parsnip_demo.ipynb - Linear regression
- 03_time_series_models.ipynb - Prophet and ARIMA
- 04_rand_forest_demo.ipynb - Random Forest
- 05_recipes_comprehensive_demo.ipynb - Feature engineering (51 steps)
- 07_rsample_demo.ipynb - Resampling and CV
- 08_workflows_demo.ipynb - Workflow pipelines
- 09_yardstick_demo.ipynb - Model metrics (17 metrics)
- 10_tune_demo.ipynb - Hyperparameter tuning
- 11_workflowsets_demo.ipynb - Multi-model comparison (20 workflows)
- 12_recursive_forecasting_demo.ipynb - Recursive/autoregressive forecasting
- 13_panel_models_demo.ipynb - Panel/grouped models (nested and global)

**WorkflowSet Grouped Modeling Notebooks (_md/ directory):**
- forecasting_workflowsets_grouped.ipynb - All workflows across all groups (NEW)
- forecasting_workflowsets_cv_grouped.ipynb - CV with group-aware evaluation (NEW)
- forecasting_advanced_workflow_grouped.ipynb - Advanced preprocessing strategies (NEW)

**Detailed Plan:** See `.claude_plans/projectplan.md` for full roadmap and architecture decisions.

## Critical Output Column Semantics

All engines must implement standardized output columns in `extract_outputs()`:

### For Observation-Level Outputs DataFrame:
- **`actuals`**: True values from the data
- **`fitted`**: Model predictions
  - Training data: in-sample predictions
  - Test data: out-of-sample predictions
  - ALWAYS contains model predictions, never NaN
- **`forecast`**: Combined actual/fitted series
  - Implementation: `pd.Series(actuals).combine_first(pd.Series(fitted)).values`
  - Shows actuals where they exist, fitted where they don't
  - Used for creating seamless train → test → future forecasts
  - MUST be the same for both train and test splits (not different logic per split)
- **`residuals`**: `actuals - fitted`
- **`split`**: "train", "test", or "forecast"

### Prediction Index for Time Series
When `.predict()` is called on time series models, predictions MUST be indexed by date:
```python
# In predict_raw() for time series engines:
date_index = new_data[predictor_name]  # or date_col
result = pd.DataFrame({".pred": predictions}, index=date_index)
```

This ensures consistency across engines and proper date alignment in visualizations.

**Code References:**
- Prophet: `py_parsnip/engines/prophet_engine.py:172-186` (predict_raw with date index)
- Prophet: `py_parsnip/engines/prophet_engine.py:284-332` (extract_outputs with combine_first)
- ARIMA: `py_parsnip/engines/statsmodels_arima.py:257-280` (predict_raw with date index)
- ARIMA: `py_parsnip/engines/statsmodels_arima.py:374-424` (extract_outputs with combine_first)
- sklearn: `py_parsnip/engines/sklearn_linear_reg.py:247-292` (extract_outputs with combine_first)

## Common Patterns

### Using WorkflowFit Extract Methods (NEW)

Extract formula and preprocessed data for debugging and inspection:

```python
# Fit workflow
wf = workflow().add_recipe(recipe().step_normalize()).add_model(linear_reg())
fit = wf.fit(train_data)

# Extract formula used (works with both explicit formulas and recipes)
formula = fit.extract_formula()
print(formula)  # e.g., "y ~ x1 + x2 + x3"

# Extract preprocessed/transformed data
train_transformed = fit.extract_preprocessed_data(train_data)
test_transformed = fit.extract_preprocessed_data(test_data)

# Inspect what the model actually sees
print(train_transformed.columns)  # Shows normalized features, dummy variables, etc.
print(train_transformed.head())

# Verify transformations (e.g., check normalization)
print(f"x1 mean: {train_transformed['x1'].mean():.4f}")  # Should be ≈ 0
print(f"x1 std: {train_transformed['x1'].std():.4f}")    # Should be ≈ 1
```

**Use cases:**
- Debug preprocessing pipelines (verify normalization, PCA components, dummy variables)
- Understand model inputs (what features the model actually sees)
- Manual analysis on transformed features
- Reproducibility (save formula for documentation)

**Code References:**
- `py_workflows/workflow.py:544-615` - extract_formula() and extract_preprocessed_data() methods
- `.claude_debugging/WORKFLOW_EXTRACT_METHODS.md` - Full documentation

### Adding a New Model Type

1. Create model specification function in `py_parsnip/models/`:
```python
def new_model(param1=default1, param2=default2, engine="default"):
    args = {"param1": param1, "param2": param2}
    return ModelSpec(model_type="new_model", engine=engine, mode="regression", args=args)
```

2. Create engine implementation in `py_parsnip/engines/`:
```python
@register_engine("new_model", "engine_name")
class NewModelEngine(Engine):
    param_map = {"param1": "engine_param1"}

    def fit(self, spec, molded): ...
    def predict(self, fit, molded, type): ...
    def extract_outputs(self, fit): ...
```

3. Register in `py_parsnip/__init__.py` and `py_parsnip/engines/__init__.py`

4. Create tests in `tests/test_parsnip/test_new_model.py`

### Using Raw Data Path (for special models)
If your model needs direct data access (e.g., datetime handling):

1. Implement `fit_raw()` instead of `fit()`:
```python
def fit_raw(self, spec, data, formula):
    # Parse formula manually
    # Fit model directly on data
    # Return (fit_data_dict, blueprint_dict)
```

2. Implement `predict_raw()` instead of `predict()`:
```python
def predict_raw(self, fit, new_data, type):
    # Make predictions directly on new_data
    # Return DataFrame with predictions
```

3. Standard methods should raise NotImplementedError

### Hybrid Model Implementation (Issue 7)
**Purpose:** Combine any two models with flexible strategies.

**Three Strategies:**

1. **Residual Strategy** (default):
```python
spec = hybrid_model(
    model1=linear_reg(),
    model2=rand_forest().set_mode('regression'),
    strategy='residual'
)
# Trains model2 on residuals from model1
# Final prediction = model1_pred + model2_pred
```

2. **Sequential Strategy** (regime changes):
```python
spec = hybrid_model(
    model1=linear_reg(),
    model2=decision_tree().set_mode('regression'),
    strategy='sequential',
    split_point='2020-06-01'  # or int index, or float proportion
)
# Uses model1 before split_point, model2 after
```

3. **Weighted Strategy** (ensembling):
```python
spec = hybrid_model(
    model1=linear_reg(),
    model2=svm_rbf().set_mode('regression'),
    strategy='weighted',
    weight1=0.6,
    weight2=0.4
)
# Final prediction = 0.6*model1_pred + 0.4*model2_pred
```

**Key Implementation Details:**
- Uses public API only (`model.fit()`, `model.predict()`, `extract_outputs()`)
- Automatically sets mode to "regression" for models with `mode='unknown'`
- Works with any two model types from the 23 available
- Returns standard three-DataFrame output

**Code References:**
- `py_parsnip/models/hybrid_model.py` - Model specification
- `py_parsnip/engines/generic_hybrid.py` - Engine implementation
- `tests/test_parsnip/test_hybrid_model.py` - 24 tests passing
- `_md/ISSUE_7_HYBRID_MODEL_SUMMARY.md` - Full documentation

### Manual Regression Implementation (Issue 8)
**Purpose:** Manually specify coefficients instead of fitting from data.

**Use Cases:**
- Compare with external forecasting tools (Excel, R, SAS)
- Incorporate domain expert knowledge
- Create simple baselines
- Reproduce legacy model forecasts

**Basic Usage:**
```python
# Domain expert coefficients
spec = manual_reg(
    coefficients={"temperature": 1.5, "humidity": -0.3},
    intercept=20.0
)
fit = spec.fit(data, 'sales ~ temperature + humidity')
predictions = fit.predict(test_data)
```

**Advanced Usage:**
```python
# Compare with external tool
external_coefs = {"x1": 2.1, "x2": 0.8}
external_model = manual_reg(coefficients=external_coefs, intercept=5.0)
fit = external_model.fit(data, 'y ~ x1 + x2')

# Standard comparison with fitted model
fitted_model = linear_reg().fit(data, 'y ~ x1 + x2')
outputs_ext, _, stats_ext = fit.extract_outputs()
outputs_fit, _, stats_fit = fitted_model.extract_outputs()
```

**Key Implementation Details:**
- Patsy adds "Intercept" column automatically - engine handles this correctly
- Partial coefficient specification (missing coefficients default to 0.0)
- Validates coefficient variables match formula predictors
- Statistical inference columns (std_error, p_value, etc.) set to NaN (not applicable)
- Returns standard three-DataFrame output

**Code References:**
- `py_parsnip/models/manual_reg.py` - Model specification
- `py_parsnip/engines/parsnip_manual_reg.py` - Engine implementation
- `tests/test_parsnip/test_manual_reg.py` - 24 tests passing
- `_md/ISSUE_8_MANUAL_MODEL_SUMMARY.md` - Full documentation

### Common Gotchas When Implementing Engines

1. **Always implement extract_outputs() with proper column semantics**
   - Use `combine_first()` for forecast column (see section above)
   - Never set `fitted` to NaN for test data
   - Include `model`, `model_group_name`, `group` columns

2. **Time series predictions must be date-indexed**
   - Extract date column from new_data
   - Set as index: `result = pd.DataFrame({...}, index=date_index)`

3. **Handle missing data in evaluation**
   - Check if test data has actuals before calling evaluate()
   - Store evaluation results in `fit.evaluation_data` dict

4. **Classification vs Regression modes**
   - Check `fit.spec.mode` to determine prediction type
   - Classification: return ".pred_class" or ".pred_prob" columns
   - Handle one-hot encoded outcomes (strip "[encoded]" suffix)

5. **Parameter translation**
   - Implement `param_map` dict for tidymodels → engine parameter mapping
   - Use `translate_params()` method to convert

6. **Statistical inference for coefficients**
   - OLS models: calculate std_error, t_stat, p_value, CI manually
   - Regularized models: set to NaN (no closed-form inference)
   - Tree models: use feature importances instead of coefficients

## Key Dependencies

- **pandas** (2.3.3): Primary data structure
- **numpy** (2.2.6): Numerical operations
- **patsy** (1.0.2): R-style formula parsing
- **scikit-learn** (1.7.2): sklearn engine backend
- **prophet** (1.2.1): Facebook's time series forecaster
- **statsmodels** (0.14.5): Statistical models (ARIMA)
- **skforecast** (0.18.0): Recursive forecasting engine
- **pytest** (8.4.2): Testing framework

## File Organization

```
py-tidymodels/
├── py_hardhat/          # Layer 1: Data preprocessing
│   ├── blueprint.py     # Blueprint dataclass
│   ├── mold_forge.py    # mold() and forge() functions
│   └── __init__.py
├── py_parsnip/          # Layer 2: Model interface
│   ├── model_spec.py    # ModelSpec and ModelFit classes
│   ├── engine_registry.py  # Engine ABC and registry
│   ├── models/          # Model specification functions
│   │   ├── linear_reg.py
│   │   ├── prophet_reg.py
│   │   ├── arima_reg.py
│   │   └── rand_forest.py
│   ├── engines/         # Engine implementations
│   │   ├── sklearn_linear_reg.py
│   │   ├── sklearn_random_forest.py
│   │   ├── prophet_engine.py
│   │   └── statsmodels_arima.py
│   └── __init__.py
├── py_rsample/          # Layer 3: Resampling
│   ├── initial_split.py
│   ├── time_series_cv.py
│   ├── vfold_cv.py
│   └── split.py
├── py_workflows/        # Layer 4: Pipelines
│   └── workflow.py
├── py_recipes/          # Layer 5: Feature engineering
│   ├── recipe.py
│   └── steps/          # 51 preprocessing steps
├── py_yardstick/        # Layer 6: Model metrics
│   └── metrics.py      # 17 evaluation metrics
├── py_tune/             # Layer 7: Hyperparameter tuning
│   └── tune.py
├── py_workflowsets/     # Layer 8: Multi-model comparison
│   └── workflowset.py
├── tests/               # Test suite (559 tests)
│   ├── test_hardhat/
│   ├── test_parsnip/
│   ├── test_rsample/
│   ├── test_workflows/
│   ├── test_recipes/
│   ├── test_yardstick/
│   ├── test_tune/
│   └── test_workflowsets/
├── examples/            # Jupyter notebooks
│   ├── 01_hardhat_demo.ipynb
│   ├── 02_parsnip_demo.ipynb
│   ├── 03_time_series_models.ipynb
│   ├── 04_rand_forest_demo.ipynb
│   ├── 05_recipes_comprehensive_demo.ipynb
│   ├── 07_rsample_demo.ipynb
│   ├── 08_workflows_demo.ipynb
│   ├── 09_yardstick_demo.ipynb
│   ├── 10_tune_demo.ipynb
│   └── 11_workflowsets_demo.ipynb
├── .claude_plans/       # Project planning documents
│   └── projectplan.md
└── requirements.txt     # Dependencies
```

## Testing Philosophy

- Tests use pytest framework
- Each component has comprehensive test coverage
- Tests verify both success and failure cases
- Model tests check: fit, predict, extract_outputs, parameter translation
- Hardhat tests check: mold, forge, categorical handling, formula parsing

## Reference Materials

The `reference/` directory contains R tidymodels source code for reference:
- `workflows-main/`: R workflows package
- `recipes-main/`: R recipes package
- `broom-main/`: R broom package (output standardization)
- `parsnip-main/`: R parsnip package (not in current directory listing but referenced)

These are for understanding design patterns, not for copying code directly.
