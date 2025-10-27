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

### Running Tests
```bash
# Activate venv first
source py-tidymodels2/bin/activate

# All tests (559 tests passing as of 2025-10-27)
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_hardhat/test_mold_forge.py -v
python -m pytest tests/test_parsnip/test_linear_reg.py -v
python -m pytest tests/test_recipes/test_recipe.py -v
python -m pytest tests/test_yardstick/test_metrics.py -v
python -m pytest tests/test_tune/test_tune.py -v

# With coverage
python -m pytest tests/ --cov=py_hardhat --cov=py_parsnip --cov=py_recipes --cov=py_yardstick --cov=py_tune --cov-report=html
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

**Implemented Models:**
- `linear_reg`:
  - sklearn engine (OLS, Ridge, Lasso, ElasticNet)
  - statsmodels engine (OLS with full statistical inference)
- `rand_forest`: sklearn engine (RandomForestRegressor, RandomForestClassifier)
  - Dual-mode: regression and classification
  - Feature importances instead of coefficients
  - Handles one-hot encoded outcomes
- `prophet_reg`: prophet engine (Facebook's time series forecaster)
- `arima_reg`: statsmodels engine (SARIMAX)

**Parameter Translation:**
- Tidymodels naming → Engine-specific naming
- Example: `penalty` → `alpha` (sklearn), `non_seasonal_ar` → `p` (statsmodels)
- Handled via `param_map` dict in each engine

**Files:**
- `py_parsnip/model_spec.py` - ModelSpec and ModelFit classes
- `py_parsnip/engine_registry.py` - Engine ABC and registry
- `py_parsnip/models/` - Model specification functions
- `py_parsnip/engines/` - Engine implementations
- `tests/test_parsnip/` - 22+ tests passing

### Layer 3: py-rsample (Resampling)
**Purpose:** Train/test splitting and cross-validation for time series and general data.

**Key Functions:**
- **initial_time_split()**: Chronological train/test split with period parsing
- **time_series_cv()**: Rolling/expanding window cross-validation
- **vfold_cv()**: Standard k-fold cross-validation with stratification support

**Features:**
- Period parsing: "2 years", "6 months", etc.
- Explicit date ranges (absolute, relative, mixed)
- Stratified sampling for classification
- Repeated CV support

**Files:**
- `py_rsample/initial_split.py` - Initial time-based splits
- `py_rsample/time_series_cv.py` - Time series CV
- `py_rsample/vfold_cv.py` - K-fold CV
- `py_rsample/split.py` - Split and RSplit classes
- `tests/test_rsample/` - 35+ tests passing

### Layer 4: py-workflows (Pipelines)
**Purpose:** Compose preprocessing + model + postprocessing into unified workflow.

**Key Classes:**
- **Workflow**: Immutable workflow specification
- **WorkflowFit**: Fitted workflow with predictions

**Key Methods:**
- `add_formula()`, `add_model()` - Compose workflow
- `fit()` - Train workflow
- `predict()` - Make predictions with automatic preprocessing
- `evaluate()` - Train/test evaluation
- `extract_outputs()` - Get three-DataFrame outputs

**Files:**
- `py_workflows/workflow.py` - Workflow and WorkflowFit classes
- `tests/test_workflows/` - 26 tests passing

### Layer 5: py-recipes (Feature Engineering)
**Purpose:** Advanced feature preprocessing and engineering pipeline.

**Key Components:**
- **Recipe**: Immutable specification of preprocessing steps
- **51 preprocessing steps** across 8 categories:
  - Imputation (median, mean, mode, KNN, bag, linear)
  - Normalization (normalize, range, center, scale)
  - Encoding (dummy, one-hot, target, ordinal, bin, date)
  - Feature engineering (polynomial, interactions, splines, PCA, log, sqrt, BoxCox, YeoJohnson)
  - Filtering (correlation, variance, missing, outliers, zero-variance)
  - Row operations (sample, filter, slice, arrange, shuffle)
  - Transformations (mutate, discretize)
  - Selectors (all_predictors, all_outcomes, all_numeric, all_nominal, has_role, has_type)

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

## Critical Implementation Notes

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

## Project Status and Planning

**Current Status:** Phase 2 Implementation (In Progress)
**Last Updated:** 2025-10-27
**Total Tests Passing:** 559 tests

**Phase 1 - COMPLETED (188 tests):**
- ✅ py-hardhat: 14/14 tests - Data preprocessing with mold/forge
- ✅ py-parsnip: 96/96 tests - 4 models (linear_reg, rand_forest, prophet_reg, arima_reg) with 5 engines
- ✅ py-rsample: 35/35 tests - Time series CV, k-fold CV, period parsing
- ✅ py-workflows: 26/26 tests - Workflow composition and pipelines
- ✅ Integration tests: 11/11 tests
- ✅ Dual-path architecture for time series models
- ✅ Comprehensive three-DataFrame outputs
- ✅ evaluate() method for train/test comparison

**Phase 2 - COMPLETED (371 tests):**
- ✅ py-recipes: 265/265 tests - 51 preprocessing steps across 8 categories
- ✅ py-yardstick: 59/59 tests - 17 evaluation metrics (regression + classification)
- ✅ py-tune: 36/36 tests - Hyperparameter tuning with grid search
- ✅ vfold_cv() added to py-rsample for standard k-fold cross-validation

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

**Pending (Phase 3):**
- py-workflowsets: Compare multiple workflows
- Additional model types (boost_tree, svm, etc.)

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
│   │   └── arima_reg.py
│   ├── engines/         # Engine implementations
│   │   ├── sklearn_linear_reg.py
│   │   ├── prophet_engine.py
│   │   └── statsmodels_arima.py
│   └── __init__.py
├── tests/               # Test suite
│   ├── test_hardhat/
│   └── test_parsnip/
├── examples/            # Jupyter notebooks
│   ├── 01_hardhat_demo.ipynb
│   ├── 02_parsnip_demo.ipynb
│   └── 03_time_series_models.ipynb
├── .claude_plans/       # Project planning documents
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
