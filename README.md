# py-tidymodels

**A Python implementation of R's tidymodels ecosystem for time series forecasting and machine learning.**

[![Tests](https://img.shields.io/badge/tests-782%2B%20passing-brightgreen)]()
[![Models](https://img.shields.io/badge/models-23-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Overview

py-tidymodels brings the power and elegance of R's tidymodels to Python, with a focus on time series forecasting and machine learning workflows. Built on familiar libraries (scikit-learn, statsmodels, Prophet, XGBoost, LightGBM, CatBoost), it provides a unified, composable API for the entire modeling workflow.

### Key Features

- **23 Production-Ready Models** - Comprehensive toolkit from baselines to hybrid models
- **Unified Model Interface** - Single API across sklearn, statsmodels, Prophet, XGBoost, LightGBM, CatBoost
- **Composable Workflows** - Build reproducible pipelines with formulas or recipes
- **Time Series Focus** - ARIMA, Prophet, ETS, STL, VARMAX, hybrid models, and recursive forecasting
- **Comprehensive Testing** - 782+ tests across all packages
- **Interactive Visualization** - Plotly-based plots for forecasts, diagnostics, and comparisons
- **Panel/Grouped Modeling** - Fit separate models per group or globally with WorkflowSet support
- **Model Ensembling** - Stack models with elastic net meta-learning
- **Production Ready** - Three-DataFrame output structure, train/test evaluation, hyperparameter tuning

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/m-deane/py-tidymodels2.git
cd py-tidymodels

# Create virtual environment
python -m venv py-tidymodels-env
source py-tidymodels-env/bin/activate  # On Windows: py-tidymodels-env\Scripts\activate

# Install package
pip install -e .

# Verify installation
python -c "from py_parsnip import linear_reg; print('✓ Installation successful!')"
```

**📖 [Complete Installation Guide](INSTALLATION.md)** - Detailed instructions, troubleshooting, Jupyter setup, and optional dependencies

## Quick Start

### Basic Time Series Forecasting

```python
import pandas as pd
from py_parsnip import linear_reg, prophet_reg
from py_workflows import workflow
from py_rsample import initial_time_split
from py_recipes import recipe, step_date, step_lag

# Load your data
data = pd.read_csv("your_data.csv")

# Split into train/test
split = initial_time_split(data, prop=0.8)
train = split.training()
test = split.testing()

# Create feature engineering recipe
rec = (
    recipe(sales ~ date, data=train)
    .step_date('date', features=['month', 'week', 'doy'])
    .step_lag('sales', lags=[1, 7, 30])
)

# Build workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

# Fit and evaluate
fit = wf.fit(train)
fit = fit.evaluate(test)

# Extract comprehensive outputs
outputs, coefficients, stats = fit.extract_outputs()

# View results
print(stats[stats['split'] == 'test'])
```

### Multi-Model Comparison

```python
from py_workflowsets import workflow_set
from py_parsnip import linear_reg, rand_forest, prophet_reg, boost_tree, exp_smoothing
from py_rsample import time_series_cv

# Create CV splits
cv = time_series_cv(train, initial=200, assess=50, skip=25)

# Define models - now 20 models available!
models = [
    linear_reg(),
    rand_forest(trees=100),
    boost_tree(trees=100).set_engine("xgboost"),
    prophet_reg(),
    exp_smoothing(seasonal_period=7)
]

# Create workflow set
wf_set = workflow_set(
    preproc=[rec],
    models=models,
    cross=True
)

# Evaluate all models
results = wf_set.fit_resamples(cv)

# Compare performance
comparison = results.rank_results(metric='rmse')
print(comparison)
```

### Interactive Visualization

```python
from py_visualize import plot_forecast, plot_residuals, plot_model_comparison

# Forecast plot with prediction intervals
fig = plot_forecast(fit, prediction_intervals=True)
fig.show()

# Diagnostic plots
fig = plot_residuals(fit, plot_type='all')  # 2x2 grid
fig.show()

# Compare multiple models
fig = plot_model_comparison(
    stats_list=[stats1, stats2, stats3],
    model_names=['Linear', 'RF', 'Prophet'],
    plot_type='bar'
)
fig.show()
```

### Model Stacking

```python
from py_stacks import stacks

# Train diverse base models and collect predictions
# ... (fit multiple models)

# Create ensemble
ensemble = (
    stacks()
    .add_candidates(pred1, name='linear')
    .add_candidates(pred2, name='random_forest')
    .add_candidates(pred3, name='prophet')
    .blend_predictions(penalty=0.01, non_negative=True)
)

# Get model weights
weights = ensemble.get_model_weights()
print(weights)  # Shows contribution of each model

# Compare to individual models
comparison = ensemble.compare_to_candidates()
print(comparison)
```

## Available Models (23 Total)

### Baseline Models (2)
- **null_model()** - Mean/median baseline for benchmarking
- **naive_reg()** - Time series baselines (naive, seasonal_naive, drift)

### Linear & Generalized Models (3)
- **linear_reg()** - Linear regression with regularization (sklearn, statsmodels)
- **poisson_reg()** - Poisson regression for count data (statsmodels)
- **gen_additive_mod()** - Generalized Additive Models (pygam)

### Tree-Based Models (2)
- **decision_tree()** - Single decision trees (sklearn)
- **rand_forest()** - Random forests (sklearn)

### Gradient Boosting (1 model, 3 engines)
- **boost_tree()** - XGBoost, LightGBM, CatBoost engines

### Support Vector Machines (2)
- **svm_rbf()** - RBF kernel SVM (sklearn)
- **svm_linear()** - Linear kernel SVM (sklearn)

### Instance-Based & Adaptive (3)
- **nearest_neighbor()** - k-NN regression (sklearn)
- **mars()** - Multivariate Adaptive Regression Splines (py-earth)
- **mlp()** - Multi-layer perceptron neural network (sklearn)

### Time Series Models (5)
- **arima_reg()** - ARIMA/SARIMAX models (statsmodels, auto_arima)
- **prophet_reg()** - Facebook Prophet (prophet)
- **exp_smoothing()** - Exponential smoothing / ETS (statsmodels)
- **seasonal_reg()** - STL decomposition models (statsmodels)
- **varmax_reg()** - Multivariate VARMAX (statsmodels) - requires 2+ outcome variables

### Hybrid Time Series (2)
- **arima_boost()** - ARIMA + XGBoost (statsmodels + xgboost)
- **prophet_boost()** - Prophet + XGBoost (prophet + xgboost)

### Recursive Forecasting (1)
- **recursive_reg()** - ML models for multi-step forecasting (skforecast)

### Special Models (3)
- **hybrid_model()** - Generic hybrid combining any two models (residual/sequential/weighted/custom_data)
- **manual_reg()** - User-specified coefficients (no fitting) - compare with external forecasts

---

## Package Overview

py-tidymodels consists of 10 integrated packages:

### Phase 1: Foundation (✅ Complete)

| Package | Purpose | Status |
|---------|---------|--------|
| **py-hardhat** | Low-level data preprocessing | ✅ 14 tests |
| **py-rsample** | Time series cross-validation | ✅ 35 tests |
| **py-parsnip** | Unified model interface | ✅ 456 tests, 20 models |
| **py-workflows** | Pipeline composition | ✅ 26 tests |

### Phase 2: Scale & Evaluate (✅ Complete)

| Package | Purpose | Status |
|---------|---------|--------|
| **py-recipes** | Feature engineering | ✅ 265 tests, 78 steps |
| **py-yardstick** | Performance metrics | ✅ 59 tests, 17 metrics |
| **py-tune** | Hyperparameter optimization | ✅ 36 tests |
| **py-workflowsets** | Multi-model comparison | ✅ 40 tests |

### Phase 3: Advanced Features (✅ Complete)

| Package | Purpose | Status |
|---------|---------|--------|
| **Recursive Forecasting** | ML for multi-step forecasting | ✅ 19 tests |
| **Panel/Grouped Models** | Per-group and global modeling | ✅ 13 tests |
| **py-visualize** | Interactive Plotly visualizations | ✅ 47+ test classes |
| **py-stacks** | Model ensembling via stacking | ✅ 10 test classes |

### Phase 4A: Model Expansion (✅ Complete)

| Feature | Status |
|---------|--------|
| **18 New Models** | ✅ Implemented (5 → 23 total) |
| **30+ Engines** | ✅ Registered |
| **WorkflowSet Grouped Modeling** | ✅ 20 new tests |

**Total: 782+ tests passing**

## Documentation

### Demo Notebooks

Comprehensive tutorials in the `examples/` directory:

1. **01_hardhat_demo.ipynb** - Data preprocessing foundations
2. **02_parsnip_demo.ipynb** - Model specifications
3. **03_time_series_models.ipynb** - ARIMA and Prophet
4. **04_rand_forest_demo.ipynb** - Random forest modeling
5. **05_time_series_forecasting_demo.ipynb** - Time series workflows
6. **06_statsmodels_ols_demo.ipynb** - OLS regression
7. **07_rsample_demo.ipynb** - Resampling and CV
8. **08_workflows_demo.ipynb** - Workflow composition
9. **09_yardstick_demo.ipynb** - Performance metrics
10. **10_tune_demo.ipynb** - Hyperparameter tuning
11. **11_workflowsets_demo.ipynb** - Multi-model comparison
12. **12_recursive_forecasting_demo.ipynb** - Recursive forecasting
13. **13_panel_models_demo.ipynb** - Grouped models
14. **14_visualization_demo.ipynb** - Interactive plotting
15. **15_stacks_demo.ipynb** - Model ensembling
16. **16_baseline_models_demo.ipynb** - Null and naive baseline models
17. **17_gradient_boosting_demo.ipynb** - XGBoost, LightGBM, CatBoost
18. **18_sklearn_regression_demo.ipynb** - Decision trees, k-NN, SVM, MLP
19. **19_time_series_ets_stl_demo.ipynb** - Exponential smoothing and STL
20. **20_hybrid_models_demo.ipynb** - ARIMA+XGBoost, Prophet+XGBoost
21. **21_advanced_regression_demo.ipynb** - MARS, Poisson, GAMs

### Architecture

See `.claude_plans/` for detailed architectural documentation:
- `projectplan.md` - Complete project roadmap
- `model_outputs.md` - Three-DataFrame output structure

## Core Concepts

### Three-DataFrame Output Structure

All models return three standardized DataFrames via `extract_outputs()`:

1. **Outputs** (Observation-level):
   - `date`, `actuals`, `fitted`, `forecast`, `residuals`, `split`

2. **Coefficients** (Variable-level):
   - `variable`, `coefficient`, `std_error`, `p_value`, `ci_0.025`, `ci_0.975`, `vif`

3. **Stats** (Model-level):
   - Performance: `rmse`, `mae`, `mape`, `r_squared`
   - Diagnostics: `durbin_watson`, `shapiro_wilk`, `ljung_box`

### Model Engines

Flexible engine system supporting:
- **sklearn**: Linear models, decision trees, random forests, SVM, k-NN, neural networks
- **statsmodels**: ARIMA, ETS, STL, Poisson regression, OLS with full inference
- **prophet**: Facebook Prophet for time series
- **xgboost**: XGBoost gradient boosting
- **lightgbm**: LightGBM gradient boosting
- **catboost**: CatBoost gradient boosting
- **pygam**: Generalized Additive Models
- **py-earth**: Multivariate Adaptive Regression Splines
- **skforecast**: Recursive forecasting with any sklearn model
- **custom hybrids**: ARIMA+XGBoost, Prophet+XGBoost combinations

### Recipe Steps

78 feature engineering steps across 14 categories:
- Time series (lag, date, rolling, diff, pct_change, fourier, trend, seasonal)
- Feature selection (PCA, VIP, Boruta, RFE, correlation, VIF, p-value, stability, LOFO, Granger, stepwise, probe)
- Scaling (normalize, center, scale, range)
- Encoding (dummy, one-hot, integer, target, ordinal)
- Imputation (mean, median, mode, KNN, linear, bag)
- Transformations (log, sqrt, Box-Cox, Yeo-Johnson)
- Basis functions (splines, polynomial, harmonic, interactions)
- And more...

## Examples

### Recursive Forecasting with Random Forest

```python
from py_parsnip import recursive_reg, rand_forest

# Create recursive forecasting model
rec_model = recursive_reg(
    base_model=rand_forest(trees=100, mode='regression'),
    lags=7,  # Use 7 most recent observations
    differentiation=1  # First-order differencing
)

wf = workflow().add_formula("sales ~ .").add_model(rec_model)
fit = wf.fit(train)

# Predict multiple steps ahead
forecast = fit.predict(test, type='pred_int')  # With prediction intervals
```

### Panel Data Modeling

```python
# Fit separate model for each group
nested_fit = wf.fit_nested(data, group_col='store_id')

# Predictions automatically routed to correct model
predictions = nested_fit.predict(test_data)

# Extract outputs with group column
outputs, coefficients, stats = nested_fit.extract_outputs()
```

### Hyperparameter Tuning

```python
from py_tune import tune, tune_grid, grid_regular

# Mark parameters for tuning
wf_tune = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg(penalty=tune(), mixture=tune()))
)

# Create parameter grid
grid = grid_regular(
    penalty={'range': (0.001, 1.0), 'trans': 'log'},
    mixture={'range': (0.0, 1.0)},
    levels=5
)

# Tune
results = tune_grid(wf_tune, resamples=cv, grid=grid)

# Show best
best = results.show_best(n=5)
print(best)

# Finalize workflow
best_params = results.select_best('rmse')
final_wf = wf_tune.finalize_workflow(best_params)
```

### Gradient Boosting (3 Engines)

```python
from py_parsnip import boost_tree

# XGBoost
xgb_model = boost_tree(trees=100, tree_depth=6, learn_rate=0.1).set_engine("xgboost")
fit_xgb = xgb_model.fit(data, "y ~ .")

# LightGBM
lgb_model = boost_tree(trees=100, tree_depth=6, learn_rate=0.1).set_engine("lightgbm")
fit_lgb = lgb_model.fit(data, "y ~ .")

# CatBoost
cat_model = boost_tree(trees=100, tree_depth=6, learn_rate=0.1).set_engine("catboost")
fit_cat = cat_model.fit(data, "y ~ .")

# All three engines support the same interface
predictions = fit_xgb.predict(new_data)
outputs, feature_importance, stats = fit_xgb.extract_outputs()
```

### Time Series Models

```python
from py_parsnip import exp_smoothing, seasonal_reg, naive_reg

# Exponential Smoothing (ETS)
ets_model = exp_smoothing(
    seasonal_period=7,
    trend="additive",
    season="additive"
)
fit_ets = ets_model.fit(data, "sales ~ date")

# STL Decomposition with Multiple Seasons
stl_model = seasonal_reg(
    seasonal_period_1=7,    # Weekly
    seasonal_period_2=365   # Yearly
)
fit_stl = stl_model.fit(data, "sales ~ date")

# Naive Baselines
naive_model = naive_reg(method="seasonal_naive", seasonal_period=7)
fit_naive = naive_model.fit(data, "sales ~ date")

# Extract decomposed components
outputs, coefs, stats = fit_stl.extract_outputs()
# outputs includes: trend, seasonal, seasonal_2, remainder columns
```

### Hybrid Time Series Models

```python
from py_parsnip import arima_boost, prophet_boost

# ARIMA + XGBoost
arima_xgb = arima_boost(
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    trees=50,
    learn_rate=0.1
)
fit_ab = arima_xgb.fit(data, "sales ~ date")

# Prophet + XGBoost
prophet_xgb = prophet_boost(
    changepoint_prior_scale=0.1,
    seasonality_mode="additive",
    trees=50,
    learn_rate=0.1
)
fit_pb = prophet_xgb.fit(data, "sales ~ date")

# Hybrid models combine classical + ML
# ARIMA captures linear patterns, XGBoost captures residual non-linearities
predictions = fit_ab.predict(test_data)
```

### Advanced Regression Models

```python
from py_parsnip import mars, poisson_reg, gen_additive_mod, svm_rbf

# MARS (Multivariate Adaptive Regression Splines)
mars_model = mars(num_terms=20, prod_degree=2)
fit_mars = mars_model.fit(data, "y ~ .")

# Poisson Regression (for count data)
pois_model = poisson_reg()
fit_pois = pois_model.fit(count_data, "count ~ x1 + x2")

# Generalized Additive Models (smooth non-parametric)
gam_model = gen_additive_mod(adjust_deg_free=10)
fit_gam = gam_model.fit(data, "y ~ .")

# Support Vector Machines
svm_model = svm_rbf(cost=1.0, rbf_sigma=0.1)
fit_svm = svm_model.fit(data, "y ~ .")

# All models support the three-DataFrame output
outputs, effects, stats = fit_gam.extract_outputs()
```

## Testing

Run the full test suite:

```bash
# Activate environment
source py-tidymodels-env/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific package
pytest tests/test_parsnip/
```

## Project Status

### Completed (Phases 1-4A + WorkflowSet Grouped Modeling)

- ✅ Core infrastructure (hardhat, workflows, rsample, parsnip)
- ✅ Feature engineering (78 recipe steps)
- ✅ Model evaluation (17 metrics)
- ✅ Hyperparameter tuning (grid search)
- ✅ Multi-model comparison (workflow sets + grouped modeling)
- ✅ Recursive forecasting (skforecast integration)
- ✅ Panel/grouped models (nested and global + WorkflowSet support)
- ✅ Interactive visualization (4 plot functions)
- ✅ Model stacking (elastic net meta-learning)
- ✅ **23 Production Models** - 360% expansion (5 → 23 models)
- ✅ **Gradient Boosting** - XGBoost, LightGBM, CatBoost engines
- ✅ **Time Series Models** - ARIMA, Prophet, ETS, STL, VARMAX, hybrid models
- ✅ **Advanced Regression** - MARS, GAMs, SVM, k-NN, neural networks
- ✅ **Baseline Models** - Null model, naive forecasting methods
- ✅ **Special Models** - Generic hybrid, manual coefficients

### Planned (Phase 4B+)

- ⏳ Additional ensemble methods (bagging variants)
- ⏳ Neural network time series (NNETAR, LSTM)
- ⏳ Interactive dashboard (Dash + Plotly)
- ⏳ MLflow integration
- ⏳ Performance optimizations (parallel processing, caching)

## Contributing

Contributions are welcome! Areas for contribution:
- Additional model engines
- New recipe steps
- Additional metrics
- Documentation improvements
- Bug reports and fixes

## Design Principles

1. **R tidymodels compatibility** - Familiar API for R users
2. **Pythonic** - Leverages Python strengths (type hints, dataclasses)
3. **Composable** - Method chaining and immutable specifications
4. **Comprehensive** - Three-DataFrame outputs with full diagnostics
5. **Time series first** - Native support for forecasting workflows
6. **Well-tested** - 657+ tests, >90% coverage

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **R tidymodels** - Inspiration and API design
- **scikit-learn** - Model implementations
- **statsmodels** - Statistical inference and ARIMA
- **Prophet** - Time series forecasting
- **skforecast** - Recursive forecasting framework
- **pytimetk** - Time series feature engineering
- **Plotly** - Interactive visualizations

## Support

- **Documentation**: See `examples/` notebooks
- **Issues**: [GitHub Issues](https://github.com/your-username/py-tidymodels/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/py-tidymodels/discussions)

---

**Built with ❤️ for data scientists who love both Python and R's tidymodels**
