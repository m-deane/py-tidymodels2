# Changelog

All notable changes to py-tidymodels will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-13

### 🎉 Production Release

First production-ready release of py-tidymodels with comprehensive feature set, extensive testing, and complete documentation.

### Added

#### Core Framework (782+ Tests)
- **py_hardhat** - Data preprocessing with mold/forge abstraction (14 tests)
- **py_parsnip** - Unified model interface with 23 production models (22+ tests)
- **py_recipes** - 51 preprocessing steps across 8 categories (265 tests)
- **py_rsample** - Time series CV and k-fold cross-validation (45+ tests)
- **py_workflows** - Composable modeling pipelines (64 tests)
- **py_yardstick** - 17 evaluation metrics for regression and classification (59 tests)
- **py_tune** - Hyperparameter tuning with grid search (36 tests)
- **py_workflowsets** - Multi-model comparison framework (72 tests)
- **py_stacks** - Model ensembling via stacking (10 test classes)
- **py_visualize** - Interactive Plotly visualizations (47+ test classes)

#### Models (23 Total)

**Baseline Models (2)**:
- `null_model()` - Mean/median/last baseline forecasts
- `naive_reg()` - Time series naive forecasting (naive, seasonal_naive, drift, window)

**Linear & Generalized Models (3)**:
- `linear_reg()` - Linear regression (sklearn, statsmodels engines)
- `poisson_reg()` - Poisson regression for count data
- `gen_additive_mod()` - Generalized Additive Models (pygam)

**Tree-Based Models (2)**:
- `decision_tree()` - Decision trees for regression/classification
- `rand_forest()` - Random forests (sklearn engine)

**Gradient Boosting (3 engines)**:
- `boost_tree()` - XGBoost, LightGBM, CatBoost engines

**Support Vector Machines (2)**:
- `svm_rbf()` - RBF kernel SVM
- `svm_linear()` - Linear kernel SVM

**Instance-Based & Adaptive (3)**:
- `nearest_neighbor()` - k-NN regression
- `mars()` - Multivariate Adaptive Regression Splines
- `mlp()` - Multi-layer perceptron neural network

**Time Series Models (5)**:
- `arima_reg()` - ARIMA/SARIMAX (statsmodels, auto_arima engines)
- `prophet_reg()` - Facebook Prophet forecasting
- `exp_smoothing()` - Exponential smoothing / ETS
- `seasonal_reg()` - STL decomposition models
- `varmax_reg()` - Multivariate VARMAX (statsmodels)

**Hybrid Time Series (2)**:
- `arima_boost()` - ARIMA + XGBoost residual modeling
- `prophet_boost()` - Prophet + XGBoost residual modeling

**Recursive Forecasting (1)**:
- `recursive_reg()` - ML models for multi-step forecasting (skforecast)

**Generic Hybrid (1)**:
- `hybrid_model()` - Combine any two models (residual, sequential, weighted, custom_data strategies)

**Manual Specification (1)**:
- `manual_reg()` - User-specified coefficients (no fitting)

#### py_agent - AI-Powered Forecasting Agent (252+ Tests)

**Phase 1 - Basic Workflow Generation** ✅
- Rule-based workflow generation from natural language requests
- Automatic model selection based on data characteristics
- Domain-aware preprocessing (retail, finance, energy)

**Phase 2 - Enhanced Intelligence** ✅
- LLM integration with Claude Sonnet 4.5
- Explainable model selection reasoning
- Budget management for API costs
- Constraint handling (speed, interpretability, accuracy)

**Phase 3 - Advanced Features** ✅
- **Phase 3.1**: Preprocessing strategy recommendations
- **Phase 3.2**: Diagnostic analysis and performance insights
- **Phase 3.3**: Multi-model comparison with diversity scoring
- **Phase 3.4**: RAG knowledge base with 8 foundational examples
- **Phase 3.5**: Autonomous iteration with try-evaluate-improve loops

#### Panel/Grouped Modeling
- `fit_nested()` - Fit separate models per group
- `fit_global()` - Fit single model with group as feature
- `NestedWorkflowFit` - Unified interface for grouped predictions
- Group-aware WorkflowSet comparison
- Per-group cross-validation
- Per-group recipe preprocessing

#### Tutorial Notebooks (5 Comprehensive Guides)

**Tutorial Series (4.5-6.5 hours total)**:
- **Notebook 22** - Complete Overview (30-45 min, beginner)
  - All phases (1, 2, 3.1-3.5)
  - Basic workflow generation
  - Multi-model comparison
  - RAG knowledge base
  - Autonomous iteration

- **Notebook 23** - LLM-Enhanced Mode (45-60 min, intermediate)
  - Claude Sonnet 4.5 integration
  - Explainable reasoning
  - Budget management
  - Constraint handling

- **Notebook 24** - Domain Examples (60-90 min, intermediate)
  - Retail forecasting (sales, inventory, promotions)
  - Finance forecasting (stock prices, volatility, risk)
  - Energy forecasting (demand, temperature, weather)

- **Notebook 25** - Advanced Features (90-120 min, advanced)
  - Performance debugging
  - Ensemble methods
  - Grouped/panel modeling
  - Production best practices

- **Notebook 26** - Real-World Data (60-90 min, advanced) 🆕
  - European gas demand (96K rows, 10 years)
  - Commodity futures (135K rows, 22 years)
  - Crude oil production (13K rows, 22 years)
  - Data quality handling
  - Multi-entity forecasting

**Tutorial Statistics**:
- 119 total cells (73 code, 46 markdown)
- 0 syntax errors across all notebooks
- 100% quality scores
- 244K+ rows of real data demonstrated
- Complete feature coverage (all phases)

#### Documentation
- Complete README with installation, quick start, and examples
- INSTALLATION.md with detailed setup instructions
- DOCUMENTATION_QUICK_START.md for rapid onboarding
- TUTORIALS_INDEX.md with learning path guidance
- VALIDATION_SUMMARY.md for production readiness
- Comprehensive API documentation in docstrings
- Architecture documentation in CLAUDE.md

### Features

#### Three-DataFrame Output Structure
All models return standardized outputs via `extract_outputs()`:
- **outputs** - Observation-level results (actuals, fitted, forecast, residuals, split)
- **coefficients** - Model parameters with statistical inference (std_error, t_stat, p_value, CI, VIF)
- **stats** - Model-level metrics by split (RMSE, MAE, R², AIC, BIC, residual diagnostics)

#### Formula Support
- R-style patsy formulas: `y ~ x1 + x2`
- Interaction terms: `y ~ x1 + x2 + I(x1*x2)`
- Polynomial features: `y ~ x1 + I(x1**2)`
- Dot notation: `y ~ .` (all columns except outcome)
- Automatic datetime exclusion from formulas

#### Preprocessing Features (51 Steps)
- **Imputation**: median, mean, mode, KNN, bag, linear
- **Normalization**: normalize, range, center, scale
- **Encoding**: dummy, one-hot, target, ordinal, bin, date
- **Feature Engineering**: polynomial, interactions, splines, PCA, ICA, kernel PCA, PLS, log, sqrt, BoxCox, YeoJohnson
- **Filtering**: correlation, variance, missing, outliers, zero-variance
- **Row Operations**: sample, filter, slice, arrange, shuffle
- **Transformations**: mutate, discretize
- **Selectors**: all_predictors, all_outcomes, all_numeric, all_nominal, has_role, has_type

#### Cross-Validation
- Time series CV with rolling/expanding windows
- K-fold CV with stratification
- Nested CV for grouped data
- Global CV for panel modeling
- Period parsing: "2 years", "6 months", etc.
- Explicit date ranges (absolute, relative, mixed)

#### Model Comparison
- WorkflowSet for multi-model evaluation
- Automatic workflow combination (preprocessors × models)
- Parallel evaluation across CV folds
- Group-aware comparison for panel data
- Rank results by any metric
- Automatic visualization with autoplot()

#### Visualization
- `plot_forecast()` - Time series forecasting plots
- `plot_residuals()` - Diagnostic plots (4 types)
- `plot_model_comparison()` - Multi-model comparison
- `plot_decomposition()` - STL/ETS component visualization
- Interactive Plotly charts with zoom, pan, hover

#### Production Features
- Comprehensive error handling
- Input validation across all modules
- Type hints and docstrings
- Reproducible workflows (serializable blueprints)
- Train/test evaluation with evaluate()
- Model persistence (pickle/joblib compatible)
- Performance monitoring utilities
- Configuration management

### Fixed
- Patsy `I()` transformation support in forge() validation
- Datetime column exclusion from auto-generated formulas
- Dot notation expansion for all model types
- MSTL seasonal component handling (statsmodels 0.14.5)
- Polynomial feature naming (`^` → `_pow_` for patsy compatibility)
- Supervised selection with group columns (WorkflowSet)
- Notebook module caching issues (kernel restart guidance)

### Security
- Input sanitization for formula parsing
- Safe evaluation of user-provided expressions
- Dependency version pinning for reproducibility
- No arbitrary code execution

### Performance
- Parallel workflow evaluation in WorkflowSet
- Efficient CV fold generation
- Optimized DataFrame operations
- Memory-efficient grouped modeling

### Testing
- 782+ total tests across all packages
- Unit tests for all core functionality
- Integration tests for workflows
- Regression tests for model outputs
- 100% of critical paths covered

### Documentation
- 5 comprehensive tutorial notebooks
- Complete API reference
- Installation and setup guides
- Architecture documentation
- Best practices and examples
- Troubleshooting guides

## [0.1.0] - 2024-11-12

### Added
- Initial development release
- Core framework implementation
- Basic model support
- Foundation for testing infrastructure

---

## Versioning Policy

py-tidymodels follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

## Release Notes

**v1.0.0 Highlights**:
- 🎉 Production-ready release with 782+ passing tests
- 🤖 AI-powered forecasting agent (py_agent) with LLM integration
- 📊 23 production models from baselines to hybrid approaches
- 📚 5 comprehensive tutorial notebooks (4.5-6.5 hours total)
- 🌍 Real-world data examples (244K+ rows across 3 datasets)
- 🔧 51 preprocessing steps for feature engineering
- 📈 Complete visualization suite with Plotly
- 🏢 Panel/grouped modeling support
- ✨ WorkflowSet for multi-model comparison
- 📖 Comprehensive documentation and guides

**What's Next**:
- PyPI package distribution
- Google Colab tutorial versions
- Video walkthroughs
- Community contributions
- Additional model engines
- Extended RAG knowledge base

---

**Project Homepage**: https://github.com/m-deane/py-tidymodels2
**Documentation**: https://github.com/m-deane/py-tidymodels2/tree/main/docs
**Bug Reports**: https://github.com/m-deane/py-tidymodels2/issues
