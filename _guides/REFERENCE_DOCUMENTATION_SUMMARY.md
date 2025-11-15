# Complete Reference Documentation Summary

**py-tidymodels Ecosystem Documentation**

This document provides a comprehensive index of all reference documentation for the py-tidymodels library - a Python port of R's tidymodels ecosystem focused on time series regression and forecasting.

**Library Status:** 782+ tests passing | 28 models | 51 recipe steps | 8 architectural layers
**Last Updated:** 2025-11-15
**Documentation Version:** 2.0

---

## Quick Stats

| Component | Count | Tests | Documentation |
|-----------|-------|-------|---------------|
| **Models** | 28 production models | 317+ | COMPLETE_MODEL_REFERENCE.md |
| **Recipe Steps** | 51 preprocessing steps | 265 | COMPLETE_RECIPE_REFERENCE.md |
| **Evaluation Metrics** | 17 metrics | 59 | (Planned: METRICS_EVALUATION_GUIDE.md) |
| **Workflow Methods** | 22+ methods | 64 | COMPLETE_WORKFLOW_REFERENCE.md |
| **Tuning Functions** | 10+ functions | 36 | COMPLETE_TUNING_REFERENCE.md |
| **WorkflowSet Classes** | 4 classes | 72 | COMPLETE_WORKFLOWSET_REFERENCE.md |
| **Example Notebooks** | 21 notebooks | - | examples/ directory |
| **Total Tests** | **782+** | - | tests/ directory |

---

## 📚 Complete Reference Guides

### 1. COMPLETE_MODEL_REFERENCE.md ✓
**Status:** UPDATED 2025-11-15
**Size:** 2,507 lines
**Purpose:** Exhaustive documentation of all 28 production models

**Contents:**
- **Universal Model Features:**
  - Grouped/Panel Modeling (fit_nested, fit_global)
  - Three-DataFrame Output Standard (outputs, coefficients, stats)
- **12 Model Categories:**
  1. Baseline Models (2): null_model, naive_reg
  2. Linear Models (4): linear_reg, poisson_reg, gen_additive_mod, pls
  3. Tree-Based Models (3): decision_tree, rand_forest, bag_tree
  4. Gradient Boosting (1 model, 3 engines): boost_tree (XGBoost, LightGBM, CatBoost)
  5. Support Vector Machines (3): svm_rbf, svm_linear, svm_poly
  6. Instance-Based & Adaptive (3): nearest_neighbor, mars, mlp
  7. Rule-Based Models (1): rule_fit
  8. Time Series Models (5): arima_reg, prophet_reg, exp_smoothing, seasonal_reg, varmax_reg
  9. Hybrid Time Series (2): arima_boost, prophet_boost
  10. Recursive & Window Models (2): recursive_reg, window_reg
  11. Generic Hybrid (1): hybrid_model (4 strategies: residual, sequential, weighted, custom_data)
  12. Manual Models (1): manual_reg

**Each Model Entry Contains:**
- Complete function signature with all parameters
- Available engines with descriptions
- Engine parameter mappings (tidymodels → backend)
- Tunable parameters with recommended grids
- Usage examples (basic to advanced)
- Common use cases and best practices
- Example notebook cross-references

**Key Features:**
- 28 complete model specifications
- 30+ engine implementations
- Tuning patterns and best practices
- Summary tables by mode, use case, and tunability
- 16 example notebook references

**Use Cases:**
- Choosing the right model for a task
- Understanding parameter mappings across engines
- Setting up hyperparameter tuning grids
- Comparing different model types

---

### 2. COMPLETE_WORKFLOW_REFERENCE.md ✓
**Status:** UPDATED 2025-11-15
**Size:** 1,079 lines
**Purpose:** Complete documentation of workflow composition system

**Contents:**

#### Workflow Class (Immutable Specification)
- `add_formula()` - Add R-style formula preprocessing
- `add_model()` - Add model specification
- `add_recipe()` - Add preprocessing recipe
- `update_formula()`, `update_model()`, `update_recipe()` - Update components
- `remove_formula()`, `remove_model()`, `remove_recipe()` - Remove components
- `fit()` - Standard training
- `fit_nested()` - Panel/grouped modeling (separate models per group)
  - **NEW:** `per_group_prep` parameter for group-specific preprocessing
  - **NEW:** `min_group_size` parameter for small group handling
- `fit_global()` - Panel/grouped modeling (single model with group feature)

#### WorkflowFit Class (Fitted Workflow)
- `predict()` - Make predictions with automatic preprocessing
- `evaluate()` - Train/test comparison with metrics
- `extract_outputs()` - Three-DataFrame pattern (outputs, coefficients, stats)
- `extract_fit_parsnip()` - Access underlying fitted model
- `extract_preprocessor()` - Access preprocessing blueprint/recipe
- `extract_spec_parsnip()` - Access model specification
- **`extract_formula()`** - NEW: Get formula used for fitting
- **`extract_preprocessed_data(data)`** - NEW: Apply preprocessing to data

#### NestedWorkflowFit Class (Panel Modeling)
- `predict()` - Automatic routing to appropriate group model
- `evaluate()` - Per-group and aggregate evaluation
- `extract_outputs()` - Three-DataFrame pattern with group column
- **`get_feature_comparison()`** - NEW: Compare features across groups (per-group prep)

**Key Features:**
- 22+ methods documented
- Per-group preprocessing capabilities (NEW)
- Extract methods for debugging (NEW)
- Common patterns and best practices
- Datetime exclusion, dot notation, I() transformations

**Use Cases:**
- Building complete ML pipelines
- Understanding workflow composition
- Panel/grouped modeling
- Extracting model outputs and diagnostics
- Debugging preprocessing pipelines (NEW)

---

### 3. COMPLETE_WORKFLOWSET_REFERENCE.md
**Status:** NEEDS UPDATE (Missing 2025-11-11 and 2025-11-12 features)
**Size:** 1,920 lines
**Purpose:** Complete documentation of multi-model comparison system

**Current Contents:**
- WorkflowSet class (from_cross, from_workflows, fit_resamples, tune_grid)
- WorkflowSetResults class (collect_metrics, rank_results, autoplot)
- 9 complete usage patterns
- Integration with tuning and resampling

**Missing Features (Need Addition):**
- ⏳ **WorkflowSetNestedResults** class (NEW - 2025-11-11)
  - fit_nested() - Fit all workflows across all groups
  - fit_global() - Fit global models with group as feature
  - collect_metrics(by_group) - Per-group or averaged metrics
  - rank_results(by_group) - Overall or per-group ranking
  - extract_best_workflow(by_group) - Best model selection
  - collect_outputs() - All predictions/actuals/forecasts
  - autoplot(by_group) - Group-aware visualizations
- ⏳ **WorkflowSetNestedResamples** class (NEW - 2025-11-12)
  - fit_nested_resamples() - Per-group models with CV
  - fit_global_resamples() - Global models with per-group CV
  - compare_train_cv() - Overfitting detection helper
  - verbose parameter for progress tracking
  - Supervised selection fix (automatic group column exclusion)

**Priority:** HIGH - Contains critical new features from last month

---

### 4. COMPLETE_PANEL_GROUPED_GUIDE.md
**Status:** NEEDS UPDATE (Missing simplified ModelSpec API)
**Size:** ~2,000 lines (estimated)
**Purpose:** Comprehensive grouped/panel modeling guide

**Current Contents:**
- Panel modeling concepts
- fit_nested() and fit_global() strategies
- Per-group and global modeling examples
- Real-world use cases

**Missing Features (Need Addition):**
- ⏳ Simplified API: `spec.fit_nested()` and `spec.fit_global()` on ModelSpec
- ⏳ When to use ModelSpec vs Workflow for grouped modeling
- ⏳ time_series_nested_cv() and time_series_global_cv() documentation
- ⏳ Integration with WorkflowSet grouped comparison
- ⏳ Updated per-group preprocessing section

**Priority:** HIGH - Critical for understanding latest grouped modeling capabilities

---

### 5. COMPLETE_TUNING_REFERENCE.md
**Status:** NEEDS MINOR UPDATE
**Size:** ~1,500 lines (estimated)
**Purpose:** Complete documentation of hyperparameter tuning system

**Contents:**
- **Core Functions:**
  - tune() - Mark parameters for tuning
  - grid_regular() - Evenly-spaced parameter grids
  - grid_random() - Random parameter sampling
  - tune_grid() - Grid search with CV
  - fit_resamples() - Evaluate without tuning
  - finalize_workflow() - Apply best parameters

- **TuneResults Class:**
  - collect_metrics() - All metrics (long/wide format support)
  - collect_predictions() - All predictions
  - show_best() - Top n configurations
  - select_best() - Best single configuration
  - select_by_one_std_err() - Parsimonious selection

- **5 Complete Workflow Examples**
- **10 Best Practice Categories**
- **Common Patterns Cheat Sheet**

**Use Cases:**
- Setting up hyperparameter tuning
- Choosing between regular and random grids
- Understanding parameter transformations
- Implementing multi-stage tuning
- Model selection strategies

---

### 6. COMPLETE_RECIPE_REFERENCE.md
**Status:** NEEDS MODERATE UPDATE
**Size:** ~4,000 lines (estimated)
**Purpose:** Exhaustive documentation of all 51 recipe steps

**Contents:**
- **8 Categories:**
  1. Imputation (6 steps)
  2. Normalization (4 steps)
  3. Encoding (6 steps)
  4. Feature Engineering (14 steps)
  5. Filtering (5 steps)
  6. Row Operations (6 steps)
  7. Transformations (2 steps)
  8. Selectors (8 functions)

**Each Entry Contains:**
- Complete function signature
- Parameter descriptions with defaults
- Usage examples
- Best practices
- Integration patterns

**Missing Features (Need Addition):**
- ⏳ Recent enhancements (2025-11-09):
  - Datetime exclusion in step_dummy() and discretization steps
  - Infinity handling in step_naomit()
  - Selector support in reduction steps
  - step_corr() removal (use step_select_corr())
- ⏳ step_poly() caret fix (x^2 → x_pow_2)

**Priority:** MEDIUM - Needs recent enhancement documentation

---

### 7. FORECASTING_GROUPED_ANALYSIS.md
**Status:** NEEDS MODERATE UPDATE
**Size:** ~500 lines (estimated)
**Purpose:** Grouped forecasting workflows and best practices

**Current Contents:**
- Grouped forecasting concepts
- Real-world examples
- Best practices

**Missing Features (Need Addition):**
- ⏳ WorkflowSet integration (fit_nested, fit_nested_resamples)
- ⏳ Per-group preprocessing strategies
- ⏳ Overfitting detection with compare_train_cv()
- ⏳ Heterogeneous pattern detection

**Priority:** MEDIUM - Important for time series users

---

## 📋 Planned New Guides

### High Priority (Critical for User Onboarding)

#### 8. GETTING_STARTED_GUIDE.md
**Status:** NOT STARTED
**Priority:** CRITICAL
**Estimated Size:** 800-1,000 lines

**Planned Contents:**
- Installation instructions (virtual environment setup)
- Package structure overview
- 5-minute quickstart examples
- "Hello World" for each major component
- Common workflows diagram
- Next steps and learning paths
- Troubleshooting installation issues

---

#### 9. DATA_PREPROCESSING_GUIDE.md
**Status:** NOT STARTED
**Priority:** CRITICAL
**Estimated Size:** 1,200-1,500 lines

**Planned Contents:**
- Hardhat architecture (Blueprint, MoldedData)
- mold() and forge() deep dive
- Formula syntax and patsy integration
- I() transformations
- Dot notation expansion
- Datetime handling
- Categorical variable encoding
- Blueprint serialization
- Common preprocessing patterns
- Troubleshooting validation errors

---

#### 10. RESAMPLING_GUIDE.md
**Status:** NOT STARTED
**Priority:** CRITICAL
**Estimated Size:** 1,000-1,200 lines

**Planned Contents:**
- Time series splits (initial_time_split)
- Time series CV (time_series_cv, rolling vs expanding windows)
- Group-aware CV (time_series_nested_cv, time_series_global_cv) - NEW
- Standard k-fold CV (vfold_cv, stratification)
- RSplit and Split classes
- Integration with tune_grid()
- Best practices for time series validation
- Avoiding data leakage

---

#### 11. TROUBLESHOOTING_GUIDE.md
**Status:** NOT STARTED
**Priority:** CRITICAL
**Estimated Size:** 1,500-1,800 lines

**Planned Contents:**
- Common error messages and solutions
- Debugging workflows (extract_formula, extract_preprocessed_data)
- Kernel restart requirements
- Mode setting patterns
- Formula syntax errors
- Date column handling
- Performance optimization
- Memory management

---

### Medium Priority (Feature Deep-Dives)

#### 12. METRICS_EVALUATION_GUIDE.md
**Status:** NOT STARTED
**Priority:** MEDIUM
**Estimated Size:** 1,000-1,200 lines

**Planned Contents:**
- All 17 metrics with formulas
- metric_set() usage
- Three-DataFrame output interpretation
- Residual diagnostics
- Model comparison strategies
- Cross-validation metric aggregation

---

#### 13. TIME_SERIES_GUIDE.md
**Status:** NOT STARTED
**Priority:** MEDIUM
**Estimated Size:** 1,500-1,800 lines

**Planned Contents:**
- Time series model taxonomy
- Choosing the right model
- Handling seasonality
- Multi-step forecasting
- Prediction intervals
- Recursive vs direct forecasting
- Multivariate time series (VARMAX)
- Auto ARIMA
- Common pitfalls

---

#### 14. FEATURE_ENGINEERING_DEEP_DIVE.md
**Status:** NOT STARTED
**Priority:** MEDIUM
**Estimated Size:** 1,800-2,000 lines

**Planned Contents:**
- Advanced recipe patterns
- Feature selection strategies (6 methods)
- Dimensionality reduction (PCA, ICA, Kernel PCA, PLS)
- Interaction and polynomial features
- Spline transformations
- Per-group feature engineering
- Recipe debugging
- Automated feature engineering

---

### Lower Priority (Specialized Topics)

#### 15-20. Additional Guides (Planned)
- VISUALIZATION_GUIDE.md
- MODEL_STACKING_GUIDE.md
- CLASSIFICATION_GUIDE.md
- MIGRATION_FROM_R_GUIDE.md
- API_REFERENCE_INDEX.md

---

## 🆕 Recent Features (2025-11-10 to 2025-11-15)

### Per-Group Preprocessing (2025-11-10)
- **Feature:** Each group can have its own recipe preprocessing
- **Use Case:** PCA, feature selection, filters where groups need different feature spaces
- **Methods:** `fit_nested(per_group_prep=True, min_group_size=30)`
- **Utility:** `get_feature_comparison()` shows feature differences across groups
- **Code:** `py_workflows/workflow.py:121-179, 255-311, 392-543, 1023-1113`
- **Tests:** `tests/test_workflows/test_per_group_prep.py` (5 tests)
- **Documentation:** `.claude_debugging/PER_GROUP_PREPROCESSING_IMPLEMENTATION.md`

### WorkflowSet Grouped Modeling (2025-11-11)
- **Feature:** Fit ALL workflows across ALL groups simultaneously
- **Methods:** `fit_nested()`, `fit_global()` on WorkflowSet
- **Class:** WorkflowSetNestedResults with 5 key methods
- **Example:** 20 workflows × 10 groups = 200 models in one call
- **Capabilities:** Group-aware ranking, per-group best model selection, visualization
- **Code:** `py_workflowsets/workflowset.py:313-1058`
- **Tests:** 20 comprehensive tests
- **Documentation:** `.claude_plans/WORKFLOWSET_GROUPED_IMPLEMENTATION_COMPLETE.md`

### Group-Aware Cross-Validation (2025-11-12)
- **Feature:** Per-group CV evaluation with automatic group column exclusion
- **Methods:** `fit_nested_resamples()`, `fit_global_resamples()`
- **Helper:** `compare_train_cv()` for overfitting detection
- **Fix:** Supervised feature selection now works with grouped data
- **Problem Solved:** "could not convert string to float" errors
- **Code:** `py_workflowsets/workflowset.py:488-716`
- **Tests:** `tests/test_workflowsets/test_fit_nested_resamples.py` (10 tests)
- **Tests:** `tests/test_workflowsets/test_compare_train_cv.py` (5 tests)
- **Documentation:** `.claude_debugging/WORKFLOWSET_NESTED_RESAMPLES_IMPLEMENTATION.md`
- **Documentation:** `.claude_debugging/COMPARE_TRAIN_CV_HELPER.md`

### Workflow Extract Methods (2025-11-09)
- **Feature:** Extract formula and preprocessed data for debugging
- **Methods:** `extract_formula()`, `extract_preprocessed_data(data)`
- **Use Cases:** Debug preprocessing, inspect transformations, verify normalization
- **Code:** `py_workflows/workflow.py:544-615`
- **Documentation:** `.claude_debugging/WORKFLOW_EXTRACT_METHODS.md`

### Recipe Enhancements (2025-11-09)
- Datetime exclusion in step_dummy() and discretization steps
- Infinity handling in step_naomit() (removes NaN and ±Inf)
- Selector support in reduction steps (step_ica, step_kpca, step_pls)
- step_corr() removed (use step_select_corr() instead)
- step_poly() caret fix (x^2 → x_pow_2 for safe formula parsing)

### Grouped Modeling on ModelSpec (2025-11-10)
- **Feature:** Simplified API without workflow
- **Methods:** `spec.fit_nested(data, formula, group_col)`
- **Methods:** `spec.fit_global(data, formula, group_col)`
- **Comparison:** 2 lines instead of 3 (no workflow creation needed)
- **Code:** `py_parsnip/model_spec.py`
- **Tests:** `tests/test_parsnip/test_nested_model_fit.py` (21 tests)

---

## 📊 Documentation Coverage Statistics

### Total Metrics (as of 2025-11-15)
- **Guides Complete:** 7/20 (35%)
  - 2 fully updated (2025-11-15)
  - 5 existing needing updates
  - 12 new guides planned
- **Models Documented:** 28/28 (100%)
- **Recipe Steps Documented:** 51/51 (100%)
- **Workflow Methods:** 22+ documented
- **Code Examples:** 200+ complete examples
- **Lines of Documentation:** 8,000+ lines (current guides)
- **Estimated Total (when complete):** 25,000+ lines

### Breakdown by Component

#### Models (COMPLETE_MODEL_REFERENCE.md)
- **Models:** 28 complete specifications
- **Engines:** 30+ implementations
- **Categories:** 12 logical groupings
- **Tuning Grids:** 28 recommended configurations
- **Examples:** 50+ from basic to advanced
- **Cross-References:** 16 example notebooks

#### Workflows (COMPLETE_WORKFLOW_REFERENCE.md)
- **Classes:** 3 (Workflow, WorkflowFit, NestedWorkflowFit)
- **Methods:** 22+ documented
- **Patterns:** 8 complete workflows
- **Best Practices:** 10 guidelines
- **NEW Features:** 4 (extract methods, per-group prep, get_feature_comparison)

#### Recipes (COMPLETE_RECIPE_REFERENCE.md)
- **Steps:** 51 preprocessing steps
- **Categories:** 8 logical groupings
- **Selectors:** 8 functions
- **Parameters:** 200+ across all steps
- **Recent Enhancements:** 5 (datetime exclusion, infinity handling, etc.)

#### Tuning (COMPLETE_TUNING_REFERENCE.md)
- **Functions:** 10 core functions
- **Classes:** 2 (TuneParameter, TuneResults)
- **Methods:** 5 TuneResults methods
- **Workflows:** 5 complete end-to-end examples
- **Best Practices:** 10 categories

#### WorkflowSets (COMPLETE_WORKFLOWSET_REFERENCE.md)
- **Classes:** 4 (WorkflowSet, WorkflowSetResults, WorkflowSetNestedResults, WorkflowSetNestedResamples)
- **Methods:** 15+ across all classes
- **Patterns:** 9+ complete usage patterns
- **NEW Features:** 7 (grouped modeling, CV helpers, compare_train_cv)

---

## 🗺️ Learning Paths

### Path 1: Quick Start (Beginners)
**Time:** 2-4 hours
1. **GETTING_STARTED_GUIDE.md** (planned) - Installation and quickstart
2. **COMPLETE_MODEL_REFERENCE.md** (Section 1-2) - Baseline and linear models
3. **COMPLETE_WORKFLOW_REFERENCE.md** (Patterns 1-2) - Basic workflows
4. **Examples:** 01_hardhat_demo.ipynb, 02_parsnip_demo.ipynb

### Path 2: Time Series Forecasting
**Time:** 6-8 hours
1. **COMPLETE_MODEL_REFERENCE.md** (Section 8-9) - Time series models
2. **TIME_SERIES_GUIDE.md** (planned) - Time series concepts
3. **COMPLETE_WORKFLOW_REFERENCE.md** - Time series workflows
4. **RESAMPLING_GUIDE.md** (planned) - Time series CV
5. **Examples:** 03_time_series_models.ipynb, 12_recursive_forecasting_demo.ipynb, 19_time_series_ets_stl_demo.ipynb, 20_hybrid_models_demo.ipynb

### Path 3: Panel/Grouped Modeling
**Time:** 4-6 hours
1. **COMPLETE_PANEL_GROUPED_GUIDE.md** - Grouped modeling concepts
2. **COMPLETE_WORKFLOW_REFERENCE.md** (fit_nested, fit_global) - Per-group methods
3. **COMPLETE_WORKFLOWSET_REFERENCE.md** (grouped features) - Multi-model comparison
4. **RESAMPLING_GUIDE.md** (planned) - Group-aware CV
5. **Examples:** 13_panel_models_demo.ipynb, _md/forecasting_workflowsets_grouped.ipynb, _md/forecasting_workflowsets_cv_grouped.ipynb

### Path 4: Advanced Feature Engineering
**Time:** 6-8 hours
1. **DATA_PREPROCESSING_GUIDE.md** (planned) - Hardhat fundamentals
2. **COMPLETE_RECIPE_REFERENCE.md** - All 51 steps
3. **FEATURE_ENGINEERING_DEEP_DIVE.md** (planned) - Advanced patterns
4. **COMPLETE_WORKFLOW_REFERENCE.md** (extract methods) - Debugging preprocessing
5. **Examples:** 05_recipes_comprehensive_demo.ipynb

### Path 5: Model Selection & Tuning
**Time:** 4-6 hours
1. **COMPLETE_TUNING_REFERENCE.md** - Hyperparameter tuning
2. **COMPLETE_WORKFLOWSET_REFERENCE.md** - Multi-model comparison
3. **COMPLETE_MODEL_REFERENCE.md** (Tuning Patterns) - Best practices
4. **Examples:** 10_tune_demo.ipynb, 11_workflowsets_demo.ipynb

### Path 6: Production Deployment
**Time:** 4-6 hours
1. **TROUBLESHOOTING_GUIDE.md** (planned) - Common errors
2. **COMPLETE_WORKFLOW_REFERENCE.md** (Best Practices) - Production patterns
3. **METRICS_EVALUATION_GUIDE.md** (planned) - Monitoring performance
4. **MODEL_STACKING_GUIDE.md** (planned) - Ensembling strategies

---

## 🔗 Cross-References & Integration

### Component Dependencies
```
py_hardhat (data preprocessing)
    ↓
py_parsnip (models) ← COMPLETE_MODEL_REFERENCE.md
    ↓
py_rsample (resampling) ← RESAMPLING_GUIDE.md (planned)
    ↓
py_workflows (pipelines) ← COMPLETE_WORKFLOW_REFERENCE.md
    ↓
py_recipes (feature engineering) ← COMPLETE_RECIPE_REFERENCE.md
    ↓
py_yardstick (metrics) ← METRICS_EVALUATION_GUIDE.md (planned)
    ↓
py_tune (hyperparameter tuning) ← COMPLETE_TUNING_REFERENCE.md
    ↓
py_workflowsets (multi-model comparison) ← COMPLETE_WORKFLOWSET_REFERENCE.md
    ↓
py_visualize (plotting) ← VISUALIZATION_GUIDE.md (planned)
    ↓
py_stacks (ensembling) ← MODEL_STACKING_GUIDE.md (planned)
```

### Guide Cross-References

**COMPLETE_MODEL_REFERENCE.md references:**
- Grouped modeling → COMPLETE_PANEL_GROUPED_GUIDE.md
- Workflows → COMPLETE_WORKFLOW_REFERENCE.md
- Tuning → COMPLETE_TUNING_REFERENCE.md
- Example notebooks → examples/ directory

**COMPLETE_WORKFLOW_REFERENCE.md references:**
- Models → COMPLETE_MODEL_REFERENCE.md
- Recipes → COMPLETE_RECIPE_REFERENCE.md
- Tuning → COMPLETE_TUNING_REFERENCE.md
- Grouped modeling → COMPLETE_PANEL_GROUPED_GUIDE.md

**COMPLETE_WORKFLOWSET_REFERENCE.md references:**
- All components above
- Resampling → RESAMPLING_GUIDE.md (planned)
- Metrics → METRICS_EVALUATION_GUIDE.md (planned)

---

## 📂 File Locations

All reference documents are located in the `_guides/` directory:

```
_guides/
├── COMPLETE_MODEL_REFERENCE.md              (2,507 lines) ✓ Updated
├── COMPLETE_WORKFLOW_REFERENCE.md           (1,079 lines) ✓ Updated
├── COMPLETE_WORKFLOWSET_REFERENCE.md        (1,920 lines) ⏳ Needs update
├── COMPLETE_PANEL_GROUPED_GUIDE.md          (~2,000 lines) ⏳ Needs update
├── COMPLETE_TUNING_REFERENCE.md             (~1,500 lines) ⏳ Needs update
├── COMPLETE_RECIPE_REFERENCE.md             (~4,000 lines) ⏳ Needs update
├── FORECASTING_GROUPED_ANALYSIS.md          (~500 lines) ⏳ Needs update
├── REFERENCE_DOCUMENTATION_SUMMARY.md       (this file) ✓ Updated
│
├── [PLANNED] GETTING_STARTED_GUIDE.md       (800-1,000 lines)
├── [PLANNED] DATA_PREPROCESSING_GUIDE.md    (1,200-1,500 lines)
├── [PLANNED] RESAMPLING_GUIDE.md            (1,000-1,200 lines)
├── [PLANNED] TROUBLESHOOTING_GUIDE.md       (1,500-1,800 lines)
├── [PLANNED] METRICS_EVALUATION_GUIDE.md    (1,000-1,200 lines)
├── [PLANNED] TIME_SERIES_GUIDE.md           (1,500-1,800 lines)
├── [PLANNED] FEATURE_ENGINEERING_DEEP_DIVE.md (1,800-2,000 lines)
├── [PLANNED] VISUALIZATION_GUIDE.md         (800-1,000 lines)
├── [PLANNED] MODEL_STACKING_GUIDE.md        (700-900 lines)
├── [PLANNED] CLASSIFICATION_GUIDE.md        (1,000-1,200 lines)
├── [PLANNED] MIGRATION_FROM_R_GUIDE.md      (1,200-1,500 lines)
└── [PLANNED] API_REFERENCE_INDEX.md         (2,000-2,500 lines)
```

**Current Total:** ~13,500 lines across 8 guides
**Estimated Final Total:** ~25,000+ lines across 20 guides

---

## 📘 Example Notebooks

All example notebooks are in the `examples/` directory:

**Core Functionality (8 notebooks):**
- 01_hardhat_demo.ipynb - Data preprocessing with mold/forge
- 02_parsnip_demo.ipynb - Linear regression with sklearn
- 03_time_series_models.ipynb - Prophet and ARIMA models
- 04_rand_forest_demo.ipynb - Random Forest (regression & classification)
- 05_recipes_comprehensive_demo.ipynb - Feature engineering (51 steps)
- 07_rsample_demo.ipynb - Resampling and CV
- 08_workflows_demo.ipynb - Workflow composition
- 09_yardstick_demo.ipynb - Model evaluation (17 metrics)

**Advanced Features (5 notebooks):**
- 10_tune_demo.ipynb - Hyperparameter tuning with grid search
- 11_workflowsets_demo.ipynb - Multi-model comparison (20 workflows)
- 12_recursive_forecasting_demo.ipynb - Recursive/autoregressive forecasting
- 13_panel_models_demo.ipynb - Panel/grouped models (nested and global)
- 14_visualization_demo.ipynb - Interactive Plotly visualizations
- 15_stacks_demo.ipynb - Model ensembling via stacking

**Phase 4A Models (6 notebooks):**
- 16_baseline_models_demo.ipynb - Null and naive forecasting
- 17_gradient_boosting_demo.ipynb - XGBoost, LightGBM, CatBoost
- 18_sklearn_regression_demo.ipynb - Decision trees, k-NN, SVM, MLP
- 19_time_series_ets_stl_demo.ipynb - ETS and STL decomposition
- 20_hybrid_models_demo.ipynb - ARIMA+XGBoost, Prophet+XGBoost
- 21_advanced_regression_demo.ipynb - MARS, Poisson, GAMs

**Grouped Modeling (_md/ directory, 3 notebooks):**
- forecasting_workflowsets_grouped.ipynb - All workflows across all groups
- forecasting_workflowsets_cv_grouped.ipynb - CV with group-aware evaluation
- forecasting_advanced_workflow_grouped.ipynb - Advanced preprocessing strategies

**Total:** 21 example notebooks demonstrating all features

---

## 🧪 Test Coverage

All tests are in the `tests/` directory:

| Package | Tests | Status |
|---------|-------|--------|
| py_hardhat | 14 | ✓ All passing |
| py_parsnip | 317+ | ✓ All passing |
| py_rsample | 45+ | ✓ All passing |
| py_workflows | 64 | ✓ All passing |
| py_recipes | 265 | ✓ All passing |
| py_yardstick | 59 | ✓ All passing |
| py_tune | 36 | ✓ All passing |
| py_workflowsets | 72 | ✓ All passing |
| **Total** | **782+** | **✓ All passing** |

---

## 🎯 Common Use Cases Mapped to Documentation

### Use Case: Build a Simple Forecasting Pipeline
1. **GETTING_STARTED_GUIDE.md** (planned) - Installation and setup
2. **COMPLETE_MODEL_REFERENCE.md** - Choose time series model (prophet_reg, arima_reg)
3. **COMPLETE_WORKFLOW_REFERENCE.md** - Compose workflow with formula + model
4. **Example:** 03_time_series_models.ipynb

### Use Case: Optimize Model Hyperparameters
1. **COMPLETE_MODEL_REFERENCE.md** - Identify tunable parameters
2. **COMPLETE_TUNING_REFERENCE.md** - Set up tune_grid() with appropriate grid
3. **COMPLETE_WORKFLOW_REFERENCE.md** - Use finalize_workflow() with best params
4. **Example:** 10_tune_demo.ipynb

### Use Case: Compare Multiple Preprocessing Strategies
1. **COMPLETE_RECIPE_REFERENCE.md** - Create multiple recipe variants
2. **COMPLETE_MODEL_REFERENCE.md** - Choose consistent model type
3. **COMPLETE_WORKFLOWSET_REFERENCE.md** - Use from_cross() and rank_results()
4. **Example:** 11_workflowsets_demo.ipynb

### Use Case: Panel/Grouped Modeling
1. **COMPLETE_PANEL_GROUPED_GUIDE.md** - Understand nested vs global approaches
2. **COMPLETE_WORKFLOW_REFERENCE.md** - Use fit_nested() or fit_global()
3. **COMPLETE_MODEL_REFERENCE.md** - Choose appropriate model
4. **Example:** 13_panel_models_demo.ipynb

### Use Case: Systematic Model Selection
1. **COMPLETE_MODEL_REFERENCE.md** - Identify candidate models
2. **COMPLETE_WORKFLOWSET_REFERENCE.md** - Set up multi-model comparison
3. **COMPLETE_TUNING_REFERENCE.md** - Fine-tune top performers
4. **Example:** 11_workflowsets_demo.ipynb

### Use Case: Debug Preprocessing Pipeline
1. **TROUBLESHOOTING_GUIDE.md** (planned) - Common preprocessing errors
2. **COMPLETE_WORKFLOW_REFERENCE.md** - Use extract_formula() and extract_preprocessed_data()
3. **DATA_PREPROCESSING_GUIDE.md** (planned) - Hardhat fundamentals
4. **COMPLETE_RECIPE_REFERENCE.md** - Verify step behavior

### Use Case: Multi-Group Cross-Validation
1. **RESAMPLING_GUIDE.md** (planned) - time_series_nested_cv() and time_series_global_cv()
2. **COMPLETE_WORKFLOWSET_REFERENCE.md** - fit_nested_resamples() and compare_train_cv()
3. **COMPLETE_PANEL_GROUPED_GUIDE.md** - Group-aware evaluation strategies
4. **Example:** _md/forecasting_workflowsets_cv_grouped.ipynb

---

## 🔧 Maintenance and Updates

### Documentation Status
- **Documentation Version:** 2.0
- **py-tidymodels Version:** Current (as of 2025-11-15)
- **Last Major Update:** 2025-11-15
- **Update Frequency:** As needed when API changes or new features added

### Change Log
- **2025-11-15:** Major update to REFERENCE_DOCUMENTATION_SUMMARY.md
  - Updated statistics (782+ tests, 28 models, 51 recipe steps)
  - Added recent features section (2025-11-10 to 2025-11-15)
  - Updated guide status and priorities
  - Added learning paths and use case mappings
  - Reorganized for better navigation
- **2025-11-15:** Updated COMPLETE_MODEL_REFERENCE.md
  - Added Universal Model Features section
  - Documented grouped/panel modeling
  - Added three-DataFrame output standard
  - Added 16 example notebook cross-references
- **2025-11-15:** Updated COMPLETE_WORKFLOW_REFERENCE.md
  - Documented per-group preprocessing (per_group_prep, min_group_size)
  - Added get_feature_comparison() method
  - Added extract_formula() and extract_preprocessed_data() methods
  - Updated method count to 22+
- **2025-11-09:** Initial creation of comprehensive references

### Next Updates Planned
1. **Immediate (High Priority):**
   - Complete COMPLETE_WORKFLOWSET_REFERENCE.md update (WorkflowSetNestedResults, WorkflowSetNestedResamples)
   - Complete COMPLETE_PANEL_GROUPED_GUIDE.md update (ModelSpec API, group-aware CV)
   - Complete COMPLETE_TUNING_REFERENCE.md minor update

2. **Short-Term (Critical New Content):**
   - Create GETTING_STARTED_GUIDE.md
   - Create TROUBLESHOOTING_GUIDE.md
   - Create DATA_PREPROCESSING_GUIDE.md
   - Create RESAMPLING_GUIDE.md

3. **Medium-Term (Feature Deep-Dives):**
   - Complete remaining existing guide updates
   - Create METRICS_EVALUATION_GUIDE.md
   - Create TIME_SERIES_GUIDE.md
   - Create FEATURE_ENGINEERING_DEEP_DIVE.md

4. **Long-Term (Specialized Topics):**
   - Create remaining 5 guides (VISUALIZATION, STACKING, CLASSIFICATION, MIGRATION, API_REFERENCE)
   - Add visual diagrams for workflow composition
   - Create searchable index of all functions/methods
   - Add interactive examples with expected outputs

---

## 📞 Contact and Contributions

For questions, issues, or contributions related to this documentation:
- **Source Code:** Check respective modules (py_recipes/, py_parsnip/, py_workflows/, etc.)
- **Tests:** Review test files in tests/ directory for additional usage examples
- **Examples:** Consult demo notebooks in examples/ directory for interactive examples
- **Planning:** See .claude_plans/ directory for development roadmaps and status

### Documentation Guidelines
When contributing to or updating documentation:
1. Follow existing format: signature → parameters → examples → notes
2. Include complete parameter lists with defaults
3. Provide at least one working example per function
4. Add to appropriate category
5. Update this REFERENCE_DOCUMENTATION_SUMMARY.md
6. Update "Last Updated" date
7. Cross-reference related guides

---

## 📚 Quick Reference Table

| Task | Guide | Priority | Status |
|------|-------|----------|--------|
| Install library | GETTING_STARTED_GUIDE.md | High | Planned |
| Choose model | COMPLETE_MODEL_REFERENCE.md | High | ✓ Updated |
| Build pipeline | COMPLETE_WORKFLOW_REFERENCE.md | High | ✓ Updated |
| Preprocess data | DATA_PREPROCESSING_GUIDE.md | High | Planned |
| Feature engineering | COMPLETE_RECIPE_REFERENCE.md | Medium | Needs update |
| Grouped modeling | COMPLETE_PANEL_GROUPED_GUIDE.md | High | Needs update |
| Multi-model compare | COMPLETE_WORKFLOWSET_REFERENCE.md | High | Needs update |
| Tune hyperparams | COMPLETE_TUNING_REFERENCE.md | Medium | Needs update |
| Cross-validation | RESAMPLING_GUIDE.md | High | Planned |
| Evaluate models | METRICS_EVALUATION_GUIDE.md | Medium | Planned |
| Time series | TIME_SERIES_GUIDE.md | Medium | Planned |
| Troubleshoot | TROUBLESHOOTING_GUIDE.md | High | Planned |
| Visualize | VISUALIZATION_GUIDE.md | Low | Planned |
| Stack models | MODEL_STACKING_GUIDE.md | Low | Planned |
| Classification | CLASSIFICATION_GUIDE.md | Low | Planned |
| Migrate from R | MIGRATION_FROM_R_GUIDE.md | Low | Planned |

---

**End of Summary**
**Version:** 2.0
**Last Updated:** 2025-11-15
**Total Guides:** 8 current (2 updated, 5 need updates, 1 summary) + 12 planned = 20 total
**Completion:** 2/20 guides fully updated (10%)
