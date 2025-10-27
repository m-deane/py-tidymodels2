# Time Series & Regression Models: Priority Analysis

**Focus:** Pure time series forecasting and regression models only
**Date:** 2025-10-27
**Current Implementation:** 5 models (3 time series, 2 regression)

---

## Currently Implemented ✅

### Time Series (3 models)
1. `arima_reg` - statsmodels ARIMA
2. `prophet_reg` - Facebook Prophet
3. `recursive_reg` - skforecast recursive forecasting

### Regression (2 models)
4. `linear_reg` - sklearn, statsmodels (regression mode)
5. `rand_forest` - sklearn (regression mode)

---

## PRIORITY 1: Essential Time Series Models 🎯

### A. Exponential Smoothing Models

#### 1. exp_smoothing() - **CRITICAL PRIORITY**
**Description:** Exponential smoothing state space models (ETS)
**Python Engines:**
- `statsmodels.holtwinters` - ExponentialSmoothing, Holt, HoltWinters
- Simple, seasonal, and damped trend variants

**Why Critical:**
- Classic forecasting method, taught in every forecasting course
- Complements ARIMA perfectly (ARIMA for linear, ETS for exponential trends)
- Excellent Python support in statsmodels
- Fast training, good for baselines

**Parameters:**
- seasonal_period - seasonality frequency
- error - additive or multiplicative error
- trend - additive, multiplicative, or damped
- season - additive, multiplicative, or none
- damping - damping parameter for trend

**Implementation Effort:** Low (statsmodels provides everything)
**Use Cases:** Sales forecasting, demand forecasting, financial time series

---

#### 2. seasonal_reg() - **HIGH PRIORITY**
**Description:** Seasonal decomposition models
**Python Engines:**
- `statsmodels.tsa.seasonal` - STL (Seasonal-Trend decomposition with LOESS)
- Custom implementation for STLM (STL + ETS/ARIMA)
- Could integrate `tbats` if Python package available

**Why High Priority:**
- Handles complex seasonal patterns (multiple seasonalities)
- STL is very robust decomposition method
- Essential for understanding seasonal structure
- Good for multi-seasonal data (hourly data with daily + weekly patterns)

**Parameters:**
- seasonal_period_1, seasonal_period_2, seasonal_period_3 - multiple seasonalities

**Implementation Effort:** Medium (STL available, STLM needs combining with other models)
**Use Cases:** Multi-seasonal data, electricity demand, retail with multiple cycles

---

#### 3. naive_reg() - **HIGH PRIORITY**
**Description:** Naive forecasting benchmarks
**Python Engines:**
- Custom implementation (very simple)
- Naive: y_t = y_{t-1}
- Seasonal naive: y_t = y_{t-s}
- Drift: y_t = y_{t-1} + average_change

**Why High Priority:**
- Essential benchmarks for ANY forecasting project
- If your model can't beat naive methods, it's useless
- Extremely fast, no parameters to tune
- Should ALWAYS be included in forecasting comparisons

**Parameters:**
- seasonal_period - for seasonal naive

**Implementation Effort:** Very Low (can implement in 50 lines)
**Use Cases:** Baseline for all forecasting projects

---

#### 4. adam() - **MEDIUM PRIORITY**
**Description:** ADAM - Automatic Dynamic Adaptive Model
**Python Engines:**
- Would need to port R's `smooth` package
- OR use statsmodels ETS with enhancements

**Why Medium Priority:**
- More advanced than basic ETS
- Handles complex patterns
- Automatic model selection

**Implementation Effort:** High (requires porting or significant custom work)
**Use Cases:** Complex time series with uncertain structure

---

### B. Hybrid Time Series Models

#### 5. arima_boost() - **HIGH PRIORITY**
**Description:** ARIMA + XGBoost hybrid
**Python Engines:**
- statsmodels (ARIMA) + xgboost
- Fit ARIMA, then boost the residuals

**Why High Priority:**
- Combines classical statistics with ML
- Often outperforms pure ARIMA
- Captures both linear (ARIMA) and non-linear (XGBoost) patterns
- Novel approach, good differentiation

**Parameters:**
- All ARIMA parameters + XGBoost parameters (trees, learning_rate, etc.)

**Implementation Effort:** Medium (need to coordinate two models)
**Use Cases:** Complex time series with both linear trends and non-linear patterns

---

#### 6. prophet_boost() - **MEDIUM PRIORITY**
**Description:** Prophet + XGBoost hybrid
**Python Engines:**
- prophet + xgboost
- Fit Prophet, then boost the residuals

**Why Medium Priority:**
- Similar concept to arima_boost
- Good for seasonal data with trends
- Prophet handles seasonality, XGBoost captures residual patterns

**Implementation Effort:** Medium
**Use Cases:** Business forecasting with complex patterns

---

### C. Neural Network Time Series

#### 7. nnetar_reg() - **MEDIUM-LOW PRIORITY**
**Description:** Neural Network AutoRegression
**Python Engines:**
- `darts.models.NLinearModel` or `NBEATSModel`
- `pytorch_forecasting.TemporalFusionTransformer`
- `sktime.forecasting.NaiveForecaster` (simpler version)

**Why Medium-Low Priority:**
- Powerful for complex patterns
- Requires more data than classical methods
- Deep learning frameworks add complexity

**Parameters:**
- seasonal_period, lags, hidden_units, epochs

**Implementation Effort:** High (requires integrating deep learning framework)
**Use Cases:** Large datasets with complex non-linear patterns

---

### D. Advanced Decomposition

#### 8. window_reg() - **LOW PRIORITY**
**Description:** Sliding window feature engineering
**Python Engines:**
- Custom implementation
- Rolling window statistics as features

**Why Low Priority:**
- More of a feature engineering approach
- Can be handled with recipes (step_rolling)
- Less value as standalone model

**Implementation Effort:** Medium
**Use Cases:** Feature engineering for ML models

---

#### 9. temporal_hierarchy() - **LOW PRIORITY**
**Description:** Temporal hierarchical reconciliation
**Python Engines:**
- Would need custom implementation
- Complex hierarchical forecasting framework

**Why Low Priority:**
- Very specialized use case
- Significant implementation effort
- Limited Python library support

**Implementation Effort:** Very High
**Use Cases:** Forecasting at multiple temporal aggregations

---

## PRIORITY 2: Essential Regression Models 🎯

### A. Gradient Boosting (CRITICAL)

#### 10. boost_tree() - **CRITICAL PRIORITY**
**Description:** Gradient boosted trees
**Python Engines:**
- `xgboost.XGBRegressor` - Industry standard
- `lightgbm.LGBMRegressor` - Fast, efficient
- `catboost.CatBoostRegressor` - Handles categoricals well

**Why Critical:**
- Most important missing model for regression
- Industry standard for tabular data
- Often best performance on structured data
- Three excellent Python implementations

**Parameters:**
- trees, tree_depth, learn_rate, min_n, sample_size, mtry, loss_reduction

**Implementation Effort:** Low-Medium (libraries handle everything)
**Use Cases:** Any regression task, especially with non-linear patterns

**Modes:** regression (focus for now), classification, censored regression

---

### B. Tree-Based Models

#### 11. decision_tree() - **HIGH PRIORITY**
**Description:** Single decision tree for regression
**Python Engines:**
- `sklearn.tree.DecisionTreeRegressor`

**Why High Priority:**
- Fundamental ML algorithm
- Interpretable
- Foundation for understanding ensembles
- Very easy to implement

**Parameters:**
- tree_depth, min_n, cost_complexity

**Implementation Effort:** Very Low
**Use Cases:** Interpretable regression, understanding data structure

**Modes:** regression, classification (focus on regression)

---

#### 12. bag_tree() - **MEDIUM PRIORITY**
**Description:** Bagged decision trees
**Python Engines:**
- `sklearn.ensemble.BaggingRegressor` with DecisionTreeRegressor base

**Why Medium Priority:**
- Ensemble method
- Reduces variance of decision trees
- Bridge between single trees and random forests

**Implementation Effort:** Low
**Use Cases:** Reducing overfitting in tree models

---

### C. Support Vector Machines

#### 13. svm_rbf() - **HIGH PRIORITY**
**Description:** RBF kernel SVM for regression
**Python Engines:**
- `sklearn.svm.SVR` with rbf kernel

**Why High Priority:**
- Powerful non-linear regression
- Works well with small-medium datasets
- Handles complex patterns

**Parameters:**
- cost (C), rbf_sigma (gamma), epsilon

**Implementation Effort:** Low (sklearn provides everything)
**Use Cases:** Non-linear regression with small-medium data

---

#### 14. svm_linear() - **MEDIUM PRIORITY**
**Description:** Linear kernel SVM for regression
**Python Engines:**
- `sklearn.svm.LinearSVR`

**Parameters:**
- cost (C), epsilon

**Implementation Effort:** Very Low
**Use Cases:** High-dimensional linear regression

---

#### 15. svm_poly() - **LOW PRIORITY**
**Description:** Polynomial kernel SVM for regression
**Python Engines:**
- `sklearn.svm.SVR` with poly kernel

**Parameters:**
- cost, degree, scale_factor, epsilon

**Implementation Effort:** Low
**Use Cases:** Polynomial relationships

---

### D. Spline & Adaptive Models

#### 16. mars() - **HIGH PRIORITY**
**Description:** Multivariate Adaptive Regression Splines
**Python Engines:**
- `pyearth.Earth` - Direct Python implementation

**Why High Priority:**
- Automatic feature interaction detection
- Spline-based non-linear regression
- Interpretable (piecewise linear)
- Good Python library available

**Parameters:**
- num_terms (max_terms), prod_degree (max_degree), prune_method

**Implementation Effort:** Low-Medium (library exists, need to wrap)
**Use Cases:** Non-linear regression with interpretability

---

#### 17. gen_additive_mod() - **MEDIUM PRIORITY**
**Description:** Generalized Additive Models
**Python Engines:**
- `pygam.LinearGAM` for regression
- Smooth non-parametric relationships

**Why Medium Priority:**
- Non-linear but interpretable
- Smooth functions of predictors
- Good for exploratory analysis

**Parameters:**
- spline types, smoothing parameters

**Implementation Effort:** Medium (library exists but complex API)
**Use Cases:** Smooth non-linear relationships, semi-parametric models

---

#### 18. bag_mars() - **LOW PRIORITY**
**Description:** Bagged MARS
**Python Engines:**
- sklearn BaggingRegressor + pyearth

**Implementation Effort:** Low (once MARS is implemented)
**Use Cases:** Ensemble MARS for better predictions

---

### E. Neural Networks

#### 19. mlp() - **MEDIUM PRIORITY**
**Description:** Multi-layer perceptron for regression
**Python Engines:**
- `sklearn.neural_network.MLPRegressor` - Simple, no external dependencies
- `pytorch` or `tensorflow` - More powerful but heavier

**Why Medium Priority:**
- Neural networks for tabular regression
- Flexible function approximation
- sklearn version is lightweight

**Parameters:**
- hidden_units, penalty, dropout, epochs, learn_rate, activation

**Implementation Effort:** Low (sklearn) to Medium (pytorch)
**Use Cases:** Complex non-linear patterns, large datasets

---

### F. Other Regression Models

#### 20. poisson_reg() - **MEDIUM PRIORITY**
**Description:** Poisson regression for count data
**Python Engines:**
- `statsmodels.genmod.GLM` with Poisson family
- `sklearn` doesn't have native Poisson

**Why Medium Priority:**
- Essential for count data
- Proper statistical framework for discrete outcomes
- Complements linear regression

**Parameters:**
- penalty, mixture (for regularized versions)

**Implementation Effort:** Low-Medium
**Use Cases:** Count data, event counts, rare events

---

#### 21. nearest_neighbor() - **HIGH PRIORITY**
**Description:** k-Nearest Neighbors regression
**Python Engines:**
- `sklearn.neighbors.KNeighborsRegressor`

**Why High Priority:**
- Simple, non-parametric
- No training required
- Good baseline
- Very easy to implement

**Parameters:**
- neighbors (k), weight_func, dist_power

**Implementation Effort:** Very Low
**Use Cases:** Local patterns, baseline comparisons

---

#### 22. pls() - **LOW PRIORITY**
**Description:** Partial Least Squares regression
**Python Engines:**
- `sklearn.cross_decomposition.PLSRegression`

**Why Low Priority:**
- Useful for high-dimensional data with collinearity
- Less commonly used than other methods
- Specialized use case

**Implementation Effort:** Low
**Use Cases:** High-dimensional data, chemometrics

---

#### 23. null_model() - **HIGH PRIORITY**
**Description:** Null/baseline model (mean prediction)
**Python Engines:**
- Custom implementation (just predict mean)

**Why High Priority:**
- Essential baseline for ANY regression project
- Simplest possible model
- Quick sanity check

**Implementation Effort:** Very Low (trivial)
**Use Cases:** Baseline for all projects

---

### G. Rule-Based Regression

#### 24. cubist_rules() - **LOW PRIORITY**
**Description:** Rule-based regression with linear models in leaves
**Python Engines:**
- No direct Python equivalent (would need porting)

**Why Low Priority:**
- No Python library available
- High implementation effort
- Specialized use case

**Implementation Effort:** Very High
**Use Cases:** Interpretable rule-based regression

---

#### 25. rule_fit() - **MEDIUM PRIORITY**
**Description:** Sparse rule ensembles with linear model
**Python Engines:**
- `imodels.RuleFitRegressor`

**Why Medium Priority:**
- Interpretable rules + linear model
- Python library available
- Good for explanation

**Implementation Effort:** Medium
**Use Cases:** Interpretable ML, feature interactions

---

### H. Bayesian Models

#### 26. bart() - **LOW PRIORITY**
**Description:** Bayesian Additive Regression Trees
**Python Engines:**
- `pymc_bart`

**Why Low Priority:**
- Requires Bayesian framework
- Computational complexity
- Specialized use case

**Implementation Effort:** High
**Use Cases:** Uncertainty quantification, Bayesian inference

---

## RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 4A: Foundation (6-8 weeks) - TIME SERIES FOCUS

**Goal:** Complete time series forecasting toolkit

1. **exp_smoothing** (statsmodels) - 1 week
   - Essential classical method
   - Complements ARIMA/Prophet
   - Low effort, high value

2. **naive_reg** (custom) - 2-3 days
   - Critical baselines
   - Trivial implementation
   - Must-have for any forecasting

3. **seasonal_reg** (statsmodels STL) - 1.5 weeks
   - Handle complex seasonality
   - STL decomposition
   - Medium effort

4. **arima_boost** (statsmodels + xgboost) - 1 week
   - Hybrid model innovation
   - Combines classical + ML
   - Good differentiation

5. **boost_tree** (xgboost, lightgbm, catboost) - 2 weeks
   - CRITICAL for regression
   - Industry standard
   - Three engines to implement

6. **null_model** (custom) - 1-2 days
   - Essential baseline
   - Trivial implementation

**Deliverables:**
- 6 new models (3 time series, 3 regression)
- Complete time series baseline toolkit (naive, null)
- Complete exponential smoothing family
- Gradient boosting for regression
- Comprehensive tests (20+ per model)
- Demo notebooks for each

**Impact:**
- Time series coverage: 27% → 55% (6/11 models)
- Regression coverage: Adds critical gradient boosting
- Essential baselines for all projects

---

### Phase 4B: Expansion (4-6 weeks) - REGRESSION FOCUS

**Goal:** Comprehensive regression toolkit

1. **decision_tree** (sklearn) - 3-4 days
   - Fundamental algorithm
   - Very easy

2. **nearest_neighbor** (sklearn) - 3-4 days
   - k-NN baseline
   - Very easy

3. **mars** (pyearth) - 1 week
   - Interpretable splines
   - Good Python library

4. **svm_rbf** (sklearn) - 4-5 days
   - Non-linear regression
   - Easy implementation

5. **svm_linear** (sklearn) - 3-4 days
   - Linear SVM
   - Very easy

6. **mlp** (sklearn) - 1 week
   - Neural networks
   - Moderate complexity

7. **gen_additive_mod** (pygam) - 1.5 weeks
   - Smooth non-parametric
   - Complex API integration

8. **poisson_reg** (statsmodels) - 4-5 days
   - Count data
   - Important for many use cases

**Deliverables:**
- 8 new regression models
- Cover tree, SVM, spline, neural network categories
- Comprehensive testing
- Demo notebooks

**Impact:**
- Comprehensive regression toolkit
- Cover all major regression paradigms
- Total models: 5 → 19

---

### Phase 4C: Advanced (4-6 weeks) - HYBRID & ADVANCED

**Goal:** Advanced models and hybrids

1. **prophet_boost** (prophet + xgboost) - 1 week
2. **nnetar_reg** (darts or pytorch_forecasting) - 2 weeks
3. **bag_tree** (sklearn) - 3-4 days
4. **bag_mars** (sklearn + pyearth) - 3-4 days
5. **rule_fit** (imodels) - 1 week
6. **pls** (sklearn) - 3-4 days
7. **svm_poly** (sklearn) - 3-4 days

**Deliverables:**
- 7 advanced models
- Complete hybrid model framework
- Neural network forecasting
- Total models: 19 → 26

---

## PRIORITY SUMMARY (Recommended Order)

### CRITICAL (Implement First - 4 weeks)
1. **exp_smoothing** - Essential time series
2. **naive_reg** - Essential baseline
3. **boost_tree** - Critical regression model
4. **null_model** - Essential baseline

### HIGH PRIORITY (Next - 4 weeks)
5. **seasonal_reg** - Complex seasonality
6. **arima_boost** - Hybrid innovation
7. **decision_tree** - Fundamental algorithm
8. **nearest_neighbor** - Simple baseline
9. **mars** - Interpretable splines
10. **svm_rbf** - Non-linear regression

### MEDIUM PRIORITY (After high priority - 6 weeks)
11. **prophet_boost** - Hybrid model
12. **mlp** - Neural networks
13. **gen_additive_mod** - GAMs
14. **poisson_reg** - Count data
15. **svm_linear** - Linear SVM
16. **bag_tree** - Bagged trees
17. **rule_fit** - Interpretable rules

### LOWER PRIORITY (As needed)
18. **nnetar_reg** - NN forecasting (requires deep learning)
19. **adam** - Advanced ETS (requires porting)
20. **bag_mars** - After MARS
21. **pls** - Specialized
22. **svm_poly** - Less common
23. **bart** - Bayesian (complex)
24. **cubist_rules** - No Python library
25. **window_reg** - Can use recipes
26. **temporal_hierarchy** - Very specialized

---

## EXPECTED COVERAGE AFTER EACH PHASE

**Current:** 5 models total
- Time Series: 3/11 (27%)
- Regression: 2 models

**After Phase 4A:** 11 models
- Time Series: 6/11 (55%) - Missing only advanced models
- Regression: 5 models (covers boosting, baselines)

**After Phase 4B:** 19 models
- Time Series: 6/11 (55%)
- Regression: 13 models (comprehensive toolkit)

**After Phase 4C:** 26 models
- Time Series: 8/11 (73%)
- Regression: 18 models (near-complete)

---

## SUMMARY

**Focus on these 10 models for maximum impact:**

### Time Series (5 models)
1. exp_smoothing - ETS family
2. naive_reg - Essential baselines
3. seasonal_reg - Complex seasonality
4. arima_boost - Hybrid classical + ML
5. null_model - Mean baseline

### Regression (5 models)
6. boost_tree - Gradient boosting (3 engines)
7. decision_tree - Single trees
8. nearest_neighbor - k-NN
9. mars - Interpretable splines
10. svm_rbf - Non-linear SVM

**Total Effort:** ~8 weeks for 10 high-impact models
**Coverage:** 5 → 15 models (200% increase)
**Time Series:** 27% → 55% coverage
**Value:** Complete forecasting toolkit + comprehensive regression

This roadmap focuses exclusively on time series and regression, providing a complete toolkit for forecasting and regression modeling without classification distractions.
