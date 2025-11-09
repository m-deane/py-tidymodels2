# Complete Model Reference

**py-tidymodels Model Library - Comprehensive Documentation**

This reference covers all 28+ models with complete parameter specifications, engine mappings, tuning capabilities, and usage examples.

---

## Table of Contents

1. [Baseline Models](#1-baseline-models)
2. [Linear Models](#2-linear-models)
3. [Tree-Based Models](#3-tree-based-models)
4. [Gradient Boosting](#4-gradient-boosting)
5. [Support Vector Machines](#5-support-vector-machines)
6. [Instance-Based & Adaptive](#6-instance-based--adaptive)
7. [Rule-Based Models](#7-rule-based-models)
8. [Time Series Models](#8-time-series-models)
9. [Hybrid Time Series](#9-hybrid-time-series)
10. [Recursive & Window Models](#10-recursive--window-models)
11. [Generic Hybrid](#11-generic-hybrid)
12. [Manual Models](#12-manual-models)
13. [Summary Tables](#summary-tables)
14. [Tuning Patterns](#common-tuning-patterns)

---

## 1. BASELINE MODELS

### 1.1 null_model()

**Purpose:** Simplest baseline predicting constant values (mean/median/last).

**Function Signature:**
```python
null_model(
    strategy: Literal["mean", "median", "last"] = "mean",
    engine: str = "parsnip"
) -> ModelSpec
```

**Parameters:**
- `strategy` (str): Baseline prediction strategy
  - `"mean"`: Mean of training outcomes (default)
  - `"median"`: Median of training outcomes
  - `"last"`: Last observed value (time series baseline)
- `engine` (str): Computational engine (default: "parsnip")

**Mode:** Regression (default), Classification (via `.set_mode('classification')`)

**Available Engines:** parsnip (built-in)

**Tunable Parameters:** None

**Use Cases:**
- Essential baseline for ANY modeling project
- If your model can't beat this, it's useless
- Regression: predicts mean/median/last
- Classification: predicts mode (most frequent class)

**Example:**
```python
from py_parsnip import null_model

# Mean baseline
spec = null_model(strategy="mean")
fit = spec.fit(train, 'y ~ x')

# Median baseline (robust to outliers)
spec = null_model(strategy="median")

# Last value baseline (time series)
spec = null_model(strategy="last")
```

---

### 1.2 naive_reg()

**Purpose:** Essential time series forecasting baselines.

**Function Signature:**
```python
naive_reg(
    strategy: Literal["naive", "seasonal_naive", "drift", "window"] = "naive",
    seasonal_period: Optional[int] = None,
    window_size: Optional[int] = None,
    engine: str = "parsnip"
) -> ModelSpec
```

**Parameters:**
- `strategy` (str): Naive forecasting strategy
  - `"naive"`: Last value (random walk forecast)
  - `"seasonal_naive"`: Last value from same season
  - `"drift"`: Linear trend from first to last observation
  - `"window"`: Rolling window average
- `seasonal_period` (int or None): Seasonal frequency (required for seasonal_naive)
  - 7 for weekly patterns in daily data
  - 12 for yearly patterns in monthly data
  - 24 for daily patterns in hourly data
- `window_size` (int or None): Window size for rolling average (required for window strategy)
- `engine` (str): Computational engine (default: "parsnip")

**Mode:** Regression only

**Available Engines:** parsnip (built-in)

**Tunable Parameters:** None (except window_size if using window strategy)

**Use Cases:**
- Critical baselines for time series forecasting
- If ML model can't beat these, don't use it
- Surprisingly effective for many time series
- Similar to sktime's NaiveForecaster

**Example:**
```python
from py_parsnip import naive_reg

# Naive forecast (last value)
spec = naive_reg(strategy="naive")
fit = spec.fit(train, 'sales ~ date')

# Seasonal naive (weekly pattern in daily data)
spec = naive_reg(strategy="seasonal_naive", seasonal_period=7)

# Drift (linear trend extrapolation)
spec = naive_reg(strategy="drift")

# Moving average
spec = naive_reg(strategy="window", window_size=7)
```

---

## 2. LINEAR MODELS

### 2.1 linear_reg()

**Purpose:** Linear regression with optional L1/L2 regularization.

**Function Signature:**
```python
linear_reg(
    penalty: Optional[float] = None,
    mixture: Optional[float] = None,
    intercept: bool = True,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `penalty` (float or None): Regularization penalty (default: None = no penalty)
  - 0 = no regularization (OLS)
  - Higher values = more regularization
  - Typical range: 0.001 to 10
- `mixture` (float or None): Mix between L1 and L2 penalty (0 to 1)
  - 0.0 = pure L2 (Ridge regression)
  - 1.0 = pure L1 (Lasso regression)
  - 0.5 = equal mix (ElasticNet)
  - Default: None (uses 0.0 if penalty is set)
- `intercept` (bool): Whether to fit intercept term (default: True)
- `engine` (str): Computational engine (default: "sklearn", also "statsmodels")

**Mode:** Regression only

**Available Engines:**
1. **sklearn** (default) - scikit-learn backend
2. **statsmodels** - Statistical inference and diagnostics

**Engine Parameter Mappings:**

**sklearn engine:**
- `penalty` → `alpha`
- `mixture` → `l1_ratio`
- `intercept` → `fit_intercept`

**Model Selection Logic:**
- No penalty → LinearRegression
- penalty + mixture=0.0 → Ridge
- penalty + mixture=1.0 → Lasso
- penalty + mixture=(0,1) → ElasticNet

**Tunable Parameters:** `penalty`, `mixture`

**Recommended Tuning Grid:**
```python
from py_tune import tune, grid_regular

spec = linear_reg(penalty=tune(), mixture=tune())

grid = grid_regular({
    "penalty": {"range": (0.001, 1.0), "trans": "log"},
    "mixture": {"range": (0, 1)}
}, levels=5)
```

**Use Cases:**
- **OLS** (no penalty): Simple linear relationships, interpretability
- **Ridge** (L2): Prevent overfitting with correlated features, multicollinearity
- **Lasso** (L1): Feature selection via sparsity, automatic variable selection
- **ElasticNet**: Combines Ridge + Lasso benefits, robust feature selection

**Example:**
```python
from py_parsnip import linear_reg

# OLS (no regularization)
spec = linear_reg()

# Ridge (L2 regularization)
spec = linear_reg(penalty=0.1, mixture=0.0)

# Lasso (L1 regularization with feature selection)
spec = linear_reg(penalty=0.1, mixture=1.0)

# ElasticNet (balanced L1/L2)
spec = linear_reg(penalty=0.1, mixture=0.5)

# No intercept
spec = linear_reg(intercept=False)
```

---

### 2.2 poisson_reg()

**Purpose:** Count data modeling using Poisson GLM.

**Function Signature:**
```python
poisson_reg(
    penalty: Optional[float] = None,
    mixture: Optional[float] = None,
    engine: str = "statsmodels"
) -> ModelSpec
```

**Parameters:**
- `penalty` (float or None): Regularization penalty (if supported by engine)
- `mixture` (float or None): L1/L2 mix (if supported)
- `engine` (str): Computational engine (default: "statsmodels")

**Mode:** Regression only

**Available Engines:** statsmodels

**Tunable Parameters:** `penalty` (if supported)

**Use Cases:**
- Count data (number of events: purchases, visits, claims)
- Rare events (accidents, defects)
- Event rates (calls per hour, emails per day)
- Non-negative integer outcomes

**Example:**
```python
from py_parsnip import poisson_reg

# Basic Poisson regression
spec = poisson_reg()
fit = spec.fit(data, 'num_purchases ~ age + income')

# Count of defects
fit = spec.fit(data, 'defect_count ~ temperature + pressure')
```

---

### 2.3 gen_additive_mod()

**Purpose:** Generalized Additive Models with automatic non-linearity detection.

**Function Signature:**
```python
gen_additive_mod(
    select_features: Optional[bool] = None,
    adjust_deg_free: Optional[Union[float, int]] = None,
    engine: str = "pygam"
) -> ModelSpec
```

**Parameters:**
- `select_features` (bool or None): Enable automatic feature selection (default: False)
- `adjust_deg_free` (float/int or None): Degrees of freedom controlling smoothness (default: 10)
  - Lower (3-5) = smoother curves (more bias, less variance)
  - Higher (15-20) = more flexible/wiggly curves (less bias, more variance)
  - Typical range: 3 to 20
- `engine` (str): Computational engine (default: "pygam")

**Mode:** Regression only

**Available Engines:** pygam (pyGAM library)

**Tunable Parameters:** `adjust_deg_free`

**Recommended Tuning Grid:**
```python
spec = gen_additive_mod(adjust_deg_free=tune())

grid = grid_regular({
    "adjust_deg_free": {"range": (3, 20)}
}, levels=6)
```

**Use Cases:**
- Automatic non-linearity detection (no manual transformations needed)
- Interpretable smooth functions (can visualize effect curves)
- Flexible alternative to polynomial regression
- Visual assessment of predictor effects (partial dependence plots)

**Example:**
```python
from py_parsnip import gen_additive_mod

# Default GAM (df=10)
spec = gen_additive_mod()

# More flexible (wiggly curves)
spec = gen_additive_mod(adjust_deg_free=15)

# Smoother curves (less overfitting)
spec = gen_additive_mod(adjust_deg_free=5)

# With feature selection
spec = gen_additive_mod(select_features=True, adjust_deg_free=10)
```

---

### 2.4 pls()

**Purpose:** Partial Least Squares regression for dimension reduction.

**Function Signature:**
```python
pls(
    num_comp: Optional[int] = None,
    predictor_prop: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `num_comp` (int or None): Number of PLS components to extract (default: 2)
  - Should be < min(n_samples, n_features, n_targets)
  - Higher = more variance explained
- `predictor_prop` (float or None): Proportion of predictor variance to explain (0-1)
  - Alternative to num_comp
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Regression only

**Available Engines:** sklearn (scikit-learn)

**Engine Parameter Mappings:**
- `num_comp` → `n_components`

**Tunable Parameters:** `num_comp`

**Recommended Tuning Grid:**
```python
spec = pls(num_comp=tune())

grid = grid_regular({
    "num_comp": {"range": (2, 20)}
}, levels=10)
```

**Use Cases:**
- Highly correlated predictors (multicollinearity)
- Number of predictors >> number of observations (p >> n)
- Dimension reduction with predictive power (unlike PCA)
- Maximizes covariance between X and y

**Example:**
```python
from py_parsnip import pls

# 2 components
spec = pls(num_comp=2)

# High-dimensional data
spec = pls(num_comp=10)

# Use with workflow
from py_workflows import workflow
wf = workflow().add_formula('y ~ .').add_model(pls(num_comp=5))
```

---

## 3. TREE-BASED MODELS

### 3.1 decision_tree()

**Purpose:** Single decision tree for regression/classification.

**Function Signature:**
```python
decision_tree(
    tree_depth: Optional[int] = None,
    min_n: Optional[int] = None,
    cost_complexity: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `tree_depth` (int or None): Maximum tree depth (default: None = unlimited)
  - Controls tree size/complexity
  - Typical range: 3 to 20
- `min_n` (int or None): Minimum samples required to split node (default: 2)
  - Higher = more conservative splits
- `cost_complexity` (float or None): Pruning parameter (ccp_alpha, default: 0.0)
  - Higher = more aggressive pruning
  - Typical range: 0.001 to 0.1
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Unknown (set via `.set_mode('regression')` or `.set_mode('classification')`)

**Available Engines:** sklearn (DecisionTreeRegressor/Classifier)

**Engine Parameter Mappings:**
- `tree_depth` → `max_depth`
- `min_n` → `min_samples_split`
- `cost_complexity` → `ccp_alpha`

**Tunable Parameters:** `tree_depth`, `min_n`, `cost_complexity`

**Recommended Tuning Grid:**
```python
spec = decision_tree(
    tree_depth=tune(),
    min_n=tune(),
    cost_complexity=tune()
).set_mode('regression')

grid = grid_regular({
    "tree_depth": {"range": (3, 20)},
    "min_n": {"range": (2, 40)},
    "cost_complexity": {"range": (0.001, 0.1), "trans": "log"}
}, levels=5)
```

**Use Cases:**
- Interpretable non-linear models (can visualize tree)
- Feature interactions automatic (no manual specification)
- No feature scaling needed
- Baseline for ensemble methods (random forest, boosting)

**Example:**
```python
from py_parsnip import decision_tree

# Default tree (unlimited depth)
spec = decision_tree().set_mode('regression')

# Constrained depth
spec = decision_tree(tree_depth=5).set_mode('regression')

# With pruning
spec = decision_tree(
    tree_depth=10,
    cost_complexity=0.01
).set_mode('regression')

# Classification
spec = decision_tree(tree_depth=5).set_mode('classification')
```

---

### 3.2 rand_forest()

**Purpose:** Random forest ensemble for robust predictions.

**Function Signature:**
```python
rand_forest(
    mtry: Optional[int] = None,
    trees: Optional[int] = None,
    min_n: Optional[int] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `mtry` (int or None): Features to sample at each split
  - Default: sqrt(n_features) for classification
  - Default: n_features/3 for regression
  - Lower = more randomness/diversity
- `trees` (int or None): Number of trees in forest (default: 500)
  - More trees = more stable, slower
  - Typical range: 100 to 2000
- `min_n` (int or None): Minimum samples to split node (default: 2)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Unknown (set via `.set_mode('regression')` or `.set_mode('classification')`)

**Available Engines:** sklearn (RandomForestRegressor/Classifier)

**Engine Parameter Mappings:**
- `mtry` → `max_features`
- `trees` → `n_estimators`
- `min_n` → `min_samples_split`

**Tunable Parameters:** `mtry`, `trees`, `min_n`

**Recommended Tuning Grid:**
```python
spec = rand_forest(
    mtry=tune(),
    trees=tune(),
    min_n=tune()
).set_mode('regression')

grid = grid_regular({
    "mtry": {"range": (1, 10)},
    "trees": {"range": (100, 1000)},
    "min_n": {"range": (2, 20)}
}, levels=4)
```

**Use Cases:**
- Robust to overfitting (ensemble averaging)
- Handles missing data well
- Feature importance built-in
- Minimal hyperparameter tuning needed
- Excellent out-of-box performance
- Parallel training (fast)

**Example:**
```python
from py_parsnip import rand_forest

# Default random forest
spec = rand_forest().set_mode('regression')

# Custom parameters
spec = rand_forest(
    mtry=5,
    trees=1000,
    min_n=10
).set_mode('regression')

# Classification
spec = rand_forest(trees=500).set_mode('classification')
```

---

### 3.3 bag_tree()

**Purpose:** Bagged decision trees for variance reduction.

**Function Signature:**
```python
bag_tree(
    trees: Optional[int] = None,
    min_n: Optional[int] = None,
    cost_complexity: Optional[float] = None,
    tree_depth: Optional[int] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `trees` (int or None): Number of bootstrap samples (default: 25)
- `min_n` (int or None): Minimum samples to split (default: 2)
- `cost_complexity` (float or None): Pruning parameter (default: 0.0)
- `tree_depth` (int or None): Maximum tree depth (default: None = unlimited)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Unknown (set via `.set_mode()`)

**Available Engines:** sklearn (BaggingRegressor/Classifier with DecisionTree)

**Engine Parameter Mappings:**
- `trees` → `n_estimators`
- `min_n` → `min_samples_split` (base estimator)
- `cost_complexity` → `ccp_alpha` (base estimator)
- `tree_depth` → `max_depth` (base estimator)

**Tunable Parameters:** `trees`, `min_n`, `tree_depth`, `cost_complexity`

**Use Cases:**
- Reduce variance compared to single tree
- Simpler than random forest (no feature sampling)
- Bootstrap aggregating improves stability
- Parallel training

**Example:**
```python
from py_parsnip import bag_tree

# Default bagged tree
spec = bag_tree().set_mode('regression')

# Custom parameters
spec = bag_tree(
    trees=50,
    tree_depth=10,
    min_n=5
).set_mode('classification')
```

---

## 4. GRADIENT BOOSTING

### 4.1 boost_tree()

**Purpose:** Gradient boosting with multiple engine backends.

**Function Signature:**
```python
boost_tree(
    trees: Optional[int] = None,
    tree_depth: Optional[int] = None,
    learn_rate: Optional[float] = None,
    mtry: Optional[int] = None,
    min_n: Optional[int] = None,
    loss_reduction: Optional[float] = None,
    sample_size: Optional[float] = None,
    stop_iter: Optional[int] = None,
    engine: str = "xgboost"
) -> ModelSpec
```

**Parameters:**
- `trees` (int or None): Number of boosting iterations/trees
  - Typical range: 100 to 2000
- `tree_depth` (int or None): Maximum tree depth
  - Lower = simpler, less overfitting
  - Typical range: 3 to 10
- `learn_rate` (float or None): Step size shrinkage (0-1)
  - Lower = more robust, needs more trees
  - Typical range: 0.001 to 0.3
- `mtry` (int or None): Features per split (int or fraction)
  - Controls column sampling
- `min_n` (int or None): Minimum samples in leaf
- `loss_reduction` (float or None): Minimum loss reduction for split
  - XGBoost: gamma parameter
- `sample_size` (float or None): Row sampling fraction (0-1)
  - Controls row subsampling
- `stop_iter` (int or None): Early stopping rounds
- `engine` (str): Computational engine (default: "xgboost")

**Mode:** Regression (default)

**Available Engines:**
1. **xgboost** (default) - XGBoost library
2. **lightgbm** - LightGBM library
3. **catboost** - CatBoost library

**Engine Parameter Mappings:**

**XGBoost:**
- `trees` → `n_estimators`
- `tree_depth` → `max_depth`
- `learn_rate` → `learning_rate`
- `mtry` → `colsample_bytree` (as fraction)
- `min_n` → `min_child_weight`
- `loss_reduction` → `gamma`
- `sample_size` → `subsample`
- `stop_iter` → `early_stopping_rounds`

**LightGBM:**
- `trees` → `n_estimators`
- `tree_depth` → `max_depth`
- `learn_rate` → `learning_rate`
- `mtry` → `colsample_bytree`
- `min_n` → `min_data_in_leaf`
- `loss_reduction` → `min_split_gain`
- `sample_size` → `subsample`

**CatBoost:**
- `trees` → `iterations`
- `tree_depth` → `depth`
- `learn_rate` → `learning_rate`
- `mtry` → `rsm` (random subspace method)
- `min_n` → `min_data_in_leaf`
- `sample_size` → `subsample`

**Tunable Parameters:** All parameters

**Recommended Tuning Grid:**
```python
spec = boost_tree(
    trees=tune(),
    tree_depth=tune(),
    learn_rate=tune(),
    min_n=tune(),
    sample_size=tune()
)

grid = grid_regular({
    "trees": {"range": (100, 1000)},
    "tree_depth": {"range": (3, 10)},
    "learn_rate": {"range": (0.001, 0.3), "trans": "log"},
    "min_n": {"range": (1, 20)},
    "sample_size": {"range": (0.6, 1.0)}
}, levels=4)
```

**Use Cases:**
- State-of-art predictive performance
- Handles missing data automatically
- Feature importance built-in
- Works well with minimal tuning
- Excellent for tabular data
- Kaggle competition winner

**Example:**
```python
from py_parsnip import boost_tree

# XGBoost (default)
spec = boost_tree(trees=100, tree_depth=6, learn_rate=0.1)

# LightGBM
spec = boost_tree(
    trees=100,
    learn_rate=0.1,
    engine="lightgbm"
)

# CatBoost with early stopping
spec = boost_tree(
    trees=1000,
    learn_rate=0.05,
    stop_iter=50,
    engine="catboost"
)
```

---

## 5. SUPPORT VECTOR MACHINES

### 5.1 svm_rbf()

**Purpose:** SVM with Radial Basis Function kernel.

**Function Signature:**
```python
svm_rbf(
    cost: Optional[float] = None,
    rbf_sigma: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `cost` (float or None): Regularization parameter C (default: 1.0)
  - Higher = less regularization (fit training data closely)
  - Lower = more regularization (smoother decision boundary)
  - Typical range: 0.1 to 100
- `rbf_sigma` (float or None): RBF kernel coefficient (gamma)
  - Higher = more influence from nearby points (tighter fit)
  - Lower = more influence from distant points (smoother)
  - Default: "scale" (1 / (n_features * X.var()))
- `margin` (float or None): Epsilon in epsilon-SVR (default: 0.1)
  - Epsilon-tube width (no penalty for errors inside tube)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Regression only

**Available Engines:** sklearn (SVR with rbf kernel)

**Engine Parameter Mappings:**
- `cost` → `C`
- `rbf_sigma` → `gamma`
- `margin` → `epsilon`

**Tunable Parameters:** `cost`, `rbf_sigma`, `margin`

**Recommended Tuning Grid:**
```python
spec = svm_rbf(cost=tune(), rbf_sigma=tune())

grid = grid_regular({
    "cost": {"range": (0.1, 10), "trans": "log"},
    "rbf_sigma": {"range": (0.001, 1), "trans": "log"}
}, levels=5)
```

**Use Cases:**
- Non-linear relationships
- Small to medium datasets
- Works well in high dimensions
- Kernel trick for complex patterns
- Regression with outlier robustness (epsilon tube)

**Example:**
```python
from py_parsnip import svm_rbf

# Default SVM-RBF
spec = svm_rbf()

# Higher cost (less regularization)
spec = svm_rbf(cost=10.0)

# Specific gamma and narrow epsilon tube
spec = svm_rbf(rbf_sigma=0.1, margin=0.05)
```

---

### 5.2 svm_linear()

**Purpose:** SVM with linear kernel.

**Function Signature:**
```python
svm_linear(
    cost: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `cost` (float or None): Regularization parameter C (default: 1.0)
- `margin` (float or None): Epsilon in epsilon-SVR (default: 0.0)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Regression only

**Available Engines:** sklearn (LinearSVR)

**Engine Parameter Mappings:**
- `cost` → `C`
- `margin` → `epsilon`

**Tunable Parameters:** `cost`, `margin`

**Use Cases:**
- Linear relationships
- Faster than RBF kernel
- Scales better to large datasets
- When linear decision boundary sufficient

**Example:**
```python
from py_parsnip import svm_linear

# Default linear SVM
spec = svm_linear()

# Less regularization
spec = svm_linear(cost=10.0)

# Wider epsilon tube
spec = svm_linear(margin=0.2)
```

---

### 5.3 svm_poly()

**Purpose:** SVM with polynomial kernel.

**Function Signature:**
```python
svm_poly(
    cost: Optional[float] = None,
    degree: Optional[int] = None,
    scale_factor: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `cost` (float or None): Regularization parameter C (default: 1.0)
- `degree` (int or None): Polynomial degree (default: 3)
  - 2 = quadratic
  - 3 = cubic
  - Higher = more complex
- `scale_factor` (float or None): Polynomial kernel coefficient (gamma)
  - Default: "scale"
- `margin` (float or None): Epsilon in epsilon-SVR (default: 0.1)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Regression only

**Available Engines:** sklearn (SVR with poly kernel)

**Engine Parameter Mappings:**
- `cost` → `C`
- `degree` → `degree`
- `scale_factor` → `gamma`
- `margin` → `epsilon`

**Tunable Parameters:** `cost`, `degree`, `scale_factor`

**Recommended Tuning Grid:**
```python
spec = svm_poly(cost=tune(), degree=tune())

grid = grid_regular({
    "cost": {"range": (0.1, 10), "trans": "log"},
    "degree": {"range": (2, 5)}
}, levels=4)
```

**Use Cases:**
- Polynomial relationships
- Interaction effects automatic
- More flexible than linear
- Interpretable polynomial features

**Example:**
```python
from py_parsnip import svm_poly

# Default (cubic polynomial)
spec = svm_poly()

# Quadratic
spec = svm_poly(degree=2)

# Quartic with higher cost
spec = svm_poly(cost=10.0, degree=4)
```

---

## 6. INSTANCE-BASED & ADAPTIVE

### 6.1 nearest_neighbor()

**Purpose:** k-Nearest Neighbors regression/classification.

**Function Signature:**
```python
nearest_neighbor(
    neighbors: Optional[int] = None,
    weight_func: Optional[str] = None,
    dist_power: Optional[float] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `neighbors` (int or None): Number of neighbors (default: 5)
  - Lower = more flexible, higher variance
  - Higher = smoother, lower variance
  - Typical range: 3 to 20
- `weight_func` (str or None): Weight function
  - `"uniform"`: All neighbors weighted equally (default)
  - `"distance"`: Closer neighbors have more influence
- `dist_power` (float or None): Minkowski distance power (default: 2)
  - 1 = Manhattan distance (city block)
  - 2 = Euclidean distance
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Unknown (set via `.set_mode()`)

**Available Engines:** sklearn (KNeighborsRegressor/Classifier)

**Engine Parameter Mappings:**
- `neighbors` → `n_neighbors`
- `weight_func` → `weights`
- `dist_power` → `p`

**Tunable Parameters:** `neighbors`, `weight_func`, `dist_power`

**Recommended Tuning Grid:**
```python
spec = nearest_neighbor(
    neighbors=tune(),
    weight_func=tune()
).set_mode('regression')

grid = grid_regular({
    "neighbors": {"range": (3, 15)},
    "weight_func": ["uniform", "distance"]
}, levels=5)
```

**Use Cases:**
- Non-parametric regression
- No training phase (lazy learning)
- Works well with local patterns
- Simple and interpretable
- Good for spatial data

**Example:**
```python
from py_parsnip import nearest_neighbor

# Default k-NN (5 neighbors)
spec = nearest_neighbor().set_mode('regression')

# 10 neighbors with distance weighting
spec = nearest_neighbor(
    neighbors=10,
    weight_func="distance"
).set_mode('regression')

# Manhattan distance
spec = nearest_neighbor(
    neighbors=7,
    dist_power=1
).set_mode('classification')
```

---

### 6.2 mars()

**Purpose:** Multivariate Adaptive Regression Splines.

**Function Signature:**
```python
mars(
    num_terms: Optional[int] = None,
    prod_degree: Optional[int] = None,
    prune_method: Optional[str] = None,
    engine: str = "pyearth"
) -> ModelSpec
```

**Parameters:**
- `num_terms` (int or None): Maximum number of terms (controls complexity)
  - Higher = more complex model
  - Typical range: 5 to 50
- `prod_degree` (int or None): Maximum interaction degree (default: 1)
  - 1 = no interactions (additive model)
  - 2 = pairwise interactions
  - Higher = more complex interactions
- `prune_method` (str or None): Pruning method
  - `"none"`: No pruning
  - `"forward"`: Forward selection only
  - `"backward"`: Forward + backward pruning (recommended)
- `engine` (str): Computational engine (default: "pyearth")

**Mode:** Regression only

**Available Engines:** pyearth

**Tunable Parameters:** `num_terms`, `prod_degree`

**Recommended Tuning Grid:**
```python
spec = mars(num_terms=tune(), prod_degree=tune())

grid = grid_regular({
    "num_terms": {"range": (5, 30)},
    "prod_degree": {"range": (1, 3)}
}, levels=4)
```

**Use Cases:**
- Automatic non-linearity detection
- Piecewise linear functions (hinge functions)
- Feature interactions automatically discovered
- Interpretable splines (can see breakpoints)

**Example:**
```python
from py_parsnip import mars

# Basic MARS
spec = mars()

# Control complexity
spec = mars(num_terms=10, prod_degree=1)

# Allow pairwise interactions
spec = mars(num_terms=20, prod_degree=2, prune_method="backward")
```

**Note:** pyearth has Python 3.10+ compatibility issues. Consider alternatives if needed.

---

### 6.3 mlp()

**Purpose:** Multi-Layer Perceptron neural network.

**Function Signature:**
```python
mlp(
    hidden_units: Optional[Union[int, Tuple[int, ...]]] = None,
    penalty: Optional[float] = None,
    epochs: Optional[int] = None,
    learn_rate: Optional[float] = None,
    activation: Optional[str] = None,
    engine: str = "sklearn"
) -> ModelSpec
```

**Parameters:**
- `hidden_units` (int, tuple, or None): Hidden layer architecture
  - int: Single layer (e.g., 50 means one layer with 50 neurons)
  - tuple: Multiple layers (e.g., (100, 50, 25) means 3 layers)
  - Default: (100,)
- `penalty` (float or None): L2 regularization (alpha, default: 0.0001)
- `epochs` (int or None): Maximum training iterations (default: 200)
- `learn_rate` (float or None): Initial learning rate (default: 0.001)
- `activation` (str or None): Activation function
  - `"relu"` (default) - Rectified Linear Unit
  - `"tanh"` - Hyperbolic tangent
  - `"logistic"` - Sigmoid
  - `"identity"` - Linear (no activation)
- `engine` (str): Computational engine (default: "sklearn")

**Mode:** Unknown (set via `.set_mode()`)

**Available Engines:** sklearn (MLPRegressor/Classifier)

**Engine Parameter Mappings:**
- `hidden_units` → `hidden_layer_sizes`
- `penalty` → `alpha`
- `epochs` → `max_iter`
- `learn_rate` → `learning_rate_init`
- `activation` → `activation`

**Tunable Parameters:** `hidden_units`, `penalty`, `epochs`, `learn_rate`

**Recommended Tuning Grid:**
```python
spec = mlp(
    hidden_units=tune(),
    penalty=tune(),
    epochs=tune()
).set_mode('regression')

grid = grid_regular({
    "hidden_units": [(50,), (100,), (100, 50)],
    "penalty": {"range": (0.0001, 0.1), "trans": "log"},
    "epochs": {"range": (100, 500)}
}, levels=3)
```

**Use Cases:**
- Complex non-linear patterns
- Deep learning for tabular data
- Feature interactions automatic
- Universal function approximation
- Works well with large datasets

**Example:**
```python
from py_parsnip import mlp

# Default MLP (single layer, 100 units)
spec = mlp().set_mode('regression')

# Custom single layer
spec = mlp(hidden_units=50).set_mode('regression')

# Two hidden layers
spec = mlp(hidden_units=(100, 50)).set_mode('regression')

# Three layers with regularization
spec = mlp(
    hidden_units=(100, 50, 25),
    penalty=0.01,
    epochs=500
).set_mode('regression')

# Tanh activation
spec = mlp(activation="tanh").set_mode('classification')
```

---

## 7. RULE-BASED MODELS

### 7.1 rule_fit()

**Purpose:** Interpretable rule-based modeling via RuleFit.

**Function Signature:**
```python
rule_fit(
    max_rules: Optional[int] = None,
    tree_depth: Optional[int] = None,
    penalty: Optional[float] = None,
    tree_generator: Optional[str] = None,
    engine: str = "imodels"
) -> ModelSpec
```

**Parameters:**
- `max_rules` (int or None): Maximum number of rules to use (default: 10)
  - Controls model complexity
- `tree_depth` (int or None): Maximum depth for tree generation (default: 3)
  - Controls rule complexity
- `penalty` (float or None): L1 regularization (alpha, default: 0.0)
  - For sparse rule selection
- `tree_generator` (str or None): Algorithm for tree generation
- `engine` (str): Computational engine (default: "imodels")

**Mode:** Unknown (set via `.set_mode()`)

**Available Engines:** imodels (interpretable models library)

**Engine Parameter Mappings:**
- `max_rules` → `max_rules`
- `tree_depth` → `tree_size`
- `penalty` → `alpha`
- `tree_generator` → `tree_generator`

**Tunable Parameters:** `max_rules`, `tree_depth`, `penalty`

**Use Cases:**
- Interpretable rules extracted from trees
- Combines linear model + rule features
- L1 regularization for sparsity
- Human-readable if-then rules
- Alternative to decision trees for interpretability

**Example:**
```python
from py_parsnip import rule_fit

# Default RuleFit
spec = rule_fit().set_mode('regression')

# More rules
spec = rule_fit(max_rules=20).set_mode('regression')

# Deeper trees for complex rules
spec = rule_fit(
    max_rules=15,
    tree_depth=5
).set_mode('classification')

# With L1 regularization
spec = rule_fit(
    max_rules=30,
    penalty=0.01
).set_mode('regression')
```

---

## 8. TIME SERIES MODELS

### 8.1 arima_reg()

**Purpose:** ARIMA/SARIMA for time series modeling.

**Function Signature:**
```python
arima_reg(
    seasonal_period: Optional[int] = None,
    non_seasonal_ar: int = 0,
    non_seasonal_differences: int = 0,
    non_seasonal_ma: int = 0,
    seasonal_ar: int = 0,
    seasonal_differences: int = 0,
    seasonal_ma: int = 0,
    engine: str = "statsmodels"
) -> ModelSpec
```

**Parameters:**
- `seasonal_period` (int or None): Seasonal cycle length
  - 12 for monthly yearly patterns
  - 7 for weekly patterns in daily data
  - 24 for daily patterns in hourly data
- `non_seasonal_ar` (int): AR order (p) - autoregressive lags (default: 0)
- `non_seasonal_differences` (int): Differencing order (d) - for stationarity (default: 0)
- `non_seasonal_ma` (int): MA order (q) - moving average lags (default: 0)
- `seasonal_ar` (int): Seasonal AR order (P) - seasonal autoregressive (default: 0)
- `seasonal_differences` (int): Seasonal differencing (D) - seasonal stationarity (default: 0)
- `seasonal_ma` (int): Seasonal MA order (Q) - seasonal moving average (default: 0)
- `engine` (str): Computational engine (default: "statsmodels")

**Mode:** Regression only

**Available Engines:**
1. **statsmodels** (default) - Manual ARIMA(p,d,q)(P,D,Q)[m] specification
2. **auto_arima** - Automatic order selection via pmdarima

**Engine Parameter Mappings:**

**statsmodels:**
- Parameters used directly to create ARIMA(p,d,q)(P,D,Q)[m]

**auto_arima:**
- Parameters become MAX constraints for automatic search
- `non_seasonal_ar` → `max_p`
- `non_seasonal_differences` → `max_d`
- `non_seasonal_ma` → `max_q`
- `seasonal_ar` → `max_P`
- `seasonal_differences` → `max_D`
- `seasonal_ma` → `max_Q`
- `seasonal_period` → `m` (exact, not max)

**Tunable Parameters:** All order parameters

**Use Cases:**
- Time series with trend (use d > 0)
- Autocorrelated data
- Seasonal patterns (use P, D, Q, seasonal_period)
- Linear temporal dynamics
- Classic time series forecasting

**Example:**
```python
from py_parsnip import arima_reg

# Simple ARIMA(1,1,1)
spec = arima_reg(
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1
)

# SARIMA(1,1,1)(1,1,1)[12] for monthly data
spec = arima_reg(
    seasonal_period=12,
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    seasonal_ar=1,
    seasonal_differences=1,
    seasonal_ma=1
)

# Auto ARIMA (automatic order selection)
spec = arima_reg().set_engine("auto_arima")

# Auto ARIMA with max constraints
spec = arima_reg(
    non_seasonal_ar=3,  # max_p=3
    non_seasonal_ma=3   # max_q=3
).set_engine("auto_arima")
```

**Note on auto_arima:** pmdarima 2.0.4 has numpy 2.x compatibility issues. Use statsmodels engine or downgrade numpy to 1.26.x.

---

### 8.2 prophet_reg()

**Purpose:** Facebook Prophet for time series with strong seasonality.

**Function Signature:**
```python
prophet_reg(
    growth: Literal["linear", "logistic"] = "linear",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Literal["additive", "multiplicative"] = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    seasonality_yearly: Union[Literal["auto"], bool] = "auto",
    seasonality_weekly: Union[Literal["auto"], bool] = "auto",
    seasonality_daily: Union[Literal["auto"], bool] = "auto",
    engine: str = "prophet"
) -> ModelSpec
```

**Parameters:**
- `growth` (str): Trend model
  - `"linear"` (default) - Linear trend
  - `"logistic"` - Logistic growth (requires cap/floor in data)
- `changepoint_prior_scale` (float): Trend flexibility (default: 0.05)
  - Larger = more flexible/responsive trend
  - Smaller = smoother/more stable trend
  - Typical range: 0.001 to 0.5
- `seasonality_prior_scale` (float): Seasonality flexibility (default: 10.0)
  - Larger = more flexible seasonality
  - Smaller = smoother seasonality
  - Typical range: 0.1 to 20
- `seasonality_mode` (str): Component combination
  - `"additive"`: y = trend + seasonality + holidays + error
  - `"multiplicative"`: y = trend * (1 + seasonality + holidays) + error
- `n_changepoints` (int): Number of potential changepoints (default: 25)
- `changepoint_range` (float): Proportion for changepoints (default: 0.8)
  - Only first 80% of data used for changepoint detection
- `seasonality_yearly` (str/bool): Yearly seasonality
  - `"auto"`: Prophet decides based on data frequency (default)
  - `True`: Force on
  - `False`: Turn off
- `seasonality_weekly` (str/bool): Weekly seasonality toggle
- `seasonality_daily` (str/bool): Daily seasonality toggle
- `engine` (str): Computational engine (default: "prophet")

**Mode:** Regression only

**Available Engines:** prophet (Facebook Prophet)

**Tunable Parameters:** `changepoint_prior_scale`, `seasonality_prior_scale`

**Recommended Tuning Grid:**
```python
spec = prophet_reg(
    changepoint_prior_scale=tune(),
    seasonality_prior_scale=tune()
)

grid = grid_regular({
    "changepoint_prior_scale": {"range": (0.001, 0.5), "trans": "log"},
    "seasonality_prior_scale": {"range": (0.1, 20), "trans": "log"}
}, levels=5)
```

**Use Cases:**
- Strong seasonal patterns (daily/weekly/yearly)
- Multiple seasonality components
- Trend changes (automatic changepoint detection)
- Holiday effects (can add custom holidays)
- Missing data robust
- Human-scale time series (minutes to years)

**Example:**
```python
from py_parsnip import prophet_reg

# Basic Prophet
spec = prophet_reg()

# More flexible trend
spec = prophet_reg(changepoint_prior_scale=0.1)

# Multiplicative seasonality (for growing amplitude)
spec = prophet_reg(seasonality_mode='multiplicative')

# Logistic growth (saturating)
spec = prophet_reg(growth='logistic')

# Turn off all seasonality (for hybrid models)
spec = prophet_reg(
    seasonality_yearly=False,
    seasonality_weekly=False,
    seasonality_daily=False
)

# Very flexible seasonality
spec = prophet_reg(
    seasonality_prior_scale=20.0,
    seasonality_mode='additive'
)
```

---

### 8.3 exp_smoothing()

**Purpose:** Exponential Smoothing (ETS) models.

**Function Signature:**
```python
exp_smoothing(
    seasonal_period: Optional[int] = None,
    error: Optional[Literal["additive", "multiplicative"]] = "additive",
    trend: Optional[Literal["additive", "multiplicative"]] = None,
    season: Optional[Literal["additive", "multiplicative"]] = None,
    damping: bool = False,
    engine: str = "statsmodels"
) -> ModelSpec
```

**Parameters:**
- `seasonal_period` (int or None): Seasonal cycle length (required if season is set)
- `error` (str or None): Error type (default: "additive")
  - `"additive"` or `"multiplicative"`
- `trend` (str or None): Trend component
  - `None` (no trend)
  - `"additive"` (linear trend)
  - `"multiplicative"` (exponential trend)
- `season` (str or None): Seasonal component
  - `None` (no seasonality)
  - `"additive"` (constant seasonal amplitude)
  - `"multiplicative"` (seasonal amplitude scales with level)
- `damping` (bool): Use damped trend (requires trend to be set, default: False)
- `engine` (str): Computational engine (default: "statsmodels")

**Mode:** Regression only

**Available Engines:** statsmodels (ExponentialSmoothing)

**Model Variants:**
- **Simple ES**: trend=None, season=None (SES)
- **Holt's Linear**: trend="additive", season=None
- **Holt-Winters Additive**: trend="additive", season="additive"
- **Holt-Winters Multiplicative**: trend="multiplicative", season="multiplicative"

**Use Cases:**
- Weighted average forecasting
- Smooth time series
- Exponentially decaying weights on past observations
- Level/trend/seasonal decomposition
- When recent past more important than distant past

**Example:**
```python
from py_parsnip import exp_smoothing

# Simple Exponential Smoothing (level only)
spec = exp_smoothing()

# Holt's Linear (level + linear trend)
spec = exp_smoothing(trend="additive")

# Holt-Winters Additive (level + trend + seasonal)
spec = exp_smoothing(
    seasonal_period=12,
    trend="additive",
    season="additive"
)

# Damped Holt-Winters
spec = exp_smoothing(
    seasonal_period=12,
    trend="additive",
    season="multiplicative",
    damping=True
)

# Multiplicative Holt-Winters (for exponential growth)
spec = exp_smoothing(
    seasonal_period=12,
    error="multiplicative",
    trend="multiplicative",
    season="multiplicative"
)
```

---

### 8.4 seasonal_reg()

**Purpose:** STL decomposition + forecasting for complex seasonality.

**Function Signature:**
```python
seasonal_reg(
    seasonal_period_1: Optional[int] = None,
    seasonal_period_2: Optional[int] = None,
    seasonal_period_3: Optional[int] = None,
    engine: str = "statsmodels"
) -> ModelSpec
```

**Parameters:**
- `seasonal_period_1` (int or None): Primary seasonal period (required)
  - 7 for weekly patterns in daily data
  - 12 for yearly patterns in monthly data
  - 24 for daily patterns in hourly data
- `seasonal_period_2` (int or None): Secondary seasonal period (optional)
  - 365 for yearly patterns in daily data (with weekly as primary)
- `seasonal_period_3` (int or None): Tertiary seasonal period (optional)
  - For triple seasonality (rare)
- `engine` (str): Computational engine (default: "statsmodels")

**Mode:** Regression only

**Available Engines:** statsmodels (STL + ARIMA)

**Use Cases:**
- Complex/multiple seasonal patterns
- When SARIMA/Holt-Winters insufficient
- Robust to outliers (LOESS-based)
- Visualize decomposed components (trend, seasonal, residual)
- Non-stationary seasonality

**Example:**
```python
from py_parsnip import seasonal_reg

# Weekly seasonality in daily data
spec = seasonal_reg(seasonal_period_1=7)

# Weekly + yearly in daily data
spec = seasonal_reg(
    seasonal_period_1=7,     # weekly
    seasonal_period_2=365    # yearly
)

# Hourly + daily + weekly patterns
spec = seasonal_reg(
    seasonal_period_1=24,    # daily (hourly data)
    seasonal_period_2=168,   # weekly (24*7)
    seasonal_period_3=8760   # yearly (24*365)
)

# Monthly yearly seasonality
spec = seasonal_reg(seasonal_period_1=12)
```

---

### 8.5 varmax_reg()

**Purpose:** Multivariate time series (Vector ARMA with exogenous variables).

**Function Signature:**
```python
varmax_reg(
    non_seasonal_ar: int = 1,
    non_seasonal_ma: int = 0,
    trend: Literal["n", "c", "t", "ct"] = "c",
    engine: str = "statsmodels"
) -> ModelSpec
```

**Parameters:**
- `non_seasonal_ar` (int): AR order (p) - past time steps (default: 1)
- `non_seasonal_ma` (int): MA order (q) - past errors (default: 0)
- `trend` (str): Trend component
  - `"n"`: No trend
  - `"c"`: Constant (intercept) - default
  - `"t"`: Linear time trend
  - `"ct"`: Both constant and time trend
- `engine` (str): Computational engine (default: "statsmodels")

**Mode:** Regression only

**Available Engines:** statsmodels (VARMAX)

**CRITICAL REQUIREMENT:** Must have 2+ outcome variables in formula

```python
# CORRECT - Multiple outcomes
fit = spec.fit(data, "y1 + y2 ~ date")              # Bivariate
fit = spec.fit(data, "y1 + y2 + y3 ~ date")         # Trivariate
fit = spec.fit(data, "sales + revenue ~ date + x")  # With exogenous

# ERROR - Single outcome not allowed
fit = spec.fit(data, "y ~ date")  # ValueError!
```

**Tunable Parameters:** `non_seasonal_ar`, `non_seasonal_ma`

**Use Cases:**
- Multiple correlated time series
- Cross-variable dynamics (how y1 affects y2)
- Joint forecasting (forecast multiple variables together)
- Interdependent outcomes (Granger causality)
- Multivariate impulse response analysis

**Example:**
```python
from py_parsnip import varmax_reg

# Basic VAR(1) model (no MA component)
spec = varmax_reg(non_seasonal_ar=1, non_seasonal_ma=0)
fit = spec.fit(data, "sales + revenue ~ date")

# VARMA(2,1) model
spec = varmax_reg(non_seasonal_ar=2, non_seasonal_ma=1)
fit = spec.fit(data, "y1 + y2 ~ date")

# VAR(2) with time trend
spec = varmax_reg(
    non_seasonal_ar=2,
    non_seasonal_ma=0,
    trend="ct"
)
fit = spec.fit(data, "price + volume + volatility ~ date")

# With exogenous variables
fit = spec.fit(data, "sales + profit ~ date + marketing_spend")
```

---

## 9. HYBRID TIME SERIES

### 9.1 arima_boost()

**Purpose:** ARIMA + XGBoost hybrid for linear + non-linear patterns.

**Function Signature:**
```python
arima_boost(
    # ARIMA parameters
    seasonal_period: Optional[int] = None,
    non_seasonal_ar: int = 0,
    non_seasonal_differences: int = 0,
    non_seasonal_ma: int = 0,
    seasonal_ar: int = 0,
    seasonal_differences: int = 0,
    seasonal_ma: int = 0,
    # XGBoost parameters
    trees: int = 100,
    tree_depth: int = 6,
    learn_rate: float = 0.1,
    min_n: int = 1,
    loss_reduction: float = 0.0,
    sample_size: float = 1.0,
    mtry: float = 1.0,
    engine: str = "hybrid_arima_xgboost"
) -> ModelSpec
```

**Strategy:**
1. Fit ARIMA to capture linear temporal patterns (trend, seasonality, autocorrelation)
2. Fit XGBoost on ARIMA residuals to capture non-linear patterns and interactions
3. Final prediction = ARIMA prediction + XGBoost prediction

**Parameters:** See arima_reg() and boost_tree() for parameter details

**Mode:** Regression only

**Available Engines:** hybrid_arima_xgboost (custom)

**Tunable Parameters:** All ARIMA and XGBoost parameters

**Use Cases:**
- Data has both linear and non-linear patterns
- ARIMA alone leaves structured residuals
- Capture complex interactions missed by ARIMA
- Best of both worlds (parametric + non-parametric)
- When pure ARIMA underfits

**Example:**
```python
from py_parsnip import arima_boost

# Simple ARIMA(1,1,1) + XGBoost
spec = arima_boost(
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    trees=100,
    tree_depth=3
)

# Seasonal ARIMA + XGBoost
spec = arima_boost(
    seasonal_period=12,
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    seasonal_ar=1,
    seasonal_differences=1,
    seasonal_ma=1,
    trees=200,
    learn_rate=0.05
)

# Light ARIMA + Heavy XGBoost
spec = arima_boost(
    non_seasonal_ar=1,
    trees=500,
    tree_depth=6,
    learn_rate=0.1
)
```

---

### 9.2 prophet_boost()

**Purpose:** Prophet + XGBoost hybrid for seasonality + non-linear patterns.

**Function Signature:**
```python
prophet_boost(
    # Prophet parameters
    growth: Literal["linear", "logistic"] = "linear",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Literal["additive", "multiplicative"] = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    seasonality_yearly: Union[Literal["auto"], bool] = "auto",
    seasonality_weekly: Union[Literal["auto"], bool] = "auto",
    seasonality_daily: Union[Literal["auto"], bool] = "auto",
    # XGBoost parameters
    trees: int = 100,
    tree_depth: int = 6,
    learn_rate: float = 0.1,
    min_n: int = 1,
    loss_reduction: float = 0.0,
    sample_size: float = 1.0,
    mtry: float = 1.0,
    engine: str = "hybrid_prophet_xgboost"
) -> ModelSpec
```

**Strategy:**
1. Fit Prophet to capture trend and seasonality
2. Fit XGBoost on Prophet residuals to capture non-linear patterns
3. Final prediction = Prophet prediction + XGBoost prediction

**Parameters:** See prophet_reg() and boost_tree() for parameter details

**Mode:** Regression only

**Available Engines:** hybrid_prophet_xgboost (custom)

**Tunable Parameters:** All Prophet and XGBoost parameters

**Use Cases:**
- Strong seasonality + non-linear patterns
- Prophet leaves structured residuals
- Capture complex interactions
- Non-linear seasonality (turn off Prophet seasonality, let XGBoost learn it)
- When pure Prophet underfits

**Example:**
```python
from py_parsnip import prophet_boost

# Basic Prophet + XGBoost
spec = prophet_boost(trees=100, tree_depth=3)

# More flexible trend + XGBoost
spec = prophet_boost(
    changepoint_prior_scale=0.1,
    seasonality_mode='multiplicative',
    trees=200,
    learn_rate=0.05
)

# Let XGBoost capture ALL seasonality
spec = prophet_boost(
    seasonality_yearly=False,
    seasonality_weekly=False,
    seasonality_daily=False,
    trees=200,
    tree_depth=6
)

# Prophet trend only, XGBoost for yearly seasonality
spec = prophet_boost(
    seasonality_yearly=False,  # Turn off Prophet yearly
    seasonality_weekly=True,   # Keep weekly
    seasonality_daily=True,    # Keep daily
    trees=150,
    tree_depth=4
)
```

---

## 10. RECURSIVE & WINDOW MODELS

### 10.1 recursive_reg()

**Purpose:** Recursive forecasting using lagged features and ML models.

**Function Signature:**
```python
recursive_reg(
    base_model: ModelSpec,
    lags: Union[int, List[int]] = 1,
    differentiation: Optional[int] = None,
    engine: str = "skforecast"
) -> ModelSpec
```

**Parameters:**
- `base_model` (ModelSpec): Base regression model (must be sklearn-compatible)
  - Examples: rand_forest(), linear_reg(), boost_tree()
  - Must have `.set_mode('regression')` for models requiring mode
- `lags` (int or list): Lagged features to use
  - int: uses lags 1 through n (e.g., 7 = [1,2,3,4,5,6,7])
  - list: specific lags (e.g., [1, 7, 14, 28] for weekly patterns)
- `differentiation` (int or None): Differencing order (None, 1, or 2)
  - For non-stationary data (makes it stationary)
  - 1 = first difference
  - 2 = second difference
- `engine` (str): Computational engine (default: "skforecast")

**Mode:** Regression only

**Available Engines:** skforecast (skforecast library)

**Use Cases:**
- Multi-step time series forecasting
- Apply ML models to time series
- Automatic lagged feature engineering
- Recursive prediction strategy (one-step ahead, then use prediction as input)
- When classic time series models insufficient

**Example:**
```python
from py_parsnip import recursive_reg, rand_forest, linear_reg, boost_tree

# Random Forest with 7 lags
spec = recursive_reg(
    base_model=rand_forest(trees=100).set_mode('regression'),
    lags=7
)

# Linear regression with specific lags (weekly patterns)
spec = recursive_reg(
    base_model=linear_reg(),
    lags=[1, 7, 14, 28]
)

# XGBoost with differencing
spec = recursive_reg(
    base_model=boost_tree(trees=50),
    lags=14,
    differentiation=1  # Make stationary
)

# LightGBM with many lags
spec = recursive_reg(
    base_model=boost_tree(trees=100, engine="lightgbm"),
    lags=list(range(1, 31))  # 30 lags
)
```

---

### 10.2 window_reg()

**Purpose:** Sliding window forecasting (moving average).

**Function Signature:**
```python
window_reg(
    window_size: int = 7,
    method: Literal["mean", "median", "weighted_mean"] = "mean",
    weights: Optional[List[float]] = None,
    min_periods: Optional[int] = None,
    engine: str = "parsnip"
) -> ModelSpec
```

**Parameters:**
- `window_size` (int): Rolling window size (default: 7)
  - Number of recent observations to use
- `method` (str): Aggregation method
  - `"mean"`: Simple moving average (default)
  - `"median"`: Median of window (robust to outliers)
  - `"weighted_mean"`: Weighted average (requires weights)
- `weights` (list or None): Optional weights for weighted_mean
  - Must have length = window_size
  - Should sum to 1.0 (normalized if not)
  - Example: [0.5, 0.3, 0.2] (recent gets more weight)
- `min_periods` (int or None): Minimum observations in window (default: None)
  - If None, uses window_size (no partial windows)
  - If < window_size, allows partial windows at start
- `engine` (str): Computational engine (default: "parsnip")

**Mode:** Regression only

**Available Engines:** parsnip (built-in)

**Use Cases:**
- Simple and interpretable baseline
- Fast computation (no model fitting)
- Smooth time series
- Recent past is best predictor
- Different from naive_reg (more than last value)
- Different from recursive_reg (no ML model, just averaging)

**Example:**
```python
from py_parsnip import window_reg

# Simple 7-day moving average
spec = window_reg(window_size=7, method="mean")

# Median of last 14 observations (robust)
spec = window_reg(window_size=14, method="median")

# Weighted moving average (more weight to recent)
spec = window_reg(
    window_size=3,
    method="weighted_mean",
    weights=[0.5, 0.3, 0.2]  # [most recent, ..., oldest]
)

# Allow partial windows at start
spec = window_reg(
    window_size=7,
    method="mean",
    min_periods=3
)

# Monthly moving average
spec = window_reg(window_size=30, method="mean")
```

---

## 11. GENERIC HYBRID

### 11.1 hybrid_model()

**Purpose:** Combine any two models with flexible strategies.

**Function Signature:**
```python
hybrid_model(
    model1: Optional[ModelSpec] = None,
    model2: Optional[ModelSpec] = None,
    strategy: Literal["residual", "sequential", "weighted", "custom_data"] = "residual",
    weight1: float = 0.5,
    weight2: float = 0.5,
    split_point: Optional[Union[int, float, str]] = None,
    blend_predictions: str = "weighted",
    engine: str = "generic_hybrid"
) -> ModelSpec
```

**Strategies:**

1. **"residual"** (default):
   - Train model1 on y
   - Train model2 on residuals from model1
   - Prediction = model1_pred + model2_pred
   - **Use case:** Capture what model1 misses

2. **"sequential"**:
   - Train model1 on early period (before split_point)
   - Train model2 on later period (after split_point)
   - Use model1 predictions before split, model2 after
   - **Requires:** split_point
   - **Use case:** Regime changes, structural breaks

3. **"weighted"**:
   - Train both models on same data
   - Prediction = weight1 * model1_pred + weight2 * model2_pred
   - **Use case:** Simple ensemble, reduce variance

4. **"custom_data"**:
   - Train model1 on data['model1']
   - Train model2 on data['model2']
   - Datasets can have different/overlapping date ranges
   - **Requires:** Pass dict to fit()
   - **Use case:** Different training periods, adaptive learning

**Parameters:**
- `model1` (ModelSpec or None): First model specification
- `model2` (ModelSpec or None): Second model specification
- `strategy` (str): How to combine models
- `weight1` (float): Weight for model1 (weighted/custom_data, default: 0.5)
- `weight2` (float): Weight for model2 (weighted/custom_data, default: 0.5)
- `split_point` (int/float/str or None): Split location (sequential strategy)
  - int: row index (e.g., 100)
  - float: proportion (e.g., 0.7 for 70% split)
  - str: date string (e.g., "2020-06-01")
- `blend_predictions` (str): Blending method (custom_data strategy)
  - `"weighted"`: weight1 * pred1 + weight2 * pred2 (default)
  - `"avg"`: simple average
  - `"sum"`: sum of predictions
  - `"model1"`: use only model1
  - `"model2"`: use only model2
- `engine` (str): Computational engine (default: "generic_hybrid")

**Mode:** Regression only

**Available Engines:** generic_hybrid (custom)

**Use Cases:**
- Combine linear + non-linear models
- Combine different time series models
- Handle regime changes (sequential)
- Different training periods (custom_data)
- Custom ensembles
- Capture complementary patterns

**Example:**
```python
from py_parsnip import hybrid_model, linear_reg, rand_forest, svm_rbf, decision_tree

# Residual: Linear + Random Forest on residuals
spec = hybrid_model(
    model1=linear_reg(),
    model2=rand_forest().set_mode('regression'),
    strategy="residual"
)
fit = spec.fit(data, 'y ~ x1 + x2')

# Sequential: Different models for different periods
spec = hybrid_model(
    model1=linear_reg(),
    model2=rand_forest().set_mode('regression'),
    strategy="sequential",
    split_point="2020-06-01"  # Structural break date
)

# Weighted: Simple ensemble
spec = hybrid_model(
    model1=linear_reg(),
    model2=svm_rbf(),
    strategy="weighted",
    weight1=0.6,
    weight2=0.4
)

# Custom data: Different training periods
spec = hybrid_model(
    model1=linear_reg(),
    model2=rand_forest().set_mode('regression'),
    strategy="custom_data",
    blend_predictions="weighted",
    weight1=0.4,  # Less weight on older model
    weight2=0.6   # More weight on recent model
)

# Fit with separate datasets (custom_data only)
early_data = df[df['date'] < '2020-07-01']
later_data = df[df['date'] >= '2020-04-01']  # 3 months overlap
fit = spec.fit({'model1': early_data, 'model2': later_data}, 'y ~ x')

# Sequential with proportion split
spec = hybrid_model(
    model1=decision_tree().set_mode('regression'),
    model2=boost_tree(),
    strategy="sequential",
    split_point=0.7  # 70% for model1, 30% for model2
)
```

---

## 12. MANUAL MODELS

### 12.1 manual_reg()

**Purpose:** User-specified coefficients (no fitting required).

**Function Signature:**
```python
manual_reg(
    coefficients: Optional[Dict[str, float]] = None,
    intercept: Optional[float] = None,
    engine: str = "parsnip"
) -> ModelSpec
```

**Parameters:**
- `coefficients` (dict or None): Dictionary mapping variable names to coefficients
  - Example: `{"x1": 2.5, "x2": -1.3, "x3": 0.8}`
  - Default: `{}` (empty)
  - Variables not in dict get coefficient of 0.0
- `intercept` (float or None): Intercept/constant term (default: 0.0)
- `engine` (str): Computational engine (default: "parsnip")

**Mode:** Regression only

**Available Engines:** parsnip (built-in)

**Tunable Parameters:** None (coefficients are user-specified)

**Use Cases:**
- Compare with external forecasts (Excel, R, SAS, legacy systems)
- Test specific coefficient combinations
- Incorporate domain expert knowledge
- Benchmark against known baseline coefficients
- Reproduce legacy model forecasts
- "What-if" scenarios with different coefficients

**Example:**
```python
from py_parsnip import manual_reg

# Domain expert coefficients
spec = manual_reg(
    coefficients={"temperature": 1.5, "humidity": -0.3},
    intercept=10.0
)
fit = spec.fit(data, 'sales ~ temperature + humidity')
predictions = fit.predict(test_data)

# Compare with external model (e.g., from Excel)
external_model = manual_reg(
    coefficients={"marketing_spend": 2.1, "seasonality": 0.8},
    intercept=5.0
)
fit = external_model.fit(train, 'revenue ~ marketing_spend + seasonality')

# Extract outputs for comparison
outputs, coefficients_df, stats = fit.extract_outputs()

# Simple baseline (1:1 relationship)
baseline = manual_reg(
    coefficients={"x": 1.0},
    intercept=0.0
)
fit = baseline.fit(data, 'y ~ x')  # y = x

# Partial specification (missing vars get 0.0)
spec = manual_reg(
    coefficients={"x1": 1.0},  # x2, x3 will be 0.0
    intercept=5.0
)
fit = spec.fit(data, 'y ~ x1 + x2 + x3')  # y = 5 + 1*x1
```

**Note:** Patsy adds "Intercept" column automatically - engine handles this correctly.

---

## SUMMARY TABLES

### Models by Mode

**Regression Only (23 models):**
- Baseline: null_model, naive_reg
- Linear: linear_reg, poisson_reg, gen_additive_mod, pls
- SVM: svm_rbf, svm_linear, svm_poly
- Adaptive: mars
- Boosting: boost_tree
- Time Series: arima_reg, prophet_reg, exp_smoothing, seasonal_reg, varmax_reg
- Hybrid TS: arima_boost, prophet_boost
- Recursive: recursive_reg, window_reg
- Hybrid: hybrid_model
- Manual: manual_reg

**Regression or Classification (6 models):**
- Trees: decision_tree, rand_forest, bag_tree
- Instance: nearest_neighbor, mlp
- Rules: rule_fit

**Setting Mode:**
```python
# For models requiring mode
spec = decision_tree().set_mode('regression')
spec = rand_forest().set_mode('classification')
spec = mlp().set_mode('regression')
```

---

### Models by Use Case

**Essential Baselines (4):**
- null_model, naive_reg, window_reg, manual_reg

**Linear Relationships (4):**
- linear_reg, poisson_reg, pls, svm_linear

**Non-Linear Interpretable (4):**
- gen_additive_mod, mars, decision_tree, rule_fit

**Non-Linear Black Box (6):**
- rand_forest, bag_tree, boost_tree, svm_rbf, svm_poly, mlp, nearest_neighbor

**Time Series Univariate (6):**
- arima_reg, prophet_reg, exp_smoothing, seasonal_reg, naive_reg, window_reg

**Time Series Multivariate (1):**
- varmax_reg

**Hybrid/Ensemble (3):**
- arima_boost, prophet_boost, hybrid_model

**Recursive/Lagged (1):**
- recursive_reg

---

### Models by Tunability

**High Tunability (6+ parameters):**
- boost_tree (8 params)
- prophet_reg (9 params)
- prophet_boost (16 params)
- arima_boost (14 params)
- mlp (5 params)
- hybrid_model (7 params)

**Medium Tunability (3-5 parameters):**
- linear_reg (3)
- rand_forest (3)
- decision_tree (3)
- bag_tree (4)
- svm_rbf (3)
- svm_poly (4)
- arima_reg (7 for SARIMA)
- nearest_neighbor (3)
- rule_fit (3)
- recursive_reg (3)

**Low Tunability (1-2 parameters):**
- gen_additive_mod (2)
- pls (2)
- mars (3)
- exp_smoothing (5 options)
- seasonal_reg (3 periods)
- varmax_reg (3)
- window_reg (4)

**No Tunability:**
- null_model, naive_reg (parameter-free)
- manual_reg (user-specified)

---

## COMMON TUNING PATTERNS

### Conservative Grid (Fast, ~50-100 combinations)
```python
from py_tune import tune, grid_regular

spec = model(param1=tune(), param2=tune())

grid = grid_regular({
    "param1": {"range": (min, max)},
    "param2": {"range": (min, max)}
}, levels=3)  # 3^2 = 9 combinations per fold
```

### Standard Grid (Balanced, ~100-300 combinations)
```python
grid = grid_regular({
    "param1": {"range": (min, max)},
    "param2": {"range": (min, max)},
    "param3": {"range": (min, max)}
}, levels=5)  # 5^3 = 125 combinations per fold
```

### Intensive Grid (Thorough, 500-1000+ combinations)
```python
grid = grid_regular({
    "param1": {"range": (min, max)},
    "param2": {"range": (min, max)},
    "param3": {"range": (min, max)},
    "param4": {"range": (min, max)}
}, levels=8)  # 8^4 = 4096 combinations per fold
```

### Log Transformation (for parameters spanning orders of magnitude)
```python
grid = grid_regular({
    "penalty": {"range": (0.001, 1.0), "trans": "log"},
    "learn_rate": {"range": (0.001, 0.3), "trans": "log"}
}, levels=5)
```

### Random Grid (for high-dimensional spaces)
```python
from py_tune import grid_random

grid = grid_random({
    "trees": {"range": (100, 1000)},
    "tree_depth": {"range": (3, 10)},
    "learn_rate": {"range": (0.001, 0.3), "trans": "log"},
    "mtry": {"range": (1, 10)}
}, size=100)  # 100 random combinations
```

### Typical Tuning Workflow
```python
from py_parsnip import boost_tree
from py_tune import tune, tune_grid, grid_regular
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae

# Define model with tune() placeholders
spec = boost_tree(
    trees=tune(),
    tree_depth=tune(),
    learn_rate=tune()
)

# Create parameter grid
grid = grid_regular({
    "trees": {"range": (100, 1000)},
    "tree_depth": {"range": (3, 10)},
    "learn_rate": {"range": (0.001, 0.3), "trans": "log"}
}, levels=5)

# Create workflow
from py_workflows import workflow
wf = workflow().add_formula('y ~ .').add_model(spec)

# Cross-validation
folds = vfold_cv(train, v=5)

# Tune
results = tune_grid(
    wf,
    resamples=folds,
    grid=grid,
    metrics=metric_set(rmse, mae)
)

# Select best
best_params = results.select_best("rmse", maximize=False)

# Finalize and fit
from py_tune import finalize_workflow
final_wf = finalize_workflow(wf, best_params)
final_fit = final_wf.fit(train)
```

---

**Total Models Documented:** 28
**Last Updated:** 2025-11-09
**Version:** py-tidymodels v1.0
