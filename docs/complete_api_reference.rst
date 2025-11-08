Complete API Reference
======================

This comprehensive reference lists all 23 model types with detailed parameter documentation, available engines, and usage examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Linear & Generalized Models
============================

linear_reg()
------------

**Purpose:** Linear regression for continuous outcomes

**Function Signature:**

.. code-block:: python

   linear_reg(
       penalty: Optional[float] = None,
       mixture: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``penalty`` (Optional[float], default=None):

  - Regularization penalty parameter (L1/L2)
  - Controls model complexity and overfitting
  - Range: [0, ∞)
  - None = no regularization (ordinary least squares)
  - Higher values = stronger regularization
  - Example: ``penalty=0.1`` for moderate regularization

* ``mixture`` (Optional[float], default=None):

  - Mix between L1 (Lasso) and L2 (Ridge) penalties
  - Only used when ``penalty`` is not None
  - Range: [0, 1]
  - 0 = pure Ridge (L2 penalty)
  - 1 = pure Lasso (L1 penalty)
  - 0.5 = Elastic Net (balanced L1/L2)
  - Example: ``mixture=0.5`` for Elastic Net

* ``engine`` (str, default="sklearn"):

  - Computational backend to use
  - Available engines:

    * ``"sklearn"``: scikit-learn LinearRegression/ElasticNet
    * ``"statsmodels"``: statsmodels OLS with statistical inference

  - Example: ``engine="statsmodels"`` for p-values and diagnostics

**Available Engines:**

1. **sklearn** (default):

   - Fast, optimized implementation
   - Supports regularization (penalty/mixture)
   - No statistical inference (p-values, confidence intervals)
   - Best for: prediction, large datasets

2. **statsmodels**:

   - Full statistical inference
   - Provides: p-values, confidence intervals, R², diagnostics
   - No regularization support
   - Best for: hypothesis testing, interpretability

**Usage Examples:**

.. code-block:: python

   # Ordinary Least Squares (no regularization)
   spec = linear_reg()

   # Ridge regression
   spec = linear_reg(penalty=0.1, mixture=0.0)

   # Lasso regression
   spec = linear_reg(penalty=0.1, mixture=1.0)

   # Elastic Net
   spec = linear_reg(penalty=0.1, mixture=0.5)

   # Statsmodels with inference
   spec = linear_reg(engine="statsmodels")
   fit = spec.fit(data, "y ~ x1 + x2")
   _, coefs, _ = fit.extract_outputs()  # Get p-values, CI

**Output Interpretation:**

* **Coefficients DataFrame**: Includes coefficient, std_error, t_stat, p_value, CI, VIF
* **Stats DataFrame**: Includes RMSE, MAE, R², adjusted R², residual diagnostics

---

poisson_reg()
-------------

**Purpose:** Poisson regression for count data (non-negative integers)

**Function Signature:**

.. code-block:: python

   poisson_reg(
       penalty: Optional[float] = None,
       mixture: Optional[float] = None,
       engine: str = "statsmodels"
   ) -> ModelSpec

**Parameters:**

* ``penalty`` (Optional[float], default=None):

  - L1/L2 regularization penalty
  - Same as ``linear_reg()``
  - Controls overfitting in Poisson models

* ``mixture`` (Optional[float], default=None):

  - L1/L2 mixture ratio
  - Same as ``linear_reg()``

* ``engine`` (str, default="statsmodels"):

  - Available engines:

    * ``"statsmodels"``: GLM with Poisson family

**Use Cases:**

- Count outcomes: number of events, purchases, visits
- Rate modeling: events per time period
- Always non-negative predictions
- Variance proportional to mean

**Usage Example:**

.. code-block:: python

   spec = poisson_reg()
   fit = spec.fit(count_data, "num_purchases ~ age + income")

---

gen_additive_mod()
------------------

**Purpose:** Generalized Additive Models with smooth non-linear relationships

**Function Signature:**

.. code-block:: python

   gen_additive_mod(
       select_features: bool = False,
       adjust_deg_free: Optional[str] = None,
       engine: str = "pygam"
   ) -> ModelSpec

**Parameters:**

* ``select_features`` (bool, default=False):

  - Enable automatic feature selection
  - True = GAM will select relevant features via regularization
  - False = use all features in formula
  - Example: ``select_features=True`` for high-dimensional data

* ``adjust_deg_free`` (Optional[str], default=None):

  - Method for adjusting degrees of freedom
  - Options: ``"GCV"``, ``"AIC"``, ``"BIC"``, ``None``
  - Controls smoothness of fitted curves
  - Example: ``adjust_deg_free="GCV"`` for cross-validated smoothing

* ``engine`` (str, default="pygam"):

  - Available engines:

    * ``"pygam"``: Python Generalized Additive Models

**Use Cases:**

- Non-linear relationships between predictors and outcome
- Interpretable non-linear models (smoother curves)
- When linearity assumption doesn't hold

**Usage Example:**

.. code-block:: python

   spec = gen_additive_mod(select_features=True, adjust_deg_free="GCV")
   fit = spec.fit(data, "y ~ x1 + x2 + x3")

---

Tree-Based Models
=================

decision_tree()
---------------

**Purpose:** Single decision tree for regression or classification

**Function Signature:**

.. code-block:: python

   decision_tree(
       tree_depth: Optional[int] = None,
       min_n: Optional[int] = None,
       cost_complexity: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``tree_depth`` (Optional[int], default=None):

  - Maximum depth of the tree
  - Controls model complexity and overfitting
  - Range: [1, ∞)
  - None = unlimited depth (grows until min_n is reached)
  - Smaller values = simpler model, less overfitting
  - Example: ``tree_depth=5`` for moderate complexity

* ``min_n`` (Optional[int], default=None):

  - Minimum number of samples required to split a node
  - Prevents overfitting by stopping early splits
  - Range: [2, ∞)
  - None = sklearn default (2 for regression, 1 for classification)
  - Larger values = simpler model
  - Example: ``min_n=20`` to prevent small splits

* ``cost_complexity`` (Optional[float], default=None):

  - Complexity parameter for pruning (alpha in sklearn)
  - Controls tree size via pruning
  - Range: [0, ∞)
  - None = no pruning
  - Higher values = more aggressive pruning
  - Example: ``cost_complexity=0.01`` for moderate pruning

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn DecisionTreeRegressor/Classifier

**Mode Setting:**

Unlike ``rand_forest()``, ``decision_tree()`` requires ``set_mode()`` call:

.. code-block:: python

   # Regression
   spec = decision_tree(tree_depth=5).set_mode("regression")

   # Classification
   spec = decision_tree(tree_depth=5).set_mode("classification")

**Usage Examples:**

.. code-block:: python

   # Simple regression tree
   spec = decision_tree(tree_depth=5, min_n=10).set_mode("regression")

   # Classification with pruning
   spec = decision_tree(
       tree_depth=8,
       min_n=5,
       cost_complexity=0.01
   ).set_mode("classification")

---

rand_forest()
-------------

**Purpose:** Random Forest ensemble for regression or classification

**Function Signature:**

.. code-block:: python

   rand_forest(
       mtry: Optional[int] = None,
       trees: int = 100,
       min_n: Optional[int] = None,
       mode: str = "unknown",
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``mtry`` (Optional[int], default=None):

  - Number of predictors randomly sampled at each split
  - Controls diversity between trees
  - Range: [1, total_predictors]
  - None = sqrt(p) for classification, p/3 for regression
  - Lower values = more diverse trees, less correlation
  - Example: ``mtry=5`` when you have 15 total predictors

* ``trees`` (int, default=100):

  - Number of trees in the forest
  - More trees = better performance but slower
  - Range: [1, ∞)
  - Typical values: 100-500
  - Example: ``trees=500`` for better accuracy

* ``min_n`` (Optional[int], default=None):

  - Minimum samples to split a node
  - Same as ``decision_tree()``
  - Controls individual tree complexity

* ``mode`` (str, default="unknown"):

  - Task type: ``"regression"`` or ``"classification"``
  - Can also use ``.set_mode()`` method
  - Example: ``mode="regression"`` or ``.set_mode("regression")``

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn RandomForestRegressor/Classifier

**Usage Examples:**

.. code-block:: python

   # Regression forest
   spec = rand_forest(mtry=5, trees=300, mode="regression")

   # Classification forest with set_mode
   spec = rand_forest(trees=500).set_mode("classification")

---

boost_tree()
------------

**Purpose:** Gradient boosting trees (XGBoost, LightGBM, CatBoost)

**Function Signature:**

.. code-block:: python

   boost_tree(
       mtry: Optional[int] = None,
       trees: int = 15,
       min_n: Optional[int] = None,
       tree_depth: Optional[int] = None,
       learn_rate: float = 0.3,
       loss_reduction: Optional[float] = None,
       sample_size: Optional[float] = None,
       stop_iter: Optional[int] = None,
       mode: str = "unknown",
       engine: str = "xgboost"
   ) -> ModelSpec

**Parameters:**

* ``mtry`` (Optional[int], default=None):

  - Number of predictors sampled per tree
  - Same concept as ``rand_forest()``
  - None = use all features

* ``trees`` (int, default=15):

  - Number of boosting iterations
  - More trees = potentially better fit but risk overfitting
  - Range: [1, ∞)
  - Typical values: 50-1000
  - Example: ``trees=100`` for moderate complexity

* ``min_n`` (Optional[int], default=None):

  - Minimum samples in terminal nodes
  - Controls leaf size and overfitting
  - Higher values = simpler model

* ``tree_depth`` (Optional[int], default=None):

  - Maximum depth of each tree
  - Typical values: 3-8
  - Shallow trees (3-4) often work well for boosting
  - Example: ``tree_depth=4`` for moderate depth

* ``learn_rate`` (float, default=0.3):

  - Learning rate (shrinkage)
  - Controls how much each tree contributes
  - Range: (0, 1]
  - Lower values = more robust but need more trees
  - Typical values: 0.01-0.3
  - Example: ``learn_rate=0.1`` with ``trees=500``

* ``loss_reduction`` (Optional[float], default=None):

  - Minimum loss reduction for split (gamma in XGBoost)
  - Controls tree growth
  - Range: [0, ∞)
  - Higher values = more conservative

* ``sample_size`` (Optional[float], default=None):

  - Fraction of data sampled per tree
  - Range: (0, 1]
  - Example: ``sample_size=0.8`` for 80% subsampling

* ``stop_iter`` (Optional[int], default=None):

  - Early stopping rounds
  - Stops if no improvement for N rounds
  - Requires validation set
  - Example: ``stop_iter=10`` to stop after 10 rounds without improvement

* ``mode`` (str, default="unknown"):

  - ``"regression"`` or ``"classification"``

* ``engine`` (str, default="xgboost"):

  - Available engines:

    * ``"xgboost"``: XGBoost library
    * ``"lightgbm"``: LightGBM library
    * ``"catboost"``: CatBoost library

**Engine Comparison:**

1. **XGBoost**: Most popular, excellent performance, GPU support
2. **LightGBM**: Faster training, lower memory, large datasets
3. **CatBoost**: Best for categorical features, less tuning needed

**Usage Examples:**

.. code-block:: python

   # XGBoost regression
   spec = boost_tree(
       trees=100,
       tree_depth=4,
       learn_rate=0.1,
       mode="regression"
   ).set_engine("xgboost")

   # LightGBM with early stopping
   spec = boost_tree(
       trees=500,
       tree_depth=6,
       learn_rate=0.05,
       stop_iter=20,
       mode="regression"
   ).set_engine("lightgbm")

---

Support Vector Machines
========================

svm_rbf()
---------

**Purpose:** Support Vector Machine with Radial Basis Function kernel

**Function Signature:**

.. code-block:: python

   svm_rbf(
       cost: Optional[float] = None,
       rbf_sigma: Optional[float] = None,
       margin: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``cost`` (Optional[float], default=None):

  - Regularization parameter (C in sklearn)
  - Controls trade-off between margin and misclassification
  - Range: (0, ∞)
  - Higher values = harder margin, less regularization
  - Lower values = softer margin, more regularization
  - Example: ``cost=1.0`` for balanced approach

* ``rbf_sigma`` (Optional[float], default=None):

  - RBF kernel bandwidth parameter (gamma in sklearn)
  - Controls influence of single training example
  - Range: (0, ∞)
  - Higher values = tighter fit, more complex
  - Lower values = smoother fit, simpler
  - None = sklearn default (1 / n_features)
  - Example: ``rbf_sigma=0.1`` for smoother decision boundary

* ``margin`` (Optional[float], default=None):

  - Epsilon in epsilon-SVR for regression
  - Width of epsilon-insensitive tube
  - Predictions within this margin have zero loss
  - Example: ``margin=0.1`` for regression

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn SVC/SVR

**Mode Setting:**

Requires ``set_mode()`` call:

.. code-block:: python

   # Regression
   spec = svm_rbf(cost=1.0, rbf_sigma=0.1).set_mode("regression")

   # Classification
   spec = svm_rbf(cost=10.0, rbf_sigma=0.5).set_mode("classification")

**Usage Examples:**

.. code-block:: python

   # Regression with moderate complexity
   spec = svm_rbf(cost=1.0, rbf_sigma=0.1, margin=0.1).set_mode("regression")

   # Classification with tight decision boundary
   spec = svm_rbf(cost=10.0, rbf_sigma=1.0).set_mode("classification")

---

svm_linear()
------------

**Purpose:** Support Vector Machine with linear kernel

**Function Signature:**

.. code-block:: python

   svm_linear(
       cost: Optional[float] = None,
       margin: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``cost`` (Optional[float], default=None):

  - Same as ``svm_rbf()``
  - Regularization parameter

* ``margin`` (Optional[float], default=None):

  - Same as ``svm_rbf()``
  - For regression only

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn LinearSVC/LinearSVR

**Usage Example:**

.. code-block:: python

   spec = svm_linear(cost=1.0).set_mode("classification")

---

Instance-Based Models
=====================

nearest_neighbor()
------------------

**Purpose:** k-Nearest Neighbors for regression or classification

**Function Signature:**

.. code-block:: python

   nearest_neighbor(
       neighbors: int = 5,
       weight_func: str = "uniform",
       dist_power: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``neighbors`` (int, default=5):

  - Number of neighbors (k)
  - Higher values = smoother predictions
  - Lower values = more complex, captures local patterns
  - Range: [1, ∞)
  - Odd numbers recommended for classification (avoid ties)
  - Example: ``neighbors=10`` for smoother predictions

* ``weight_func`` (str, default="uniform"):

  - Weighting function for neighbors
  - Options:

    * ``"uniform"``: All neighbors weighted equally
    * ``"distance"``: Closer neighbors weighted more heavily

  - Example: ``weight_func="distance"`` for distance-weighted k-NN

* ``dist_power`` (Optional[float], default=None):

  - Minkowski distance power parameter
  - Range: [1, ∞)
  - 1 = Manhattan distance
  - 2 = Euclidean distance (default)
  - Example: ``dist_power=1`` for Manhattan distance

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn KNeighborsRegressor/Classifier

**Mode Setting:**

Requires ``set_mode()`` call:

.. code-block:: python

   spec = nearest_neighbor(neighbors=10).set_mode("regression")

**Usage Examples:**

.. code-block:: python

   # Regression with distance weighting
   spec = nearest_neighbor(
       neighbors=10,
       weight_func="distance"
   ).set_mode("regression")

   # Classification with 5 neighbors
   spec = nearest_neighbor(neighbors=5).set_mode("classification")

---

mars()
------

**Purpose:** Multivariate Adaptive Regression Splines

**Function Signature:**

.. code-block:: python

   mars(
       num_terms: Optional[int] = None,
       prod_degree: Optional[int] = None,
       prune_method: str = "backward",
       engine: str = "pyearth"
   ) -> ModelSpec

**Parameters:**

* ``num_terms`` (Optional[int], default=None):

  - Maximum number of terms in model
  - Controls model complexity
  - None = unlimited (pruned by GCV)
  - Example: ``num_terms=20`` for moderate complexity

* ``prod_degree`` (Optional[int], default=None):

  - Maximum degree of interaction terms
  - 1 = additive model (no interactions)
  - 2 = pairwise interactions
  - Higher values = more complex interactions
  - Example: ``prod_degree=2`` for pairwise interactions

* ``prune_method`` (str, default="backward"):

  - Pruning strategy
  - Options: ``"backward"``, ``"none"``
  - ``"backward"`` = remove terms via cross-validation

* ``engine`` (str, default="pyearth"):

  - Available engines:

    * ``"pyearth"``: py-earth library

**Usage Example:**

.. code-block:: python

   spec = mars(num_terms=20, prod_degree=2)

---

mlp()
-----

**Purpose:** Multi-Layer Perceptron neural network

**Function Signature:**

.. code-block:: python

   mlp(
       hidden_units: Optional[int] = None,
       penalty: Optional[float] = None,
       epochs: int = 100,
       activation: str = "relu",
       learn_rate: Optional[float] = None,
       engine: str = "sklearn"
   ) -> ModelSpec

**Parameters:**

* ``hidden_units`` (Optional[int], default=None):

  - Number of neurons in hidden layer
  - Controls network capacity
  - None = 100 neurons
  - Example: ``hidden_units=50`` for moderate network

* ``penalty`` (Optional[float], default=None):

  - L2 regularization (alpha in sklearn)
  - Controls overfitting
  - Example: ``penalty=0.001`` for light regularization

* ``epochs`` (int, default=100):

  - Number of training iterations
  - More epochs = potentially better fit
  - Example: ``epochs=200`` for more training

* ``activation`` (str, default="relu"):

  - Activation function
  - Options: ``"relu"``, ``"tanh"``, ``"logistic"``
  - ``"relu"`` = Rectified Linear Unit (most common)

* ``learn_rate`` (Optional[float], default=None):

  - Learning rate for optimizer
  - Controls gradient descent step size
  - Example: ``learn_rate=0.001`` for slower learning

* ``engine`` (str, default="sklearn"):

  - Available engines:

    * ``"sklearn"``: scikit-learn MLPRegressor/Classifier

**Mode Setting:**

Requires ``set_mode()`` call:

.. code-block:: python

   spec = mlp(hidden_units=100, epochs=200).set_mode("regression")

**Usage Example:**

.. code-block:: python

   spec = mlp(
       hidden_units=50,
       penalty=0.001,
       epochs=200,
       activation="relu"
   ).set_mode("regression")

---

Time Series Models
==================

arima_reg()
-----------

**Purpose:** ARIMA/SARIMAX models for time series forecasting

**Function Signature:**

.. code-block:: python

   arima_reg(
       non_seasonal_ar: int = 0,
       non_seasonal_differences: int = 0,
       non_seasonal_ma: int = 0,
       seasonal_ar: int = 0,
       seasonal_differences: int = 0,
       seasonal_ma: int = 0,
       seasonal_period: Optional[int] = None,
       engine: str = "statsmodels"
   ) -> ModelSpec

**Parameters:**

* ``non_seasonal_ar`` (int, default=0):

  - Non-seasonal autoregressive order (p)
  - Number of lagged observations to use
  - Range: [0, ∞)
  - Example: ``non_seasonal_ar=2`` uses y(t-1) and y(t-2)

* ``non_seasonal_differences`` (int, default=0):

  - Non-seasonal differencing order (d)
  - Number of differencing operations for stationarity
  - Range: [0, 2]
  - 0 = stationary series
  - 1 = trend removal (most common)
  - 2 = second-order differencing

* ``non_seasonal_ma`` (int, default=0):

  - Non-seasonal moving average order (q)
  - Number of lagged forecast errors
  - Range: [0, ∞)
  - Example: ``non_seasonal_ma=1`` uses previous error term

* ``seasonal_ar`` (int, default=0):

  - Seasonal autoregressive order (P)
  - Seasonal lag component
  - Example: ``seasonal_ar=1`` for seasonal AR

* ``seasonal_differences`` (int, default=0):

  - Seasonal differencing order (D)
  - Remove seasonal patterns
  - Range: [0, 1]
  - Example: ``seasonal_differences=1`` for seasonal differencing

* ``seasonal_ma`` (int, default=0):

  - Seasonal moving average order (Q)
  - Seasonal error component

* ``seasonal_period`` (Optional[int], default=None):

  - Seasonal period length (m)
  - Number of observations per season
  - Examples:

    * Daily data with weekly pattern: ``seasonal_period=7``
    * Monthly data with yearly pattern: ``seasonal_period=12``
    * Hourly data with daily pattern: ``seasonal_period=24``

* ``engine`` (str, default="statsmodels"):

  - Available engines:

    * ``"statsmodels"``: statsmodels SARIMAX (exact parameters)
    * ``"auto_arima"``: pmdarima auto_arima (parameters become MAX constraints)

**Engine Behavior:**

1. **statsmodels** (exact specification):

   .. code-block:: python

      spec = arima_reg(non_seasonal_ar=2, non_seasonal_ma=1)
      # Fits ARIMA(2,0,1) exactly

2. **auto_arima** (automatic selection):

   .. code-block:: python

      spec = arima_reg(non_seasonal_ar=5).set_engine("auto_arima")
      # Searches for best p in [0, 5] using AIC/BIC

**ARIMA Order Notation:**

* ARIMA(p,d,q): Non-seasonal components
* SARIMA(p,d,q)(P,D,Q,m): With seasonal components

**Usage Examples:**

.. code-block:: python

   # Simple ARIMA(1,1,1)
   spec = arima_reg(
       non_seasonal_ar=1,
       non_seasonal_differences=1,
       non_seasonal_ma=1
   )

   # Seasonal ARIMA(1,1,1)(1,1,1,12)
   spec = arima_reg(
       non_seasonal_ar=1,
       non_seasonal_differences=1,
       non_seasonal_ma=1,
       seasonal_ar=1,
       seasonal_differences=1,
       seasonal_ma=1,
       seasonal_period=12
   )

   # Auto ARIMA (search up to ARIMA(5,2,5))
   spec = arima_reg(
       non_seasonal_ar=5,
       non_seasonal_differences=2,
       non_seasonal_ma=5
   ).set_engine("auto_arima")

---

prophet_reg()
-------------

**Purpose:** Facebook Prophet for time series with strong seasonality and holidays

**Function Signature:**

.. code-block:: python

   prophet_reg(
       growth: str = "linear",
       n_changepoints: int = 25,
       changepoint_range: float = 0.8,
       changepoint_prior_scale: float = 0.05,
       seasonality_mode: str = "additive",
       seasonality_prior_scale: float = 10.0,
       holidays_prior_scale: float = 10.0,
       yearly_seasonality: Union[bool, str, int] = "auto",
       weekly_seasonality: Union[bool, str, int] = "auto",
       daily_seasonality: Union[bool, str, int] = "auto",
       engine: str = "prophet"
   ) -> ModelSpec

**Parameters:**

* ``growth`` (str, default="linear"):

  - Trend model type
  - Options:

    * ``"linear"``: Linear trend with changepoints
    * ``"logistic"``: Saturating growth (requires cap in data)

  - Example: ``growth="logistic"`` for market saturation

* ``n_changepoints`` (int, default=25):

  - Number of potential trend changepoints
  - More changepoints = more flexible trend
  - Range: [0, ∞)
  - Example: ``n_changepoints=50`` for more flexibility

* ``changepoint_range`` (float, default=0.8):

  - Proportion of history in which changepoints can occur
  - Range: (0, 1]
  - 0.8 = changepoints in first 80% of data
  - Prevents overfitting to recent data

* ``changepoint_prior_scale`` (float, default=0.05):

  - Regularization for trend flexibility
  - Range: (0, ∞)
  - Higher values = more flexible trend
  - Lower values = smoother trend
  - Example: ``changepoint_prior_scale=0.5`` for more flexible

* ``seasonality_mode`` (str, default="additive"):

  - Seasonal component type
  - Options:

    * ``"additive"``: Constant seasonal effect
    * ``"multiplicative"``: Seasonal effect proportional to trend

  - Example: ``seasonality_mode="multiplicative"`` when seasonal variation increases with level

* ``seasonality_prior_scale`` (float, default=10.0):

  - Regularization for seasonality strength
  - Range: (0, ∞)
  - Higher values = stronger seasonal patterns
  - Lower values = weaker seasonal patterns

* ``holidays_prior_scale`` (float, default=10.0):

  - Regularization for holiday effects
  - Same interpretation as ``seasonality_prior_scale``

* ``yearly_seasonality`` (Union[bool, str, int], default="auto"):

  - Yearly seasonal component
  - Options:

    * ``"auto"``: Enabled if data spans >2 years
    * ``True``: Always enabled
    * ``False``: Disabled
    * int: Custom Fourier order (flexibility)

  - Example: ``yearly_seasonality=10`` for more flexible yearly pattern

* ``weekly_seasonality`` (Union[bool, str, int], default="auto"):

  - Weekly seasonal component
  - Same options as ``yearly_seasonality``

* ``daily_seasonality`` (Union[bool, str, int], default="auto"):

  - Daily seasonal component (for sub-daily data)
  - Same options as ``yearly_seasonality``

* ``engine`` (str, default="prophet"):

  - Available engines:

    * ``"prophet"``: Facebook Prophet library

**Usage Examples:**

.. code-block:: python

   # Standard Prophet with daily data
   spec = prophet_reg(
       n_changepoints=25,
       changepoint_prior_scale=0.05,
       seasonality_prior_scale=10.0,
       yearly_seasonality="auto",
       weekly_seasonality="auto"
   )

   # More flexible Prophet
   spec = prophet_reg(
       n_changepoints=50,
       changepoint_prior_scale=0.5,
       seasonality_mode="multiplicative"
   )

   # Logistic growth with capacity
   spec = prophet_reg(growth="logistic")
   # Requires 'cap' column in data

---

exp_smoothing()
---------------

**Purpose:** Exponential Smoothing (ETS) for trend and seasonal patterns

**Function Signature:**

.. code-block:: python

   exp_smoothing(
       seasonal_period: Optional[int] = None,
       error: str = "add",
       trend: str = "add",
       season: str = "add",
       damped_trend: bool = False,
       engine: str = "statsmodels"
   ) -> ModelSpec

**Parameters:**

* ``seasonal_period`` (Optional[int], default=None):

  - Seasonal period length
  - Same as ``arima_reg()``
  - Examples: 12 for monthly, 7 for daily/weekly

* ``error`` (str, default="add"):

  - Error component type
  - Options: ``"add"`` (additive), ``"mul"`` (multiplicative)

* ``trend`` (str, default="add"):

  - Trend component type
  - Options: ``"add"``, ``"mul"``, ``"none"``

* ``season`` (str, default="add"):

  - Seasonal component type
  - Options: ``"add"``, ``"mul"``, ``"none"``

* ``damped_trend`` (bool, default=False):

  - Apply damping to trend
  - True = trend flattens over time
  - False = trend continues indefinitely

* ``engine`` (str, default="statsmodels"):

  - Available engines:

    * ``"statsmodels"``: statsmodels ExponentialSmoothing

**ETS Models:**

* ETS(A,A,A): Additive error, trend, season
* ETS(M,A,M): Multiplicative error/season, additive trend
* etc.

**Usage Example:**

.. code-block:: python

   spec = exp_smoothing(
       seasonal_period=12,
       error="add",
       trend="add",
       season="add",
       damped_trend=True
   )

---

seasonal_reg()
--------------

**Purpose:** STL (Seasonal-Trend decomposition using Loess) models

**Function Signature:**

.. code-block:: python

   seasonal_reg(
       seasonal_period_1: Optional[int] = None,
       seasonal_period_2: Optional[int] = None,
       seasonal_period_3: Optional[int] = None,
       engine: str = "statsmodels"
   ) -> ModelSpec

**Parameters:**

* ``seasonal_period_1`` (Optional[int], default=None):

  - Primary seasonal period
  - Example: ``seasonal_period_1=7`` for weekly pattern in daily data

* ``seasonal_period_2`` (Optional[int], default=None):

  - Secondary seasonal period (for multiple seasonality)
  - Example: ``seasonal_period_2=365`` for yearly pattern

* ``seasonal_period_3`` (Optional[int], default=None):

  - Tertiary seasonal period

* ``engine`` (str, default="statsmodels"):

  - Available engines:

    * ``"statsmodels"``: statsmodels STL/MSTL

**Multiple Seasonality:**

When multiple periods specified, uses MSTL (Multiple STL):

.. code-block:: python

   spec = seasonal_reg(
       seasonal_period_1=7,     # Weekly
       seasonal_period_2=365    # Yearly
   )

**Usage Example:**

.. code-block:: python

   # Single seasonality
   spec = seasonal_reg(seasonal_period_1=12)

   # Multiple seasonality (daily data)
   spec = seasonal_reg(
       seasonal_period_1=7,     # Weekly
       seasonal_period_2=365    # Yearly
   )

---

recursive_reg()
---------------

**Purpose:** Recursive multi-step forecasting with any sklearn-compatible model

**Function Signature:**

.. code-block:: python

   recursive_reg(
       lags: Union[int, List[int]] = 3,
       differentiation: Optional[int] = None,
       base_model: Optional[ModelSpec] = None,
       engine: str = "skforecast"
   ) -> ModelSpec

**Parameters:**

* ``lags`` (Union[int, List[int]], default=3):

  - Lag configuration for autoregressive features
  - Options:

    * int: Use lags 1 through N (e.g., ``lags=7`` → lags [1,2,3,4,5,6,7])
    * List[int]: Specific lags (e.g., ``lags=[1,7,14]`` for weekly patterns)

  - Example: ``lags=[1, 7, 14, 28]`` for 1-day, 1-week, 2-week, 4-week lags

* ``differentiation`` (Optional[int], default=None):

  - Order of differencing for non-stationary series
  - Range: [0, 2]
  - None = no differencing
  - 1 = first-order differencing (most common)
  - Example: ``differentiation=1`` to remove trend

* ``base_model`` (Optional[ModelSpec], default=None):

  - Underlying model for forecasting
  - None = linear regression
  - Can use any sklearn-compatible model
  - Examples:

    * ``base_model=linear_reg()``
    * ``base_model=rand_forest(trees=100, mode="regression")``
    * ``base_model=boost_tree(trees=50, mode="regression")``

* ``engine`` (str, default="skforecast"):

  - Available engines:

    * ``"skforecast"``: skforecast library

**How Recursive Forecasting Works:**

1. Creates lagged features from target variable
2. Trains model: y(t) ~ y(t-1), y(t-2), ..., y(t-k)
3. For multi-step ahead forecasting:

   - Step 1: Predict y(t+1) using actual lags
   - Step 2: Use predicted y(t+1) to predict y(t+2)
   - Step 3: Continue recursively

**Usage Examples:**

.. code-block:: python

   # Simple AR(7) with linear regression
   spec = recursive_reg(lags=7)

   # Custom lags with Random Forest
   spec = recursive_reg(
       lags=[1, 7, 14, 28],
       base_model=rand_forest(trees=100, mode="regression")
   )

   # With differencing for non-stationary series
   spec = recursive_reg(
       lags=14,
       differentiation=1,
       base_model=boost_tree(trees=50, mode="regression")
   )

---

varmax_reg()
------------

**Purpose:** Vector Autoregression with Moving Average (multivariate time series)

**Function Signature:**

.. code-block:: python

   varmax_reg(
       non_seasonal_ar: int = 1,
       non_seasonal_ma: int = 0,
       trend: str = "c",
       engine: str = "statsmodels"
   ) -> ModelSpec

**Parameters:**

* ``non_seasonal_ar`` (int, default=1):

  - VAR order (p)
  - Number of lagged observations
  - Applies to ALL outcome variables

* ``non_seasonal_ma`` (int, default=0):

  - VMA order (q)
  - Number of lagged errors

* ``trend`` (str, default="c"):

  - Trend specification
  - Options:

    * ``"n"``: No trend
    * ``"c"``: Constant (intercept)
    * ``"ct"``: Constant + time trend
    * ``"ctt"``: Constant + time trend + quadratic trend

* ``engine`` (str, default="statsmodels"):

  - Available engines:

    * ``"statsmodels"``: statsmodels VARMAX

**Critical Requirement:**

VARMAX requires **at least 2 outcome variables** in the formula:

.. code-block:: python

   # CORRECT - Multiple outcomes
   fit = spec.fit(data, "y1 + y2 ~ date")           # Bivariate
   fit = spec.fit(data, "y1 + y2 + y3 ~ date")      # Trivariate

   # ERROR - Single outcome
   fit = spec.fit(data, "y ~ date")  # Raises ValueError

**Multi-Outcome Predictions:**

Predictions include separate columns for each outcome:

.. code-block:: python

   predictions = fit.predict(forecast_data)
   # Returns: .pred_y1, .pred_y2 columns

   predictions = fit.predict(forecast_data, type="conf_int")
   # Returns: .pred_y1, .pred_y1_lower, .pred_y1_upper
   #          .pred_y2, .pred_y2_lower, .pred_y2_upper

**Usage Example:**

.. code-block:: python

   spec = varmax_reg(non_seasonal_ar=2, non_seasonal_ma=1)
   fit = spec.fit(data, "sales + revenue ~ date + promo")

---

Hybrid Time Series Models
==========================

arima_boost()
-------------

**Purpose:** ARIMA + XGBoost hybrid for combining linear and non-linear patterns

**Function Signature:**

.. code-block:: python

   arima_boost(
       # ARIMA parameters
       non_seasonal_ar: int = 0,
       non_seasonal_differences: int = 0,
       non_seasonal_ma: int = 0,
       seasonal_ar: int = 0,
       seasonal_differences: int = 0,
       seasonal_ma: int = 0,
       seasonal_period: Optional[int] = None,
       # XGBoost parameters
       trees: int = 15,
       tree_depth: Optional[int] = None,
       learn_rate: float = 0.3,
       engine: str = "arima_xgboost"
   ) -> ModelSpec

**Parameters:**

Combines all ``arima_reg()`` parameters with ``boost_tree()`` parameters:

* ARIMA parameters: Same as ``arima_reg()``
* XGBoost parameters: Same as ``boost_tree()``

**How It Works:**

1. Fits ARIMA to capture linear time series patterns
2. Fits XGBoost to ARIMA residuals to capture non-linear patterns
3. Final prediction = ARIMA prediction + XGBoost prediction

**Usage Example:**

.. code-block:: python

   spec = arima_boost(
       # ARIMA component
       non_seasonal_ar=1,
       non_seasonal_differences=1,
       non_seasonal_ma=1,
       seasonal_period=12,
       # XGBoost component
       trees=100,
       tree_depth=4,
       learn_rate=0.1
   )

---

prophet_boost()
---------------

**Purpose:** Prophet + XGBoost hybrid for seasonality + non-linear patterns

**Function Signature:**

.. code-block:: python

   prophet_boost(
       # Prophet parameters
       growth: str = "linear",
       n_changepoints: int = 25,
       changepoint_prior_scale: float = 0.05,
       seasonality_prior_scale: float = 10.0,
       # XGBoost parameters
       trees: int = 15,
       tree_depth: Optional[int] = None,
       learn_rate: float = 0.3,
       engine: str = "prophet_xgboost"
   ) -> ModelSpec

**Parameters:**

Combines ``prophet_reg()`` and ``boost_tree()`` parameters.

**How It Works:**

1. Fits Prophet to capture trend and seasonality
2. Fits XGBoost to Prophet residuals
3. Final prediction = Prophet prediction + XGBoost prediction

**Usage Example:**

.. code-block:: python

   spec = prophet_boost(
       # Prophet component
       n_changepoints=50,
       changepoint_prior_scale=0.5,
       # XGBoost component
       trees=100,
       tree_depth=6,
       learn_rate=0.1
   )

---

Generic Hybrid Models
=====================

hybrid_model()
--------------

**Purpose:** Combine any two arbitrary models with four strategies

**Function Signature:**

.. code-block:: python

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

**Parameters:**

* ``model1`` (Optional[ModelSpec], default=None):

  - First model specification
  - Required parameter
  - Can be any model type
  - Example: ``model1=linear_reg()``

* ``model2`` (Optional[ModelSpec], default=None):

  - Second model specification
  - Required parameter
  - Can be any model type
  - Example: ``model2=rand_forest(trees=100, mode="regression")``

* ``strategy`` (str, default="residual"):

  - How to combine the two models
  - Options:

    1. **"residual"** (default):

       - Train model1 on y
       - Train model2 on residuals from model1
       - Prediction = model1_pred + model2_pred
       - Use case: Capture what model1 misses

    2. **"sequential"**:

       - Train model1 on early period (before split_point)
       - Train model2 on later period (after split_point)
       - Use model1 before split, model2 after
       - Use case: Handle regime changes, structural breaks
       - **Requires ``split_point`` parameter**

    3. **"weighted"**:

       - Train both models on same data
       - Prediction = weight1 × model1_pred + weight2 × model2_pred
       - Use case: Simple ensemble, reduce variance

    4. **"custom_data"** (NEW):

       - Train model1 on data['model1']
       - Train model2 on data['model2']
       - Datasets can have different/overlapping date ranges
       - Prediction blends both models based on ``blend_predictions``
       - Use case: Different training periods, adaptive learning
       - **Requires dict input to fit()**

* ``weight1`` (float, default=0.5):

  - Weight for model1 in weighted/custom_data strategies
  - Range: [0, 1]
  - Should sum to 1.0 with weight2 for interpretability
  - Example: ``weight1=0.6`` gives more weight to model1

* ``weight2`` (float, default=0.5):

  - Weight for model2
  - Same interpretation as weight1

* ``split_point`` (Optional[Union[int, float, str]], default=None):

  - For sequential strategy - where to split time periods
  - Options:

    * int: Row index (e.g., ``split_point=100``)
    * float: Proportion (e.g., ``split_point=0.7`` for 70/30 split)
    * str: Date string (e.g., ``split_point="2020-06-01"``)

  - Required for sequential strategy

* ``blend_predictions`` (str, default="weighted"):

  - For custom_data strategy - how to combine predictions
  - Options:

    * ``"weighted"``: weight1 × pred1 + weight2 × pred2 (default)
    * ``"avg"``: simple average (0.5 × pred1 + 0.5 × pred2)
    * ``"model1"``: use only model1 predictions
    * ``"model2"``: use only model2 predictions

* ``engine`` (str, default="generic_hybrid"):

  - Available engines:

    * ``"generic_hybrid"``: Generic hybrid engine

**Strategy Details:**

**1. Residual Strategy (Default):**

.. code-block:: python

   spec = hybrid_model(
       model1=linear_reg(),
       model2=rand_forest(trees=100, mode="regression"),
       strategy="residual"
   )
   fit = spec.fit(data, "y ~ x1 + x2")

Flow:
1. Fit linear_reg to predict y
2. Calculate residuals = y - linear_pred
3. Fit rand_forest to predict residuals
4. Final prediction = linear_pred + forest_pred

**2. Sequential Strategy:**

.. code-block:: python

   spec = hybrid_model(
       model1=linear_reg(),
       model2=rand_forest(trees=100, mode="regression"),
       strategy="sequential",
       split_point="2020-06-01"  # Regime change date
   )
   fit = spec.fit(data, "y ~ x1 + x2")

Flow:
1. Split data at 2020-06-01
2. Fit linear_reg on data before 2020-06-01
3. Fit rand_forest on data after 2020-06-01
4. Use appropriate model based on prediction date

**3. Weighted Strategy:**

.. code-block:: python

   spec = hybrid_model(
       model1=linear_reg(),
       model2=svm_rbf(cost=1.0).set_mode("regression"),
       strategy="weighted",
       weight1=0.6,
       weight2=0.4
   )
   fit = spec.fit(data, "y ~ x1 + x2")

Flow:
1. Fit both models on full data
2. Prediction = 0.6 × linear_pred + 0.4 × svm_pred

**4. Custom Data Strategy (NEW):**

.. code-block:: python

   spec = hybrid_model(
       model1=linear_reg(),
       model2=rand_forest(trees=100, mode="regression"),
       strategy="custom_data",
       blend_predictions="weighted",
       weight1=0.4,  # Less weight on older model
       weight2=0.6   # More weight on recent model
   )

   # Fit with separate datasets (can overlap)
   early_data = df[df['date'] < '2020-07-01']     # 100 obs
   later_data = df[df['date'] >= '2020-04-01']    # 80 obs, 3-month overlap

   fit = spec.fit({'model1': early_data, 'model2': later_data}, 'y ~ x1 + x2')

Flow:
1. Fit linear_reg on early_data (2020-01-01 to 2020-06-30)
2. Fit rand_forest on later_data (2020-04-01 to 2020-08-31)
3. For predictions: 0.4 × linear_pred + 0.6 × forest_pred

**Use Cases:**

* **Residual**: Combine complementary models (linear + non-linear)
* **Sequential**: Handle regime changes, structural breaks in data
* **Weighted**: Ensemble for variance reduction
* **Custom Data**: Adaptive learning, evolving data distributions

**Validation:**

.. code-block:: python

   # Error: missing models
   hybrid_model(model1=linear_reg(), model2=None)  # ValueError

   # Error: sequential without split_point
   hybrid_model(
       model1=linear_reg(),
       model2=linear_reg(),
       strategy="sequential"
   )  # ValueError: split_point required

   # Warning: weights don't sum to 1
   hybrid_model(
       model1=linear_reg(),
       model2=linear_reg(),
       strategy="weighted",
       weight1=0.7,
       weight2=0.5
   )  # Warning: weights sum to 1.2

**Output Structure:**

Three DataFrames from ``extract_outputs()``:

1. **Coefficients**: Includes strategy, model1_type, model2_type, weights (if applicable)
2. **Stats**: Includes n_obs_model1, n_obs_model2 (for custom_data)
3. **Outputs**: Standard actuals, fitted, residuals, forecast

---

Baseline Models
===============

null_model()
------------

**Purpose:** Simple baseline models (mean, median)

**Function Signature:**

.. code-block:: python

   null_model(
       method: str = "mean",
       engine: str = "baseline"
   ) -> ModelSpec

**Parameters:**

* ``method`` (str, default="mean"):

  - Prediction method
  - Options:

    * ``"mean"``: Use mean of training data
    * ``"median"``: Use median of training data

* ``engine`` (str, default="baseline"):

  - Available engines:

    * ``"baseline"``: Simple baseline implementation

**Usage Example:**

.. code-block:: python

   # Mean baseline
   spec = null_model(method="mean")

   # Median baseline
   spec = null_model(method="median")

---

naive_reg()
-----------

**Purpose:** Naive time series forecasting baselines

**Function Signature:**

.. code-block:: python

   naive_reg(
       method: str = "naive",
       seasonal_period: Optional[int] = None,
       engine: str = "baseline"
   ) -> ModelSpec

**Parameters:**

* ``method`` (str, default="naive"):

  - Forecasting method
  - Options:

    * ``"naive"``: Last observation carried forward
    * ``"seasonal_naive"``: Last seasonal observation (requires ``seasonal_period``)
    * ``"drift"``: Linear trend from first to last observation

* ``seasonal_period`` (Optional[int], default=None):

  - Required for ``method="seasonal_naive"``
  - Seasonal period length
  - Example: ``seasonal_period=12`` for monthly data with yearly seasonality

* ``engine`` (str, default="baseline"):

  - Available engines:

    * ``"baseline"``: Simple baseline implementation

**Usage Examples:**

.. code-block:: python

   # Naive: y_hat(t) = y(t-1)
   spec = naive_reg(method="naive")

   # Seasonal naive: y_hat(t) = y(t-m)
   spec = naive_reg(method="seasonal_naive", seasonal_period=12)

   # Drift: y_hat(t) = y(t-1) + (y(T) - y(1)) / (T-1)
   spec = naive_reg(method="drift")

---

Manual Coefficient Models
==========================

manual_reg()
------------

**Purpose:** User-specified coefficients (no model fitting)

**Function Signature:**

.. code-block:: python

   manual_reg(
       coefficients: Dict[str, float],
       intercept: float = 0.0,
       engine: str = "manual"
   ) -> ModelSpec

**Parameters:**

* ``coefficients`` (Dict[str, float]):

  - Dictionary mapping variable names to coefficient values
  - Keys must match predictor column names
  - Values are the coefficients to use
  - Example: ``coefficients={"x1": 0.5, "x2": -0.3}``

* ``intercept`` (float, default=0.0):

  - Intercept term
  - Added to all predictions
  - Example: ``intercept=10.0``

* ``engine`` (str, default="manual"):

  - Available engines:

    * ``"manual"``: Manual coefficient implementation

**Use Cases:**

1. Compare with external models (Excel, R, SAS)
2. Incorporate domain knowledge
3. Reproduce published models
4. Benchmark against established baselines

**Usage Example:**

.. code-block:: python

   # Manual coefficients from external source
   spec = manual_reg(
       coefficients={
           "price": -0.5,
           "advertising": 2.3,
           "competitor_price": 0.8
       },
       intercept=100.0
   )

   fit = spec.fit(data, "sales ~ price + advertising + competitor_price")
   predictions = fit.predict(new_data)

**Notes:**

* No actual fitting occurs - coefficients are fixed
* ``fit()`` validates coefficients match formula predictors
* Useful for comparing py-tidymodels predictions with external forecasts

---

Cross-Cutting Concepts
======================

Engine Selection
----------------

Most models support multiple engines. Set via ``.set_engine()`` method:

.. code-block:: python

   spec = linear_reg().set_engine("statsmodels")
   spec = boost_tree(trees=100, mode="regression").set_engine("lightgbm")

Mode Setting
------------

Some models require explicit mode setting:

.. code-block:: python

   # Method 1: Constructor parameter (rand_forest, boost_tree)
   spec = rand_forest(trees=100, mode="regression")

   # Method 2: set_mode() method (decision_tree, svm_rbf, etc.)
   spec = decision_tree(tree_depth=5).set_mode("regression")

Tunable Parameters
------------------

Mark parameters for tuning with ``tune()``:

.. code-block:: python

   from py_tune import tune

   spec = boost_tree(
       trees=tune(),
       tree_depth=tune(),
       learn_rate=tune()
   )

Prediction Types
----------------

All models support ``type`` parameter in ``predict()``:

.. code-block:: python

   # Numeric predictions (default)
   preds = fit.predict(new_data, type="numeric")

   # Confidence intervals (time series models)
   preds = fit.predict(new_data, type="conf_int")
   # Returns: .pred, .pred_lower, .pred_upper

   # Classification predictions
   preds = fit.predict(new_data, type="class")     # Class labels
   preds = fit.predict(new_data, type="prob")      # Probabilities

Output Structure
----------------

All models return three DataFrames from ``extract_outputs()``:

.. code-block:: python

   outputs, coefficients, stats = fit.extract_outputs()

1. **Outputs DataFrame**: Observation-level results

   - ``actuals``: True values
   - ``fitted``: Model predictions
   - ``forecast``: Combined actual/fitted series
   - ``residuals``: actuals - fitted
   - ``split``: "train", "test", or "forecast"
   - ``model``, ``model_group_name``, ``group``: Metadata

2. **Coefficients DataFrame**: Model parameters

   - ``variable``: Parameter name
   - ``coefficient``: Parameter value
   - ``std_error``, ``t_stat``, ``p_value``: Statistical inference (if available)
   - ``ci_0.025``, ``ci_0.975``: 95% confidence interval
   - ``vif``: Variance Inflation Factor
   - Tree models: Feature importances instead
   - Time series: Hyperparameters as "coefficients"

3. **Stats DataFrame**: Model-level metrics

   - ``metric``: Metric name (RMSE, MAE, R², etc.)
   - ``value``: Metric value
   - ``split``: "train", "test", ""
   - Additional: formula, model_type, n_obs_train, date ranges

Formula Syntax
--------------

Uses Patsy formula syntax (R-style):

.. code-block:: python

   # Simple formula
   "y ~ x1 + x2"

   # All predictors
   "y ~ ."

   # Interactions
   "y ~ x1 + x2 + x1:x2"
   "y ~ x1 * x2"  # Expands to x1 + x2 + x1:x2

   # Transformations with I()
   "y ~ x1 + I(x1**2)"           # Polynomial
   "y ~ x1 + x2 + I(x1*x2)"      # Manual interaction
   "y ~ I(x1 + x2)"              # Sum

   # Categorical encoding
   "y ~ C(category)"             # Dummy encoding

   # Multiple outcomes (VARMAX only)
   "y1 + y2 ~ date"              # Bivariate
   "y1 + y2 + y3 ~ date + x1"    # Trivariate with exog

---

Performance Considerations
==========================

Model Selection Guide
---------------------

**For Interpretability:**
1. ``linear_reg(engine="statsmodels")`` - Full statistical inference
2. ``decision_tree()`` - Visual decision paths
3. ``gen_additive_mod()`` - Smooth interpretable curves

**For Accuracy (Tabular Data):**
1. ``boost_tree(engine="xgboost")`` - Often best performance
2. ``boost_tree(engine="lightgbm")`` - Faster, large datasets
3. ``rand_forest()`` - Good default, robust

**For Time Series:**
1. ``prophet_reg()`` - Strong seasonality, holidays
2. ``arima_reg()`` - Linear temporal patterns
3. ``prophet_boost()``/``arima_boost()`` - Complex patterns
4. ``recursive_reg()`` - Multi-step ML forecasting

**For Categorical Features:**
1. ``boost_tree(engine="catboost")`` - Native categorical support
2. ``rand_forest()`` - Handles categoricals well

**For Large Datasets:**
1. ``boost_tree(engine="lightgbm")`` - Memory efficient
2. ``linear_reg()`` - Fast, scalable
3. ``svm_linear()`` - Linear scaling

Computational Complexity
------------------------

**Fast (< 1 second for 10k rows):**
- ``linear_reg()``
- ``null_model()``, ``naive_reg()``
- ``decision_tree()``

**Moderate (1-10 seconds for 10k rows):**
- ``rand_forest(trees=100)``
- ``boost_tree(trees=100)``
- ``svm_rbf()``

**Slow (10+ seconds for 10k rows):**
- ``prophet_reg()`` - Complex seasonal decomposition
- ``arima_reg()`` - MLE estimation
- ``nearest_neighbor()`` - Distance calculations
- ``boost_tree(trees=1000)`` - Many iterations

Memory Usage
------------

**Low Memory:**
- All linear models
- ``decision_tree()``
- Time series models

**High Memory:**
- ``rand_forest(trees=1000)`` - Stores all trees
- ``nearest_neighbor()`` - Stores all training data
- ``boost_tree(engine="xgboost")`` - Tree storage

Parallelization
---------------

**Multi-core Support:**
- ``rand_forest()`` - Parallel tree building
- ``boost_tree()`` - Some engines support parallel training
- Cross-validation in ``tune_grid()`` - Parallel fold evaluation

---

Common Workflows
================

Basic Regression
----------------

.. code-block:: python

   from py_parsnip import linear_reg
   from py_workflows import Workflow

   # Create and fit workflow
   spec = linear_reg()
   wf = Workflow().add_formula("y ~ x1 + x2").add_model(spec)
   fit = wf.fit(train_data)

   # Predict and evaluate
   preds = fit.predict(test_data)
   fit = fit.evaluate(test_data)

   # Extract outputs
   outputs, coefs, stats = fit.extract_outputs()

Time Series Forecasting
------------------------

.. code-block:: python

   from py_parsnip import prophet_reg

   # Create and fit model
   spec = prophet_reg(
       n_changepoints=50,
       changepoint_prior_scale=0.5,
       seasonality_mode="multiplicative"
   )
   fit = spec.fit(train_data, "y ~ date + regressor1")

   # Forecast future
   future_data = pd.DataFrame({
       'date': pd.date_range('2024-01-01', periods=30),
       'regressor1': future_values
   })
   forecast = fit.predict(future_data, type="conf_int")

Hyperparameter Tuning
----------------------

.. code-block:: python

   from py_parsnip import boost_tree
   from py_tune import tune, tune_grid, grid_regular
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, r_squared

   # Mark parameters for tuning
   spec = boost_tree(
       trees=tune(),
       tree_depth=tune(),
       learn_rate=tune(),
       mode="regression"
   )

   # Create workflow
   wf = Workflow().add_formula("y ~ .").add_model(spec)

   # Create CV folds
   folds = vfold_cv(train_data, v=5)

   # Create grid
   grid = grid_regular(
       {"trees": {"range": (50, 500), "levels": 5},
        "tree_depth": {"range": (3, 8), "levels": 3},
        "learn_rate": {"range": (0.01, 0.3), "trans": "log", "levels": 4}},
       levels=3
   )

   # Tune
   results = tune_grid(wf, folds, grid=grid, metrics=metric_set(rmse, r_squared))

   # Select best
   best = results.select_best("rmse", maximize=False)
   final_wf = finalize_workflow(wf, best)
   final_fit = final_wf.fit(train_data)

Multi-Model Comparison
-----------------------

.. code-block:: python

   from py_workflowsets import WorkflowSet
   from py_parsnip import linear_reg, rand_forest, boost_tree

   # Define models
   models = [
       linear_reg(),
       rand_forest(trees=300, mode="regression"),
       boost_tree(trees=100, mode="regression").set_engine("xgboost")
   ]

   # Define formulas
   formulas = [
       "y ~ x1 + x2",
       "y ~ x1 + x2 + I(x1**2)",
       "y ~ x1 * x2"  # With interaction
   ]

   # Create all combinations (3 × 3 = 9 workflows)
   wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

   # Evaluate all
   folds = vfold_cv(train_data, v=5)
   results = wf_set.fit_resamples(resamples=folds, metrics=metric_set(rmse, mae))

   # Rank and visualize
   top_models = results.rank_results("rmse", n=5)
   results.autoplot("rmse")
