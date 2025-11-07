Time Series Models
==================

Specialized models for forecasting and time series analysis.

arima_reg()
-----------

**Purpose**: ARIMA/SARIMAX models

**Engines**:
- statsmodels (manual specification)
- auto_arima (automatic order selection)

**Example**:

.. code-block:: python

   from py_parsnip import arima_reg

   # Manual ARIMA
   spec = arima_reg(
       non_seasonal_ar=1,
       non_seasonal_differences=1,
       non_seasonal_ma=1,
       seasonal_period=7
   )

   # Auto ARIMA
   spec = arima_reg(non_seasonal_ar=5).set_engine("auto_arima")

   fit = spec.fit(train, "sales ~ date")

prophet_reg()
-------------

**Purpose**: Facebook Prophet for additive time series

**Best For**: Multiple seasonality, holidays, trend changes

**Example**:

.. code-block:: python

   from py_parsnip import prophet_reg

   spec = prophet_reg(
       n_changepoints=25,
       changepoint_prior_scale=0.05,
       seasonality_prior_scale=10.0
   )

   fit = spec.fit(train, "sales ~ date + promotion")

exp_smoothing()
---------------

**Purpose**: Exponential smoothing / ETS models

**Example**:

.. code-block:: python

   from py_parsnip import exp_smoothing

   # Automatic selection
   spec = exp_smoothing()

   # Specific model
   spec = exp_smoothing(error="add", trend="add", season="add")

   fit = spec.fit(train, "sales ~ date")

seasonal_reg()
--------------

**Purpose**: STL decomposition

**Example**:

.. code-block:: python

   from py_parsnip import seasonal_reg

   spec = seasonal_reg(seasonal_period=7)
   fit = spec.fit(train, "sales ~ date")

recursive_reg()
---------------

**Purpose**: Multi-step ML forecasting

**Example**:

.. code-block:: python

   from py_parsnip import recursive_reg, rand_forest

   spec = recursive_reg(
       base_model=rand_forest(trees=100),
       lags=[1, 7, 14, 28]
   )

   fit = spec.fit(train, "sales ~ .")

Hybrid Models
-------------

arima_boost()
~~~~~~~~~~~~~

ARIMA + XGBoost:

.. code-block:: python

   from py_parsnip import arima_boost

   spec = arima_boost(non_seasonal_ar=1, trees=100)

prophet_boost()
~~~~~~~~~~~~~~~

Prophet + XGBoost:

.. code-block:: python

   from py_parsnip import prophet_boost

   spec = prophet_boost(n_changepoints=25, trees=100)

When to Use
-----------

**ARIMA**: Statistical approach, linear patterns, few covariates
**Prophet**: Multiple seasonality, holidays, interpretable decomposition
**ETS**: Automatic model selection, simple exponential smoothing
**STL**: Decompose trend/seasonal/residual
**Recursive**: ML forecasting, many covariates, complex patterns
**Hybrid**: Combine strengths of statistical + ML

See Also
--------

* :doc:`../user_guide/time_series` - Detailed time series guide
* :doc:`../api/parsnip` - Complete API reference
