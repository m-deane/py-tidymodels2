Time Series Modeling
====================

This guide covers time series forecasting with py-tidymodels, including specialized models, data preparation, and best practices.

Time Series Models
------------------

py-tidymodels provides 5 native time series models plus 2 hybrid models:

Native Time Series Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ARIMA Models** (arima_reg):

.. code-block:: python

   from py_parsnip import arima_reg

   # Manual ARIMA specification
   spec = arima_reg(
       non_seasonal_ar=1,          # p: AR order
       non_seasonal_differences=1,  # d: differencing
       non_seasonal_ma=1,           # q: MA order
       seasonal_period=7            # Weekly seasonality
   )

   # Auto ARIMA (automatic order selection)
   spec = arima_reg(
       non_seasonal_ar=2,    # max_p=2
       seasonal_period=12
   ).set_engine("auto_arima")

**Prophet Models** (prophet_reg):

.. code-block:: python

   from py_parsnip import prophet_reg

   spec = prophet_reg(
       n_changepoints=25,
       changepoint_prior_scale=0.05,
       seasonality_prior_scale=10.0,
       seasonality_mode="additive"  # or "multiplicative"
   )

**Exponential Smoothing** (exp_smoothing):

.. code-block:: python

   from py_parsnip import exp_smoothing

   # Automatic selection
   spec = exp_smoothing(error="add", trend="add", season="add")

   # Or specific model
   spec = exp_smoothing(error="mul", trend="add", season="mul", seasonal_period=12)

**STL Decomposition** (seasonal_reg):

.. code-block:: python

   from py_parsnip import seasonal_reg

   spec = seasonal_reg(seasonal_period=7)

**Recursive Forecasting** (recursive_reg):

.. code-block:: python

   from py_parsnip import recursive_reg, rand_forest

   # Use any ML model for multi-step forecasting
   spec = recursive_reg(
       base_model=rand_forest(trees=100),
       lags=7,          # Use lags 1-7
       # Or: lags=[1, 7, 14, 28]  # Specific lags
   )

Hybrid Models
~~~~~~~~~~~~~

Combine time series with gradient boosting:

.. code-block:: python

   from py_parsnip import arima_boost, prophet_boost

   # ARIMA + XGBoost
   spec = arima_boost(
       non_seasonal_ar=1,
       trees=100,
       tree_depth=6
   )

   # Prophet + XGBoost
   spec = prophet_boost(
       n_changepoints=25,
       trees=100
   )

Data Preparation
----------------

Date Column Handling
~~~~~~~~~~~~~~~~~~~~

Time series models automatically detect date columns:

.. code-block:: python

   # Date column auto-detected
   fit = spec.fit(data, "sales ~ date + feature1 + feature2")

   # Or specify explicitly
   spec = spec.set_args(date_col="date")
   fit = spec.fit(data, "sales ~ .")

Date Requirements:

* Must be pandas datetime64 dtype
* Should have regular frequency (daily, weekly, monthly)
* Missing dates can cause issues (fill gaps first)

Time Series Splits
~~~~~~~~~~~~~~~~~~

Use time-aware splitting:

.. code-block:: python

   from py_rsample import initial_time_split, time_series_cv

   # Simple train/test split
   split = initial_time_split(
       data,
       date_column="date",
       train_start="2020-01-01",
       train_end="2023-12-31",
       test_start="2024-01-01",
       test_end="2024-06-30"
   )

   train = split.training()
   test = split.testing()

   # Time series cross-validation
   folds = time_series_cv(
       data,
       date_column="date",
       initial="6 months",    # Initial training size
       assess="1 month",      # Test size per fold
       skip="0 months",       # Gap between folds
       cumulative=False       # Rolling window (False) or expanding (True)
   )

Feature Engineering for Time Series
------------------------------------

Lagged Features
~~~~~~~~~~~~~~~

Create lagged variables for ML models:

.. code-block:: python

   from py_recipes import recipe

   rec = (
       recipe()
       .step_lag(["sales"], lags=[1, 7, 14, 28])  # Create sales_lag1, sales_lag7, etc.
       .step_impute_median()  # Handle NAs from lagging
   )

Rolling Statistics
~~~~~~~~~~~~~~~~~~

Create moving averages and other rolling features:

.. code-block:: python

   rec = (
       recipe()
       .step_mutate(
           sales_ma7="sales.rolling(7).mean()",
           sales_ma28="sales.rolling(28).mean()",
           sales_std7="sales.rolling(7).std()"
       )
       .step_impute_median()
   )

Time Features
~~~~~~~~~~~~~

Extract temporal features from dates:

.. code-block:: python

   rec = (
       recipe()
       .step_timeseries_signature(["date"])  # Creates: year, month, day, dow, etc.
       .step_dummy(["date_month", "date_dow"])  # Encode month, day of week
   )

Differencing
~~~~~~~~~~~~

Make series stationary:

.. code-block:: python

   rec = (
       recipe()
       .step_diff(["sales"], lag=1)      # First difference
       .step_diff(["sales"], lag=7)      # Seasonal difference
       .step_impute_median()
   )

Complete Examples
-----------------

Prophet Forecasting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import prophet_reg
   from py_rsample import initial_time_split, training, testing

   # Prepare data
   split = initial_time_split(
       data,
       date_column="date",
       train_start="2020-01-01",
       train_end="2023-12-31",
       test_start="2024-01-01",
       test_end="2024-06-30"
   )

   train = training(split)
   test = testing(split)

   # Fit Prophet model
   spec = prophet_reg(
       n_changepoints=25,
       changepoint_prior_scale=0.05,
       seasonality_prior_scale=10.0
   )

   fit = spec.fit(train, "sales ~ date + promotion + price")
   fit = fit.evaluate(test)

   # Extract outputs with dates
   outputs, coefs, stats = fit.extract_outputs()

   # Visualize
   from py_visualize import plot_forecast
   plot_forecast(outputs, title="Sales Forecast")

ARIMA with Auto Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import arima_reg

   # Auto ARIMA finds best (p,d,q)
   spec = arima_reg(
       non_seasonal_ar=5,         # Search p in [0, 5]
       non_seasonal_differences=2, # Search d in [0, 2]
       non_seasonal_ma=5,          # Search q in [0, 5]
       seasonal_ar=2,              # Search P in [0, 2]
       seasonal_ma=2,              # Search Q in [0, 2]
       seasonal_period=12          # Monthly data
   ).set_engine("auto_arima")

   fit = spec.fit(train, "sales ~ date")

   # Check selected order
   order = fit.fit_data["order"]
   print(f"Selected ARIMA order: {order}")

Recursive ML Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import recursive_reg, boost_tree
   from py_recipes import recipe

   # Feature engineering
   rec = (
       recipe()
       .step_lag(["sales"], lags=[1, 7, 14, 28])
       .step_timeseries_signature(["date"])
       .step_normalize()
       .step_impute_median()
   )

   rec_prepped = rec.prep(train)
   train_proc = rec_prepped.bake(train)
   test_proc = rec_prepped.bake(test)

   # Recursive forecasting with XGBoost
   spec = recursive_reg(
       base_model=boost_tree(trees=100, tree_depth=6),
       lags=[1, 7, 14, 28]
   )

   fit = spec.fit(train_proc, "sales ~ .")
   fit = fit.evaluate(test_proc)

   outputs, _, stats = fit.extract_outputs()

Panel/Grouped Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import prophet_reg

   # Data with multiple stores
   # Columns: date, store_id, sales, price

   wf = (
       workflow()
       .add_formula("sales ~ date + price")
       .add_model(prophet_reg())
   )

   # Nested: separate Prophet model per store
   nested_fit = wf.fit_nested(train, group_col="store_id")

   # Predict for all stores
   predictions = nested_fit.predict(test)

   # Extract outputs (includes store_id column)
   outputs, coefs, stats = nested_fit.extract_outputs()

Hyperparameter Tuning for Time Series
--------------------------------------

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_tune import tune, tune_grid, grid_regular, finalize_workflow
   from py_workflows import workflow
   from py_parsnip import prophet_reg
   from py_rsample import time_series_cv
   from py_yardstick import metric_set, rmse, mae

   # Mark parameters for tuning
   spec = prophet_reg(
       changepoint_prior_scale=tune(),
       seasonality_prior_scale=tune()
   )

   wf = workflow().add_formula("sales ~ date").add_model(spec)

   # Time series CV
   folds = time_series_cv(
       train,
       date_column="date",
       initial="6 months",
       assess="1 month",
       cumulative=True  # Expanding window
   )

   # Parameter grid
   grid = grid_regular({
       "changepoint_prior_scale": {"range": (0.001, 0.5), "trans": "log"},
       "seasonality_prior_scale": {"range": (0.01, 10), "trans": "log"}
   }, levels=5)

   # Tune
   results = tune_grid(
       wf,
       resamples=folds,
       grid=grid,
       metrics=metric_set(rmse, mae)
   )

   # Finalize
   best = results.select_best("rmse", maximize=False)
   final_wf = finalize_workflow(wf, best)
   final_fit = final_wf.fit(train)

Best Practices
--------------

Check Stationarity
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statsmodels.tsa.stattools import adfuller

   # ADF test
   result = adfuller(data['sales'])
   print(f"ADF Statistic: {result[0]}")
   print(f"p-value: {result[1]}")

   if result[1] > 0.05:
       print("Series is non-stationary, consider differencing")

Handle Seasonality
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For ARIMA: specify seasonal period
   spec = arima_reg(seasonal_period=7)  # Weekly

   # For Prophet: add custom seasonality
   spec = prophet_reg()
   # Prophet auto-detects daily, weekly, yearly

   # For ML: create seasonal features
   rec = recipe().step_timeseries_signature(["date"])

Validate on Holdout
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Always use chronological split
   split = initial_time_split(
       data,
       date_column="date",
       train_end="2023-12-31",
       test_start="2024-01-01"
   )

   # Never shuffle time series data!

Check Residuals
~~~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_residuals

   # Fit model
   fit = spec.fit(train, "sales ~ date")
   fit = fit.evaluate(test)

   outputs, _, stats = fit.extract_outputs()

   # Plot diagnostics
   plot_residuals(outputs, plot_type="all")

   # Check for:
   # - No pattern in residuals (random)
   # - Normally distributed
   # - No autocorrelation

Use Appropriate Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_yardstick import metric_set, rmse, mae, mape, smape

   # For time series:
   metrics = metric_set(
       rmse,    # Root Mean Squared Error
       mae,     # Mean Absolute Error
       mape,    # Mean Absolute Percentage Error
       smape    # Symmetric MAPE (handles zeros better)
   )

Common Issues
-------------

Missing Dates
~~~~~~~~~~~~~

.. code-block:: python

   # Fill missing dates
   date_range = pd.date_range(
       start=data['date'].min(),
       end=data['date'].max(),
       freq='D'
   )

   data = data.set_index('date').reindex(date_range).reset_index()
   data = data.rename(columns={'index': 'date'})

   # Forward fill values
   data['sales'] = data['sales'].fillna(method='ffill')

Irregular Frequency
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Resample to regular frequency
   data = data.set_index('date').resample('D').mean().reset_index()

Multiple Seasonality
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use STL or Prophet
   spec = seasonal_reg(seasonal_period=[7, 365])  # Weekly + yearly

   # Or Prophet (auto-detects)
   spec = prophet_reg()

Next Steps
----------

* :doc:`../api/parsnip` - All time series models
* :doc:`../models/time_series` - Model reference
* :doc:`recipes` - Feature engineering for time series
* :doc:`tuning` - Hyperparameter optimization
* :doc:`../examples/time_series_forecasting` - More examples
