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

Using ML Models for Time Series Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to Use ML Models vs Traditional Time Series Models:**

Machine learning models (Random Forest, XGBoost, k-NN, Neural Networks, etc.) are excellent for time series when:

* You have rich feature sets (exogenous variables)
* Relationships are non-linear
* You need to capture complex interactions
* Multiple seasonalities or irregular patterns exist
* You're doing supervised regression (not pure forecasting)

Traditional models (ARIMA, Prophet) are better when:

* You have univariate time series with limited features
* You need probabilistic forecasts with prediction intervals
* The data shows clear trend/seasonality patterns
* You need interpretable decomposition

**Date-Indexed Outputs:**

All models in py-tidymodels return date-indexed outputs from ``extract_outputs()`` when the data contains a date column. This enables proper visualization and time series analysis.

Random Forest Example
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_parsnip import rand_forest
   from py_rsample import initial_time_split, training, testing

   # Prepare time series data (must include 'date' column)
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
       'sales': [...],
       'price': [...],
       'promotion': [...],
       'temperature': [...]
   })

   # Time-based split (chronological order preserved)
   split = initial_time_split(
       data,
       date_column="date",
       train_end="2022-09-30",
       test_start="2022-10-01"
   )

   train = training(split)
   test = testing(split)

   # Random Forest for time series regression
   spec_rf = rand_forest(
       trees=500,
       mtry=3,      # Number of features to consider
       min_n=10     # Minimum samples per leaf
   ).set_mode('regression')

   # Fit model - date column automatically handled
   # Date is NOT used as predictor (excluded from formula)
   fit_rf = spec_rf.fit(train, "sales ~ price + promotion + temperature")

   # Evaluate on test set
   fit_rf = fit_rf.evaluate(test)

   print("Random Forest fitted successfully!")

   # Extract date-indexed outputs
   outputs_rf, coefs_rf, stats_rf = fit_rf.extract_outputs()

   # outputs_rf is indexed by date:
   # Columns: date, actuals, fitted, forecast, residuals, split
   print(outputs_rf.head())
   #          date   actuals    fitted   forecast  residuals  split
   # 0  2020-01-01     245.3     243.1      243.1       2.2   train
   # 1  2020-01-02     251.7     249.3      249.3       2.4   train
   # ...

   # Filter to test period only
   test_outputs = outputs_rf[outputs_rf['split'] == 'test']

   # Visualize forecast
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12, 6))
   plt.plot(test_outputs['date'], test_outputs['actuals'], label='Actual', marker='o')
   plt.plot(test_outputs['date'], test_outputs['fitted'], label='Predicted', marker='x')
   plt.xlabel('Date')
   plt.ylabel('Sales')
   plt.title('Random Forest Time Series Forecast')
   plt.legend()
   plt.show()

Gradient Boosting Example (XGBoost/LightGBM/CatBoost)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_parsnip import boost_tree

   # XGBoost for time series
   spec_xgb = boost_tree(
       trees=1000,
       tree_depth=6,
       learn_rate=0.01,
       min_n=5
   ).set_engine('xgboost')

   fit_xgb = spec_xgb.fit(train, "sales ~ price + promotion + temperature")
   fit_xgb = fit_xgb.evaluate(test)

   outputs_xgb, coefs_xgb, stats_xgb = fit_xgb.extract_outputs()

   # LightGBM for time series (faster training)
   spec_lgbm = boost_tree(
       trees=1000,
       tree_depth=6,
       learn_rate=0.01
   ).set_engine('lightgbm')

   fit_lgbm = spec_lgbm.fit(train, "sales ~ price + promotion + temperature")
   fit_lgbm = fit_lgbm.evaluate(test)

   outputs_lgbm, coefs_lgbm, stats_lgbm = fit_lgbm.extract_outputs()

   # CatBoost for time series (handles categoricals well)
   spec_cat = boost_tree(
       trees=1000,
       tree_depth=6,
       learn_rate=0.01
   ).set_engine('catboost')

   fit_cat = spec_cat.fit(train, "sales ~ price + promotion + temperature")
   fit_cat = fit_cat.evaluate(test)

   outputs_cat, coefs_cat, stats_cat = fit_cat.extract_outputs()

Support Vector Machines Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_parsnip import svm_rbf, svm_linear

   # SVM with RBF kernel
   spec_svm = svm_rbf(
       cost=1.0,
       rbf_sigma=0.1
   ).set_mode('regression')

   fit_svm = spec_svm.fit(train, "sales ~ price + promotion + temperature")
   fit_svm = fit_svm.evaluate(test)

   outputs_svm, coefs_svm, stats_svm = fit_svm.extract_outputs()

   # Outputs indexed by date
   print(f"SVM Test RMSE: {stats_svm[stats_svm['split']=='test']['rmse'].values[0]:.2f}")

k-Nearest Neighbors Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_parsnip import nearest_neighbor

   # k-NN with k=10 and distance weighting
   spec_knn = nearest_neighbor(
       neighbors=10,
       weight_func='distance'  # Weight by inverse distance
   ).set_mode('regression')

   fit_knn = spec_knn.fit(train, "sales ~ price + promotion + temperature")
   fit_knn = fit_knn.evaluate(test)

   outputs_knn, coefs_knn, stats_knn = fit_knn.extract_outputs()

Neural Network (MLP) Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_parsnip import mlp

   # Multi-layer perceptron
   spec_mlp = mlp(
       hidden_units=50,
       epochs=200,
       learn_rate=0.01
   ).set_mode('regression')

   fit_mlp = spec_mlp.fit(train, "sales ~ price + promotion + temperature")
   fit_mlp = fit_mlp.evaluate(test)

   outputs_mlp, coefs_mlp, stats_mlp = fit_mlp.extract_outputs()

Feature Engineering for ML Time Series Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ML models need engineered features to capture temporal patterns:

.. code-block:: python

   from py_recipes import recipe
   from py_workflows import workflow
   from py_parsnip import boost_tree

   # Comprehensive feature engineering for time series
   rec = (
       recipe()
       # 1. Lagged features (autoregressive)
       .step_lag(["sales"], lags=[1, 7, 14, 28])

       # 2. Rolling statistics
       .step_mutate(
           sales_ma7="sales.rolling(7).mean()",
           sales_ma28="sales.rolling(28).mean()",
           sales_std7="sales.rolling(7).std()"
       )

       # 3. Time-based features
       .step_timeseries_signature(["date"])  # year, month, day, dow, etc.

       # 4. Differencing (if needed for stationarity)
       .step_diff(["price"], lag=1)

       # 5. Encoding categorical features
       .step_dummy(["date_month", "date_dow", "promotion_type"])

       # 6. Normalization
       .step_normalize()

       # 7. Handle missing values from lagging
       .step_impute_median()

       # 8. Remove correlated features
       .step_corr(threshold=0.95)
   )

   # Use in workflow
   wf = (
       workflow()
       .add_recipe(rec)
       .add_model(boost_tree(trees=500).set_engine('xgboost'))
   )

   fit = wf.fit(train)
   fit = fit.evaluate(test)

   outputs, coefs, stats = fit.extract_outputs()

   # outputs is STILL indexed by date even after all preprocessing!
   print(outputs[outputs['split']=='test'][['date', 'actuals', 'fitted']].head(10))

Understanding the Three-DataFrame Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All models return three DataFrames from ``extract_outputs()``:

**1. outputs** - Observation-level results (date-indexed):

.. code-block:: python

   # Columns in outputs DataFrame:
   # - date: Original date from input data
   # - actuals: True values
   # - fitted: Model predictions
   # - forecast: Combined series (actuals where available, fitted elsewhere)
   # - residuals: actuals - fitted
   # - split: 'train', 'test', or 'forecast'
   # - model: Model name
   # - model_group_name: Model group identifier
   # - group: Group identifier (for panel models)

   outputs_rf.head()

**2. coefficients** - Model parameters/importances:

.. code-block:: python

   # For tree models: feature importances
   # Columns: variable, importance, model, model_group_name, group

   coefs_rf.head()
   #     variable  importance     model  model_group_name  group
   # 0      price      0.453  rand_forest      default      all
   # 1  promotion      0.321  rand_forest      default      all
   # 2temperature      0.226  rand_forest      default      all

**3. stats** - Model-level metrics by split:

.. code-block:: python

   # Columns: metric, value, split, model, model_group_name, group

   stats_rf[stats_rf['split']=='test']
   #    metric   value  split       model  model_group_name  group
   # 0    rmse   12.34   test  rand_forest      default      all
   # 1     mae    9.87   test  rand_forest      default      all
   # 2 r_squared  0.892   test  rand_forest      default      all

Comparing Multiple ML Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from py_workflowsets import WorkflowSet
   from py_yardstick import metric_set, rmse, mae, r_squared

   # Define multiple models
   models = [
       ("rf", rand_forest(trees=500).set_mode('regression')),
       ("xgb", boost_tree(trees=500).set_engine('xgboost')),
       ("svm", svm_rbf().set_mode('regression')),
       ("knn", nearest_neighbor(neighbors=10).set_mode('regression'))
   ]

   # Create workflow set
   wf_set = WorkflowSet.from_workflows([
       (name, workflow().add_formula("sales ~ price + promotion + temperature").add_model(model))
       for name, model in models
   ])

   # Evaluate all models
   from py_rsample import vfold_cv
   folds = vfold_cv(train, v=5)

   results = wf_set.fit_resamples(
       resamples=folds,
       metrics=metric_set(rmse, mae, r_squared)
   )

   # Rank models
   rankings = results.rank_results("rmse")
   print(rankings)

   # Select best model and fit on full training set
   best_wf_id = rankings.iloc[0]["wflow_id"]
   best_wf = wf_set[best_wf_id]

   best_fit = best_wf.fit(train)
   best_fit = best_fit.evaluate(test)

   # Extract date-indexed outputs from best model
   outputs, coefs, stats = best_fit.extract_outputs()

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
