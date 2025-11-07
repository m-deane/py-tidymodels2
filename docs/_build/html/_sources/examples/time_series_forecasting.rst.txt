Time Series Forecasting Examples
=================================

Complete examples for time series forecasting.

Prophet Forecasting
-------------------

.. code-block:: python

   from py_parsnip import prophet_reg
   from py_rsample import initial_time_split, training, testing

   # Split
   split = initial_time_split(
       data,
       date_column="date",
       train_end="2023-12-31",
       test_start="2024-01-01"
   )

   train = training(split)
   test = testing(split)

   # Fit
   spec = prophet_reg(n_changepoints=25)
   fit = spec.fit(train, "sales ~ date + promotion")
   fit = fit.evaluate(test)

   # Extract and visualize
   outputs, _, stats = fit.extract_outputs()

   from py_visualize import plot_forecast
   plot_forecast(outputs, title="Sales Forecast")

ARIMA Forecasting
-----------------

.. code-block:: python

   from py_parsnip import arima_reg

   # Auto ARIMA
   spec = arima_reg(non_seasonal_ar=5).set_engine("auto_arima")
   fit = spec.fit(train, "sales ~ date")

   print(f"Selected order: {fit.fit_data['order']}")

See Also
--------

* :doc:`../user_guide/time_series` - Time series guide
* :doc:`../models/time_series` - All time series models
