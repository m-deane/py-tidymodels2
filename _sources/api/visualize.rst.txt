py-visualize: Interactive Visualizations
==========================================

The ``py_visualize`` package provides interactive Plotly visualizations for model diagnostics and results.

Main Functions
--------------

Forecast Visualization
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: py_visualize.plot_forecast

Residual Diagnostics
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: py_visualize.plot_residuals

Model Comparison
~~~~~~~~~~~~~~~~

.. autofunction:: py_visualize.plot_model_comparison

Decomposition Plots
~~~~~~~~~~~~~~~~~~~

.. autofunction:: py_visualize.plot_decomposition

Examples
--------

Forecast Plot
~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_forecast
   from py_parsnip import prophet_reg

   # Fit model
   spec = prophet_reg()
   fit = spec.fit(train_data, "sales ~ date + feature1")
   fit = fit.evaluate(test_data)

   # Extract outputs
   outputs, coefs, stats = fit.extract_outputs()

   # Create interactive forecast plot
   fig = plot_forecast(outputs, title="Sales Forecast")
   fig.show()

Residual Diagnostics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_residuals

   # Four diagnostic plots
   fig = plot_residuals(
       outputs,
       plot_type="all",  # "scatter", "histogram", "qq", "acf", "all"
       title="Residual Diagnostics"
   )
   fig.show()

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_model_comparison
   import pandas as pd

   # Combine stats from multiple models
   all_stats = pd.concat([
       stats_prophet.assign(model="Prophet"),
       stats_arima.assign(model="ARIMA"),
       stats_linear.assign(model="Linear")
   ])

   # Compare test metrics
   fig = plot_model_comparison(
       all_stats[all_stats['split'] == 'test'],
       metric="rmse",
       title="Model Comparison: Test RMSE"
   )
   fig.show()

Decomposition Plot
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_decomposition
   from py_parsnip import seasonal_reg

   # Fit STL model
   spec = seasonal_reg(seasonal_period=7)
   fit = spec.fit(train_data, "sales ~ date")

   # Extract components
   components = fit.fit_data["components"]

   # Plot decomposition
   fig = plot_decomposition(
       components,
       title="STL Decomposition"
   )
   fig.show()

Customization
~~~~~~~~~~~~~

.. code-block:: python

   from py_visualize import plot_forecast
   import plotly.graph_objects as go

   # Create base plot
   fig = plot_forecast(outputs)

   # Customize layout
   fig.update_layout(
       template="plotly_dark",
       width=1200,
       height=600,
       font=dict(size=14)
   )

   # Add custom annotations
   fig.add_annotation(
       text="Forecast period",
       x="2024-01-01",
       y=100,
       showarrow=True
   )

   fig.show()
