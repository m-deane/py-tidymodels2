Visualization Guide
===================

The **py_visualize** module provides interactive Plotly-based visualizations for model analysis, diagnostics, and comparison. All plotting functions return Plotly figures that can be displayed in Jupyter notebooks or saved as HTML files.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

py-tidymodels includes five main visualization functions:

- :func:`plot_forecast` - Time series forecasting plots with actuals, fitted, and predictions
- :func:`plot_forecast_multi` - Multi-model forecast comparison
- :func:`plot_residuals` - Diagnostic plots for model validation
- :func:`plot_model_comparison` - Compare multiple models by metrics
- :func:`plot_tune_results` - Visualize hyperparameter tuning results

All plots are **interactive** with Plotly's built-in features:

- Hover tooltips showing exact values
- Zoom and pan capabilities
- Toggle traces on/off by clicking legend items
- Export to PNG via camera icon
- Auto-scaling and responsive layouts

Installation
------------

The visualization module requires Plotly:

.. code-block:: bash

   pip install plotly>=5.0.0

Basic Usage Pattern
-------------------

All visualization functions follow a consistent pattern:

.. code-block:: python

   from py_visualize import plot_forecast, plot_residuals, plot_model_comparison

   # 1. Fit your model
   fit = workflow().add_model(...).fit(train).evaluate(test)

   # 2. Create plot
   fig = plot_forecast(fit)

   # 3. Display (Jupyter) or save
   fig.show()  # Display in Jupyter
   fig.write_html("plot.html")  # Save as HTML file

----

Forecast Visualization
----------------------

plot_forecast()
~~~~~~~~~~~~~~~

Create interactive time series plots showing actuals, fitted values, and forecasts with optional prediction intervals.

**Function Signature:**

.. code-block:: python

   plot_forecast(
       fit,                        # WorkflowFit or NestedWorkflowFit
       prediction_intervals=True,  # Show prediction intervals
       title=None,                 # Custom title
       height=500,                 # Plot height in pixels
       width=None,                 # Plot width (None = auto)
       show_legend=True            # Display legend
   )

**Basic Example:**

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import prophet_reg
   from py_visualize import plot_forecast
   import pandas as pd

   # Prepare time series data
   dates = pd.date_range('2020-01-01', periods=365, freq='D')
   df = pd.DataFrame({
       'date': dates,
       'sales': [...],  # Your sales data
       'temperature': [...]
   })

   # Split data
   train = df.iloc[:300]
   test = df.iloc[300:]

   # Fit Prophet model
   wf = workflow().add_formula("sales ~ date + temperature").add_model(prophet_reg())
   fit = wf.fit(train).evaluate(test)

   # Create forecast plot
   fig = plot_forecast(fit, prediction_intervals=True, title="Sales Forecast")
   fig.show()

**What the Plot Shows:**

- **Blue line**: Training data (actual values)
- **Orange line**: Test data (actual values)
- **Green line**: Fitted values (in-sample predictions on training data)
- **Red line**: Forecasts (out-of-sample predictions on test data)
- **Shaded regions**: Prediction intervals (if available and requested)

**With Linear Regression:**

.. code-block:: python

   from py_parsnip import linear_reg

   # Linear regression for time series
   wf = workflow().add_formula("sales ~ temperature + lag_7_sales").add_model(linear_reg())
   fit = wf.fit(train).evaluate(test)

   # Simple forecast plot without prediction intervals
   fig = plot_forecast(fit, prediction_intervals=False, height=600)
   fig.show()

**Panel/Grouped Models:**

For nested models (per-group forecasts), ``plot_forecast()`` automatically creates subplots for each group:

.. code-block:: python

   from py_parsnip import arima_reg

   # Fit nested model (one ARIMA per store)
   wf = workflow().add_formula("sales ~ date").add_model(arima_reg())
   nested_fit = wf.fit_nested(panel_data, group_col="store_id")

   # Creates one subplot per store
   fig = plot_forecast(nested_fit, height=800)
   fig.show()

**Customization:**

.. code-block:: python

   # Custom styling
   fig = plot_forecast(
       fit,
       prediction_intervals=True,
       title="Q4 Sales Forecast - Prophet Model",
       height=700,
       width=1200,
       show_legend=True
   )

   # Further customize with Plotly API
   fig.update_layout(
       font=dict(size=14),
       hovermode='x unified',
       template='plotly_white'
   )

   # Save to file
   fig.write_html("sales_forecast.html")

plot_forecast_multi()
~~~~~~~~~~~~~~~~~~~~~

Compare forecasts from multiple models on the same plot.

.. code-block:: python

   from py_visualize import plot_forecast_multi

   # Fit multiple models
   fit_prophet = wf_prophet.fit(train).evaluate(test)
   fit_arima = wf_arima.fit(train).evaluate(test)
   fit_xgb = wf_xgb.fit(train).evaluate(test)

   # Compare all three
   fig = plot_forecast_multi(
       fits=[fit_prophet, fit_arima, fit_xgb],
       model_names=["Prophet", "ARIMA", "XGBoost"],
       title="Model Comparison: Sales Forecasting"
   )
   fig.show()

This creates overlaid forecast lines with different colors for each model, making it easy to visually compare model predictions.

----

Residual Diagnostics
--------------------

plot_residuals()
~~~~~~~~~~~~~~~~

Create diagnostic plots to validate model assumptions and identify potential issues.

**Function Signature:**

.. code-block:: python

   plot_residuals(
       fit,              # WorkflowFit or NestedWorkflowFit
       plot_type="all",  # Type of diagnostic plot
       title=None,       # Custom title
       height=600,       # Plot height in pixels
       width=None        # Plot width (None = auto)
   )

**Plot Types:**

- ``"all"`` (default): 2×2 grid with all four diagnostic plots
- ``"fitted"``: Residuals vs fitted values (homoscedasticity check)
- ``"qq"``: Q-Q plot (normality check)
- ``"time"``: Residuals vs time (autocorrelation check)
- ``"hist"``: Histogram of residuals (distribution check)

**Complete Diagnostics (All Four Plots):**

.. code-block:: python

   from py_visualize import plot_residuals

   # Fit model
   wf = workflow().add_formula("sales ~ .").add_model(linear_reg())
   fit = wf.fit(train).evaluate(test)

   # Create all diagnostic plots
   fig = plot_residuals(fit, plot_type="all", height=800)
   fig.show()

**What Each Plot Shows:**

1. **Residuals vs Fitted** (Top Left):

   - Check for **homoscedasticity** (constant variance)
   - Residuals should be randomly scattered around zero
   - Funnel shape indicates heteroscedasticity
   - Patterns indicate non-linearity

2. **Q-Q Plot** (Top Right):

   - Check if residuals are **normally distributed**
   - Points should follow the diagonal line
   - Deviations at extremes indicate heavy tails
   - S-curve indicates skewness

3. **Residuals vs Time** (Bottom Left):

   - Check for **temporal patterns** in residuals
   - Residuals should be randomly scattered
   - Trends indicate missing time-dependent features
   - Cycles indicate seasonality not captured

4. **Histogram** (Bottom Right):

   - Visual check of **residual distribution**
   - Should approximate normal distribution
   - Skewness or multiple peaks indicate issues

**Individual Diagnostic Plots:**

.. code-block:: python

   # Just Q-Q plot for normality check
   fig_qq = plot_residuals(fit, plot_type="qq", title="Normality Check")
   fig_qq.show()

   # Just residuals vs fitted for variance check
   fig_fitted = plot_residuals(fit, plot_type="fitted", height=500)
   fig_fitted.show()

   # Time series autocorrelation check
   fig_time = plot_residuals(fit, plot_type="time")
   fig_time.show()

**Interpreting Results:**

.. code-block:: python

   # Good model - residuals look random
   fit_good = linear_reg().fit(data, "y ~ x1 + x2")
   plot_residuals(fit_good)  # ✓ Random scatter, normal Q-Q

   # Problematic model - patterns in residuals
   fit_bad = linear_reg().fit(data, "y ~ x1")  # Missing x2
   plot_residuals(fit_bad)  # ✗ Funnel shape, curved Q-Q

**Common Issues Detected:**

- **Funnel shape in fitted plot**: Heteroscedasticity → Try log transformation
- **Curved line in Q-Q plot**: Non-normality → Check for outliers
- **Pattern in time plot**: Autocorrelation → Add lag features or use ARIMA
- **Multiple peaks in histogram**: Mixture of distributions → Check for subgroups

----

Model Comparison
----------------

plot_model_comparison()
~~~~~~~~~~~~~~~~~~~~~~~

Compare performance of multiple models across different metrics.

**Function Signature:**

.. code-block:: python

   plot_model_comparison(
       stats_list,          # List of stats DataFrames
       model_names=None,    # Model names for legend
       metrics=None,        # Metrics to compare
       split="test",        # Which split to compare
       plot_type="bar",     # Visualization type
       title=None,          # Custom title
       height=500,          # Plot height
       width=None,          # Plot width
       show_legend=True     # Show legend
   )

**Basic Comparison (Bar Chart):**

.. code-block:: python

   from py_visualize import plot_model_comparison
   from py_parsnip import linear_reg, rand_forest, boost_tree

   # Fit multiple models
   models = {
       "Linear Regression": linear_reg(),
       "Random Forest": rand_forest(trees=100).set_mode("regression"),
       "XGBoost": boost_tree(trees=100).set_mode("regression")
   }

   # Train and collect stats
   stats_list = []
   model_names = []

   for name, model in models.items():
       wf = workflow().add_formula("sales ~ .").add_model(model)
       fit = wf.fit(train).evaluate(test)
       _, _, stats = fit.extract_outputs()
       stats_list.append(stats)
       model_names.append(name)

   # Compare models
   fig = plot_model_comparison(
       stats_list,
       model_names=model_names,
       metrics=["rmse", "mae", "r_squared"],
       split="test",
       plot_type="bar",
       title="Model Performance Comparison"
   )
   fig.show()

**What the Plot Shows:**

- Grouped bar chart with one group per metric
- Different colored bars for each model
- Easy visual comparison of which model performs best
- Lower is better for RMSE/MAE, higher is better for R²

**Heatmap Comparison (Many Models):**

For comparing many models across many metrics, use heatmap:

.. code-block:: python

   # Compare 10 models across 5 metrics
   fig = plot_model_comparison(
       stats_list,
       model_names=model_names,
       metrics=["rmse", "mae", "mape", "r_squared", "adj_r_squared"],
       plot_type="heatmap",
       height=600
   )
   fig.show()

**Radar Chart (Normalized Metrics):**

Radar charts work well when you want to see the overall "shape" of model performance:

.. code-block:: python

   # All metrics normalized to 0-1 scale
   fig = plot_model_comparison(
       stats_list,
       model_names=["Model A", "Model B", "Model C"],
       metrics=["rmse", "mae", "mape", "r_squared"],
       plot_type="radar",
       title="Normalized Model Performance"
   )
   fig.show()

**Train vs Test Comparison:**

.. code-block:: python

   # Compare both training and test performance
   fig = plot_model_comparison(
       stats_list,
       model_names=model_names,
       metrics=["rmse", "r_squared"],
       split="both",  # Show both train and test
       plot_type="bar"
   )
   fig.show()

This helps identify overfitting (large gap between train and test performance).

**Custom Metric Selection:**

.. code-block:: python

   # Compare only specific metrics
   fig = plot_model_comparison(
       stats_list,
       model_names=["Prophet", "ARIMA", "Linear"],
       metrics=["rmse", "mape"],  # Only these two
       split="test",
       height=400,
       width=800
   )
   fig.show()

----

Hyperparameter Tuning Visualization
------------------------------------

plot_tune_results()
~~~~~~~~~~~~~~~~~~~

Visualize how model performance varies with hyperparameter values during tuning.

**Function Signature:**

.. code-block:: python

   plot_tune_results(
       tune_results,     # TuneResults object
       metric="rmse",    # Metric to visualize
       plot_type="auto", # Visualization type
       title=None,       # Custom title
       height=500,       # Plot height
       width=None,       # Plot width
       show_best=0       # Highlight N best configs
   )

**Plot Types:**

- ``"auto"``: Automatically choose based on number of parameters
- ``"line"``: Line plot for single parameter (1D)
- ``"heatmap"``: Heatmap for two parameters (2D)
- ``"parallel"``: Parallel coordinates for 3+ parameters
- ``"scatter"``: Scatter plot matrix for 2+ parameters

**Single Parameter Tuning:**

.. code-block:: python

   from py_tune import tune, tune_grid, grid_regular
   from py_visualize import plot_tune_results
   from py_rsample import vfold_cv

   # Create tuning workflow (tune number of trees)
   wf = workflow().add_formula("sales ~ .").add_model(
       rand_forest(trees=tune(), mtry=3, min_n=5).set_mode("regression")
   )

   # Define parameter grid
   grid = grid_regular(
       {"trees": {"range": (50, 500), "levels": 10}},
       levels=10
   )

   # Tune
   cv_splits = vfold_cv(train, v=5)
   results = tune_grid(wf, resamples=cv_splits, grid=grid)

   # Visualize (line plot)
   fig = plot_tune_results(
       results,
       metric="rmse",
       plot_type="line",  # or "auto"
       title="RMSE vs Number of Trees",
       show_best=3  # Highlight 3 best
   )
   fig.show()

**Two Parameter Tuning:**

.. code-block:: python

   # Tune two parameters
   wf = workflow().add_formula("sales ~ .").add_model(
       rand_forest(
           trees=tune(),
           min_n=tune(),
           mtry=3
       ).set_mode("regression")
   )

   # 2D grid
   grid = grid_regular(
       {
           "trees": {"range": (50, 500), "levels": 5},
           "min_n": {"range": (2, 40), "levels": 5}
       }
   )

   results = tune_grid(wf, resamples=cv_splits, grid=grid)

   # Heatmap visualization
   fig = plot_tune_results(
       results,
       metric="rmse",
       plot_type="heatmap",
       title="RMSE Heatmap: Trees vs Min N"
   )
   fig.show()

**What the Heatmap Shows:**

- X-axis: First parameter (trees)
- Y-axis: Second parameter (min_n)
- Color: Metric value (darker/lighter = better/worse)
- Easy to spot optimal parameter combinations

**Multiple Parameter Tuning (3+):**

For 3 or more parameters, use parallel coordinates:

.. code-block:: python

   # Tune 4 parameters
   wf = workflow().add_formula("sales ~ .").add_model(
       boost_tree(
           trees=tune(),
           tree_depth=tune(),
           learn_rate=tune(),
           min_n=tune()
       ).set_mode("regression")
   )

   # Random grid for efficiency
   from py_tune import grid_random
   grid = grid_random(
       {
           "trees": {"range": (50, 500)},
           "tree_depth": {"range": (3, 10)},
           "learn_rate": {"range": (0.001, 0.1), "trans": "log"},
           "min_n": {"range": (2, 40)}
       },
       size=50  # 50 random combinations
   )

   results = tune_grid(wf, resamples=cv_splits, grid=grid)

   # Parallel coordinates plot
   fig = plot_tune_results(
       results,
       metric="rmse",
       plot_type="parallel",
       show_best=5,  # Highlight 5 best configurations
       height=600
   )
   fig.show()

**What the Parallel Plot Shows:**

- Each vertical axis represents one parameter
- Lines connect parameter values for each configuration
- Color indicates metric value
- Best configurations are highlighted
- Helps identify parameter interactions

**Scatter Matrix (2-3 Parameters):**

.. code-block:: python

   # Pairwise scatter plots
   fig = plot_tune_results(
       results,
       metric="r_squared",
       plot_type="scatter",
       title="Parameter Relationships"
   )
   fig.show()

Shows scatter plots for all parameter pairs, helping identify correlations and interactions.

**Highlighting Best Configurations:**

.. code-block:: python

   # Show top 10 configurations
   fig = plot_tune_results(
       results,
       metric="rmse",
       show_best=10,  # Highlight best 10 with markers
       plot_type="auto"
   )
   fig.show()

Best configurations are marked with special symbols and/or colors, making them easy to identify.

----

Saving and Exporting Plots
---------------------------

All plotting functions return Plotly Figure objects that can be saved in multiple formats:

**HTML (Interactive):**

.. code-block:: python

   # Save interactive HTML file
   fig = plot_forecast(fit)
   fig.write_html("forecast.html")

   # Open in browser
   import webbrowser
   webbrowser.open("forecast.html")

**Static Images:**

.. code-block:: python

   # Requires kaleido: pip install kaleido
   fig = plot_residuals(fit)

   # Save as PNG
   fig.write_image("diagnostics.png", width=1200, height=800)

   # Save as PDF
   fig.write_image("diagnostics.pdf")

   # Save as SVG
   fig.write_image("diagnostics.svg")

**Jupyter Notebook Display:**

.. code-block:: python

   # Default: show() displays inline
   fig = plot_forecast(fit)
   fig.show()

   # Control renderer
   fig.show(renderer="browser")  # Open in browser
   fig.show(renderer="notebook")  # Inline (default)
   fig.show(renderer="colab")  # Google Colab

----

Advanced Customization
----------------------

All plots can be customized using Plotly's extensive API:

**Layout Customization:**

.. code-block:: python

   fig = plot_forecast(fit)

   # Update layout
   fig.update_layout(
       title="Sales Forecast - Q4 2024",
       title_font_size=20,
       title_x=0.5,  # Center title
       font=dict(family="Arial", size=12),
       hovermode='x unified',
       template='plotly_white',  # or 'plotly_dark', 'ggplot2', etc.
       showlegend=True,
       legend=dict(
           orientation="h",
           yanchor="bottom",
           y=1.02,
           xanchor="right",
           x=1
       )
   )

   fig.show()

**Axis Customization:**

.. code-block:: python

   fig = plot_model_comparison(stats_list, model_names=["A", "B", "C"])

   # Customize axes
   fig.update_xaxes(title="Metric", title_font_size=14)
   fig.update_yaxes(title="Value", title_font_size=14, gridcolor='lightgray')

   fig.show()

**Color Schemes:**

.. code-block:: python

   fig = plot_tune_results(results, metric="rmse")

   # Change color scale
   fig.update_traces(marker=dict(colorscale='Viridis'))

   fig.show()

**Adding Annotations:**

.. code-block:: python

   fig = plot_forecast(fit)

   # Add custom annotations
   fig.add_annotation(
       x='2024-10-15',
       y=1000,
       text="Promotion Start",
       showarrow=True,
       arrowhead=2
   )

   # Add horizontal line
   fig.add_hline(y=950, line_dash="dash", line_color="red",
                 annotation_text="Target")

   fig.show()

**Combining Plots:**

.. code-block:: python

   from plotly.subplots import make_subplots

   # Create custom subplot grid
   fig = make_subplots(
       rows=2, cols=1,
       subplot_titles=("Forecast", "Residuals"),
       vertical_spacing=0.15
   )

   # Get individual plots
   fig_forecast = plot_forecast(fit)
   fig_residuals = plot_residuals(fit, plot_type="fitted")

   # Add traces to subplots
   for trace in fig_forecast.data:
       fig.add_trace(trace, row=1, col=1)

   for trace in fig_residuals.data:
       fig.add_trace(trace, row=2, col=1)

   fig.update_layout(height=800, showlegend=True)
   fig.show()

----

Best Practices
--------------

**1. Always Check Diagnostics:**

.. code-block:: python

   # After fitting any model, check residuals
   fit = workflow().add_model(linear_reg()).fit(train).evaluate(test)
   plot_residuals(fit, plot_type="all")

**2. Compare Multiple Models:**

.. code-block:: python

   # Don't rely on a single model
   models = [prophet_reg(), arima_reg(), linear_reg()]
   fits = [fit_model(m, train, test) for m in models]

   # Visual comparison
   stats = [f.extract_outputs()[2] for f in fits]
   plot_model_comparison(stats, model_names=["Prophet", "ARIMA", "Linear"])

**3. Visualize Tuning Results:**

.. code-block:: python

   # Understand hyperparameter effects
   results = tune_grid(wf, resamples, grid)
   plot_tune_results(results, metric="rmse", show_best=5)

**4. Save Important Plots:**

.. code-block:: python

   # Document your work
   fig = plot_forecast(final_fit, title="Production Model - v1.2")
   fig.write_html(f"forecast_{date.today()}.html")

**5. Use Appropriate Plot Types:**

- **Forecasting**: ``plot_forecast()`` for time series
- **Diagnostics**: ``plot_residuals(plot_type="all")`` for validation
- **Selection**: ``plot_model_comparison()`` for choosing best model
- **Tuning**: ``plot_tune_results()`` for hyperparameter optimization

----

Troubleshooting
---------------

**"No data found" errors:**

.. code-block:: python

   # Ensure you've evaluated the model
   fit = wf.fit(train).evaluate(test)  # Don't forget .evaluate()
   plot_forecast(fit)  # Now it works

**"Metric not found" errors:**

.. code-block:: python

   # Check available metrics
   _, _, stats = fit.extract_outputs()
   print(stats['metric'].unique())

   # Use available metric
   plot_tune_results(results, metric="rmse")  # Not "RMSE"

**Empty plots:**

.. code-block:: python

   # Check if data has date column for time plots
   outputs, _, _ = fit.extract_outputs()
   print(outputs.columns)  # Should include 'date'

   # Or use observation index
   plot_residuals(fit, plot_type="time")  # Works without date

**Slow rendering:**

.. code-block:: python

   # For large datasets, downsample
   outputs_sample = outputs.iloc[::10]  # Every 10th point

   # Or reduce plot complexity
   plot_forecast(fit, prediction_intervals=False)  # Faster

----

API Reference
-------------

For complete API documentation, see:

- :doc:`../api/visualize` - Complete py_visualize API reference

Examples
--------

For more examples, see:

- :doc:`../examples/time_series_forecasting` - Time series visualization examples
- :doc:`../examples/hyperparameter_tuning` - Tuning visualization examples
- :doc:`../examples/model_stacking` - Multi-model comparison examples

.. seealso::

   - `Plotly Documentation <https://plotly.com/python/>`_ - For advanced customization
   - :doc:`time_series` - Time series modeling guide
   - :doc:`tuning` - Hyperparameter tuning guide
   - :doc:`workflows` - Workflow composition guide
