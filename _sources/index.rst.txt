py-tidymodels Documentation
============================

**py-tidymodels** is a Python port of R's tidymodels ecosystem focused on time series regression and forecasting. The project implements a unified, composable interface for machine learning models with emphasis on clean architecture patterns and extensibility.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/version-0.1.0-blue.svg
   :alt: Version 0.1.0

Features
--------

* **22 Production-Ready Models**: Linear regression, gradient boosting, time series models (Prophet, ARIMA), and more
* **Unified Interface**: Consistent API across all model types
* **51 Preprocessing Steps**: Comprehensive recipe system for feature engineering
* **Time Series Support**: First-class support for forecasting and time series analysis
* **Model Tuning**: Grid search and hyperparameter optimization
* **Model Evaluation**: 17 metrics for regression and classification
* **Workflow Pipelines**: Compose preprocessing + model + postprocessing
* **Panel Modeling**: Fit separate models per group or global models
* **Model Ensembling**: Stack models with elastic net meta-learning
* **Interactive Visualizations**: Plotly-based plots for forecasting, diagnostics, and model comparison

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/py-tidymodels.git
   cd py-tidymodels

   # Create virtual environment
   python -m venv py-tidymodels2
   source py-tidymodels2/bin/activate

   # Install in editable mode
   pip install -e .

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from py_workflows import workflow
   from py_parsnip import linear_reg
   from py_recipes import recipe
   from py_yardstick import rmse, r_squared

   # Create workflow
   wf = (
       workflow()
       .add_formula("sales ~ price + advertising")
       .add_model(linear_reg())
   )

   # Fit model
   fit = wf.fit(train_data)

   # Predict and evaluate
   fit = fit.evaluate(test_data)
   outputs, coefs, stats = fit.extract_outputs()

Package Structure
-----------------

The project follows a layered architecture:

Layer 1: py-hardhat
~~~~~~~~~~~~~~~~~~~
Data preprocessing with mold/forge for consistent train/predict transformations.

Layer 2: py-parsnip
~~~~~~~~~~~~~~~~~~~
Unified model specification interface with 22 models and 28+ engine implementations.

Layer 3: py-rsample
~~~~~~~~~~~~~~~~~~~
Train/test splitting and cross-validation for time series and general data.

Layer 4: py-workflows
~~~~~~~~~~~~~~~~~~~~~
Compose preprocessing + model + postprocessing into unified workflows.

Layer 5: py-recipes
~~~~~~~~~~~~~~~~~~~~
51 preprocessing steps for advanced feature engineering.

Layer 6: py-yardstick
~~~~~~~~~~~~~~~~~~~~~
17 comprehensive model evaluation metrics.

Layer 7: py-tune
~~~~~~~~~~~~~~~~
Hyperparameter tuning with grid search and cross-validation.

Layer 8: py-workflowsets
~~~~~~~~~~~~~~~~~~~~~~~~
Multi-model comparison across different preprocessing strategies.

Additional Packages
~~~~~~~~~~~~~~~~~~~
* **py-visualize**: Interactive Plotly visualizations
* **py-stacks**: Model ensembling via stacking

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/concepts
   user_guide/recipes
   user_guide/time_series
   user_guide/tuning
   user_guide/workflows
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   complete_api_reference
   api/hardhat
   api/parsnip
   api/rsample
   api/workflows
   api/recipes
   api/yardstick
   api/tune
   api/workflowsets
   api/visualize
   api/stacks

.. toctree::
   :maxdepth: 2
   :caption: Model Reference

   models/linear_models
   models/tree_models
   models/time_series
   models/ensemble_models
   models/baseline_models

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_regression
   examples/time_series_forecasting
   examples/hyperparameter_tuning
   examples/panel_models
   examples/model_stacking

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/architecture
   development/testing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
