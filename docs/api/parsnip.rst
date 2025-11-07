py-parsnip: Model Specification
=================================

The ``py_parsnip`` package provides a unified interface for specifying and fitting machine learning models.

Core Classes
------------

.. autoclass:: py_parsnip.ModelSpec
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_parsnip.ModelFit
   :members:
   :undoc-members:
   :show-inheritance:

Linear Models
-------------

.. autofunction:: py_parsnip.linear_reg

.. autofunction:: py_parsnip.poisson_reg

.. autofunction:: py_parsnip.gen_additive_mod

Tree-Based Models
-----------------

.. autofunction:: py_parsnip.decision_tree

.. autofunction:: py_parsnip.rand_forest

.. autofunction:: py_parsnip.boost_tree

Support Vector Machines
------------------------

.. autofunction:: py_parsnip.svm_rbf

.. autofunction:: py_parsnip.svm_linear

Instance-Based Models
---------------------

.. autofunction:: py_parsnip.nearest_neighbor

.. autofunction:: py_parsnip.mars

.. autofunction:: py_parsnip.mlp

Time Series Models
------------------

.. autofunction:: py_parsnip.arima_reg

.. autofunction:: py_parsnip.prophet_reg

.. autofunction:: py_parsnip.exp_smoothing

.. autofunction:: py_parsnip.seasonal_reg

.. autofunction:: py_parsnip.recursive_reg

Hybrid Time Series Models
--------------------------

.. autofunction:: py_parsnip.arima_boost

.. autofunction:: py_parsnip.prophet_boost

Baseline Models
---------------

.. autofunction:: py_parsnip.null_model

.. autofunction:: py_parsnip.naive_reg

Generic Hybrid Models
---------------------

.. autofunction:: py_parsnip.hybrid_model

Manual Coefficient Models
--------------------------

.. autofunction:: py_parsnip.manual_reg

Engine Registry
---------------

.. automodule:: py_parsnip.engine_registry
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import linear_reg

   # Create model specification
   spec = linear_reg(penalty=0.1, mixture=0.5)

   # Set engine
   spec = spec.set_engine("sklearn")

   # Fit model
   fit = spec.fit(train_data, "y ~ x1 + x2")

   # Predict
   predictions = fit.predict(test_data)

   # Extract outputs
   outputs, coefs, stats = fit.extract_outputs()

Time Series Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import prophet_reg

   # Create Prophet specification
   spec = prophet_reg(
       n_changepoints=25,
       changepoint_prior_scale=0.05,
       seasonality_prior_scale=10.0
   )

   # Fit model
   fit = spec.fit(train_data, "target ~ date + feature1 + feature2")

   # Evaluate on test set
   fit = fit.evaluate(test_data)

   # Extract comprehensive outputs
   outputs, coefs, stats = fit.extract_outputs()

Gradient Boosting
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_parsnip import boost_tree

   # Try different engines
   xgb_spec = boost_tree(trees=100, tree_depth=6).set_engine("xgboost")
   lgb_spec = boost_tree(trees=100, tree_depth=6).set_engine("lightgbm")
   cat_spec = boost_tree(trees=100, tree_depth=6).set_engine("catboost")

   # Fit and compare
   xgb_fit = xgb_spec.fit(train_data, "y ~ .")
   lgb_fit = lgb_spec.fit(train_data, "y ~ .")
   cat_fit = cat_spec.fit(train_data, "y ~ .")
