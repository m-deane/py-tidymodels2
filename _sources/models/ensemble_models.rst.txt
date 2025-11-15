Ensemble Models
===============

Models that combine multiple base learners.

boost_tree()
------------

**Gradient Boosting** via XGBoost, LightGBM, or CatBoost.

See :doc:`tree_models` for details.

Hybrid Time Series
------------------

arima_boost()
~~~~~~~~~~~~~

Combines ARIMA with gradient boosting:

.. code-block:: python

   from py_parsnip import arima_boost

   spec = arima_boost(
       non_seasonal_ar=1,
       trees=100,
       tree_depth=6
   )

prophet_boost()
~~~~~~~~~~~~~~~

Combines Prophet with gradient boosting:

.. code-block:: python

   from py_parsnip import prophet_boost

   spec = prophet_boost(
       n_changepoints=25,
       trees=100
   )

Model Stacking
--------------

Stack multiple models with elastic net:

.. code-block:: python

   from py_stacks import create_stack
   from py_parsnip import linear_reg, rand_forest, boost_tree

   # Fit base models
   fit1 = linear_reg().fit(train, "y ~ .")
   fit2 = rand_forest(trees=100).fit(train, "y ~ .")
   fit3 = boost_tree(trees=100).fit(train, "y ~ .")

   # Stack
   stack = create_stack()
   stack.add_members([fit1, fit2, fit3])

   stack_fit = stack.blend_predictions(penalty=0.1, non_negative=True)

   # Get weights
   weights = stack_fit.get_weights()

See Also
--------

* :doc:`../api/stacks` - Stacking API
* :doc:`time_series` - Time series models
