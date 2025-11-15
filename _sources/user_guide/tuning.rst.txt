Hyperparameter Tuning
======================

This guide covers hyperparameter optimization with grid search and cross-validation.

Quick Start
-----------

.. code-block:: python

   from py_tune import tune, tune_grid, grid_regular, finalize_workflow
   from py_workflows import workflow
   from py_parsnip import linear_reg
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, r_squared

   # Mark parameters for tuning
   spec = linear_reg(penalty=tune(), mixture=tune())

   # Create workflow
   wf = workflow().add_formula("y ~ .").add_model(spec)

   # Create parameter grid
   grid = grid_regular({
       "penalty": {"range": (0.001, 1.0), "trans": "log"},
       "mixture": {"range": (0, 1)}
   }, levels=5)

   # Create CV folds
   folds = vfold_cv(train, v=5)

   # Tune
   results = tune_grid(wf, resamples=folds, grid=grid, 
                       metrics=metric_set(rmse, r_squared))

   # Get best parameters
   best = results.select_best("rmse", maximize=False)

   # Finalize workflow
   final_wf = finalize_workflow(wf, best)
   final_fit = final_wf.fit(train)

Grid Types
----------

Regular Grid
~~~~~~~~~~~~

Evenly-spaced grid:

.. code-block:: python

   from py_tune import grid_regular

   grid = grid_regular({
       "penalty": {"range": (0.001, 1.0), "trans": "log"},
       "mixture": {"range": (0, 1)},
       "tree_depth": {"range": (3, 10)}
   }, levels=5)  # 5 values per parameter = 5^3 = 125 combinations

Random Grid
~~~~~~~~~~~

Random sampling:

.. code-block:: python

   from py_tune import grid_random

   grid = grid_random({
       "penalty": {"range": (0.001, 1.0), "trans": "log"},
       "mixture": {"range": (0, 1)},
       "tree_depth": {"range": (3, 10)}
   }, size=50, seed=123)  # 50 random combinations

Analyzing Results
-----------------

.. code-block:: python

   # Show best models
   top_5 = results.show_best("rmse", n=5, maximize=False)

   # Select best by metric
   best_params = results.select_best("rmse", maximize=False)

   # Select by one-standard-error rule (simpler model)
   simple_params = results.select_by_one_std_err(
       "rmse", maximize=False, metric="penalty"
   )

Time Series Tuning
------------------

Use time series CV:

.. code-block:: python

   from py_rsample import time_series_cv

   folds = time_series_cv(
       data,
       date_column="date",
       initial="6 months",
       assess="1 month",
       cumulative=True  # Expanding window
   )

   results = tune_grid(wf, resamples=folds, grid=grid, metrics=metrics)

Best Practices
--------------

1. **Use appropriate CV**: Time series → time_series_cv, Classification → stratified vfold_cv
2. **Log transform**: Use "trans": "log" for penalty-like parameters
3. **Start coarse**: Begin with levels=3, then refine around best region
4. **Check multiple metrics**: Don't optimize for just one metric
5. **Validate on holdout**: Always test on separate test set

See Also
--------

* :doc:`../api/tune` - Complete API reference
* :doc:`time_series` - Time series specific tuning
* :doc:`../examples/hyperparameter_tuning` - More examples
