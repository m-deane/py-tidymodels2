py-tune: Hyperparameter Tuning
================================

The ``py_tune`` package provides grid search and hyperparameter optimization with cross-validation.

Core Classes
------------

.. autoclass:: py_tune.TuneResults
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

Tuning
~~~~~~

.. autofunction:: py_tune.tune

.. autofunction:: py_tune.tune_grid

.. autofunction:: py_tune.fit_resamples

Grid Generation
~~~~~~~~~~~~~~~

.. autofunction:: py_tune.grid_regular

.. autofunction:: py_tune.grid_random

Finalization
~~~~~~~~~~~~

.. autofunction:: py_tune.finalize_workflow

Examples
--------

Basic Tuning
~~~~~~~~~~~~

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

   # Create resamples
   folds = vfold_cv(train_data, v=5)

   # Tune grid search
   results = tune_grid(
       wf,
       resamples=folds,
       grid=grid,
       metrics=metric_set(rmse, r_squared)
   )

Analyze Results
~~~~~~~~~~~~~~~

.. code-block:: python

   # Show best models
   best = results.show_best("rmse", n=5, maximize=False)
   print(best)

   # Select best by metric
   best_params = results.select_best("rmse", maximize=False)

   # Select by one-std-error rule
   simple_params = results.select_by_one_std_err(
       "rmse",
       maximize=False,
       metric="penalty"
   )

Finalize Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Apply best parameters
   final_wf = finalize_workflow(wf, best_params)

   # Fit on full training data
   final_fit = final_wf.fit(train_data)

   # Evaluate on test data
   final_fit = final_fit.evaluate(test_data)
   outputs, coefs, stats = final_fit.extract_outputs()

Random Grid
~~~~~~~~~~~

.. code-block:: python

   from py_tune import grid_random

   # Random parameter sampling
   grid = grid_random({
       "penalty": {"range": (0.001, 1.0), "trans": "log"},
       "mixture": {"range": (0, 1)},
       "tree_depth": {"range": (3, 10)}
   }, size=20, seed=123)

   # Tune with random grid
   results = tune_grid(wf, resamples=folds, grid=grid, metrics=metrics)

Fit Resamples Without Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_tune import fit_resamples

   # Evaluate fixed model across resamples
   spec = linear_reg()  # No tune() markers
   wf = workflow().add_formula("y ~ .").add_model(spec)

   results = fit_resamples(
       wf,
       resamples=folds,
       metrics=metric_set(rmse, mae, r_squared)
   )

   # Aggregate metrics
   mean_metrics = results.collect_metrics()
