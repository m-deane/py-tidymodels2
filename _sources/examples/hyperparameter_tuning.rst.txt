Hyperparameter Tuning Examples
===============================

Complete examples for hyperparameter optimization.

Grid Search
-----------

.. code-block:: python

   from py_tune import tune, tune_grid, grid_regular, finalize_workflow
   from py_workflows import workflow
   from py_parsnip import boost_tree
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, mae

   # Mark for tuning
   spec = boost_tree(
       trees=tune(),
       tree_depth=tune(),
       learn_rate=tune()
   )

   wf = workflow().add_formula("y ~ .").add_model(spec)

   # Grid
   grid = grid_regular({
       "trees": {"range": (50, 200)},
       "tree_depth": {"range": (3, 10)},
       "learn_rate": {"range": (0.001, 0.1), "trans": "log"}
   }, levels=3)

   # CV and tune
   folds = vfold_cv(train, v=5)
   results = tune_grid(wf, folds, grid, metric_set(rmse, mae))

   # Finalize
   best = results.select_best("rmse")
   final_wf = finalize_workflow(wf, best)
   final_fit = final_wf.fit(train)

See Also
--------

* :doc:`../user_guide/tuning` - Tuning guide
* :doc:`../api/tune` - Tuning API
