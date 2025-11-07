py-workflowsets: Multi-Model Comparison
=========================================

The ``py_workflowsets`` package provides tools for efficiently comparing multiple workflows across different preprocessing strategies and models.

Core Classes
------------

.. autoclass:: py_workflowsets.WorkflowSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_workflowsets.WorkflowSetResults
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Workflow Set
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_workflowsets import WorkflowSet
   from py_parsnip import linear_reg, rand_forest
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, mae

   # Define preprocessing strategies (formulas)
   formulas = [
       "y ~ x1 + x2",
       "y ~ x1 + x2 + x3",
       "y ~ x1 + x2 + I(x1*x2)",  # Interaction
   ]

   # Define models
   models = [
       linear_reg(),
       linear_reg(penalty=0.1, mixture=1.0),  # Lasso
       rand_forest(trees=100, mtry=3),
   ]

   # Create all combinations (3 formulas × 3 models = 9 workflows)
   wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

Evaluate Workflows
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create resamples
   folds = vfold_cv(train_data, v=5)

   # Evaluate all workflows
   results = wf_set.fit_resamples(
       resamples=folds,
       metrics=metric_set(rmse, mae)
   )

   # Aggregate metrics
   metrics_summary = results.collect_metrics()

   # Rank by performance
   top_models = results.rank_results("rmse", n=5)

   # Visualize results
   results.autoplot("rmse")

Select Best Workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get best workflow ID
   best_wf_id = top_models.iloc[0]["wflow_id"]

   # Extract and fit best workflow
   best_wf = wf_set[best_wf_id]
   best_fit = best_wf.fit(train_data)

   # Evaluate on test data
   best_fit = best_fit.evaluate(test_data)
   outputs, coefs, stats = best_fit.extract_outputs()

From Explicit Workflows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_workflows import workflow

   # Create explicit workflows
   wf1 = workflow().add_formula("y ~ x1").add_model(linear_reg())
   wf2 = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
   wf3 = workflow().add_formula("y ~ .").add_model(rand_forest())

   # Create workflow set
   wf_set = WorkflowSet.from_workflows([
       ("minimal", wf1),
       ("medium", wf2),
       ("full_rf", wf3),
   ])

   # Evaluate
   results = wf_set.fit_resamples(resamples=folds, metrics=metrics)

Access Individual Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access by ID
   wf = wf_set["minimal_linear_reg_1"]

   # Iterate over workflows
   for wf_id, wf in wf_set.items():
       print(f"{wf_id}: {wf.spec.model_type}")

   # Get all workflow IDs
   ids = list(wf_set.keys())
