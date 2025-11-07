Basic Regression Examples
==========================

Complete examples for regression modeling.

Simple Linear Regression
-------------------------

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import linear_reg
   import pandas as pd

   # Create workflow
   wf = (
       workflow()
       .add_formula("sales ~ price + advertising")
       .add_model(linear_reg())
   )

   # Fit and evaluate
   fit = wf.fit(train)
   fit = fit.evaluate(test)

   # Extract results
   outputs, coefs, stats = fit.extract_outputs()

   print("Coefficients:")
   print(coefs[['variable', 'coefficient', 'p_value']])

   print("\nTest Metrics:")
   test_metrics = stats[stats['split'] == 'test']
   print(test_metrics[['metric', 'value']])

With Feature Engineering
-------------------------

.. code-block:: python

   from py_recipes import recipe
   from py_workflows import workflow
   from py_parsnip import linear_reg

   # Preprocessing
   rec = (
       recipe()
       .step_impute_median()
       .step_log(["price", "income"])
       .step_normalize()
       .step_dummy(["category"])
   )

   rec_prepped = rec.prep(train)
   train_proc = rec_prepped.bake(train)
   test_proc = rec_prepped.bake(test)

   # Workflow
   wf = workflow().add_formula("y ~ .").add_model(linear_reg())
   fit = wf.fit(train_proc).evaluate(test_proc)

   outputs, coefs, stats = fit.extract_outputs()

See Also
--------

* :doc:`../user_guide/quickstart` - More examples
* :doc:`../api/workflows` - Workflow API
