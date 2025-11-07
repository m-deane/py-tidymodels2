Workflow Guide
==============

Workflows compose preprocessing, model fitting, and prediction into unified pipelines.

Basic Workflow
--------------

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import linear_reg

   wf = (
       workflow()
       .add_formula("y ~ x1 + x2")
       .add_model(linear_reg())
   )

   # Fit
   wf_fit = wf.fit(train)

   # Predict
   predictions = wf_fit.predict(test)

   # Evaluate
   wf_fit = wf_fit.evaluate(test)
   outputs, coefs, stats = wf_fit.extract_outputs()

Method Chaining
---------------

Complete pipeline in one chain:

.. code-block:: python

   outputs, coefs, stats = (
       workflow()
       .add_formula("y ~ .")
       .add_model(linear_reg())
       .fit(train)
       .evaluate(test)
       .extract_outputs()
   )

With Recipes
------------

Integrate preprocessing:

.. code-block:: python

   from py_recipes import recipe

   rec = (
       recipe()
       .step_normalize()
       .step_dummy(["category"])
   )

   rec_prepped = rec.prep(train)
   train_proc = rec_prepped.bake(train)
   test_proc = rec_prepped.bake(test)

   wf = (
       workflow()
       .add_formula("y ~ .")
       .add_model(linear_reg())
   )

   wf_fit = wf.fit(train_proc)
   wf_fit = wf_fit.evaluate(test_proc)

Panel/Grouped Models
--------------------

Nested Approach
~~~~~~~~~~~~~~~

Separate model per group:

.. code-block:: python

   # One model per store
   nested_fit = wf.fit_nested(data, group_col="store_id")

   # Predict for all stores
   predictions = nested_fit.predict(test)

   # Extract outputs (includes store_id column)
   outputs, coefs, stats = nested_fit.extract_outputs()

Global Approach
~~~~~~~~~~~~~~~

Single model with group as feature:

.. code-block:: python

   # Group becomes a predictor
   global_fit = wf.fit_global(data, group_col="store_id")

When to Use Each:
- **Nested**: Different patterns per group (e.g., premium vs budget stores)
- **Global**: Similar patterns, or limited data per group

Extracting Components
----------------------

.. code-block:: python

   # Fit workflow
   wf_fit = wf.fit(train)

   # Extract individual pieces
   model_fit = wf_fit.extract_fit_parsnip()
   preprocessor = wf_fit.extract_preprocessor()
   model_spec = wf_fit.extract_spec_parsnip()

   # Access underlying engine
   sklearn_model = model_fit.extract_fit_engine()

Best Practices
--------------

1. **Use workflows for consistency**: Ensures same preprocessing for train/test
2. **Method chain for clarity**: Read top-to-bottom pipeline
3. **Extract for diagnostics**: Get fitted model for detailed analysis
4. **Panel models for heterogeneity**: Use nested when groups differ significantly

See Also
--------

* :doc:`../api/workflows` - Complete API reference
* :doc:`recipes` - Preprocessing with recipes
* :doc:`../examples/panel_models` - Panel modeling examples
