Panel/Grouped Model Examples
=============================

Examples for modeling grouped/panel data.

Nested Approach
---------------

Separate model per group:

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import rand_forest

   wf = workflow().add_formula("sales ~ x1 + x2").add_model(rand_forest(trees=100))

   # Fit per store
   nested_fit = wf.fit_nested(data, group_col="store_id")

   # Predict
   predictions = nested_fit.predict(test)

   # Extract (includes store_id column)
   outputs, coefs, stats = nested_fit.extract_outputs()

Global Approach
---------------

Single model with group as feature:

.. code-block:: python

   global_fit = wf.fit_global(data, group_col="store_id")

See Also
--------

* :doc:`../user_guide/workflows` - Panel modeling guide
* :doc:`../api/workflows` - Workflow API
