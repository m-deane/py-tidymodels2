Model Stacking Examples
=======================

Examples for model ensembling via stacking.

Basic Stacking
--------------

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

   stack_fit = stack.blend_predictions(
       penalty=0.1,
       mixture=0.5,
       non_negative=True
   )

   # View weights
   weights = stack_fit.get_weights()
   print(weights)

   # Predict
   predictions = stack_fit.predict(test)

See Also
--------

* :doc:`../api/stacks` - Stacking API
