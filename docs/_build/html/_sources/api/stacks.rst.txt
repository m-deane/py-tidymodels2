py-stacks: Model Ensembling
=============================

The ``py_stacks`` package provides model ensembling via stacking with elastic net meta-learning.

Core Classes
------------

.. autoclass:: py_stacks.ModelStack
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: py_stacks.create_stack

.. autofunction:: py_stacks.blend_predictions

Examples
--------

Basic Stacking
~~~~~~~~~~~~~~

.. code-block:: python

   from py_stacks import create_stack, blend_predictions
   from py_parsnip import linear_reg, rand_forest, boost_tree

   # Fit base models
   spec1 = linear_reg()
   spec2 = rand_forest(trees=100)
   spec3 = boost_tree(trees=100)

   fit1 = spec1.fit(train_data, "y ~ .")
   fit2 = spec2.fit(train_data, "y ~ .")
   fit3 = spec3.fit(train_data, "y ~ .")

   # Create stack
   stack = create_stack()

   # Add members
   stack.add_members([fit1, fit2, fit3])

   # Blend with elastic net meta-learner
   stack_fit = stack.blend_predictions(
       penalty=0.1,
       mixture=0.5,
       non_negative=True
   )

   # Predict
   predictions = stack_fit.predict(test_data)

Model Weights
~~~~~~~~~~~~~

.. code-block:: python

   # Get model weights
   weights = stack_fit.get_weights()
   print(weights)
   #           model    weight
   # 0    linear_reg  0.325000
   # 1   rand_forest  0.450000
   # 2    boost_tree  0.225000

   # Visualize weights
   from py_visualize import plot_model_weights
   fig = plot_model_weights(weights)
   fig.show()

Non-Negative Weights
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Force non-negative weights (no negative contributions)
   stack_fit = stack.blend_predictions(
       penalty=0.1,
       mixture=0.5,
       non_negative=True  # Ensures all weights >= 0
   )

   # Weights sum to 1 and are >= 0
   weights = stack_fit.get_weights()

Cross-Validation Stacking
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_rsample import vfold_cv

   # Create CV folds
   folds = vfold_cv(train_data, v=5)

   # Fit base models on each fold
   cv_predictions = []
   for fold in folds:
       fold_train = fold.training()
       fold_test = fold.testing()

       # Fit and predict
       pred1 = spec1.fit(fold_train, "y ~ .").predict(fold_test)
       pred2 = spec2.fit(fold_train, "y ~ .").predict(fold_test)
       pred3 = spec3.fit(fold_train, "y ~ .").predict(fold_test)

       cv_predictions.append((pred1, pred2, pred3))

   # Blend CV predictions
   stack = create_stack()
   stack_fit = stack.blend_cv_predictions(
       cv_predictions,
       train_data,
       penalty=0.1
   )

Evaluate Stack
~~~~~~~~~~~~~~

.. code-block:: python

   from py_yardstick import metric_set, rmse, mae, r_squared

   # Evaluate stacked model
   stack_fit = stack_fit.evaluate(test_data)
   outputs, coefs, stats = stack_fit.extract_outputs()

   # Compare to base models
   metrics = metric_set(rmse, mae, r_squared)

   stack_metrics = metrics(test['y'], stack_predictions['.pred'])
   fit1_metrics = metrics(test['y'], fit1_predictions['.pred'])
   fit2_metrics = metrics(test['y'], fit2_predictions['.pred'])
