Quick Start Guide
=================

This guide will get you started with py-tidymodels in 5 minutes.

Basic Regression
----------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from py_workflows import workflow
   from py_parsnip import linear_reg
   from py_yardstick import rmse, mae, r_squared

   # Create sample data
   np.random.seed(42)
   df = pd.DataFrame({
       'price': np.random.rand(100) * 100,
       'advertising': np.random.rand(100) * 50,
       'sales': np.random.rand(100) * 1000
   })

   # Split data
   train = df.iloc[:80]
   test = df.iloc[80:]

   # Create workflow
   wf = (
       workflow()
       .add_formula("sales ~ price + advertising")
       .add_model(linear_reg())
   )

   # Fit model
   wf_fit = wf.fit(train)

   # Evaluate on test data
   wf_fit = wf_fit.evaluate(test)

   # Extract results
   outputs, coefs, stats = wf_fit.extract_outputs()

   print("Coefficients:")
   print(coefs[['variable', 'coefficient', 'p_value']])

   print("\nTest Metrics:")
   test_stats = stats[stats['split'] == 'test']
   print(test_stats[test_stats['metric'].isin(['rmse', 'mae', 'r_squared'])])

Time Series Forecasting
------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from py_parsnip import prophet_reg
   from py_rsample import initial_time_split, training, testing

   # Create time series data
   dates = pd.date_range('2020-01-01', periods=100, freq='D')
   df = pd.DataFrame({
       'date': dates,
       'sales': np.cumsum(np.random.randn(100)) + 100,
       'promotion': np.random.choice([0, 1], 100)
   })

   # Time series split
   split = initial_time_split(
       df,
       date_column="date",
       train_start="2020-01-01",
       train_end="2020-03-01",
       test_start="2020-03-02",
       test_end="2020-04-09"
   )

   train = training(split)
   test = testing(split)

   # Create Prophet model
   spec = prophet_reg(
       n_changepoints=25,
       seasonality_prior_scale=10.0
   )

   # Fit and evaluate
   fit = spec.fit(train, "sales ~ date + promotion")
   fit = fit.evaluate(test)

   # Extract outputs with dates
   outputs, coefs, stats = fit.extract_outputs()

   print("Forecast:")
   print(outputs[outputs['split'] == 'test'][['date', 'actuals', 'fitted', 'residuals']])

Feature Engineering with Recipes
---------------------------------

.. code-block:: python

   from py_recipes import recipe
   from py_workflows import workflow
   from py_parsnip import linear_reg

   # Create recipe
   rec = (
       recipe()
       .step_impute_median()  # Fill missing values
       .step_normalize()  # Z-score normalization
       .step_dummy(["category"])  # One-hot encoding
       .step_interact([("price", "advertising")])  # Interaction
   )

   # Prep and bake
   rec_prepped = rec.prep(train)
   train_processed = rec_prepped.bake(train)
   test_processed = rec_prepped.bake(test)

   # Create workflow with processed data
   wf = (
       workflow()
       .add_formula("sales ~ .")
       .add_model(linear_reg())
   )

   # Fit and evaluate
   fit = wf.fit(train_processed)
   fit = fit.evaluate(test_processed)

   outputs, coefs, stats = fit.extract_outputs()

Hyperparameter Tuning
----------------------

.. code-block:: python

   from py_tune import tune, tune_grid, grid_regular, finalize_workflow
   from py_workflows import workflow
   from py_parsnip import linear_reg
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, r_squared

   # Mark parameters for tuning
   spec = linear_reg(penalty=tune(), mixture=tune())

   # Create workflow
   wf = workflow().add_formula("sales ~ .").add_model(spec)

   # Create parameter grid
   grid = grid_regular({
       "penalty": {"range": (0.001, 1.0), "trans": "log"},
       "mixture": {"range": (0, 1)}
   }, levels=5)

   # Create CV folds
   folds = vfold_cv(train, v=5)

   # Tune
   results = tune_grid(
       wf,
       resamples=folds,
       grid=grid,
       metrics=metric_set(rmse, r_squared)
   )

   # Get best parameters
   best = results.select_best("rmse", maximize=False)
   print("Best parameters:", best)

   # Finalize workflow
   final_wf = finalize_workflow(wf, best)
   final_fit = final_wf.fit(train)

Multi-Model Comparison
-----------------------

.. code-block:: python

   from py_workflowsets import WorkflowSet
   from py_parsnip import linear_reg, rand_forest, boost_tree
   from py_rsample import vfold_cv
   from py_yardstick import metric_set, rmse, mae

   # Define formulas and models
   formulas = [
       "sales ~ price + advertising",
       "sales ~ price + advertising + I(price*advertising)",
   ]

   models = [
       linear_reg(),
       rand_forest(trees=50),
       boost_tree(trees=50)
   ]

   # Create workflow set (2 formulas × 3 models = 6 workflows)
   wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

   # Evaluate all workflows
   folds = vfold_cv(train, v=5)
   results = wf_set.fit_resamples(
       resamples=folds,
       metrics=metric_set(rmse, mae)
   )

   # Rank by performance
   top_models = results.rank_results("rmse", n=3)
   print("Top 3 Models:")
   print(top_models[['wflow_id', 'mean', 'std_err']])

   # Select and fit best model
   best_wf_id = top_models.iloc[0]["wflow_id"]
   best_wf = wf_set[best_wf_id]
   best_fit = best_wf.fit(train)

Model Stacking
--------------

.. code-block:: python

   from py_stacks import create_stack
   from py_parsnip import linear_reg, rand_forest, boost_tree

   # Fit base models
   fit1 = linear_reg().fit(train, "sales ~ .")
   fit2 = rand_forest(trees=100).fit(train, "sales ~ .")
   fit3 = boost_tree(trees=100).fit(train, "sales ~ .")

   # Create stack
   stack = create_stack()
   stack.add_members([fit1, fit2, fit3])

   # Blend with elastic net
   stack_fit = stack.blend_predictions(
       penalty=0.1,
       mixture=0.5,
       non_negative=True
   )

   # Get model weights
   weights = stack_fit.get_weights()
   print("Model Weights:")
   print(weights)

   # Predict
   predictions = stack_fit.predict(test)

Next Steps
----------

* :doc:`concepts` - Learn core concepts and architecture
* :doc:`time_series` - Deep dive into time series modeling
* :doc:`recipes` - Master feature engineering with recipes
* :doc:`../api/parsnip` - Explore all 22 available models
* :doc:`../examples/basic_regression` - More detailed examples
