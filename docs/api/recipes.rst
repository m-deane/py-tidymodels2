py-recipes: Feature Engineering
=================================

The ``py_recipes`` package provides 51 preprocessing steps for advanced feature engineering.

Core Classes
------------

.. autoclass:: py_recipes.Recipe
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: py_recipes.recipe

Selectors
---------

.. automodule:: py_recipes.selectors
   :members:
   :undoc-members:

Imputation Steps (6)
--------------------

.. autofunction:: py_recipes.steps.step_impute_median

.. autofunction:: py_recipes.steps.step_impute_mean

.. autofunction:: py_recipes.steps.step_impute_mode

.. autofunction:: py_recipes.steps.step_impute_knn

.. autofunction:: py_recipes.steps.step_impute_bag

.. autofunction:: py_recipes.steps.step_impute_linear

Normalization Steps (4)
------------------------

.. autofunction:: py_recipes.steps.step_normalize

.. autofunction:: py_recipes.steps.step_range

.. autofunction:: py_recipes.steps.step_center

.. autofunction:: py_recipes.steps.step_scale

Encoding Steps (6)
-------------------

.. autofunction:: py_recipes.steps.step_dummy

.. autofunction:: py_recipes.steps.step_one_hot

.. autofunction:: py_recipes.steps.step_target_encode

.. autofunction:: py_recipes.steps.step_ordinal

.. autofunction:: py_recipes.steps.step_bin

.. autofunction:: py_recipes.steps.step_date

Feature Engineering Steps (8)
------------------------------

.. autofunction:: py_recipes.steps.step_poly

.. autofunction:: py_recipes.steps.step_interact

.. autofunction:: py_recipes.steps.step_ns

.. autofunction:: py_recipes.steps.step_bs

.. autofunction:: py_recipes.steps.step_pca

.. autofunction:: py_recipes.steps.step_log

.. autofunction:: py_recipes.steps.step_sqrt

.. autofunction:: py_recipes.steps.step_inverse

Filtering Steps (6)
--------------------

.. autofunction:: py_recipes.steps.step_corr

.. autofunction:: py_recipes.steps.step_nzv

.. autofunction:: py_recipes.steps.step_filter_missing

.. autofunction:: py_recipes.steps.step_select

.. autofunction:: py_recipes.steps.step_rm

.. autofunction:: py_recipes.steps.step_filter

Row Operations (6)
-------------------

.. autofunction:: py_recipes.steps.step_sample

.. autofunction:: py_recipes.steps.step_slice

.. autofunction:: py_recipes.steps.step_arrange

.. autofunction:: py_recipes.steps.step_shuffle

.. autofunction:: py_recipes.steps.step_lag

Transformation Steps (6)
-------------------------

.. autofunction:: py_recipes.steps.step_mutate

.. autofunction:: py_recipes.steps.step_discretize

.. autofunction:: py_recipes.steps.step_cut

.. autofunction:: py_recipes.steps.step_BoxCox

.. autofunction:: py_recipes.steps.step_YeoJohnson

.. autofunction:: py_recipes.steps.step_other

Time Series Steps (4)
----------------------

.. autofunction:: py_recipes.steps.step_diff

.. autofunction:: py_recipes.steps.step_timeseries_signature

Examples
--------

Basic Recipe
~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe

   # Create recipe
   rec = (
       recipe()
       .step_normalize()  # Z-score normalization
       .step_dummy(["category"])  # One-hot encoding
   )

   # Prep on training data
   rec_prepped = rec.prep(train_data)

   # Apply to train and test
   train_processed = rec_prepped.bake(train_data)
   test_processed = rec_prepped.bake(test_data)

Advanced Recipe
~~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe
   from py_recipes.selectors import all_numeric, all_nominal

   # Complex preprocessing pipeline
   rec = (
       recipe()
       .step_impute_median(all_numeric())
       .step_impute_mode(all_nominal())
       .step_log(["price", "income"])
       .step_normalize(all_numeric())
       .step_dummy(all_nominal())
       .step_poly(["age"], degree=2)
       .step_interact([("income", "age")])
       .step_corr(threshold=0.9)
       .step_nzv()
   )

   rec_prepped = rec.prep(train_data)
   train_processed = rec_prepped.bake(train_data)

Time Series Recipe
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe

   # Time series feature engineering
   rec = (
       recipe()
       .step_lag(["sales"], lags=[1, 7, 14, 28])
       .step_diff(["sales"], lag=1)
       .step_timeseries_signature(["date"])
       .step_impute_median(all_numeric())
       .step_normalize(all_numeric())
   )

   rec_prepped = rec.prep(train_data)
   train_processed = rec_prepped.bake(train_data)
