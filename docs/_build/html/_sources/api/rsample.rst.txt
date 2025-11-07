py-rsample: Resampling & Cross-Validation
===========================================

The ``py_rsample`` package provides train/test splitting and cross-validation for time series and general data.

Core Classes
------------

.. autoclass:: py_rsample.Split
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_rsample.RSplit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_rsample.Resample
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

Train/Test Split
~~~~~~~~~~~~~~~~

.. autofunction:: py_rsample.initial_split

.. autofunction:: py_rsample.initial_time_split

.. autofunction:: py_rsample.training

.. autofunction:: py_rsample.testing

Cross-Validation
~~~~~~~~~~~~~~~~

.. autofunction:: py_rsample.vfold_cv

.. autofunction:: py_rsample.time_series_cv

Examples
--------

Basic Split
~~~~~~~~~~~

.. code-block:: python

   from py_rsample import initial_split, training, testing

   # Create 75/25 split
   split = initial_split(data, prop=0.75, seed=123)
   train = training(split)
   test = testing(split)

Time Series Split
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_rsample import initial_time_split

   # Absolute dates
   split = initial_time_split(
       data,
       date_column="date",
       train_start="2022-01-01",
       train_end="2023-12-01",
       test_start="2024-01-01",
       test_end="2024-06-01"
   )

   # Relative periods
   split = initial_time_split(
       data,
       date_column="date",
       train_start="start",
       train_end="start + 18 months",
       test_start="end - 6 months",
       test_end="end"
   )

K-Fold Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_rsample import vfold_cv

   # Standard k-fold
   folds = vfold_cv(data, v=5, seed=123)

   # Stratified k-fold
   folds = vfold_cv(data, v=5, strata="target", seed=123)

   # Repeated k-fold
   folds = vfold_cv(data, v=5, repeats=3, seed=123)

Time Series Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_rsample import time_series_cv

   # Rolling window CV
   folds = time_series_cv(
       data,
       date_column="date",
       initial="6 months",
       assess="1 month",
       skip="0 months",
       cumulative=False
   )

   # Expanding window CV
   folds = time_series_cv(
       data,
       date_column="date",
       initial="6 months",
       assess="1 month",
       cumulative=True
   )
