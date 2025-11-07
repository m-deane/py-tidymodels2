py-yardstick: Model Metrics
=============================

The ``py_yardstick`` package provides 17 comprehensive model evaluation metrics for regression and classification.

Main Functions
--------------

Metric Set
~~~~~~~~~~

.. autofunction:: py_yardstick.metric_set

Regression Metrics (7)
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: py_yardstick.rmse

.. autofunction:: py_yardstick.mae

.. autofunction:: py_yardstick.mape

.. autofunction:: py_yardstick.smape

.. autofunction:: py_yardstick.r_squared

.. autofunction:: py_yardstick.adj_r_squared

.. autofunction:: py_yardstick.rse

Classification Metrics (10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: py_yardstick.accuracy

.. autofunction:: py_yardstick.precision

.. autofunction:: py_yardstick.recall

.. autofunction:: py_yardstick.f1_score

.. autofunction:: py_yardstick.specificity

.. autofunction:: py_yardstick.balanced_accuracy

.. autofunction:: py_yardstick.mcc

.. autofunction:: py_yardstick.roc_auc

.. autofunction:: py_yardstick.log_loss

.. autofunction:: py_yardstick.brier_score

Examples
--------

Individual Metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_yardstick import rmse, mae, r_squared

   # Calculate metrics (returns DataFrames)
   rmse_df = rmse(y_true, y_pred)
   mae_df = mae(y_true, y_pred)
   r2_df = r_squared(y_true, y_pred)

   # Extract scalar values
   rmse_val = rmse(y_true, y_pred).iloc[0]["value"]
   mae_val = mae(y_true, y_pred).iloc[0]["value"]
   r2_val = r_squared(y_true, y_pred).iloc[0]["value"]

Metric Set
~~~~~~~~~~

.. code-block:: python

   from py_yardstick import metric_set, rmse, mae, r_squared

   # Create metric set
   metrics = metric_set(rmse, mae, r_squared)

   # Evaluate
   results = metrics(y_true, y_pred)
   print(results)
   #    .metric     value
   # 0     rmse  5.234567
   # 1      mae  4.123456
   # 2  r_squared  0.856789

With DataFrames
~~~~~~~~~~~~~~~

.. code-block:: python

   from py_yardstick import metric_set, rmse, mae
   import pandas as pd

   # DataFrame with predictions
   df = pd.DataFrame({
       'truth': [1, 2, 3, 4, 5],
       'estimate': [1.1, 2.2, 2.9, 4.1, 4.8]
   })

   # Evaluate directly on DataFrame
   metrics = metric_set(rmse, mae)
   results = metrics(df['truth'], df['estimate'])

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_yardstick import metric_set, accuracy, precision, recall, f1_score
   import numpy as np

   # Binary classification
   y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1])
   y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])

   # Evaluate classification metrics
   metrics = metric_set(accuracy, precision, recall, f1_score)
   results = metrics(y_true, y_pred)
   print(results)
   #       .metric     value
   # 0    accuracy  0.875000
   # 1   precision  1.000000
   # 2      recall  0.800000
   # 3    f1_score  0.888889
