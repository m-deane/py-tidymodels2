py-hardhat: Data Preprocessing
================================

The ``py_hardhat`` package provides low-level data preprocessing abstraction for consistent transformations between training and prediction.

Core Classes
------------

.. autoclass:: py_hardhat.Blueprint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_hardhat.MoldedData
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: py_hardhat.mold

.. autofunction:: py_hardhat.forge

Examples
--------

Basic Mold/Forge
~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_hardhat import mold, forge

   # Mold training data
   molded = mold("y ~ x1 + x2", train_data)

   # Access components
   X_train = molded.predictors
   y_train = molded.outcomes
   blueprint = molded.blueprint

   # Forge test data with same blueprint
   forged = forge(test_data, blueprint)
   X_test = forged.predictors

Formula Transformations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_hardhat import mold

   # Patsy formula with interactions
   molded = mold("y ~ x1 + x2 + I(x1*x2)", train_data)

   # Polynomial features
   molded = mold("y ~ x1 + I(x1**2) + I(x1**3)", train_data)

   # Categorical encoding (automatic)
   molded = mold("y ~ x1 + category", train_data)

Categorical Handling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_hardhat import mold, forge

   # Mold creates factor levels from training data
   molded = mold("y ~ category", train_data)
   # category has levels: ['A', 'B', 'C']

   # Forge validates test data has same levels
   forged = forge(test_data, molded.blueprint)
   # Raises error if test has unseen level 'D'

   # Missing levels get zero columns
   # If test only has ['A', 'B'], column for 'C' is added with zeros
