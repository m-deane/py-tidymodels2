Tree-Based Models
=================

Tree-based models for regression and classification.

decision_tree()
---------------

**Purpose**: Single decision tree

**Modes**: regression, classification

**Parameters**:
- tree_depth: Maximum depth (None = unlimited)
- min_n: Minimum samples per leaf

**Example**:

.. code-block:: python

   from py_parsnip import decision_tree

   # Regression
   spec = decision_tree(tree_depth=5, min_n=10).set_mode("regression")
   fit = spec.fit(train, "y ~ .")

   # Classification
   spec = decision_tree(tree_depth=5).set_mode("classification")
   fit = spec.fit(train, "species ~ .")

rand_forest()
-------------

**Purpose**: Random Forest ensemble

**Parameters**:
- trees: Number of trees (default: 100)
- mtry: Features per split (default: sqrt(p))
- min_n: Minimum samples per leaf

**Example**:

.. code-block:: python

   from py_parsnip import rand_forest

   spec = rand_forest(trees=100, mtry=3, min_n=5, mode="regression")
   fit = spec.fit(train, "y ~ .")

boost_tree()
------------

**Purpose**: Gradient boosting

**Engines**: xgboost, lightgbm, catboost

**Parameters**:
- trees: Number of boosting rounds
- tree_depth: Maximum depth per tree
- learn_rate: Shrinkage parameter

**Example**:

.. code-block:: python

   from py_parsnip import boost_tree

   # XGBoost
   spec = boost_tree(trees=100, tree_depth=6, learn_rate=0.1, mode="regression")
   spec = spec.set_engine("xgboost")

   # LightGBM
   spec = spec.set_engine("lightgbm")

   # CatBoost
   spec = spec.set_engine("catboost")

   fit = spec.fit(train, "y ~ .")

When to Use
-----------

**decision_tree()**: 
- Interpretability critical
- Simple baseline
- Small datasets

**rand_forest()**:
- General-purpose strong baseline
- Handles non-linearity well
- Robust to outliers

**boost_tree()**:
- Maximum performance needed
- Have tuning budget
- Large datasets

See Also
--------

* :doc:`../api/parsnip` - Complete API reference
* :doc:`ensemble_models` - Ensemble methods
