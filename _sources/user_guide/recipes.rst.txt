Recipe Guide
============

Recipes provide a powerful framework for feature engineering and data preprocessing in py-tidymodels. This guide covers the core concepts and common patterns.

What Are Recipes?
-----------------

Recipes are preprocessing specifications that:

* Define a sequence of preprocessing steps
* Learn parameters from training data (prep)
* Apply transformations consistently to new data (bake)
* Prevent data leakage by separating training and application
* Enable reusable preprocessing pipelines

The prep/bake Pattern
----------------------

Basic Workflow
~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe

   # 1. Define recipe (specification)
   rec = (
       recipe()
       .step_normalize()
       .step_dummy(["category"])
   )

   # 2. Prep on training data (learn parameters)
   rec_prepped = rec.prep(train_data)

   # 3. Bake data (apply transformations)
   train_processed = rec_prepped.bake(train_data)
   test_processed = rec_prepped.bake(test_data)

Why Separate prep and bake?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**prep()** learns parameters from training data:
- Normalization: means and standard deviations
- Encoding: factor levels for categorical variables
- Imputation: median/mean values for filling missing data
- PCA: principal components and loadings

**bake()** applies learned parameters to any data:
- Uses training means/SDs for test data normalization
- Uses training factor levels for test data encoding
- Uses training medians for test data imputation
- Uses training PCA components for test data transformation

This prevents **data leakage** - test data never influences preprocessing parameters.

Recipe Steps by Category
-------------------------

Imputation Steps
~~~~~~~~~~~~~~~~

Fill missing values:

.. code-block:: python

   rec = (
       recipe()
       .step_impute_median(["numeric_col1", "numeric_col2"])
       .step_impute_mean(["other_numeric"])
       .step_impute_mode(["categorical_col"])
       .step_impute_knn(["complex_numeric"], neighbors=5)
   )

Normalization Steps
~~~~~~~~~~~~~~~~~~~

Scale numeric features:

.. code-block:: python

   rec = (
       recipe()
       .step_normalize()  # Z-score: (x - mean) / sd
       .step_range()      # Min-max: (x - min) / (max - min)
       .step_center()     # Center only: x - mean
       .step_scale()      # Scale only: x / sd
   )

Encoding Steps
~~~~~~~~~~~~~~

Transform categorical variables:

.. code-block:: python

   from py_recipes.selectors import all_nominal

   rec = (
       recipe()
       .step_dummy(all_nominal())         # One-hot encoding
       .step_target_encode(["category"])  # Target encoding
       .step_ordinal(["size"])            # Ordinal: small=1, medium=2, large=3
   )

Feature Engineering Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create new features:

.. code-block:: python

   rec = (
       recipe()
       .step_poly(["x1"], degree=3)                    # x1, x1^2, x1^3
       .step_interact([("x1", "x2")])                  # x1 * x2
       .step_ns(["age"], df=4)                         # Natural splines
       .step_log(["price", "income"])                  # Log transform
       .step_pca(["var1", "var2", "var3"], num_comp=2) # PCA
   )

Filtering Steps
~~~~~~~~~~~~~~~

Remove problematic features:

.. code-block:: python

   rec = (
       recipe()
       .step_corr(threshold=0.9)          # Remove highly correlated
       .step_nzv()                        # Remove zero variance
       .step_filter_missing(threshold=0.5) # Remove columns with >50% missing
   )

Time Series Steps
~~~~~~~~~~~~~~~~~

Create temporal features:

.. code-block:: python

   rec = (
       recipe()
       .step_lag(["sales"], lags=[1, 7, 14, 28])  # Lagged features
       .step_diff(["sales"], lag=1)                # First difference
       .step_timeseries_signature(["date"])        # Extract time features
   )

Complete Example
----------------

Real-World Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe
   from py_recipes.selectors import all_numeric, all_nominal
   from py_workflows import workflow
   from py_parsnip import boost_tree

   # Complex preprocessing pipeline
   rec = (
       recipe()
       # 1. Handle missing data
       .step_impute_median(all_numeric())
       .step_impute_mode(all_nominal())

       # 2. Create features
       .step_log(["price", "income"])
       .step_poly(["age"], degree=2)
       .step_interact([("price", "income")])

       # 3. Normalize numerics
       .step_normalize(all_numeric())

       # 4. Encode categoricals
       .step_dummy(all_nominal())

       # 5. Remove problematic features
       .step_corr(threshold=0.9)
       .step_nzv()
   )

   # Prep and bake
   rec_prepped = rec.prep(train_data)
   train_processed = rec_prepped.bake(train_data)
   test_processed = rec_prepped.bake(test_data)

   # Use with workflow
   wf = (
       workflow()
       .add_formula("y ~ .")
       .add_model(boost_tree(trees=100))
   )

   fit = wf.fit(train_processed)
   fit = fit.evaluate(test_processed)

Time Series Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes import recipe

   # Time series feature engineering
   rec = (
       recipe()
       # Create lagged features
       .step_lag(["sales", "price"], lags=[1, 7, 14, 21, 28])

       # Create rolling statistics
       .step_mutate(
           sales_ma7="sales.rolling(7).mean()",
           sales_ma28="sales.rolling(28).mean()"
       )

       # Extract time features from date
       .step_timeseries_signature(["date"])

       # First difference for stationarity
       .step_diff(["sales"], lag=1)

       # Handle missing (from lag/diff)
       .step_impute_median()

       # Normalize
       .step_normalize()
   )

   rec_prepped = rec.prep(train_ts)
   train_processed = rec_prepped.bake(train_ts)
   test_processed = rec_prepped.bake(test_ts)

Selectors
---------

Use selectors to apply steps to groups of columns:

Built-in Selectors
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_recipes.selectors import (
       all_numeric,      # All numeric columns
       all_nominal,      # All categorical columns
       all_predictors,   # All predictor columns
       all_outcomes,     # All outcome columns
       has_role,         # Columns with specific role
       has_type,         # Columns with specific type
   )

   rec = (
       recipe()
       .step_normalize(all_numeric())
       .step_dummy(all_nominal())
   )

Custom Selection
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Specific columns
   rec = recipe().step_normalize(["x1", "x2", "x3"])

   # Pattern matching
   rec = recipe().step_log(starts_with("price_"))
   rec = recipe().step_center(ends_with("_amount"))

Best Practices
--------------

Order of Operations
~~~~~~~~~~~~~~~~~~~

Apply steps in this order for best results:

1. **Imputation** - Fill missing values first
2. **Feature Creation** - Log, polynomial, interactions
3. **Normalization** - Scale numeric features
4. **Encoding** - Transform categoricals
5. **Filtering** - Remove problematic features

.. code-block:: python

   rec = (
       recipe()
       .step_impute_median()      # 1. Impute
       .step_log(["price"])       # 2. Transform
       .step_normalize()          # 3. Normalize
       .step_dummy(["category"])  # 4. Encode
       .step_corr(threshold=0.9)  # 5. Filter
   )

Avoid Data Leakage
~~~~~~~~~~~~~~~~~~

**Always prep on training data only:**

.. code-block:: python

   # CORRECT
   rec_prepped = rec.prep(train_data)
   train_baked = rec_prepped.bake(train_data)
   test_baked = rec_prepped.bake(test_data)

   # WRONG - Data leakage!
   rec_prepped = rec.prep(all_data)  # Don't use test data

Reuse Prepped Recipes
~~~~~~~~~~~~~~~~~~~~~~

Save and reuse preprocessing:

.. code-block:: python

   import pickle

   # Save prepped recipe
   with open('recipe.pkl', 'wb') as f:
       pickle.dump(rec_prepped, f)

   # Load and apply to new data
   with open('recipe.pkl', 'rb') as f:
       rec_loaded = pickle.load(f)

   new_data_processed = rec_loaded.bake(new_data)

Common Patterns
---------------

Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Strategy 1: Median imputation
   rec = recipe().step_impute_median()

   # Strategy 2: Indicator variables
   rec = (
       recipe()
       .step_mutate(price_missing="price.isna()")
       .step_impute_median(["price"])
   )

   # Strategy 3: Advanced imputation
   rec = (
       recipe()
       .step_impute_knn(["numeric_col"], neighbors=5)
       .step_impute_bag(["complex_col"])
   )

Dealing with Skewness
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Log transform for right-skewed data
   rec = recipe().step_log(["price", "income"])

   # Box-Cox for more flexibility
   rec = recipe().step_BoxCox(["price"])

   # Yeo-Johnson (handles zeros/negatives)
   rec = recipe().step_YeoJohnson(["price"])

Reducing Dimensionality
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PCA for many correlated features
   rec = (
       recipe()
       .step_normalize()  # PCA needs normalized data
       .step_pca(all_numeric(), num_comp=10)
   )

   # Remove correlated features
   rec = recipe().step_corr(threshold=0.9)

Rare Categories
~~~~~~~~~~~~~~~

.. code-block:: python

   # Pool infrequent categories
   rec = recipe().step_other(["category"], threshold=0.05)

   # Or remove rare categories
   rec = (
       recipe()
       .step_filter("category.value_counts() / len(category) > 0.05")
   )

Next Steps
----------

* :doc:`../api/recipes` - Complete API reference
* :doc:`workflows` - Integrate recipes with workflows
* :doc:`time_series` - Time series specific recipes
* :doc:`../examples/basic_regression` - More examples
