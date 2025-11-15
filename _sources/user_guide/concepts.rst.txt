Core Concepts
=============

Architecture Overview
---------------------

py-tidymodels follows a layered architecture inspired by R's tidymodels ecosystem:

.. image:: ../_static/architecture.png
   :alt: Architecture Diagram
   :align: center

Layer 1: py-hardhat
~~~~~~~~~~~~~~~~~~~

**Purpose**: Low-level data preprocessing abstraction

**Key Components**:

* **Blueprint**: Immutable preprocessing metadata (formula, factor levels, column order)
* **MoldedData**: Preprocessed data ready for modeling (predictors, outcomes, extras)
* **mold()**: Formula → model matrix conversion (training phase)
* **forge()**: Apply blueprint to new data (prediction phase)

**Why It Matters**: Ensures consistent transformations between training and prediction. The same categorical encodings, column order, and transformations are applied to test data.

Layer 2: py-parsnip
~~~~~~~~~~~~~~~~~~~

**Purpose**: Unified model specification interface

**Key Design Patterns**:

1. **Immutable Specifications**: ModelSpec is a frozen dataclass
2. **Registry-Based Engines**: Decorator pattern for engine registration
3. **Dual-Path Preprocessing**: Standard vs Raw data handling
4. **Standardized Outputs**: Three-DataFrame pattern

**Models Available**: 22 production-ready models across 8 categories

Layer 3: py-rsample
~~~~~~~~~~~~~~~~~~~

**Purpose**: Train/test splitting and cross-validation

**Features**:

* Time series splits with period parsing
* K-fold cross-validation with stratification
* Rolling and expanding window CV
* Explicit date range support

Layer 4-8: Higher-Level Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **py-workflows**: Compose preprocessing + model + postprocessing
* **py-recipes**: 51 preprocessing steps for feature engineering
* **py-yardstick**: 17 evaluation metrics
* **py-tune**: Hyperparameter optimization
* **py-workflowsets**: Multi-model comparison

Key Design Principles
---------------------

Immutability
~~~~~~~~~~~~

Model specifications are immutable to prevent side effects:

.. code-block:: python

   spec = linear_reg(penalty=0.1)

   # Create new spec instead of modifying
   new_spec = spec.set_args(penalty=0.2)

   # Original spec unchanged
   assert spec.args["penalty"] == 0.1
   assert new_spec.args["penalty"] == 0.2

Composability
~~~~~~~~~~~~~

Components can be combined in flexible ways:

.. code-block:: python

   # Compose preprocessing + model
   wf = (
       workflow()
       .add_formula("y ~ x1 + x2")
       .add_model(linear_reg())
   )

   # Or use recipes
   rec = recipe().step_normalize().step_dummy(["category"])
   wf = workflow().add_recipe(rec).add_model(linear_reg())

Method Chaining
~~~~~~~~~~~~~~~

Fluent interfaces enable readable pipelines:

.. code-block:: python

   outputs, coefs, stats = (
       workflow()
       .add_formula("y ~ .")
       .add_model(linear_reg())
       .fit(train)
       .evaluate(test)
       .extract_outputs()
   )

Standardized Outputs
~~~~~~~~~~~~~~~~~~~~

All models return three DataFrames:

1. **Outputs**: Observation-level results (actuals, fitted, residuals)
2. **Coefficients**: Model parameters with statistical inference
3. **Stats**: Model-level metrics and diagnostics

This consistency makes it easy to compare models and visualize results.

Data Flow
---------

Training Flow
~~~~~~~~~~~~~

.. code-block:: text

   Raw Data → mold() → MoldedData → Engine.fit() → ModelFit
                ↓
            Blueprint (saved for prediction)

Prediction Flow
~~~~~~~~~~~~~~~

.. code-block:: text

   New Data → forge(blueprint) → Transformed Data → Engine.predict() → Predictions

The blueprint ensures test data gets the same transformations as training data.

Working with Formulas
---------------------

Basic Formulas
~~~~~~~~~~~~~~

.. code-block:: python

   # Simple formula
   "y ~ x1 + x2"

   # All predictors
   "y ~ ."

   # Interactions
   "y ~ x1 + x2 + I(x1*x2)"

   # Polynomials
   "y ~ x1 + I(x1**2) + I(x1**3)"

Categorical Variables
~~~~~~~~~~~~~~~~~~~~~

Automatically one-hot encoded:

.. code-block:: python

   # If 'category' has levels ['A', 'B', 'C']
   molded = mold("y ~ category", data)

   # Creates columns: category_B, category_C
   # (A is reference level)

Working with Recipes
--------------------

Recipe Workflow
~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Define recipe
   rec = recipe().step_normalize().step_dummy(["cat"])

   # 2. Prep on training data (learn parameters)
   rec_prepped = rec.prep(train)

   # 3. Bake data (apply transformations)
   train_processed = rec_prepped.bake(train)
   test_processed = rec_prepped.bake(test)

Why prep/bake?
~~~~~~~~~~~~~~

* **prep()**: Learns parameters from training data (means, std devs, factor levels)
* **bake()**: Applies learned parameters to any data (train or test)
* This prevents data leakage - test data never influences preprocessing parameters

Time Series Considerations
--------------------------

Date Column Handling
~~~~~~~~~~~~~~~~~~~~

Time series models need special handling:

.. code-block:: python

   # Date as predictor (auto-detected)
   spec = prophet_reg()
   fit = spec.fit(data, "sales ~ date + feature1")

   # Or specify explicitly
   spec = spec.set_args(date_col="date")

Raw vs Standard Path
~~~~~~~~~~~~~~~~~~~~

* **Standard Path**: Uses mold/forge (linear models, tree models)
* **Raw Path**: Bypasses mold/forge (Prophet, ARIMA)

Raw path is needed when datetime columns cause issues with patsy's categorical encoding.

Panel/Grouped Modeling
----------------------

Nested Approach
~~~~~~~~~~~~~~~

Fit separate model per group:

.. code-block:: python

   wf = workflow().add_formula("y ~ x").add_model(linear_reg())

   # One model per store
   nested_fit = wf.fit_nested(data, group_col="store_id")

Global Approach
~~~~~~~~~~~~~~~

Single model with group as feature:

.. code-block:: python

   # Group becomes a predictor
   global_fit = wf.fit_global(data, group_col="store_id")

When to Use Each
~~~~~~~~~~~~~~~~

* **Nested**: Groups have different patterns (e.g., premium vs budget stores)
* **Global**: Groups share patterns, or limited data per group

Next Steps
----------

* :doc:`recipes` - Master feature engineering
* :doc:`time_series` - Deep dive into time series
* :doc:`workflows` - Build complex pipelines
* :doc:`../api/parsnip` - Explore all models
