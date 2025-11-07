Linear Models
=============

py-tidymodels provides 3 linear modeling approaches for regression tasks.

linear_reg()
------------

**Purpose**: Linear regression with optional regularization

**Engines**:
- sklearn (Ridge, Lasso, ElasticNet, OLS)
- statsmodels (OLS with full statistical inference)

**Parameters**:
- penalty: Regularization strength (0 = OLS)
- mixture: L1/L2 mixing (0 = Ridge, 1 = Lasso, 0.5 = ElasticNet)

**Example**:

.. code-block:: python

   from py_parsnip import linear_reg

   # OLS
   spec = linear_reg()

   # Ridge regression
   spec = linear_reg(penalty=0.1, mixture=0)

   # Lasso
   spec = linear_reg(penalty=0.1, mixture=1)

   # ElasticNet
   spec = linear_reg(penalty=0.1, mixture=0.5)

   # With statsmodels engine (full inference)
   spec = linear_reg().set_engine("statsmodels")

   fit = spec.fit(train, "y ~ x1 + x2")
   outputs, coefs, stats = fit.extract_outputs()

poisson_reg()
-------------

**Purpose**: Poisson regression for count data

**Use Cases**:
- Count data (0, 1, 2, 3, ...)
- Event counts
- Rare events

**Example**:

.. code-block:: python

   from py_parsnip import poisson_reg

   spec = poisson_reg()
   fit = spec.fit(train, "count ~ x1 + x2")

gen_additive_mod()
------------------

**Purpose**: Generalized Additive Models (GAMs) with splines

**Benefits**:
- Non-linear relationships
- Interpretable via partial dependence plots
- Automatic smoothing

**Example**:

.. code-block:: python

   from py_parsnip import gen_additive_mod

   spec = gen_additive_mod()
   fit = spec.fit(train, "y ~ s(x1) + s(x2)")  # Splines on x1, x2

When to Use
-----------

**linear_reg()**: 
- Linear relationships
- Need interpretability
- Want statistical inference
- Regularization for high-dimensional data

**poisson_reg()**:
- Count data
- Non-negative outcomes
- Rare events

**gen_additive_mod()**:
- Non-linear relationships
- Want interpretability
- Smooth curves needed

See Also
--------

* :doc:`../api/parsnip` - Complete API reference
* :doc:`../user_guide/quickstart` - Examples
