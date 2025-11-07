Baseline Models
===============

Simple baseline models for comparison.

null_model()
------------

**Purpose**: Simple statistical baselines

**Strategies**:
- mean: Predict mean of training data
- median: Predict median of training data

**Example**:

.. code-block:: python

   from py_parsnip import null_model

   # Mean baseline
   spec = null_model(strategy="mean")
   fit = spec.fit(train, "y ~ .")

   # Median baseline
   spec = null_model(strategy="median")

naive_reg()
-----------

**Purpose**: Time series naive forecasts

**Strategies**:
- naive: Last observation
- seasonal_naive: Last seasonal observation
- drift: Linear trend from first to last

**Example**:

.. code-block:: python

   from py_parsnip import naive_reg

   # Naive: y_t = y_{t-1}
   spec = naive_reg(strategy="naive")

   # Seasonal naive: y_t = y_{t-s}
   spec = naive_reg(strategy="seasonal_naive", seasonal_period=7)

   # Drift: linear trend
   spec = naive_reg(strategy="drift")

   fit = spec.fit(train, "sales ~ date")

When to Use
-----------

Always fit baseline models first:

1. Establish performance floor
2. Quick sanity check
3. Compare complex models against simple benchmarks

If complex model doesn't beat baseline → investigate!

See Also
--------

* :doc:`../api/parsnip` - Complete API reference
