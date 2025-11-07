py-workflows: Pipeline Composition
====================================

The ``py_workflows`` package provides tools for composing preprocessing + model + postprocessing into unified workflows.

Core Classes
------------

.. autoclass:: py_workflows.Workflow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_workflows.WorkflowFit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: py_workflows.NestedWorkflowFit
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: py_workflows.workflow

Examples
--------

Basic Workflow
~~~~~~~~~~~~~~

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import linear_reg

   # Create workflow
   wf = (
       workflow()
       .add_formula("sales ~ price + advertising")
       .add_model(linear_reg())
   )

   # Fit and evaluate
   wf_fit = wf.fit(train_data)
   wf_fit = wf_fit.evaluate(test_data)

   # Extract outputs
   outputs, coefs, stats = wf_fit.extract_outputs()

Method Chaining
~~~~~~~~~~~~~~~

.. code-block:: python

   # Full pipeline in one chain
   outputs, coefs, stats = (
       workflow()
       .add_formula("y ~ x1 + x2 + x3")
       .add_model(linear_reg().set_engine("sklearn"))
       .fit(train_data)
       .evaluate(test_data)
       .extract_outputs()
   )

Panel/Grouped Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_workflows import workflow
   from py_parsnip import recursive_reg, rand_forest

   # Nested approach: separate model per group
   wf = (
       workflow()
       .add_formula("sales ~ lag1 + lag2 + lag3")
       .add_model(recursive_reg(base_model=rand_forest(), lags=7))
   )

   nested_fit = wf.fit_nested(data, group_col="store_id")
   predictions = nested_fit.predict(test_data)
   outputs, coefs, stats = nested_fit.extract_outputs()

   # Global approach: single model with group as feature
   global_fit = wf.fit_global(data, group_col="store_id")
   predictions = global_fit.predict(test_data)

Extracting Components
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit workflow
   wf_fit = workflow().add_formula("y ~ x").add_model(linear_reg()).fit(train_data)

   # Extract components
   model_fit = wf_fit.extract_fit_parsnip()
   preprocessor = wf_fit.extract_preprocessor()
   model_spec = wf_fit.extract_spec_parsnip()

   # Access underlying engine model
   sklearn_model = model_fit.extract_fit_engine()
