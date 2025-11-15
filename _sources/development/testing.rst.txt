Testing Guide
=============

Guidelines for testing py-tidymodels.

Test Structure
--------------

.. code-block:: bash

   tests/
   ├── test_hardhat/     # Data preprocessing tests
   ├── test_parsnip/     # Model tests
   ├── test_rsample/     # Resampling tests
   ├── test_workflows/   # Workflow tests
   ├── test_recipes/     # Recipe tests
   ├── test_yardstick/   # Metric tests
   ├── test_tune/        # Tuning tests
   └── test_workflowsets/ # Multi-model tests

Running Tests
-------------

.. code-block:: bash

   # All tests
   pytest tests/ -v

   # Specific test file
   pytest tests/test_parsnip/test_linear_reg.py -v

   # Specific test
   pytest tests/test_parsnip/test_linear_reg.py::TestLinearRegFit::test_fit_with_formula -v

Test Coverage
-------------

Current coverage: 900+ tests passing

.. code-block:: bash

   pytest tests/ --cov=py_parsnip --cov=py_workflows --cov-report=html
   open htmlcov/index.html

See Also
--------

* CLAUDE.md - Testing instructions
