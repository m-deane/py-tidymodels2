Contributing Guide
==================

Guidelines for contributing to py-tidymodels.

Development Setup
-----------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-repo/py-tidymodels.git
   cd py-tidymodels

   # Create virtual environment
   python -m venv py-tidymodels2
   source py-tidymodels2/bin/activate

   # Install in editable mode
   pip install -e .
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   # All tests
   pytest tests/ -v

   # Specific module
   pytest tests/test_parsnip/ -v

   # With coverage
   pytest tests/ --cov=py_parsnip --cov-report=html

Code Style
----------

- Follow PEP 8
- Use Google-style docstrings
- Type hints encouraged
- Maximum line length: 100

Documentation
-------------

Update documentation after code changes:

.. code-block:: bash

   cd docs
   make html

See Also
--------

* CLAUDE.md - Detailed development guidelines
* :doc:`architecture` - Architecture overview
* :doc:`testing` - Testing guidelines
