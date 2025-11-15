Installation
============

Requirements
------------

* Python 3.10 or higher
* pip package manager

Dependencies
------------

Core dependencies (automatically installed):

* pandas >= 2.3.3
* numpy >= 2.2.6
* patsy >= 1.0.2
* scikit-learn >= 1.7.2
* prophet >= 1.2.1
* statsmodels >= 0.14.5
* scipy >= 1.14.1

Optional dependencies:

* skforecast >= 0.18.0 (for recursive forecasting)
* xgboost >= 2.0.0 (for XGBoost engine)
* lightgbm >= 4.1.0 (for LightGBM engine)
* catboost >= 1.2.0 (for CatBoost engine)
* pygam >= 0.9.0 (for GAM models)
* plotly >= 5.18.0 (for visualizations)

Installation Steps
------------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-repo/py-tidymodels.git
   cd py-tidymodels

2. Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv py-tidymodels2

   # Activate on macOS/Linux
   source py-tidymodels2/bin/activate

   # Activate on Windows
   py-tidymodels2\Scripts\activate

3. Install Package
~~~~~~~~~~~~~~~~~~

Development Installation (Editable)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended for development and testing:

.. code-block:: bash

   pip install -e .

This installs the package in editable mode, so changes to source code are immediately available without reinstalling.

Standard Installation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install .

4. Install Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full functionality, install optional dependencies:

.. code-block:: bash

   # Gradient boosting engines
   pip install xgboost lightgbm catboost

   # Recursive forecasting
   pip install skforecast

   # GAM models
   pip install pygam

   # Visualizations
   pip install plotly

   # Or install all extras
   pip install -e ".[dev,viz,boost,forecast]"

5. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import py_parsnip
   import py_workflows
   import py_recipes

   print(f"py-parsnip version: {py_parsnip.__version__}")

   # Test a simple model
   from py_parsnip import linear_reg
   import pandas as pd
   import numpy as np

   # Create test data
   np.random.seed(42)
   data = pd.DataFrame({
       'x': range(10),
       'y': range(10) + np.random.randn(10)
   })

   # Fit model
   spec = linear_reg()
   fit = spec.fit(data, "y ~ x")
   print("Installation successful!")

Jupyter Notebook Setup
-----------------------

For using py-tidymodels in Jupyter notebooks:

.. code-block:: bash

   # Install Jupyter
   pip install jupyter notebook

   # Install kernel
   python -m ipykernel install --user --name=py-tidymodels2

   # Launch Jupyter
   jupyter notebook

**Important:** After modifying source code, restart the Jupyter kernel to see changes.

Development Installation
------------------------

For contributing to py-tidymodels:

.. code-block:: bash

   # Install with dev dependencies
   pip install -e ".[dev]"

   # Install documentation dependencies
   cd docs
   pip install -r requirements.txt

   # Run tests
   pytest tests/ -v

   # Check test coverage
   pytest tests/ --cov=py_parsnip --cov=py_workflows --cov-report=html

Troubleshooting
---------------

Prophet Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Prophet installation fails:

.. code-block:: bash

   # On macOS, install pystan first
   pip install pystan

   # Then install prophet
   pip install prophet

Windows Issues
~~~~~~~~~~~~~~

Some packages may require Visual C++ Build Tools on Windows:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Retry package installation

Import Errors
~~~~~~~~~~~~~

If you get import errors after installation:

.. code-block:: bash

   # Ensure virtual environment is activated
   source py-tidymodels2/bin/activate

   # Reinstall in editable mode
   pip install -e .

   # Verify Python path
   python -c "import sys; print(sys.path)"

Kernel Not Found in Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List available kernels
   jupyter kernelspec list

   # Install kernel if missing
   python -m ipykernel install --user --name=py-tidymodels2

   # Restart Jupyter and select the kernel
