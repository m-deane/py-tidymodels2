# py-tidymodels Installation Guide

Complete installation instructions for **py-tidymodels**, a Python port of R's tidymodels ecosystem for time series regression and forecasting.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Jupyter Setup](#jupyter-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Optional Dependencies](#optional-dependencies)

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: 4GB minimum (8GB+ recommended for large datasets)
- **Disk Space**: ~2GB for virtual environment with all dependencies

### Required Tools
- `pip` (Python package installer)
- `git` (for cloning the repository)
- `virtualenv` or `venv` (for creating isolated Python environments)

---

## Quick Start

For users who want to get started immediately:

```bash
# 1. Clone the repository
git clone https://github.com/m-deane/py-tidymodels2.git
cd py-tidymodels

# 2. Create and activate virtual environment
python3 -m venv py-tidymodels-env
source py-tidymodels-env/bin/activate  # On Windows: py-tidymodels-env\Scripts\activate

# 3. Install package in editable mode
pip install -e .

# 4. Install development tools (optional)
pip install -e ".[dev]"

# 5. Verify installation
python -c "from py_parsnip import linear_reg; print('✓ Installation successful!')"
```

---

## Detailed Installation

### Step 1: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/m-deane/py-tidymodels2.git

# Or using SSH (if you have SSH keys configured)
git clone git@github.com:m-deane/py-tidymodels2.git

# Navigate to project directory
cd py-tidymodels
```

### Step 2: Create Virtual Environment

**Option A: Using venv (built-in)**
```bash
# Create virtual environment
python3 -m venv py-tidymodels-env

# Activate on macOS/Linux
source py-tidymodels-env/bin/activate

# Activate on Windows
py-tidymodels-env\Scripts\activate
```

**Option B: Using virtualenv**
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create virtual environment
virtualenv py-tidymodels-env

# Activate (same commands as above)
source py-tidymodels-env/bin/activate
```

**Option C: Using conda**
```bash
# Create conda environment
conda create -n py-tidymodels python=3.10

# Activate environment
conda activate py-tidymodels
```

### Step 3: Install the Package

**For Users (Regular Installation):**
```bash
pip install -e .
```

**For Developers (With Dev Tools):**
```bash
# Install with development dependencies
pip install -e ".[dev]"

# This includes: pytest, pytest-cov, jupyter, notebook
```

**Install All Optional Dependencies:**
```bash
# Install from requirements.txt for complete feature set
pip install -r requirements.txt
```

### Step 4: Install Additional Time Series Packages (Optional)

Some advanced models require additional packages:

```bash
# XGBoost, LightGBM, CatBoost (for gradient boosting)
pip install xgboost lightgbm catboost

# skforecast (for recursive forecasting)
pip install skforecast

# pygam (for generalized additive models)
pip install pygam

# plotly (for interactive visualizations)
pip install plotly matplotlib seaborn

# All plotting tools
pip install plotly matplotlib seaborn adjustText
```

---

## Jupyter Setup

### Step 1: Install Jupyter (if not already installed)

```bash
pip install jupyter notebook ipykernel
```

### Step 2: Register Kernel with Jupyter

This allows Jupyter to use your virtual environment:

```bash
python -m ipykernel install --user --name=py-tidymodels --display-name="Python (py-tidymodels)"
```

### Step 3: Launch Jupyter

```bash
# From the project directory
jupyter notebook

# Or for JupyterLab
jupyter lab
```

### Step 4: Select the Kernel

When opening a notebook:
1. Click **Kernel** → **Change Kernel**
2. Select **Python (py-tidymodels)**

### Important Note on Code Changes

The package is installed in **editable mode** (`-e` flag), which means:
- ✅ Code changes are immediately available
- ⚠️ **You must restart the Jupyter kernel** after making changes to Python files
- ⚠️ Clear bytecode cache if changes don't appear (see [Troubleshooting](#troubleshooting))

---

## Verification

### Basic Verification

Test that core modules import correctly:

```bash
python << 'EOF'
# Test core imports
from py_hardhat import mold, forge
from py_parsnip import linear_reg, rand_forest, prophet_reg, arima_reg
from py_workflows import workflow
from py_recipes import recipe
from py_rsample import initial_split, time_series_cv
from py_yardstick import rmse, mae, r_squared, metric_set
from py_tune import tune_grid, grid_regular
from py_workflowsets import WorkflowSet

print("✅ All core modules imported successfully!")
print("\nInstalled packages:")
print(f"  • py-hardhat (data preprocessing)")
print(f"  • py-parsnip (23 model types)")
print(f"  • py-workflows (pipeline composition)")
print(f"  • py-recipes (78 preprocessing steps)")
print(f"  • py-rsample (cross-validation)")
print(f"  • py-yardstick (17 evaluation metrics)")
print(f"  • py-tune (hyperparameter tuning)")
print(f"  • py-workflowsets (multi-model comparison)")
EOF
```

### Run Tests

Verify everything works correctly:

```bash
# Run all tests (782+ tests)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_parsnip/test_linear_reg.py -v

# Run with coverage report
python -m pytest tests/ --cov=py_hardhat --cov=py_parsnip --cov=py_recipes --cov=py_yardstick --cov=py_tune --cov-report=html

# Get exact test count
python -m pytest tests/ --collect-only -q | tail -1
```

### Try an Example

```python
import pandas as pd
import numpy as np
from py_parsnip import linear_reg
from py_workflows import workflow

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'y': np.random.randn(100)
})

# Fit a simple model
wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
fit = wf.fit(data)

# Make predictions
predictions = fit.predict(data)
print("✅ Simple workflow executed successfully!")
print(predictions.head())
```

---

## Troubleshooting

### Problem: Import Errors After Code Changes

**Symptom:** Changes to Python files don't appear in Jupyter notebooks

**Solution:**
```bash
# 1. Clear Python bytecode cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 2. Reinstall package
pip install -e . --force-reinstall --no-deps

# 3. Restart Jupyter kernel: Kernel → Restart & Clear Output
```

### Problem: Module Not Found Errors

**Symptom:** `ModuleNotFoundError: No module named 'py_parsnip'`

**Solutions:**
```bash
# Check if package is installed
pip list | grep py-tidymodels

# If not installed, install it
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify you're in the virtual environment
which python  # Should show path to venv
```

### Problem: Prophet/ARIMA Installation Issues

**Prophet on macOS:**
```bash
# Install using conda (recommended)
conda install -c conda-forge prophet

# Or install dependencies first
brew install cmake
pip install prophet
```

**statsmodels compatibility:**
```bash
# If you get statsmodels errors
pip install statsmodels==0.14.5 --force-reinstall
```

### Problem: pmdarima numpy 2.x Incompatibility

**Symptom:** `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Solutions:**
```bash
# Option 1: Use statsmodels ARIMA instead (recommended)
# No code changes needed - just use default engine

# Option 2: Downgrade numpy (not recommended)
pip install 'numpy<2.0'

# Option 3: Skip pmdarima
pip install -r requirements.txt --no-deps
pip install $(grep -v pmdarima requirements.txt)
```

### Problem: Jupyter Kernel Not Available

**Symptom:** Kernel not showing in Jupyter notebook

**Solution:**
```bash
# Reinstall kernel
python -m ipykernel install --user --name=py-tidymodels --display-name="Python (py-tidymodels)" --force

# List available kernels
jupyter kernelspec list

# Remove old kernel if needed
jupyter kernelspec remove old-kernel-name
```

### Problem: Permission Errors on macOS/Linux

**Solution:**
```bash
# Don't use sudo with pip in virtual environments
# If you see permission errors, ensure you're in venv:
which python  # Should point to venv

# Deactivate and reactivate if needed
deactivate
source py-tidymodels-env/bin/activate
```

### Problem: Slow Test Execution

**Solution:**
```bash
# Run tests in parallel
pip install pytest-xdist
python -m pytest tests/ -n auto

# Run only fast tests
python -m pytest tests/ -m "not slow"

# Skip specific slow tests
python -m pytest tests/ --ignore=tests/test_tune/
```

---

## Optional Dependencies

### Visualization Tools

```bash
pip install plotly matplotlib seaborn adjustText
```

### Additional Model Engines

```bash
# Gradient boosting
pip install xgboost lightgbm catboost

# Recursive forecasting
pip install skforecast

# GAMs
pip install pygam

# MARS (may require compilation)
pip install scikit-learn-contrib/py-earth
# Or: conda install -c conda-forge py-earth
```

### Development Tools

```bash
# Linting and formatting
pip install black flake8 mypy

# Documentation
pip install sphinx sphinx-rtd-theme

# Profiling
pip install line-profiler memory-profiler
```

### Time Series Utilities

```bash
pip install pytimetk tsfeatures antropy supersmoother
```

---

## Updating the Package

### Pull Latest Changes

```bash
# Ensure you're on main branch
git checkout main

# Pull latest changes
git pull origin main

# Reinstall package (updates any changes)
pip install -e . --force-reinstall --no-deps

# Restart Jupyter kernel if using notebooks
```

### Update Dependencies

```bash
# Update all packages to latest compatible versions
pip install --upgrade -r requirements.txt

# Or update specific packages
pip install --upgrade pandas numpy scikit-learn
```

---

## Package Structure

After installation, you'll have access to:

```
py-tidymodels/
├── py_hardhat/          # Data preprocessing (mold/forge)
├── py_parsnip/          # 23 model types with 30+ engines
├── py_rsample/          # Cross-validation and resampling
├── py_workflows/        # Pipeline composition
├── py_recipes/          # 78 preprocessing steps
├── py_yardstick/        # 17 evaluation metrics
├── py_tune/             # Hyperparameter tuning
├── py_workflowsets/     # Multi-model comparison
├── py_visualize/        # Interactive visualizations (optional)
└── py_stacks/           # Model ensembling (optional)
```

---

## Getting Help

### Documentation
- **Main docs**: `CLAUDE.md` (comprehensive guide)
- **Examples**: `examples/` directory (21+ Jupyter notebooks)
- **API Reference**: Coming soon

### Support
- **Issues**: https://github.com/m-deane/py-tidymodels2/issues
- **Discussions**: https://github.com/m-deane/py-tidymodels2/discussions

### Example Notebooks

Located in `examples/` directory:
- `01_hardhat_demo.ipynb` - Data preprocessing
- `02_parsnip_demo.ipynb` - Model fitting basics
- `05_recipes_comprehensive_demo.ipynb` - Feature engineering
- `11_workflowsets_demo.ipynb` - Multi-model comparison
- `13_panel_models_demo.ipynb` - Grouped/panel modeling

---

## Next Steps

After installation:

1. **Explore Examples**: Run Jupyter notebooks in `examples/` directory
2. **Read Documentation**: Review `CLAUDE.md` for detailed API reference
3. **Run Tests**: Verify everything works with `pytest tests/`
4. **Start Building**: Create your first workflow!

```python
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

# Build a complete modeling pipeline
wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_impute_median(all_numeric_predictors())
        .step_normalize(all_numeric_predictors())
    )
    .add_model(linear_reg())
)

# Fit and predict
fit = wf.fit(train_data)
predictions = fit.predict(test_data)
```

Happy modeling! 🎉
