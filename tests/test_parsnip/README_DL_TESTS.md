# Deep Learning Model Tests - NHITS & NBEATS

This directory contains comprehensive test suites for the deep learning time series models:
- **NHITS** (Neural Hierarchical Interpolation for Time Series)
- **NBEATS** (Neural Basis Expansion Analysis for Time Series)

## Test Files

### 1. `test_nhits_reg.py` (23 tests)
Tests for NHITS model covering:
- Model specification and parameter validation
- Univariate and multivariate fitting
- Predictions with confidence intervals
- Device management (CPU, CUDA, MPS, auto)
- Output extraction (three-DataFrame pattern)
- Integration workflows
- Error handling

### 2. `test_nbeats_reg.py` (24 tests)
Tests for NBEATS model covering:
- Model specification and stack type validation
- Univariate fitting (NBEATS does NOT support exogenous variables)
- Decomposition (trend, seasonality, generic stacks)
- Predictions with confidence intervals
- Device management
- Output extraction
- Integration workflows
- Comparison with NHITS

## Installation Requirements

### Core Dependencies
```bash
# Activate virtual environment
source py-tidymodels2/bin/activate

# Install NeuralForecast (required)
pip install neuralforecast

# Install PyTorch (required by NeuralForecast)
pip install torch>=1.13.0
```

### GPU Support (Optional)

**NVIDIA GPU (CUDA):**
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (M1/M2):**
```bash
# PyTorch with MPS support (built-in for PyTorch >= 1.12)
pip install torch>=1.12.0
```

## Running Tests

### Run All Deep Learning Tests
```bash
# From project root
cd /home/user/py-tidymodels2

# Activate virtual environment
source py-tidymodels2/bin/activate

# Run both NHITS and NBEATS tests
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -v
```

### Run NHITS Tests Only
```bash
pytest tests/test_parsnip/test_nhits_reg.py -v
```

### Run NBEATS Tests Only
```bash
pytest tests/test_parsnip/test_nbeats_reg.py -v
```

### Run Specific Test Class
```bash
# Example: Run only NHITS specification tests
pytest tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec -v

# Example: Run only NBEATS decomposition tests
pytest tests/test_parsnip/test_nbeats_reg.py::TestNBEATSDecomposition -v
```

### Run Specific Test
```bash
# Example: Run single test
pytest tests/test_parsnip/test_nhits_reg.py::TestNHITSFit::test_fit_with_exogenous_variables -v
```

### Skip GPU Tests (Default)
GPU tests are marked with `@pytest.mark.gpu` and can be skipped:
```bash
# Skip GPU tests explicitly
pytest tests/test_parsnip/test_nhits_reg.py -m "not gpu" -v
```

### Run GPU Tests Only (Requires GPU)
```bash
# Run only GPU tests
pytest tests/test_parsnip/test_nhits_reg.py -m gpu -v
pytest tests/test_parsnip/test_nbeats_reg.py -m gpu -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py \
  --cov=py_parsnip.models.nhits_reg \
  --cov=py_parsnip.models.nbeats_reg \
  --cov=py_parsnip.engines.neuralforecast_nhits \
  --cov=py_parsnip.engines.neuralforecast_nbeats \
  --cov=py_parsnip.engines.base_dl_engine \
  --cov-report=html \
  --cov-report=term
```

### Run in Quiet Mode
```bash
# Less verbose output
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -q
```

### Run with Warnings
```bash
# Show all warnings (useful for debugging)
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -v -W all
```

## Test Behavior When NeuralForecast Not Installed

If NeuralForecast is not installed, all tests will be **automatically skipped** with a helpful message:

```
tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec::test_default_spec
SKIPPED (NeuralForecast not installed. Install with: pip install neuralforecast)
```

To run the tests, install NeuralForecast:
```bash
pip install neuralforecast
```

## Test Structure

Both test files follow the same organizational pattern:

### Test Classes

1. **TestSpec** - Model specification creation and validation
   - Default parameters
   - Custom parameters
   - Parameter validation
   - Error handling

2. **TestFit** - Model fitting
   - Univariate data
   - Multivariate data (NHITS only)
   - Insufficient data handling
   - Validation splits
   - Early stopping

3. **TestPredict** - Predictions
   - Numeric predictions
   - Confidence intervals
   - Future forecasting
   - Exogenous variables (NHITS only)

4. **TestExtract** - Output extraction
   - Three-DataFrame pattern
   - Outputs DataFrame (actuals, fitted, residuals)
   - Coefficients DataFrame (hyperparameters)
   - Stats DataFrame (metrics)

5. **TestDevice** - Device management
   - Auto-detection
   - CPU
   - CUDA (GPU tests)
   - MPS (Apple Silicon)

6. **TestDecomposition** (NBEATS only)
   - Trend stack
   - Seasonality stack
   - Generic stack
   - Combined decomposition

7. **TestIntegration** - End-to-end workflows
   - Complete fit → predict → extract workflows
   - Different time series frequencies
   - Long-horizon forecasting
   - Model comparisons

## Key Test Features

### Fixtures
- **daily_univariate_data**: 300-day univariate time series with trend + seasonality
- **daily_multivariate_data**: 300-day multivariate time series with exogenous variables
- **fitted_nhits/fitted_nbeats**: Pre-fitted models for prediction tests

### Synthetic Data Generation
All tests use synthetic data with known patterns:
- Linear trends
- Weekly seasonality (7-day period)
- Gaussian noise
- Exogenous effects (price, promo)

This ensures tests are:
- Fast (no real data loading)
- Reproducible (fixed random seeds)
- Interpretable (known ground truth)

### CPU-Only Testing
All tests force `device='cpu'` to ensure:
- CI/CD compatibility
- Fast execution
- No GPU requirement

GPU tests are separate and marked with `@pytest.mark.gpu`.

## Expected Test Results

### NHITS Tests
```
tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec::test_default_spec PASSED
tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec::test_spec_with_custom_horizon PASSED
...
23 passed in XX.XXs
```

### NBEATS Tests
```
tests/test_parsnip/test_nbeats_reg.py::TestNBEATSSpec::test_default_spec PASSED
tests/test_parsnip/test_nbeats_reg.py::TestNBEATSSpec::test_spec_interpretable_stacks PASSED
...
24 passed in XX.XXs
```

### Combined
```
============================== test session starts ==============================
tests/test_parsnip/test_nhits_reg.py ....................... [ 48%]
tests/test_parsnip/test_nbeats_reg.py ........................ [100%]
============================== 47 passed in XX.XXs ==============================
```

## Troubleshooting

### Tests Skipped
**Problem:** All tests skipped with "NeuralForecast not installed"

**Solution:**
```bash
pip install neuralforecast
```

### Import Errors
**Problem:** `ImportError: cannot import name 'NHITS' from 'neuralforecast.models'`

**Solution:** Update NeuralForecast:
```bash
pip install --upgrade neuralforecast
```

### PyTorch Not Found
**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch:
```bash
pip install torch>=1.13.0
```

### GPU Tests Failing
**Problem:** GPU tests fail with CUDA errors

**Solution:** Either:
1. Skip GPU tests: `pytest -m "not gpu"`
2. Install CUDA-enabled PyTorch
3. Use CPU tests only

### Slow Tests
**Problem:** Tests take very long to run

**Solution:** Tests use `max_steps=50` for speed. If still slow:
- Reduce to `max_steps=20` in fixtures
- Run tests in parallel: `pytest -n auto` (requires pytest-xdist)

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Deep Learning Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install neuralforecast torch
        pip install pytest pytest-cov

    - name: Run DL tests
      run: |
        pytest tests/test_parsnip/test_nhits_reg.py \
               tests/test_parsnip/test_nbeats_reg.py \
               -v --cov --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Coverage Summary

### NHITS Tests (23 tests)
- ✅ Specification validation (7 tests)
- ✅ Fitting (6 tests)
- ✅ Prediction (4 tests)
- ✅ Output extraction (3 tests)
- ✅ Device management (3 tests)
- ✅ Integration (4 tests)

### NBEATS Tests (24 tests)
- ✅ Specification validation (6 tests)
- ✅ Fitting (6 tests)
- ✅ Prediction (3 tests)
- ✅ Output extraction (4 tests)
- ✅ Decomposition (3 tests)
- ✅ Device management (3 tests)
- ✅ Integration (4 tests)

**Total: 47 tests covering all major functionality**

## Next Steps

After running these tests successfully:

1. **Add to main test suite:**
   ```bash
   # Run all parsnip tests including DL models
   pytest tests/test_parsnip/ -v
   ```

2. **Check coverage:**
   ```bash
   pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py \
     --cov --cov-report=html
   # View coverage report at htmlcov/index.html
   ```

3. **Update project documentation:**
   - Add NHITS and NBEATS to model list
   - Document deep learning model requirements
   - Update CLAUDE.md with DL model info

4. **Create example notebooks:**
   - `examples/XX_nhits_demo.ipynb`
   - `examples/XX_nbeats_demo.ipynb`
   - Show real-world forecasting examples

## Contact & Support

For issues or questions about these tests:
- Check error messages carefully
- Verify NeuralForecast installation
- Review test output with `-v` flag
- Check GitHub issues for known problems
