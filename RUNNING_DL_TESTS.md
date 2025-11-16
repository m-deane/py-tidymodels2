# Running Deep Learning Model Tests - Quick Start Guide

## Overview

47 comprehensive tests have been created for NHITS and NBEATS deep learning models:
- **test_nhits_reg.py**: 23 tests
- **test_nbeats_reg.py**: 24 tests

## Prerequisites

### 1. Activate Virtual Environment
```bash
cd /home/user/py-tidymodels2
source py-tidymodels2/bin/activate
```

### 2. Install NeuralForecast (Required)
```bash
pip install neuralforecast
```

This will automatically install:
- PyTorch (>= 1.13.0)
- Lightning (for training)
- Other dependencies

### 3. Verify Installation
```bash
python -c "import neuralforecast; print('✓ NeuralForecast installed')"
```

## Running Tests

### Option 1: Run All DL Tests (Recommended)
```bash
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -v
```

**Expected output:**
```
============================== test session starts ==============================
tests/test_parsnip/test_nhits_reg.py ....................... [ 48%]
tests/test_parsnip/test_nbeats_reg.py ........................ [100%]
============================== 47 passed in XX.XXs ==============================
```

### Option 2: Run NHITS Tests Only
```bash
pytest tests/test_parsnip/test_nhits_reg.py -v
```

### Option 3: Run NBEATS Tests Only
```bash
pytest tests/test_parsnip/test_nbeats_reg.py -v
```

### Option 4: Quick Test (No Verbose)
```bash
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -q
```

### Option 5: Run with Coverage
```bash
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py \
  --cov=py_parsnip.models.nhits_reg \
  --cov=py_parsnip.models.nbeats_reg \
  --cov=py_parsnip.engines.neuralforecast_nhits \
  --cov=py_parsnip.engines.neuralforecast_nbeats \
  --cov=py_parsnip.engines.base_dl_engine \
  --cov-report=term \
  --cov-report=html
```

Then view coverage report:
```bash
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## If NeuralForecast Not Installed

If you run tests without NeuralForecast, they will be automatically skipped:

```
tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec::test_default_spec
SKIPPED (NeuralForecast not installed. Install with: pip install neuralforecast)
```

All 47 tests will be skipped with helpful installation message.

## Test Execution Time

With `max_steps=50` (fast mode):
- NHITS tests: ~30-60 seconds
- NBEATS tests: ~30-60 seconds
- **Total: ~1-2 minutes** (CPU mode)

With GPU (if available):
- Tests run 2-5x faster on CUDA devices

## GPU Tests (Optional)

GPU tests are marked with `@pytest.mark.gpu` and skipped by default.

To run GPU tests (requires NVIDIA GPU + CUDA):
```bash
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -m gpu -v
```

To skip GPU tests explicitly:
```bash
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -m "not gpu" -v
```

## Running Specific Tests

### Run Single Test Class
```bash
# NHITS specification tests
pytest tests/test_parsnip/test_nhits_reg.py::TestNHITSSpec -v

# NBEATS decomposition tests
pytest tests/test_parsnip/test_nbeats_reg.py::TestNBEATSDecomposition -v
```

### Run Single Test
```bash
# Specific NHITS test
pytest tests/test_parsnip/test_nhits_reg.py::TestNHITSFit::test_fit_with_exogenous_variables -v

# Specific NBEATS test
pytest tests/test_parsnip/test_nbeats_reg.py::TestNBEATSSpec::test_stack_type_validation -v
```

## Troubleshooting

### Problem: Tests Skipped
```
SKIPPED (NeuralForecast not installed...)
```
**Solution:** Install NeuralForecast: `pip install neuralforecast`

### Problem: Import Errors
```
ImportError: cannot import name 'NHITS' from 'neuralforecast.models'
```
**Solution:** Update NeuralForecast: `pip install --upgrade neuralforecast`

### Problem: PyTorch Not Found
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Install PyTorch: `pip install torch>=1.13.0`

### Problem: Tests Very Slow
**Solution:** Tests use `max_steps=50` for speed. If still slow:
- Check if running on CPU vs GPU
- Reduce batch_size in fixtures
- Run tests in parallel: `pytest -n auto` (requires pytest-xdist)

## Test Coverage Summary

### NHITS Tests (23)
| Category | Tests | Coverage |
|----------|-------|----------|
| Specification | 7 | Model creation, parameters, validation |
| Fitting | 6 | Univariate, multivariate, validation splits |
| Prediction | 4 | Numeric, intervals, exogenous, forecasting |
| Extraction | 3 | Three-DataFrame output, metrics |
| Device | 3 | CPU, CUDA, auto-detection |
| Integration | 4 | Full workflows, different frequencies |

### NBEATS Tests (24)
| Category | Tests | Coverage |
|----------|-------|----------|
| Specification | 6 | Model creation, stack types, validation |
| Fitting | 6 | Univariate, stacks, validation splits |
| Prediction | 3 | Numeric, intervals, forecasting |
| Extraction | 4 | Three-DataFrame output, decomposition |
| Decomposition | 3 | Trend, seasonality, combined |
| Device | 3 | CPU, CUDA, auto-detection |
| Integration | 4 | Full workflows, comparisons |

## Next Steps After Testing

1. **Check test results:** All 47 tests should pass
2. **Review coverage report:** `htmlcov/index.html`
3. **Run all parsnip tests:** `pytest tests/test_parsnip/ -v`
4. **Update documentation:** Add NHITS/NBEATS to CLAUDE.md
5. **Create example notebooks:** Demonstrate real-world usage

## Documentation

For detailed information, see:
- `tests/test_parsnip/README_DL_TESTS.md` - Comprehensive testing guide
- `.claude_plans/DL_TESTS_SUMMARY.md` - Implementation details
- `py_parsnip/models/nhits_reg.py` - NHITS model documentation
- `py_parsnip/models/nbeats_reg.py` - NBEATS model documentation

## Quick Reference

```bash
# Full command sequence
cd /home/user/py-tidymodels2
source py-tidymodels2/bin/activate
pip install neuralforecast
pytest tests/test_parsnip/test_nhits_reg.py tests/test_parsnip/test_nbeats_reg.py -v

# Expected result: 47 passed
```

---

**Status:** ✅ Ready to run
**Requirements:** NeuralForecast, PyTorch
**Execution time:** ~1-2 minutes (CPU)
**Total tests:** 47
