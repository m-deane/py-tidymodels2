# Comprehensive Testing Guide for Examples 27-37

## Objective
Test all 11 new example notebooks (27-37) to verify:
- All imports work correctly
- All code cells execute without errors
- Data loading succeeds
- Model fitting completes
- Outputs are generated correctly
- Notebooks complete in reasonable time

## Quick Start - Batch Test All Notebooks

```bash
cd /home/user/py-tidymodels2

# Run all notebooks in sequence
for nb in examples/{27..37}_*.ipynb; do
    echo "Testing: $(basename $nb)"
    jupyter nbconvert --clear-output --inplace "$nb"
    jupyter nbconvert --to notebook --execute "$nb" \
      --output "/tmp/$(basename $nb)" \
      --ExecutePreprocessor.timeout=900 2>&1 | head -50
    echo "Status: $?"
    echo ""
done
```

## Individual Notebook Test Commands

### Example 27: Agent Complete Forecasting Pipeline
```bash
jupyter nbconvert --clear-output --inplace examples/27_agent_complete_forecasting_pipeline.ipynb
jupyter nbconvert --to notebook --execute examples/27_agent_complete_forecasting_pipeline.ipynb \
  --output /tmp/27_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-5 minutes | **Dataset**: european_gas_demand_weather_data.csv

### Example 28: WorkflowSet Nested Resamples CV
```bash
jupyter nbconvert --clear-output --inplace examples/28_workflowset_nested_resamples_cv.ipynb
jupyter nbconvert --to notebook --execute examples/28_workflowset_nested_resamples_cv.ipynb \
  --output /tmp/28_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 2-4 minutes | **Dataset**: jodi_refinery_production_data.csv

### Example 29: Hybrid Models Comprehensive
```bash
jupyter nbconvert --clear-output --inplace examples/29_hybrid_models_comprehensive.ipynb
jupyter nbconvert --to notebook --execute examples/29_hybrid_models_comprehensive.ipynb \
  --output /tmp/29_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-5 minutes | **Dataset**: all_commodities_futures_collection.csv

### Example 30: Manual Regression Comparison
```bash
jupyter nbconvert --clear-output --inplace examples/30_manual_regression_comparison.ipynb
jupyter nbconvert --to notebook --execute examples/30_manual_regression_comparison.ipynb \
  --output /tmp/30_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 2-3 minutes | **Dataset**: refinery_margins.csv

### Example 31: Per-Group Preprocessing
```bash
jupyter nbconvert --clear-output --inplace examples/31_per_group_preprocessing.ipynb
jupyter nbconvert --to notebook --execute examples/31_per_group_preprocessing.ipynb \
  --output /tmp/31_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-4 minutes | **Dataset**: european_gas_demand_weather_data.csv

### Example 32: New Baseline Models
```bash
jupyter nbconvert --clear-output --inplace examples/32_new_baseline_models.ipynb
jupyter nbconvert --to notebook --execute examples/32_new_baseline_models.ipynb \
  --output /tmp/32_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 2-3 minutes | **Dataset**: jodi_crude_production_data.csv

### Example 33: Recursive Multistep Forecasting
```bash
jupyter nbconvert --clear-output --inplace examples/33_recursive_multistep_forecasting.ipynb
jupyter nbconvert --to notebook --execute examples/33_recursive_multistep_forecasting.ipynb \
  --output /tmp/33_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-5 minutes | **Dataset**: european_gas_demand_weather_data.csv

### Example 34: Boosting Engines Comparison
```bash
jupyter nbconvert --clear-output --inplace examples/34_boosting_engines_comparison.ipynb
jupyter nbconvert --to notebook --execute examples/34_boosting_engines_comparison.ipynb \
  --output /tmp/34_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 2-4 minutes | **Dataset**: refinery_margins.csv
**Requires**: `pip install xgboost lightgbm catboost`

### Example 35: Hybrid Time Series Models
```bash
jupyter nbconvert --clear-output --inplace examples/35_hybrid_timeseries_models.ipynb
jupyter nbconvert --to notebook --execute examples/35_hybrid_timeseries_models.ipynb \
  --output /tmp/35_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 4-6 minutes | **Dataset**: european_gas_demand_weather_data.csv
**Requires**: `pip install prophet`

### Example 36: Multivariate VARMAX
```bash
jupyter nbconvert --clear-output --inplace examples/36_multivariate_varmax.ipynb
jupyter nbconvert --to notebook --execute examples/36_multivariate_varmax.ipynb \
  --output /tmp/36_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-5 minutes | **Dataset**: all_commodities_futures_collection.csv

### Example 37: Advanced sklearn Models
```bash
jupyter nbconvert --clear-output --inplace examples/37_advanced_sklearn_models.ipynb
jupyter nbconvert --to notebook --execute examples/37_advanced_sklearn_models.ipynb \
  --output /tmp/37_test.ipynb --ExecutePreprocessor.timeout=900
```
**Duration**: 3-5 minutes | **Dataset**: refinery_margins.csv

## Test Import Verification

Before running notebooks, test all imports:

```python
# Test core imports
from py_parsnip import (
    linear_reg, rand_forest, decision_tree, nearest_neighbor,
    svm_rbf, svm_linear, mlp, boost_tree,
    arima_reg, prophet_reg, arima_boost, prophet_boost,
    varmax_reg, recursive_reg, hybrid_model, manual_reg,
    null_model, naive_reg
)
from py_workflows import Workflow
from py_recipes import recipe, step_normalize, all_numeric_predictors
from py_rsample import initial_time_split, time_series_cv, time_series_nested_cv
from py_yardstick import rmse, mae, r_squared
from py_tune import tune_grid, metric_set
from py_workflowsets import WorkflowSet
from py_agent import ForecastAgent

print("✅ All imports successful!")
```

## Common Issues and Solutions

### Missing Packages
```bash
pip install xgboost lightgbm catboost prophet
```

### Data Files Not Found
```bash
# Verify data directory
ls -lh _md/__data/*.csv
```

### Timeout Errors
```bash
# Increase timeout to 30 minutes
--ExecutePreprocessor.timeout=1800
```

### LLM API Key (Example 27 only)
```bash
# Set API key or skip LLM cells
export ANTHROPIC_API_KEY=your-key
```

## Success Criteria

Each notebook should:
- ✅ Execute without errors
- ✅ Complete in < 15 minutes
- ✅ Generate expected outputs
- ✅ All imports work
- ✅ All data loads correctly
- ✅ Models fit successfully

## Expected Total Testing Time

- **All 11 notebooks**: 35-55 minutes
- **Average per notebook**: 3-5 minutes
- **Longest**: Example 35 (~6 minutes)
- **Shortest**: Example 30 (~2 minutes)

## Verification Checklist

After testing, verify:
```bash
# Check all output files created
ls -lh /tmp/{27..37}_test.ipynb

# Count successful tests
grep -c "SUCCESS" /tmp/notebook_test_results.txt

# Check for errors
grep -i "error\|exception" /tmp/*_test_log.txt
```
