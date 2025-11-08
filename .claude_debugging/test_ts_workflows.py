"""
Test time series workflows to verify fit_raw parameter handling
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

from py_workflows import workflow
from py_parsnip import prophet_reg, arima_reg
from py_rsample import time_series_cv
from py_yardstick import metric_set, rmse, mae
from py_tune import fit_resamples

# Create simple time series data
dates = pd.date_range('2020-01-01', periods=50, freq='M')
df = pd.DataFrame({
    'date': dates,
    'target': [100 + i*2 + (i%12)*5 for i in range(50)],
    'x1': range(50)
})

# Test Prophet workflow
print("Testing Prophet workflow...")
wf_prophet = workflow().add_formula("target ~ date + x1").add_model(prophet_reg())

# Create time series CV
cv_folds = time_series_cv(df, date_column='date', initial='12 months', assess='3 months', skip='3 months')
print(f"Created {len(cv_folds)} CV folds")

# Fit resamples (should not error about original_training_data)
try:
    results = fit_resamples(wf_prophet, cv_folds, metrics=metric_set(rmse, mae))
    metrics = results.collect_metrics()
    print(f"✓ Prophet workflow succeeded: {len(metrics)} metrics collected")
except Exception as e:
    print(f"✗ Prophet workflow failed: {e}")
    sys.exit(1)

# Test ARIMA workflow
print("\nTesting ARIMA workflow...")
wf_arima = workflow().add_formula("target ~ date + x1").add_model(arima_reg(seasonal_period=12))

try:
    results = fit_resamples(wf_arima, cv_folds, metrics=metric_set(rmse, mae))
    metrics = results.collect_metrics()
    print(f"✓ ARIMA workflow succeeded: {len(metrics)} metrics collected")
except Exception as e:
    print(f"✗ ARIMA workflow failed: {e}")
    sys.exit(1)

print("\n✓ All time series workflows work correctly!")
