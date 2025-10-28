import pandas as pd
import numpy as np
from py_parsnip import rand_forest
from py_workflows import workflow
from py_rsample import time_series_cv
from py_tune import tune, tune_grid, grid_regular

# Create simple time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'time_index': np.arange(100),
    'value': np.arange(100) * 0.5 + np.random.randn(100) * 5 + 100
})

# Split data
train_data = data.iloc[:80]

# Create CV splits
cv_splits = time_series_cv(
    train_data,
    date_column='date',
    initial=40,
    assess=20,
    skip=10,
    cumulative=False
)

# Tune random forest
wf_tune_multi = (
    workflow()
    .add_formula("value ~ time_index")
    .add_model(
        rand_forest(
            trees=tune(),
            min_n=tune()
        ).set_mode('regression')
    )
)

# Create multi-parameter grid
grid_multi = grid_regular(
    {
        'trees': {'range': (50, 100)},
        'min_n': {'range': (2, 10)}
    },
    levels=2
)

print("Tuning random forest...")
results_multi = tune_grid(
    wf_tune_multi,
    resamples=cv_splits,
    grid=grid_multi
)

print("\n" + "="*60)
print("Inspecting results_multi.metrics:")
print("="*60)
print(f"Type: {type(results_multi.metrics)}")
print(f"Shape: {results_multi.metrics.shape}")
print(f"Columns: {results_multi.metrics.columns.tolist()}")
print("\nFirst 10 rows:")
print(results_multi.metrics.head(10))

print("\n" + "="*60)
print("Checking for 'metric' column:")
print("="*60)
print(f"'metric' in results_multi.metrics.columns: {'metric' in results_multi.metrics.columns}")
print(f"Has 'rmse' column: {'rmse' in results_multi.metrics.columns}")

print("\n" + "="*60)
print("Inspecting results_multi.grid:")
print("="*60)
print(f"Columns: {results_multi.grid.columns.tolist()}")
print(results_multi.grid)
