#!/usr/bin/env python3
"""Minimal reproduction of plot_tune_results() bug - under 50 lines"""

import pandas as pd
import numpy as np
from py_parsnip import linear_reg
from py_tune import tune, tune_grid
from py_rsample import vfold_cv
from py_visualize import plot_tune_results
from py_workflows import workflow

# Create minimal synthetic data (15 rows)
np.random.seed(42)
data = pd.DataFrame({'x': np.random.randn(15), 'y': np.random.randn(15)})

# Create workflow with 1 tunable parameter
wf = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=tune()))

# Create small parameter grid (3 values)
grid = pd.DataFrame({'penalty': [0.001, 0.01, 0.1]})

# Create minimal resampling (2 folds)
resamples = vfold_cv(data, v=2)

# Run tune_grid
print("Running tune_grid()...")
results = tune_grid(wf, resamples, grid=grid)

# Show diagnostic info
print(f"\nTuneResults structure:")
print(f"  metrics columns: {list(results.metrics.columns)}")
print(f"  grid columns: {list(results.grid.columns)}")
print(f"  'penalty' in metrics: {'penalty' in results.metrics.columns}")
print(f"  'penalty' in grid: {'penalty' in results.grid.columns}")

# Attempt to plot - this will fail with "No tunable parameters found"
print(f"\nAttempting plot_tune_results()...")
try:
    plot_tune_results(results)
    print("SUCCESS!")
except ValueError as e:
    print(f"ERROR: {e}")
    print(f"\nBUG: plot_tune_results() can't find 'penalty' parameter")
    print(f"     because it only looks in metrics, not in grid")
