#!/usr/bin/env python3
"""
Minimal test case to reproduce plot_tune_results() failure.
Tests the bug where plot_tune_results() can't find tunable parameters.
"""

import pandas as pd
import numpy as np
from py_parsnip import linear_reg
from py_tune import tune, tune_grid
from py_rsample import vfold_cv
from py_visualize import plot_tune_results
from py_workflows import workflow

# Create minimal synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(15),
    'y': np.random.randn(15)
})

print("=" * 60)
print("REPRODUCING TUNING VISUALIZATION BUG")
print("=" * 60)

# Create workflow with tunable parameter
print("\n1. Creating workflow with penalty=tune()...")
wf = (
    workflow()
    .add_formula("y ~ x")
    .add_model(linear_reg(penalty=tune()))
)
print(f"   Workflow: {wf}")

# Create small parameter grid
print("\n2. Creating parameter grid...")
grid = pd.DataFrame({'penalty': [0.001, 0.01, 0.1]})
print(f"   Grid shape: {grid.shape}")
print(f"   Grid columns: {list(grid.columns)}")

# Create minimal resampling
print("\n3. Creating 2-fold cross-validation...")
resamples = vfold_cv(data, v=2)

# Run tune_grid
print("\n4. Running tune_grid()...")
results = tune_grid(wf, resamples, grid=grid)
print(f"   Results type: {type(results)}")

# Diagnostic prints
print("\n5. DIAGNOSTICS - Inspecting TuneResults:")
print(f"   - results.grid type: {type(results.grid)}")
print(f"   - results.grid:\n{results.grid}")
print(f"   - results.metrics shape: {results.metrics.shape}")
print(f"   - results.metrics columns: {list(results.metrics.columns)}")
print(f"   - results.metrics head:\n{results.metrics.head()}")

# Show the bug - parameters are missing from metrics
print("\n6. THE PROBLEM:")
print(f"   - results.metrics has columns: {list(results.metrics.columns)}")
print(f"   - results.metrics has 'metric' column: {'metric' in results.metrics.columns}")
print(f"   - This triggers LONG FORMAT code path (line 78-86)")
print(f"   - Line 86 looks for params in metric_data.columns")
print(f"   - But 'penalty' is NOT in metrics: {'penalty' in results.metrics.columns}")
print(f"   - 'penalty' is ONLY in grid: {'penalty' in results.grid.columns}")
print(f"")
print(f"   ROOT CAUSE: Long format code path doesn't merge with grid!")
print(f"   - It expects params to be in metrics DataFrame already")
print(f"   - But tune_grid() only stores params in grid, not metrics")

# Attempt to plot - this should fail
print("\n7. Attempting plot_tune_results()...")
try:
    plot_tune_results(results)
    print("   SUCCESS: Plot created without error!")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    print(f"\n   BUG CONFIRMED:")
    print(f"   - plot_tune_results() cannot find the tunable parameters")
    print(f"   - Parameter 'penalty' IS in results.grid: {'penalty' in results.grid.columns}")
    print(f"   - But plot function fails to detect it")

print("\n" + "=" * 60)
print("SUMMARY OF BUG")
print("=" * 60)
print("FILE: /Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_visualize/tuning.py")
print("FUNCTION: plot_tune_results()")
print("ISSUE: Lines 78-86 handle 'long format' data with 'metric' column")
print("       Line 86 tries to find params in metric_data.columns")
print("       But params are NOT in results.metrics - only in results.grid")
print("")
print("FIX NEEDED: Long format path must merge with results.grid")
print("           Similar to lines 107-128 in the wide format path")
print("=" * 60)
