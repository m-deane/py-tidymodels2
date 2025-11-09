# Complete py_tune Hyperparameter Tuning Reference

**Version:** py-tidymodels v1.0
**Module:** `py_tune`
**Last Updated:** 2025-11-09

---

## Table of Contents

1. [Overview](#overview)
2. [Core Classes and Functions](#core-classes-and-functions)
3. [Parameter Marking](#parameter-marking)
4. [Grid Generation](#grid-generation)
5. [Tuning Functions](#tuning-functions)
6. [Results Analysis](#results-analysis)
7. [Complete Workflow Examples](#complete-workflow-examples)
8. [Best Practices](#best-practices)

---

## Overview

The `py_tune` module provides tidymodels-style hyperparameter optimization with grid search and cross-validation evaluation. It enables:

- **Parameter marking** with `tune()` function
- **Grid generation** (regular and random)
- **Grid search** with cross-validation
- **Model evaluation** without tuning
- **Results analysis** with multiple selection strategies

**Key Design Philosophy:**
- Declarative parameter marking
- Flexible grid generation
- Comprehensive result analysis
- Integration with py_workflows and py_rsample

---

## Core Classes and Functions

### Quick Reference

```python
# Imports
from py_tune import (
    tune,                    # Mark parameters for tuning
    TuneParameter,          # Parameter marker class
    grid_regular,           # Regular grid generation
    grid_random,            # Random grid generation
    tune_grid,              # Grid search with CV
    fit_resamples,          # Evaluate without tuning
    finalize_workflow,      # Apply best parameters
    TuneResults            # Results container class
)
```

---

## Parameter Marking

### `tune()` Function

Mark a model parameter for hyperparameter tuning.

**Signature:**
```python
def tune(id: Optional[str] = None) -> TuneParameter
```

**Parameters:**
- `id` (Optional[str], default=None): Optional identifier for the parameter
  - If None, auto-generates unique ID like `"tune_140234567890"`
  - If provided, uses custom ID (useful for clarity and debugging)

**Returns:**
- `TuneParameter`: Marker object indicating parameter should be tuned

**Examples:**

```python
from py_parsnip import linear_reg
from py_tune import tune

# Basic usage - auto-generated IDs
spec = linear_reg(penalty=tune(), mixture=tune())

# With custom IDs (recommended for clarity)
spec = linear_reg(
    penalty=tune(id='penalty'),
    mixture=tune(id='mixture')
)

# Multiple model types
from py_parsnip import rand_forest, boost_tree

# Random forest tuning
rf_spec = rand_forest(
    trees=tune(id='trees'),
    tree_depth=tune(id='depth'),
    min_n=tune(id='min_samples')
)

# Boosting tuning
boost_spec = boost_tree(
    trees=tune(id='trees'),
    tree_depth=tune(id='depth'),
    learn_rate=tune(id='learning_rate')
)
```

### `TuneParameter` Class

**Class Definition:**
```python
class TuneParameter:
    """Marker for parameters to be tuned."""

    def __init__(self, id: Optional[str] = None):
        if id is None:
            self.id = f"tune_{id(self)}"
        else:
            self.id = id

    def __repr__(self):
        return f"tune(id='{self.id}')"
```

**Attributes:**
- `id` (str): Unique identifier for the parameter

**Notes:**
- Each `tune()` call without ID creates unique marker
- Multiple `tune()` calls have different IDs
- Use explicit IDs for better debugging

---

## Grid Generation

### `grid_regular()` - Regular Grid

Create an evenly-spaced grid of parameter combinations.

**Signature:**
```python
def grid_regular(
    param_info: Dict[str, Dict[str, Any]],
    levels: int = 3
) -> pd.DataFrame
```

**Parameters:**

- `param_info` (Dict[str, Dict[str, Any]], required): Parameter specifications

  **For each parameter, specify either:**

  **A) Range-based specification:**
  ```python
  {
      'parameter_name': {
          'range': (min_value, max_value),  # Required: tuple of (min, max)
          'trans': 'identity' or 'log',      # Optional: transformation
          'type': 'auto', 'int', or 'float'  # Optional: value type
      }
  }
  ```

  **B) Explicit values:**
  ```python
  {
      'parameter_name': {
          'values': [val1, val2, val3, ...]  # Explicit list of values
      }
  }
  ```

- `levels` (int, default=3): Number of levels for each parameter
  - Ignored if 'values' is specified
  - Total grid size = levels^(number of parameters)

**Parameter Specification Keys:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `range` | tuple | Yes (if no values) | (min, max) bounds for parameter |
| `values` | list | Yes (if no range) | Explicit parameter values |
| `trans` | str | No (default='identity') | Transformation: 'identity' or 'log' |
| `type` | str | No (default='auto') | Type conversion: 'auto', 'int', or 'float' |

**Type Handling:**
- `'auto'`: Auto-detects integers for common parameters:
  - `trees`, `tree_depth`, `min_n`, `stop_iter`, `mtry`, `neighbors`, `epochs`
- `'int'`: Forces integer conversion with rounding
- `'float'`: Keeps as floating point

**Transformation Details:**
- `'identity'`: Linear spacing using `np.linspace(min, max, levels)`
- `'log'`: Log spacing using `np.logspace(log10(min), log10(max), levels)`
  - Ideal for penalty, learn_rate, and other exponential-scale parameters

**Returns:**
- `pd.DataFrame`: Grid with columns for each parameter plus `.config` column
  - Config names: `"config_001"`, `"config_002"`, etc.

**Examples:**

```python
from py_tune import grid_regular

# Example 1: Basic linear spacing
param_info = {
    'penalty': {'range': (0.001, 1.0)},
    'mixture': {'range': (0, 1)}
}
grid = grid_regular(param_info, levels=3)
# Output: 9 rows (3 x 3)
#    penalty  mixture  .config
# 0  0.001    0.0      config_001
# 1  0.001    0.5      config_002
# 2  0.001    1.0      config_003
# 3  0.5005   0.0      config_004
# ...

# Example 2: Log transformation for penalty-like parameters
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'range': (0, 1)}
}
grid = grid_regular(param_info, levels=3)
# Penalty values: [0.001, 0.0316, 1.0] (log-spaced)

# Example 3: Integer parameters
param_info = {
    'trees': {'range': (50, 200), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'}
}
grid = grid_regular(param_info, levels=4)
# Output: 16 rows (4 x 4) with integer values

# Example 4: Explicit values
param_info = {
    'mtry': {'values': [2, 4, 6, 8]},
    'trees': {'values': [100, 500, 1000]}
}
grid = grid_regular(param_info)
# Output: 12 rows (4 x 3), levels parameter ignored

# Example 5: Mixed specification
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'values': [0, 0.5, 1.0]},
    'trees': {'range': (100, 1000), 'type': 'int'}
}
grid = grid_regular(param_info, levels=3)
# Output: 27 rows (3 penalty × 3 mixture × 3 trees)

# Example 6: Single parameter tuning
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
}
grid = grid_regular(param_info, levels=10)
# Output: 10 rows with single parameter varied
```

**Common Parameter Patterns:**

```python
# Regularization parameters (log scale)
{
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'range': (0, 1)}  # elastic net mixing
}

# Tree-based models (integers)
{
    'trees': {'range': (50, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 15), 'type': 'int'},
    'min_n': {'range': (2, 20), 'type': 'int'}
}

# Neural networks (log scale for learning rate)
{
    'epochs': {'range': (10, 100), 'type': 'int'},
    'learn_rate': {'range': (0.0001, 0.1), 'trans': 'log'},
    'hidden_units': {'values': [32, 64, 128, 256]}
}

# Boosting models (mixed types)
{
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'},
    'learn_rate': {'range': (0.001, 0.3), 'trans': 'log'},
    'min_n': {'range': (5, 30), 'type': 'int'}
}
```

---

### `grid_random()` - Random Grid

Create a random grid of parameter combinations.

**Signature:**
```python
def grid_random(
    param_info: Dict[str, Dict[str, Any]],
    size: int = 10,
    seed: Optional[int] = None
) -> pd.DataFrame
```

**Parameters:**

- `param_info` (Dict[str, Dict[str, Any]], required): Parameter specifications

  **Structure (similar to grid_regular but requires 'range'):**
  ```python
  {
      'parameter_name': {
          'range': (min_value, max_value),   # Required for random sampling
          'trans': 'identity' or 'log',       # Optional: sampling distribution
          'type': 'float' or 'int'            # Optional: type conversion
      }
  }
  ```

- `size` (int, default=10): Number of random parameter combinations to generate
  - Unlike grid_regular, total combinations = size (not exponential)

- `seed` (Optional[int], default=None): Random seed for reproducibility
  - If None, results differ each run
  - If set, same grid generated for same seed

**Sampling Distributions:**
- `trans='identity'`: Uniform sampling using `np.random.uniform(min, max, size)`
- `trans='log'`: Log-uniform sampling using `np.exp(np.random.uniform(log(min), log(max), size))`
  - Ideal for parameters with exponential scale (penalty, learn_rate)

**Type Handling:**
- `'float'` (default): Keep as floating point
- `'int'`: Round to nearest integer

**Returns:**
- `pd.DataFrame`: Grid with columns for each parameter plus `.config` column

**Examples:**

```python
from py_tune import grid_random

# Example 1: Basic random sampling
param_info = {
    'penalty': {'range': (0.001, 1.0)},
    'mixture': {'range': (0, 1)}
}
grid = grid_random(param_info, size=20, seed=42)
# Output: 20 rows with random uniform samples

# Example 2: Log-uniform sampling (better for wide ranges)
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
}
grid = grid_random(param_info, size=50, seed=42)
# More samples near lower values (0.001) than upper (1.0)

# Example 3: Integer parameters
param_info = {
    'trees': {'range': (10, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 20), 'type': 'int'}
}
grid = grid_random(param_info, size=30, seed=42)
# All values rounded to integers

# Example 4: Mixed types (neural network)
param_info = {
    'hidden_units': {'range': (32, 512), 'type': 'int'},
    'learn_rate': {'range': (0.0001, 0.1), 'trans': 'log'},
    'dropout': {'range': (0, 0.5)}
}
grid = grid_random(param_info, size=25, seed=123)

# Example 5: Reproducibility
grid1 = grid_random(param_info, size=10, seed=42)
grid2 = grid_random(param_info, size=10, seed=42)
assert grid1.equals(grid2)  # Identical grids

# Example 6: Large random search (efficient for many parameters)
param_info = {
    'param1': {'range': (0, 1)},
    'param2': {'range': (0, 100)},
    'param3': {'range': (0.001, 10), 'trans': 'log'},
    'param4': {'range': (5, 50), 'type': 'int'}
}
# Regular grid with 4 params × 3 levels = 81 combinations
# Random grid can sample just 30 combinations
grid = grid_random(param_info, size=30, seed=42)
```

**When to Use Random vs Regular Grid:**

| Situation | Recommended Grid | Reason |
|-----------|------------------|--------|
| 1-2 parameters | Regular | Full coverage with few combinations |
| 3+ parameters | Random | Exponential explosion of regular grid |
| Wide parameter ranges | Random with log | Better sampling across scales |
| Integer parameters | Regular | Discrete values benefit from full coverage |
| Limited compute budget | Random | Control exact number of fits |
| Initial exploration | Random (large) | Quick broad search |
| Fine-tuning | Regular (narrow range) | Thorough local search |

---

## Tuning Functions

### `tune_grid()` - Grid Search with Cross-Validation

Perform hyperparameter tuning via grid search with cross-validation.

**Signature:**
```python
def tune_grid(
    workflow,
    resamples,
    grid: Optional[Union[int, pd.DataFrame]] = None,
    metrics = None,
    param_info: Optional[Dict[str, Dict[str, Any]]] = None,
    control: Optional[Dict[str, Any]] = None
) -> TuneResults
```

**Parameters:**

- `workflow` (Workflow, required): Workflow object with `tune()` parameter markers
  - Must have parameters marked with `tune()` in model spec
  - Example: `workflow().add_formula("y ~ x").add_model(linear_reg(penalty=tune()))`

- `resamples` (resampling object, required): Cross-validation folds from `py_rsample`
  - From `vfold_cv()`, `time_series_cv()`, or other resampling functions
  - Each fold used to evaluate each parameter combination

- `grid` (Optional[Union[int, pd.DataFrame]], default=None): Grid specification
  - **If int**: Number of levels for `grid_regular()` (requires `param_info`)
  - **If DataFrame**: Pre-generated grid (from `grid_regular()` or `grid_random()`)
  - **If None**: Uses 3 levels with `grid_regular()` (requires `param_info`)

- `metrics` (optional, default=None): Metric set or list of metric functions
  - From `py_yardstick`: `metric_set(rmse, mae, r_squared)`
  - Or list: `[rmse, mae, r_squared]`
  - If None, uses default: `metric_set(rmse, mae, r_squared)` for regression

- `param_info` (Optional[Dict[str, Dict[str, Any]]], default=None): Parameter specifications
  - **Required when** `grid` is int or None
  - Same format as `grid_regular()` and `grid_random()`
  - Not needed if `grid` is pre-generated DataFrame

- `control` (Optional[Dict[str, Any]], default=None): Control parameters
  - **Keys:**
    - `'save_pred'` (bool, default=False): Whether to save predictions for each fold
    - Future: `'verbose'`, `'allow_par'`, etc.

**Returns:**
- `TuneResults`: Object containing metrics, predictions, and grid information
  - See [TuneResults Class](#tuneresults-class) for details

**Process Flow:**
1. Generate or validate parameter grid
2. For each parameter combination:
   - Update workflow with current parameters
   - For each CV fold:
     - Fit workflow on training data
     - Predict on test data
     - Calculate metrics
3. Combine all metrics and predictions
4. Return TuneResults object

**Examples:**

```python
from py_tune import tune, tune_grid, grid_regular, grid_random
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared

# Example 1: Basic grid search with integer levels
spec = linear_reg(penalty=tune(), mixture=tune())
wf = workflow().add_formula("y ~ .").add_model(spec)

param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'range': (0, 1)}
}

folds = vfold_cv(train_data, v=5)
results = tune_grid(wf, folds, grid=3, param_info=param_info)
# Grid: 3 × 3 = 9 configurations
# Total fits: 9 configs × 5 folds = 45 models

# Example 2: Pre-generated regular grid
grid_df = grid_regular(param_info, levels=5)
results = tune_grid(wf, folds, grid=grid_df)
# Grid: 5 × 5 = 25 configurations
# Total fits: 25 configs × 5 folds = 125 models

# Example 3: Random grid for efficiency
grid_df = grid_random(param_info, size=20, seed=42)
results = tune_grid(wf, folds, grid=grid_df)
# Grid: 20 random configurations
# Total fits: 20 configs × 5 folds = 100 models

# Example 4: Custom metrics
my_metrics = metric_set(rmse, mae, r_squared)
results = tune_grid(wf, folds, grid=3, param_info=param_info, metrics=my_metrics)

# Example 5: Save predictions
control = {'save_pred': True}
results = tune_grid(wf, folds, grid=3, param_info=param_info, control=control)
# Access predictions: results.collect_predictions()

# Example 6: Boosting model tuning
from py_parsnip import boost_tree

spec = boost_tree(
    trees=tune(),
    tree_depth=tune(),
    learn_rate=tune()
).set_engine('xgboost')

param_info = {
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'},
    'learn_rate': {'range': (0.001, 0.3), 'trans': 'log'}
}

wf = workflow().add_formula("target ~ .").add_model(spec)
results = tune_grid(wf, folds, grid=4, param_info=param_info)
# Grid: 4 × 4 × 4 = 64 configurations (full factorial)

# Example 7: Large parameter space with random grid
grid_df = grid_random(param_info, size=30, seed=42)
results = tune_grid(wf, folds, grid=grid_df)
# Only 30 configurations instead of 64 (more efficient)

# Example 8: Time series cross-validation
from py_rsample import time_series_cv

ts_folds = time_series_cv(
    data,
    date_col='date',
    initial='2 years',
    assess='3 months',
    skip='1 month',
    cumulative=True
)

spec = linear_reg(penalty=tune())
wf = workflow().add_formula("sales ~ .").add_model(spec)

param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}
results = tune_grid(wf, ts_folds, grid=10, param_info=param_info)
```

**Performance Considerations:**

Total number of model fits = `(grid size) × (number of folds)`

| Grid Size | Folds | Total Fits | Estimated Time* |
|-----------|-------|------------|-----------------|
| 9 (3×3) | 5 | 45 | 1-5 minutes |
| 25 (5×5) | 5 | 125 | 3-15 minutes |
| 64 (4×4×4) | 5 | 320 | 10-40 minutes |
| 100 (random) | 10 | 1000 | 30-120 minutes |

*Depends on model complexity, data size, and hardware

---

### `fit_resamples()` - Evaluate Without Tuning

Evaluate a single workflow configuration across cross-validation folds without hyperparameter tuning.

**Signature:**
```python
def fit_resamples(
    workflow,
    resamples,
    metrics = None,
    control: Optional[Dict[str, Any]] = None
) -> TuneResults
```

**Parameters:**

- `workflow` (Workflow, required): Workflow with fixed parameters
  - Should NOT have `tune()` markers (or they'll be ignored)
  - Evaluates the workflow as-is across folds

- `resamples` (resampling object, required): Cross-validation folds

- `metrics` (optional, default=None): Metric set or list of metrics
  - If None, uses default regression metrics

- `control` (Optional[Dict[str, Any]], default=None): Control parameters
  - `'save_pred'` (bool): Save predictions from each fold

**Returns:**
- `TuneResults`: Object with metrics across folds (single configuration)

**Use Cases:**
- Estimate model performance without tuning
- Compare different model types at default parameters
- Baseline performance assessment
- Quick CV evaluation

**Examples:**

```python
from py_tune import fit_resamples
from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae

# Example 1: Basic evaluation
wf = workflow().add_formula("y ~ .").add_model(linear_reg())
folds = vfold_cv(data, v=5)
results = fit_resamples(wf, folds)

# Check performance
metrics_df = results.collect_metrics()
print(metrics_df.groupby('metric')['value'].mean())

# Example 2: Compare multiple models
models = {
    'linear': linear_reg(),
    'ridge': linear_reg(penalty=0.1, mixture=0),
    'lasso': linear_reg(penalty=0.1, mixture=1),
    'rf': rand_forest(trees=100).set_mode('regression')
}

results = {}
for name, model in models.items():
    wf = workflow().add_formula("y ~ .").add_model(model)
    results[name] = fit_resamples(wf, folds)

# Compare
for name, res in results.items():
    metrics = res.collect_metrics()
    mean_rmse = metrics[metrics['metric'] == 'rmse']['value'].mean()
    print(f"{name}: RMSE = {mean_rmse:.4f}")

# Example 3: Custom metrics
my_metrics = metric_set(rmse, mae)
results = fit_resamples(wf, folds, metrics=my_metrics)

# Example 4: Save predictions for analysis
control = {'save_pred': True}
results = fit_resamples(wf, folds, control=control)
preds = results.collect_predictions()

# Analyze prediction errors
import matplotlib.pyplot as plt
plt.scatter(preds['.pred'], preds['y'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
```

---

### `finalize_workflow()` - Apply Best Parameters

Create a finalized workflow with the best hyperparameters.

**Signature:**
```python
def finalize_workflow(
    workflow,
    best_params: Dict[str, Any]
) -> Workflow
```

**Parameters:**

- `workflow` (Workflow, required): Original workflow with `tune()` markers

- `best_params` (Dict[str, Any], required): Dictionary of best parameter values
  - Usually from `TuneResults.select_best()` or `select_by_one_std_err()`
  - Keys must match tuned parameter names
  - Values replace `tune()` markers

**Returns:**
- `Workflow`: New workflow with parameters finalized (ready for fitting)

**Process:**
1. Takes original workflow (immutable)
2. Replaces `tune()` markers with actual values from `best_params`
3. Returns new workflow (original unchanged)

**Examples:**

```python
from py_tune import tune, tune_grid, finalize_workflow

# Example 1: Basic finalization
spec = linear_reg(penalty=tune(), mixture=tune())
wf = workflow().add_formula("y ~ .").add_model(spec)

# Tune
results = tune_grid(wf, folds, grid=grid)

# Get best parameters
best = results.select_best('rmse', maximize=False)
# best = {'penalty': 0.0316, 'mixture': 0.5}

# Finalize
final_wf = finalize_workflow(wf, best)

# Fit on full training data
final_fit = final_wf.fit(train_data)

# Predict on test data
test_preds = final_fit.predict(test_data)

# Example 2: One-standard-error rule (simpler model)
best_simple = results.select_by_one_std_err('rmse', maximize=False)
final_wf_simple = finalize_workflow(wf, best_simple)

# Example 3: Manual parameter selection
custom_params = {'penalty': 0.1, 'mixture': 0.8}
final_wf_custom = finalize_workflow(wf, custom_params)

# Example 4: Full tuning workflow
from py_parsnip import boost_tree
from py_rsample import vfold_cv

# Step 1: Define tunable model
spec = boost_tree(
    trees=tune(),
    tree_depth=tune(),
    learn_rate=tune()
).set_engine('xgboost')

wf = workflow().add_formula("target ~ .").add_model(spec)

# Step 2: Create grid
param_info = {
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'},
    'learn_rate': {'range': (0.001, 0.3), 'trans': 'log'}
}
grid = grid_random(param_info, size=30, seed=42)

# Step 3: Tune
folds = vfold_cv(train_data, v=5)
results = tune_grid(wf, folds, grid=grid)

# Step 4: Select best
best = results.select_best('rmse', maximize=False)

# Step 5: Finalize
final_wf = finalize_workflow(wf, best)

# Step 6: Fit and evaluate
final_fit = final_wf.fit(train_data)
test_preds = final_fit.predict(test_data)

# Step 7: Calculate test performance
from py_yardstick import rmse, mae
test_rmse = rmse(test_data['target'], test_preds['.pred'])
test_mae = mae(test_data['target'], test_preds['.pred'])
```

---

## Results Analysis

### `TuneResults` Class

Container class for hyperparameter tuning results.

**Class Definition:**
```python
@dataclass
class TuneResults:
    """Results from hyperparameter tuning."""

    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    workflow: Any = None
    resamples: Any = None
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
```

**Attributes:**

- `metrics` (pd.DataFrame): Metrics for each configuration and fold
  - Columns: `.config`, `.resample`, `metric`, `value` (long format)
  - Or: `.config`, `.resample`, metric columns (wide format)

- `predictions` (pd.DataFrame): Predictions from each configuration/fold
  - Only populated if `control={'save_pred': True}`
  - Columns: `.config`, `.resample`, `.row`, `.pred`

- `workflow` (Workflow): Original workflow object

- `resamples` (resampling object): Cross-validation folds used

- `grid` (pd.DataFrame): Parameter grid used
  - Columns: parameter names + `.config`

**Methods:**

---

### `collect_metrics()` - Get All Metrics

Retrieve all metrics from tuning.

**Signature:**
```python
def collect_metrics(self) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: Copy of metrics DataFrame

**Example:**
```python
results = tune_grid(wf, folds, grid=grid)
all_metrics = results.collect_metrics()

# Long format (default)
#    .config    .resample  metric     value
# 0  config_001 Fold01     rmse       0.523
# 1  config_001 Fold01     mae        0.412
# 2  config_001 Fold02     rmse       0.561
# ...

# Summarize by metric
summary = all_metrics.groupby('metric')['value'].agg(['mean', 'std'])
```

---

### `collect_predictions()` - Get All Predictions

Retrieve all predictions from tuning.

**Signature:**
```python
def collect_predictions(self) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: Copy of predictions DataFrame (empty if `save_pred=False`)

**Example:**
```python
control = {'save_pred': True}
results = tune_grid(wf, folds, grid=grid, control=control)

all_preds = results.collect_predictions()
#    .config    .resample  .row  .pred
# 0  config_001 Fold01     15    2.34
# 1  config_001 Fold01     23    3.12
# ...

# Analyze by configuration
config_001_preds = all_preds[all_preds['.config'] == 'config_001']
```

---

### `show_best()` - Top Configurations

Show the best n parameter configurations ranked by a metric.

**Signature:**
```python
def show_best(
    self,
    metric: str,
    n: int = 5,
    maximize: bool = True
) -> pd.DataFrame
```

**Parameters:**

- `metric` (str, required): Metric name to rank by
  - Must be in metrics DataFrame (e.g., 'rmse', 'mae', 'r_squared')

- `n` (int, default=5): Number of top configurations to return

- `maximize` (bool, default=True): Whether to maximize the metric
  - `True`: Higher values are better (accuracy, r_squared)
  - `False`: Lower values are better (rmse, mae)

**Returns:**
- `pd.DataFrame`: Top n configurations with:
  - `.config`: Configuration ID
  - `mean`: Mean metric value across folds
  - Parameter columns from grid

**Process:**
1. Calculate mean metric value for each configuration across folds
2. Sort by mean (ascending if minimize, descending if maximize)
3. Return top n with merged parameter values

**Examples:**

```python
results = tune_grid(wf, folds, grid=grid)

# Example 1: Show best for RMSE (minimize)
top_5 = results.show_best('rmse', n=5, maximize=False)
#    .config    mean    penalty  mixture
# 0  config_015 0.423   0.0316   0.5
# 1  config_023 0.431   0.1      0.3
# 2  config_008 0.445   0.01     0.7
# ...

# Example 2: Show best for R² (maximize)
top_3 = results.show_best('r_squared', n=3, maximize=True)

# Example 3: Show all configurations
all_configs = results.show_best('rmse', n=len(results.grid), maximize=False)

# Example 4: Visual comparison
import matplotlib.pyplot as plt
top_10 = results.show_best('rmse', n=10, maximize=False)
plt.barh(range(len(top_10)), top_10['mean'])
plt.yticks(range(len(top_10)), top_10['.config'])
plt.xlabel('RMSE')
plt.title('Top 10 Configurations')
```

---

### `select_best()` - Best Parameters

Select the single best parameter configuration.

**Signature:**
```python
def select_best(
    self,
    metric: str,
    maximize: bool = True
) -> Dict[str, Any]
```

**Parameters:**

- `metric` (str, required): Metric name to rank by

- `maximize` (bool, default=True): Whether to maximize the metric

**Returns:**
- `Dict[str, Any]`: Dictionary of best parameter values
  - Keys: parameter names
  - Values: best parameter values
  - Ready for `finalize_workflow()`

**Examples:**

```python
results = tune_grid(wf, folds, grid=grid)

# Example 1: Select best RMSE
best_rmse = results.select_best('rmse', maximize=False)
# {'penalty': 0.0316, 'mixture': 0.5}

# Example 2: Select best R²
best_r2 = results.select_best('r_squared', maximize=True)

# Example 3: Use with finalize_workflow
best = results.select_best('rmse', maximize=False)
final_wf = finalize_workflow(wf, best)

# Example 4: Multiple metrics
best_rmse = results.select_best('rmse', maximize=False)
best_mae = results.select_best('mae', maximize=False)

print(f"Best for RMSE: {best_rmse}")
print(f"Best for MAE: {best_mae}")

# Example 5: Compare to show_best
top_config = results.show_best('rmse', n=1, maximize=False)
best_params = results.select_best('rmse', maximize=False)

# top_config has .config and mean columns
# best_params is just the parameter dict
```

---

### `select_by_one_std_err()` - One-Standard-Error Rule

Select the simplest model within one standard error of the best performance.

**Signature:**
```python
def select_by_one_std_err(
    self,
    metric: str,
    maximize: bool = True
) -> Dict[str, Any]
```

**Parameters:**

- `metric` (str, required): Metric name to rank by

- `maximize` (bool, default=True): Whether to maximize the metric

**Returns:**
- `Dict[str, Any]`: Dictionary of selected parameter values (simpler model)

**Algorithm:**
1. Find best mean metric value and its standard error
2. Calculate threshold: `best_mean ± std_error`
3. Find all configurations within threshold (candidate set)
4. Among candidates, select simplest model (lowest value of first parameter)

**Rationale:**
- Prefer simpler models when performance is similar
- Reduces overfitting risk
- Improves interpretability and generalization
- From Breiman et al. (1984) CART methodology

**Simplicity Assumption:**
- Assumes first parameter represents model complexity
- Lower values = simpler models
- Examples:
  - `penalty`: Higher penalty = simpler (more regularization)
  - `trees`: Fewer trees = simpler
  - `tree_depth`: Shallower = simpler

**Examples:**

```python
results = tune_grid(wf, folds, grid=grid)

# Example 1: Select simpler model (one-SE rule)
best_complex = results.select_best('rmse', maximize=False)
best_simple = results.select_by_one_std_err('rmse', maximize=False)

print(f"Complex model: {best_complex}")  # {'penalty': 0.0316, 'mixture': 0.5}
print(f"Simple model: {best_simple}")    # {'penalty': 0.1, 'mixture': 0.5}

# Example 2: Comparison
final_wf_complex = finalize_workflow(wf, best_complex)
final_wf_simple = finalize_workflow(wf, best_simple)

fit_complex = final_wf_complex.fit(train_data)
fit_simple = final_wf_simple.fit(train_data)

# Evaluate on test
preds_complex = fit_complex.predict(test_data)
preds_simple = fit_simple.predict(test_data)

# Example 3: Visualize one-SE threshold
import matplotlib.pyplot as plt

all_results = results.show_best('rmse', n=len(results.grid), maximize=False)
best_mean = all_results['mean'].min()

# Calculate std error for best config (approximation)
best_config = all_results.iloc[0]['.config']
best_values = results.metrics[
    (results.metrics['.config'] == best_config) &
    (results.metrics['metric'] == 'rmse')
]['value']
best_std = best_values.std()
threshold = best_mean + best_std

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(all_results)), all_results['mean'])
plt.axhline(best_mean, color='g', linestyle='--', label='Best')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold (1 SE)')
plt.xlabel('Configuration')
plt.ylabel('RMSE')
plt.legend()

# Example 4: When simplicity matters
# Tree models: prefer fewer trees if performance similar
spec = boost_tree(trees=tune(), tree_depth=tune()).set_engine('xgboost')
wf = workflow().add_formula("y ~ .").add_model(spec)

param_info = {
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'}
}

results = tune_grid(wf, folds, grid=5, param_info=param_info)

# One-SE rule favors fewer trees (faster inference)
best_simple = results.select_by_one_std_err('rmse', maximize=False)
```

**When to Use:**

| Use select_best() | Use select_by_one_std_err() |
|-------------------|----------------------------|
| Maximum performance critical | Prefer simpler models |
| Small performance differences matter | Risk of overfitting high |
| Production latency not a concern | Production speed matters |
| Interpretability not required | Interpretability important |
| Training data representative | Training data noisy |

---

## Complete Workflow Examples

### Example 1: Linear Regression Tuning (Basic)

```python
import pandas as pd
from py_tune import tune, tune_grid, grid_regular, finalize_workflow
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared

# Load data
data = pd.read_csv('data.csv')
train_data = data[:800]
test_data = data[800:]

# Step 1: Define tunable model
spec = linear_reg(penalty=tune(id='penalty'), mixture=tune(id='mixture'))
wf = workflow().add_formula("target ~ .").add_model(spec)

# Step 2: Define parameter space
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'range': (0, 1)}
}

# Step 3: Create cross-validation folds
folds = vfold_cv(train_data, v=5, seed=42)

# Step 4: Define metrics
my_metrics = metric_set(rmse, mae, r_squared)

# Step 5: Tune
results = tune_grid(wf, folds, grid=5, param_info=param_info, metrics=my_metrics)

# Step 6: Analyze results
print("Top 5 configurations:")
print(results.show_best('rmse', n=5, maximize=False))

# Step 7: Select best parameters
best = results.select_best('rmse', maximize=False)
print(f"\nBest parameters: {best}")

# Step 8: Finalize workflow
final_wf = finalize_workflow(wf, best)

# Step 9: Fit on all training data
final_fit = final_wf.fit(train_data)

# Step 10: Evaluate on test set
test_preds = final_fit.predict(test_data)
test_rmse = rmse(test_data['target'], test_preds['.pred']).iloc[0]['value']
print(f"\nTest RMSE: {test_rmse:.4f}")
```

---

### Example 2: Random Forest Tuning (Intermediate)

```python
from py_tune import tune, tune_grid, grid_random, finalize_workflow
from py_parsnip import rand_forest
from py_rsample import vfold_cv

# Large parameter space - use random grid
spec = rand_forest(
    trees=tune(id='trees'),
    tree_depth=tune(id='depth'),
    min_n=tune(id='min_samples'),
    mtry=tune(id='features')
).set_mode('regression')

wf = workflow().add_formula("sales ~ .").add_model(spec)

# Define parameter ranges
param_info = {
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (5, 30), 'type': 'int'},
    'min_n': {'range': (2, 20), 'type': 'int'},
    'mtry': {'range': (2, 10), 'type': 'int'}
}

# Random grid (4^4 = 256 combinations, sample only 50)
grid = grid_random(param_info, size=50, seed=123)

# Tune
folds = vfold_cv(train_data, v=5, seed=123)
results = tune_grid(wf, folds, grid=grid)

# Analyze
top_10 = results.show_best('rmse', n=10, maximize=False)

# One-SE rule for simpler model
best_simple = results.select_by_one_std_err('rmse', maximize=False)
final_wf = finalize_workflow(wf, best_simple)

# Fit and evaluate
final_fit = final_wf.fit(train_data)
test_preds = final_fit.predict(test_data)
```

---

### Example 3: XGBoost Tuning (Advanced)

```python
from py_tune import tune, tune_grid, grid_random, finalize_workflow
from py_parsnip import boost_tree
from py_rsample import vfold_cv

# Multi-stage tuning strategy

# Stage 1: Coarse grid search (find general region)
spec_coarse = boost_tree(
    trees=tune(),
    tree_depth=tune(),
    learn_rate=tune(),
    min_n=tune()
).set_engine('xgboost')

wf = workflow().add_formula("target ~ .").add_model(spec_coarse)

param_info_coarse = {
    'trees': {'range': (100, 1000), 'type': 'int'},
    'tree_depth': {'range': (3, 10), 'type': 'int'},
    'learn_rate': {'range': (0.001, 0.3), 'trans': 'log'},
    'min_n': {'range': (5, 50), 'type': 'int'}
}

grid_coarse = grid_random(param_info_coarse, size=30, seed=42)
folds = vfold_cv(train_data, v=5, seed=42)

results_coarse = tune_grid(wf, folds, grid=grid_coarse)
best_coarse = results_coarse.select_best('rmse', maximize=False)
print("Coarse search best:", best_coarse)

# Stage 2: Fine grid search (refine around best region)
spec_fine = boost_tree(
    trees=tune(),
    tree_depth=best_coarse['tree_depth'],  # Fix from stage 1
    learn_rate=tune(),
    min_n=best_coarse['min_n']  # Fix from stage 1
).set_engine('xgboost')

wf_fine = workflow().add_formula("target ~ .").add_model(spec_fine)

# Narrow ranges around coarse best
param_info_fine = {
    'trees': {
        'range': (
            max(100, int(best_coarse['trees'] * 0.7)),
            int(best_coarse['trees'] * 1.3)
        ),
        'type': 'int'
    },
    'learn_rate': {
        'range': (
            best_coarse['learn_rate'] * 0.5,
            best_coarse['learn_rate'] * 2.0
        ),
        'trans': 'log'
    }
}

grid_fine = grid_regular(param_info_fine, levels=7)
results_fine = tune_grid(wf_fine, folds, grid=grid_fine)

best_fine = results_fine.select_best('rmse', maximize=False)
print("Fine search best:", best_fine)

# Finalize with best from fine search
final_params = {
    'trees': best_fine['trees'],
    'tree_depth': best_coarse['tree_depth'],
    'learn_rate': best_fine['learn_rate'],
    'min_n': best_coarse['min_n']
}

# Rebuild workflow with all parameters
spec_final = boost_tree(
    trees=final_params['trees'],
    tree_depth=final_params['tree_depth'],
    learn_rate=final_params['learn_rate'],
    min_n=final_params['min_n']
).set_engine('xgboost')

wf_final = workflow().add_formula("target ~ .").add_model(spec_final)
final_fit = wf_final.fit(train_data)
```

---

### Example 4: Time Series Tuning

```python
from py_tune import tune, tune_grid, grid_regular, finalize_workflow
from py_parsnip import linear_reg
from py_rsample import time_series_cv

# Time series data
ts_data = pd.DataFrame({
    'date': pd.date_range('2018-01-01', periods=1000, freq='D'),
    'sales': np.random.randn(1000).cumsum() + 100,
    'trend': range(1000),
    'day_of_week': (pd.date_range('2018-01-01', periods=1000, freq='D').dayofweek)
})

# Time series CV (expanding window)
ts_folds = time_series_cv(
    ts_data,
    date_col='date',
    initial='1 year',
    assess='1 month',
    skip='15 days',
    cumulative=True
)

# Tune model
spec = linear_reg(penalty=tune(), mixture=tune())
wf = workflow().add_formula("sales ~ trend + day_of_week").add_model(spec)

param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'mixture': {'range': (0, 1)}
}

results = tune_grid(wf, ts_folds, grid=5, param_info=param_info)

# Select best
best = results.select_best('rmse', maximize=False)
final_wf = finalize_workflow(wf, best)

# Fit and forecast
final_fit = final_wf.fit(ts_data[:900])
forecast = final_fit.predict(ts_data[900:])
```

---

### Example 5: Model Comparison with fit_resamples()

```python
from py_tune import fit_resamples
from py_parsnip import linear_reg, rand_forest, boost_tree

# Define multiple model types
models = {
    'linear': linear_reg(),
    'ridge': linear_reg(penalty=0.1, mixture=0),
    'lasso': linear_reg(penalty=0.1, mixture=1),
    'elastic_net': linear_reg(penalty=0.1, mixture=0.5),
    'random_forest': rand_forest(trees=500).set_mode('regression'),
    'xgboost': boost_tree(trees=500, tree_depth=6).set_engine('xgboost')
}

# Evaluate each model
folds = vfold_cv(train_data, v=10, seed=42)
results = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    wf = workflow().add_formula("target ~ .").add_model(model)
    results[name] = fit_resamples(wf, folds)

# Compare performance
comparison = []
for name, res in results.items():
    metrics = res.collect_metrics()
    rmse_values = metrics[metrics['metric'] == 'rmse']['value']
    comparison.append({
        'model': name,
        'mean_rmse': rmse_values.mean(),
        'std_rmse': rmse_values.std()
    })

comparison_df = pd.DataFrame(comparison).sort_values('mean_rmse')
print(comparison_df)

# Select best model type for further tuning
best_model_name = comparison_df.iloc[0]['model']
print(f"\nBest model: {best_model_name}")
```

---

## Best Practices

### 1. Parameter Grid Design

**Regular vs Random Grids:**

```python
# Use REGULAR grid when:
# - 1-2 parameters (full coverage feasible)
# - Narrow, well-understood range
# - Discrete parameter choices

param_info = {
    'penalty': {'values': [0.001, 0.01, 0.1, 1.0]},
    'mixture': {'values': [0, 0.5, 1.0]}
}
grid = grid_regular(param_info)  # 4 × 3 = 12 configs

# Use RANDOM grid when:
# - 3+ parameters (exponential explosion)
# - Wide parameter ranges
# - Initial exploration phase

param_info = {
    'param1': {'range': (0, 1)},
    'param2': {'range': (0, 100)},
    'param3': {'range': (0.001, 10), 'trans': 'log'},
    'param4': {'range': (5, 50), 'type': 'int'}
}
# Regular: 3^4 = 81 configs
# Random: control exact number
grid = grid_random(param_info, size=30, seed=42)
```

---

### 2. Transformation Choice

**Use log transformation for exponential-scale parameters:**

```python
# Penalty, learning rate (log scale)
param_info = {
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
    'learn_rate': {'range': (0.0001, 0.1), 'trans': 'log'}
}

# Linear parameters (identity scale)
param_info = {
    'mixture': {'range': (0, 1)},  # Alpha in elastic net
    'dropout': {'range': (0, 0.5)}  # Dropout rate
}

# Tree depth, number of samples (identity scale for small ranges)
param_info = {
    'tree_depth': {'range': (3, 15)},
    'min_n': {'range': (5, 50)}
}

# Number of trees (log or identity depending on range)
param_info = {
    'trees': {'range': (100, 1000), 'type': 'int'}  # Log might help
}
```

---

### 3. Multi-Stage Tuning Strategy

**For complex models with many parameters:**

```python
# Stage 1: Coarse search (all parameters, wide range)
param_info_stage1 = {
    'param1': {'range': (0.001, 1), 'trans': 'log'},
    'param2': {'range': (10, 1000), 'type': 'int'},
    'param3': {'range': (0, 1)}
}
grid1 = grid_random(param_info_stage1, size=50, seed=42)
results1 = tune_grid(wf, folds, grid=grid1)
best1 = results1.select_best('rmse', maximize=False)

# Stage 2: Fine search (narrow range around best)
param_info_stage2 = {
    'param1': {
        'range': (best1['param1'] * 0.5, best1['param1'] * 2.0),
        'trans': 'log'
    },
    'param2': {
        'range': (best1['param2'] - 100, best1['param2'] + 100),
        'type': 'int'
    }
}
grid2 = grid_regular(param_info_stage2, levels=7)
results2 = tune_grid(wf, folds, grid=grid2)
best2 = results2.select_best('rmse', maximize=False)
```

---

### 4. Cross-Validation Strategy

```python
# Standard data: k-fold CV
folds = vfold_cv(data, v=10, seed=42)

# Time series: expanding window
ts_folds = time_series_cv(
    data,
    date_col='date',
    initial='2 years',
    assess='3 months',
    skip='1 month',
    cumulative=True
)

# Small data: repeated CV
from py_rsample import vfold_cv
folds = vfold_cv(data, v=5, repeats=3, seed=42)

# Stratified for classification
folds = vfold_cv(data, v=5, strata='class_column', seed=42)
```

---

### 5. Metric Selection

```python
# Regression: primary metric + diagnostics
my_metrics = metric_set(
    rmse,        # Primary: overall error
    mae,         # Robust to outliers
    r_squared    # Variance explained
)

# Compare models across multiple metrics
results = tune_grid(wf, folds, grid=grid, metrics=my_metrics)

# Rank by primary
best_rmse = results.select_best('rmse', maximize=False)

# Check other metrics for best config
best_config = results.show_best('rmse', n=1, maximize=False).iloc[0]['.config']
all_metrics = results.collect_metrics()
config_metrics = all_metrics[all_metrics['.config'] == best_config]
print(config_metrics.groupby('metric')['value'].mean())
```

---

### 6. Computational Budget Management

```python
# Calculate expected runtime
n_configs = len(grid)
n_folds = len(folds)
total_fits = n_configs * n_folds

# Example: 50 configs × 5 folds = 250 fits
# If each fit takes 10 seconds = 2500 seconds = 42 minutes

# Strategies for large budgets:
# 1. Reduce folds (faster but less robust)
folds = vfold_cv(data, v=3, seed=42)  # Instead of 10

# 2. Use random grid with smaller size
grid = grid_random(param_info, size=20, seed=42)  # Instead of 50

# 3. Parallel processing (future feature)
# control = {'allow_par': True, 'cores': 4}

# 4. Early stopping for iterative models
spec = boost_tree(
    trees=tune(),
    stop_iter=10  # Stop if no improvement for 10 iterations
).set_engine('xgboost')
```

---

### 7. Reproducibility

```python
# Always set seeds for reproducibility

# Grid generation
grid = grid_random(param_info, size=30, seed=42)

# CV splits
folds = vfold_cv(data, v=5, seed=42)

# Time series CV
ts_folds = time_series_cv(
    data,
    date_col='date',
    initial='2 years',
    assess='3 months',
    seed=42  # Future feature
)

# Model-level randomness
spec = rand_forest(trees=500).set_mode('regression')
wf = workflow().add_formula("y ~ .").add_model(spec)

# Set seed in model fit (via engine-specific args)
spec = rand_forest(trees=500, seed=42).set_mode('regression')
```

---

### 8. Overfitting Prevention

```python
# 1. Use one-standard-error rule
best_simple = results.select_by_one_std_err('rmse', maximize=False)

# 2. Regularization in parameter space
param_info = {
    'penalty': {'range': (0.01, 1.0), 'trans': 'log'},  # Exclude very small
    'trees': {'range': (100, 500)}  # Cap at moderate size
}

# 3. Monitor train vs validation gap
control = {'save_pred': True}
results = tune_grid(wf, folds, grid=grid, control=control)

# Calculate per-fold performance
all_metrics = results.collect_metrics()
fold_performance = all_metrics.groupby(['.config', '.resample'])['value'].mean()

# Check variance across folds (high variance = overfitting)
config_variance = all_metrics.groupby('.config')['value'].std()

# 4. Hold-out test set (separate from CV)
final_fit = final_wf.fit(train_data)
test_preds = final_fit.predict(test_data)
test_rmse = rmse(test_data['target'], test_preds['.pred'])

# Compare CV RMSE to test RMSE
cv_rmse = results.show_best('rmse', n=1, maximize=False).iloc[0]['mean']
print(f"CV RMSE: {cv_rmse:.4f}")
print(f"Test RMSE: {test_rmse.iloc[0]['value']:.4f}")
```

---

### 9. Debugging Failed Configurations

```python
# tune_grid prints warnings for failed configs
# Example output:
# "Warning: Config config_015, Fold 3 failed: Invalid parameter combination"

# Investigate failed configs
results = tune_grid(wf, folds, grid=grid)

# Check which configs succeeded
successful_configs = results.metrics['.config'].unique()
all_configs = grid['.config'].tolist()
failed_configs = [c for c in all_configs if c not in successful_configs]

print(f"Failed configs: {failed_configs}")

# Get parameter values for failed configs
failed_params = grid[grid['.config'].isin(failed_configs)]
print(failed_params)

# Common failure reasons:
# - Invalid parameter combinations (e.g., min_n > n_samples)
# - Numerical issues (e.g., penalty too small)
# - Convergence failures

# Fix by adjusting parameter ranges
param_info = {
    'min_n': {'range': (5, 50), 'type': 'int'},  # Ensure min_n reasonable
    'penalty': {'range': (0.001, 1.0), 'trans': 'log'}  # Avoid too small
}
```

---

### 10. Saving and Loading Results

```python
import pickle

# Save results
with open('tune_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load results
with open('tune_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Re-analyze
best = results.select_best('rmse', maximize=False)
final_wf = finalize_workflow(wf, best)

# Save final model
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_fit, f)
```

---

## Common Patterns Cheat Sheet

### Pattern 1: Quick Tuning Pipeline

```python
from py_tune import tune, tune_grid, grid_regular, finalize_workflow
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv

# Define → Grid → Tune → Finalize → Fit
spec = linear_reg(penalty=tune(), mixture=tune())
wf = workflow().add_formula("y ~ .").add_model(spec)
param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
              'mixture': {'range': (0, 1)}}
folds = vfold_cv(train_data, v=5)
results = tune_grid(wf, folds, grid=5, param_info=param_info)
best = results.select_best('rmse', maximize=False)
final_wf = finalize_workflow(wf, best)
final_fit = final_wf.fit(train_data)
```

### Pattern 2: Model Comparison

```python
from py_tune import fit_resamples

models = {'linear': linear_reg(), 'rf': rand_forest().set_mode('regression')}
results = {name: fit_resamples(
    workflow().add_formula("y ~ .").add_model(model), folds
) for name, model in models.items()}
comparison = {name: res.collect_metrics().groupby('metric')['value'].mean()
              for name, res in results.items()}
```

### Pattern 3: Time Series Tuning

```python
from py_rsample import time_series_cv

ts_folds = time_series_cv(data, date_col='date', initial='2 years',
                          assess='3 months', cumulative=True)
results = tune_grid(wf, ts_folds, grid=grid)
```

---

## Summary Table

| Function | Purpose | Returns |
|----------|---------|---------|
| `tune(id)` | Mark parameter for tuning | TuneParameter |
| `grid_regular(param_info, levels)` | Regular grid | DataFrame |
| `grid_random(param_info, size, seed)` | Random grid | DataFrame |
| `tune_grid(wf, resamples, grid, ...)` | Grid search CV | TuneResults |
| `fit_resamples(wf, resamples, ...)` | Evaluate single config | TuneResults |
| `finalize_workflow(wf, params)` | Apply best params | Workflow |
| `TuneResults.collect_metrics()` | Get all metrics | DataFrame |
| `TuneResults.show_best(metric, n)` | Top n configs | DataFrame |
| `TuneResults.select_best(metric)` | Best params | Dict |
| `TuneResults.select_by_one_std_err(metric)` | Simpler model | Dict |

---

**End of Complete Tuning Reference**
