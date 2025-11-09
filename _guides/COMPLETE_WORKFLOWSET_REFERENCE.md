# Complete py_workflowsets Reference

**Module:** `py_workflowsets`
**Purpose:** Multi-model comparison framework for efficiently evaluating multiple preprocessing strategies and model specifications
**Documentation Date:** 2025-11-09
**Version:** Current (py-tidymodels)

---

## Table of Contents

1. [Overview](#overview)
2. [WorkflowSet Class](#workflowset-class)
3. [WorkflowSetResults Class](#workflowsetresults-class)
4. [Complete Usage Patterns](#complete-usage-patterns)
5. [Integration with Other Modules](#integration-with-other-modules)
6. [Best Practices](#best-practices)
7. [Common Patterns](#common-patterns)

---

## Overview

The `py_workflowsets` module provides a framework for comparing multiple modeling workflows in a systematic and efficient manner. It allows you to:

- Create cross-products of preprocessors (formulas/recipes) and models
- Evaluate all workflow combinations across resampling folds
- Compare and rank workflows by performance metrics
- Visualize results with automatic plotting
- Integrate with `py_tune` for hyperparameter tuning

**Key Benefits:**
- Reduces boilerplate code for multi-model comparison
- Ensures consistent evaluation across all workflows
- Provides standardized result format for easy analysis
- Supports parallel evaluation (designed for extensibility)

---

## WorkflowSet Class

### Class Definition

```python
@dataclass
class WorkflowSet:
    """
    Collection of workflows for multi-model comparison.

    Attributes:
        workflows: Dictionary mapping workflow IDs to Workflow objects
        info: Metadata DataFrame about each workflow
    """
    workflows: Dict[str, Any]
    info: pd.DataFrame
```

### Attributes

#### `workflows: Dict[str, Any]`
Dictionary mapping workflow IDs to Workflow objects.

**Structure:**
```python
{
    "workflow_id_1": Workflow(...),
    "workflow_id_2": Workflow(...),
    ...
}
```

**Access Pattern:**
```python
# Get specific workflow
wf = wf_set["workflow_id_1"]
wf = wf_set.workflows["workflow_id_1"]  # Equivalent

# Iterate over workflow IDs
for wf_id in wf_set:
    print(wf_id)
```

#### `info: pd.DataFrame`
Metadata DataFrame containing information about each workflow.

**Columns:**
- `wflow_id` (str): Workflow identifier
- `info` (str): Combined preprocessor and model type (e.g., "formula_linear_reg")
- `option` (str): Preprocessor ID or option name
- `preprocessor` (str): Preprocessor type ("formula", "recipe", or "none")
- `model` (str): Model type (e.g., "linear_reg", "rand_forest")

**Example:**
```python
   wflow_id                info       option preprocessor       model
0  minimal_linear_reg_1  formula_linear_reg  minimal      formula  linear_reg
1  minimal_linear_reg_2  formula_linear_reg  minimal      formula  linear_reg
```

---

### Class Methods

#### `from_workflows()`

Create WorkflowSet from an explicit list of workflows.

**Signature:**
```python
@classmethod
def from_workflows(
    cls,
    workflows: List[Any],
    ids: Optional[List[str]] = None
) -> WorkflowSet
```

**Parameters:**
- **`workflows`** (List[Any]):
  - List of Workflow objects, OR
  - List of (id, workflow) tuples
- **`ids`** (Optional[List[str]]):
  - List of IDs for workflows
  - Auto-generated as `["workflow_1", "workflow_2", ...]` if None
  - Ignored if workflows is a list of tuples
  - Must match length of workflows

**Returns:**
- `WorkflowSet` instance

**Raises:**
- `ValueError`: If length of ids doesn't match length of workflows

**Examples:**

```python
# Method 1: Separate workflows and IDs
wf1 = workflow().add_formula("y ~ x1").add_model(linear_reg())
wf2 = workflow().add_formula("y ~ x1 + x2").add_model(rand_forest())

wf_set = WorkflowSet.from_workflows(
    [wf1, wf2],
    ids=["linear", "rf"]
)

# Method 2: List of tuples (id, workflow)
wf_set = WorkflowSet.from_workflows([
    ("linear", wf1),
    ("rf", wf2)
])

# Method 3: Auto-generated IDs
wf_set = WorkflowSet.from_workflows([wf1, wf2])
# Creates IDs: "workflow_1", "workflow_2"
```

---

#### `from_cross()`

Create WorkflowSet from cross-product of preprocessors and models.

**Signature:**
```python
@classmethod
def from_cross(
    cls,
    preproc: List[Union[str, Any]],
    models: List[Any],
    ids: Optional[List[str]] = None
) -> WorkflowSet
```

**Parameters:**
- **`preproc`** (List[Union[str, Any]]):
  - List of formulas (strings) OR Recipe objects
  - Each preprocessor will be combined with each model
- **`models`** (List[Any]):
  - List of ModelSpec objects
  - Each model will be combined with each preprocessor
- **`ids`** (Optional[List[str]]):
  - List of ID prefixes for each preprocessor
  - Auto-generated as `["prep_1", "prep_2", ...]` if None
  - Must match length of preproc

**Returns:**
- `WorkflowSet` instance with len(preproc) × len(models) workflows

**Workflow ID Format:**
- `{preproc_id}_{model_type}_{model_index}`
- Example: `"minimal_linear_reg_1"`, `"full_rand_forest_2"`

**Examples:**

```python
# Example 1: Cross-product with formulas
formulas = [
    "y ~ x1",
    "y ~ x1 + x2",
    "y ~ x1 + x2 + x3"
]
models = [
    linear_reg(),
    rand_forest().set_mode('regression')
]

wf_set = WorkflowSet.from_cross(formulas, models)
# Creates 3 × 2 = 6 workflows
# IDs: prep_1_linear_reg_1, prep_1_rand_forest_2, ...

# Example 2: With custom IDs
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["simple", "medium", "full"]
)
# IDs: simple_linear_reg_1, simple_rand_forest_2, ...

# Example 3: With recipes
from py_recipes import recipe, step_normalize, step_dummy

rec1 = recipe().step_normalize(all_numeric())
rec2 = (recipe()
    .step_normalize(all_numeric())
    .step_dummy(all_nominal()))

wf_set = WorkflowSet.from_cross(
    preproc=[rec1, rec2],
    models=[linear_reg(), svm_rbf()],
    ids=["normalized", "normalized_dummy"]
)
# Creates 2 × 2 = 4 workflows

# Example 4: Complex comparison
formulas = [
    "y ~ x1 + x2",                           # Baseline
    "y ~ x1 + x2 + x3",                      # More features
    "y ~ x1 + x2 + I(x1*x2)",                # Interaction
    "y ~ x1 + x2 + I(x1**2) + I(x2**2)"      # Polynomial
]
models = [
    linear_reg(),                            # OLS
    linear_reg(penalty=0.1, mixture=1.0),    # Lasso
    linear_reg(penalty=0.1, mixture=0.5),    # Elastic Net
]

wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["baseline", "extended", "interaction", "polynomial"]
)
# Creates 4 × 3 = 12 workflows
```

---

### Instance Methods

#### `__len__()`

Get number of workflows in the set.

**Signature:**
```python
def __len__(self) -> int
```

**Returns:**
- int: Number of workflows

**Example:**
```python
wf_set = WorkflowSet.from_cross(formulas, models)
n_workflows = len(wf_set)
print(f"Evaluating {n_workflows} workflows")
```

---

#### `__iter__()`

Iterate over workflow IDs.

**Signature:**
```python
def __iter__(self) -> Iterator[str]
```

**Returns:**
- Iterator over workflow IDs

**Example:**
```python
for wf_id in wf_set:
    print(f"Workflow: {wf_id}")
    wf = wf_set[wf_id]
    # Process workflow...
```

---

#### `__getitem__()`

Get workflow by ID (indexing support).

**Signature:**
```python
def __getitem__(self, key: str) -> Workflow
```

**Parameters:**
- **`key`** (str): Workflow ID

**Returns:**
- `Workflow` object

**Raises:**
- `KeyError`: If workflow ID not found

**Example:**
```python
# Get specific workflow
best_wf = wf_set["minimal_linear_reg_1"]

# Fit on full data
fitted = best_wf.fit(train_data)

# Make predictions
predictions = fitted.predict(test_data)
```

---

#### `workflow_map()`

Apply a function to all workflows (unified interface).

**Signature:**
```python
def workflow_map(
    self,
    fn: str,
    resamples: Any = None,
    metrics: Any = None,
    grid: Any = None,
    **kwargs
) -> WorkflowSetResults
```

**Parameters:**
- **`fn`** (str): Function name to apply
  - `"fit_resamples"`: Evaluate without tuning
  - `"tune_grid"`: Hyperparameter tuning
- **`resamples`** (Any): Resampling object (VFoldCV, TimeSeriesCV, etc.)
- **`metrics`** (Any): Metric set from `py_yardstick.metric_set()`
- **`grid`** (Any): Parameter grid (for tune_grid only)
- **`**kwargs`**: Additional arguments passed to the function

**Returns:**
- `WorkflowSetResults` object

**Raises:**
- `ValueError`: If unknown function name

**Examples:**

```python
# Example 1: fit_resamples via workflow_map
results = wf_set.workflow_map(
    "fit_resamples",
    resamples=folds,
    metrics=metric_set(rmse, mae)
)

# Example 2: tune_grid via workflow_map
results = wf_set.workflow_map(
    "tune_grid",
    resamples=folds,
    grid=param_grid,
    metrics=metric_set(rmse, r_squared)
)

# Note: Direct methods are preferred over workflow_map
# results = wf_set.fit_resamples(folds, metrics)  # Preferred
```

---

#### `fit_resamples()`

Fit all workflows to all resamples and evaluate (no tuning).

**Signature:**
```python
def fit_resamples(
    self,
    resamples: Any,
    metrics: Any = None,
    control: Optional[Dict[str, Any]] = None
) -> WorkflowSetResults
```

**Parameters:**
- **`resamples`** (Any): Resampling object
  - `VFoldCV` from `vfold_cv()`
  - `TimeSeriesCV` from `time_series_cv()`
  - `InitialSplit` from `initial_split()`
- **`metrics`** (Any): Metric set for evaluation
  - Created via `metric_set(metric1, metric2, ...)`
  - If None, uses default metrics
- **`control`** (Optional[Dict[str, Any]]): Control parameters
  - `{'save_pred': True}`: Save predictions for each resample
  - `{'verbose': True}`: Print progress (default behavior)

**Returns:**
- `WorkflowSetResults` containing metrics and predictions for all workflows

**Side Effects:**
- Prints progress messages: "Fitting {wf_id}..."

**Examples:**

```python
# Example 1: Basic usage
folds = vfold_cv(data, v=5, seed=42)
metrics = metric_set(rmse, mae, r_squared)

results = wf_set.fit_resamples(
    resamples=folds,
    metrics=metrics
)

# Example 2: Save predictions
results = wf_set.fit_resamples(
    resamples=folds,
    metrics=metrics,
    control={'save_pred': True}
)
preds_df = results.collect_predictions()

# Example 3: Time series CV
ts_folds = time_series_cv(
    data,
    initial="1 year",
    assess="3 months",
    cumulative=True
)
results = wf_set.fit_resamples(ts_folds, metrics)

# Example 4: Different metrics for different tasks
# Regression metrics
reg_metrics = metric_set(rmse, mae, r_squared, mape)

# Classification metrics
clf_metrics = metric_set(accuracy, precision, recall, f1)

results = wf_set.fit_resamples(folds, reg_metrics)
```

---

#### `tune_grid()`

Tune all workflows over parameter grids with cross-validation.

**Signature:**
```python
def tune_grid(
    self,
    resamples: Any,
    grid: Any,
    metrics: Any = None,
    control: Optional[Dict[str, Any]] = None
) -> WorkflowSetResults
```

**Parameters:**
- **`resamples`** (Any): Resampling object (same as `fit_resamples`)
- **`grid`** (Any): Parameter grid specification
  - Single grid: Applied to all workflows
  - Dict of grids: `{wf_id: grid}` for per-workflow grids
  - Created via `grid_regular()` or `grid_random()`
- **`metrics`** (Any): Metric set for evaluation
- **`control`** (Optional[Dict[str, Any]]): Control parameters
  - Same as `fit_resamples`

**Returns:**
- `WorkflowSetResults` containing tuning results for all workflows

**Side Effects:**
- Prints progress messages: "Tuning {wf_id}..."

**Examples:**

```python
# Example 1: Single grid for all workflows
from py_tune import grid_regular, tune

# Mark parameters for tuning
spec = linear_reg(penalty=tune(), mixture=tune())

# Create grid
grid = grid_regular(
    {"penalty": {"range": (0.001, 1.0), "trans": "log"},
     "mixture": {"range": (0, 1)}},
    levels=5
)

# Tune all workflows with same grid
results = wf_set.tune_grid(
    resamples=folds,
    grid=grid,
    metrics=metric_set(rmse, mae)
)

# Example 2: Per-workflow grids
wf_set = WorkflowSet.from_workflows([
    ("lasso", workflow().add_formula("y ~ .").add_model(
        linear_reg(penalty=tune(), mixture=1.0)
    )),
    ("ridge", workflow().add_formula("y ~ .").add_model(
        linear_reg(penalty=tune(), mixture=0.0)
    ))
])

# Different grids for different workflows
grids = {
    "lasso": grid_regular({"penalty": {"range": (0.001, 1.0)}}, levels=10),
    "ridge": grid_regular({"penalty": {"range": (0.01, 10.0)}}, levels=10)
}

results = wf_set.tune_grid(folds, grid=grids, metrics=metrics)

# Example 3: Random grid search
grid = grid_random(
    {"penalty": {"range": (0.001, 1.0), "trans": "log"},
     "mixture": {"range": (0, 1)}},
    size=20  # 20 random combinations
)

results = wf_set.tune_grid(folds, grid=grid, metrics=metrics)
```

---

## WorkflowSetResults Class

### Class Definition

```python
@dataclass
class WorkflowSetResults:
    """
    Results from fitting a WorkflowSet.

    Attributes:
        results: List of dictionaries containing results for each workflow
        workflow_set: The original WorkflowSet
        metrics: The metric set used for evaluation
    """
    results: List[Dict[str, Any]]
    workflow_set: WorkflowSet
    metrics: Any
```

### Attributes

#### `results: List[Dict[str, Any]]`
List of result dictionaries, one per workflow.

**Structure of each dictionary:**
```python
{
    "wflow_id": str,           # Workflow ID
    "tune_results": TuneResults,  # Results from fit_resamples/tune_grid
    "metrics": pd.DataFrame    # Metrics with wflow_id column
}
```

#### `workflow_set: WorkflowSet`
The original WorkflowSet that was evaluated.

**Usage:**
```python
# Access workflow info
info_df = results.workflow_set.info

# Access specific workflow
best_wf = results.workflow_set["workflow_id"]
```

#### `metrics: Any`
The metric set used for evaluation.

**Usage:**
```python
# Typically a metric_set() object
# Used internally for metric collection
```

---

### Methods

#### `collect_metrics()`

Collect all metrics from all workflows.

**Signature:**
```python
def collect_metrics(
    self,
    summarize: bool = True
) -> pd.DataFrame
```

**Parameters:**
- **`summarize`** (bool): Default `True`
  - `True`: Return mean and std across resamples (summarized)
  - `False`: Return raw metrics from each resample (unsummarized)

**Returns:**
- `pd.DataFrame` with metrics for all workflows

**DataFrame Schema (summarize=True):**
```python
Columns:
- wflow_id (str): Workflow ID
- metric (str): Metric name (e.g., "rmse", "mae")
- mean (float): Mean metric value across resamples
- std (float): Standard deviation across resamples
- n (int): Number of resamples
- preprocessor (str): Preprocessor type
- model (str): Model type
```

**DataFrame Schema (summarize=False):**
```python
Columns:
- wflow_id (str): Workflow ID
- metric (str): Metric name
- value (float): Metric value for this resample
- id (str): Resample ID (e.g., "Fold1", "Fold2")
- Additional columns from TuneResults.collect_metrics()
```

**Examples:**

```python
# Example 1: Summarized metrics (default)
metrics_df = results.collect_metrics()

print(metrics_df)
#            wflow_id     metric      mean       std  n preprocessor       model
# 0  minimal_linear_reg_1  rmse  1.234567  0.123456  5      formula  linear_reg
# 1  minimal_linear_reg_1   mae  0.987654  0.098765  5      formula  linear_reg

# Example 2: Unsummarized (raw) metrics
raw_metrics = results.collect_metrics(summarize=False)

print(raw_metrics)
#            wflow_id metric     value     id
# 0  minimal_linear_reg_1   rmse  1.250000  Fold1
# 1  minimal_linear_reg_1   rmse  1.220000  Fold2
# 2  minimal_linear_reg_1   rmse  1.235000  Fold3

# Example 3: Filter and analyze
metrics_df = results.collect_metrics()

# Get only RMSE values
rmse_df = metrics_df[metrics_df['metric'] == 'rmse'].copy()
rmse_df = rmse_df.sort_values('mean')

# Compare models
model_comparison = metrics_df.groupby(['model', 'metric'])['mean'].mean()

# Best workflow per model type
best_per_model = (metrics_df[metrics_df['metric'] == 'rmse']
                  .sort_values('mean')
                  .groupby('model')
                  .first())
```

---

#### `collect_predictions()`

Collect all predictions from all workflows.

**Signature:**
```python
def collect_predictions(self) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame` with predictions for all workflows

**DataFrame Schema:**
```python
Columns:
- wflow_id (str): Workflow ID
- .pred (float): Predicted values
- .row (int): Row index in original data
- id (str): Resample ID
- [outcome_col] (float): Actual values (if available)
- Additional columns from TuneResults.collect_predictions()
```

**Requirements:**
- Must use `control={'save_pred': True}` in `fit_resamples()` or `tune_grid()`
- Otherwise, predictions are not saved and this returns empty DataFrame

**Examples:**

```python
# Example 1: Collect predictions
results = wf_set.fit_resamples(
    folds,
    metrics=metrics,
    control={'save_pred': True}
)

preds_df = results.collect_predictions()

print(preds_df.head())
#            wflow_id     .pred  .row     id        y
# 0  minimal_linear_reg_1  5.123     0  Fold1  5.000
# 1  minimal_linear_reg_1  6.456     1  Fold1  6.200

# Example 2: Analyze predictions by workflow
for wf_id in results.workflow_set:
    wf_preds = preds_df[preds_df['wflow_id'] == wf_id]

    # Calculate residuals
    wf_preds['residual'] = wf_preds['y'] - wf_preds['.pred']

    # Plot residuals
    plt.scatter(wf_preds['.pred'], wf_preds['residual'])
    plt.title(f"Residuals: {wf_id}")
    plt.show()

# Example 3: Ensemble predictions
# Average predictions across multiple workflows
ensemble_preds = (preds_df.groupby(['.row', 'id'])
                  .agg({'.pred': 'mean', 'y': 'first'})
                  .reset_index())
```

---

#### `rank_results()`

Rank workflows by a specific metric.

**Signature:**
```python
def rank_results(
    self,
    rank_metric: Optional[str] = None,
    metric: Optional[str] = None,
    select_best: bool = False,
    n: int = 10
) -> pd.DataFrame
```

**Parameters:**
- **`rank_metric`** (Optional[str]):
  - Metric name to rank by (e.g., "rmse")
  - **DEPRECATED**: Use `metric` parameter instead
- **`metric`** (Optional[str]):
  - Metric name to rank by (e.g., "rmse")
  - **PREFERRED** parameter name
- **`select_best`** (bool): Default `False`
  - `True`: Return only the best workflow per model type
  - `False`: Return top N workflows overall
- **`n`** (int): Default `10`
  - Number of top workflows to return (if select_best=False)

**Returns:**
- `pd.DataFrame` with ranked workflows in **wide format**

**DataFrame Schema (Wide Format):**
```python
Columns:
- rank (int): Rank (1 = best)
- wflow_id (str): Workflow ID
- preprocessor (str): Preprocessor type
- model (str): Model type
- {metric}_mean (float): Mean value for each metric
- {metric}_std (float): Std value for each metric
- {metric}_n (int): Count for each metric

Example columns:
- rmse_mean, rmse_std, rmse_n
- mae_mean, mae_std, mae_n
- r_squared_mean, r_squared_std, r_squared_n
```

**Ranking Logic:**
- Automatically determines whether to minimize or maximize metric
- **Minimize metrics**: rmse, mae, mape, smape, mse, log_loss, brier_score
- **Maximize metrics**: r_squared, accuracy, precision, recall, f1, auc, etc.

**Raises:**
- `ValueError`: If metric not found in results
- `ValueError`: If neither `metric` nor `rank_metric` provided

**Examples:**

```python
# Example 1: Top 10 workflows by RMSE
top10 = results.rank_results(metric="rmse", n=10)

print(top10)
#    rank                  wflow_id preprocessor       model  rmse_mean  rmse_std
# 0     1  interaction_linear_reg_1      formula  linear_reg   1.158548  0.107885
# 1     2       medium_linear_reg_2      formula  linear_reg   1.269358  0.192810

# Example 2: Top 5 by R² (higher is better)
top5_r2 = results.rank_results(metric="r_squared", n=5)

# Example 3: Best workflow per model type
best_per_model = results.rank_results(metric="rmse", select_best=True)

print(best_per_model)
# Returns one row per model type, ranked by RMSE

# Example 4: Access best workflow
best_wf_id = top10.iloc[0]["wflow_id"]
best_wf = results.workflow_set[best_wf_id]

# Fit on full data
final_fit = best_wf.fit(train_data)

# Example 5: Compare multiple metrics
ranked_df = results.rank_results(metric="rmse", n=5)

# Wide format has all metrics as columns
print(ranked_df[['rank', 'wflow_id', 'rmse_mean', 'mae_mean', 'r_squared_mean']])

# Example 6: Legacy parameter name (still works)
top10 = results.rank_results(rank_metric="rmse", n=10)  # DEPRECATED
```

---

#### `autoplot()`

Plot workflow comparison results with automatic visualization.

**Signature:**
```python
def autoplot(
    self,
    metric: Optional[str] = None,
    select_best: bool = False,
    top_n: int = 15
) -> plt.Figure
```

**Parameters:**
- **`metric`** (Optional[str]):
  - Metric to plot
  - If None, plots first metric in results
- **`select_best`** (bool): Default `False`
  - `True`: Show only best workflow per model type
  - `False`: Show top N workflows overall
- **`top_n`** (int): Default `15`
  - Number of top workflows to show

**Returns:**
- `matplotlib.pyplot.Figure` object

**Plot Characteristics:**
- **Type**: Horizontal bar chart with error bars
- **Y-axis**: Workflow IDs
- **X-axis**: Metric value (mean ± std)
- **Ordering**: Best workflows at top
- **Colors**: Color-coded by model type
- **Error bars**: Standard deviation across resamples
- **Legend**: Model type legend

**Examples:**

```python
# Example 1: Basic usage
fig = results.autoplot("rmse")
plt.show()

# Example 2: Show top 10 workflows
fig = results.autoplot("rmse", top_n=10)
plt.show()

# Example 3: Best per model type
fig = results.autoplot("rmse", select_best=True)
plt.show()

# Example 4: Save figure
fig = results.autoplot("rmse", top_n=20)
fig.savefig("workflow_comparison.png", dpi=300, bbox_inches='tight')

# Example 5: Customize after creation
fig = results.autoplot("rmse")
ax = fig.gca()
ax.set_xlabel("RMSE (Test Set)")
ax.set_title("Model Comparison: RMSE")
plt.tight_layout()
plt.show()

# Example 6: Multiple metrics side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot RMSE
results.autoplot("rmse", top_n=10)
# Plot R²
results.autoplot("r_squared", top_n=10)

plt.tight_layout()
plt.show()

# Example 7: Use default metric
fig = results.autoplot()  # Uses first metric
plt.show()
```

---

## Complete Usage Patterns

### Pattern 1: Basic Multi-Model Comparison

**Scenario:** Compare different feature combinations with linear regression.

```python
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared

# 1. Load data
data = pd.read_csv("data.csv")

# 2. Define preprocessing strategies
formulas = [
    "y ~ x1",
    "y ~ x1 + x2",
    "y ~ x1 + x2 + x3",
]

# 3. Define models
models = [linear_reg()]

# 4. Create workflow set
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["simple", "medium", "full"]
)

# 5. Create CV folds
folds = vfold_cv(data, v=5, seed=42)

# 6. Define metrics
metrics = metric_set(rmse, mae, r_squared)

# 7. Evaluate all workflows
results = wf_set.fit_resamples(folds, metrics=metrics)

# 8. Rank results
top_workflows = results.rank_results("rmse", n=5)
print(top_workflows)

# 9. Visualize
fig = results.autoplot("rmse")
plt.show()

# 10. Select and finalize best workflow
best_wf_id = top_workflows.iloc[0]["wflow_id"]
best_wf = wf_set[best_wf_id]
final_fit = best_wf.fit(data)
```

---

### Pattern 2: Comparing Multiple Model Types

**Scenario:** Compare different model types with same preprocessing.

```python
from py_parsnip import linear_reg, rand_forest, svm_rbf, decision_tree

# Define multiple model types
models = [
    linear_reg(),
    rand_forest(trees=500, min_n=5).set_mode('regression'),
    svm_rbf(cost=1.0),
    decision_tree(tree_depth=5).set_mode('regression')
]

# Single formula
formulas = ["y ~ ."]

# Create workflow set
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["all_features"]
)

# Evaluate
results = wf_set.fit_resamples(folds, metrics=metrics)

# Compare model types
best_per_model = results.rank_results("rmse", select_best=True)
print(best_per_model[['rank', 'model', 'rmse_mean', 'rmse_std']])
```

---

### Pattern 3: Advanced Feature Engineering Comparison

**Scenario:** Compare multiple feature engineering strategies.

```python
# Define complex formulas
formulas = [
    "y ~ x1 + x2",                           # Baseline
    "y ~ x1 + x2 + x3 + x4",                 # More features
    "y ~ x1 + x2 + I(x1*x2)",                # Interaction
    "y ~ x1 + x2 + I(x1**2) + I(x2**2)",     # Polynomial
    "y ~ x1 + x2 + I(x1*x2) + I(x1**2)",     # Both
]

# Define model variations
models = [
    linear_reg(),                            # OLS
    linear_reg(penalty=0.01, mixture=1.0),   # Light Lasso
    linear_reg(penalty=0.1, mixture=1.0),    # Strong Lasso
    linear_reg(penalty=0.1, mixture=0.5),    # Elastic Net
]

# Create workflow set (5 × 4 = 20 workflows)
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["baseline", "extended", "interaction", "polynomial", "combined"]
)

# Evaluate
results = wf_set.fit_resamples(folds, metrics=metrics)

# Analyze by preprocessor
metrics_df = results.collect_metrics()
by_preproc = (metrics_df[metrics_df['metric'] == 'rmse']
              .groupby('preprocessor')
              .agg({'mean': ['mean', 'min', 'max']}))
print(by_preproc)

# Best overall
top5 = results.rank_results("rmse", n=5)
print(top5)
```

---

### Pattern 4: Recipe-Based Comparison

**Scenario:** Compare different recipe preprocessing pipelines.

```python
from py_recipes import (
    recipe, step_normalize, step_dummy, step_interact,
    step_poly, step_pca, all_numeric, all_nominal
)

# Define preprocessing recipes
rec1 = (recipe()
    .step_normalize(all_numeric()))

rec2 = (recipe()
    .step_normalize(all_numeric())
    .step_dummy(all_nominal()))

rec3 = (recipe()
    .step_normalize(all_numeric())
    .step_poly("x1", "x2", degree=2)
    .step_dummy(all_nominal()))

rec4 = (recipe()
    .step_normalize(all_numeric())
    .step_interact(["x1", "x2"])
    .step_dummy(all_nominal()))

rec5 = (recipe()
    .step_normalize(all_numeric())
    .step_pca(all_numeric(), num_comp=3)
    .step_dummy(all_nominal()))

# Models
models = [
    linear_reg(),
    rand_forest().set_mode('regression')
]

# Create workflow set
wf_set = WorkflowSet.from_cross(
    preproc=[rec1, rec2, rec3, rec4, rec5],
    models=models,
    ids=["normalized", "dummy", "poly", "interact", "pca"]
)

# Evaluate
results = wf_set.fit_resamples(folds, metrics=metrics)

# Compare preprocessing strategies
ranked = results.rank_results("rmse", n=10)
print(ranked[['rank', 'wflow_id', 'preprocessor', 'model', 'rmse_mean']])
```

---

### Pattern 5: Hyperparameter Tuning with WorkflowSets

**Scenario:** Tune hyperparameters across multiple workflows.

```python
from py_tune import tune, grid_regular, finalize_workflow

# Define tunable models
models = [
    linear_reg(penalty=tune(), mixture=tune()),
    rand_forest(trees=tune(), min_n=tune()).set_mode('regression')
]

formulas = ["y ~ x1 + x2", "y ~ x1 + x2 + x3"]

wf_set = WorkflowSet.from_cross(formulas, models)

# Define grids per model
grids = {
    # Linear regression grids
    "prep_1_linear_reg_1": grid_regular({
        "penalty": {"range": (0.001, 1.0), "trans": "log"},
        "mixture": {"range": (0, 1)}
    }, levels=5),
    "prep_2_linear_reg_1": grid_regular({
        "penalty": {"range": (0.001, 1.0), "trans": "log"},
        "mixture": {"range": (0, 1)}
    }, levels=5),

    # Random forest grids
    "prep_1_rand_forest_2": grid_regular({
        "trees": {"range": (100, 1000)},
        "min_n": {"range": (2, 20)}
    }, levels=5),
    "prep_2_rand_forest_2": grid_regular({
        "trees": {"range": (100, 1000)},
        "min_n": {"range": (2, 20)}
    }, levels=5),
}

# Tune all workflows
results = wf_set.tune_grid(
    resamples=folds,
    grid=grids,
    metrics=metrics
)

# Find best configurations
best_configs = results.rank_results("rmse", n=5)
print(best_configs)

# Finalize best workflow with best parameters
best_wf_id = best_configs.iloc[0]["wflow_id"]
best_wf = wf_set[best_wf_id]

# Get best parameters from TuneResults
tune_result = next(r["tune_results"] for r in results.results
                   if r["wflow_id"] == best_wf_id)
best_params = tune_result.select_best("rmse")

# Finalize workflow
final_wf = finalize_workflow(best_wf, best_params)
final_fit = final_wf.fit(train_data)
```

---

### Pattern 6: Time Series Workflow Comparison

**Scenario:** Compare time series models with different preprocessing.

```python
from py_parsnip import arima_reg, prophet_reg, exp_smoothing
from py_rsample import time_series_cv

# Time series specific formulas
formulas = [
    "y ~ date",
    "y ~ date + x1",
    "y ~ date + x1 + x2"
]

# Time series models
models = [
    arima_reg(non_seasonal_ar=1, non_seasonal_differences=1, non_seasonal_ma=1),
    prophet_reg(),
    exp_smoothing(seasonal_periods=12)
]

# Create workflow set
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["date_only", "with_x1", "with_x1_x2"]
)

# Time series CV
ts_folds = time_series_cv(
    data,
    initial="2 years",
    assess="3 months",
    cumulative=True
)

# Time series metrics
ts_metrics = metric_set(rmse, mae, mape, smape)

# Evaluate
results = wf_set.fit_resamples(ts_folds, metrics=ts_metrics)

# Compare
ranked = results.rank_results("mape", n=5)
print(ranked[['rank', 'wflow_id', 'model', 'mape_mean', 'rmse_mean']])
```

---

### Pattern 7: Explicit Workflow List with Custom IDs

**Scenario:** Create workflows with full control over each configuration.

```python
# Create explicit workflows
workflows = [
    ("baseline_ols",
     workflow()
        .add_formula("y ~ x1 + x2")
        .add_model(linear_reg())),

    ("baseline_ridge",
     workflow()
        .add_formula("y ~ x1 + x2")
        .add_model(linear_reg(penalty=0.1, mixture=0.0))),

    ("extended_ols",
     workflow()
        .add_formula("y ~ x1 + x2 + x3 + x4")
        .add_model(linear_reg())),

    ("interaction_elastic",
     workflow()
        .add_formula("y ~ x1 + x2 + I(x1*x2)")
        .add_model(linear_reg(penalty=0.1, mixture=0.5))),

    ("recipe_pca_rf",
     workflow()
        .add_recipe(recipe().step_normalize(all_numeric()).step_pca(all_numeric(), num_comp=5))
        .add_model(rand_forest(trees=500).set_mode('regression'))),
]

# Create workflow set from list
wf_set = WorkflowSet.from_workflows(workflows)

# Evaluate
results = wf_set.fit_resamples(folds, metrics=metrics)
ranked = results.rank_results("rmse", n=5)
```

---

### Pattern 8: Prediction Analysis Across Workflows

**Scenario:** Compare prediction patterns across multiple workflows.

```python
# Evaluate with predictions saved
results = wf_set.fit_resamples(
    folds,
    metrics=metrics,
    control={'save_pred': True}
)

# Collect all predictions
preds_df = results.collect_predictions()

# Analyze by workflow
for wf_id in wf_set:
    wf_preds = preds_df[preds_df['wflow_id'] == wf_id].copy()

    # Calculate residuals
    wf_preds['residual'] = wf_preds['y'] - wf_preds['.pred']
    wf_preds['abs_residual'] = wf_preds['residual'].abs()

    # Summary statistics
    print(f"\n{wf_id}:")
    print(f"  Mean Abs Residual: {wf_preds['abs_residual'].mean():.4f}")
    print(f"  Max Abs Residual: {wf_preds['abs_residual'].max():.4f}")

    # Residual plot
    plt.figure(figsize=(8, 4))
    plt.scatter(wf_preds['.pred'], wf_preds['residual'], alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals: {wf_id}")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()

# Ensemble predictions (average across workflows)
ensemble = (preds_df.groupby(['.row', 'id'])
            .agg({'.pred': 'mean', 'y': 'first'})
            .reset_index())

# Evaluate ensemble
from py_yardstick import rmse
ensemble_rmse = rmse(ensemble['y'].values, ensemble['.pred'].values)
print(f"\nEnsemble RMSE: {ensemble_rmse.iloc[0]['value']:.4f}")
```

---

### Pattern 9: Progressive Model Selection

**Scenario:** Systematically narrow down from many models to the best few.

```python
# Stage 1: Broad screening (50+ workflows)
formulas_broad = [
    "y ~ x1", "y ~ x2", "y ~ x3",
    "y ~ x1 + x2", "y ~ x1 + x3", "y ~ x2 + x3",
    "y ~ x1 + x2 + x3",
    "y ~ x1 + x2 + x3 + x4 + x5"
]

models_broad = [
    linear_reg(),
    linear_reg(penalty=0.1, mixture=1.0),
    rand_forest(trees=100).set_mode('regression'),
    decision_tree(tree_depth=5).set_mode('regression')
]

wf_set_broad = WorkflowSet.from_cross(formulas_broad, models_broad)

# Quick evaluation (fewer folds)
folds_quick = vfold_cv(data, v=3, seed=42)
results_broad = wf_set_broad.fit_resamples(folds_quick, metrics=metrics)

# Select top 10
top10 = results_broad.rank_results("rmse", n=10)
print("Stage 1 - Top 10 workflows:")
print(top10[['rank', 'wflow_id', 'rmse_mean']])

# Stage 2: Refine top candidates (more thorough evaluation)
top10_ids = top10['wflow_id'].tolist()
refined_workflows = [(wf_id, wf_set_broad[wf_id]) for wf_id in top10_ids]

wf_set_refined = WorkflowSet.from_workflows(refined_workflows)

# More thorough evaluation
folds_thorough = vfold_cv(data, v=10, seed=42)
results_refined = wf_set_refined.fit_resamples(folds_thorough, metrics=metrics)

# Final ranking
final_ranking = results_refined.rank_results("rmse", n=5)
print("\nStage 2 - Final top 5:")
print(final_ranking[['rank', 'wflow_id', 'rmse_mean', 'rmse_std']])

# Stage 3: Hyperparameter tuning on best workflow
best_wf_id = final_ranking.iloc[0]['wflow_id']
best_wf = wf_set_refined[best_wf_id]
# ... continue with tuning ...
```

---

## Integration with Other Modules

### Integration with py_rsample

**WorkflowSets work with all resampling strategies:**

```python
from py_rsample import (
    vfold_cv,           # Standard k-fold CV
    time_series_cv,     # Time series CV
    initial_split,      # Simple train/test split
    bootstraps          # Bootstrap resampling (if available)
)

# Standard CV
folds = vfold_cv(data, v=10, seed=42, stratify="outcome")
results = wf_set.fit_resamples(folds, metrics=metrics)

# Time series CV
ts_folds = time_series_cv(
    data,
    initial="1 year",
    assess="3 months",
    skip="1 month",
    cumulative=True
)
results = wf_set.fit_resamples(ts_folds, metrics=metrics)

# Repeated CV
folds = vfold_cv(data, v=5, repeats=3, seed=42)
results = wf_set.fit_resamples(folds, metrics=metrics)
```

---

### Integration with py_yardstick

**WorkflowSets work with all metric types:**

```python
from py_yardstick import (
    # Regression metrics
    rmse, mae, mape, smape, r_squared, rsq_trad,

    # Classification metrics
    accuracy, precision, recall, f1, auc, log_loss,

    # Metric set creation
    metric_set
)

# Regression metrics
reg_metrics = metric_set(rmse, mae, r_squared, mape)
results = wf_set.fit_resamples(folds, metrics=reg_metrics)

# Classification metrics
clf_metrics = metric_set(accuracy, precision, recall, f1, auc)
results = wf_set.fit_resamples(folds, metrics=clf_metrics)

# Custom metric selection
custom_metrics = metric_set(rmse, mae)  # Minimal set
results = wf_set.fit_resamples(folds, metrics=custom_metrics)
```

---

### Integration with py_tune

**WorkflowSets integrate seamlessly with hyperparameter tuning:**

```python
from py_tune import (
    tune,              # Mark parameters for tuning
    grid_regular,      # Regular grid
    grid_random,       # Random grid
    finalize_workflow  # Apply best parameters
)

# Example: Tune multiple workflows
# 1. Create workflows with tunable parameters
spec1 = linear_reg(penalty=tune(), mixture=tune())
spec2 = rand_forest(trees=tune(), min_n=tune()).set_mode('regression')

wf1 = workflow().add_formula("y ~ .").add_model(spec1)
wf2 = workflow().add_formula("y ~ .").add_model(spec2)

wf_set = WorkflowSet.from_workflows([
    ("elastic_net", wf1),
    ("random_forest", wf2)
])

# 2. Create grids
grids = {
    "elastic_net": grid_regular({
        "penalty": {"range": (0.001, 1.0), "trans": "log"},
        "mixture": {"range": (0, 1)}
    }, levels=5),
    "random_forest": grid_regular({
        "trees": {"range": (100, 1000)},
        "min_n": {"range": (2, 20)}
    }, levels=5)
}

# 3. Tune all workflows
results = wf_set.tune_grid(folds, grid=grids, metrics=metrics)

# 4. Extract best configurations
for result in results.results:
    wf_id = result['wflow_id']
    tune_results = result['tune_results']

    best_params = tune_results.select_best("rmse")
    print(f"{wf_id} best params:")
    print(best_params)

    # Finalize this workflow
    final_wf = finalize_workflow(wf_set[wf_id], best_params)
    final_fit = final_wf.fit(train_data)
```

---

### Integration with py_recipes

**WorkflowSets work with recipe preprocessing:**

```python
from py_recipes import (
    recipe,
    step_normalize, step_dummy, step_interact, step_poly,
    step_pca, step_filter_missing, step_impute_median,
    all_numeric, all_nominal, all_predictors
)

# Create recipe pipelines
recipes = [
    recipe()
        .step_impute_median(all_numeric())
        .step_normalize(all_numeric()),

    recipe()
        .step_impute_median(all_numeric())
        .step_normalize(all_numeric())
        .step_dummy(all_nominal()),

    recipe()
        .step_impute_median(all_numeric())
        .step_normalize(all_numeric())
        .step_poly(all_numeric(), degree=2)
        .step_dummy(all_nominal()),

    recipe()
        .step_impute_median(all_numeric())
        .step_normalize(all_numeric())
        .step_pca(all_numeric(), num_comp=5)
        .step_dummy(all_nominal())
]

# Models
models = [
    linear_reg(),
    rand_forest().set_mode('regression'),
    svm_rbf()
]

# Create workflow set
wf_set = WorkflowSet.from_cross(
    preproc=recipes,
    models=models,
    ids=["baseline", "dummy", "poly", "pca"]
)

# Evaluate (4 recipes × 3 models = 12 workflows)
results = wf_set.fit_resamples(folds, metrics=metrics)
```

---

### Integration with py_workflows

**WorkflowSets are built on top of py_workflows:**

```python
from py_workflows import workflow

# Manual workflow creation
wf1 = (workflow()
    .add_formula("y ~ x1 + x2")
    .add_model(linear_reg()))

wf2 = (workflow()
    .add_recipe(recipe().step_normalize(all_numeric()))
    .add_model(rand_forest().set_mode('regression')))

# Add to WorkflowSet
wf_set = WorkflowSet.from_workflows([
    ("formula_linear", wf1),
    ("recipe_rf", wf2)
])

# All workflow methods available
for wf_id in wf_set:
    wf = wf_set[wf_id]

    # Workflow methods
    fitted = wf.fit(train_data)
    preds = fitted.predict(test_data)
    eval_results = fitted.evaluate(test_data)
```

---

## Best Practices

### 1. Naming Conventions

**Use descriptive IDs that convey information:**

```python
# GOOD: Descriptive IDs
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["baseline", "extended", "interaction", "polynomial"]
)

# AVOID: Generic IDs
wf_set = WorkflowSet.from_cross(
    preproc=formulas,
    models=models,
    ids=["prep_1", "prep_2", "prep_3", "prep_4"]
)
```

### 2. Start Simple, Iterate

**Progressive refinement strategy:**

```python
# Stage 1: Quick screening with fewer folds
folds_quick = vfold_cv(data, v=3)
results_quick = wf_set.fit_resamples(folds_quick, metrics)
top10 = results_quick.rank_results("rmse", n=10)

# Stage 2: Thorough evaluation of top candidates
top_workflows = [wf_set[wf_id] for wf_id in top10['wflow_id']]
wf_set_refined = WorkflowSet.from_workflows(
    [(wf_id, wf) for wf_id, wf in zip(top10['wflow_id'], top_workflows)]
)

folds_thorough = vfold_cv(data, v=10)
results_final = wf_set_refined.fit_resamples(folds_thorough, metrics)
```

### 3. Use Appropriate Metrics

**Match metrics to your problem:**

```python
# Regression: Standard metrics
reg_metrics = metric_set(rmse, mae, r_squared)

# Time series: Include percentage errors
ts_metrics = metric_set(rmse, mae, mape, smape)

# Imbalanced classification: Use F1, precision, recall
clf_metrics = metric_set(f1, precision, recall, accuracy)

# Multiple objectives: Track all relevant metrics
multi_metrics = metric_set(rmse, mae, r_squared, mape)
```

### 4. Save Predictions Selectively

**Only save predictions when needed (memory intensive):**

```python
# For final evaluation - save predictions
results = wf_set.fit_resamples(
    folds,
    metrics=metrics,
    control={'save_pred': True}
)
preds_df = results.collect_predictions()

# For screening - don't save predictions
results = wf_set.fit_resamples(folds, metrics=metrics)
```

### 5. Document Workflow Decisions

**Keep track of what you tried:**

```python
# Create workflow set with documentation
formulas = {
    "baseline": "y ~ x1 + x2",              # Initial hypothesis
    "extended": "y ~ x1 + x2 + x3",         # Add seasonality
    "interaction": "y ~ x1 + x2 + I(x1*x2)", # Test interaction
}

wf_set = WorkflowSet.from_cross(
    preproc=list(formulas.values()),
    models=models,
    ids=list(formulas.keys())
)

# Save results with context
results_summary = results.rank_results("rmse", n=10)
results_summary.to_csv("model_comparison_2024-11-09.csv", index=False)

# Document in notebook/script
"""
Model Comparison Results - 2024-11-09
=====================================
Best workflow: interaction_linear_reg_1
CV RMSE: 1.158 ± 0.108
Test RMSE: 1.174

Key findings:
- Interaction term improved RMSE by 8.7%
- Polynomial features didn't help (overfitting)
- Regularization (Lasso) slightly worse than OLS
"""
```

### 6. Validate on Held-Out Test Set

**Always confirm CV results on test data:**

```python
from py_rsample import initial_split, training, testing

# Split data
split = initial_split(data, prop=0.75)
train = training(split)
test = testing(split)

# Cross-validate on training data only
folds = vfold_cv(train, v=5)
results = wf_set.fit_resamples(folds, metrics=metrics)

# Get best workflow
best_wf_id = results.rank_results("rmse").iloc[0]['wflow_id']
best_wf = wf_set[best_wf_id]

# Fit on full training data
final_fit = best_wf.fit(train)

# Evaluate on test set
test_preds = final_fit.predict(test)
test_rmse = rmse(test['y'].values, test_preds['.pred'].values)

print(f"CV RMSE: {results.rank_results('rmse').iloc[0]['rmse_mean']:.4f}")
print(f"Test RMSE: {test_rmse.iloc[0]['value']:.4f}")
```

### 7. Visualize Multiple Metrics

**Compare workflows across multiple dimensions:**

```python
# Create comparison plots
metrics_list = ['rmse', 'mae', 'r_squared']

fig, axes = plt.subplots(1, len(metrics_list), figsize=(18, 6))

for i, metric in enumerate(metrics_list):
    results.autoplot(metric, top_n=10)
    plt.sca(axes[i])

plt.tight_layout()
plt.savefig("multi_metric_comparison.png", dpi=300)
plt.show()
```

### 8. Handle Different Model Types Appropriately

**Be aware of model-specific requirements:**

```python
# Classification models need mode set
clf_models = [
    rand_forest(trees=500).set_mode('classification'),
    decision_tree(tree_depth=10).set_mode('classification'),
    svm_rbf().set_mode('classification')
]

# Time series models need date column
ts_formulas = ["y ~ date", "y ~ date + x1"]
ts_models = [arima_reg(), prophet_reg()]

# Some models require specific data formats
# (e.g., prophet needs 'ds' and 'y' columns)
```

---

## Common Patterns

### Pattern: Comparing Regularization Strengths

```python
# Create grid of regularization strengths
penalties = [0.001, 0.01, 0.1, 1.0, 10.0]
mixtures = [0.0, 0.25, 0.5, 0.75, 1.0]  # Ridge to Lasso

models = []
for penalty in penalties:
    for mixture in mixtures:
        models.append(linear_reg(penalty=penalty, mixture=mixture))

wf_set = WorkflowSet.from_cross(
    preproc=["y ~ ."],
    models=models
)

results = wf_set.fit_resamples(folds, metrics=metrics)
ranked = results.rank_results("rmse", n=10)
```

### Pattern: Ensemble via Averaging

```python
# Get top 5 workflows
top5 = results.rank_results("rmse", n=5)
top5_ids = top5['wflow_id'].tolist()

# Fit all on training data
fitted_models = {}
for wf_id in top5_ids:
    wf = wf_set[wf_id]
    fitted_models[wf_id] = wf.fit(train_data)

# Predict with all models
all_preds = []
for wf_id, fitted in fitted_models.items():
    preds = fitted.predict(test_data)
    all_preds.append(preds['.pred'])

# Average predictions
ensemble_pred = pd.DataFrame(all_preds).mean(axis=0)

# Evaluate ensemble
test_y = test_data['y'].values
ensemble_rmse = np.sqrt(np.mean((test_y - ensemble_pred.values)**2))
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
```

### Pattern: Workflow Filtering by Model Type

```python
# Get metrics for all workflows
metrics_df = results.collect_metrics()

# Filter by model type
linear_models = metrics_df[metrics_df['model'] == 'linear_reg']
rf_models = metrics_df[metrics_df['model'] == 'rand_forest']

# Compare
print("Best Linear Model:")
print(linear_models[linear_models['metric'] == 'rmse'].nsmallest(1, 'mean'))

print("\nBest Random Forest:")
print(rf_models[rf_models['metric'] == 'rmse'].nsmallest(1, 'mean'))
```

### Pattern: Export Results for Reporting

```python
# Collect all results
metrics_summary = results.collect_metrics(summarize=True)
ranked_results = results.rank_results("rmse", n=20)

# Export to CSV
metrics_summary.to_csv("workflow_metrics_summary.csv", index=False)
ranked_results.to_csv("workflow_rankings_rmse.csv", index=False)

# Export workflow info
wf_set.info.to_csv("workflow_definitions.csv", index=False)

# Create report
report = {
    'date': '2024-11-09',
    'n_workflows': len(wf_set),
    'n_folds': folds.v,
    'best_workflow': ranked_results.iloc[0]['wflow_id'],
    'best_rmse': ranked_results.iloc[0]['rmse_mean'],
    'best_rmse_std': ranked_results.iloc[0]['rmse_std']
}

import json
with open('workflow_comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

---

## Summary

**py_workflowsets provides:**

1. **Efficient multi-model comparison** via cross-products and explicit lists
2. **Standardized evaluation** across all workflows with consistent metrics
3. **Easy ranking and selection** of best-performing workflows
4. **Automatic visualization** for quick insights
5. **Seamless integration** with all py-tidymodels modules

**Key Classes:**
- **WorkflowSet**: Collection of workflows for comparison
- **WorkflowSetResults**: Results with metrics, predictions, and analysis methods

**Key Methods:**
- `from_cross()`: Create cross-product of preprocessors and models
- `from_workflows()`: Create from explicit list
- `fit_resamples()`: Evaluate without tuning
- `tune_grid()`: Evaluate with hyperparameter tuning
- `collect_metrics()`: Aggregate metrics across workflows
- `rank_results()`: Rank workflows by performance
- `autoplot()`: Automatic visualization

**Typical Workflow:**
1. Define preprocessing strategies and models
2. Create WorkflowSet via `from_cross()` or `from_workflows()`
3. Create resampling folds
4. Evaluate with `fit_resamples()` or `tune_grid()`
5. Analyze with `rank_results()` and `autoplot()`
6. Select best workflow and finalize on full data

---

**End of Reference Document**
