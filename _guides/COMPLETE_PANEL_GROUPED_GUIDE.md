# Complete Panel/Grouped Modeling Reference

**py-tidymodels Panel Data Modeling Guide**

## Table of Contents
1. [Overview](#overview)
2. [Panel Data Concepts](#panel-data-concepts)
3. [Nested Approach (fit_nested)](#nested-approach-fit_nested)
4. [Global Approach (fit_global)](#global-approach-fit_global)
5. [NestedWorkflowFit Class](#nestedworkflowfit-class)
6. [Data Preparation](#data-preparation)
7. [Model Integration](#model-integration)
8. [Outputs and Analysis](#outputs-and-analysis)
9. [Complete Workflow Examples](#complete-workflow-examples)
10. [Decision Framework](#decision-framework)
11. [Best Practices](#best-practices)
12. [Common Patterns](#common-patterns)
13. [Troubleshooting](#troubleshooting)

---

## Overview

**Panel/grouped modeling** enables fitting models for datasets with multiple groups or entities observed over time or across different conditions.

### What is Panel Data?

Panel data contains observations for **multiple entities (groups)** over time or conditions:

- **Retail**: Multiple stores with daily sales (one model per store)
- **Finance**: Multiple customers with transaction histories (one model per customer)
- **Healthcare**: Multiple hospitals with patient outcomes (one model per hospital)
- **Supply Chain**: Multiple warehouses with inventory levels (one model per warehouse)
- **Energy**: Multiple regions with demand patterns (one model per region)

### Key Features

✅ **Unified API**: Same workflow for both nested and global approaches
✅ **Works with any model**: linear_reg, rand_forest, recursive_reg, prophet_reg, etc.
✅ **Three-DataFrame outputs**: All outputs include group column
✅ **Easy comparison**: Compare performance across groups
✅ **Handles evaluation**: Test on held-out data per group

---

## Panel Data Concepts

### Structure Requirements

Panel data must have:
1. **Group column**: Identifier for each entity (store_id, customer_id, region, etc.)
2. **Multiple groups**: At least 2 distinct group values
3. **Observations per group**: Sufficient data for modeling (varies by approach)

**Example Structure:**
```python
import pandas as pd

# Panel data with 3 stores
data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
    'store_id': ['A', 'A', 'B', 'B', 'C', 'C'],
    'sales': [100, 105, 200, 210, 150, 155],
    'temperature': [70, 72, 68, 71, 75, 77]
})

# Group column: store_id
# 3 groups: A, B, C
# 2 time points per group
```

### Two Modeling Approaches

#### 1. Nested (Per-Group) Modeling
- **Fits separate models for each group**
- Each group gets **independent parameters**
- Best when groups have **different patterns**

**Example Scenario:**
```
Store A (Premium): High prices, steep growth, affluent customers
Store B (Standard): Medium prices, steady growth, average customers
Store C (Budget): Low prices, slow growth, price-sensitive customers

→ Nested approach: Each store gets its own model with unique coefficients
```

#### 2. Global Modeling
- **Fits one model for all groups**
- Group ID becomes a **feature** in the model
- Best when groups share **similar patterns** with different levels

**Example Scenario:**
```
Store A, B, C: All follow seasonal patterns, differ only in baseline sales

→ Global approach: Single model with store_id as a categorical feature
```

---

## Nested Approach (fit_nested)

### Method Signature

```python
Workflow.fit_nested(
    data: pd.DataFrame,
    group_col: str
) -> NestedWorkflowFit
```

**Parameters:**
- `data` (pd.DataFrame): Training data with group column
- `group_col` (str): Column name containing group identifiers

**Returns:**
- `NestedWorkflowFit`: Fitted workflow with dict of models per group

**Raises:**
- `ValueError`: If group_col not in data or workflow doesn't have model

### How It Works Internally

```python
# Pseudocode for fit_nested()
1. Extract unique group values from data[group_col]
2. For each group:
   a. Filter data to group subset: group_data = data[data[group_col] == group]
   b. Remove group_col from group_data (it's not a predictor)
   c. For recursive models: Set date as index if present
   d. Fit workflow on group_data: group_fit = self.fit(group_data)
   e. Store in dict: group_fits[group] = group_fit
3. Return NestedWorkflowFit(workflow, group_col, group_fits)
```

### Basic Usage

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Create workflow
wf = (
    workflow()
    .add_formula("sales ~ temperature + humidity")
    .add_model(linear_reg())
)

# Fit nested models (one per store)
nested_fit = wf.fit_nested(data, group_col="store_id")

# Structure
print(f"Number of models: {len(nested_fit.group_fits)}")
print(f"Group IDs: {list(nested_fit.group_fits.keys())}")
```

**Output:**
```
Number of models: 3
Group IDs: ['A', 'B', 'C']
```

### When to Use Nested Approach

✅ **Use nested when:**
- Groups have **fundamentally different patterns**
- You have **sufficient data per group** (50+ observations recommended)
- You want **group-specific parameters** for interpretation
- **Prediction accuracy per group** is critical
- Groups represent **distinct markets/segments**

**Example Scenarios:**
1. **Retail**: Premium vs budget stores (different price sensitivities)
2. **Finance**: High-risk vs low-risk customers (different default patterns)
3. **Healthcare**: Urban vs rural hospitals (different patient demographics)
4. **Energy**: Residential vs commercial regions (different usage patterns)

### Advantages

1. **Maximum flexibility**: Each group has completely independent model
2. **Best accuracy**: Optimizes for each group's specific pattern
3. **Easy interpretation**: Clear per-group coefficients
4. **No information leakage**: Groups don't influence each other
5. **Handles heterogeneity**: Works when groups are very different

### Disadvantages

1. **Requires more data**: Each group needs sufficient observations
2. **Computationally expensive**: Fits N models instead of 1
3. **No information sharing**: Small groups don't benefit from large groups
4. **More parameters**: Can overfit with limited data per group

---

## Global Approach (fit_global)

### Method Signature

```python
Workflow.fit_global(
    data: pd.DataFrame,
    group_col: str
) -> WorkflowFit
```

**Parameters:**
- `data` (pd.DataFrame): Training data with group column
- `group_col` (str): Column name containing group identifiers (used as feature)

**Returns:**
- `WorkflowFit`: Standard fitted workflow (single model)

**Raises:**
- `ValueError`: If group_col not in data, workflow doesn't have model, or invalid formula format

### How It Works Internally

```python
# Pseudocode for fit_global()
1. Check if formula already includes group_col
2. If not, update formula: "y ~ x" → "y ~ x + group_col"
3. Fit single model on all data with updated formula
4. Return standard WorkflowFit (not nested)
```

**Formula Update Logic:**
```python
# Original formula
"sales ~ temperature"

# Updated formula (group_col = "store_id")
"sales ~ temperature + store_id"

# Group becomes a categorical feature with dummy encoding
```

### Basic Usage

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Create workflow
wf = (
    workflow()
    .add_formula("sales ~ temperature + humidity")
    .add_model(linear_reg())
)

# Fit global model (store_id added as feature)
global_fit = wf.fit_global(data, group_col="store_id")

# This is a standard WorkflowFit (not nested)
from py_workflows.workflow import WorkflowFit
print(isinstance(global_fit, WorkflowFit))  # True
```

### Formula Handling

#### Case 1: Formula without group_col

```python
# Original formula
wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())

# After fit_global(data, group_col="store_id")
# Actual formula used: "y ~ x1 + x2 + store_id"
```

#### Case 2: Formula already includes group_col

```python
# Original formula already has store_id
wf = workflow().add_formula("y ~ x1 + store_id").add_model(linear_reg())

# After fit_global(data, group_col="store_id")
# Actual formula used: "y ~ x1 + store_id" (no duplication)
```

#### Case 3: Formula with dot notation

```python
# Original formula uses "." for all columns
wf = workflow().add_formula("y ~ .").add_model(linear_reg())

# After fit_global(data, group_col="store_id")
# Actual formula used: "y ~ ." (group_col already included via ".")
```

### When to Use Global Approach

✅ **Use global when:**
- Groups have **similar patterns** with different levels
- You have **limited data per group** (< 50 observations per group)
- You want to **share information** across groups
- **Computational efficiency** is important
- Groups represent **same market with different locations**

**Example Scenarios:**
1. **Retail**: Franchise stores following same business model (different baselines)
2. **Finance**: Customers in same segment with regional differences
3. **Healthcare**: Hospitals using same protocols (different patient volumes)
4. **Energy**: Regions with similar climate (different population sizes)

### Advantages

1. **Data efficient**: Shares information across all groups
2. **Computationally fast**: Fits only 1 model
3. **Fewer parameters**: Less risk of overfitting
4. **Handles small groups**: Even groups with few observations contribute
5. **Leverages patterns**: Small groups benefit from large groups

### Disadvantages

1. **Less flexible**: All groups share same functional form
2. **May underfit**: Can't capture group-specific patterns beyond levels
3. **Assumes similarity**: Performance degrades if groups are very different
4. **Harder interpretation**: Group effects are categorical coefficients

---

## NestedWorkflowFit Class

### Class Definition

```python
@dataclass
class NestedWorkflowFit:
    """
    Fitted workflow with separate models for each group.

    Attributes:
        workflow: Original Workflow specification
        group_col: Column name containing group identifiers
        group_fits: Dict mapping group values to WorkflowFit objects
    """
    workflow: Workflow
    group_col: str
    group_fits: dict  # {group_value: WorkflowFit}
```

### Accessing Group Models

```python
# Fit nested models
nested_fit = wf.fit_nested(data, group_col="store_id")

# Access specific group's model
store_a_fit = nested_fit.group_fits["A"]
store_b_fit = nested_fit.group_fits["B"]

# Get underlying ModelFit for a group
store_a_model = store_a_fit.extract_fit_parsnip()

# Access fitted sklearn model
sklearn_model = store_a_model.fit_data["model"]
```

### Methods

#### predict()

```python
NestedWorkflowFit.predict(
    new_data: pd.DataFrame,
    type: str = "numeric"
) -> pd.DataFrame
```

Generate predictions for all groups with automatic routing.

**Parameters:**
- `new_data` (pd.DataFrame): New data with group column
- `type` (str): Prediction type
  - `"numeric"`: Point predictions (default)
  - `"conf_int"`: Confidence intervals
  - `"pred_int"`: Prediction intervals (if model supports)

**Returns:**
- `pd.DataFrame`: Predictions with columns:
  - `.pred`: Point predictions
  - `store_id` (or whatever group_col is): Group identifier
  - Additional columns for intervals if type != "numeric"

**How it works:**
```python
# Pseudocode
1. Check group_col exists in new_data
2. For each group in group_fits:
   a. Filter new_data to group subset
   b. Remove group_col from subset
   c. Get predictions from group model: preds = group_fit.predict(subset)
   d. Add group_col back to predictions
   e. Append to list
3. Concatenate all predictions
4. Return combined DataFrame
```

**Example:**
```python
# Predict on test data
predictions = nested_fit.predict(test_data)

# Prediction intervals
predictions_with_intervals = nested_fit.predict(test_data, type="pred_int")
```

#### evaluate()

```python
NestedWorkflowFit.evaluate(
    test_data: pd.DataFrame,
    outcome_col: Optional[str] = None
) -> NestedWorkflowFit
```

Evaluate all group models on test data.

**Parameters:**
- `test_data` (pd.DataFrame): Test data with actual outcomes and group column
- `outcome_col` (str, optional): Name of outcome column (auto-detected if None)

**Returns:**
- `NestedWorkflowFit`: Self for method chaining

**Example:**
```python
# Fit, evaluate, extract in chain
nested_fit = (
    wf.fit_nested(train, "store_id")
    .evaluate(test)
)

outputs, coeffs, stats = nested_fit.extract_outputs()
```

#### extract_outputs()

```python
NestedWorkflowFit.extract_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

Extract comprehensive three-DataFrame outputs for all groups.

**Returns:**
Tuple of (outputs, coefficients, stats) DataFrames:
- **outputs**: Observation-level results with group column
- **coefficients**: Model parameters with group column
- **stats**: Model-level metrics with group column

**Example:**
```python
outputs, coefficients, stats = nested_fit.extract_outputs()

# Filter to specific group
store_a_outputs = outputs[outputs["store_id"] == "A"]

# Compare metrics across groups
test_rmse = stats[
    (stats["metric"] == "rmse") &
    (stats["split"] == "test")
][["store_id", "value"]].sort_values("value")
```

---

## Data Preparation

### Required Structure

```python
# Minimum requirements
data = pd.DataFrame({
    'group_id': [...],      # Group identifier
    'outcome': [...],       # Target variable
    'feature_1': [...],     # Predictors
    'feature_2': [...],
    # ... more features
})

# For time series
data = pd.DataFrame({
    'date': [...],          # Date column (for time series models)
    'store_id': [...],      # Group identifier
    'sales': [...],         # Target variable
    'temperature': [...],   # Predictors
    # ... more features
})
```

### Group Column

**Requirements:**
- Must be present in both train and test data
- Should have consistent naming across datasets
- Can be any data type (string, int, categorical)

**Good Practices:**
```python
# Use descriptive names
group_col = "store_id"      # ✓ Clear
group_col = "customer_id"   # ✓ Clear
group_col = "region"        # ✓ Clear

# Avoid ambiguous names
group_col = "id"            # ✗ Too generic
group_col = "group"         # ✗ Not descriptive
```

### Date Column Handling

#### For Recursive Models

**Critical**: Recursive models (`recursive_reg`) require date as index.

**fit_nested() automatically handles this:**
```python
# Input data
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'store_id': [...],
    'sales': [...]
})

# fit_nested() does this internally for recursive models:
# 1. Filter to group: group_data = data[data['store_id'] == 'A']
# 2. Set date as index: group_data = group_data.set_index('date')
# 3. Remove group_col: group_data = group_data.drop(columns=['store_id'])
# 4. Fit: group_fit = workflow.fit(group_data)
```

**Manual setup (if needed):**
```python
# For recursive models, prepare data with date as index
data_indexed = data.set_index('date')

# Then reset before calling fit_nested (it will re-index per group)
nested_fit = wf.fit_nested(data_indexed.reset_index(), group_col='store_id')
```

#### For Standard Models

**Standard models** (linear_reg, rand_forest, etc.) keep date as a regular column.

```python
# Date stays as column (not index)
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'store_id': [...],
    'sales': [...]
})

# fit_nested() does this internally for standard models:
# 1. Filter to group: group_data = data[data['store_id'] == 'A']
# 2. Remove group_col: group_data = group_data.drop(columns=['store_id'])
# 3. Fit: group_fit = workflow.fit(group_data)
# (Date remains as column, can be used in formula)
```

### Missing Groups in Test Data

**Behavior:**
- `predict()`: Skips groups not in training data (raises error if NO groups match)
- `evaluate()`: Skips groups not in training data

**Example:**
```python
# Train has groups: A, B, C
nested_fit = wf.fit_nested(train, group_col='store_id')

# Test has groups: A, B, D
# Predictions returned for A and B only
# Group D raises warning or is skipped
predictions = nested_fit.predict(test)  # Works for A, B
```

**Best Practice:**
```python
# Ensure test groups are subset of train groups
train_groups = set(train['store_id'].unique())
test_groups = set(test['store_id'].unique())

assert test_groups.issubset(train_groups), \
    f"Test has unseen groups: {test_groups - train_groups}"
```

### Unbalanced Panels

**Unbalanced panels** have different numbers of observations per group.

**Handling:**
- **Nested approach**: Works fine (each group fitted independently)
- **Global approach**: Works fine (group imbalance handled by model)

**Example:**
```python
data = pd.DataFrame({
    'store_id': ['A']*100 + ['B']*50 + ['C']*200,  # Unbalanced
    'sales': [...],
    'temperature': [...]
})

# Both work without issues
nested_fit = wf.fit_nested(data, group_col='store_id')    # ✓
global_fit = wf.fit_global(data, group_col='store_id')    # ✓
```

**Considerations:**
- **Nested**: Groups with few observations may have high variance
- **Global**: Model may be dominated by large groups
- **Solution**: Weight samples or use stratified validation

---

## Model Integration

### Which Models Work?

**All py_parsnip models work with panel modeling:**

#### Linear Models
```python
from py_parsnip import linear_reg, poisson_reg, gen_additive_mod

# Linear regression per group
wf = workflow().add_formula("y ~ x").add_model(linear_reg())
nested_fit = wf.fit_nested(data, group_col='group')
```

#### Tree-Based Models
```python
from py_parsnip import decision_tree, rand_forest, boost_tree

# Random forest per group
wf = workflow().add_formula("y ~ x1 + x2").add_model(
    rand_forest(trees=100).set_mode('regression')
)
nested_fit = wf.fit_nested(data, group_col='group')
```

#### Time Series Models
```python
from py_parsnip import prophet_reg, arima_reg, exp_smoothing

# Prophet per group
wf = workflow().add_formula("y ~ date").add_model(prophet_reg())
nested_fit = wf.fit_nested(data, group_col='group')
```

#### Recursive Forecasting
```python
from py_parsnip import recursive_reg

# Recursive model per group (special date handling)
wf = workflow().add_formula("sales ~ .").add_model(
    recursive_reg(base_model=linear_reg(), lags=7)
)
nested_fit = wf.fit_nested(data, group_col='group')
```

### Special Considerations

#### recursive_reg

**Date indexing is automatic:**
```python
# Input: data with 'date' column
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'store_id': ['A']*50 + ['B']*50,
    'sales': [...]
})

# fit_nested() automatically:
# 1. Checks if model_type == "recursive_reg"
# 2. Sets date as index per group
# 3. Removes group_col before fitting

wf = workflow().add_formula("sales ~ .").add_model(
    recursive_reg(base_model=linear_reg(), lags=7)
)

# Just call fit_nested - date handling is automatic
nested_fit = wf.fit_nested(data, group_col='store_id')
```

#### prophet_reg and arima_reg

**Work seamlessly with panel data:**
```python
# Prophet per store
wf = workflow().add_formula("sales ~ date").add_model(
    prophet_reg(
        seasonality_yearly=True,
        seasonality_weekly=True,
        changepoint_prior_scale=0.05
    )
)

nested_fit = wf.fit_nested(data, group_col='store_id')

# ARIMA per store
wf = workflow().add_formula("sales ~ date").add_model(
    arima_reg(non_seasonal_ar=2, non_seasonal_differences=1, non_seasonal_ma=1)
)

nested_fit = wf.fit_nested(data, group_col='store_id')
```

#### varmax_reg (Multivariate)

**Note**: VARMAX requires 2+ outcome variables, which complicates panel setup.

**Not recommended for panel modeling** - use nested univariate models instead:
```python
# Instead of VARMAX per group (complex):
# Use nested ARIMA or Prophet per outcome per group
```

#### boost_tree, svm_*, nearest_neighbor

**Work perfectly with both nested and global:**
```python
# XGBoost per group
wf = workflow().add_formula("y ~ .").add_model(
    boost_tree(trees=100, tree_depth=6).set_mode('regression').set_engine('xgboost')
)
nested_fit = wf.fit_nested(data, group_col='group')

# SVM global model
wf = workflow().add_formula("y ~ .").add_model(
    svm_rbf().set_mode('regression')
)
global_fit = wf.fit_global(data, group_col='group')
```

### Model Mode Setting

**Critical**: Some models require explicit mode setting.

```python
# Models that need .set_mode('regression') or .set_mode('classification')
rand_forest(trees=100).set_mode('regression')
decision_tree(tree_depth=10).set_mode('classification')
svm_rbf().set_mode('regression')
nearest_neighbor(neighbors=5).set_mode('regression')
mlp(hidden_units=50, epochs=100).set_mode('regression')

# Models with default mode (no set_mode needed)
linear_reg()             # Always regression
prophet_reg()            # Always regression
recursive_reg(...)       # Always regression
```

---

## Outputs and Analysis

### Three-DataFrame Pattern

All panel models return **three DataFrames** with the **group column** included.

#### 1. Outputs DataFrame

**Observation-level results:**

```python
outputs, coeffs, stats = nested_fit.extract_outputs()

print(outputs.columns)
# ['actuals', 'fitted', 'forecast', 'residuals', 'split',
#  'model', 'model_group_name', 'group', 'store_id']
```

**Columns:**
- `actuals`: True values from data
- `fitted`: Model predictions
- `forecast`: Combined actual/fitted (actuals where available, fitted otherwise)
- `residuals`: actuals - fitted
- `split`: 'train', 'test', or 'forecast'
- `model`: Model type (e.g., 'linear_reg')
- `model_group_name`: Model grouping (usually 'global')
- `group`: Model tracking column
- `store_id` (or whatever group_col is): **Group identifier**

**Usage:**
```python
# Filter to specific group
store_a_outputs = outputs[outputs['store_id'] == 'A']

# Filter to test split
test_outputs = outputs[outputs['split'] == 'test']

# Group-specific residuals
store_a_test_residuals = outputs[
    (outputs['store_id'] == 'A') &
    (outputs['split'] == 'test')
]['residuals']
```

#### 2. Coefficients DataFrame

**Variable-level parameters:**

```python
print(coefficients.columns)
# ['variable', 'coefficient', 'std_error', 't_stat', 'p_value',
#  'ci_0.025', 'ci_0.975', 'vif', 'model', 'model_group_name',
#  'group', 'store_id']
```

**Columns:**
- `variable`: Predictor name (e.g., 'temperature', 'Intercept')
- `coefficient`: Parameter estimate
- `std_error`: Standard error (if available)
- `t_stat`: t-statistic (if available)
- `p_value`: p-value (if available)
- `ci_0.025`, `ci_0.975`: 95% confidence interval bounds
- `vif`: Variance Inflation Factor (multicollinearity check)
- `model`, `model_group_name`, `group`: Model tracking
- `store_id`: **Group identifier**

**Usage:**
```python
# Compare coefficients across groups
temp_coefs = coefficients[coefficients['variable'] == 'temperature']
print(temp_coefs[['store_id', 'coefficient', 'p_value']])

# Find significant predictors per group
sig_coefs = coefficients[coefficients['p_value'] < 0.05]
print(sig_coefs.groupby('store_id')['variable'].apply(list))
```

#### 3. Stats DataFrame

**Model-level metrics:**

```python
print(stats.columns)
# ['metric', 'value', 'split', 'model', 'model_group_name',
#  'group', 'store_id']
```

**Columns:**
- `metric`: Metric name (e.g., 'rmse', 'mae', 'r_squared')
- `value`: Metric value
- `split`: 'train', 'test', or blank (overall)
- `model`, `model_group_name`, `group`: Model tracking
- `store_id`: **Group identifier**

**Available Metrics:**
- **Regression**: rmse, mae, mape, smape, r_squared, adj_r_squared, rse
- **Classification**: accuracy, precision, recall, f1, specificity
- **Model Info**: formula, model_type, model_class, n_obs_train, n_obs_test
- **Residual Diagnostics**: durbin_watson, ljung_box_p, jarque_bera_p, breusch_pagan_p

**Usage:**
```python
# Compare test RMSE across groups
test_rmse = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'] == 'test')
][['store_id', 'value']].sort_values('value')

# Get R² for all groups
r2_by_group = stats[
    (stats['metric'] == 'r_squared') &
    (stats['split'] == 'train')
].pivot(index='store_id', columns='split', values='value')
```

### Per-Group Metrics

**Extract metrics for each group:**

```python
outputs, coeffs, stats = nested_fit.extract_outputs()

# Method 1: Filter stats DataFrame
for group in ['A', 'B', 'C']:
    group_stats = stats[
        (stats['store_id'] == group) &
        (stats['split'] == 'test')
    ]

    rmse = group_stats[group_stats['metric'] == 'rmse']['value'].values[0]
    mae = group_stats[group_stats['metric'] == 'mae']['value'].values[0]

    print(f"Store {group}: RMSE={rmse:.4f}, MAE={mae:.4f}")

# Method 2: Pivot for comparison
test_metrics = stats[
    (stats['split'] == 'test') &
    (stats['metric'].isin(['rmse', 'mae', 'r_squared']))
]

comparison = test_metrics.pivot_table(
    index='store_id',
    columns='metric',
    values='value'
)

print(comparison)
```

### Aggregate Metrics

**Compute overall performance:**

```python
# Average RMSE across all groups
avg_rmse = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'] == 'test')
]['value'].mean()

# Weighted average (by group size)
group_sizes = outputs.groupby('store_id').size()
weighted_rmse = (
    stats[
        (stats['metric'] == 'rmse') &
        (stats['split'] == 'test')
    ].merge(group_sizes.rename('n'), left_on='store_id', right_index=True)
)
weighted_rmse['weighted_value'] = weighted_rmse['value'] * weighted_rmse['n']
overall_rmse = weighted_rmse['weighted_value'].sum() / weighted_rmse['n'].sum()

print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Weighted RMSE: {overall_rmse:.4f}")
```

### Comparing Groups

**Identify best/worst performing groups:**

```python
# Rank groups by test RMSE
test_rmse_ranked = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'] == 'test')
][['store_id', 'value']].sort_values('value')

print("Groups ranked by RMSE (lower is better):")
print(test_rmse_ranked)

# Find groups needing attention
high_error_groups = test_rmse_ranked[test_rmse_ranked['value'] > threshold]
print(f"\nGroups with RMSE > {threshold}:")
print(high_error_groups['store_id'].tolist())
```

---

## Complete Workflow Examples

### Example 1: Basic Nested Linear Models

```python
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg

# Generate panel data (3 stores, 120 days each)
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=120)

data = pd.concat([
    pd.DataFrame({
        'date': dates,
        'store_id': 'A',
        'sales': np.linspace(200, 300, 120) + np.random.normal(0, 8, 120)
    }),
    pd.DataFrame({
        'date': dates,
        'store_id': 'B',
        'sales': np.linspace(150, 200, 120) + np.random.normal(0, 6, 120)
    }),
    pd.DataFrame({
        'date': dates,
        'store_id': 'C',
        'sales': np.linspace(80, 110, 120) + np.random.normal(0, 4, 120)
    })
], ignore_index=True)

# Add time variable
data['time'] = data.groupby('store_id').cumcount()

# Train/test split
train = data[data['time'] < 100]
test = data[data['time'] >= 100]

# Create workflow
wf = workflow().add_formula("sales ~ time").add_model(linear_reg())

# Fit nested models
nested_fit = wf.fit_nested(train, group_col='store_id')

# Predict
predictions = nested_fit.predict(test)

# Evaluate
nested_fit = nested_fit.evaluate(test)

# Extract outputs
outputs, coeffs, stats = nested_fit.extract_outputs()

# Compare test RMSE
test_rmse = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'] == 'test')
][['store_id', 'value']]

print("Test RMSE by Store:")
print(test_rmse)
```

### Example 2: Global Model Comparison

```python
# Same data as Example 1

# Create workflow
wf_global = workflow().add_formula("sales ~ time").add_model(linear_reg())

# Fit global model (store_id added as feature)
global_fit = wf_global.fit_global(train, group_col='store_id')

# Evaluate
global_fit = global_fit.evaluate(test)

# Extract outputs
outputs_global, coeffs_global, stats_global = global_fit.extract_outputs()

# Get coefficients (now includes store_id dummies)
print("Global Model Coefficients:")
print(coeffs_global[['variable', 'coefficient']])

# Example output:
#         variable  coefficient
# 0      Intercept     0.000000
# 1  store_id[T.B]   -74.921082  # Effect of being store B vs A
# 2  store_id[T.C]  -154.327527  # Effect of being store C vs A
# 3           time     0.519394  # Common time trend

# Get test RMSE (overall)
global_rmse = stats_global[
    (stats_global['metric'] == 'rmse') &
    (stats_global['split'] == 'test')
]['value'].values[0]

print(f"Global Model Test RMSE: {global_rmse:.4f}")
```

### Example 3: Nested Random Forest

```python
from py_parsnip import rand_forest

# Create workflow with Random Forest
wf_rf = (
    workflow()
    .add_formula("sales ~ time")
    .add_model(rand_forest(trees=100, min_n=5).set_mode('regression'))
)

# Fit nested models
nested_rf_fit = wf_rf.fit_nested(train, group_col='store_id')

# Evaluate
nested_rf_fit = nested_rf_fit.evaluate(test)

# Extract outputs
outputs_rf, coeffs_rf, stats_rf = nested_rf_fit.extract_outputs()

# Feature importances (for tree models, "coefficients" are importances)
print("Feature Importances by Store:")
print(coeffs_rf[['store_id', 'variable', 'coefficient']])

# Example output:
#   store_id variable  coefficient
# 0        A     time          1.0  # Time is only feature, so importance = 1.0
# 1        B     time          1.0
# 2        C     time          1.0

# Compare test RMSE: Linear vs Random Forest
rf_rmse = stats_rf[
    (stats_rf['metric'] == 'rmse') &
    (stats_rf['split'] == 'test')
][['store_id', 'value']]

print("\nRandom Forest Test RMSE:")
print(rf_rmse)
```

### Example 4: Recursive Forecasting per Group

```python
from py_parsnip import recursive_reg

# Prepare data with date as index
data_for_recursive = data.copy()

# Create workflow with recursive model
wf_recursive = (
    workflow()
    .add_formula("sales ~ .")
    .add_model(recursive_reg(base_model=linear_reg(), lags=7))
)

# Fit nested recursive models
# fit_nested() handles date indexing automatically
nested_recursive_fit = wf_recursive.fit_nested(train, group_col='store_id')

# Predict future values
predictions_recursive = nested_recursive_fit.predict(test)

# Evaluate
nested_recursive_fit = nested_recursive_fit.evaluate(test)

# Extract outputs
outputs_rec, coeffs_rec, stats_rec = nested_recursive_fit.extract_outputs()

# Show lag coefficients for each store
print("Lag Coefficients by Store:")
lag_coefs = coeffs_rec[coeffs_rec['variable'].str.contains('lag_')]
print(lag_coefs[['store_id', 'variable', 'coefficient']])

# Example output:
#   store_id variable  coefficient
# 0        A    lag_1    -0.041636  # Effect of lag 1 on store A
# 1        A    lag_2    -0.066337
# ...
# 8        B    lag_1     0.008633  # Different lag effects for store B
# 9        B    lag_2     0.003536
# ...

# Test RMSE
recursive_rmse = stats_rec[
    (stats_rec['metric'] == 'rmse') &
    (stats_rec['split'] == 'test')
][['store_id', 'value']]

print("\nRecursive Model Test RMSE:")
print(recursive_rmse)
```

### Example 5: Time Series with Prophet per Group

```python
from py_parsnip import prophet_reg

# Prepare data with date column
data_prophet = data.copy()

# Create workflow with Prophet
wf_prophet = (
    workflow()
    .add_formula("sales ~ date")
    .add_model(prophet_reg(
        seasonality_yearly=False,  # Not enough data for yearly
        seasonality_weekly=True,
        changepoint_prior_scale=0.05
    ))
)

# Fit nested Prophet models
nested_prophet_fit = wf_prophet.fit_nested(train, group_col='store_id')

# Predict
predictions_prophet = nested_prophet_fit.predict(test)

# Evaluate
nested_prophet_fit = nested_prophet_fit.evaluate(test)

# Extract outputs
outputs_prophet, coeffs_prophet, stats_prophet = nested_prophet_fit.extract_outputs()

# Prophet-specific coefficients (trend, seasonality components)
print("Prophet Hyperparameters by Store:")
print(coeffs_prophet[['store_id', 'variable', 'coefficient']])

# Test RMSE
prophet_rmse = stats_prophet[
    (stats_prophet['metric'] == 'rmse') &
    (stats_prophet['split'] == 'test')
][['store_id', 'value']]

print("\nProphet Test RMSE:")
print(prophet_rmse)
```

### Example 6: Multi-Model Comparison per Group

```python
# Collect test RMSE from all approaches
comparison_data = []

# Nested Linear
for _, row in nested_test_rmse.iterrows():
    comparison_data.append({
        'model': 'Nested Linear',
        'store_id': row['store_id'],
        'rmse': row['value']
    })

# Global Linear (need to compute per-group RMSE manually)
test_with_global_preds = test.copy()
test_with_global_preds['.pred'] = global_fit.predict(test)['.pred'].values
test_with_global_preds['squared_error'] = (
    test_with_global_preds['sales'] - test_with_global_preds['.pred']
) ** 2

global_rmse_by_store = (
    test_with_global_preds
    .groupby('store_id')['squared_error']
    .mean() ** 0.5
)

for store_id, rmse in global_rmse_by_store.items():
    comparison_data.append({
        'model': 'Global Linear',
        'store_id': store_id,
        'rmse': rmse
    })

# Nested Random Forest
for _, row in rf_rmse.iterrows():
    comparison_data.append({
        'model': 'Nested RF',
        'store_id': row['store_id'],
        'rmse': row['value']
    })

# Nested Recursive
for _, row in recursive_rmse.iterrows():
    comparison_data.append({
        'model': 'Nested Recursive',
        'store_id': row['store_id'],
        'rmse': row['value']
    })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_pivot = comparison_df.pivot(
    index='store_id',
    columns='model',
    values='rmse'
)

print("=" * 80)
print("COMPREHENSIVE MODEL COMPARISON - TEST SET RMSE")
print("=" * 80)
print(comparison_pivot)

# Find best model for each store
print("\n" + "=" * 80)
print("BEST MODEL FOR EACH STORE")
print("=" * 80)
for store_id in comparison_pivot.index:
    best_model = comparison_pivot.loc[store_id].idxmin()
    best_rmse = comparison_pivot.loc[store_id].min()
    print(f"Store {store_id}: {best_model:25s} (RMSE: {best_rmse:.4f})")

# Overall best
print("\n" + "=" * 80)
print("AVERAGE RMSE ACROSS ALL STORES")
print("=" * 80)
avg_rmse = comparison_pivot.mean().sort_values()
print(avg_rmse)
print(f"\nBest overall: {avg_rmse.idxmin()} (Avg RMSE: {avg_rmse.min():.4f})")
```

### Example 7: Panel Data with Recipes

```python
from py_recipes import recipe

# Add more features for demonstration
data_enriched = data.copy()
data_enriched['day_of_week'] = data_enriched['date'].dt.dayofweek
data_enriched['month'] = data_enriched['date'].dt.month
data_enriched['is_weekend'] = data_enriched['day_of_week'].isin([5, 6]).astype(int)

train_enriched = data_enriched[data_enriched['time'] < 100]
test_enriched = data_enriched[data_enriched['time'] >= 100]

# Create recipe for preprocessing
rec = (
    recipe()
    .step_normalize(['time'])
    .step_dummy(['day_of_week', 'month'])
)

# Create workflow with recipe
wf_recipe = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

# Fit nested models
nested_recipe_fit = wf_recipe.fit_nested(train_enriched, group_col='store_id')

# Evaluate
nested_recipe_fit = nested_recipe_fit.evaluate(test_enriched)

# Extract outputs
outputs_recipe, coeffs_recipe, stats_recipe = nested_recipe_fit.extract_outputs()

# Show coefficients (now includes all dummy variables)
print("Model Coefficients by Store:")
print(coeffs_recipe[['store_id', 'variable', 'coefficient']].head(20))

# Test RMSE
recipe_rmse = stats_recipe[
    (stats_recipe['metric'] == 'rmse') &
    (stats_recipe['split'] == 'test')
][['store_id', 'value']]

print("\nRecipe-Based Model Test RMSE:")
print(recipe_rmse)
```

### Example 8: Handling Missing Groups

```python
# Train on groups A, B
train_ab = train[train['store_id'].isin(['A', 'B'])]

# Fit nested models
nested_fit_ab = wf.fit_nested(train_ab, group_col='store_id')

# Test includes group C (not in training)
test_abc = test.copy()

# Predict - will skip group C
try:
    predictions_ab = nested_fit_ab.predict(test_abc)
    print("Predictions generated for groups A and B only")
    print(predictions_ab['store_id'].unique())
except ValueError as e:
    print(f"Error: {e}")

# Filter test to only trained groups
test_ab = test[test['store_id'].isin(['A', 'B'])]
predictions_ab = nested_fit_ab.predict(test_ab)
print("Predictions for trained groups:")
print(predictions_ab.head())
```

### Example 9: Large-Scale Panel Data

```python
# Generate large panel (100 stores, 365 days each)
np.random.seed(42)
n_stores = 100
n_days = 365

large_data = []
for i in range(n_stores):
    store_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_days),
        'store_id': f'Store_{i:03d}',
        'sales': np.cumsum(np.random.randn(n_days)) + 100 + i * 10,
        'temperature': 70 + 20 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 5, n_days)
    })
    large_data.append(store_data)

large_data = pd.concat(large_data, ignore_index=True)
large_data['time'] = large_data.groupby('store_id').cumcount()

# Split
large_train = large_data[large_data['time'] < 300]
large_test = large_data[large_data['time'] >= 300]

# Fit nested models (100 models)
import time
start = time.time()

wf_large = workflow().add_formula("sales ~ time + temperature").add_model(linear_reg())
nested_fit_large = wf_large.fit_nested(large_train, group_col='store_id')

elapsed = time.time() - start
print(f"Fitted {len(nested_fit_large.group_fits)} models in {elapsed:.2f} seconds")
print(f"Average time per model: {elapsed / len(nested_fit_large.group_fits):.3f} seconds")

# Evaluate
nested_fit_large = nested_fit_large.evaluate(large_test)

# Extract outputs
outputs_large, coeffs_large, stats_large = nested_fit_large.extract_outputs()

# Distribution of test RMSE across all stores
test_rmse_large = stats_large[
    (stats_large['metric'] == 'rmse') &
    (stats_large['split'] == 'test')
]['value']

print(f"\nTest RMSE Distribution:")
print(f"Mean: {test_rmse_large.mean():.4f}")
print(f"Std: {test_rmse_large.std():.4f}")
print(f"Min: {test_rmse_large.min():.4f}")
print(f"Max: {test_rmse_large.max():.4f}")
print(f"Median: {test_rmse_large.median():.4f}")

# Identify outlier stores
high_error_stores = stats_large[
    (stats_large['metric'] == 'rmse') &
    (stats_large['split'] == 'test') &
    (stats_large['value'] > test_rmse_large.quantile(0.95))
]['store_id']

print(f"\nTop 5% error stores (need attention):")
print(high_error_stores.tolist())
```

### Example 10: Cross-Validation with Panel Data

```python
from py_rsample import time_series_cv

# Create time series CV folds (5 folds)
cv_folds = time_series_cv(
    data,
    date_col='date',
    initial_train='60 days',
    assess='20 days',
    skip='0 days',
    cumulative=True
)

# Fit and evaluate across all folds
cv_results = []

for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
    # Get fold data
    fold_train = data.iloc[train_idx]
    fold_test = data.iloc[test_idx]

    # Fit nested models
    nested_fit_cv = wf.fit_nested(fold_train, group_col='store_id')
    nested_fit_cv = nested_fit_cv.evaluate(fold_test)

    # Extract stats
    _, _, stats_cv = nested_fit_cv.extract_outputs()

    # Add fold index
    stats_cv['fold'] = fold_idx
    cv_results.append(stats_cv)

# Combine all folds
cv_stats = pd.concat(cv_results, ignore_index=True)

# Average RMSE across folds per store
cv_rmse = cv_stats[
    (cv_stats['metric'] == 'rmse') &
    (cv_stats['split'] == 'test')
].groupby('store_id')['value'].agg(['mean', 'std'])

print("Cross-Validated RMSE by Store:")
print(cv_rmse)

# Overall CV performance
overall_cv_rmse = cv_rmse['mean'].mean()
overall_cv_std = cv_rmse['mean'].std()

print(f"\nOverall CV RMSE: {overall_cv_rmse:.4f} ± {overall_cv_std:.4f}")
```

---

## Decision Framework

### Choosing Between Nested and Global

Use this decision tree:

```
START
  |
  ├─ Do groups have FUNDAMENTALLY different patterns?
  |  (e.g., premium vs budget stores)
  |  ├─ YES → Use NESTED
  |  └─ NO → Continue
  |
  ├─ Do you have SUFFICIENT data per group?
  |  (50+ observations per group)
  |  ├─ NO → Use GLOBAL
  |  └─ YES → Continue
  |
  ├─ Is per-group ACCURACY critical?
  |  (e.g., store-level forecasting for inventory)
  |  ├─ YES → Use NESTED
  |  └─ NO → Continue
  |
  ├─ Do you need INTERPRETABLE per-group parameters?
  |  (e.g., understanding store-specific sensitivities)
  |  ├─ YES → Use NESTED
  |  └─ NO → Continue
  |
  ├─ Is COMPUTATIONAL EFFICIENCY important?
  |  (e.g., many groups, limited compute)
  |  ├─ YES → Use GLOBAL
  |  └─ NO → Use NESTED (default for flexibility)
```

### Decision Matrix

| Criterion | Nested | Global | Winner |
|-----------|--------|--------|--------|
| **Groups have different patterns** | ✓✓✓ | ✗ | Nested |
| **Limited data per group (< 50 obs)** | ✗ | ✓✓✓ | Global |
| **Per-group accuracy critical** | ✓✓✓ | ✓ | Nested |
| **Interpretability important** | ✓✓✓ | ✓ | Nested |
| **Computational efficiency** | ✗ | ✓✓✓ | Global |
| **Many groups (> 100)** | ✗ | ✓✓✓ | Global |
| **Unbalanced panels** | ✓✓ | ✓✓ | Tie |
| **Information sharing needed** | ✗ | ✓✓✓ | Global |

### Hybrid Approach

**Best of both worlds:**

```python
# Step 1: Start with nested to understand group differences
nested_fit = wf.fit_nested(train, group_col='store_id')
outputs_nested, coeffs_nested, stats_nested = nested_fit.extract_outputs()

# Analyze coefficient variation across groups
temp_coefs = coeffs_nested[coeffs_nested['variable'] == 'temperature']
coef_std = temp_coefs['coefficient'].std()
coef_mean = temp_coefs['coefficient'].mean()
cv = coef_std / coef_mean  # Coefficient of variation

print(f"Coefficient variation: {cv:.4f}")

# Step 2: If coefficients are similar (low CV), use global
if cv < 0.3:  # Less than 30% variation
    print("Coefficients similar across groups → Using global model")
    global_fit = wf.fit_global(train, group_col='store_id')
    final_fit = global_fit
else:
    print("Coefficients vary across groups → Using nested models")
    final_fit = nested_fit

# Step 3: Use final model
final_fit = final_fit.evaluate(test)
outputs, coeffs, stats = final_fit.extract_outputs()
```

### Sample Size Guidelines

| Observations per Group | Recommendation |
|------------------------|----------------|
| < 30 | **Global only** (nested will overfit) |
| 30-50 | **Global preferred**, nested possible with regularization |
| 50-100 | **Either approach**, test both |
| 100-500 | **Nested preferred**, global if many groups (>100) |
| > 500 | **Nested strongly preferred** |

**Adjustment factors:**
- **Few features** (< 5): Can use nested with less data (divide by 2)
- **Many features** (> 20): Need more data (multiply by 2)
- **Regularized models** (Lasso, Ridge): Can use nested with less data

---

## Best Practices

### 1. Data Preparation

#### Ensure Group Column Consistency
```python
# Before fitting
train_groups = set(train['store_id'].unique())
test_groups = set(test['store_id'].unique())

# Validate test groups
if not test_groups.issubset(train_groups):
    unseen = test_groups - train_groups
    raise ValueError(f"Test has unseen groups: {unseen}")
```

#### Handle Missing Values
```python
# Check for missing values per group
missing_by_group = (
    data.groupby('store_id')
    .apply(lambda x: x.isnull().sum())
)

print("Missing values by group:")
print(missing_by_group)

# Impute or drop
from py_recipes import recipe
rec = recipe().step_impute_median(all_numeric())
```

#### Standardize Group IDs
```python
# Use consistent naming
data['store_id'] = data['store_id'].astype('category')
data['store_id'] = data['store_id'].cat.remove_unused_categories()
```

### 2. Model Selection

#### Start Simple
```python
# 1. Start with linear model
wf_linear = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
nested_linear = wf_linear.fit_nested(train, group_col='group')

# 2. Add complexity if needed
wf_rf = workflow().add_formula("y ~ x1 + x2").add_model(
    rand_forest(trees=100).set_mode('regression')
)
nested_rf = wf_rf.fit_nested(train, group_col='group')

# 3. Compare
_, _, stats_linear = nested_linear.evaluate(test).extract_outputs()
_, _, stats_rf = nested_rf.evaluate(test).extract_outputs()
```

#### Use Appropriate Models
```python
# Time series data → Time series models
if data_has_temporal_patterns:
    models = [prophet_reg(), arima_reg(), recursive_reg(...)]

# Cross-sectional data → Standard ML models
else:
    models = [linear_reg(), rand_forest(), boost_tree()]

# Non-linear patterns → Tree-based models
if non_linear_relationships:
    models = [rand_forest(), boost_tree(), svm_rbf()]
```

### 3. Evaluation Strategy

#### Use Appropriate Metrics
```python
# Regression: RMSE, MAE, MAPE
metrics_regression = ['rmse', 'mae', 'mape', 'r_squared']

# If outcome scale varies across groups, use relative metrics
# (MAPE, SMAPE) instead of absolute metrics (RMSE, MAE)
if group_sales_vary_widely:
    primary_metric = 'mape'
else:
    primary_metric = 'rmse'
```

#### Compare Train vs Test
```python
# Check for overfitting per group
train_test_comparison = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'].isin(['train', 'test']))
].pivot_table(index='store_id', columns='split', values='value')

train_test_comparison['ratio'] = (
    train_test_comparison['test'] / train_test_comparison['train']
)

# Flag groups with high train/test ratio (overfitting)
overfit_groups = train_test_comparison[
    train_test_comparison['ratio'] > 1.5
]

print("Groups with overfitting:")
print(overfit_groups)
```

#### Time Series Validation
```python
# Use time series CV for panel data
from py_rsample import time_series_cv

# Create CV folds
cv_folds = time_series_cv(
    data,
    date_col='date',
    initial_train='60 days',
    assess='20 days',
    cumulative=True
)

# Evaluate across folds
cv_rmse = []
for train_idx, test_idx in cv_folds:
    fold_train = data.iloc[train_idx]
    fold_test = data.iloc[test_idx]

    nested_fit = wf.fit_nested(fold_train, group_col='store_id')
    nested_fit = nested_fit.evaluate(fold_test)

    _, _, stats = nested_fit.extract_outputs()
    rmse = stats[(stats['metric'] == 'rmse') & (stats['split'] == 'test')]['value'].mean()
    cv_rmse.append(rmse)

print(f"CV RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
```

### 4. Memory Management

#### For Large Panels (Many Groups)

```python
# Option 1: Fit groups in batches
def fit_nested_batched(workflow, data, group_col, batch_size=10):
    groups = data[group_col].unique()
    all_group_fits = {}

    for i in range(0, len(groups), batch_size):
        batch_groups = groups[i:i+batch_size]
        batch_data = data[data[group_col].isin(batch_groups)]

        batch_fit = workflow.fit_nested(batch_data, group_col=group_col)
        all_group_fits.update(batch_fit.group_fits)

        print(f"Fitted batch {i//batch_size + 1}/{len(groups)//batch_size + 1}")

    return NestedWorkflowFit(workflow, group_col, all_group_fits)

# Option 2: Use global model for efficiency
if len(data['store_id'].unique()) > 100:
    print("Many groups detected → Using global model for efficiency")
    fit = wf.fit_global(data, group_col='store_id')
else:
    fit = wf.fit_nested(data, group_col='store_id')
```

#### Delete Intermediate Objects

```python
# Clear memory after extracting outputs
outputs, coeffs, stats = nested_fit.extract_outputs()

# Delete the fit object if no longer needed
del nested_fit
import gc
gc.collect()
```

### 5. Parallel Processing

**Note**: py-workflows doesn't currently support parallel fitting, but you can implement it:

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def fit_single_group(group_id, data, workflow, group_col):
    """Fit model for a single group"""
    group_data = data[data[group_col] == group_id].copy()
    group_data = group_data.drop(columns=[group_col])

    group_fit = workflow.fit(group_data)
    return (group_id, group_fit)

def fit_nested_parallel(workflow, data, group_col, n_workers=4):
    """Fit nested models in parallel"""
    groups = data[group_col].unique()

    # Create partial function with fixed arguments
    fit_func = partial(fit_single_group, data=data, workflow=workflow, group_col=group_col)

    # Fit in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(fit_func, groups))

    # Reconstruct group_fits dict
    group_fits = dict(results)

    return NestedWorkflowFit(workflow, group_col, group_fits)

# Use parallel fitting
nested_fit_parallel = fit_nested_parallel(
    wf,
    train,
    group_col='store_id',
    n_workers=4
)
```

### 6. Reproducibility

```python
# Set seeds for reproducibility
import numpy as np
import random

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # If using tensorflow/pytorch, set their seeds too

set_seeds(42)

# Document model configuration
model_config = {
    'approach': 'nested',
    'group_col': 'store_id',
    'formula': 'sales ~ time + temperature',
    'model': 'linear_reg',
    'train_period': '2023-01-01 to 2023-04-10',
    'test_period': '2023-04-11 to 2023-04-30',
    'seed': 42
}

# Save configuration
import json
with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
```

### 7. Documentation

```python
# Document each group's model
for group_id, group_fit in nested_fit.group_fits.items():
    outputs_g, coeffs_g, stats_g = group_fit.extract_outputs()

    # Extract key metrics
    train_rmse = stats_g[
        (stats_g['metric'] == 'rmse') &
        (stats_g['split'] == 'train')
    ]['value'].values[0]

    test_rmse = stats_g[
        (stats_g['metric'] == 'rmse') &
        (stats_g['split'] == 'test')
    ]['value'].values[0]

    # Log or print
    print(f"\nGroup {group_id}:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Coefficients:")
    print(coeffs_g[['variable', 'coefficient']])
```

---

## Common Patterns

### Pattern 1: Store-Level Forecasting

```python
# Scenario: Retail chain with multiple stores
# Goal: Forecast sales per store for inventory planning

# Data: Daily sales for 50 stores over 2 years
data = pd.DataFrame({
    'date': ...,
    'store_id': ...,
    'sales': ...,
    'day_of_week': ...,
    'is_holiday': ...,
    'temperature': ...
})

# Approach: Nested recursive models (each store has own seasonality)
wf = (
    workflow()
    .add_formula("sales ~ day_of_week + is_holiday + temperature")
    .add_model(recursive_reg(
        base_model=rand_forest(trees=100).set_mode('regression'),
        lags=[1, 2, 3, 7, 14]  # Recent days + weekly pattern
    ))
)

# Fit per store
nested_fit = wf.fit_nested(train, group_col='store_id')

# Generate 30-day forecasts per store
future_dates = pd.date_range(test['date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.concat([
    pd.DataFrame({
        'date': future_dates,
        'store_id': store_id,
        'day_of_week': future_dates.dayofweek,
        'is_holiday': [0] * 30,  # Assume no holidays
        'temperature': [70] * 30  # Use historical average
    })
    for store_id in data['store_id'].unique()
])

forecasts = nested_fit.predict(future_data)
```

### Pattern 2: Product-Level Demand

```python
# Scenario: E-commerce with many products
# Goal: Predict demand per product

# Data: Weekly demand for 500 products
data = pd.DataFrame({
    'week': ...,
    'product_id': ...,
    'demand': ...,
    'price': ...,
    'promotion': ...,
    'competitor_price': ...
})

# Approach: Global model (products share demand patterns, differ in levels)
wf = (
    workflow()
    .add_formula("demand ~ price + promotion + competitor_price")
    .add_model(boost_tree(trees=500, tree_depth=6).set_mode('regression'))
)

# Fit global model
global_fit = wf.fit_global(train, group_col='product_id')

# product_id becomes a categorical feature
# Model learns common price elasticity, but product-specific levels
```

### Pattern 3: Region-Level Predictions

```python
# Scenario: National company with regional divisions
# Goal: Predict revenue per region

# Data: Monthly revenue for 10 regions
data = pd.DataFrame({
    'month': ...,
    'region': ...,
    'revenue': ...,
    'gdp_growth': ...,
    'unemployment_rate': ...,
    'marketing_spend': ...
})

# Approach: Nested models (regions have different economic patterns)
wf = (
    workflow()
    .add_formula("revenue ~ gdp_growth + unemployment_rate + marketing_spend")
    .add_model(linear_reg())
)

nested_fit = wf.fit_nested(train, group_col='region')

# Extract region-specific sensitivities
outputs, coeffs, stats = nested_fit.extract_outputs()

# Marketing ROI per region
marketing_coefs = coeffs[coeffs['variable'] == 'marketing_spend']
print("Marketing ROI by Region:")
print(marketing_coefs[['region', 'coefficient']].sort_values('coefficient', ascending=False))
```

### Pattern 4: Customer Segmentation

```python
# Scenario: B2B company with customer segments
# Goal: Predict churn per segment

# Data: Monthly customer metrics by segment
data = pd.DataFrame({
    'month': ...,
    'segment': ...,  # SMB, Enterprise, etc.
    'churn': ...,    # Binary outcome
    'support_tickets': ...,
    'usage_score': ...,
    'contract_value': ...
})

# Approach: Nested classification models
wf = (
    workflow()
    .add_formula("churn ~ support_tickets + usage_score + contract_value")
    .add_model(rand_forest(trees=200).set_mode('classification'))
)

nested_fit = wf.fit_nested(train, group_col='segment')

# Predict churn probabilities per segment
churn_probs = nested_fit.predict(test, type='prob')

# High-risk customers per segment
high_risk = churn_probs[churn_probs['.pred_1'] > 0.7]  # Assuming class 1 is churn
print(f"High-risk customers by segment:")
print(high_risk.groupby('segment').size())
```

### Pattern 5: A/B Testing Analysis

```python
# Scenario: A/B test running across multiple stores
# Goal: Estimate treatment effect per store

# Data: Daily metrics during A/B test
data = pd.DataFrame({
    'date': ...,
    'store_id': ...,
    'treatment': ...,  # 0 = control, 1 = treatment
    'conversion_rate': ...,
    'avg_order_value': ...
})

# Approach: Nested models to estimate store-specific treatment effects
wf = (
    workflow()
    .add_formula("conversion_rate ~ treatment + avg_order_value")
    .add_model(linear_reg())
)

nested_fit = wf.fit_nested(data, group_col='store_id')

# Extract treatment effects per store
outputs, coeffs, stats = nested_fit.extract_outputs()
treatment_effects = coeffs[coeffs['variable'] == 'treatment']

print("Treatment Effect by Store:")
print(treatment_effects[['store_id', 'coefficient', 'p_value']])

# Identify stores with significant positive effect
winners = treatment_effects[
    (treatment_effects['coefficient'] > 0) &
    (treatment_effects['p_value'] < 0.05)
]

print(f"\nStores with significant lift: {len(winners)}")
```

---

## Troubleshooting

### Issue 1: Missing Groups in Test Data

**Problem:**
```python
nested_fit = wf.fit_nested(train, group_col='store_id')
predictions = nested_fit.predict(test)
# ValueError: No matching groups found in new_data
```

**Cause:** Test data contains groups not in training data.

**Solution:**
```python
# Check groups
train_groups = set(train['store_id'].unique())
test_groups = set(test['store_id'].unique())

unseen_groups = test_groups - train_groups
print(f"Unseen groups in test: {unseen_groups}")

# Filter test to only include trained groups
test_filtered = test[test['store_id'].isin(train_groups)]
predictions = nested_fit.predict(test_filtered)

# Or handle unseen groups with fallback model
if unseen_groups:
    # Fit global model as fallback
    fallback_fit = wf.fit_global(train, group_col='store_id')

    # Predict on unseen groups with fallback
    test_unseen = test[test['store_id'].isin(unseen_groups)]
    predictions_unseen = fallback_fit.predict(test_unseen)

    # Combine predictions
    predictions_seen = nested_fit.predict(test_filtered)
    all_predictions = pd.concat([predictions_seen, predictions_unseen])
```

### Issue 2: Varying Time Ranges per Group

**Problem:**
```python
# Different groups have different date ranges
data.groupby('store_id')['date'].agg(['min', 'max'])
# Store A: 2023-01-01 to 2023-12-31 (365 days)
# Store B: 2023-06-01 to 2023-12-31 (214 days)
# Store C: 2023-01-01 to 2023-06-30 (181 days)
```

**Impact:** Groups with shorter ranges have less training data.

**Solution 1: Filter to common date range**
```python
# Find common date range
min_date = data.groupby('store_id')['date'].min().max()  # Latest start
max_date = data.groupby('store_id')['date'].max().min()  # Earliest end

print(f"Common range: {min_date} to {max_date}")

# Filter data
data_common = data[
    (data['date'] >= min_date) &
    (data['date'] <= max_date)
]

# All groups now have same date range
nested_fit = wf.fit_nested(data_common, group_col='store_id')
```

**Solution 2: Accept varying ranges**
```python
# Fit with available data per group
nested_fit = wf.fit_nested(data, group_col='store_id')

# Check sample sizes per group
outputs, _, stats = nested_fit.extract_outputs()
sample_sizes = outputs.groupby('store_id').size()

print("Sample sizes by group:")
print(sample_sizes.sort_values())

# Flag groups with insufficient data
min_samples = 50
small_groups = sample_sizes[sample_sizes < min_samples]

if len(small_groups) > 0:
    print(f"\nWarning: {len(small_groups)} groups have < {min_samples} observations")
    print(small_groups)
```

### Issue 3: Group-Specific Errors

**Problem:**
```python
# Some groups fail to fit
nested_fit = wf.fit_nested(train, group_col='store_id')
# Some groups may have errors (singular matrices, convergence issues)
```

**Solution: Graceful error handling**
```python
# Implement custom fit_nested with error handling
from py_workflows.workflow import NestedWorkflowFit

def fit_nested_safe(workflow, data, group_col):
    """Fit nested models with error handling"""
    groups = data[group_col].unique()
    group_fits = {}
    failed_groups = []

    for group in groups:
        try:
            group_data = data[data[group_col] == group].copy()
            group_data = group_data.drop(columns=[group_col])

            group_fits[group] = workflow.fit(group_data)
            print(f"✓ Fitted {group}")

        except Exception as e:
            print(f"✗ Failed {group}: {e}")
            failed_groups.append(group)

    if failed_groups:
        print(f"\nFailed groups: {failed_groups}")
        print("Consider:")
        print("  - Checking data quality for these groups")
        print("  - Using simpler models")
        print("  - Excluding these groups")

    return NestedWorkflowFit(workflow, group_col, group_fits), failed_groups

# Use safe fitting
nested_fit, failed = fit_nested_safe(wf, train, 'store_id')

# Predict only on successful groups
test_filtered = test[~test['store_id'].isin(failed)]
predictions = nested_fit.predict(test_filtered)
```

### Issue 4: Memory Issues with Many Groups

**Problem:**
```python
# 1000+ groups causes memory issues
nested_fit = wf.fit_nested(large_data, group_col='store_id')
# MemoryError
```

**Solution 1: Batch processing**
```python
# Process groups in batches
def fit_nested_batched(workflow, data, group_col, batch_size=50):
    groups = data[group_col].unique()
    all_fits = {}

    n_batches = len(groups) // batch_size + 1

    for i in range(0, len(groups), batch_size):
        batch_groups = groups[i:i+batch_size]
        batch_data = data[data[group_col].isin(batch_groups)]

        print(f"Processing batch {i//batch_size + 1}/{n_batches}...")

        batch_fit = workflow.fit_nested(batch_data, group_col=group_col)
        all_fits.update(batch_fit.group_fits)

        # Clear memory
        del batch_fit
        import gc
        gc.collect()

    return NestedWorkflowFit(workflow, group_col, all_fits)

nested_fit = fit_nested_batched(wf, large_data, 'store_id', batch_size=50)
```

**Solution 2: Use global model**
```python
# For very large panels, global model is more efficient
if len(data['store_id'].unique()) > 500:
    print("Many groups → Using global model")
    fit = wf.fit_global(data, group_col='store_id')
else:
    fit = wf.fit_nested(data, group_col='store_id')
```

**Solution 3: Simpler models**
```python
# Use linear models instead of tree-based
wf_simple = (
    workflow()
    .add_formula("y ~ x1 + x2")
    .add_model(linear_reg())  # Much less memory than rand_forest
)

nested_fit = wf_simple.fit_nested(large_data, group_col='store_id')
```

### Issue 5: Prediction Indices Don't Match

**Problem:**
```python
predictions = nested_fit.predict(test)
# predictions has RangeIndex(0, n)
# test has original DatetimeIndex

# This fails:
test['predictions'] = predictions['.pred']
# ValueError: Length mismatch
```

**Solution:**
```python
# Reset test index before adding predictions
test_reset = test.reset_index(drop=False)
predictions_reset = predictions.reset_index(drop=True)

# Add predictions
test_reset['.pred'] = predictions_reset['.pred']

# Restore original index if needed
test_with_preds = test_reset.set_index('date')
```

### Issue 6: Date Column Handling

**Problem:**
```python
# Recursive model needs date as index, but it's a column
data = pd.DataFrame({'date': ..., 'store_id': ..., 'sales': ...})

wf = workflow().add_formula("sales ~ .").add_model(
    recursive_reg(base_model=linear_reg(), lags=7)
)

nested_fit = wf.fit_nested(data, group_col='store_id')
# May fail or give unexpected results
```

**Solution:**
```python
# fit_nested() handles this automatically for recursive models!
# Just ensure 'date' column is named 'date' (or adjust code to look for it)

# If date column has different name:
data_renamed = data.rename(columns={'timestamp': 'date'})

nested_fit = wf.fit_nested(data_renamed, group_col='store_id')
# Works! fit_nested detects recursive model and sets date as index per group
```

### Issue 7: Formula Already Has Group Column

**Problem:**
```python
# Formula explicitly includes group column
wf = workflow().add_formula("y ~ x + store_id").add_model(linear_reg())

# Using fit_global adds store_id again
global_fit = wf.fit_global(data, group_col='store_id')
# May cause duplication issues
```

**Solution:**
```python
# fit_global() checks for this and doesn't duplicate!
# The updated formula logic handles this case:

# Original: "y ~ x + store_id"
# After fit_global(..., group_col='store_id'): "y ~ x + store_id" (no change)

# So you're safe - just use fit_global normally
global_fit = wf.fit_global(data, group_col='store_id')
```

---

## Summary

### Key Takeaways

1. **Panel modeling enables per-group or group-aware modeling** for datasets with multiple entities
2. **Nested approach** fits separate models per group (best for different patterns)
3. **Global approach** fits one model with group as feature (best for similar patterns)
4. **Works with any model type**: linear, tree-based, time series, etc.
5. **Three-DataFrame outputs include group column** for easy analysis
6. **Unified API** makes it easy to switch between approaches

### Quick Reference

```python
# Nested (per-group models)
nested_fit = wf.fit_nested(train, group_col='store_id')
predictions = nested_fit.predict(test)
nested_fit = nested_fit.evaluate(test)
outputs, coeffs, stats = nested_fit.extract_outputs()

# Global (single model with group feature)
global_fit = wf.fit_global(train, group_col='store_id')
predictions = global_fit.predict(test)
global_fit = global_fit.evaluate(test)
outputs, coeffs, stats = global_fit.extract_outputs()

# Access specific group model (nested only)
store_a_fit = nested_fit.group_fits['A']

# Compare metrics across groups
test_rmse = stats[
    (stats['metric'] == 'rmse') &
    (stats['split'] == 'test')
][['store_id', 'value']].sort_values('value')
```

### When to Use

- **Use nested** when groups are fundamentally different
- **Use global** when groups share patterns with different levels
- **Compare both** when unsure (often global is surprisingly good)
- **Use hybrid** for large-scale projects (analyze nested, deploy global)

### Next Steps

1. Try both approaches on your panel data
2. Compare performance metrics
3. Analyze coefficient variation across groups
4. Choose based on decision framework
5. Iterate with different models and features

---

**End of Guide**
