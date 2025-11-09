# Complete Workflow Reference

**py-tidymodels Workflow System - Comprehensive Documentation**

This reference covers the complete workflow system for composing preprocessing and modeling pipelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Class](#workflow-class)
3. [WorkflowFit Class](#workflowfit-class)
4. [NestedWorkflowFit Class](#nestedworkflowfit-class)
5. [Common Patterns](#common-patterns)
6. [Best Practices](#best-practices)

---

## Overview

The `py_workflows` module provides pipeline composition for preprocessing + modeling. Workflows are immutable specifications that combine:
- **Preprocessor**: Formula (string) or Recipe (preprocessing steps)
- **Model**: ModelSpec from py_parsnip
- **Optional**: Post-processing and case weights

**Key Benefits:**
- Unified interface for different preprocessing strategies
- Immutable design prevents side effects
- Consistent predict/evaluate interface
- Seamless integration with tuning and resampling

---

## Workflow Class

**Purpose:** Immutable workflow specification composing preprocessing + modeling.

### Creation

```python
from py_workflows import workflow

wf = workflow()  # Create empty workflow
```

---

### Methods

#### `add_formula(formula: str) -> Workflow`

Add R-style formula for preprocessing.

**Parameters:**
- `formula` (str): Formula string (e.g., `"y ~ x1 + x2"`)

**Returns:** New Workflow with formula added

**Raises:** `ValueError` if workflow already has a preprocessor

**Examples:**
```python
# Simple formula
wf = workflow().add_formula("sales ~ price + advertising")

# With interactions
wf = workflow().add_formula("sales ~ price * advertising")

# Patsy I() transformations
wf = workflow().add_formula("sales ~ price + I(price**2)")

# Dot notation (all predictors)
wf = workflow().add_formula("target ~ .")
```

**Dot Notation Behavior:**
- `.` expands to all columns except outcome
- Datetime columns automatically excluded
- Example: `"y ~ ."` with columns [date, x1, x2, y] → `"y ~ x1 + x2"`

**Supported Transformations:**
- Interactions: `x1 * x2`, `I(x1*x2)`
- Polynomials: `I(x1**2)`, `I(x1**3)`
- Arithmetic: `I(x1 + x2)`, `I(x1 / x2)`

---

#### `add_model(spec: ModelSpec) -> Workflow`

Add model specification.

**Parameters:**
- `spec` (ModelSpec): Model from py_parsnip (e.g., `linear_reg()`, `rand_forest()`)

**Returns:** New Workflow with model added

**Raises:** `ValueError` if workflow already has a model

**Examples:**
```python
from py_parsnip import linear_reg, rand_forest, prophet_reg

# Linear regression
wf = workflow().add_model(linear_reg())

# Random forest with parameters
wf = workflow().add_model(
    rand_forest(trees=100, min_n=5).set_mode('regression')
)

# Time series model
wf = workflow().add_model(prophet_reg())
```

**Mode Setting:**
Some models require explicit mode:
```python
# Models requiring .set_mode()
decision_tree().set_mode('regression')
rand_forest().set_mode('classification')
mlp().set_mode('regression')
nearest_neighbor().set_mode('classification')
```

---

#### `add_recipe(recipe: Recipe) -> Workflow`

Add preprocessing recipe.

**Parameters:**
- `recipe` (Recipe): Recipe object with preprocessing steps

**Returns:** New Workflow with recipe added

**Raises:** `ValueError` if workflow already has a preprocessor

**Examples:**
```python
from py_recipes import recipe

# Recipe with normalization
rec = (recipe()
    .step_normalize()
    .step_dummy(['category']))

wf = workflow().add_recipe(rec).add_model(linear_reg())
```

**Recipe Behavior:**
- Recipe is prepped during `fit()`, not during `add_recipe()`
- Auto-generates formula internally (excludes datetime columns)
- Outcome column detected from first step or inferred during fit

---

#### `update_formula(formula: str) -> Workflow`

Replace existing formula.

**Parameters:**
- `formula` (str): New formula string

**Returns:** New Workflow with updated formula

**Examples:**
```python
wf = workflow().add_formula("y ~ x1")
wf = wf.update_formula("y ~ x1 + x2")  # Replace
```

---

#### `update_model(spec: ModelSpec) -> Workflow`

Replace existing model.

**Parameters:**
- `spec` (ModelSpec): New model specification

**Returns:** New Workflow with updated model

**Examples:**
```python
wf = workflow().add_formula("y ~ x").add_model(linear_reg())
wf = wf.update_model(rand_forest().set_mode('regression'))
```

---

#### `remove_formula() -> Workflow`

Remove preprocessor from workflow.

**Returns:** New Workflow without preprocessor

---

#### `remove_model() -> Workflow`

Remove model from workflow.

**Returns:** New Workflow without model

---

#### `fit(data: pd.DataFrame) -> WorkflowFit`

Fit workflow on training data.

**Parameters:**
- `data` (pd.DataFrame): Training data

**Returns:** WorkflowFit object containing fitted pipeline

**Raises:**
- `ValueError` if workflow doesn't have a model
- `ValueError` if workflow doesn't have a preprocessor

**Examples:**
```python
# Basic fit
wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
wf_fit = wf.fit(train_data)

# Method chaining
wf_fit = (workflow()
    .add_formula("sales ~ price + advertising")
    .add_model(linear_reg())
    .fit(train_data))
```

**Execution Flow:**
1. **For formulas**: Pass directly to model's `fit()` method
2. **For recipes**:
   - Prep recipe on training data
   - Bake training data through recipe
   - Auto-generate formula from processed columns
   - Fit model on processed data

---

#### `fit_nested(data: pd.DataFrame, group_col: str) -> NestedWorkflowFit`

Fit separate models for each group (panel/grouped modeling).

**Parameters:**
- `data` (pd.DataFrame): Training data with group column
- `group_col` (str): Column name containing group identifiers

**Returns:** NestedWorkflowFit containing dict of fitted models per group

**Raises:**
- `ValueError` if `group_col` not in data
- `ValueError` if workflow doesn't have model

**Use Cases:**
- Multi-store sales forecasting (one model per store)
- Multi-product demand forecasting (one model per product)
- Multi-region time series (one model per region)
- When groups have different patterns/behaviors

**Examples:**
```python
from py_parsnip import recursive_reg, rand_forest

# Per-store forecasting
wf = (workflow()
    .add_formula("sales ~ date")
    .add_model(recursive_reg(
        base_model=rand_forest().set_mode('regression'),
        lags=7
    )))

nested_fit = wf.fit_nested(data, group_col="store_id")

# Predict for all groups
predictions = nested_fit.predict(test_data)

# Extract outputs with group column
outputs, coeffs, stats = nested_fit.extract_outputs()
print(outputs[["date", "store_id", "actuals", "forecast"]])
```

**Key Behaviors:**
- Fits one model per unique value in `group_col`
- Removes `group_col` before fitting each model
- For `recursive_reg`: sets date as index if present
- Returns unified interface via `NestedWorkflowFit`
- All outputs include `group_col` for filtering

---

#### `fit_global(data: pd.DataFrame, group_col: str) -> WorkflowFit`

Fit single global model using group as a feature.

**Parameters:**
- `data` (pd.DataFrame): Training data with group column
- `group_col` (str): Column name to use as predictor

**Returns:** Standard WorkflowFit with group as feature

**Raises:**
- `ValueError` if `group_col` not in data
- `ValueError` if workflow doesn't have model

**Use Cases:**
- Groups share similar patterns
- Insufficient data per group for separate models
- Want to capture cross-group effects
- When pooling information across groups beneficial

**Examples:**
```python
# Global model with store_id as feature
wf = (workflow()
    .add_formula("sales ~ date + store_id")
    .add_model(rand_forest().set_mode('regression')))

global_fit = wf.fit_global(data, group_col="store_id")
predictions = global_fit.predict(test_data)
```

**Key Behaviors:**
- Automatically adds `group_col` to formula if not present
- For formulas: updates formula to include group column
- For recipes: group column included if in data
- Returns standard `WorkflowFit` (not nested)

---

## WorkflowFit Class

**Purpose:** Fitted workflow containing preprocessing and model fit.

### Attributes

```python
@dataclass
class WorkflowFit:
    workflow: Workflow  # Original workflow spec
    pre: Any  # Fitted preprocessor (formula or PreparedRecipe)
    fit: ModelFit  # Fitted model
    post: Optional[Any] = None
    formula: Optional[str] = None  # Formula used for fitting
```

---

### Methods

#### `predict(new_data: pd.DataFrame, type: str = "numeric") -> pd.DataFrame`

Make predictions on new data.

**Parameters:**
- `new_data` (pd.DataFrame): New data for prediction
- `type` (str): Prediction type
  - `"numeric"`: Point predictions (default)
  - `"conf_int"`: Confidence intervals
  - `"pred_int"`: Prediction intervals

**Returns:** DataFrame with predictions

**Column Names:**
- `"numeric"`: `.pred`
- `"conf_int"`: `.pred`, `.pred_lower`, `.pred_upper`
- `"pred_int"`: `.pred`, `.pred_lower`, `.pred_upper`
- For VARMAX (multivariate): `.pred_y1`, `.pred_y2`, etc.

**Examples:**
```python
# Point predictions
predictions = wf_fit.predict(test_data)
print(predictions[[".pred"]])

# Prediction intervals
predictions = wf_fit.predict(test_data, type="pred_int")
print(predictions[[".pred", ".pred_lower", ".pred_upper"]])

# Access as series
pred_series = predictions[".pred"]
```

**Execution Flow:**
1. **For formulas**: Uses `forge()` to create model matrix
2. **For recipes**: Uses `bake()` to transform new data
3. Calls model's `predict()` method
4. Returns formatted DataFrame

**Error Handling:**
- Missing categorical levels: Handled via forge (adds zeros)
- Novel categorical levels: Depends on recipe steps (e.g., step_novel)
- Missing columns: Raises error with clear message

---

#### `evaluate(test_data: pd.DataFrame, outcome_col: Optional[str] = None) -> WorkflowFit`

Evaluate workflow on test data for train/test comparison.

**Parameters:**
- `test_data` (pd.DataFrame): Test data with actual outcomes
- `outcome_col` (Optional[str]): Outcome column name (auto-detected if None)

**Returns:** Self (for method chaining)

**Examples:**
```python
# Fit and evaluate
wf_fit = wf.fit(train_data).evaluate(test_data)

# Extract train/test metrics
outputs, coeffs, stats = wf_fit.extract_outputs()
test_stats = stats[stats["split"] == "test"]
print(test_stats[["metric", "value"]])
```

**Key Behaviors:**
- Applies preprocessing to test data
- Passes original test data to engines needing raw values
- Stores test predictions in underlying ModelFit
- Enables split-wise metrics in `extract_outputs()`

**Outcome Column Detection:**
- Strips one-hot encoding suffixes: `"species[setosa]"` → `"species"`
- Uses formula outcome if available
- Falls back to common names: "y", "target", "outcome"

---

#### `extract_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

Extract comprehensive model outputs (three-DataFrame pattern).

**Returns:** Tuple of three DataFrames:

**1. Outputs (observation-level):**
- `actuals`: True values from data
- `fitted`: Model predictions
- `forecast`: Combined actuals/fitted (seamless series)
- `residuals`: actuals - fitted
- `split`: "train", "test", or "forecast"
- `model`: Model type name
- `model_group_name`: Model identifier
- `group`: Group identifier (if panel modeling)

**2. Coefficients (variable-level):**
- `variable`: Predictor name
- `coefficient`: Parameter value
- `std_error`: Standard error (if available)
- `t_stat`: t-statistic (if available)
- `p_value`: p-value (if available)
- `ci_lower`, `ci_upper`: Confidence intervals
- `vif`: Variance Inflation Factor (if available)
- For tree models: feature importances instead

**3. Stats (model-level metrics):**
- `metric`: Metric name (rmse, mae, r_squared, etc.)
- `value`: Metric value
- `split`: "train", "test", or "all"
- Plus residual diagnostics (normality, autocorrelation, etc.)

**Examples:**
```python
outputs, coefficients, stats = wf_fit.extract_outputs()

# Observation-level results
print(outputs[["actuals", "fitted", "forecast", "split"]].head())

# Variable-level parameters
sig_vars = coefficients[coefficients["p_value"] < 0.05]
print(sig_vars[["variable", "coefficient", "p_value"]])

# Model-level metrics
train_rmse = stats[
    (stats["metric"] == "rmse") &
    (stats["split"] == "train")
]["value"].iloc[0]

test_rmse = stats[
    (stats["metric"] == "rmse") &
    (stats["split"] == "test")
]["value"].iloc[0]

print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
```

**Forecast Column Semantics:**
- Uses `pd.Series(actuals).combine_first(pd.Series(fitted))`
- Shows actuals where they exist, fitted where they don't
- Creates seamless series for visualization
- Same calculation for both train and test splits

---

#### `extract_formula() -> str`

Get formula string used for model fitting.

**Returns:** Formula string

**Examples:**
```python
# Explicit formula
wf_fit = workflow().add_formula("y ~ x1 + x2").add_model(spec).fit(train)
formula = wf_fit.extract_formula()
print(formula)  # "y ~ x1 + x2"

# Auto-generated from recipe
rec = recipe().step_normalize()
wf_fit = workflow().add_recipe(rec).add_model(spec).fit(train)
formula = wf_fit.extract_formula()
print(formula)  # "y ~ x1 + x2 + x3" (auto-generated, excludes datetime)
```

---

#### `extract_fit_parsnip() -> ModelFit`

Get underlying parsnip ModelFit object.

**Returns:** ModelFit object

**Examples:**
```python
model_fit = wf_fit.extract_fit_parsnip()

# Access raw sklearn model
sklearn_model = model_fit.fit_data["model"]

# Access fit metadata
print(model_fit.spec)  # ModelSpec
print(model_fit.fit_data.keys())  # Available fit artifacts
```

---

#### `extract_preprocessor() -> Union[str, PreparedRecipe]`

Get fitted preprocessor.

**Returns:** Formula string or PreparedRecipe

**Examples:**
```python
prep = wf_fit.extract_preprocessor()

# For formulas
if isinstance(prep, str):
    print(f"Formula: {prep}")

# For recipes
if hasattr(prep, 'steps'):
    print(f"Recipe with {len(prep.steps)} steps")
```

---

#### `extract_spec_parsnip() -> ModelSpec`

Get original model specification.

**Returns:** ModelSpec object

---

#### `extract_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame`

Get preprocessed data that model sees (debugging/inspection).

**Parameters:**
- `data` (pd.DataFrame): Data to preprocess (train or test)

**Returns:** DataFrame with preprocessing applied

**Examples:**
```python
# Inspect transformed features
train_transformed = wf_fit.extract_preprocessed_data(train)
print(train_transformed.columns)
print(train_transformed.describe())

# Check test data transformation
test_transformed = wf_fit.extract_preprocessed_data(test)

# Verify normalization
print(train_transformed.mean())  # Should be ~0 if normalized
print(train_transformed.std())   # Should be ~1 if normalized
```

---

## NestedWorkflowFit Class

**Purpose:** Fitted workflow with separate models for each group.

### Attributes

```python
@dataclass
class NestedWorkflowFit:
    workflow: Workflow  # Original workflow spec
    group_col: str  # Group column name
    group_fits: dict  # {group_value: WorkflowFit}
```

---

### Methods

#### `predict(new_data: pd.DataFrame, type: str = "numeric") -> pd.DataFrame`

Make predictions on new data for all groups.

**Parameters:**
- `new_data` (pd.DataFrame): New data with group column
- `type` (str): Prediction type ("numeric", "conf_int", "pred_int")

**Returns:** DataFrame with predictions and group column

**Raises:** `ValueError` if `group_col` not in `new_data`

**Examples:**
```python
predictions = nested_fit.predict(test_data)
print(predictions[["store_id", "date", ".pred"]])

# With prediction intervals
predictions = nested_fit.predict(test_data, type="pred_int")
print(predictions[["store_id", ".pred", ".pred_lower", ".pred_upper"]])

# Filter to specific group
store_a_preds = predictions[predictions["store_id"] == "A"]
```

**Key Behaviors:**
- Automatically routes each row to appropriate group model
- Skips groups not present in `new_data`
- Adds group column back to predictions
- Combines predictions from all groups
- Preserves original row order

---

#### `evaluate(test_data: pd.DataFrame, outcome_col: Optional[str] = None) -> NestedWorkflowFit`

Evaluate all group models on test data.

**Parameters:**
- `test_data` (pd.DataFrame): Test data with outcomes and group column
- `outcome_col` (Optional[str]): Outcome column name

**Returns:** Self (for method chaining)

**Examples:**
```python
nested_fit = wf.fit_nested(train, "store_id")
nested_fit = nested_fit.evaluate(test)
outputs, coeffs, stats = nested_fit.extract_outputs()
```

---

#### `extract_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

Extract outputs from all group models.

**Returns:** Tuple of three DataFrames with group column added

**Examples:**
```python
outputs, coefficients, stats = nested_fit.extract_outputs()

# Filter to specific group
store_a = outputs[outputs["store_id"] == "A"]
print(store_a[["date", "actuals", "forecast"]])

# Compare metrics across groups
group_rmse = stats[
    (stats["metric"] == "rmse") &
    (stats["split"] == "test")
][["store_id", "value"]].sort_values("value")

print("Worst performing stores:")
print(group_rmse.tail())

# Group-wise coefficients
store_a_coeffs = coefficients[coefficients["store_id"] == "A"]
print(store_a_coeffs[["variable", "coefficient", "p_value"]])
```

**Key Behaviors:**
- Combines outputs from all group models
- Adds `group_col` to all three DataFrames
- Enables group-wise filtering and comparison
- Useful for identifying problematic groups

---

## Common Patterns

### Pattern 1: Simple Formula + Model

```python
from py_workflows import workflow
from py_parsnip import linear_reg

wf = (workflow()
    .add_formula("sales ~ price + advertising")
    .add_model(linear_reg()))

wf_fit = wf.fit(train_data)
predictions = wf_fit.predict(test_data)
```

---

### Pattern 2: Recipe + Model

```python
from py_recipes import recipe

rec = (recipe()
    .step_impute_median()
    .step_normalize()
    .step_dummy(['category']))

wf = (workflow()
    .add_recipe(rec)
    .add_model(linear_reg()))

wf_fit = wf.fit(train_data)
predictions = wf_fit.predict(test_data)
```

---

### Pattern 3: Train/Test Evaluation

```python
# Fit on train, evaluate on test
wf_fit = wf.fit(train_data).evaluate(test_data)

# Extract comprehensive outputs
outputs, coeffs, stats = wf_fit.extract_outputs()

# Compare train vs test
metrics_by_split = stats.pivot_table(
    index="metric",
    columns="split",
    values="value"
)
print(metrics_by_split)
```

---

### Pattern 4: Panel/Grouped Modeling - Nested

```python
from py_parsnip import recursive_reg, rand_forest

# Per-store models
wf = (workflow()
    .add_formula("sales ~ date")
    .add_model(recursive_reg(
        base_model=rand_forest().set_mode('regression'),
        lags=7
    )))

nested_fit = wf.fit_nested(train, group_col="store_id")
nested_fit = nested_fit.evaluate(test)

# Predictions include group column
predictions = nested_fit.predict(test)

# Compare performance across groups
outputs, coeffs, stats = nested_fit.extract_outputs()
group_performance = stats[
    (stats["metric"] == "rmse") &
    (stats["split"] == "test")
][["store_id", "value"]]
```

---

### Pattern 5: Panel/Grouped Modeling - Global

```python
# Single model with group as feature
wf = (workflow()
    .add_formula("sales ~ date + store_id")
    .add_model(rand_forest().set_mode('regression')))

global_fit = wf.fit_global(train, group_col="store_id")
predictions = global_fit.predict(test)

# Standard evaluation
global_fit = global_fit.evaluate(test)
outputs, coeffs, stats = global_fit.extract_outputs()
```

---

### Pattern 6: Method Chaining

```python
# Complete pipeline in one chain
predictions = (
    workflow()
    .add_formula("y ~ x1 + x2 + I(x1*x2)")
    .add_model(linear_reg(penalty=0.1, mixture=1.0))
    .fit(train_data)
    .evaluate(test_data)
    .predict(future_data)
)
```

---

### Pattern 7: Workflow Reuse

```python
# Define once, use multiple times
base_wf = workflow().add_formula("y ~ .")

# Try different models
linear_fit = base_wf.add_model(linear_reg()).fit(train)
rf_fit = base_wf.add_model(rand_forest().set_mode('regression')).fit(train)
boost_fit = base_wf.add_model(boost_tree()).fit(train)

# Compare
for name, fit in [("Linear", linear_fit), ("RF", rf_fit), ("Boost", boost_fit)]:
    _, _, stats = fit.evaluate(test).extract_outputs()
    rmse = stats[(stats["metric"] == "rmse") & (stats["split"] == "test")]["value"].iloc[0]
    print(f"{name} RMSE: {rmse:.3f}")
```

---

### Pattern 8: Extract and Visualize

```python
import matplotlib.pyplot as plt

wf_fit = wf.fit(train).evaluate(test)
outputs, coeffs, stats = wf_fit.extract_outputs()

# Plot actuals vs fitted
fig, ax = plt.subplots(figsize=(10, 6))
train_data = outputs[outputs["split"] == "train"]
test_data = outputs[outputs["split"] == "test"]

ax.scatter(train_data["actuals"], train_data["fitted"], alpha=0.5, label="Train")
ax.scatter(test_data["actuals"], test_data["fitted"], alpha=0.5, label="Test")
ax.plot([outputs["actuals"].min(), outputs["actuals"].max()],
        [outputs["actuals"].min(), outputs["actuals"].max()],
        'k--', label="Perfect")
ax.set_xlabel("Actual")
ax.set_ylabel("Fitted")
ax.legend()
plt.show()

# Coefficient plot
coeffs_sorted = coeffs.sort_values("coefficient")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(coeffs_sorted["variable"], coeffs_sorted["coefficient"])
ax.set_xlabel("Coefficient")
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Best Practices

### 1. Workflow Design

**DO:**
- Use method chaining for concise workflow construction
- Separate preprocessing (formula/recipe) from modeling
- Use `.evaluate()` for proper train/test metrics
- Extract outputs using three-DataFrame pattern

**DON'T:**
- Mutate workflows (they're immutable by design)
- Mix formula and recipe in same workflow
- Skip evaluation on test data
- Hardcode outcome column names (use auto-detection)

---

### 2. Formula vs Recipe

**Use Formula when:**
- Simple transformations sufficient
- R-style formula syntax preferred
- Direct variable specification desired
- Minimal preprocessing needed

**Use Recipe when:**
- Complex multi-step preprocessing
- Reusable preprocessing pipelines
- Advanced feature engineering
- Consistent transformation across train/test

---

### 3. Panel Modeling

**Choose Nested when:**
- Groups have different patterns/behaviors
- Sufficient data per group (>100 observations)
- Want group-specific parameters
- Need to identify problematic groups

**Choose Global when:**
- Groups share similar patterns
- Limited data per group
- Want to pool information across groups
- Computational efficiency important

---

### 4. Evaluation

**Always:**
- Use separate train/test data
- Call `.evaluate()` before extracting outputs
- Compare train vs test metrics (overfitting check)
- Inspect residuals for model diagnostics

**Metrics to Monitor:**
- RMSE/MAE: Prediction error magnitude
- R²: Variance explained
- Train vs Test gap: Overfitting indicator
- Residual patterns: Model assumptions

---

### 5. Error Handling

**Common Issues:**
- **Missing preprocessor**: Ensure `.add_formula()` or `.add_recipe()` called
- **Missing model**: Ensure `.add_model()` called
- **Mode not set**: Some models require `.set_mode('regression'/'classification')`
- **Datetime in formula**: Automatically excluded, but check if needed
- **Novel categories**: Use `step_novel()` in recipe or ensure categories align

---

## Summary

The workflow system provides a unified, immutable interface for composing preprocessing and modeling pipelines. Key features:

✅ **Immutable design** prevents side effects
✅ **Flexible preprocessing** via formulas or recipes
✅ **Consistent interface** for predict/evaluate
✅ **Panel modeling** support (nested and global)
✅ **Comprehensive outputs** via three-DataFrame pattern
✅ **Seamless integration** with tuning and resampling

**Total Methods Documented:** 20+
**Last Updated:** 2025-11-09
**Version:** py-tidymodels v1.0
