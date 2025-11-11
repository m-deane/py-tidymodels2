# Code Change Reference: Converting to Grouped Modeling

## Quick Reference: Before → After

### 1. ModelSpec.fit() → ModelSpec with Workflows

**BEFORE (Standard fitting - 39 instances)**:
```python
spec_prophet = prophet_reg(n_changepoints=25, ...)
fit_prophet = spec_prophet.fit(train_data, FORMULA_STR)
fit_prophet = fit_prophet.evaluate(test_data)
outputs, coefs, stats = fit_prophet.extract_outputs()
```

**AFTER - NESTED (Per-Country Models)**:
```python
from py_workflows import workflow

spec_prophet = prophet_reg(n_changepoints=25, ...)

# Create workflow
wf_prophet = workflow().add_formula(FORMULA_STR).add_model(spec_prophet)

# Fit nested (10 separate models, one per country)
fit_prophet_nested = wf_prophet.fit_nested(train_data, group_col='country')

# Evaluate (automatically uses group_col from fit)
fit_prophet_nested = fit_prophet_nested.evaluate(test_data)

# Extract outputs (now includes 'group' column with country names)
outputs, coefs, stats = fit_prophet_nested.extract_outputs()

# Filter by country if needed
algeria_outputs = outputs[outputs['group'] == 'Algeria']
```

**AFTER - GLOBAL (Single Model, Country as Feature)**:
```python
from py_workflows import workflow

spec_prophet = prophet_reg(n_changepoints=25, ...)

# Create workflow
wf_prophet = workflow().add_formula(FORMULA_STR).add_model(spec_prophet)

# Fit global (ONE model using all countries)
fit_prophet_global = wf_prophet.fit_global(train_data, group_col='country')

# Evaluate
fit_prophet_global = fit_prophet_global.evaluate(test_data)

# Extract outputs (includes 'group' column)
outputs, coefs, stats = fit_prophet_global.extract_outputs()
```

### 2. Visualization Changes

**BEFORE**:
```python
from py_visualize import plot_forecast, plot_residuals

fig = plot_forecast(fit_prophet, title="Sales Forecast")
fig.show()

fig = plot_residuals(fit_prophet, title="Sales Forecast")
fig.show()
```

**AFTER - Option A (All Groups)**:
```python
from py_visualize import plot_forecast, plot_residuals

# Plot all countries together (may be crowded with 10 countries)
fig = plot_forecast(fit_prophet_nested, title="Refinery Forecast - All Countries")
fig.show()
```

**AFTER - Option B (Filtered to One Group)**:
```python
# Extract outputs for one country
algeria_outputs = outputs[outputs['group'] == 'Algeria']

# Create temporary fit object for visualization
# (May need custom wrapper if plot_forecast requires fit object)

# OR: Use outputs directly
import plotly.graph_objects as go

fig = go.Figure()
algeria_data = algeria_outputs[algeria_outputs['split'] == 'test']
fig.add_trace(go.Scatter(
    x=algeria_data.index,
    y=algeria_data['actuals'],
    name='Actual',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=algeria_data.index,
    y=algeria_data['fitted'],
    name='Forecast',
    mode='lines'
))
fig.update_layout(title="Algeria Refinery Forecast")
fig.show()
```

**AFTER - Option C (Loop Through Groups)**:
```python
# Plot each country separately
for country in outputs['group'].unique():
    country_outputs = outputs[outputs['group'] == country]
    
    # Create custom plot for this country
    fig = go.Figure()
    test_data = country_outputs[country_outputs['split'] == 'test']
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['actuals'], name='Actual'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['fitted'], name='Forecast'))
    fig.update_layout(title=f"{country} Refinery Forecast")
    fig.show()
```

### 3. Formula Changes

**BEFORE**:
```python
FORMULA_STR = "refinery_kbd ~ ."
```

**AFTER - For NESTED (exclude country from predictors)**:
```python
# Option A: Explicit exclusion
FORMULA_STR = "refinery_kbd ~ . -country"

# Option B: Explicit predictors
FORMULA_STR = "refinery_kbd ~ brent + wti + brent_cracking_nw_europe + ..."
```

**AFTER - For GLOBAL (keep country, or let fit_global handle)**:
```python
# Option A: Let fit_global handle it automatically
FORMULA_STR = "refinery_kbd ~ ."

# Option B: Explicitly include country
FORMULA_STR = "refinery_kbd ~ country + brent + wti + ..."
```

## Cell-by-Cell Changes

### Cells Requiring Updates

#### Cell 13 - Formula Definition
```python
# BEFORE
FORMULA_STR = "refinery_kbd ~ ."

# AFTER
FORMULA_STR_NESTED = "refinery_kbd ~ . -country"  # For nested approach
FORMULA_STR_GLOBAL = "refinery_kbd ~ ."           # For global approach
```

#### Cell 15 - Prophet (First Example)
```python
# BEFORE
spec_prophet = prophet_reg(n_changepoints=25, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
fit_prophet = spec_prophet.fit(train_data, FORMULA_STR)
fit_prophet = fit_prophet.evaluate(test_data)
outputs_prophet, coefs_prophet, stats_prophet = fit_prophet.extract_outputs()

# AFTER - GLOBAL (demonstrate global approach first)
from py_workflows import workflow

spec_prophet = prophet_reg(n_changepoints=25, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
wf_prophet_global = workflow().add_formula(FORMULA_STR_GLOBAL).add_model(spec_prophet)
fit_prophet_global = wf_prophet_global.fit_global(train_data, group_col='country')
fit_prophet_global = fit_prophet_global.evaluate(test_data)
outputs_prophet, coefs_prophet, stats_prophet = fit_prophet_global.extract_outputs()

print(f"Global model fitted. Group column values: {outputs_prophet['group'].unique()}")
```

#### Cell 17 - ARIMA (Second Example)
```python
# BEFORE
spec_arima = arima_reg(seasonal_period=7, non_seasonal_ar=1, ...)
fit_arima = spec_arima.fit(train_data, FORMULA_STR)
fit_arima = fit_arima.evaluate(test_data)
outputs_arima, coefs_arima, stats_arima = fit_arima.extract_outputs()

# AFTER - NESTED (demonstrate nested approach)
from py_workflows import workflow

spec_arima = arima_reg(seasonal_period=7, non_seasonal_ar=1, ...)
wf_arima_nested = workflow().add_formula(FORMULA_STR_NESTED).add_model(spec_arima)
fit_arima_nested = wf_arima_nested.fit_nested(train_data, group_col='country')
fit_arima_nested = fit_arima_nested.evaluate(test_data)
outputs_arima, coefs_arima, stats_arima = fit_arima_nested.extract_outputs()

print(f"Nested models fitted. Number of groups: {outputs_arima['group'].nunique()}")
print(f"Groups: {outputs_arima['group'].unique()}")
```

#### Pattern for All Other Models (Cells 19, 22-82)
```python
# BEFORE (template)
spec_<model> = <model_function>(<params>)
fit_<model> = spec_<model>.fit(train_data, FORMULA_STR)
fit_<model> = fit_<model>.evaluate(test_data)
outputs, coefs, stats = fit_<model>.extract_outputs()

# AFTER - NESTED (apply to most models)
from py_workflows import workflow

spec_<model> = <model_function>(<params>)
wf_<model> = workflow().add_formula(FORMULA_STR_NESTED).add_model(spec_<model>)
fit_<model>_nested = wf_<model>.fit_nested(train_data, group_col='country')
fit_<model>_nested = fit_<model>_nested.evaluate(test_data)
outputs, coefs, stats = fit_<model>_nested.extract_outputs()
```

#### Cell 84 - Workflow Example
```python
# BEFORE
wf_formula = (
    workflow()
    .add_formula(FORMULA_STR)
    .add_model(linear_reg())
)
fit_formula = wf_formula.fit(train_processed)
fit_formula = fit_formula.evaluate(test_processed)

# AFTER - NESTED
wf_formula = (
    workflow()
    .add_formula(FORMULA_STR_NESTED)
    .add_model(linear_reg())
)
fit_formula_nested = wf_formula.fit_nested(train_processed, group_col='country')
fit_formula_nested = fit_formula_nested.evaluate(test_processed)
outputs_formula, coefs_formula, stats_formula = fit_formula_nested.extract_outputs()
```

#### Cell 89 - Recipe + Workflow
```python
# BEFORE
rec_for_model = recipe().step_impute_mean().step_normalize()
wf = workflow().add_recipe(rec_for_model).add_model(linear_reg().set_engine("sklearn"))
wf_fit = wf.fit(train_data)

# AFTER - NESTED
rec_for_model = recipe().step_impute_mean().step_normalize()
wf = workflow().add_recipe(rec_for_model).add_model(linear_reg().set_engine("sklearn"))
wf_fit_nested = wf.fit_nested(train_data, group_col='country')
wf_fit_nested = wf_fit_nested.evaluate(test_data)
outputs, coefs, stats = wf_fit_nested.extract_outputs()
```

## Special Cases

### Recursive Forecasting (Cell 79)
```python
# BEFORE
train_indexed = train_data.set_index('date')
spec_recursive = recursive_reg(...)
fit_recursive = spec_recursive.fit(train_indexed, FORMULA_STR)

# AFTER - NESTED (handles date indexing per group)
from py_workflows import workflow

spec_recursive = recursive_reg(...)
wf_recursive = workflow().add_formula(FORMULA_STR_NESTED).add_model(spec_recursive)

# fit_nested will handle date indexing per group automatically
fit_recursive_nested = wf_recursive.fit_nested(train_data, group_col='country')
fit_recursive_nested = fit_recursive_nested.evaluate(test_data)
outputs, coefs, stats = fit_recursive_nested.extract_outputs()
```

### Models with Special Modes (Decision Tree, SVM, etc.)
```python
# BEFORE
spec_dt = decision_tree(tree_depth=5, min_n=2).set_mode('regression')
fit_dt = spec_dt.fit(train_data, FORMULA_STR)

# AFTER - NESTED (same pattern)
from py_workflows import workflow

spec_dt = decision_tree(tree_depth=5, min_n=2).set_mode('regression')
wf_dt = workflow().add_formula(FORMULA_STR_NESTED).add_model(spec_dt)
fit_dt_nested = wf_dt.fit_nested(train_data, group_col='country')
fit_dt_nested = fit_dt_nested.evaluate(test_data)
```

## Summary of Changes

### Total Changes Required
- **1 cell**: Formula definition (cell 13) → Add two formulas (nested vs global)
- **~40 cells**: Model fitting (cells 15-82) → Convert to workflow + fit_nested/fit_global
- **~3 cells**: Workflow examples (cells 84, 89, etc.) → Add fit_nested/fit_global
- **~40 cells**: Visualization (all plot_forecast/plot_residuals calls) → Add group handling

### Recommended Approach
1. **Phase 1**: Update first 5 models to show both global and nested approaches
2. **Phase 2**: Batch update remaining 35 models to use nested approach
3. **Phase 3**: Update workflow examples
4. **Phase 4**: Add group-aware visualization examples

### Key Benefits
- Properly handles country-specific patterns
- Outputs include 'group' column for filtering
- Consistent with panel modeling best practices
- Supports both nested (per-group) and global (pooled) approaches
