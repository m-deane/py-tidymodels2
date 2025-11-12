# MLflow Integration Quick Start Guide

## TL;DR

MLflow is the recommended experiment tracking solution for py-tidymodels due to:
- **Open-source** and free (aligns with project philosophy)
- **Strong sklearn integration** via `autolog()` (works with 15+ py-parsnip engines)
- **Nested run architecture** (perfect match for WorkflowSet multi-model comparison)
- **Three-DataFrame artifact support** (preserves py-tidymodels' standardized outputs)
- **Production model registry** (enables deployment via REST API with version control)

---

## Minimal Working Example

```python
import mlflow
import mlflow.sklearn
from py_workflows import workflow
from py_parsnip import linear_reg

# Enable auto-logging
mlflow.sklearn.autolog()

# Create and fit workflow
wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())

with mlflow.start_run(run_name="my_first_experiment"):
    # Tag the run
    mlflow.set_tag("model_type", "linear_reg")

    # Fit (autolog captures params and metrics)
    fit = wf.fit(train_data)

    # Evaluate
    fit_eval = fit.evaluate(test_data)

    # Extract and log outputs
    outputs, coeffs, stats = fit.extract_outputs()
    mlflow.log_table(stats, "model_stats.json")

    # View in MLflow UI: http://localhost:5000
```

Start MLflow UI: `mlflow ui`

---

## Quick Integration Points

### 1. Single Workflow → Single MLflow Run

```python
with mlflow.start_run():
    fit = workflow.fit(data)
    # MLflow auto-captures sklearn metrics
```

### 2. WorkflowSet → Parent + Child Nested Runs

```python
with mlflow.start_run(run_name="WorkflowSet_Experiment") as parent:
    results = wf_set.fit_resamples(folds)

    for wf_id in wf_set.workflow_ids:
        with mlflow.start_run(run_name=wf_id, nested=True):
            # Log per-workflow metrics
```

### 3. Hyperparameter Tuning → Parent + Child Nested Runs

```python
with mlflow.start_run(run_name="Grid_Search") as parent:
    results = tune_grid(wf, resamples, grid)

    for idx, params in enumerate(grid):
        with mlflow.start_run(nested=True):
            # Log each parameter combination
```

### 4. Panel Data → Nested Runs Per Group

```python
with mlflow.start_run(run_name="Nested_By_Country") as parent:
    fit = wf.fit_nested(data, group_col='country')

    for group in data['country'].unique():
        with mlflow.start_run(run_name=f"country_{group}", nested=True):
            # Log per-group metrics
```

---

## Critical Implementation Details

### Three-DataFrame Artifact Logging

```python
outputs, coeffs, stats = fit.extract_outputs()

# Option 1: Log as JSON tables (queryable in MLflow)
mlflow.log_table(stats, "model_stats.json")
mlflow.log_table(coeffs, "coefficients.json")

# Option 2: Log as Parquet (5-10x smaller for large data)
outputs.to_parquet("/tmp/outputs.parquet")
mlflow.log_artifact("/tmp/outputs.parquet", "model_outputs")
```

### Time Series Diagnostic Plots

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... create plots ...
mlflow.log_figure(fig, "diagnostics.png")
```

### Interactive Plotly Plots

```python
import plotly.graph_objects as go

fig = go.Figure()
# ... create interactive plot ...

# IMPORTANT: Must use .html extension for Plotly
mlflow.log_figure(fig, "forecast.html")
```

### Recipe Tracking

```python
rec = recipe(data).step_pca(num_comp=5).step_normalize()
prepped = rec.prep()

# Log step sequence
step_names = [step.__class__.__name__ for step in rec.steps]
mlflow.log_param("recipe_steps", ",".join(step_names))

# Log PreparedRecipe artifact
import pickle
with open("/tmp/recipe.pkl", "wb") as f:
    pickle.dump(prepped, f)
mlflow.log_artifact("/tmp/recipe.pkl", "recipe")
```

---

## MLflow Python API Essentials

### Logging Functions

```python
# Parameters (strings or numbers)
mlflow.log_param("penalty", 0.1)
mlflow.log_params({"penalty": 0.1, "mixture": 0.5})

# Metrics (numbers only)
mlflow.log_metric("rmse", 2.5)
mlflow.log_metrics({"rmse": 2.5, "mae": 1.8})

# Tags (metadata)
mlflow.set_tag("model_type", "linear_reg")
mlflow.set_tags({"dataset": "sales_2024", "split": "test"})

# Artifacts (files)
mlflow.log_artifact("outputs.csv")  # Single file
mlflow.log_artifacts("/tmp/plots/")  # Directory

# Tables (DataFrames)
mlflow.log_table(df, "results.json")

# Figures (matplotlib or Plotly)
mlflow.log_figure(fig, "plot.png")  # matplotlib
mlflow.log_figure(fig, "plot.html")  # Plotly (must use .html)

# Models (sklearn, Prophet, etc.)
mlflow.sklearn.log_model(model, "model")
mlflow.prophet.log_model(model, "model")
```

### Run Management

```python
# Start run
with mlflow.start_run(run_name="experiment_1"):
    # ... logging code ...
    pass

# Nested run
with mlflow.start_run(run_name="parent") as parent:
    with mlflow.start_run(run_name="child", nested=True):
        # ... child run logging ...
        pass

# Manual run management (not recommended)
mlflow.start_run()
# ... logging code ...
mlflow.end_run()
```

---

## Common Patterns

### Pattern 1: Auto-Log Everything (Simplest)

```python
mlflow.sklearn.autolog()  # Enable before any sklearn training

with mlflow.start_run():
    fit = wf.fit(data)  # Auto-logs params, metrics, model
```

**Pros:** Zero boilerplate, captures sklearn metrics automatically
**Cons:** Less control over what's logged

### Pattern 2: Selective Manual Logging (Most Control)

```python
with mlflow.start_run():
    mlflow.log_params({"model_type": "linear_reg", "formula": "y ~ x1 + x2"})

    fit = wf.fit(data)
    outputs, coeffs, stats = fit.extract_outputs()

    mlflow.log_metrics({
        "train_rmse": stats[stats['split']=='train']['rmse'].iloc[0],
        "test_rmse": stats[stats['split']=='test']['rmse'].iloc[0]
    })

    mlflow.log_table(stats, "stats.json")
```

**Pros:** Full control, logs exactly what you need
**Cons:** More code, must manually extract metrics

### Pattern 3: Hybrid (Recommended)

```python
mlflow.sklearn.autolog(log_models=False)  # Auto-log metrics, but not models

with mlflow.start_run():
    # Custom logging
    mlflow.set_tags({"dataset": "sales_2024", "group_col": "country"})

    # Fit (auto-logs sklearn metrics)
    fit = wf.fit(data)

    # Custom artifact logging
    outputs, coeffs, stats = fit.extract_outputs()
    mlflow.log_table(stats, "stats.json")
```

**Pros:** Best of both worlds - auto-capture sklearn metrics, manual control over artifacts
**Cons:** Need to understand what autolog() captures

---

## Next Steps

### Immediate Actions (This Week)
1. Install MLflow: `pip install mlflow`
2. Start MLflow UI: `mlflow ui` (view at http://localhost:5000)
3. Run the minimal working example above
4. Explore the MLflow UI (runs, metrics, artifacts, model registry)

### Short-Term (Next 2-4 Weeks)
1. Implement basic workflow tracking wrapper (`track_workflow()` context manager)
2. Add MLflow tracking to 5-10 existing py-tidymodels notebooks
3. Test with all 23 model types to identify edge cases
4. Document MLflow integration patterns in py-tidymodels docs

### Medium-Term (1-3 Months)
1. Implement WorkflowSet nested run tracking
2. Create visualization functions for common plots (diagnostics, comparisons)
3. Add recipe tracking with PreparedRecipe serialization
4. Write integration tests for MLflow tracking

### Long-Term (3-6 Months)
1. Build custom MLflow plugin for py-tidymodels UI components
2. Create interactive dashboards for multi-model comparison
3. Integrate with production deployment pipelines (Docker, Kubernetes)
4. Write comprehensive user guide and tutorials

---

## Resources

### Essential Links
- **MLflow Quickstart:** https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
- **MLflow sklearn Integration:** https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/guide/
- **MLflow Python API:** https://mlflow.org/docs/latest/python_api/index.html
- **MLflow Model Registry:** https://mlflow.org/docs/latest/model-registry/

### Example Repositories
- **amesar/mlflow-examples:** https://github.com/amesar/mlflow-examples
- **Official MLflow Examples:** https://github.com/mlflow/mlflow/tree/master/examples

### Tutorials
- **5 Tips for MLflow Experiment Tracking:** https://towardsdatascience.com/5-tips-for-mlflow-experiment-tracking-c70ae117b03f
- **GridSearchCV with MLflow:** https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5

---

## Troubleshooting

### Issue: Plotly plots not rendering in MLflow UI
**Solution:** Use `.html` extension when logging Plotly figures:
```python
mlflow.log_figure(plotly_fig, "forecast.html")  # ✓ Correct
mlflow.log_figure(plotly_fig, "forecast.png")   # ✗ Wrong
```

### Issue: MLflow UI not showing nested runs
**Solution:** Check that `nested=True` is set:
```python
with mlflow.start_run(nested=True):  # ✓ Correct
    # ... logging code ...
```

### Issue: Large artifacts slowing down MLflow UI
**Solution:** Use Parquet instead of CSV, or partition large DataFrames:
```python
outputs.to_parquet("/tmp/outputs.parquet")  # 5-10x smaller
mlflow.log_artifact("/tmp/outputs.parquet")
```

### Issue: Can't find logged artifacts
**Solution:** Check artifact URI and ensure proper directory structure:
```python
mlflow.log_artifact("file.csv", "subdir/")  # Logs to subdir/file.csv
```

### Issue: sklearn autolog() not capturing metrics
**Solution:** Enable autolog() BEFORE training:
```python
mlflow.sklearn.autolog()  # Must be before fit()
fit = wf.fit(data)
```

---

## Contact and Support

- **Full Research Report:** `.claude_plans/MLFLOW_INTEGRATION_RESEARCH.json`
- **Detailed Documentation:** `.claude_plans/MLFLOW_INTEGRATION_RESEARCH.md`
- **MLflow Official Docs:** https://mlflow.org/docs/latest/
- **MLflow GitHub Issues:** https://github.com/mlflow/mlflow/issues

For py-tidymodels-specific MLflow questions, refer to the comprehensive research documentation in this directory.
