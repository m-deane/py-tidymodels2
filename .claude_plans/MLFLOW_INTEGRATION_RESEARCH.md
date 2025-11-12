# MLflow Integration Research for py-tidymodels

**Research Date:** 2025-11-12
**Researcher:** Technical Researcher Agent
**Focus:** MLflow integration opportunities for py-tidymodels library

---

## Executive Summary

This research explores how MLflow can enhance the py-tidymodels ecosystem by providing experiment tracking, model versioning, and visualization capabilities. MLflow emerged as the optimal choice due to its open-source nature, strong sklearn integration, and nested run architecture that naturally aligns with py-tidymodels' multi-model comparison patterns.

### Top 5 Benefits

1. **Unified Experiment Tracking Across 23 Model Types**
   - Single interface for all py-tidymodels models (linear_reg, prophet_reg, arima_reg, boost_tree, etc.)
   - Automatic parameter and metric logging via `autolog()` for sklearn, XGBoost, LightGBM, Prophet, pmdarima
   - Critical for exploratory modeling with 20+ workflows

2. **Nested Run Structure for WorkflowSet Multi-Model Comparison**
   - Parent-child run architecture perfectly matches `WorkflowSet.fit_resamples()` pattern
   - Parent run = WorkflowSet experiment, child runs = individual workflow evaluations
   - Natural fit for tracking 100+ models (e.g., 20 workflows × 5 CV folds)

3. **Three-DataFrame Output Persistence**
   - MLflow's `log_table()`, `log_dict()`, and `log_artifact()` persist py-tidymodels' standardized outputs
   - Supports CSV, Parquet, and JSON formats for queryable artifacts
   - Enables post-hoc analysis without re-fitting models

4. **Production-Ready Model Registry**
   - Version control with aliases (champion, challenger, candidate)
   - Seamless deployment via REST API
   - Decouple model versions from serving code

5. **Custom Visualization Dashboard**
   - `log_figure()` supports matplotlib and Plotly interactive plots
   - Transform static plots into explorable visualizations
   - Critical for time series diagnostics and multi-model comparison

---

## Architecture Integration

### Layer 8: py-workflowsets (Multi-Model Comparison)

**Tracking Strategy:** Parent-Child Nested Runs

#### WorkflowSet.fit_resamples()
- **Parent run:** WorkflowSet experiment with all 20 workflows
- **Child runs:** One per workflow evaluation
- **Logged artifacts:**
  - `collect_metrics()` DataFrame (CV performance)
  - `rank_results()` DataFrame (workflow ranking)
  - Per-workflow PreparedRecipe or formula

#### WorkflowSet.fit_nested()
- **Parent run:** WorkflowSet grouped experiment
- **Child runs (Level 1):** Per-workflow runs
- **Child runs (Level 2):** Per-group runs within workflow
- **Logged artifacts:**
  - Per-group outputs, coefficients, stats DataFrames
  - Group comparison metrics
  - Best workflow per group identification

### Layer 4: py-workflows (Single Pipeline)

**Tracking Strategy:** Single Run with Detailed Artifacts

#### Workflow.fit()
- **Parameters:** Model type, hyperparameters, formula, group column
- **Metrics:** Train/test RMSE, MAE, R², per-group metrics
- **Artifacts:** Three-DataFrame outputs, fitted workflow, formula, preprocessed data sample

### Layer 5: py-recipes (Feature Engineering)

**Tracking Strategy:** Parameters + Learned State Artifacts

#### Recipe.prep()
- **Parameters:** Step sequence, each step's parameters, selector functions
- **Artifacts:**
  - PreparedRecipe pickle object
  - PCA components matrix (if `step_pca` used)
  - Feature correlation matrix
  - Imputation values
  - Feature names before/after comparison

### Layer 7: py-tune (Hyperparameter Tuning)

**Tracking Strategy:** Parent + Child Runs for Grid Search

#### tune_grid()
- **Parent run:** Hyperparameter tuning experiment
- **Child runs:** One per parameter combination tested
- **Per-child logs:** Parameter values, CV metrics (mean + std), fitted model
- **Parent-level logs:** `TuneResults.show_best()` DataFrame, grid spec, best parameters, performance plots

---

## Code Examples

### Example 1: Basic Workflow Tracking

```python
import mlflow
import mlflow.sklearn
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Enable sklearn autologging
mlflow.sklearn.autolog()

# Create workflow
rec = recipe(train_data).step_normalize(all_numeric()).step_dummy(all_nominal())
wf = workflow().add_recipe(rec).add_model(linear_reg())

# Track with MLflow
with mlflow.start_run(run_name="linear_reg_normalized"):
    # Log custom tags
    mlflow.set_tags({
        "model_type": "linear_reg",
        "preprocessing": "normalize_dummy",
        "dataset": "sales_2024"
    })

    # Fit workflow (autolog captures sklearn metrics)
    fit = wf.fit(train_data)

    # Evaluate on test data
    fit_eval = fit.evaluate(test_data)

    # Extract outputs
    outputs, coeffs, stats = fit.extract_outputs()

    # Log three-DataFrame outputs
    mlflow.log_table(stats, "model_stats.json")
    mlflow.log_table(coeffs, "coefficients.json")

    # Log outputs as Parquet (more efficient for large data)
    outputs.to_parquet("/tmp/outputs.parquet")
    mlflow.log_artifact("/tmp/outputs.parquet", "model_outputs")

    # Log additional custom metrics
    test_stats = stats[stats['split'] == 'test']
    mlflow.log_metrics({
        "test_rmse": test_stats['rmse'].iloc[0],
        "test_mae": test_stats['mae'].iloc[0],
        "test_r2": test_stats['rsq'].iloc[0]
    })
```

### Example 2: WorkflowSet Multi-Model Comparison

```python
import mlflow
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest

# Define preprocessing strategies and models
formulas = [
    "y ~ x1 + x2",
    "y ~ x1 + x2 + x3",
    "y ~ x1 + x2 + I(x1*x2)",
    "y ~ x1 + x2 + I(x1**2)",
    "y ~ ."
]

models = [
    linear_reg(),
    linear_reg(penalty=0.1, mixture=1.0),  # Lasso
    linear_reg(penalty=0.1, mixture=0.5),  # ElasticNet
    rand_forest().set_mode('regression')
]

# Create WorkflowSet (5 formulas × 4 models = 20 workflows)
wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

# Parent run for entire WorkflowSet experiment
with mlflow.start_run(run_name="WorkflowSet_20_Workflows") as parent_run:
    mlflow.set_tags({
        "experiment_type": "workflowset",
        "n_workflows": 20,
        "preprocessing_strategies": 5,
        "model_types": 4
    })

    # Evaluate all workflows
    folds = vfold_cv(train_data, v=5)
    results = wf_set.fit_resamples(resamples=folds, metrics=metric_set(rmse, mae))

    # Log each workflow as child run
    for wf_id in wf_set.workflow_ids:
        with mlflow.start_run(run_name=wf_id, nested=True):
            # Log workflow-specific parameters
            wf = wf_set[wf_id]
            model_spec = wf.extract_spec_parsnip()
            mlflow.log_params({
                "model_type": model_spec.model_type,
                "formula": wf.extract_formula()
            })

            # Get workflow metrics
            wf_metrics = results.collect_metrics()
            wf_metrics = wf_metrics[wf_metrics['wflow_id'] == wf_id]

            # Log aggregated metrics
            for _, row in wf_metrics.iterrows():
                mlflow.log_metric(row['metric'], row['mean'])
                mlflow.log_metric(f"{row['metric']}_std", row['std'])

    # Log overall results at parent level
    ranked = results.rank_results('rmse', n=20)
    mlflow.log_table(ranked, "workflow_rankings.json")

    # Log best workflow details
    best_wf_id = ranked.iloc[0]['wflow_id']
    mlflow.log_param("best_workflow", best_wf_id)
    mlflow.log_metric("best_rmse", ranked.iloc[0]['mean'])
```

### Example 3: Time Series Diagnostics with Plotly

```python
import mlflow
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from py_parsnip import prophet_reg

# Fit time series model
spec = prophet_reg()
fit = spec.fit(train_data, "y ~ date")

# Extract outputs
outputs, coeffs, stats = fit.extract_outputs()
residuals = outputs[outputs['split'] == 'train']['residuals'].dropna()

with mlflow.start_run(run_name="Prophet_Diagnostics"):
    # Log model parameters
    mlflow.log_params({
        "model_type": "prophet_reg",
        "formula": "y ~ date"
    })

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residual plot
    axes[0, 0].scatter(range(len(residuals)), residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residual Plot')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Residuals')

    # 2. ACF plot
    plot_acf(residuals, ax=axes[0, 1], lags=40)
    axes[0, 1].set_title('ACF of Residuals')

    # 3. PACF plot
    plot_pacf(residuals, ax=axes[1, 0], lags=40, method='ywm')
    axes[1, 0].set_title('PACF of Residuals')

    # 4. Residual distribution
    axes[1, 1].hist(residuals, bins=30, edgecolor='black')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()

    # Log the figure
    mlflow.log_figure(fig, "time_series_diagnostics.png")
    plt.close(fig)

    # Log residual statistics as metrics
    mlflow.log_metrics({
        "residual_mean": residuals.mean(),
        "residual_std": residuals.std(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis()
    })
```

### Example 4: Panel Data with Nested Models

```python
import mlflow
import pandas as pd
from py_workflows import workflow
from py_parsnip import linear_reg

# Create workflow
wf = workflow().add_formula("sales ~ price + promotion").add_model(linear_reg())

# Fit nested models (one per country)
with mlflow.start_run(run_name="Nested_Models_By_Country") as parent_run:
    mlflow.set_tags({
        "model_type": "nested_workflow",
        "group_col": "country",
        "n_groups": train_data['country'].nunique()
    })

    # Fit nested models
    nested_fit = wf.fit_nested(train_data, group_col='country')

    # Extract outputs (includes group column)
    outputs, coeffs, stats = nested_fit.extract_outputs()

    # Log per-group metrics as DataFrame artifact
    per_group_stats = stats[stats['split'] == 'test'].copy()
    mlflow.log_table(per_group_stats, "per_group_performance.json")

    # Create child run for each group
    for group in train_data['country'].unique():
        with mlflow.start_run(run_name=f"country_{group}", nested=True):
            # Get group-specific data
            group_stats = stats[(stats['group'] == group) & (stats['split'] == 'test')].iloc[0]
            group_coeffs = coeffs[coeffs['group'] == group]

            # Log group as tag
            mlflow.set_tag("country", group)

            # Log group-specific metrics
            mlflow.log_metrics({
                "rmse": group_stats['rmse'],
                "mae": group_stats['mae'],
                "r2": group_stats['rsq']
            })

            # Log group-specific coefficients
            coeff_dict = group_coeffs.set_index('term')['estimate'].to_dict()
            mlflow.log_params(coeff_dict)

    # Aggregate metrics at parent level
    avg_metrics = per_group_stats.groupby('metric')[['rmse', 'mae', 'rsq']].mean()
    for metric_name in ['rmse', 'mae', 'rsq']:
        mlflow.log_metric(f"avg_{metric_name}", avg_metrics.loc[metric_name].mean())
```

---

## Custom Visualization Gallery

### Multi-Model Comparison Plots

#### 1. Workflow Performance Bar Chart
**Description:** Horizontal bar chart showing RMSE for all 20 workflows, sorted by performance, color-coded by model type.

**Use Case:** Quick identification of best-performing workflows in WorkflowSet evaluation.

**Implementation:**
```python
fig, ax = plt.subplots(figsize=(10, 8))
ranked = results.rank_results('rmse', n=20)
ax.barh(ranked['wflow_id'], ranked['mean'],
        color=['blue' if 'linear' in x else 'green' for x in ranked['wflow_id']])
ax.set_xlabel('RMSE')
ax.set_title('Workflow Performance Comparison')
mlflow.log_figure(fig, 'workflow_comparison.png')
```

**Priority:** Critical

#### 2. Parameter Sensitivity Heatmap
**Description:** 2D heatmap showing how two hyperparameters (e.g., penalty × mixture) affect performance metric (e.g., RMSE).

**Use Case:** Understanding hyperparameter interactions in `tune_grid()` results.

**Implementation:**
```python
pivot = results.metrics.pivot(index='penalty', columns='mixture', values='rmse')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax)
ax.set_title('RMSE by Penalty and Mixture')
mlflow.log_figure(fig, 'parameter_sensitivity.png')
```

**Priority:** High

#### 3. CV Stability Plot (Mean ± Std)
**Description:** Bar chart with error bars showing mean performance ± standard deviation across CV folds.

**Use Case:** Assessing model stability and identifying workflows with high variance.

**Priority:** Critical

### Time Series Diagnostic Plots

#### 1. Forecast vs Actuals with Prediction Intervals
**Description:** Line plot showing historical actuals, model fit, and future forecasts with 95% confidence intervals.

**Use Case:** Primary visualization for time series model evaluation.

**Implementation:** See Example 3 above (Plotly version in JSON).

**Priority:** Critical

#### 2. Residual Diagnostic 4-Panel
**Description:** 2×2 subplot: residual scatter, ACF, PACF, and residual distribution histogram.

**Use Case:** Comprehensive residual analysis for time series models.

**Priority:** Critical

#### 3. Decomposition Plot (Trend + Seasonal + Residual)
**Description:** Stacked time series showing original data decomposed into trend, seasonal, and residual components.

**Use Case:** Understanding `seasonal_reg` and `exp_smoothing` models.

**Priority:** High

### Panel/Grouped Model Plots

#### 1. Per-Group Performance Heatmap
**Description:** Heatmap showing RMSE for each group (rows) × workflow (columns). Darker = better performance.

**Use Case:** Identifying which workflows perform best for each group in nested modeling.

**Implementation:**
```python
per_group = stats[stats['split'] == 'test'].pivot(index='group', columns='model', values='rmse')
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(per_group, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
ax.set_title('RMSE by Group and Model')
mlflow.log_figure(fig, 'per_group_heatmap.png')
```

**Priority:** Critical

#### 2. Coefficient Variation Across Groups
**Description:** Box plot or violin plot showing distribution of coefficient estimates across groups.

**Use Case:** Understanding parameter heterogeneity in nested models.

**Priority:** High

---

## Key Challenges and Solutions

### Challenge 1: Logging 51 Recipe Preprocessing Steps

**Problem:** py-recipes has 51 step types (step_normalize, step_pca, step_dummy, etc.). Each step has parameters and learned state (e.g., PCA components, normalization means).

**Solution:**
- Use tags to mark recipe type
- Log each step's parameters via `log_params()`
- Persist PreparedRecipe object as pickle artifact via `log_artifact()`
- For complex steps (PCA, ICA), log component matrices as separate DataFrames via `log_table()`

### Challenge 2: Panel/Grouped Model Tracking

**Problem:** py-tidymodels' NestedWorkflowFit trains separate models per group (e.g., 10 countries × 5 models = 50 fitted models). MLflow's native UI doesn't support per-group metric aggregation.

**Solution:**
- Leverage MLflow Diviner integration pattern
- Log per-group metrics as DataFrame artifacts via `log_dict()` or `log_table()`
- Use tags for group identification (country:USA, country:Germany)
- Create custom dashboards using MLflow Search API to query and aggregate

### Challenge 3: Time Series Cross-Validation with Rolling Windows

**Problem:** py-rsample's `time_series_cv()` creates multiple train/test splits with rolling or expanding windows. Need to track performance across all folds while preserving temporal structure.

**Solution:**
- Use nested runs: Parent = workflow evaluation, Children = individual CV folds
- Log fold metrics with step parameter (e.g., `log_metric('rmse', value, step=fold_idx)`)
- Log split boundaries as tags (train_start, train_end, test_start, test_end)

### Challenge 4: Large Output DataFrames

**Problem:** Time series models often produce large outputs DataFrames with actuals, fitted, forecast, residuals for thousands of time steps. Logging as artifacts can be slow and consume storage.

**Solution:**
- Use Parquet format for efficient compression and columnar storage (5-10x smaller than CSV)
- Log summary statistics as metrics (RMSE, MAE) and only log full outputs for best N models
- For panel data, partition outputs by group and log separately

---

## Implementation Roadmap

### Phase 1: Basic Workflow Tracking (1-2 weeks, Low Complexity)

**Deliverables:**
- MLflow integration wrapper for `Workflow.fit()`
- Automatic parameter logging (model type, hyperparameters, formula)
- Automatic metric logging (RMSE, MAE, R² from stats DataFrame)
- Three-DataFrame artifact logging (outputs, coefficients, stats)
- Basic tags (model_type, dataset, split)

**Steps:**
1. Create `mlflow_tracking.py` module in py_workflows
2. Implement `track_workflow()` context manager
3. Add optional `mlflow_tracking` parameter to `Workflow.fit()`
4. Extract parameters from ModelSpec and log via `mlflow.log_params()`
5. Extract metrics from stats DataFrame and log via `mlflow.log_metrics()`
6. Save three DataFrames as CSV/Parquet and log via `mlflow.log_artifact()`
7. Add basic tags (model_type from spec, dataset from user input)
8. Write unit tests for tracking integration
9. Update documentation with tracking examples

### Phase 2: Recipe Artifact Logging and Visualization (2-3 weeks, Medium Complexity)

**Deliverables:**
- PreparedRecipe serialization and logging
- Step parameter extraction and logging
- PCA component matrix logging
- Feature correlation before/after plots
- Feature name comparison DataFrame
- Recipe step sequence as parameter

**Steps:**
1. Add `track_recipe()` function to mlflow_tracking.py
2. Implement recipe step parameter extraction (loop through steps, get params)
3. Serialize PreparedRecipe with pickle and log as artifact
4. For step_pca: extract PCA object, log explained variance metrics, save component matrix
5. For step_normalize: log normalization means/stds as parameters
6. Create visualization functions: `plot_correlation_comparison()`, `plot_pca_scree()`
7. Integrate `track_recipe()` into `Workflow.fit()` when recipe is present
8. Add recipe_steps tag with comma-separated step names
9. Write unit tests for recipe tracking
10. Update documentation with recipe tracking examples

### Phase 3: WorkflowSet Multi-Model Comparison (3-4 weeks, High Complexity)

**Deliverables:**
- Nested runs for `WorkflowSet.fit_resamples()`
- Per-workflow child runs with metrics
- Parent-level aggregation and ranking
- Workflow comparison bar chart
- Parameter sensitivity heatmaps
- CV stability plots (mean ± std)
- Support for `fit_nested()` with per-group tracking

**Steps:**
1. Add `track_workflowset()` context manager to mlflow_tracking.py
2. Modify `WorkflowSet.fit_resamples()` to accept mlflow_tracking parameter
3. Start parent run with WorkflowSet experiment name
4. For each workflow, start nested child run with `mlflow.start_run(nested=True)`
5. Log workflow-specific parameters (model type, formula, hyperparameters)
6. Log workflow-specific metrics (from `collect_metrics()`)
7. At parent level, log aggregated results (`rank_results()` DataFrame)
8. Create visualization functions: `plot_workflow_comparison()`, `plot_parameter_sensitivity()`
9. Integrate with `fit_nested()` for per-group tracking (double-nested runs)
10. Write unit tests for WorkflowSet tracking
11. Update documentation with multi-model comparison examples

### Phase 4: Custom Dashboards and Advanced Visualizations (4-5 weeks, High Complexity)

**Deliverables:**
- Interactive Plotly forecast plots with prediction intervals
- Time series diagnostic 4-panel plots (residuals, ACF, PACF, distribution)
- Decomposition plots (trend + seasonal + residual)
- Per-group performance heatmaps for panel data
- Coefficient variation plots across groups
- Parallel coordinates plots for hyperparameter tuning
- Custom MLflow plugin for py-tidymodels (optional)

**Steps:**
1. Create `visualizations.py` module with plotting functions
2. Implement `plot_forecast_interactive()` using Plotly
3. Implement `plot_time_series_diagnostics()` with matplotlib subplots
4. Implement `plot_decomposition()` for seasonal models
5. Implement `plot_per_group_heatmap()` for panel data
6. Implement `plot_coefficient_variation()` for nested models
7. Implement `plot_hyperparameter_parallel()` using plotly parallel coordinates
8. Integrate plotting functions into tracking workflow (auto-generate plots)
9. Add `mlflow_plots` parameter to control which plots to generate
10. Test all plots with various model types and data scenarios
11. (Optional) Create MLflow plugin for custom UI components
12. Update documentation with visualization gallery

---

## MLflow vs Alternatives

### MLflow
**Pros:**
- Open-source and free
- Strong sklearn integration with autolog()
- Built-in model registry for production deployment
- Flexible artifact storage (local, S3, Azure, GCS)
- REST API for serving models
- Nested runs for hierarchical experiments
- Large community and extensive documentation

**Cons:**
- UI is basic compared to W&B (no native metric grouping)
- Requires infrastructure setup for remote tracking server
- Limited support for grouped/panel data metrics out of the box
- No native support for real-time collaboration features

### Weights & Biases
**Pros:**
- Superior UI with advanced visualizations
- Real-time collaboration features (shared workspaces)
- Hyperparameter sweep automation
- Native support for grouped metrics and dashboards
- Better for large-scale deep learning

**Cons:**
- Not free (pricing can be high for teams)
- Requires internet connection (cloud-hosted)
- Vendor lock-in concerns
- More complex for simple use cases

### Neptune
**Pros:**
- Fast performance (100x faster than W&B for some operations)
- Usage-based pricing (more cost-effective)
- Excellent scalability for large experiments
- Good UI and visualization capabilities

**Cons:**
- Smaller community than MLflow or W&B
- Less documentation and examples
- Not open-source (proprietary platform)

### Recommendation

**MLflow is the best fit for py-tidymodels** due to:
1. Open-source aligns with py-tidymodels philosophy
2. Strong sklearn integration via autolog() works seamlessly with py-parsnip's sklearn engines
3. Nested runs architecture matches WorkflowSet pattern
4. Model registry enables production deployment
5. Self-hosted option avoids vendor lock-in

For teams with budget, W&B could be used alongside MLflow for enhanced visualizations.

---

## Best Practices

1. **Consistent Naming:** Use format `ProjectName_DatasetVersion_Date` for experiments
2. **Tags Everything:** Add metadata tags - model_type, dataset, preprocessing, split
3. **Use Parquet:** Log three-DataFrame outputs as Parquet for efficiency (5-10x smaller than CSV)
4. **Nested Runs:** Use for hierarchical experiments (WorkflowSet, hyperparameter tuning)
5. **Interactive Plots:** Log Plotly figures as .html for explorable visualizations
6. **Per-Group Tracking:** Create separate child runs per group with group tags for panel data
7. **Recipe Artifacts:** Log PreparedRecipe as pickle artifact for reproducibility
8. **Model Registry:** Use aliases (champion, challenger) for production deployment
9. **Compress Artifacts:** Store large artifacts (>10MB) in compressed format (Parquet, pickle)
10. **Use log_table():** For structured data (DataFrames) instead of CSV files

---

## Common Pitfalls

1. Forgetting to set `.html` extension when logging Plotly figures (won't render in UI)
2. Logging too many artifacts per run (slows down MLflow UI)
3. Not using nested runs for hierarchical experiments (poor organization)
4. Using CSV for large DataFrames instead of Parquet (storage bloat)
5. Not tagging runs with metadata (hard to filter and search later)
6. Logging per-fold metrics at parent level instead of child runs (loses granularity)
7. Not using MLflow's autolog() for sklearn models (manual logging is error-prone)
8. Forgetting to log formula or recipe details (hard to reproduce experiments)
9. Not testing MLflow integration with all 23 model types (surprises in production)
10. Not considering MLflow server infrastructure for team use (local tracking has limits)

---

## Reference Links

### Official Documentation
- [MLflow Tracking - Official Guide](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow scikit-learn Integration](https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/guide/)
- [MLflow Prophet Integration](https://mlflow.org/docs/latest/ml/traditional-ml/prophet/guide/)
- [MLflow pmdarima API](https://mlflow.org/docs/latest/python_api/mlflow.pmdarima.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry/)
- [Hyperparameter Tuning with Child Runs](https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/)
- [MLflow Python API Reference](https://mlflow.org/docs/latest/python_api/index.html)

### Tutorials and Blogs
- [Machine Learning Model Development and Deployment with MLflow and Scikit-learn Pipelines](https://towardsdatascience.com/machine-learning-model-development-and-deployment-with-mlflow-and-scikit-learn-pipelines-f658c39e4d58)
- [5 Tips for MLflow Experiment Tracking](https://towardsdatascience.com/5-tips-for-mlflow-experiment-tracking-c70ae117b03f)
- [Bayesian Hyperparameter Optimization with MLflow](https://www.phdata.io/blog/bayesian-hyperparameter-optimization-with-mlflow/)
- [Log Sklearn Pipelines with MLflow & Deploy](https://www.gokhan.io/python/track-sklearn-pipeline-with-mlflow/)
- [ML Lifecycle with MLflow: Forecasting Retail data with Facebook Prophet](https://medium.com/@jagdungu/ml-lifecycle-with-mlflow-forecasting-retail-data-with-facebook-prophet-7244dc05c771)

### GitHub Repositories
- [mlflow/mlflow - Official Repository](https://github.com/mlflow/mlflow)
- [amesar/mlflow-examples](https://github.com/amesar/mlflow-examples)
- [GridSearchCV with MLflow Gist](https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5)

### Comparison Articles
- [Weights & Biases vs MLflow vs Neptune](https://neptune.ai/vs/wandb-mlflow) (2025)
- [Best MLflow Alternatives](https://neptune.ai/blog/best-mlflow-alternatives)
- [Benchmarking Experiment Tracking Frameworks](https://mltraq.com/benchmarks/speed/)

---

## Conclusion

MLflow integration with py-tidymodels offers significant value for experiment tracking, model versioning, and visualization. The nested run architecture naturally aligns with py-tidymodels' multi-model comparison patterns, making it an ideal choice for exploratory modeling workflows.

**Key Recommendations:**
1. Start with Phase 1 (basic workflow tracking) to establish integration patterns
2. Prioritize WorkflowSet tracking (Phase 3) as it provides maximum value for multi-model comparison
3. Invest in custom visualizations (Phase 4) to differentiate py-tidymodels from alternatives
4. Consider creating an MLflow plugin for py-tidymodels-specific UI components
5. Document best practices and common pitfalls to accelerate user adoption

The research provides concrete code examples, implementation roadmaps, and architectural guidance to support successful MLflow integration into the py-tidymodels ecosystem.
