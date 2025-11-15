# Notebooks 29-32: New Features Demonstration

This document summarizes the 4 comprehensive Jupyter notebooks demonstrating new features in py-tidymodels.

## Created Notebooks

### 29_bayesian_modeling_demo.ipynb (21 KB, 24 cells)
**Bayesian Modeling with PyMC Integration**

**Topics Covered:**
1. Basic Bayesian linear regression with default priors
2. Prior specification (default and custom)
3. All 4 prediction types:
   - `type="numeric"`: Posterior mean predictions
   - `type="conf_int"`: 95% credible intervals
   - `type="posterior"`: Full posterior samples
   - `type="predictive"`: Posterior predictive (includes noise)
4. Convergence diagnostics (Rhat, ESS)
5. Comparing Bayesian vs Frequentist models
6. Coefficient interpretation with credible intervals
7. Model diagnostics and residual analysis
8. Uncertainty quantification for individual predictions

**Use Case:** Sales forecasting with uncertainty quantification

**Key Features Demonstrated:**
- `linear_reg().set_engine("pymc", chains=4, draws=2000)`
- `check_convergence(fit)` for convergence diagnostics
- Custom priors: `prior_intercept="normal(100, 50)"`
- Posterior analysis and credible intervals
- Heteroscedastic uncertainty modeling

**Code Cells:** 14 | **Markdown Cells:** 10

---

### 30_shap_interpretability_demo.ipynb (28 KB, 32 cells)
**SHAP Interpretability for Model Explanations**

**Topics Covered:**
1. SHAP explanations for different model types (linear, tree-based)
2. Auto-explainer selection (TreeExplainer, LinearExplainer, KernelExplainer)
3. Global feature importance (mean |SHAP|)
4. Local explanations for individual predictions
5. SHAP with workflows and recipes
6. Grouped model SHAP (per-group feature importance)
7. Identifying prediction errors with SHAP
8. SHAP dependence plots (feature interactions)

**Use Case:** Customer churn prediction with model explanations

**Key Features Demonstrated:**
- `fit.explain(test_data)` for SHAP computation
- `method="auto"` for auto-explainer selection
- Global importance: `shap_df.groupby('variable')['abs_shap'].mean()`
- Local explanations with waterfall plots
- `nested_fit.explain(test_data)` for grouped models
- SHAP dependence analysis for feature interactions

**Code Cells:** 22 | **Markdown Cells:** 10

---

### 31_backtest_vintages_demo.ipynb (30 KB, 35 cells)
**Backtesting with Data Vintages**

**Topics Covered:**
1. Creating synthetic vintage data with revisions
2. VintageCV setup for time-series cross-validation
3. WorkflowSet backtesting across multiple models
4. Vintage drift analysis (performance over time)
5. Forecast horizon performance degradation
6. Comparing vintage vs final data (demonstrating data leakage)
7. Production forecasting workflow with point-in-time data

**Use Case:** Commodity price forecasting with data revisions

**Key Features Demonstrated:**
- `create_vintage_data()` for synthetic vintages
- `validate_vintage_data()` for data validation
- `VintageCV()` for vintage-aware cross-validation
- `wf_set.fit_backtests()` for multi-model backtesting
- `backtest_results.analyze_vintage_drift('rmse')`
- `backtest_results.analyze_forecast_horizon('rmse')`
- Point-in-time constraints (no future information leakage)

**Code Cells:** 24 | **Markdown Cells:** 11

---

### 32_mlflow_integration_demo.ipynb (29 KB, 35 cells)
**MLflow Integration for Model Lifecycle Management**

**Topics Covered:**
1. Basic save/load workflow for ModelFit
2. Saving workflows with recipes (preprocessing pipelines)
3. Saving grouped/nested models (per-group models)
4. Version compatibility checking
5. Model signatures for input/output validation
6. Loading and deploying models
7. Model comparison across saved artifacts
8. Production deployment workflow

**Use Case:** Model lifecycle management for production deployment

**Key Features Demonstrated:**
- `fit.save_mlflow(path, signature="auto", input_example=data)`
- `load_model(path)` for deserialization
- `get_model_info(path)` for metadata retrieval
- `validate_model_exists(path)` for validation
- Model signatures for schema validation
- Version tracking and compatibility
- Grouped model serialization
- Production model selection and deployment

**Code Cells:** 24 | **Markdown Cells:** 11

---

## Technical Requirements Met

### Notebook Structure (All notebooks)
✅ **Introduction** - Clear explanation of feature and use case
✅ **Setup** - Import libraries, load data
✅ **Basic Usage** - Simple end-to-end example
✅ **Advanced Features** - Demonstrate key capabilities
✅ **Integration** - Show ecosystem integration
✅ **Best Practices** - Tips and recommendations
✅ **Summary** - Key takeaways

### Code Requirements
✅ Realistic synthetic data (not iris/mtcars)
✅ Clear markdown explanations between code cells
✅ Multiple visualizations (matplotlib/seaborn)
✅ Graceful error handling
✅ Follows py-tidymodels conventions

### Data Generation
✅ Realistic business contexts:
   - Sales forecasting (Bayesian, MLflow)
   - Customer churn (SHAP)
   - Commodity prices (Vintages)
✅ Appropriately sized (120-300 rows)
✅ Dates for time series examples
✅ Categorical variables for demonstrations

---

## What Each Notebook Demonstrates

### 29_bayesian_modeling_demo.ipynb
**Core Feature:** Bayesian inference with PyMC backend

**Key Concepts:**
- Posterior distributions vs point estimates
- Credible intervals (95% HDI)
- Prior specification and sensitivity
- Convergence diagnostics (Rhat, ESS)
- Uncertainty quantification
- Comparison with frequentist methods

**When to Use:**
- Need full uncertainty quantification
- Small sample sizes
- Decision-making under uncertainty
- Incorporating prior knowledge

---

### 30_shap_interpretability_demo.ipynb
**Core Feature:** Model-agnostic explanations via SHAP

**Key Concepts:**
- Shapley values for feature attribution
- Global importance (mean |SHAP|)
- Local explanations (individual predictions)
- Auto-explainer selection
- Grouped model interpretability
- Error analysis with SHAP

**When to Use:**
- Need model interpretability
- Regulatory requirements (explainability)
- Feature selection validation
- Error diagnosis
- Comparing heterogeneous models

---

### 31_backtest_vintages_demo.ipynb
**Core Feature:** Production-realistic backtesting

**Key Concepts:**
- Point-in-time data constraints
- Data revision timelines
- Vintage-aware cross-validation
- Concept drift detection
- Forecast horizon analysis
- Avoiding data leakage

**When to Use:**
- Economic/financial forecasting
- Data with revisions (GDP, earnings)
- Production deployment planning
- Realistic performance estimation
- Time series model selection

---

### 32_mlflow_integration_demo.ipynb
**Core Feature:** Model persistence and lifecycle management

**Key Concepts:**
- Model serialization with metadata
- Version tracking and compatibility
- Model signatures for validation
- Workflow/recipe persistence
- Grouped model serialization
- Production deployment workflow

**When to Use:**
- Production model deployment
- Model versioning and tracking
- Cross-team collaboration
- Model registry integration
- Reproducibility requirements

---

## Verification

All notebooks have been created and verified:

```bash
# Created notebooks
examples/29_bayesian_modeling_demo.ipynb    (21 KB)
examples/30_shap_interpretability_demo.ipynb (28 KB)
examples/31_backtest_vintages_demo.ipynb     (30 KB)
examples/32_mlflow_integration_demo.ipynb    (29 KB)

# Cell counts
29: 24 cells (10 markdown, 14 code)
30: 32 cells (10 markdown, 22 code)
31: 35 cells (11 markdown, 24 code)
32: 35 cells (11 markdown, 24 code)
```

## Running the Notebooks

### Prerequisites
```bash
# Activate virtual environment
source py-tidymodels2/bin/activate

# Install dependencies (if needed)
pip install pymc>=5.10.0 arviz>=0.16.0  # For Bayesian modeling
pip install shap>=0.43.0                 # For SHAP interpretability
pip install mlflow>=2.0.0                # For MLflow integration
```

### Launch Jupyter
```bash
jupyter notebook
```

Then navigate to `examples/` and open any of the 4 notebooks (29-32).

### Execution Notes
- Bayesian modeling (Notebook 29) may take 2-5 minutes for MCMC sampling
- SHAP computation (Notebook 30) is fast for tree models, slower for kernel method
- Vintage backtesting (Notebook 31) evaluates multiple workflows across CV folds
- MLflow integration (Notebook 32) saves models locally (no tracking server needed)

---

## Integration with Existing Notebooks

These notebooks complement the existing 28 demonstration notebooks:

**Existing Notebooks (1-28):**
- Core features: hardhat, parsnip, recipes, workflows, yardstick, tune
- 23 model types
- Grouped/nested modeling
- WorkflowSets and model comparison
- Genetic algorithms and NSGA-2

**New Notebooks (29-32):**
- Advanced features: Bayesian inference, interpretability, backtesting, deployment
- Production readiness: Uncertainty quantification, explainability, versioning
- Enterprise capabilities: Model registry, lifecycle management, drift detection

---

## Summary

Four comprehensive Jupyter notebooks have been created to demonstrate the latest features in py-tidymodels:

1. **Bayesian Modeling** - Full uncertainty quantification with PyMC
2. **SHAP Interpretability** - Model-agnostic explanations for any model
3. **Vintage Backtesting** - Production-realistic performance evaluation
4. **MLflow Integration** - Model lifecycle management and deployment

Each notebook includes:
- Clear narrative structure
- Realistic use cases
- End-to-end examples
- Best practices
- Integration with existing ecosystem
- Comprehensive visualizations

All notebooks are ready for use and follow py-tidymodels conventions.
