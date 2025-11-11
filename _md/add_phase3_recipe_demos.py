"""
Script to add Phase 3 advanced selection step demonstrations to forecasting_recipes_grouped.ipynb
"""

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

# Load the notebook
nb_path = '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_recipes_grouped.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Define the 7 Phase 3 recipe step demonstrations
demos = [
    {
        "title": "### Phase 3 Step 1: VIF-Based Multicollinearity Removal",
        "description": """**`step_vif()`** - Iteratively removes features with high VIF (Variance Inflation Factor) to reduce multicollinearity.

**Use case**: Remove highly correlated predictors that cause instability in coefficient estimates.""",
        "code": """# Create recipe with VIF multicollinearity removal
rec_vif = (
    recipe()
    .step_normalize()  # Normalize first
    .step_vif(
        threshold=10.0  # Remove features with VIF > 10
    )
)

# Create workflow
wf_vif = (
    workflow()
    .add_recipe(rec_vif)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("vif_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with VIF multicollinearity removal...")
fit_vif = wf_vif.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_vif = fit_vif.evaluate(test_data)

outputs, coefs, stats = fit_vif.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast for a sample group
fig = plot_forecast(fit_vif, title="VIF Multicollinearity Removal")
fig.show()

# Show which features were kept per group
feature_comparison = fit_vif.get_feature_comparison()
print("\\nFeatures selected per group (VIF filtering):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 2: Statistical Significance Selection (P-Value)",
        "description": """**`step_pvalue()`** - Selects features based on statistical significance (p-values) from OLS regression.

**Use case**: Keep only features with significant relationships to the outcome variable.""",
        "code": """# Create recipe with p-value based selection
rec_pvalue = (
    recipe()
    .step_normalize()  # Normalize first
    .step_pvalue(
        outcome='refinery_kbd',
        threshold=0.05  # Keep features with p < 0.05
    )
)

# Create workflow
wf_pvalue = (
    workflow()
    .add_recipe(rec_pvalue)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("pvalue_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with p-value based selection...")
fit_pvalue = wf_pvalue.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_pvalue = fit_pvalue.evaluate(test_data)

outputs, coefs, stats = fit_pvalue.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_pvalue, title="P-Value Based Selection")
fig.show()

# Show which features were kept per group
feature_comparison = fit_pvalue.get_feature_comparison()
print("\\nFeatures selected per group (p-value < 0.05):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 3: Bootstrap Stability Selection",
        "description": """**`step_select_stability()`** - Uses bootstrap resampling to identify stably important features.

**Use case**: Select features that are consistently important across different data samples (more robust than single-model importance).""",
        "code": """# Create recipe with stability selection
rec_stability = (
    recipe()
    .step_normalize()  # Normalize first
    .step_select_stability(
        outcome='refinery_kbd',
        n_bootstrap=20,  # Number of bootstrap samples
        threshold=0.6  # Keep features selected in >60% of bootstraps
    )
)

# Create workflow
wf_stability = (
    workflow()
    .add_recipe(rec_stability)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("stability_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with bootstrap stability selection...")
fit_stability = wf_stability.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_stability = fit_stability.evaluate(test_data)

outputs, coefs, stats = fit_stability.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_stability, title="Bootstrap Stability Selection")
fig.show()

# Show which features were kept per group
feature_comparison = fit_stability.get_feature_comparison()
print("\\nFeatures selected per group (stable across bootstraps):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 4: Leave-One-Feature-Out (LOFO) Importance",
        "description": """**`step_select_lofo()`** - Measures feature importance by the performance drop when each feature is removed.

**Use case**: Identify truly important features based on their contribution to model performance.""",
        "code": """# Create recipe with LOFO importance selection
rec_lofo = (
    recipe()
    .step_normalize()  # Normalize first
    .step_select_lofo(
        outcome='refinery_kbd',
        top_n=10  # Keep top 10 most important features
    )
)

# Create workflow
wf_lofo = (
    workflow()
    .add_recipe(rec_lofo)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("lofo_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with LOFO importance selection...")
fit_lofo = wf_lofo.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_lofo = fit_lofo.evaluate(test_data)

outputs, coefs, stats = fit_lofo.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_lofo, title="LOFO Importance Selection")
fig.show()

# Show which features were kept per group
feature_comparison = fit_lofo.get_feature_comparison()
print("\\nFeatures selected per group (top LOFO importance):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 5: Granger Causality Selection",
        "description": """**`step_select_granger()`** - Selects features based on Granger causality tests for time series.

**Use case**: Identify lagged predictors that have predictive power for future outcomes (time series context).""",
        "code": """# Create recipe with Granger causality selection
rec_granger = (
    recipe()
    .step_normalize()  # Normalize first
    .step_select_granger(
        outcome='refinery_kbd',
        max_lag=5,  # Test up to 5 lags
        alpha=0.05  # Significance level
    )
)

# Create workflow
wf_granger = (
    workflow()
    .add_recipe(rec_granger)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("granger_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with Granger causality selection...")
fit_granger = wf_granger.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_granger = fit_granger.evaluate(test_data)

outputs, coefs, stats = fit_granger.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_granger, title="Granger Causality Selection")
fig.show()

# Show which features were kept per group
feature_comparison = fit_granger.get_feature_comparison()
print("\\nFeatures selected per group (Granger causality):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 6: Stepwise Selection (Forward/Backward/Bidirectional)",
        "description": """**`step_select_stepwise()`** - Performs stepwise feature selection using AIC/BIC criteria.

**Use case**: Build parsimonious models by iteratively adding/removing features based on information criteria.""",
        "code": """# Create recipe with stepwise selection
rec_stepwise = (
    recipe()
    .step_normalize()  # Normalize first
    .step_select_stepwise(
        outcome='refinery_kbd',
        direction='both',  # 'forward', 'backward', or 'both'
        criterion='aic'  # 'aic' or 'bic'
    )
)

# Create workflow
wf_stepwise = (
    workflow()
    .add_recipe(rec_stepwise)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("stepwise_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with stepwise selection...")
fit_stepwise = wf_stepwise.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_stepwise = fit_stepwise.evaluate(test_data)

outputs, coefs, stats = fit_stepwise.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_stepwise, title="Stepwise Selection (AIC)")
fig.show()

# Show which features were kept per group
feature_comparison = fit_stepwise.get_feature_comparison()
print("\\nFeatures selected per group (stepwise AIC):")
display(feature_comparison)"""
    },
    {
        "title": "### Phase 3 Step 7: Random Probe Threshold Selection",
        "description": """**`step_select_probe()`** - Uses random noise features (probes) to determine importance threshold.

**Use case**: Objectively determine feature importance threshold by comparing against random noise.""",
        "code": """# Create recipe with probe-based selection
rec_probe = (
    recipe()
    .step_normalize()  # Normalize first
    .step_select_probe(
        outcome='refinery_kbd',
        n_probes=5  # Number of random probe features to generate
    )
)

# Create workflow
wf_probe = (
    workflow()
    .add_recipe(rec_probe)
    .add_model(linear_reg().set_engine("sklearn"))
    .add_model_name("probe_selection")
    .add_model_group_name("feature_selection_models")
)

# Fit with per-group preprocessing
print("Fitting model with random probe threshold selection...")
fit_probe = wf_probe.fit_nested(train_data, group_col='country', per_group_prep=True)
fit_probe = fit_probe.evaluate(test_data)

outputs, coefs, stats = fit_probe.extract_outputs()

# Display test performance by group
test_stats = stats[stats['split'] == 'test'][['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse')
print("\\nTest Performance by Group:")
display(test_stats)

# Plot forecast
fig = plot_forecast(fit_probe, title="Random Probe Threshold Selection")
fig.show()

# Show which features were kept per group
feature_comparison = fit_probe.get_feature_comparison()
print("\\nFeatures selected per group (above probe threshold):")
display(feature_comparison)"""
    }
]

# Add new cells at the end of the notebook
for demo in demos:
    # Add markdown cell with title and description
    nb.cells.append(new_markdown_cell(f"{demo['title']}\\n\\n{demo['description']}"))

    # Add code cell
    nb.cells.append(new_code_cell(demo['code']))

# Save the updated notebook
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print(f"✅ Added {len(demos)} Phase 3 recipe step demonstrations to the notebook")
print(f"✅ Total cells in notebook: {len(nb.cells)}")
