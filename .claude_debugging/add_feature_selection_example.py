"""
Script to add comprehensive feature selection example to forecasting_recipes_grouped.ipynb
"""

import json

notebook_path = "_md/forecasting_recipes_grouped.ipynb"

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Cells to add at the end
new_cells = []

# 1. Markdown header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "# Feature Selection with Per-Group Preprocessing\n",
        "\n",
        "This section demonstrates how different feature selection methods can select **different features for each group** when using `per_group_prep=True`. This is powerful for grouped data where different variables may be important for different groups.\n",
        "\n",
        "We'll demonstrate:\n",
        "1. **step_select_permutation()** - Permutation importance-based selection\n",
        "2. **step_select_shap()** - SHAP value-based selection\n",
        "3. **step_safe_v2()** - SAFE feature engineering + selection\n",
        "4. **step_filter_rf_importance()** - Random Forest importance filtering\n",
        "\n",
        "**Key Point**: With `per_group_prep=True`, each group's recipe will select features independently based on that group's data patterns."
    ]
})

# 2. Setup code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import feature selection steps\n",
        "from py_recipes import (\n",
        "    step_select_permutation,\n",
        "    step_select_shap, \n",
        "    step_safe_v2,\n",
        "    step_filter_rf_importance\n",
        ")\n",
        "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
        "\n",
        "print(\"✓ Feature selection imports complete\")"
    ]
})

# 3. Markdown - Permutation
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Permutation Importance Feature Selection\n",
        "\n",
        "`step_select_permutation()` uses permutation importance to rank features by how much model performance degrades when that feature is shuffled.\n",
        "\n",
        "**How it works with grouped data**:\n",
        "- Each group's recipe fits a model on that group's training data\n",
        "- Calculates permutation importance for each feature\n",
        "- Selects top features based on `top_n` or `top_p` threshold\n",
        "- Different groups may select different features"
    ]
})

# 4. Code - Permutation
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create recipe with permutation importance feature selection\n",
        "rec_perm = (\n",
        "    recipe()\n",
        "    .step_normalize()  # Normalize first\n",
        "    .step_select_permutation(\n",
        "        outcome='refinery_kbd',\n",
        "        model=RandomForestRegressor(n_estimators=50, random_state=42),\n",
        "        top_n=3,  # Select top 3 features per group\n",
        "        n_repeats=5,  # Repeat permutation 5 times for stability\n",
        "        random_state=42\n",
        "    )\n",
        ")\n",
        "\n",
        "# Create workflow\n",
        "wf_perm = (\n",
        "    workflow()\n",
        "    .add_recipe(rec_perm)\n",
        "    .add_model(linear_reg().set_engine(\"sklearn\"))\n",
        "    .add_model_name(\"permutation_selection\")\n",
        "    .add_model_group_name(\"feature_selection_models\")\n",
        ")\n",
        "\n",
        "# Fit with per-group preprocessing\n",
        "print(\"Fitting model with permutation importance feature selection...\")\n",
        "fit_perm = wf_perm.fit_nested(train_data, group_col='country')\n",
        "fit_perm = fit_perm.evaluate(test_data)\n",
        "\n",
        "print(\"\\n✓ Permutation importance model fitted successfully!\")\n",
        "\n",
        "# Extract preprocessed data to see which features were selected per group\n",
        "processed_perm = fit_perm.extract_preprocessed_data(train_data, split='train')\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Features Selected Per Group (Permutation Importance):\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "for group in processed_perm['country'].unique():\n",
        "    group_data = processed_perm[processed_perm['country'] == group]\n",
        "    # Get feature columns (exclude date, country, refinery_kbd, split)\n",
        "    feature_cols = [col for col in group_data.columns \n",
        "                   if col not in ['date', 'country', 'refinery_kbd', 'split']]\n",
        "    print(f\"\\n{group}:\")\n",
        "    print(f\"  Features: {feature_cols}\")\n",
        "    print(f\"  Count: {len(feature_cols)}\")\n",
        "\n",
        "# Show performance\n",
        "_, _, stats_perm = fit_perm.extract_outputs()\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Model Performance:\")\n",
        "print(\"=\"*70)\n",
        "print(stats_perm[['country', 'split', 'rmse', 'mae', 'r_squared']].to_string(index=False))"
    ]
})

# 5. Markdown - SHAP
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. SHAP Value Feature Selection\n",
        "\n",
        "`step_select_shap()` uses SHAP (SHapley Additive exPlanations) values to determine feature importance based on game theory.\n",
        "\n",
        "**How it works with grouped data**:\n",
        "- Each group's recipe fits a model and calculates SHAP values\n",
        "- Features are ranked by mean absolute SHAP value\n",
        "- Top features are selected based on `top_n`, `top_p`, or `threshold`\n",
        "- Groups may identify different important features"
    ]
})

# 6. Code - SHAP
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create recipe with SHAP-based feature selection\n",
        "rec_shap = (\n",
        "    recipe()\n",
        "    .step_normalize()  # Normalize first\n",
        "    .step_select_shap(\n",
        "        outcome='refinery_kbd',\n",
        "        model=GradientBoostingRegressor(n_estimators=50, random_state=42),\n",
        "        top_n=4,  # Select top 4 features per group\n",
        "        shap_samples=100,  # Use 100 samples for SHAP calculation (faster)\n",
        "        random_state=42\n",
        "    )\n",
        ")\n",
        "\n",
        "# Create workflow\n",
        "wf_shap = (\n",
        "    workflow()\n",
        "    .add_recipe(rec_shap)\n",
        "    .add_model(linear_reg().set_engine(\"sklearn\"))\n",
        "    .add_model_name(\"shap_selection\")\n",
        "    .add_model_group_name(\"feature_selection_models\")\n",
        ")\n",
        "\n",
        "# Fit with per-group preprocessing\n",
        "print(\"Fitting model with SHAP-based feature selection...\")\n",
        "fit_shap = wf_shap.fit_nested(train_data, group_col='country')\n",
        "fit_shap = fit_shap.evaluate(test_data)\n",
        "\n",
        "print(\"\\n✓ SHAP-based model fitted successfully!\")\n",
        "\n",
        "# Extract preprocessed data to see which features were selected per group\n",
        "processed_shap = fit_shap.extract_preprocessed_data(train_data, split='train')\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Features Selected Per Group (SHAP Values):\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "for group in processed_shap['country'].unique():\n",
        "    group_data = processed_shap[processed_shap['country'] == group]\n",
        "    feature_cols = [col for col in group_data.columns \n",
        "                   if col not in ['date', 'country', 'refinery_kbd', 'split']]\n",
        "    print(f\"\\n{group}:\")\n",
        "    print(f\"  Features: {feature_cols}\")\n",
        "    print(f\"  Count: {len(feature_cols)}\")\n",
        "\n",
        "# Show performance\n",
        "_, _, stats_shap = fit_shap.extract_outputs()\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Model Performance:\")\n",
        "print(\"=\"*70)\n",
        "print(stats_shap[['country', 'split', 'rmse', 'mae', 'r_squared']].to_string(index=False))"
    ]
})

# 7. Markdown - SAFE
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. SAFE Feature Engineering + Selection\n",
        "\n",
        "`step_safe_v2()` performs **Surrogate Assisted Feature Extraction** - it creates interpretable transformations of features (thresholds, bins) and selects the most important ones.\n",
        "\n",
        "**How it works with grouped data**:\n",
        "- Each group's recipe fits a surrogate model\n",
        "- Creates partial dependence plots to find optimal feature thresholds\n",
        "- Generates binary features based on thresholds (e.g., `brent_gt_50`, `wti_lt_70`)\n",
        "- Selects top transformed features based on importance\n",
        "- Different groups may have different thresholds and transformations\n",
        "\n",
        "**Note**: SAFE is computationally intensive, so we use `top_n` to limit features."
    ]
})

# 8. Code - SAFE
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create recipe with SAFE v2 feature engineering + selection\n",
        "rec_safe = (\n",
        "    recipe()\n",
        "    .step_normalize()  # Normalize first\n",
        "    .step_safe_v2(\n",
        "        surrogate_model=GradientBoostingRegressor(n_estimators=50, random_state=42),\n",
        "        outcome='refinery_kbd',\n",
        "        penalty=10.0,  # Changepoint penalty (higher = fewer thresholds)\n",
        "        top_n=5,  # Select top 5 transformed features per group\n",
        "        max_thresholds=3,  # Max 3 thresholds per feature\n",
        "        keep_original_cols=False,  # Only keep transformed features\n",
        "        feature_type='numeric'  # Only numeric features\n",
        "    )\n",
        ")\n",
        "\n",
        "# Create workflow\n",
        "wf_safe = (\n",
        "    workflow()\n",
        "    .add_recipe(rec_safe)\n",
        "    .add_model(linear_reg().set_engine(\"sklearn\"))\n",
        "    .add_model_name(\"safe_selection\")\n",
        "    .add_model_group_name(\"feature_selection_models\")\n",
        ")\n",
        "\n",
        "# Fit with per-group preprocessing\n",
        "print(\"Fitting model with SAFE feature engineering + selection...\")\n",
        "print(\"(This may take a minute due to PDP calculations)\")\n",
        "fit_safe = wf_safe.fit_nested(train_data, group_col='country')\n",
        "fit_safe = fit_safe.evaluate(test_data)\n",
        "\n",
        "print(\"\\n✓ SAFE-based model fitted successfully!\")\n",
        "\n",
        "# Extract preprocessed data to see which features were created per group\n",
        "processed_safe = fit_safe.extract_preprocessed_data(train_data, split='train')\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Features Created Per Group (SAFE Transformations):\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "for group in processed_safe['country'].unique():\n",
        "    group_data = processed_safe[processed_safe['country'] == group]\n",
        "    feature_cols = [col for col in group_data.columns \n",
        "                   if col not in ['date', 'country', 'refinery_kbd', 'split']]\n",
        "    print(f\"\\n{group}:\")\n",
        "    print(f\"  Features: {feature_cols}\")\n",
        "    print(f\"  Count: {len(feature_cols)}\")\n",
        "    print(f\"  (Note: SAFE creates threshold-based features like 'brent_gt_50', 'wti_lt_70')\")\n",
        "\n",
        "# Show performance\n",
        "_, _, stats_safe = fit_safe.extract_outputs()\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Model Performance:\")\n",
        "print(\"=\"*70)\n",
        "print(stats_safe[['country', 'split', 'rmse', 'mae', 'r_squared']].to_string(index=False))"
    ]
})

# 9. Markdown - RF Importance
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Random Forest Importance Feature Selection\n",
        "\n",
        "`step_filter_rf_importance()` uses Random Forest's built-in feature importance (mean decrease in impurity) to select features.\n",
        "\n",
        "**How it works with grouped data**:\n",
        "- Each group's recipe fits a Random Forest on that group's data\n",
        "- Features are ranked by importance scores\n",
        "- Top features are selected based on `top_n`, `top_p`, or `threshold`\n",
        "- Fast and reliable for initial feature screening"
    ]
})

# 10. Code - RF Importance
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create recipe with RF importance feature selection\n",
        "rec_rf = (\n",
        "    recipe()\n",
        "    .step_normalize()  # Normalize first\n",
        "    .step_filter_rf_importance(\n",
        "        outcome='refinery_kbd',\n",
        "        top_n=3,  # Select top 3 features per group\n",
        "        n_estimators=100,  # RF trees\n",
        "        random_state=42\n",
        "    )\n",
        ")\n",
        "\n",
        "# Create workflow\n",
        "wf_rf = (\n",
        "    workflow()\n",
        "    .add_recipe(rec_rf)\n",
        "    .add_model(linear_reg().set_engine(\"sklearn\"))\n",
        "    .add_model_name(\"rf_importance_selection\")\n",
        "    .add_model_group_name(\"feature_selection_models\")\n",
        ")\n",
        "\n",
        "# Fit with per-group preprocessing\n",
        "print(\"Fitting model with RF importance feature selection...\")\n",
        "fit_rf = wf_rf.fit_nested(train_data, group_col='country')\n",
        "fit_rf = fit_rf.evaluate(test_data)\n",
        "\n",
        "print(\"\\n✓ RF importance model fitted successfully!\")\n",
        "\n",
        "# Extract preprocessed data to see which features were selected per group\n",
        "processed_rf = fit_rf.extract_preprocessed_data(train_data, split='train')\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Features Selected Per Group (RF Importance):\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "for group in processed_rf['country'].unique():\n",
        "    group_data = processed_rf[processed_rf['country'] == group]\n",
        "    feature_cols = [col for col in group_data.columns \n",
        "                   if col not in ['date', 'country', 'refinery_kbd', 'split']]\n",
        "    print(f\"\\n{group}:\")\n",
        "    print(f\"  Features: {feature_cols}\")\n",
        "    print(f\"  Count: {len(feature_cols)}\")\n",
        "\n",
        "# Show performance\n",
        "_, _, stats_rf = fit_rf.extract_outputs()\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Model Performance:\")\n",
        "print(\"=\"*70)\n",
        "print(stats_rf[['country', 'split', 'rmse', 'mae', 'r_squared']].to_string(index=False))"
    ]
})

# 11. Markdown - Comparison
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Comparison of Feature Selection Methods\n",
        "\n",
        "Let's compare how different feature selection methods performed and what features they selected for each group."
    ]
})

# 12. Code - Comparison
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "\n",
        "# Combine all stats\n",
        "all_stats = pd.concat([\n",
        "    stats_perm[['country', 'model', 'split', 'rmse', 'mae', 'r_squared']],\n",
        "    stats_shap[['country', 'model', 'split', 'rmse', 'mae', 'r_squared']],\n",
        "    stats_safe[['country', 'model', 'split', 'rmse', 'mae', 'r_squared']],\n",
        "    stats_rf[['country', 'model', 'split', 'rmse', 'mae', 'r_squared']]\n",
        "], ignore_index=True)\n",
        "\n",
        "print(\"=\"*90)\n",
        "print(\"Performance Comparison: Feature Selection Methods\")\n",
        "print(\"=\"*90)\n",
        "\n",
        "# Show test set performance\n",
        "test_stats = all_stats[all_stats['split'] == 'test'].copy()\n",
        "test_stats = test_stats.sort_values(['country', 'rmse'])\n",
        "\n",
        "print(\"\\nTest Set Performance (sorted by RMSE):\")\n",
        "print(test_stats.to_string(index=False))\n",
        "\n",
        "# Summary by method\n",
        "print(\"\\n\" + \"=\"*90)\n",
        "print(\"Average Performance by Method (across all groups):\")\n",
        "print(\"=\"*90)\n",
        "summary = test_stats.groupby('model')[['rmse', 'mae', 'r_squared']].mean()\n",
        "summary = summary.sort_values('rmse')\n",
        "print(summary)\n",
        "\n",
        "print(\"\\n\" + \"=\"*90)\n",
        "print(\"Feature Selection Summary\")\n",
        "print(\"=\"*90)\n",
        "\n",
        "# Show which features were selected by each method\n",
        "for method_name, processed_data in [\n",
        "    ('Permutation Importance', processed_perm),\n",
        "    ('SHAP Values', processed_shap),\n",
        "    ('SAFE Transformations', processed_safe),\n",
        "    ('RF Importance', processed_rf)\n",
        "]:\n",
        "    print(f\"\\n{method_name}:\")\n",
        "    for group in processed_data['country'].unique():\n",
        "        group_data = processed_data[processed_data['country'] == group]\n",
        "        feature_cols = [col for col in group_data.columns \n",
        "                       if col not in ['date', 'country', 'refinery_kbd', 'split']]\n",
        "        print(f\"  {group}: {len(feature_cols)} features - {', '.join(feature_cols[:3])}{'...' if len(feature_cols) > 3 else ''}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*90)\n",
        "print(\"Key Insights:\")\n",
        "print(\"=\"*90)\n",
        "print(\"1. Different methods select different features for each group\")\n",
        "print(\"2. SAFE creates interpretable threshold-based features (e.g., 'brent_gt_50')\")\n",
        "print(\"3. Permutation and SHAP tend to select similar features (model-agnostic)\")\n",
        "print(\"4. RF importance is fastest but may be biased toward high-cardinality features\")\n",
        "print(\"5. With per_group_prep=True, each group can have completely different feature sets\")"
    ]
})

# 13. Markdown - Conclusion
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Key Takeaways\n",
        "\n",
        "### Per-Group Feature Selection Benefits\n",
        "\n",
        "1. **Adaptive to Group Patterns**: Each group selects features most relevant to its data\n",
        "2. **Improved Interpretability**: Different groups may have different drivers\n",
        "3. **Better Performance**: Group-specific features can improve accuracy\n",
        "\n",
        "### Method Selection Guide\n",
        "\n",
        "| Method | Speed | Interpretability | Best For |\n",
        "|--------|-------|------------------|----------|\n",
        "| **step_filter_rf_importance()** | ⚡⚡⚡ Fast | ⭐⭐ Good | Initial screening, high-dimensional data |\n",
        "| **step_select_permutation()** | ⚡⚡ Medium | ⭐⭐⭐ Very Good | Model-agnostic importance, any model type |\n",
        "| **step_select_shap()** | ⚡ Slow | ⭐⭐⭐ Very Good | Detailed explanations, tree-based models |\n",
        "| **step_safe_v2()** | 🐌 Very Slow | ⭐⭐⭐⭐ Excellent | Interpretable rules, threshold discovery |\n",
        "\n",
        "### Recommendations\n",
        "\n",
        "1. **Start with RF importance** for quick screening\n",
        "2. **Use permutation importance** for robust, model-agnostic selection\n",
        "3. **Use SHAP** for detailed explanations and feature interactions\n",
        "4. **Use SAFE** when you need interpretable thresholds and rules\n",
        "\n",
        "### Next Steps\n",
        "\n",
        "- Try combining multiple selection methods in sequence\n",
        "- Experiment with different `top_n` values per group\n",
        "- Compare per-group vs shared preprocessing performance\n",
        "- Use `.extract_preprocessed_data()` to inspect selected features"
    ]
})

# Add all cells to the notebook
for cell in new_cells:
    notebook['cells'].append(cell)

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Successfully added {len(new_cells)} cells to notebook")
print(f"  Total cells now: {len(notebook['cells'])}")
print(f"\nCells added:")
print(f"  - 1 header markdown")
print(f"  - 1 setup code cell")
print(f"  - 2 cells per method × 4 methods = 8 cells")
print(f"  - 1 markdown + 1 code for comparison = 2 cells")
print(f"  - 1 conclusion markdown")
print(f"  Total: 13 cells")
