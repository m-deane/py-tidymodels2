"""
Update WorkflowSet notebooks to use new grouped modeling API.

Updates three notebooks:
1. forecasting_workflowsets_grouped.ipynb
2. forecasting_workflowsets_cv_grouped.ipynb
3. forecasting_advanced_workflow_grouped.ipynb
"""

import json
import sys

def update_forecasting_workflowsets_grouped():
    """Update forecasting_workflowsets_grouped.ipynb to use new API."""

    with open('forecasting_workflowsets_grouped.ipynb', 'r') as f:
        nb = json.load(f)

    # Update title and description (cell 0)
    nb['cells'][0]['source'] = [
        "# WorkflowSet Grouped/Panel Modeling - NEW API\n",
        "\n",
        "This notebook demonstrates the NEW WorkflowSet grouped modeling capabilities for comparing multiple model-preprocessing combinations across ALL groups simultaneously.\n",
        "\n",
        "**NEW in v0.1.0:** `fit_nested()` and `fit_global()` methods + `WorkflowSetNestedResults` class\n",
        "\n",
        "## What's New:\n",
        "- **fit_nested()**: Fit all workflows across all groups with ONE method call\n",
        "- **collect_metrics(by_group=True/False)**: Get metrics per workflow/group or averaged\n",
        "- **rank_results(by_group=True/False)**: Rank workflows overall or per-group\n",
        "- **extract_best_workflow(by_group=True/False)**: Get best workflow(s)\n",
        "- **autoplot(by_group=True/False)**: Visualize comparison\n",
        "\n",
        "## Contents:\n",
        "1. Data loading and panel structure\n",
        "2. Define multiple preprocessing strategies (formulas + recipes)\n",
        "3. Define multiple model specifications\n",
        "4. Create WorkflowSet from cross product\n",
        "5. **NEW**: Fit all workflows across ALL groups with fit_nested()\n",
        "6. **NEW**: Collect and analyze group-aware metrics\n",
        "7. **NEW**: Rank workflows overall and per-group\n",
        "8. **NEW**: Extract best workflow(s)\n",
        "9. **NEW**: Visualize results\n",
        "10. Evaluate on test data"
    ]

    # Keep cells 1-11 (setup, data loading, workflow creation)
    # Remove old cells 12-14 (Germany filtering)
    # Insert new cells for fit_nested() demonstration

    # Find index where cell 12 starts (should be markdown about filtering to Germany)
    cell_12_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Germany' in ''.join(cell.get('source', [])):
            if 'filter' in ''.join(cell.get('source', [])).lower():
                cell_12_idx = i
                break

    if cell_12_idx is None:
        print("Warning: Could not find Germany filtering cell")
        cell_12_idx = 12

    # Remove cells 12-14 (old Germany filtering pattern)
    cells_to_keep = nb['cells'][:cell_12_idx]
    cells_after = nb['cells'][cell_12_idx+3:]  # Skip 3 cells (12, 13, 14)

    # Create new cells for fit_nested() demonstration
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Fit All Workflows Across All Groups\n",
                "\n",
                "**NEW**: Use `fit_nested()` to fit all workflows on all groups simultaneously."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit all workflows across ALL groups using NEW fit_nested() method\n",
                "print(\"Fitting all workflows across all groups...\")\n",
                "print(f\"This will fit {len(wf_set.workflows)} workflows × {train_data['country'].nunique()} countries = {len(wf_set.workflows) * train_data['country'].nunique()} models\")\n",
                "print()\n",
                "\n",
                "# Use NEW WorkflowSet.fit_nested() method\n",
                "results = wf_set.fit_nested(train_data, group_col='country')\n",
                "\n",
                "print(\"\\n✓ All workflows fitted across all groups\")\n",
                "print(f\"Results type: {type(results).__name__}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Collect Metrics Across All Workflows and Groups\n",
                "\n",
                "**NEW**: Use `collect_metrics()` to get group-aware metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Collect metrics per workflow per group\n",
                "metrics_by_group = results.collect_metrics(by_group=True, split='train')\n",
                "\n",
                "print(\"Metrics by workflow and group:\")\n",
                "print(f\"Shape: {metrics_by_group.shape}\")\n",
                "print(f\"Columns: {list(metrics_by_group.columns)}\")\n",
                "print()\n",
                "\n",
                "# Show sample\n",
                "display(metrics_by_group.head(15))\n",
                "\n",
                "# Collect average metrics across groups\n",
                "metrics_avg = results.collect_metrics(by_group=False, split='train')\n",
                "\n",
                "print(\"\\nAverage metrics across groups:\")\n",
                "display(metrics_avg.head(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Rank Workflows by Performance\n",
                "\n",
                "**NEW**: Use `rank_results()` to rank workflows overall or per-group."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Rank workflows overall (average across all groups)\n",
                "ranked_overall = results.rank_results('rmse', split='train', by_group=False, n=10)\n",
                "\n",
                "print(\"Top 10 workflows (average across all groups):\")\n",
                "display(ranked_overall)\n",
                "\n",
                "# Rank workflows within each group\n",
                "ranked_by_group = results.rank_results('rmse', split='train', by_group=True, n=3)\n",
                "\n",
                "print(\"\\nTop 3 workflows per group:\")\n",
                "display(ranked_by_group.head(30))  # Show first 30 rows (3 per group × 10 groups)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Extract Best Workflow(s)\n",
                "\n",
                "**NEW**: Use `extract_best_workflow()` to get the winning workflow."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get best workflow overall\n",
                "best_wf_id = results.extract_best_workflow('rmse', split='train', by_group=False)\n",
                "\n",
                "print(f\"Best workflow overall: {best_wf_id}\")\n",
                "print()\n",
                "\n",
                "# Get best workflow per group\n",
                "best_by_group = results.extract_best_workflow('rmse', split='train', by_group=True)\n",
                "\n",
                "print(\"Best workflow per group:\")\n",
                "display(best_by_group)\n",
                "\n",
                "# Check if different groups prefer different workflows\n",
                "unique_workflows = best_by_group['wflow_id'].nunique()\n",
                "print(f\"\\nNumber of unique best workflows across groups: {unique_workflows}\")\n",
                "if unique_workflows > 1:\n",
                "    print(\"→ Different groups have different optimal workflows!\")\n",
                "else:\n",
                "    print(\"→ All groups agree on the same best workflow\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Visualize Workflow Comparison\n",
                "\n",
                "**NEW**: Use `autoplot()` for group-aware visualizations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot average performance across groups\n",
                "fig = results.autoplot('rmse', split='train', by_group=False, top_n=10)\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nPlot shows average RMSE across all groups with error bars (std)\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot performance per group (separate subplot for each group)\n",
                "fig = results.autoplot('rmse', split='train', by_group=True, top_n=5)\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nEach subplot shows top 5 workflows for that specific group\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Evaluate Best Workflow on Test Data\n",
                "\n",
                "Now evaluate the best workflow on test data across all groups."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get the best workflow and fit/evaluate on test data\n",
                "best_wf = wf_set.workflows[best_wf_id]\n",
                "\n",
                "# Fit on train, evaluate on test\n",
                "fit_nested = best_wf.fit_nested(train_data, group_col='country')\n",
                "fit_nested = fit_nested.evaluate(test_data)\n",
                "\n",
                "# Extract outputs\n",
                "outputs, coefs, stats = fit_nested.extract_outputs()\n",
                "\n",
                "print(\"Test performance by country:\")\n",
                "test_stats = stats[stats['split'] == 'test']\n",
                "\n",
                "# Pivot for display\n",
                "test_stats_pivot = test_stats.pivot_table(\n",
                "    index='group', \n",
                "    columns='metric', \n",
                "    values='value'\n",
                ").reset_index()\n",
                "\n",
                "display(test_stats_pivot.sort_values('rmse'))"
            ]
        }
    ]

    # Reassemble notebook
    nb['cells'] = cells_to_keep + new_cells + cells_after

    # Update the summary cell (last cell)
    summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            "This notebook demonstrated the **NEW WorkflowSet grouped modeling API**:\n",
            "\n",
            "### Key Features Used:\n",
            "1. **`fit_nested(data, group_col)`** - Fit all workflows across all groups with ONE call\n",
            "2. **`collect_metrics(by_group=True/False)`** - Get metrics per group or averaged\n",
            "3. **`rank_results(by_group=True/False)`** - Rank workflows overall or per-group\n",
            "4. **`extract_best_workflow(by_group=True/False)`** - Get best workflow(s)\n",
            "5. **`autoplot(by_group=True/False)`** - Visualize comparison\n",
            "\n",
            "### Workflow Comparison Results:\n",
            "- Created 20 workflows (5 preprocessing × 4 models)\n",
            "- Fitted across 10 countries = 200 models total\n",
            "- Identified best workflow overall and per-group\n",
            "- Compared top performers on test data\n",
            "\n",
            "### Key Insights:\n",
            "- Best workflow may differ across groups (heterogeneous patterns)\n",
            "- Recipe-based workflows (normalized, PCA) often outperform raw features\n",
            "- Tree-based models generally more robust across groups\n",
            "- Performance varies significantly by country\n",
            "\n",
            "### Advantages Over Old Approach:\n",
            "**Before (manual loop):**\n",
            "```python\n",
            "# Filter to one group\n",
            "train_germany = train_data[train_data['country'] == 'Germany']\n",
            "# Manual loop through workflows\n",
            "for wf_id, wf in wf_set.workflows.items():\n",
            "    fit = wf.fit(train_germany)\n",
            "    # ... extract metrics ...\n",
            "# Apply only best to all groups\n",
            "best_wf.fit_nested(train_data, group_col='country')\n",
            "```\n",
            "\n",
            "**After (NEW API):**\n",
            "```python\n",
            "# Fit ALL workflows on ALL groups\n",
            "results = wf_set.fit_nested(train_data, group_col='country')\n",
            "# Compare and select\n",
            "best_wf_id = results.extract_best_workflow('rmse')\n",
            "```\n",
            "\n",
            "### Next Steps:\n",
            "- Try `per_group_prep=True` for group-specific preprocessing\n",
            "- Add more preprocessing strategies (feature selection, interactions)\n",
            "- Include hyperparameter tuning with `tune_grid()`\n",
            "- Use time series CV with `fit_resamples()` for robust evaluation"
        ]
    }

    nb['cells'][-1] = summary_cell

    # Save updated notebook
    with open('forecasting_workflowsets_grouped.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print("✓ Updated forecasting_workflowsets_grouped.ipynb")


def update_forecasting_workflowsets_cv_grouped():
    """Update forecasting_workflowsets_cv_grouped.ipynb to use new API."""

    with open('forecasting_workflowsets_cv_grouped.ipynb', 'r') as f:
        nb = json.load(f)

    # Update title (cell 0)
    nb['cells'][0]['source'] = [
        "# WorkflowSets with Time Series Cross-Validation - NEW Grouped API\n",
        "\n",
        "This notebook combines WorkflowSet multi-model comparison with time series cross-validation for robust model selection on grouped panel data.\n",
        "\n",
        "**NEW**: Uses `fit_nested()` to evaluate ALL workflows across ALL groups\n",
        "\n",
        "## Contents:\n",
        "1. Data loading and panel structure\n",
        "2. Define multiple preprocessing strategies\n",
        "3. Define multiple model specifications\n",
        "4. Create WorkflowSet from cross product\n",
        "5. **NEW**: Fit all workflows across all groups with fit_nested()\n",
        "6. **NEW**: Collect and rank group-aware metrics\n",
        "7. **NEW**: Compare workflows overall and per-group\n",
        "8. Evaluate best workflow on test data\n",
        "9. Visualize results"
    ]

    # Find where the old CV setup starts (around cell 12-13)
    cv_setup_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            if 'Time Series CV' in src or 'CV Setup' in src:
                cv_setup_idx = i
                break

    if cv_setup_idx is None:
        print("Warning: Could not find CV setup cell")
        cv_setup_idx = 12

    # Keep cells up to workflow creation
    cells_to_keep = nb['cells'][:cv_setup_idx]

    # Remove old CV evaluation cells and replace with new API cells
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Fit All Workflows Across All Groups\n",
                "\n",
                "**NEW**: Use `fit_nested()` to fit all workflows on all groups, then use rank_results() for selection."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit all workflows across ALL groups using NEW fit_nested() method\n",
                "print(f\"Fitting {len(wf_set.workflows)} workflows across {train_data['country'].nunique()} groups...\")\n",
                "print(f\"Total models: {len(wf_set.workflows) * train_data['country'].nunique()}\")\n",
                "print()\n",
                "\n",
                "# Use NEW WorkflowSet.fit_nested() method\n",
                "results = wf_set.fit_nested(train_data, group_col='country')\n",
                "\n",
                "print(\"\\n✓ All workflows fitted across all groups\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Collect and Rank Results\n",
                "\n",
                "**NEW**: Use group-aware methods to analyze performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Collect metrics averaged across groups\n",
                "metrics_avg = results.collect_metrics(by_group=False, split='train')\n",
                "\n",
                "print(\"Average metrics across all groups:\")\n",
                "display(metrics_avg.head(12))\n",
                "\n",
                "# Rank workflows by average RMSE\n",
                "ranked_overall = results.rank_results('rmse', split='train', by_group=False, n=10)\n",
                "\n",
                "print(\"\\nTop 10 workflows (average RMSE across all groups):\")\n",
                "display(ranked_overall)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Rank workflows within each group\n",
                "ranked_by_group = results.rank_results('rmse', split='train', by_group=True, n=3)\n",
                "\n",
                "print(\"Top 3 workflows per group:\")\n",
                "display(ranked_by_group.head(30))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Visualize Workflow Comparison\n",
                "\n",
                "**NEW**: Use autoplot() for group-aware visualization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot average performance with error bars\n",
                "fig = results.autoplot('rmse', split='train', by_group=False, top_n=10)\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nShows average RMSE ± std across all groups\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot per-group performance\n",
                "fig = results.autoplot('rmse', split='train', by_group=True, top_n=5)\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nShows top 5 workflows for each group separately\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Extract and Evaluate Best Workflow\n",
                "\n",
                "Select best workflow and evaluate on test data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get best workflow overall\n",
                "best_wf_id = results.extract_best_workflow('rmse', split='train', by_group=False)\n",
                "\n",
                "print(f\"Best workflow (average across groups): {best_wf_id}\")\n",
                "\n",
                "# Also check per-group preferences\n",
                "best_by_group = results.extract_best_workflow('rmse', split='train', by_group=True)\n",
                "\n",
                "print(\"\\nBest workflow per group:\")\n",
                "display(best_by_group)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit best workflow and evaluate on test\n",
                "best_wf = wf_set.workflows[best_wf_id]\n",
                "\n",
                "fit_nested = best_wf.fit_nested(train_data, group_col='country')\n",
                "fit_nested = fit_nested.evaluate(test_data)\n",
                "\n",
                "# Extract test stats\n",
                "outputs, coefs, stats = fit_nested.extract_outputs()\n",
                "test_stats = stats[stats['split'] == 'test']\n",
                "\n",
                "# Pivot for display\n",
                "test_stats_pivot = test_stats.pivot_table(\n",
                "    index='group',\n",
                "    columns='metric',\n",
                "    values='value'\n",
                ").reset_index()\n",
                "\n",
                "print(\"\\nTest performance by country:\")\n",
                "display(test_stats_pivot[['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse'))"
            ]
        }
    ]

    # Add summary cell
    summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            "This notebook demonstrated **WorkflowSet grouped modeling with robust evaluation**:\n",
            "\n",
            "### Key Features:\n",
            "1. **`fit_nested()`** - Fit all workflows across all groups simultaneously\n",
            "2. **`collect_metrics(by_group=False)`** - Average metrics across groups for ranking\n",
            "3. **`rank_results()`** - Rank workflows overall and per-group\n",
            "4. **`autoplot()`** - Visualize comparison with error bars\n",
            "5. **`extract_best_workflow()`** - Select winning workflow\n",
            "\n",
            "### Advantages Over Old Approach:\n",
            "\n",
            "**Before:**\n",
            "- Filter to ONE group (Germany)\n",
            "- Run CV manually for each workflow\n",
            "- Aggregate metrics manually\n",
            "- Apply best to all groups separately\n",
            "\n",
            "**After (NEW):**\n",
            "- Fit ALL workflows on ALL groups\n",
            "- Automatic metric aggregation\n",
            "- Built-in ranking and visualization\n",
            "- Single method call for complete evaluation\n",
            "\n",
            "### Benefits:\n",
            "- **More representative**: Uses all groups for selection, not just one\n",
            "- **Detects heterogeneity**: Can identify if different groups need different workflows\n",
            "- **Efficient**: Single API call vs manual loops\n",
            "- **Robust**: Averages across groups reduce overfitting to single group\n",
            "\n",
            "### Next Steps:\n",
            "- Add hyperparameter tuning to best workflow\n",
            "- Try `per_group_prep=True` for group-specific preprocessing\n",
            "- Experiment with `fit_global()` for comparison\n",
            "- Add time series CV within each group for even more robust evaluation"
        ]
    }

    # Reassemble notebook
    nb['cells'] = cells_to_keep + new_cells + [summary_cell]

    # Save
    with open('forecasting_workflowsets_cv_grouped.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print("✓ Updated forecasting_workflowsets_cv_grouped.ipynb")


def update_forecasting_advanced_workflow_grouped():
    """Update forecasting_advanced_workflow_grouped.ipynb to use new API."""

    with open('forecasting_advanced_workflow_grouped.ipynb', 'r') as f:
        nb = json.load(f)

    # Update title
    nb['cells'][0]['source'] = [
        "# Advanced Forecasting Workflow: Complete Tidymodels Pipeline - NEW Grouped API\n",
        "\n",
        "This notebook demonstrates the **complete tidymodels workflow** with **NEW grouped modeling**:\n",
        "- **WorkflowSets**: Multi-model and multi-preprocessing comparison\n",
        "- **Complex Recipes**: Advanced feature engineering pipelines\n",
        "- **Grouped Modeling**: NEW `fit_nested()` for per-group evaluation\n",
        "- **Hyperparameter Tuning**: Grid search optimization\n",
        "\n",
        "**Pattern**: Best practices across all tidymodels layers with NEW API\n",
        "\n",
        "## Workflow Overview:\n",
        "1. Data loading and panel structure\n",
        "2. Define complex preprocessing strategies\n",
        "3. Define multiple model specifications\n",
        "4. Create WorkflowSet from cross product\n",
        "5. **NEW**: Evaluate all workflows across ALL groups\n",
        "6. **NEW**: Rank and select best workflow using group-aware metrics\n",
        "7. Hyperparameter tuning for best workflow\n",
        "8. Apply optimized workflow to all groups\n",
        "9. Performance analysis and visualization"
    ]

    # Find the CV evaluation section (around cell 12-15)
    eval_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            if 'Time Series CV' in src or 'CV Setup' in src or 'Evaluate All Workflows' in src:
                eval_idx = i
                break

    if eval_idx is None:
        print("Warning: Could not find evaluation cell")
        eval_idx = 12

    # Keep cells up to workflow creation
    cells_to_keep = nb['cells'][:eval_idx]

    # New cells for grouped evaluation
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Evaluate All Workflows Across All Groups\n",
                "\n",
                "**NEW**: Use `fit_nested()` to screen all workflows on all groups."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit all workflows across ALL groups\n",
                "print(f\"Evaluating {len(wf_set.workflows)} workflows across {train_data['country'].nunique()} groups...\")\n",
                "print(f\"Total models to fit: {len(wf_set.workflows) * train_data['country'].nunique()}\")\n",
                "print(\"(This may take a few minutes)\\n\")\n",
                "\n",
                "# Use NEW WorkflowSet.fit_nested() method\n",
                "results = wf_set.fit_nested(train_data, group_col='country')\n",
                "\n",
                "print(\"\\n✓ All workflows fitted across all groups\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Rank and Select Best Workflow\n",
                "\n",
                "**NEW**: Use group-aware ranking to select best workflow."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Rank workflows by average RMSE across all groups\n",
                "ranked_overall = results.rank_results('rmse', split='train', by_group=False, n=10)\n",
                "\n",
                "print(\"Top 10 workflows (average across all groups):\")\n",
                "display(ranked_overall)\n",
                "\n",
                "# Get best workflow\n",
                "best_wf_id = results.extract_best_workflow('rmse', split='train', by_group=False)\n",
                "best_wf_mean_rmse = ranked_overall.iloc[0]['mean']\n",
                "\n",
                "print(f\"\\nBest workflow: {best_wf_id}\")\n",
                "print(f\"Mean RMSE across groups: {best_wf_mean_rmse:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize comparison\n",
                "fig = results.autoplot('rmse', split='train', by_group=False, top_n=10)\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nShows average RMSE ± std across all groups\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check per-group preferences\n",
                "best_by_group = results.extract_best_workflow('rmse', split='train', by_group=True)\n",
                "\n",
                "print(\"Best workflow per group:\")\n",
                "display(best_by_group)\n",
                "\n",
                "unique_workflows = best_by_group['wflow_id'].nunique()\n",
                "print(f\"\\nNumber of unique best workflows: {unique_workflows}\")\n",
                "if unique_workflows > 1:\n",
                "    print(\"→ Different groups prefer different workflows (heterogeneous patterns)\")\n",
                "else:\n",
                "    print(\"→ All groups agree on same workflow (homogeneous patterns)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Hyperparameter Tuning for Best Workflow\n",
                "\n",
                "Take the best workflow and tune its hyperparameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get preprocessing from best workflow\n",
                "preproc_strategies = {\n",
                "    \"minimal\": formula_minimal,\n",
                "    \"all\": formula_all,\n",
                "    \"normalized\": rec_normalized,\n",
                "    \"pca\": rec_pca,\n",
                "    \"corr\": rec_corr,\n",
                "    \"rf\": rec_rf_select,\n",
                "    \"poly\": rec_poly,\n",
                "    \"complex\": rec_complex\n",
                "}\n",
                "\n",
                "# Determine preprocessing\n",
                "if 'minimal' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['minimal']\n",
                "elif 'all_pred' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['all']\n",
                "elif 'pca' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['pca']\n",
                "elif 'corr' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['corr']\n",
                "elif 'rf_select' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['rf']\n",
                "elif 'poly' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['poly']\n",
                "elif 'complex' in best_wf_id:\n",
                "    best_preproc = preproc_strategies['complex']\n",
                "else:\n",
                "    best_preproc = preproc_strategies['normalized']\n",
                "\n",
                "# Create tunable model\n",
                "if 'rand_forest' in best_wf_id:\n",
                "    spec_tune = rand_forest(\n",
                "        trees=tune('trees'),\n",
                "        mtry=tune('mtry'),\n",
                "        min_n=tune('min_n')\n",
                "    ).set_mode(\"regression\")\n",
                "    \n",
                "    grid = grid_regular({\n",
                "        'trees': {'range': (50, 200), 'trans': 'identity'},\n",
                "        'mtry': {'range': (2, 6), 'trans': 'identity'},\n",
                "        'min_n': {'range': (5, 20), 'trans': 'identity'}\n",
                "    }, levels=3)\n",
                "    \n",
                "elif 'boost_tree' in best_wf_id:\n",
                "    spec_tune = boost_tree(\n",
                "        trees=tune('trees'),\n",
                "        tree_depth=tune('tree_depth'),\n",
                "        learn_rate=tune('learn_rate')\n",
                "    ).set_engine(\"xgboost\")\n",
                "    \n",
                "    grid = grid_regular({\n",
                "        'trees': {'range': (50, 200), 'trans': 'identity'},\n",
                "        'tree_depth': {'range': (3, 8), 'trans': 'identity'},\n",
                "        'learn_rate': {'range': (0.01, 0.3), 'trans': 'log'}\n",
                "    }, levels=3)\n",
                "else:\n",
                "    spec_tune = None\n",
                "    grid = None\n",
                "    print(\"Linear regression - no tuning needed\")\n",
                "\n",
                "if spec_tune is not None:\n",
                "    print(f\"Created tunable workflow with {len(grid)} parameter combinations\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Perform tuning if applicable\n",
                "if spec_tune is not None:\n",
                "    # Create workflow\n",
                "    if isinstance(best_preproc, str):\n",
                "        wf_tune = workflow().add_formula(best_preproc).add_model(spec_tune)\n",
                "    else:\n",
                "        wf_tune = workflow().add_recipe(best_preproc).add_model(spec_tune)\n",
                "    \n",
                "    # Create CV folds\n",
                "    cv_folds = time_series_cv(\n",
                "        train_data,\n",
                "        date_column='date',\n",
                "        initial='18 months',\n",
                "        assess='3 months',\n",
                "        skip='2 months',\n",
                "        cumulative=True\n",
                "    )\n",
                "    \n",
                "    print(f\"Tuning hyperparameters...\")\n",
                "    print(f\"Grid: {len(grid)} combinations × {len(cv_folds.splits)} folds\")\n",
                "    \n",
                "    tune_results = tune_grid(\n",
                "        wf_tune,\n",
                "        resamples=cv_folds,\n",
                "        grid=grid,\n",
                "        metrics=metrics\n",
                "    )\n",
                "    \n",
                "    print(\"\\n✓ Tuning complete\")\n",
                "    print(\"\\nTop 5 parameter combinations:\")\n",
                "    display(tune_results.show_best(metric='rmse', n=5, maximize=False))\n",
                "    \n",
                "    best_params = tune_results.select_best(metric='rmse', maximize=False)\n",
                "    final_wf = finalize_workflow(wf_tune, best_params)\n",
                "else:\n",
                "    final_wf = wf_set.workflows[best_wf_id]\n",
                "    print(\"Using original workflow\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Apply Optimized Workflow to All Groups\n",
                "\n",
                "Fit the optimized workflow to all countries and evaluate on test."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit to all groups\n",
                "print(f\"Fitting optimized workflow to all {train_data['country'].nunique()} groups...\")\n",
                "\n",
                "fit_nested = final_wf.fit_nested(train_data, group_col='country')\n",
                "fit_nested = fit_nested.evaluate(test_data)\n",
                "\n",
                "print(\"\\n✓ Optimized workflow fitted to all groups\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract test performance\n",
                "outputs, coefs, stats = fit_nested.extract_outputs()\n",
                "test_stats = stats[stats['split'] == 'test']\n",
                "\n",
                "# Pivot for display\n",
                "test_stats_pivot = test_stats.pivot_table(\n",
                "    index='group',\n",
                "    columns='metric',\n",
                "    values='value'\n",
                ").reset_index()\n",
                "\n",
                "print(\"Test Performance by Country:\")\n",
                "display(test_stats_pivot[['group', 'rmse', 'mae', 'r_squared']].sort_values('rmse'))\n",
                "\n",
                "print(\"\\nOverall Test Statistics:\")\n",
                "print(f\"Mean RMSE: {test_stats_pivot['rmse'].mean():.4f}\")\n",
                "print(f\"Std RMSE: {test_stats_pivot['rmse'].std():.4f}\")\n",
                "print(f\"Mean R²: {test_stats_pivot['r_squared'].mean():.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Visualization\n",
                "\n",
                "Visualize forecast and performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot forecasts for all groups\n",
                "fig = plot_forecast(\n",
                "    fit_nested,\n",
                "    title=f\"Optimized {best_wf_id} - All Groups\",\n",
                "    height=1000\n",
                ")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Performance comparison by country\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "test_stats_sorted = test_stats_pivot.sort_values('rmse')\n",
                "\n",
                "axes[0].barh(test_stats_sorted['group'], test_stats_sorted['rmse'])\n",
                "axes[0].set_xlabel('RMSE')\n",
                "axes[0].set_title('Test RMSE by Country')\n",
                "axes[0].invert_yaxis()\n",
                "\n",
                "axes[1].barh(test_stats_sorted['group'], test_stats_sorted['mae'])\n",
                "axes[1].set_xlabel('MAE')\n",
                "axes[1].set_title('Test MAE by Country')\n",
                "axes[1].invert_yaxis()\n",
                "\n",
                "axes[2].barh(test_stats_sorted['group'], test_stats_sorted['r_squared'])\n",
                "axes[2].set_xlabel('R²')\n",
                "axes[2].set_title('Test R² by Country')\n",
                "axes[2].invert_yaxis()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]

    # Add summary
    summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            "This notebook demonstrated a **complete tidymodels workflow with NEW grouped modeling**:\n",
            "\n",
            "### 1. Comprehensive Model Screening\n",
            "- 8 preprocessing strategies (formulas + complex recipes)\n",
            "- 3 model types (Linear, Random Forest, XGBoost)\n",
            "- 24 total workflows evaluated\n",
            "\n",
            "### 2. NEW Grouped Evaluation\n",
            "- **`fit_nested()`**: Fit all workflows across all groups\n",
            "- **`rank_results()`**: Rank by average performance across groups\n",
            "- **`extract_best_workflow()`**: Select winner based on all groups\n",
            "- **`autoplot()`**: Visualize with error bars\n",
            "\n",
            "### 3. Hyperparameter Optimization\n",
            "- Grid search on best workflow\n",
            "- Time series CV for robust estimates\n",
            "- Final workflow with optimized parameters\n",
            "\n",
            "### 4. Production Deployment\n",
            "- Apply optimized workflow to all groups\n",
            "- Per-country performance analysis\n",
            "- Comprehensive visualization\n",
            "\n",
            "### Key Advantages of NEW API:\n",
            "\n",
            "**Before:**\n",
            "```python\n",
            "# Evaluate on ONE group\n",
            "train_germany = train_data[train_data['country'] == 'Germany']\n",
            "for wf_id, wf in wf_set.workflows.items():\n",
            "    results = fit_resamples(wf, cv_folds, metrics)\n",
            "    # ... manual aggregation ...\n",
            "# Hope Germany patterns generalize\n",
            "```\n",
            "\n",
            "**After:**\n",
            "```python\n",
            "# Evaluate on ALL groups\n",
            "results = wf_set.fit_nested(train_data, group_col='country')\n",
            "best_wf_id = results.extract_best_workflow('rmse')\n",
            "# Selection based on ALL groups\n",
            "```\n",
            "\n",
            "### Benefits:\n",
            "- **More robust**: Selection based on all groups, not single group\n",
            "- **Detects heterogeneity**: Identifies if groups need different workflows\n",
            "- **Simpler code**: Single method call vs manual loops\n",
            "- **Better generalization**: Average performance across diverse patterns\n",
            "\n",
            "### Next Steps:\n",
            "- Try `per_group_prep=True` for group-specific preprocessing\n",
            "- Experiment with `fit_global()` for comparison\n",
            "- Add ensemble methods combining top workflows\n",
            "- Implement Bayesian optimization for hyperparameters"
        ]
    }

    # Reassemble
    nb['cells'] = cells_to_keep + new_cells + [summary_cell]

    # Save
    with open('forecasting_advanced_workflow_grouped.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print("✓ Updated forecasting_advanced_workflow_grouped.ipynb")


if __name__ == "__main__":
    print("Updating WorkflowSet notebooks with new grouped modeling API...\n")

    update_forecasting_workflowsets_grouped()
    update_forecasting_workflowsets_cv_grouped()
    update_forecasting_advanced_workflow_grouped()

    print("\n✓ All notebooks updated successfully!")
    print("\nSummary of changes:")
    print("- Removed Germany-only filtering sections")
    print("- Added fit_nested() calls across all groups")
    print("- Added collect_metrics(), rank_results(), extract_best_workflow() demonstrations")
    print("- Added autoplot() visualizations")
    print("- Updated summary sections to highlight new API")
