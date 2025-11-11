# Grouped/Panel Data Notebook Update Summary

**File:** `_md/forecasting_grouped.ipynb`
**Date:** 2025-11-09
**Status:** Complete - All updates applied successfully

## Overview

Updated the forecasting notebook to properly work with grouped/panel data (multiple countries). All workflow demonstrations and visualizations now use `fit_nested()` for per-country model fitting and include appropriate country-specific visualizations.

## Update Statistics

### Notebook Structure
- **Total cells:** 138 (increased from 94)
- **Code cells:** 107 (increased from 63)
- **Markdown cells:** 31 (unchanged)

### Updates Applied

1. **fit_nested() Calls:** 44 workflows updated
   - All `.fit(train_data)` calls updated to `.fit_nested(train_data, group_col='country')`
   - Method chaining examples updated
   - All models now fit separately per country

2. **Visualization Cells Added:** 44 cells
   - One visualization cell added after each `extract_outputs()` call
   - Each viz shows top 3 countries by test RMSE
   - Subplots with train/test actual vs fitted values
   - RMSE displayed in subplot titles

3. **Metrics Display:** 44 cells enhanced
   - Per-country test RMSE tables added
   - Best/worst country identification
   - Sorted by performance

4. **Output Inspection:** 45 calls enhanced
   - Shows output DataFrame shape
   - Lists countries in outputs
   - Displays column names including 'country' column

## Visualization Pattern Applied

All 44 visualization cells follow this pattern:

```python
import matplotlib.pyplot as plt

# Get RMSE by country from stats DataFrame
country_rmse = stats_MODEL[
    (stats_MODEL['metric'] == 'rmse') &
    (stats_MODEL['split'] == 'test')
].sort_values('value')

# Plot top 3 countries by RMSE
top_countries = country_rmse.head(3)['country'].values
print(f"Plotting top 3 countries for MODEL: {list(top_countries)}")

# Create subplots for top countries
fig, axes = plt.subplots(len(top_countries), 1, figsize=(12, 4*len(top_countries)))
if len(top_countries) == 1:
    axes = [axes]

for idx, country in enumerate(top_countries):
    ax = axes[idx]
    country_data = outputs_MODEL[outputs_MODEL['country'] == country].copy()

    # Get train/test data
    train_data = country_data[country_data['split'] == 'train']
    test_data = country_data[country_data['split'] == 'test']

    # Plot actual values
    if len(train_data) > 0:
        ax.plot(train_data.index, train_data['actuals'], 'o-', label='Train Actual', alpha=0.6)
        ax.plot(train_data.index, train_data['fitted'], '--', label='Train Fitted', alpha=0.8)

    if len(test_data) > 0:
        ax.plot(test_data.index, test_data['actuals'], 'o-', label='Test Actual', color='green', alpha=0.6)
        ax.plot(test_data.index, test_data['fitted'], '--', label='Test Predicted', color='red', alpha=0.8)

    # Get RMSE for this country
    country_rmse_val = country_rmse[country_rmse['country'] == country]['value'].iloc[0]

    ax.set_title(f'{country} - MODEL (Test RMSE: {country_rmse_val:.2f})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Refinery KBD')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Models with Visualizations (39 total)

All fitted models now have country-specific visualizations:

**Time Series Models:**
- prophet, arima, auto_arima, exp_smoothing, stl
- ets_aaa, ets_aam
- arima_boost, prophet_boost
- prophet_flex, prophet_cons
- arima_ns (non-seasonal ARIMA)
- recursive (recursive forecasting)

**ML Regression Models:**
- linear_reg (sklearn and statsmodels)
- elasticnet, ridge, lasso
- decision_tree, rand_forest
- boost_tree, xgboost, catboost
- svm_rbf, svm_linear
- knn_5, knn_10
- mlp (neural network)
- poisson, mars, gam

**Baseline Models:**
- null_model (mean/median baseline)
- naive_model (time series baseline)

**Advanced Models:**
- hybrid_model (residual/sequential/weighted strategies)
- manual_reg (user-specified coefficients)

**Workflow Examples:**
- formula (formula-based workflow)
- recipe (recipe-based workflow)
- manual (manual workflow construction)

## Metrics Display Pattern

Each model section now includes:

```python
# Show per-country test metrics
print("\n=== Test RMSE by Country (MODEL) ===")
country_metrics = stats_MODEL[
    (stats_MODEL['metric'] == 'rmse') &
    (stats_MODEL['split'] == 'test')
][['country', 'value']].sort_values('value')

display(country_metrics)

if len(country_metrics) > 0:
    print(f"\nBest country: {country_metrics.iloc[0]['country']} (RMSE: {country_metrics.iloc[0]['value']:.4f})")
    print(f"Worst country: {country_metrics.iloc[-1]['country']} (RMSE: {country_metrics.iloc[-1]['value']:.4f}}")
```

## Cells Requiring Manual Review

**None** - All cells have been successfully updated.

Two cells initially flagged for review:
- Cell 130: Commented code (skipped, no action needed)
- Cell 134: Method chaining example (updated to use `fit_nested()`)

## Key Design Decisions

### Pattern A: Multi-Country Comparison (Applied to all viz cells)
- Shows top 3 countries by test RMSE performance
- Separate subplot for each country
- Train and test data clearly distinguished
- RMSE value in title for easy comparison

### Pattern B: Single Country (Not used in this notebook)
Alternative pattern if space-constrained:
```python
country_to_plot = 'Germany'
outputs_single = outputs[outputs['country'] == country_to_plot]
# ... plot single country
```

**Rationale:** Multi-country comparison provides better insights into model performance across different groups.

### Grouped Diagnostics
All `extract_outputs()` calls now include:
- Output shape verification
- Country list verification
- Column inspection (ensuring 'country' column present)

## Testing Recommendations

Before running the notebook:

1. **Verify data has 'country' column:**
   ```python
   assert 'country' in train_data.columns
   assert 'country' in test_data.columns
   ```

2. **Check group consistency:**
   ```python
   train_countries = set(train_data['country'].unique())
   test_countries = set(test_data['country'].unique())
   assert test_countries.issubset(train_countries)
   ```

3. **Memory considerations:**
   - 44 models × multiple countries = many fitted models
   - Consider reducing number of countries for testing
   - Or run notebook in sections

## File Location

**Updated notebook:** `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_grouped.ipynb`

## Next Steps

1. Test notebook execution with actual grouped data
2. Verify all visualizations render correctly
3. Check memory usage with full country set
4. Consider adding:
   - Country-level metric comparison table
   - Overall best model identification
   - Cross-country model performance heatmap

## Notes

- All visualizations use matplotlib (already imported in cell 1)
- All models use FORMULA_NESTED which excludes 'date' via dot notation
- Test/train split maintains group consistency
- NestedWorkflowFit handles prediction routing automatically
