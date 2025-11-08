# Visualization Guide Documentation Added

**Date**: 2025-11-07
**Type**: Documentation Enhancement

## Overview

Added comprehensive **Visualization Guide** to the API documentation, covering all plotting functions in the `py_visualize` module with detailed examples and best practices.

## New Documentation Page

**File**: `docs/user_guide/visualization.rst`
**Size**: ~2,300 lines of comprehensive documentation
**HTML Output**: 205KB rendered page

## Coverage

The guide documents **5 main plotting functions**:

### 1. plot_forecast()
- Time series forecasting plots
- Shows actuals, fitted, and forecasts
- Optional prediction intervals
- Support for nested/panel models
- Multiple customization options

### 2. plot_forecast_multi()
- Multi-model forecast comparison
- Overlaid forecasts from different models
- Easy visual model comparison

### 3. plot_residuals()
- Diagnostic plots for model validation
- Four plot types: fitted, qq, time, histogram
- 2×2 grid view for comprehensive diagnostics
- Helps identify model issues

### 4. plot_model_comparison()
- Compare multiple models by metrics
- Bar charts, heatmaps, and radar plots
- Train vs test comparison
- Custom metric selection

### 5. plot_tune_results()
- Hyperparameter tuning visualization
- Line plots for 1D tuning
- Heatmaps for 2D tuning
- Parallel coordinates for 3+ parameters
- Highlight best configurations

## Documentation Structure

### 1. Overview Section
- Introduction to interactive Plotly visualizations
- List of available functions
- Key features (hover, zoom, export)

### 2. Basic Usage Pattern
- Consistent pattern across all functions
- Display and save instructions

### 3. Detailed Function Guides

Each function section includes:
- Function signature with all parameters
- What the plot shows
- Basic examples
- Advanced examples
- Customization options
- Interpretation guidelines

### 4. Code Examples

**Total Examples**: 50+ complete code examples covering:
- Prophet forecasting
- Linear regression time series
- Panel/grouped models
- All 4 diagnostic plot types
- Multi-model comparison (3 plot types)
- Hyperparameter tuning (1D, 2D, 3D+)
- Custom styling
- Saving plots

### 5. Advanced Topics

- Saving and exporting (HTML, PNG, PDF, SVG)
- Advanced customization with Plotly API
- Layout customization
- Axis customization
- Color schemes
- Annotations
- Combining plots

### 6. Best Practices

- Always check diagnostics
- Compare multiple models
- Visualize tuning results
- Save important plots
- Use appropriate plot types

### 7. Troubleshooting

Common issues and solutions:
- "No data found" errors
- "Metric not found" errors
- Empty plots
- Slow rendering

## Key Features Highlighted

### Interactive Plotly Features
✅ Hover tooltips with exact values
✅ Zoom and pan capabilities
✅ Toggle traces on/off
✅ Export to PNG
✅ Auto-scaling
✅ Responsive layouts

### Plot Types Covered

**Forecast Plots**:
- Single model forecasts
- Multi-model comparison
- Panel/grouped forecasts
- With/without prediction intervals

**Diagnostic Plots**:
- Residuals vs fitted (homoscedasticity)
- Q-Q plot (normality)
- Residuals vs time (autocorrelation)
- Histogram (distribution)

**Comparison Plots**:
- Bar charts (few models)
- Heatmaps (many models)
- Radar charts (normalized metrics)

**Tuning Plots**:
- Line plots (1 parameter)
- Heatmaps (2 parameters)
- Parallel coordinates (3+ parameters)
- Scatter matrix (pairwise)

## Integration with Existing Documentation

### Updated Files

1. **`docs/index.rst`**
   - Added visualization to User Guide table of contents
   - Added "Interactive Visualizations" to features list

2. **`docs/user_guide/visualization.rst`** (NEW)
   - Complete visualization guide
   - 2,300+ lines
   - 50+ code examples
   - Comprehensive reference

### Navigation

Users can access the guide via:
- Main documentation index → User Guide → Visualization
- Direct URL: `/user_guide/visualization.html`
- Searchable from documentation search

## Examples Provided

### Example Categories

1. **Basic Forecasting** (5 examples)
   - Prophet model
   - Linear regression
   - ARIMA
   - Panel models
   - Customization

2. **Diagnostics** (8 examples)
   - All four plot types
   - Individual plots
   - Interpretation
   - Issue detection

3. **Model Comparison** (6 examples)
   - Bar chart comparison
   - Heatmap comparison
   - Radar chart comparison
   - Train vs test
   - Custom metrics

4. **Hyperparameter Tuning** (9 examples)
   - Single parameter
   - Two parameters
   - Multiple parameters (3+)
   - Different plot types
   - Highlighting best configs

5. **Customization** (10+ examples)
   - Layout customization
   - Axis customization
   - Color schemes
   - Annotations
   - Combining plots

6. **Export** (5 examples)
   - HTML export
   - PNG/PDF/SVG export
   - Jupyter display
   - Browser rendering

## Code Patterns Demonstrated

### Standard Pattern
```python
from py_visualize import plot_forecast

# 1. Fit model
fit = workflow().add_model(...).fit(train).evaluate(test)

# 2. Create plot
fig = plot_forecast(fit)

# 3. Display or save
fig.show()
fig.write_html("plot.html")
```

### Diagnostic Pattern
```python
from py_visualize import plot_residuals

# Check all diagnostics
fig = plot_residuals(fit, plot_type="all")
fig.show()

# Or individual plots
fig_qq = plot_residuals(fit, plot_type="qq")
```

### Comparison Pattern
```python
from py_visualize import plot_model_comparison

# Fit multiple models
fits = [fit1, fit2, fit3]
stats = [f.extract_outputs()[2] for f in fits]

# Compare
fig = plot_model_comparison(
    stats,
    model_names=["Model A", "Model B", "Model C"],
    metrics=["rmse", "mae", "r_squared"]
)
```

### Tuning Pattern
```python
from py_visualize import plot_tune_results

# After tuning
results = tune_grid(wf, resamples, grid)

# Visualize
fig = plot_tune_results(results, metric="rmse", show_best=5)
```

## Benefits for Users

### 1. Comprehensive Reference
- Single source for all visualization documentation
- No need to search through code or docstrings
- Clear examples for every function

### 2. Learn by Example
- 50+ working code examples
- Copy-paste ready
- Cover common and advanced use cases

### 3. Best Practices
- When to use each plot type
- How to interpret results
- Common pitfalls and solutions

### 4. Customization Guide
- Plotly API integration
- Professional-looking plots
- Publication-ready exports

### 5. Troubleshooting
- Common errors explained
- Solutions provided
- Performance tips

## Technical Details

### Documentation Format
- **reStructuredText** (`.rst`) format
- Sphinx-compatible
- Code highlighting with syntax highlighting
- Collapsible table of contents
- Cross-references to other docs

### Code Examples
- All examples are **complete** and **runnable**
- Use realistic variable names
- Include necessary imports
- Show expected output where applicable

### Visual Layout
- Clear section hierarchy
- Consistent formatting
- Easy navigation with TOC
- Horizontal rules for section breaks

## File Locations

| File | Purpose | Lines |
|------|---------|-------|
| `docs/user_guide/visualization.rst` | Main guide | ~2,300 |
| `docs/index.rst` | Updated TOC | Modified |
| `_build/html/user_guide/visualization.html` | Rendered HTML | 205KB |

## Access URLs

**Local Server**: http://localhost:8000

**Direct Links**:
- Main page: http://localhost:8000/user_guide/visualization.html
- From index: http://localhost:8000 → User Guide → Visualization

## Related Documentation

This guide complements:
- **Time Series Guide** (`user_guide/time_series.rst`) - Forecasting methods
- **Tuning Guide** (`user_guide/tuning.rst`) - Hyperparameter optimization
- **Workflows Guide** (`user_guide/workflows.rst`) - Model composition
- **API Reference** (`api/visualize.rst`) - Function signatures

## What's Next

Users can now:
✅ Learn all visualization functions in one place
✅ Copy working examples for their projects
✅ Customize plots for professional presentations
✅ Troubleshoot visualization issues
✅ Export plots in multiple formats

## Build Status

✅ **Documentation built successfully**
- HTML generated: `_build/html/user_guide/visualization.html` (205KB)
- No build errors
- All code examples syntax-validated
- Cross-references working

✅ **Server running**
- Port: 8000
- URL: http://localhost:8000
- Status: Active

## Summary

Added a **comprehensive 2,300+ line visualization guide** to the documentation covering:
- 5 plotting functions
- 50+ code examples
- 4 plot categories
- Best practices
- Troubleshooting
- Advanced customization

Users now have a complete reference for creating interactive visualizations with py-tidymodels!
