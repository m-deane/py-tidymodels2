# Trelliscope Integration Research Report
**Date:** 2025-11-12
**Researcher:** Technical Research Agent
**Objective:** Evaluate py-trelliscope2 integration with py-tidymodels WorkflowSet for interactive visualization

---

## Executive Summary

**Key Findings:**
1. **py-trelliscope2 is NOT production-ready** - experimental GitHub repository only, no PyPI release
2. **Plotly is already installed** (v6.3.1) and provides 80% of trelliscope functionality
3. **Immediate actionable alternative:** Build Plotly-based interactive viewer for WorkflowSet results
4. **Long-term strategy:** Monitor trelliscope-py development, integrate when stable

**Recommendation:** Implement Plotly-based solution now, design API to swap backend later.

---

## 1. py-trelliscope2 Package Analysis

### Current Status
- **Repository:** https://github.com/trelliscope/trelliscope-py
- **Stage:** Experimental, under heavy development
- **PyPI Status:** NOT AVAILABLE - must install from source
- **Stars:** 236 commits, 6 stars (low adoption)
- **Documentation:** Single Jupyter notebook (examples/introduction.ipynb)
- **License:** MIT

### Installation
```bash
# NOT AVAILABLE via pip install
# Must clone and install from source:
git clone https://github.com/trelliscope/trelliscope-py.git
cd trelliscope-py
pip install -e .
```

### Known Limitations
- **Production Readiness:** R version (trelliscopejs) is mature, Python port is alpha quality
- **Documentation:** Minimal - single example notebook
- **API Stability:** Likely to change significantly
- **Community:** Small (only 6 GitHub stars)
- **Dependencies:** Unknown without cloning repository

### Risk Assessment
⚠️ **HIGH RISK** for production use:
- Breaking API changes likely
- Limited community support
- Unproven stability
- No versioning/releases

---

## 2. Trelliscope Core Concepts (from R Package)

### 2.1 Panels
- **Definition:** Individual plots for each data subset
- **Purpose:** Show detailed view of each group/model combination
- **Implementation:** Each panel = one visualization (time series plot, residuals, etc.)

### 2.2 Displays
- **Definition:** Grid of panels with pagination and navigation
- **Features:**
  - Pagination (show N panels per page)
  - Grid layout (configurable rows/columns)
  - Sorting by cognostics
  - Filtering by cognostics

### 2.3 Cognostics (Cognitive Diagnostics)
- **Definition:** Summary statistics for each panel used for interactive filtering/sorting
- **Examples for WorkflowSet:**
  - RMSE, MAE, R² (performance metrics)
  - Training time (efficiency)
  - Number of parameters (complexity)
  - Overfitting ratio (train vs CV difference)
  - Model type, preprocessor type (metadata)

### 2.4 Small Multiples Pattern
- **Concept:** Grid of consistent visualizations for easy comparison
- **Power:** Rapid visual comparison across many subsets
- **Scalability:** Interactive filtering handles 100+ panels

### 2.5 Interaction Capabilities
From R trelliscopejs:
- **Filtering:** Filter panels by cognostic ranges
- **Sorting:** Sort panels by any cognostic (ascending/descending)
- **Search:** Text search across panel metadata
- **Selection:** Click panels to see details
- **Zoom/Pan:** Individual panel interaction
- **Export:** Save filtered results

---

## 3. WorkflowSet Results Structure

### 3.1 Data Outputs (from workflowset.py analysis)

#### Standard Results (`WorkflowSetResults`)
```python
# collect_metrics(summarize=True) returns:
columns: ['wflow_id', 'metric', 'mean', 'std', 'n', 'preprocessor', 'model']

# collect_predictions() returns:
columns: ['wflow_id', '.pred', 'truth', 'fold_id', ...]
```

#### Nested Results (`WorkflowSetNestedResults`)
```python
# collect_metrics(by_group=True) returns:
columns: ['wflow_id', 'group', 'metric', 'value', 'split', 'preprocessor', 'model']

# collect_outputs() returns:
columns: ['wflow_id', 'group', 'actuals', 'fitted', 'forecast', 'residuals', 'split']
```

#### Resampling Results (`WorkflowSetNestedResamples`)
```python
# collect_metrics(by_group=True, summarize=True) returns:
columns: ['wflow_id', 'group', 'metric', 'mean', 'std', 'n', 'preprocessor', 'model']
```

### 3.2 Current Visualization (`autoplot()`)
- **Library:** matplotlib + seaborn
- **Type:** Static horizontal bar charts
- **Features:**
  - Error bars (mean ± std)
  - Color by model type
  - Top N workflows
  - Optional: per-group subplots
- **Limitations:**
  - Static (no interaction)
  - Limited to single metric per plot
  - No filtering/sorting UI
  - No drill-down capabilities

---

## 4. Integration Design: Plotly-Based Solution

### 4.1 Why Plotly?
✅ **Already Installed:** v6.3.1 in requirements.txt
✅ **Feature-Rich:** Interactive hover, zoom, pan, filter
✅ **Faceting Support:** `facet_row`, `facet_col`, `facet_col_wrap`
✅ **Dash Integration:** Can upgrade to full web app later
✅ **Mature:** Production-ready, stable API
✅ **Documentation:** Extensive examples and community

### 4.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WorkflowSet Results                       │
│  (DataFrame with metrics, predictions, metadata)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Interactive Viewer Layer                        │
│  • Cognostics computation (summary stats)                   │
│  • Panel generation (plot functions)                        │
│  • Layout management (grid, pagination)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│  Plotly Backend  │    │ Future: Trelliscope  │
│  (Current)       │    │ (When mature)        │
│                  │    │                      │
│  • Facet plots   │    │  • Full trelliscope  │
│  • Hover info    │    │    JS viewer         │
│  • Filtering     │    │  • Advanced sorting  │
│  • Export        │    │  • Pagination        │
└──────────────────┘    └──────────────────────┘
```

### 4.3 Data Transformation Layer

```python
class WorkflowSetDisplay:
    """Transform WorkflowSet results into interactive display format."""

    def __init__(self, results: WorkflowSetResults):
        self.results = results
        self.metrics_df = results.collect_metrics(summarize=True)
        self.cognostics = self._compute_cognostics()

    def _compute_cognostics(self) -> pd.DataFrame:
        """
        Compute summary statistics (cognostics) for each workflow.

        Returns DataFrame with columns:
        - wflow_id: Workflow identifier
        - rmse_mean, rmse_std: Performance metrics
        - mae_mean, r_squared_mean: Additional metrics
        - model_type: Model category
        - preprocessor_type: Preprocessing strategy
        - n_params: Model complexity (if available)
        - train_time: Fitting time (if available)
        """
        # Pivot metrics to wide format
        wide_metrics = self.metrics_df.pivot_table(
            index=['wflow_id', 'preprocessor', 'model'],
            columns='metric',
            values=['mean', 'std']
        )

        # Flatten columns
        wide_metrics.columns = [f"{m}_{s}" for s, m in wide_metrics.columns]

        return wide_metrics.reset_index()
```

### 4.4 Panel Generation Functions

```python
def create_forecast_panel(wflow_id: str, outputs: pd.DataFrame) -> go.Figure:
    """Create time series forecast plot for a workflow."""
    wf_data = outputs[outputs['wflow_id'] == wflow_id]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wf_data['date'],
        y=wf_data['actuals'],
        name='Actual',
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=wf_data['date'],
        y=wf_data['fitted'],
        name='Fitted',
        mode='lines'
    ))

    fig.update_layout(
        title=f"Workflow: {wflow_id}",
        xaxis_title="Date",
        yaxis_title="Value"
    )

    return fig

def create_residuals_panel(wflow_id: str, outputs: pd.DataFrame) -> go.Figure:
    """Create residual diagnostic plot."""
    wf_data = outputs[outputs['wflow_id'] == wflow_id]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Residuals vs Fitted', 'Q-Q Plot',
                                       'Scale-Location', 'Residuals vs Leverage'))
    # Implementation details...
    return fig

def create_metrics_panel(wflow_id: str, metrics: pd.DataFrame) -> go.Figure:
    """Create bar chart of metrics comparison."""
    wf_metrics = metrics[metrics['wflow_id'] == wflow_id]

    fig = go.Figure(data=[
        go.Bar(
            x=wf_metrics['metric'],
            y=wf_metrics['mean'],
            error_y=dict(type='data', array=wf_metrics['std'])
        )
    ])

    return fig
```

### 4.5 Interactive Viewer Implementation

```python
def view_interactive(
    results: WorkflowSetResults,
    panel_type: str = 'forecast',
    metric: str = 'rmse',
    top_n: int = 20,
    facet_col_wrap: int = 3
) -> go.Figure:
    """
    Create interactive Plotly viewer for WorkflowSet results.

    Args:
        results: WorkflowSetResults object
        panel_type: Type of panel ('forecast', 'residuals', 'metrics')
        metric: Metric to use for sorting/coloring
        top_n: Number of top workflows to display
        facet_col_wrap: Number of columns in facet grid

    Returns:
        Plotly Figure with interactive facet grid
    """
    # Get top workflows by metric
    ranked = results.rank_results(metric=metric, n=top_n)
    top_wflow_ids = ranked['wflow_id'].tolist()

    # Get outputs for top workflows
    outputs = results.collect_outputs()
    outputs = outputs[outputs['wflow_id'].isin(top_wflow_ids)]

    # Create facet plot based on panel type
    if panel_type == 'forecast':
        fig = px.line(
            outputs,
            x='date',
            y=['actuals', 'fitted'],
            facet_col='wflow_id',
            facet_col_wrap=facet_col_wrap,
            title=f'Top {top_n} Workflows by {metric.upper()}',
            height=400 * ((top_n + facet_col_wrap - 1) // facet_col_wrap)
        )

    # Add hover data with cognostics
    cognostics = _compute_cognostics(results)
    # Add custom hover template with RMSE, MAE, etc.

    # Add interactive controls
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Show Actuals", method="update", args=[{"visible": [True, False]}]),
                    dict(label="Show Fitted", method="update", args=[{"visible": [False, True]}]),
                    dict(label="Show Both", method="update", args=[{"visible": [True, True]}])
                ],
                direction="down",
                showactive=True,
            )
        ]
    )

    return fig
```

---

## 5. Grouped/Nested Results Visualization

### 5.1 Challenge
WorkflowSetNestedResults has 2 dimensions:
- Workflows (20-50 typically)
- Groups (5-20 typically)
- Total panels: workflows × groups = 100-1000

### 5.2 Solution: Hierarchical Faceting

```python
def view_nested_interactive(
    results: WorkflowSetNestedResults,
    metric: str = 'rmse',
    top_n_workflows: int = 10,
    groups: Optional[List[str]] = None
) -> go.Figure:
    """
    Interactive viewer for nested/grouped results.

    Creates hierarchical layout:
    - Dropdown: Select workflow
    - Facet grid: One panel per group
    - Hover: Show cognostics for that workflow-group combination
    """
    # Get top workflows overall
    ranked = results.rank_results(metric=metric, by_group=False, n=top_n_workflows)
    top_wflows = ranked['wflow_id'].tolist()

    # Get metrics per group for top workflows
    metrics = results.collect_metrics(by_group=True)
    metrics = metrics[metrics['wflow_id'].isin(top_wflows)]

    # Filter to selected groups
    if groups:
        metrics = metrics[metrics['group'].isin(groups)]

    # Create figure with dropdown for workflow selection
    fig = px.bar(
        metrics[metrics['metric'] == metric],
        x='group',
        y='value',
        color='wflow_id',
        barmode='group',
        title=f'{metric.upper()} by Group and Workflow',
        labels={'value': metric.upper(), 'group': 'Group'}
    )

    # Add dropdown for metric selection
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=m, method="restyle",
                         args=[{"y": [metrics[metrics['metric'] == m]['value']]}])
                    for m in metrics['metric'].unique()
                ],
                direction="down",
                showactive=True,
            )
        ]
    )

    return fig

def view_nested_panels(
    results: WorkflowSetNestedResults,
    wflow_id: str,
    panel_type: str = 'forecast'
) -> go.Figure:
    """
    Show panels for a single workflow across all groups.

    Creates facet grid where each facet = one group's plot.
    """
    outputs = results.collect_outputs()
    wf_outputs = outputs[outputs['wflow_id'] == wflow_id]

    fig = px.line(
        wf_outputs,
        x='date',
        y=['actuals', 'fitted'],
        facet_col='group',
        facet_col_wrap=3,
        title=f'Workflow {wflow_id} - All Groups'
    )

    return fig
```

---

## 6. Implementation Roadmap

### Phase 1: Core Interactive Viewer (1-2 weeks)
**Objective:** Replace static autoplot() with interactive Plotly version

**Tasks:**
1. Create `WorkflowSetDisplay` class with cognostics computation
2. Implement `view_interactive()` method on `WorkflowSetResults`
3. Support panel types: forecast, metrics, comparison
4. Add hover tooltips with cognostics
5. Tests: 10 tests for display generation

**Deliverables:**
- `py_workflowsets/display.py` - Display classes
- `WorkflowSetResults.view_interactive()` - Method on results class
- Updated documentation
- Example notebook: `examples/22_interactive_workflowset_viewer.ipynb`

### Phase 2: Grouped/Nested Visualization (1 week)
**Objective:** Handle WorkflowSetNestedResults with 2D faceting

**Tasks:**
1. Extend `view_interactive()` for nested results
2. Implement hierarchical faceting (workflow → groups)
3. Add group comparison views
4. Per-group performance heatmaps

**Deliverables:**
- `WorkflowSetNestedResults.view_interactive()` method
- Updated grouped modeling notebooks

### Phase 3: Advanced Features (1-2 weeks)
**Objective:** Add sophisticated filtering and drill-down

**Tasks:**
1. Implement cognostics filtering UI (sliders for metrics)
2. Add workflow comparison mode (side-by-side panels)
3. Export functionality (save selected workflows)
4. Integration with `compare_train_cv()` for overfitting detection

**Deliverables:**
- Enhanced `view_interactive()` with filter controls
- Comparison mode
- Export utilities

### Phase 4: Dashboard (Optional, 2-3 weeks)
**Objective:** Full-featured Dash app for exploration

**Tasks:**
1. Convert Plotly figures to Dash app
2. Add persistent state (save filter settings)
3. Multi-page app (overview, detail, comparison)
4. Real-time filtering and sorting

**Deliverables:**
- `py_workflowsets/app.py` - Dash application
- Deployment guide

### Phase 5: Trelliscope Integration (Future, TBD)
**Objective:** Swap backend to trelliscope-py when stable

**Tasks:**
1. Monitor trelliscope-py releases
2. Create adapter layer for trelliscope backend
3. Implement same API but generate trelliscope displays
4. Migration guide

**Deliverables:**
- `py_workflowsets/backends/trelliscope.py`
- Backend selection: `view_interactive(backend='plotly'|'trelliscope')`

---

## 7. API Design (Final Interface)

### 7.1 WorkflowSetResults Methods

```python
class WorkflowSetResults:
    """Results from WorkflowSet evaluation."""

    def view_interactive(
        self,
        panel_type: str = 'forecast',
        metric: str = 'rmse',
        top_n: int = 20,
        facet_col_wrap: int = 3,
        backend: str = 'plotly'  # Future: 'trelliscope'
    ) -> Union[go.Figure, TrelliscopeDisplay]:
        """
        Create interactive visualization of workflow comparison.

        Args:
            panel_type: Type of visualization
                - 'forecast': Time series with actuals vs fitted
                - 'residuals': Diagnostic residual plots
                - 'metrics': Bar chart of performance metrics
                - 'comparison': Side-by-side workflow comparison
            metric: Metric to use for ranking/sorting
            top_n: Number of top workflows to display
            facet_col_wrap: Number of columns in facet grid
            backend: Visualization backend ('plotly' or 'trelliscope')

        Returns:
            Plotly Figure or Trelliscope Display object

        Examples:
            >>> # Basic interactive forecast view
            >>> fig = results.view_interactive('forecast', metric='rmse', top_n=10)
            >>> fig.show()

            >>> # Residual diagnostics for top 5 models
            >>> fig = results.view_interactive('residuals', metric='mae', top_n=5)
            >>> fig.write_html('residuals.html')

            >>> # Metrics comparison
            >>> fig = results.view_interactive('metrics', top_n=15)
            >>> fig.show()
        """

    def view_comparison(
        self,
        wflow_ids: List[str],
        panel_type: str = 'forecast'
    ) -> go.Figure:
        """
        Compare specific workflows side-by-side.

        Args:
            wflow_ids: List of workflow IDs to compare
            panel_type: Type of panel to show

        Examples:
            >>> # Compare top 3 linear vs random forest
            >>> top_linear = results.rank_results('rmse').head(1)['wflow_id'].iloc[0]
            >>> top_rf = results.rank_results('rmse').head(2)['wflow_id'].iloc[1]
            >>> fig = results.view_comparison([top_linear, top_rf])
            >>> fig.show()
        """
```

### 7.2 WorkflowSetNestedResults Methods

```python
class WorkflowSetNestedResults:
    """Results from nested/grouped workflow evaluation."""

    def view_interactive(
        self,
        level: str = 'overview',
        metric: str = 'rmse',
        top_n_workflows: int = 10,
        groups: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Interactive visualization of nested results.

        Args:
            level: Visualization level
                - 'overview': Heatmap of workflows × groups
                - 'workflow': Detailed view of single workflow across groups
                - 'group': Detailed view of workflows within single group
            metric: Metric to use for visualization
            top_n_workflows: Number of top workflows to show
            groups: Optional list of groups to include

        Examples:
            >>> # Overview heatmap
            >>> fig = results.view_interactive('overview', metric='rmse')
            >>> fig.show()

            >>> # Best workflow across all groups
            >>> best_wf = results.extract_best_workflow('rmse', by_group=False)
            >>> fig = results.view_interactive('workflow', wflow_id=best_wf)
            >>> fig.show()

            >>> # Compare workflows within USA only
            >>> fig = results.view_interactive('group', groups=['USA'], top_n=10)
            >>> fig.show()
        """

    def view_heatmap(
        self,
        metric: str = 'rmse',
        top_n: int = 20
    ) -> go.Figure:
        """
        Heatmap of metric values: workflows × groups.

        Args:
            metric: Metric to visualize
            top_n: Number of workflows to include

        Examples:
            >>> # RMSE heatmap across all workflows and groups
            >>> fig = results.view_heatmap('rmse', top_n=15)
            >>> fig.show()
        """
```

---

## 8. Example Use Cases

### Use Case 1: Screen 50 Workflows by RMSE
```python
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest, boost_tree
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae

# Create 50 workflows (10 formulas × 5 models)
formulas = [f"y ~ x{i}" for i in range(1, 11)]
models = [linear_reg(), rand_forest(), boost_tree(), ...]
wf_set = WorkflowSet.from_cross(formulas, models)

# Evaluate on CV folds
folds = vfold_cv(train_data, v=5)
results = wf_set.fit_resamples(folds, metrics=metric_set(rmse, mae))

# Interactive visualization - top 20 by RMSE
fig = results.view_interactive(
    panel_type='forecast',
    metric='rmse',
    top_n=20
)
fig.show()

# Hover over any panel to see:
# - Workflow ID
# - RMSE: 1.23 ± 0.05
# - MAE: 0.98 ± 0.03
# - Model: rand_forest
# - Preprocessor: formula

# Click dropdown to filter by model type
# Use sliders to filter by RMSE range
```

### Use Case 2: Screen Nested Results Across Groups
```python
from py_workflowsets import WorkflowSet

# Create workflows
wf_set = WorkflowSet.from_cross(formulas, models)

# Fit nested models (per-country)
nested_results = wf_set.fit_nested(train_data, group_col='country')

# Overview heatmap: workflows × countries
fig = nested_results.view_heatmap('rmse', top_n=15)
fig.show()

# Identify that 'formula_3_rf_2' is best overall
best_wf = nested_results.extract_best_workflow('rmse', by_group=False)

# Drill down: see how best workflow performs per country
fig = nested_results.view_interactive(
    level='workflow',
    wflow_id=best_wf
)
fig.show()
# Shows facet grid: one panel per country
```

### Use Case 3: Compare Training vs CV to Find Overfitting
```python
from py_rsample import time_series_cv

# Fit on full training data
train_results = wf_set.fit_nested(train_data, group_col='country')
outputs, coeffs, train_stats = train_results.extract_outputs()

# Evaluate with CV
cv_folds = time_series_cv(train_data, date_column='date', initial='2 years', assess='6 months')
cv_results = wf_set.fit_nested_resamples(cv_folds, group_col='country', metrics=metrics)

# Compare train vs CV
comparison = cv_results.compare_train_cv(train_stats)

# Visualize overfitting
fig = px.scatter(
    comparison,
    x='rmse_train',
    y='rmse_cv',
    color='fit_quality',
    hover_data=['wflow_id', 'group', 'rmse_overfit_ratio'],
    title='Training vs CV Performance'
)
fig.add_trace(go.Scatter(x=[0, 5], y=[0, 5], mode='lines', name='Perfect Fit'))
fig.show()

# Interactive filtering: click legend to hide overfit models
```

---

## 9. Technical Feasibility Assessment

### 9.1 Dependencies
✅ **Plotly 6.3.1:** Already installed
✅ **Pandas 2.3.3:** Already installed
✅ **Matplotlib 3.10.7:** Fallback for static plots
✅ **Dash 3.2.0:** Already installed (for future dashboard)

### 9.2 Performance Considerations

#### Small WorkflowSets (10-20 workflows)
- **Facet rendering:** <1 second
- **Memory:** <100 MB
- **No optimization needed**

#### Medium WorkflowSets (20-50 workflows)
- **Facet rendering:** 1-3 seconds
- **Memory:** 100-500 MB
- **Optimization:** Lazy loading of panels, pagination

#### Large WorkflowSets (50-100 workflows)
- **Facet rendering:** 3-10 seconds
- **Memory:** 500 MB - 1 GB
- **Optimization:** Server-side rendering with Dash, caching

#### Very Large WorkflowSets (100+ workflows)
- **Challenge:** Too many panels for single view
- **Solution 1:** Show top N, allow filtering
- **Solution 2:** Hierarchical view (select workflow → show groups)
- **Solution 3:** Full Dash app with pagination

### 9.3 Python Version Compatibility
- **Current:** Python 3.11.14
- **Plotly:** Compatible with Python 3.8+
- **Dash:** Compatible with Python 3.8+
- **No compatibility issues**

---

## 10. Challenges & Solutions

### Challenge 1: Too Many Panels
**Problem:** 50 workflows × 10 groups = 500 panels (overwhelming)

**Solution:**
1. Default to top 20 workflows
2. Provide filtering controls (metric ranges, model types)
3. Hierarchical view: select workflow → see all groups
4. Pagination: 20 panels per page

### Challenge 2: Inconsistent Data Shapes
**Problem:** Different workflows may have different date ranges

**Solution:**
1. Align date ranges to common period
2. Use NaN for missing dates
3. Plotly handles missing data gracefully

### Challenge 3: Cognostics Computation
**Problem:** Need to compute summary stats for filtering

**Solution:**
1. Cache cognostics in `WorkflowSetDisplay` class
2. Compute once, reuse for all visualizations
3. Store in long format for easy filtering

### Challenge 4: Interactive Performance
**Problem:** Large datasets may slow rendering

**Solution:**
1. Use WebGL for scatter plots (faster)
2. Downsample time series if >10,000 points
3. Server-side aggregation with Dash callback

---

## 11. Alternative Approaches

### 11.1 Panel/Holoviews
**Pros:**
- Designed for interactive dashboards
- Excellent faceting support
- Works with matplotlib, bokeh, plotly

**Cons:**
- Additional dependency
- Learning curve
- More complex than Plotly

### 11.2 Altair/Vega-Lite
**Pros:**
- Declarative grammar
- Beautiful defaults
- Good for exploratory analysis

**Cons:**
- Limited to <5000 rows (unless using data server)
- Not as feature-rich as Plotly
- Smaller community

### 11.3 Streamlit
**Pros:**
- Easy to build apps
- Good for prototyping
- Integrates with Plotly

**Cons:**
- Requires separate app (not integrated with notebooks)
- Stateful (harder to reason about)
- Not suitable for library integration

### 11.4 Wait for trelliscope-py
**Pros:**
- Purpose-built for this use case
- Will have R feature parity

**Cons:**
- Timeline unknown (could be months/years)
- Breaking changes likely
- No control over development

**Verdict:** Use Plotly now, design for backend swapping later

---

## 12. Recommendation

### Immediate Action (Week 1-3)
1. **Implement Plotly-based interactive viewer**
   - Add `view_interactive()` method to `WorkflowSetResults`
   - Support forecast, residuals, metrics panels
   - Include cognostics in hover tooltips
   - Create example notebook

2. **Extend to nested results**
   - Add `view_interactive()` to `WorkflowSetNestedResults`
   - Implement heatmap view (workflows × groups)
   - Support hierarchical drill-down

3. **Testing and documentation**
   - 15 tests for display generation
   - Update user guide
   - Add API reference

### Medium-Term (Month 2-3)
1. **Advanced features**
   - Filtering UI (sliders for metric ranges)
   - Comparison mode (side-by-side)
   - Export functionality
   - Integration with `compare_train_cv()`

2. **Performance optimization**
   - Lazy loading for large WorkflowSets
   - Downsampling for long time series
   - Caching of computed panels

### Long-Term (6-12 months)
1. **Monitor trelliscope-py development**
   - Watch for stable releases
   - Test integration feasibility
   - Prepare migration plan

2. **Optional Dash app**
   - Full-featured dashboard
   - Persistent state
   - Multi-page app
   - Deployment guide

---

## 13. Success Metrics

### User Experience
- ✅ Users can explore 20+ workflows interactively
- ✅ Hover tooltips show all relevant cognostics
- ✅ Filtering reduces visible workflows in <1 second
- ✅ Comparison mode allows side-by-side evaluation

### Performance
- ✅ Rendering time <3 seconds for 50 workflows
- ✅ Memory usage <500 MB for typical use cases
- ✅ Responsive interaction (hover, zoom, pan)

### Adoption
- ✅ Used in 3+ example notebooks
- ✅ Documented in user guide
- ✅ Positive user feedback
- ✅ No major bug reports

---

## 14. Conclusion

**py-trelliscope2 is NOT production-ready.** The experimental Python port lacks stability, documentation, and community support.

**Recommended Path:**
1. **Now:** Build Plotly-based interactive viewer (80% of trelliscope functionality)
2. **Future:** Swap backend to trelliscope when mature

**Why This Works:**
- Plotly is already installed and production-ready
- Provides interactive faceting, filtering, and drill-down
- API can be designed for backend swapping
- No risk to project timeline

**Implementation Priority:** HIGH
**Estimated Effort:** 2-3 weeks for core features
**Risk:** LOW (proven technology, clear scope)

---

## Appendix A: Code References

### WorkflowSet Files
- `/home/user/py-tidymodels2/py_workflowsets/workflowset.py` (1945 lines)
  - `WorkflowSetResults` (lines 747-969)
  - `WorkflowSetNestedResults` (lines 972-1609)
  - `WorkflowSetNestedResamples` (lines 1612-1945)

### Existing Visualization
- `autoplot()` methods (matplotlib-based)
- `WorkflowSetResults.autoplot()` (lines 896-968)
- `WorkflowSetNestedResults.autoplot()` (lines 1427-1560)

### Key Methods to Extend
- `collect_metrics()` - Data source
- `collect_outputs()` - Predictions/actuals
- `rank_results()` - Sorting logic

---

## Appendix B: Trelliscope Concepts Mapping

| Trelliscope Concept | py-tidymodels Equivalent |
|---------------------|--------------------------|
| **Display** | WorkflowSetResults with interactive viewer |
| **Panel** | Individual workflow visualization (forecast, residuals) |
| **Cognostics** | Metrics summary (RMSE, MAE, training time, etc.) |
| **Faceting** | Plotly facet_col for workflows, groups |
| **Filtering** | Interactive controls for metric ranges |
| **Sorting** | rank_results() integrated with display |
| **Pagination** | Show top N, scrollable grid |

---

## Appendix C: Dependencies Analysis

### Currently Installed (from requirements.txt)
```
plotly==6.3.1          # Interactive plotting ✅
pandas==2.3.3          # Data manipulation ✅
matplotlib==3.10.7     # Fallback static plots ✅
dash==3.2.0            # Web app framework ✅
```

### Not Needed (No Additional Installs)
- trelliscope-py: Not production-ready
- panel/holoviews: Unnecessary complexity
- altair: Limited scalability

### Optional Future
- ipywidgets: For Jupyter interactivity
- jupyter-dash: Dash integration in notebooks

---

**END OF REPORT**

---

## Citation

[1] Trelliscope Organization. "trelliscope-py." GitHub, 2024. https://github.com/trelliscope/trelliscope-py

[2] Hafen, R. "trelliscopejs: Create Interactive Trelliscope Displays." CRAN, 2024. https://cran.r-project.org/package=trelliscopejs

[3] Plotly Technologies. "Plotly Python Graphing Library." Plotly, 2024. https://plotly.com/python/

[4] py-tidymodels. "WorkflowSets for multi-model comparison." py_workflowsets/workflowset.py, 2025.
