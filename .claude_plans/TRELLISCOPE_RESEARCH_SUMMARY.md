# Trelliscope Integration Research - Executive Summary
**Date:** 2025-11-12
**Status:** RESEARCH COMPLETE
**Recommendation:** BUILD PLOTLY-BASED SOLUTION NOW

---

## Key Finding

**py-trelliscope2 is NOT production-ready.**

The Python port of Trelliscope is experimental, has minimal documentation, and is not available on PyPI. Using it would introduce significant risk to the project.

---

## Recommended Solution

### Immediate: Plotly-Based Interactive Viewer

**Why Plotly:**
- ✅ Already installed (v6.3.1)
- ✅ Production-ready and stable
- ✅ 80% of trelliscope functionality
- ✅ Rich interactive features
- ✅ Easy HTML export

**What It Provides:**
1. **Interactive facet grids** - Multiple workflow panels in responsive grid
2. **Cognostics tooltips** - Hover to see RMSE, MAE, R², etc.
3. **Zoom and pan** - Explore individual panels in detail
4. **Filtering** - Interactive controls for metric ranges
5. **Comparison mode** - Side-by-side workflow evaluation
6. **Export** - Self-contained HTML files for sharing

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER CALLS                                      │
│                                                                       │
│  results = wf_set.fit_resamples(cv_folds, metrics)                  │
│  fig = results.view_interactive('forecast', metric='rmse', top_n=20)│
│  fig.show()                                                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WorkflowSetResults                                │
│                                                                       │
│  Data:                                                               │
│  • results: List[Dict] - TuneResults per workflow                   │
│  • workflow_set: WorkflowSet - Original workflows                   │
│  • metrics: MetricSet - Evaluation metrics                          │
│                                                                       │
│  Methods:                                                            │
│  • collect_metrics() → DataFrame (wflow_id, metric, mean, std)     │
│  • collect_outputs() → DataFrame (wflow_id, actuals, fitted)       │
│  • rank_results() → DataFrame (ranked workflows)                    │
│  • view_interactive() → Plotly Figure ⭐ NEW                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WorkflowSetDisplay                                │
│                    (NEW CLASS)                                       │
│                                                                       │
│  Responsibilities:                                                   │
│  1. Compute cognostics (summary stats)                              │
│  2. Generate panels (forecast, residuals, metrics)                  │
│  3. Create facet layouts                                            │
│  4. Add interactive controls                                        │
│                                                                       │
│  Key Methods:                                                        │
│  • _compute_cognostics() → DataFrame                                │
│  • _create_forecast_panel(wflow_id) → go.Figure                     │
│  • _create_residuals_panel(wflow_id) → go.Figure                    │
│  • create_facet_display() → go.Figure                               │
│  • add_interactive_controls(fig) → go.Figure                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Cognostics Computation                            │
│                                                                       │
│  Input: collect_metrics() DataFrame                                  │
│         [wflow_id, metric, mean, std, n]                            │
│                                                                       │
│  Transform: Pivot to wide format + add derived metrics              │
│                                                                       │
│  Output: Cognostics DataFrame                                        │
│         [wflow_id, rmse_mean, rmse_std, mae_mean, r_squared_mean,  │
│          model, preprocessor, rank_rmse, complexity, fit_category]  │
│                                                                       │
│  Used For:                                                           │
│  • Sorting panels                                                    │
│  • Hover tooltips                                                    │
│  • Interactive filtering (Phase 3)                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Panel Generation                                  │
│                                                                       │
│  For Each Top N Workflows:                                          │
│                                                                       │
│  1. Forecast Panel:                                                 │
│     • Line plot: actuals (black) vs fitted (blue/red by split)     │
│     • Title with cognostics: "RMSE: 1.23 | MAE: 0.98 | R²: 0.85"   │
│     • Hover tooltips with date and value                            │
│                                                                       │
│  2. Residuals Panel (4 subplots):                                   │
│     • Residuals vs Fitted                                           │
│     • Q-Q Plot (normality check)                                    │
│     • Scale-Location (homoscedasticity)                             │
│     • Residuals Distribution (histogram)                            │
│                                                                       │
│  3. Metrics Panel:                                                   │
│     • Bar chart: RMSE, MAE, R², MAPE                                │
│     • Error bars (std dev)                                          │
│     • Color-coded by metric type                                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Facet Grid Assembly                               │
│                                                                       │
│  Layout:                                                             │
│  • Grid size: top_n workflows                                       │
│  • Columns: facet_col_wrap (default: 3)                            │
│  • Rows: ⌈top_n / facet_col_wrap⌉                                  │
│  • Height: 300px per row                                            │
│                                                                       │
│  Sorting:                                                            │
│  • By metric (RMSE, MAE, R²)                                        │
│  • Ascending for error metrics                                      │
│  • Descending for goodness-of-fit                                   │
│                                                                       │
│  Example: top_n=12, facet_col_wrap=3                               │
│  ┌─────────┬─────────┬─────────┐                                    │
│  │ Panel 1 │ Panel 2 │ Panel 3 │  ← Best workflows                  │
│  ├─────────┼─────────┼─────────┤                                    │
│  │ Panel 4 │ Panel 5 │ Panel 6 │                                    │
│  ├─────────┼─────────┼─────────┤                                    │
│  │ Panel 7 │ Panel 8 │ Panel 9 │                                    │
│  ├─────────┼─────────┼─────────┤                                    │
│  │Panel 10 │Panel 11 │Panel 12 │                                    │
│  └─────────┴─────────┴─────────┘                                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Interactive Controls                              │
│                                                                       │
│  Dropdown 1: Metric Selection                                       │
│  • RMSE                                                              │
│  • MAE                                                               │
│  • R²                                                                │
│  • MAPE                                                              │
│  → Updates sorting and highlighting                                 │
│                                                                       │
│  Dropdown 2: Panel Type                                             │
│  • Forecast                                                          │
│  • Residuals                                                         │
│  • Metrics                                                           │
│  → Switches visualization type                                      │
│                                                                       │
│  Slider (Phase 3): Filter by Metric Range                          │
│  • Min RMSE: [----●--------] Max RMSE                              │
│  → Filters visible workflows                                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Plotly Figure Output                              │
│                                                                       │
│  Capabilities:                                                       │
│  • Hover: Show detailed cognostics                                  │
│  • Zoom: Box select and zoom in                                     │
│  • Pan: Click and drag to explore                                   │
│  • Reset: Double-click to reset view                                │
│  • Export: Download as PNG                                          │
│  • HTML: Save interactive figure to file                            │
│                                                                       │
│  fig.show()         → Display in browser                            │
│  fig.write_html()   → Save to HTML file                             │
│  fig.write_image()  → Save to PNG/PDF                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Nested/Grouped Results Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER CALLS                                      │
│                                                                       │
│  nested_results = wf_set.fit_nested(data, group_col='country')     │
│  fig = nested_results.view_heatmap('rmse', top_n=15)                │
│  fig.show()                                                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  WorkflowSetNestedResults                            │
│                                                                       │
│  Data Structure:                                                     │
│  • results: List[Dict] per workflow                                 │
│    - wflow_id: "formula_1_rf_2"                                     │
│    - nested_fit: NestedWorkflowFit object                           │
│    - outputs: DataFrame with 'group' column                         │
│    - stats: DataFrame with 'group' column                           │
│                                                                       │
│  2D Structure:                                                       │
│  • Dimension 1: Workflows (20-50 typically)                         │
│  • Dimension 2: Groups (5-20 typically)                             │
│  • Total combinations: workflows × groups (100-1000)                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                NestedWorkflowSetDisplay                              │
│                    (NEW CLASS)                                       │
│                                                                       │
│  Three View Levels:                                                  │
│                                                                       │
│  1. OVERVIEW (Heatmap):                                             │
│     • Rows: Workflows                                               │
│     • Columns: Groups                                               │
│     • Cell color: Metric value                                      │
│     • Hover: Show exact value                                       │
│                                                                       │
│  2. WORKFLOW (Drill-down):                                          │
│     • Select single workflow                                        │
│     • Show facet grid: one panel per group                          │
│     • Compare how workflow performs across groups                   │
│                                                                       │
│  3. GROUP (Drill-down):                                             │
│     • Select single group                                           │
│     • Show top N workflows for that group                           │
│     • Compare workflows within single group                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Heatmap Visualization                             │
│                                                                       │
│                     Groups                                           │
│              USA   Germany  Japan  France                           │
│         ┌─────────────────────────────────┐                         │
│  formula│ 1.23│  1.45 │ 1.12 │ 1.67 │  ← Workflow 1                │
│  _1_rf_2├─────────────────────────────────┤                         │
│  formula│ 1.35│  1.28 │ 1.19 │ 1.52 │  ← Workflow 2                │
│  _2_xgb ├─────────────────────────────────┤                         │
│  rec_pca│ 1.41│  1.33 │ 1.25 │ 1.48 │  ← Workflow 3                │
│  _linear├─────────────────────────────────┤                         │
│    ...  │ ... │  ...  │ ...  │ ...  │                              │
│         └─────────────────────────────────┘                         │
│                                                                       │
│  Color Scale: Green (best) → Yellow → Red (worst)                   │
│  Interactive: Click cell → Drill down to that workflow+group        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### ✅ Phase 1: Core Display (Week 1)
- Create `WorkflowSetDisplay` class
- Implement cognostics computation
- Build panel generators (forecast, residuals, metrics)
- Create facet grid assembly
- **Deliverable:** 10 unit tests passing

### ✅ Phase 2: Integration (Week 2)
- Add `view_interactive()` to `WorkflowSetResults`
- Add `view_interactive()` to `WorkflowSetNestedResults`
- Create `NestedWorkflowSetDisplay` class
- Implement heatmap and drill-down views
- **Deliverable:** 15 unit tests passing

### ✅ Phase 3: Example & Docs (Week 3)
- Create comprehensive example notebook
- Update user guide
- Write API documentation
- Performance testing and optimization
- **Deliverable:** Production-ready feature

### 🔮 Phase 4: Advanced Features (Future)
- Interactive filtering UI
- Dash dashboard
- Trelliscope backend (when mature)

---

## Code Impact

### New Files
```
py_workflowsets/
├── display.py                    # NEW: Display classes
│   ├── WorkflowSetDisplay        # ~300 lines
│   └── NestedWorkflowSetDisplay  # ~200 lines
│
tests/test_workflowsets/
└── test_display.py               # NEW: 15 tests

examples/
└── 22_interactive_workflowset_viewer.ipynb  # NEW: Demo notebook
```

### Modified Files
```
py_workflowsets/
└── workflowset.py                # ADD: view_interactive() methods
    ├── WorkflowSetResults.view_interactive()      # ~50 lines
    ├── WorkflowSetResults.view_comparison()       # ~30 lines
    ├── WorkflowSetNestedResults.view_interactive() # ~60 lines
    └── WorkflowSetNestedResults.view_heatmap()    # ~30 lines

py_workflowsets/__init__.py       # ADD: Export display classes
```

### Total Lines of Code
- New code: ~800 lines
- Modified code: ~170 lines
- Tests: ~500 lines
- Documentation: ~400 lines
- **Total: ~1,870 lines**

---

## Example Usage

### Standard Workflow Comparison

```python
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest, boost_tree

# Create 18 workflows (6 formulas × 3 models)
formulas = ["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3",
            "y ~ x1 + x2 + I(x1*x2)", "y ~ .", "y ~ . + I(x1**2)"]
models = [linear_reg(), rand_forest(), boost_tree()]
wf_set = WorkflowSet.from_cross(formulas, models)

# Evaluate
results = wf_set.fit_resamples(cv_folds, metrics=metric_set(rmse, mae))

# Interactive view - top 12 by RMSE
fig = results.view_interactive('forecast', metric='rmse', top_n=12)
fig.show()

# Hover over any panel to see:
# - RMSE: 1.23 ± 0.05
# - MAE: 0.98 ± 0.03
# - R²: 0.85 ± 0.02
# - Model: rand_forest
# - Rank: 3

# Save to HTML
fig.write_html('workflow_comparison.html')
```

### Grouped/Nested Results

```python
# Fit per-country models
nested_results = wf_set.fit_nested(train_data, group_col='country')

# Overview heatmap: workflows × countries
fig = nested_results.view_heatmap('rmse', top_n=15)
fig.show()
# → See which workflows excel in which countries

# Find best workflow overall
best_wf = nested_results.extract_best_workflow('rmse', by_group=False)
print(f"Best: {best_wf}")  # "formula_3_rf_2"

# Drill down: how does best workflow perform per country?
fig = nested_results.view_interactive('workflow', wflow_id=best_wf)
fig.show()
# → Facet grid with one panel per country
```

### Overfitting Detection

```python
from py_rsample import time_series_cv

# Fit on full training
train_results = wf_set.fit_nested(train_data, group_col='country')
outputs, coeffs, train_stats = train_results.extract_outputs()

# Evaluate with CV
cv_folds = time_series_cv(train_data, date_column='date',
                          initial='2 years', assess='6 months')
cv_results = wf_set.fit_nested_resamples(cv_folds, group_col='country')

# Compare
comparison = cv_results.compare_train_cv(train_stats)

# Visualize train vs CV
fig = px.scatter(
    comparison,
    x='rmse_train',
    y='rmse_cv',
    color='fit_quality',  # 🟢 Good, 🟡 Moderate Overfit, 🔴 Severe
    hover_data=['wflow_id', 'group', 'rmse_overfit_ratio'],
    title='Training vs CV Performance'
)
fig.add_trace(go.Scatter(x=[0, 5], y=[0, 5],
                         mode='lines', name='Perfect Fit'))
fig.show()
# → Interactive: click legend to filter out overfit models
```

---

## Performance Characteristics

### Small WorkflowSets (10-20 workflows)
- **Rendering:** <1 second
- **Memory:** <100 MB
- **Optimization:** None needed

### Medium WorkflowSets (20-50 workflows)
- **Rendering:** 1-3 seconds
- **Memory:** 100-500 MB
- **Optimization:** Panel caching

### Large WorkflowSets (50-100 workflows)
- **Rendering:** 3-10 seconds
- **Memory:** 500 MB - 1 GB
- **Optimization:** Lazy loading, pagination

### Nested Results (workflows × groups)
- **10 workflows × 5 groups = 50 combinations:** <2 seconds
- **20 workflows × 10 groups = 200 combinations:** 5-8 seconds
- **50 workflows × 20 groups = 1000 combinations:** Heatmap only, drill-down on demand

---

## Risk Assessment

### Low Risk ✅
- **Technology:** Plotly is mature and stable
- **Dependencies:** Already installed
- **Implementation:** Clear architecture
- **Timeline:** 3 weeks is achievable
- **Testing:** Comprehensive test coverage

### Medium Risk ⚠️
- **Performance:** Large WorkflowSets (50+) may need optimization
- **Mitigation:** Implement caching and lazy loading in Phase 3

### High Risk ❌
- **NONE:** Using trelliscope-py would be high risk (unstable, no docs)

---

## Success Metrics

### User Experience
- ✅ Users can explore 20+ workflows interactively
- ✅ Hover tooltips show relevant cognostics
- ✅ Filtering reduces visible workflows in <1 second
- ✅ Comparison mode enables side-by-side evaluation

### Performance
- ✅ Rendering <3 seconds for 50 workflows
- ✅ Memory <500 MB for typical use cases
- ✅ Responsive interaction (hover, zoom, pan)

### Adoption
- ✅ Used in 3+ example notebooks
- ✅ Documented in user guide
- ✅ Positive user feedback

---

## Comparison: Plotly vs Trelliscope-py

| Feature | Plotly (Recommended) | trelliscope-py |
|---------|---------------------|----------------|
| **Production Ready** | ✅ Yes | ❌ No (experimental) |
| **Documentation** | ✅ Extensive | ❌ Single notebook |
| **Installation** | ✅ pip install | ❌ From source |
| **Stability** | ✅ Stable API | ❌ Breaking changes likely |
| **Community** | ✅ Large (millions) | ❌ Small (6 stars) |
| **Interactive Faceting** | ✅ Yes | ✅ Yes |
| **Cognostics Filtering** | ⚠️ Manual (Phase 3) | ✅ Built-in |
| **HTML Export** | ✅ Yes | ✅ Yes |
| **Pagination** | ⚠️ Manual | ✅ Built-in |
| **Timeline** | ✅ 3 weeks | ❌ Unknown (6+ months?) |
| **Risk** | ✅ Low | ❌ High |

**Verdict:** Plotly provides 80% of functionality with 10% of the risk.

---

## Future Trelliscope Integration

When trelliscope-py matures (stable release, PyPI package, documentation):

### Backend Abstraction Layer

```python
# User code remains the same
fig = results.view_interactive('forecast', backend='trelliscope')
```

### Implementation

```python
# In display.py
def create_display(self, backend='plotly'):
    if backend == 'plotly':
        return self._create_plotly_display()
    elif backend == 'trelliscope':
        return self._create_trelliscope_display()
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

### Migration Path
1. Monitor trelliscope-py releases
2. Test stability and features
3. Implement adapter layer
4. Add to documentation
5. Users can choose: `backend='plotly'` or `backend='trelliscope'`

---

## Conclusion

**Build Plotly-based solution immediately.**

- ✅ Production-ready technology
- ✅ Low risk, clear timeline
- ✅ 80% of desired functionality
- ✅ Easy migration path to trelliscope later
- ✅ No project delays

**Estimated Effort:** 2-3 weeks
**Priority:** HIGH
**Status:** READY TO IMPLEMENT

---

## Next Steps

1. **Week 1:** Implement core `WorkflowSetDisplay` class
2. **Week 2:** Integrate with results classes, add nested support
3. **Week 3:** Create example notebook, update docs, test performance
4. **Week 4+:** (Optional) Advanced features, Dash app

---

## Related Documents

- **Full Research Report:** `TRELLISCOPE_INTEGRATION_RESEARCH_REPORT.md` (14,000+ words)
- **Implementation Plan:** `INTERACTIVE_VIEWER_IMPLEMENTATION_PLAN.md` (detailed code)
- **Data Flow:** This document (visualizations)

---

**END OF SUMMARY**
