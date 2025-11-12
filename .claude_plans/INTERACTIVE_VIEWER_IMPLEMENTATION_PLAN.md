# Interactive WorkflowSet Viewer - Implementation Plan
**Date:** 2025-11-12
**Status:** PROPOSED
**Priority:** HIGH
**Estimated Effort:** 2-3 weeks

---

## Overview

Build Plotly-based interactive viewer for WorkflowSet results to replace static matplotlib plots with rich, interactive visualizations including filtering, drill-down, and cognostics display.

---

## Phase 1: Core Display Class (Week 1)

### 1.1 Create Display Module Structure

**File:** `py_workflowsets/display.py`

```python
"""
Interactive display utilities for WorkflowSet results.

Provides Plotly-based interactive visualizations with cognostics,
filtering, and drill-down capabilities.
"""

from typing import List, Dict, Any, Optional, Union, Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class WorkflowSetDisplay:
    """
    Interactive display for WorkflowSet evaluation results.

    Transforms WorkflowSet results into interactive Plotly visualizations
    with cognostics (summary statistics) for filtering and exploration.

    Attributes:
        results: WorkflowSetResults or WorkflowSetNestedResults object
        cognostics: DataFrame with summary statistics for each workflow
        panel_cache: Cache of generated panels for performance
    """

    def __init__(self, results):
        """
        Initialize display from WorkflowSet results.

        Args:
            results: WorkflowSetResults or WorkflowSetNestedResults
        """
        self.results = results
        self.cognostics = None
        self.panel_cache = {}
        self._compute_cognostics()

    def _compute_cognostics(self) -> pd.DataFrame:
        """
        Compute summary statistics (cognostics) for each workflow.

        Cognostics are the statistics used for interactive filtering
        and sorting in the display.

        Returns:
            DataFrame with columns:
            - wflow_id: Workflow identifier
            - {metric}_mean: Mean metric value
            - {metric}_std: Std dev of metric
            - model: Model type
            - preprocessor: Preprocessor type
            - complexity: Model complexity score (if available)
        """
        # Implementation details in section 1.2
        pass

    def create_display(
        self,
        panel_type: Literal['forecast', 'residuals', 'metrics', 'comparison'],
        metric: str = 'rmse',
        top_n: int = 20,
        layout: Dict[str, Any] = None
    ) -> go.Figure:
        """
        Create interactive display with specified panel type.

        Args:
            panel_type: Type of visualization to create
            metric: Metric for ranking/sorting
            top_n: Number of workflows to display
            layout: Custom layout options

        Returns:
            Plotly Figure with interactive controls
        """
        # Implementation details in section 1.3
        pass

    # Additional methods in sections 1.4-1.6
```

### 1.2 Cognostics Computation

```python
def _compute_cognostics(self) -> pd.DataFrame:
    """Compute cognostics from results."""
    # Get metrics in wide format
    metrics_df = self.results.collect_metrics(summarize=True)

    # Pivot to wide: one column per metric
    wide = metrics_df.pivot_table(
        index=['wflow_id', 'preprocessor', 'model'],
        columns='metric',
        values=['mean', 'std']
    )

    # Flatten multi-level columns
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()

    # Add derived cognostics
    if 'rmse_mean' in wide.columns and 'rmse_std' in wide.columns:
        # Coefficient of variation (stability indicator)
        wide['rmse_cv'] = wide['rmse_std'] / wide['rmse_mean']

    if 'r_squared_mean' in wide.columns:
        # Goodness of fit category
        wide['fit_category'] = pd.cut(
            wide['r_squared_mean'],
            bins=[0, 0.5, 0.7, 0.9, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )

    # Add rank by primary metric (typically RMSE)
    if 'rmse_mean' in wide.columns:
        wide['rank_rmse'] = wide['rmse_mean'].rank()

    # Add model complexity (if available from fit objects)
    wide['complexity'] = self._estimate_complexity()

    self.cognostics = wide
    return wide

def _estimate_complexity(self) -> pd.Series:
    """
    Estimate model complexity for each workflow.

    Returns:
        Series with complexity scores (higher = more complex)
    """
    complexity_map = {
        'linear_reg': 1,
        'decision_tree': 2,
        'rand_forest': 3,
        'boost_tree': 4,
        'mlp': 5
    }

    return self.cognostics['model'].map(complexity_map).fillna(3)
```

### 1.3 Panel Creation Functions

```python
def _create_forecast_panel(
    self,
    wflow_id: str,
    outputs: pd.DataFrame
) -> go.Figure:
    """
    Create time series forecast panel for a workflow.

    Shows actuals vs fitted values over time with train/test split.
    """
    wf_data = outputs[outputs['wflow_id'] == wflow_id].copy()

    # Sort by date if available
    if 'date' in wf_data.columns:
        wf_data = wf_data.sort_values('date')

    fig = go.Figure()

    # Add actuals
    fig.add_trace(go.Scatter(
        x=wf_data.index if 'date' not in wf_data.columns else wf_data['date'],
        y=wf_data['actuals'],
        name='Actual',
        mode='lines',
        line=dict(color='black', width=2),
        hovertemplate='<b>Actual</b><br>%{y:.2f}<extra></extra>'
    ))

    # Add fitted (colored by split)
    for split in wf_data['split'].unique():
        split_data = wf_data[wf_data['split'] == split]
        color = 'blue' if split == 'train' else 'red'

        fig.add_trace(go.Scatter(
            x=split_data.index if 'date' not in split_data.columns else split_data['date'],
            y=split_data['fitted'],
            name=f'Fitted ({split})',
            mode='lines',
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>Fitted ({split})</b><br>%{{y:.2f}}<extra></extra>'
        ))

    # Get cognostics for this workflow
    cogs = self.cognostics[self.cognostics['wflow_id'] == wflow_id].iloc[0]

    # Update layout with title and cognostics
    title_text = f"<b>{wflow_id}</b><br>"
    title_text += f"<sub>RMSE: {cogs.get('rmse_mean', 'N/A'):.3f} | "
    title_text += f"MAE: {cogs.get('mae_mean', 'N/A'):.3f} | "
    title_text += f"R²: {cogs.get('r_squared_mean', 'N/A'):.3f}</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title="Time" if 'date' not in wf_data.columns else "Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=300,
        margin=dict(l=50, r=20, t=80, b=40)
    )

    return fig

def _create_residuals_panel(
    self,
    wflow_id: str,
    outputs: pd.DataFrame
) -> go.Figure:
    """
    Create residual diagnostic panel (4 subplots).

    1. Residuals vs Fitted
    2. Q-Q Plot
    3. Scale-Location
    4. Residuals vs Leverage (if available)
    """
    wf_data = outputs[outputs['wflow_id'] == wflow_id].copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals vs Fitted',
            'Normal Q-Q',
            'Scale-Location',
            'Residuals Distribution'
        )
    )

    # 1. Residuals vs Fitted
    fig.add_trace(
        go.Scatter(
            x=wf_data['fitted'],
            y=wf_data['residuals'],
            mode='markers',
            marker=dict(size=4, opacity=0.6),
            name='Residuals',
            hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2. Q-Q Plot
    from scipy import stats
    residuals = wf_data['residuals'].dropna()
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(size=4, opacity=0.6),
            name='Q-Q',
            hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    # Add reference line
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Scale-Location (sqrt of standardized residuals vs fitted)
    std_residuals = (wf_data['residuals'] - wf_data['residuals'].mean()) / wf_data['residuals'].std()
    sqrt_std_resid = np.sqrt(np.abs(std_residuals))

    fig.add_trace(
        go.Scatter(
            x=wf_data['fitted'],
            y=sqrt_std_resid,
            mode='markers',
            marker=dict(size=4, opacity=0.6),
            name='Scale-Location',
            hovertemplate='Fitted: %{x:.2f}<br>√|Std Resid|: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Residuals Distribution
    fig.add_trace(
        go.Histogram(
            x=wf_data['residuals'],
            nbinsx=30,
            name='Distribution',
            marker=dict(opacity=0.7)
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
    fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=1)

    fig.update_xaxes(title_text="Residuals", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        title=f"<b>{wflow_id}</b> - Residual Diagnostics",
        height=600,
        showlegend=False
    )

    return fig

def _create_metrics_panel(
    self,
    wflow_id: str,
    metrics: pd.DataFrame
) -> go.Figure:
    """Create bar chart comparing metrics for a workflow."""
    wf_metrics = metrics[
        (metrics['wflow_id'] == wflow_id) &
        (metrics['metric'].isin(['rmse', 'mae', 'r_squared', 'mape']))
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=wf_metrics['metric'],
            y=wf_metrics['mean'],
            error_y=dict(
                type='data',
                array=wf_metrics['std'],
                visible=True
            ),
            marker=dict(
                color=['red', 'orange', 'green', 'blue'],
                opacity=0.7
            ),
            text=wf_metrics['mean'].round(3),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f"<b>{wflow_id}</b> - Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=300,
        showlegend=False
    )

    return fig
```

### 1.4 Facet Grid Generation

```python
def create_facet_display(
    self,
    panel_type: str = 'forecast',
    metric: str = 'rmse',
    top_n: int = 20,
    facet_col_wrap: int = 3
) -> go.Figure:
    """
    Create facet grid display with top N workflows.

    Args:
        panel_type: Type of panel ('forecast', 'residuals', 'metrics')
        metric: Metric for ranking
        top_n: Number of workflows to display
        facet_col_wrap: Number of columns in grid

    Returns:
        Plotly Figure with facet grid
    """
    # Rank workflows by metric
    ranked = self.results.rank_results(metric=metric, n=top_n)
    top_wflow_ids = ranked['wflow_id'].tolist()

    # Get outputs for top workflows
    outputs = self.results.collect_outputs() if hasattr(self.results, 'collect_outputs') else None
    metrics_df = self.results.collect_metrics(summarize=True)

    # Calculate subplot layout
    n_rows = (top_n + facet_col_wrap - 1) // facet_col_wrap

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=facet_col_wrap,
        subplot_titles=top_wflow_ids,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Generate panels
    for idx, wflow_id in enumerate(top_wflow_ids):
        row = (idx // facet_col_wrap) + 1
        col = (idx % facet_col_wrap) + 1

        # Get panel figure
        if panel_type == 'forecast' and outputs is not None:
            panel = self._create_forecast_panel(wflow_id, outputs)
        elif panel_type == 'metrics':
            panel = self._create_metrics_panel(wflow_id, metrics_df)
        else:
            continue

        # Add panel traces to subplot
        for trace in panel.data:
            fig.add_trace(trace, row=row, col=col)

    # Update layout
    fig.update_layout(
        title=f"Top {top_n} Workflows by {metric.upper()}",
        height=300 * n_rows,
        showlegend=True,
        hovermode='closest'
    )

    return fig
```

### 1.5 Interactive Controls

```python
def add_interactive_controls(self, fig: go.Figure) -> go.Figure:
    """
    Add interactive controls to figure.

    Adds:
    - Dropdown for metric selection
    - Buttons for panel type
    - Range sliders for filtering
    """
    # Add dropdown for metric selection
    metrics = ['rmse', 'mae', 'r_squared', 'mape']
    metric_buttons = []

    for metric in metrics:
        metric_buttons.append(
            dict(
                label=metric.upper(),
                method="update",
                args=[
                    {"visible": [True] * len(fig.data)},
                    {"title": f"Sorted by {metric.upper()}"}
                ]
            )
        )

    # Add buttons for panel type
    panel_buttons = [
        dict(label="Forecast", method="update", args=[{"visible": [True]}]),
        dict(label="Residuals", method="update", args=[{"visible": [False]}]),
        dict(label="Metrics", method="update", args=[{"visible": [False]}])
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=metric_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            ),
            dict(
                buttons=panel_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    return fig
```

### 1.6 Testing Strategy

**File:** `tests/test_workflowsets/test_display.py`

```python
import pytest
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet, WorkflowSetDisplay
from py_parsnip import linear_reg, rand_forest
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae


class TestWorkflowSetDisplay:
    """Tests for WorkflowSetDisplay class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample WorkflowSet results for testing."""
        # Create simple workflows
        formulas = ["y ~ x1", "y ~ x1 + x2"]
        models = [linear_reg(), rand_forest()]
        wf_set = WorkflowSet.from_cross(formulas, models)

        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })

        # Fit resamples
        folds = vfold_cv(data, v=3)
        results = wf_set.fit_resamples(folds, metrics=metric_set(rmse, mae))

        return results

    def test_display_init(self, sample_results):
        """Test Display initialization."""
        display = WorkflowSetDisplay(sample_results)

        assert display.results is not None
        assert display.cognostics is not None
        assert isinstance(display.cognostics, pd.DataFrame)

    def test_cognostics_computation(self, sample_results):
        """Test cognostics DataFrame has required columns."""
        display = WorkflowSetDisplay(sample_results)
        cogs = display.cognostics

        # Check required columns
        assert 'wflow_id' in cogs.columns
        assert 'rmse_mean' in cogs.columns
        assert 'mae_mean' in cogs.columns
        assert 'model' in cogs.columns
        assert 'preprocessor' in cogs.columns

    def test_forecast_panel_creation(self, sample_results):
        """Test forecast panel generation."""
        display = WorkflowSetDisplay(sample_results)
        wflow_id = display.cognostics['wflow_id'].iloc[0]

        # Get outputs
        outputs = sample_results.collect_outputs() if hasattr(sample_results, 'collect_outputs') else None

        if outputs is not None:
            fig = display._create_forecast_panel(wflow_id, outputs)

            assert fig is not None
            assert len(fig.data) > 0  # Has traces
            assert fig.layout.title.text is not None  # Has title

    def test_facet_display_creation(self, sample_results):
        """Test facet grid generation."""
        display = WorkflowSetDisplay(sample_results)

        fig = display.create_facet_display(
            panel_type='forecast',
            metric='rmse',
            top_n=4,
            facet_col_wrap=2
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_interactive_controls(self, sample_results):
        """Test addition of interactive controls."""
        display = WorkflowSetDisplay(sample_results)

        fig = display.create_facet_display(panel_type='metrics', top_n=4)
        fig = display.add_interactive_controls(fig)

        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0

# Additional tests for nested results, edge cases, etc.
```

---

## Phase 2: Integration with Results Classes (Week 2)

### 2.1 Add Methods to WorkflowSetResults

**File:** `py_workflowsets/workflowset.py` (add to existing class)

```python
class WorkflowSetResults:
    """Results from fitting a WorkflowSet."""

    # ... existing methods ...

    def view_interactive(
        self,
        panel_type: Literal['forecast', 'residuals', 'metrics', 'comparison'] = 'forecast',
        metric: str = 'rmse',
        top_n: int = 20,
        facet_col_wrap: int = 3,
        backend: str = 'plotly'
    ) -> go.Figure:
        """
        Create interactive visualization of workflow comparison.

        Args:
            panel_type: Type of visualization
                - 'forecast': Time series with actuals vs fitted
                - 'residuals': Diagnostic residual plots
                - 'metrics': Bar chart of performance metrics
                - 'comparison': Side-by-side workflow comparison
            metric: Metric to use for ranking/sorting (default: 'rmse')
            top_n: Number of top workflows to display (default: 20)
            facet_col_wrap: Number of columns in facet grid (default: 3)
            backend: Visualization backend (default: 'plotly')
                Currently only 'plotly' supported. Future: 'trelliscope'

        Returns:
            Plotly Figure with interactive controls

        Examples:
            >>> # Basic forecast view
            >>> fig = results.view_interactive('forecast', metric='rmse', top_n=10)
            >>> fig.show()

            >>> # Save to HTML
            >>> fig.write_html('workflow_comparison.html')

            >>> # Residual diagnostics
            >>> fig = results.view_interactive('residuals', top_n=5)
            >>> fig.show()
        """
        from py_workflowsets.display import WorkflowSetDisplay

        if backend != 'plotly':
            raise ValueError("Only 'plotly' backend currently supported")

        display = WorkflowSetDisplay(self)

        if panel_type == 'comparison':
            return display.create_comparison_display(metric=metric, top_n=top_n)
        else:
            fig = display.create_facet_display(
                panel_type=panel_type,
                metric=metric,
                top_n=top_n,
                facet_col_wrap=facet_col_wrap
            )
            fig = display.add_interactive_controls(fig)
            return fig

    def view_comparison(
        self,
        wflow_ids: List[str],
        panel_type: str = 'forecast'
    ) -> go.Figure:
        """
        Compare specific workflows side-by-side.

        Args:
            wflow_ids: List of 2-4 workflow IDs to compare
            panel_type: Type of panel to show

        Returns:
            Plotly Figure with side-by-side comparison

        Examples:
            >>> # Compare top 2 workflows
            >>> top2 = results.rank_results('rmse', n=2)['wflow_id'].tolist()
            >>> fig = results.view_comparison(top2)
            >>> fig.show()
        """
        from py_workflowsets.display import WorkflowSetDisplay

        if len(wflow_ids) < 2 or len(wflow_ids) > 4:
            raise ValueError("Can compare 2-4 workflows at a time")

        display = WorkflowSetDisplay(self)
        return display.create_comparison_display(wflow_ids=wflow_ids, panel_type=panel_type)
```

### 2.2 Add Methods to WorkflowSetNestedResults

```python
class WorkflowSetNestedResults:
    """Results from nested/grouped workflow evaluation."""

    # ... existing methods ...

    def view_interactive(
        self,
        level: Literal['overview', 'workflow', 'group'] = 'overview',
        metric: str = 'rmse',
        top_n_workflows: int = 10,
        groups: Optional[List[str]] = None,
        wflow_id: Optional[str] = None
    ) -> go.Figure:
        """
        Interactive visualization of nested results.

        Args:
            level: Visualization level
                - 'overview': Heatmap of workflows × groups
                - 'workflow': Single workflow across all groups
                - 'group': All workflows within single group
            metric: Metric to visualize
            top_n_workflows: Number of workflows to include
            groups: Optional list of groups to include
            wflow_id: Workflow ID for 'workflow' level

        Returns:
            Plotly Figure

        Examples:
            >>> # Overview heatmap
            >>> fig = results.view_interactive('overview', metric='rmse')
            >>> fig.show()

            >>> # Drill down to best workflow
            >>> best_wf = results.extract_best_workflow('rmse')
            >>> fig = results.view_interactive('workflow', wflow_id=best_wf)
            >>> fig.show()
        """
        from py_workflowsets.display import NestedWorkflowSetDisplay

        display = NestedWorkflowSetDisplay(self)

        if level == 'overview':
            return display.create_heatmap(metric=metric, top_n=top_n_workflows, groups=groups)
        elif level == 'workflow':
            if wflow_id is None:
                wflow_id = self.extract_best_workflow(metric, by_group=False)
            return display.create_workflow_view(wflow_id=wflow_id, groups=groups)
        elif level == 'group':
            if groups is None or len(groups) != 1:
                raise ValueError("Must specify exactly one group for 'group' level")
            return display.create_group_view(group=groups[0], metric=metric, top_n=top_n_workflows)
        else:
            raise ValueError(f"Unknown level: {level}")

    def view_heatmap(
        self,
        metric: str = 'rmse',
        top_n: int = 20,
        groups: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Heatmap of metric values: workflows × groups.

        Args:
            metric: Metric to visualize
            top_n: Number of workflows to include
            groups: Optional list of groups

        Returns:
            Plotly Figure with heatmap

        Examples:
            >>> fig = results.view_heatmap('rmse', top_n=15)
            >>> fig.show()
        """
        from py_workflowsets.display import NestedWorkflowSetDisplay

        display = NestedWorkflowSetDisplay(self)
        return display.create_heatmap(metric=metric, top_n=top_n, groups=groups)
```

---

## Phase 3: Nested Display Implementation (Week 2)

### 3.1 NestedWorkflowSetDisplay Class

**File:** `py_workflowsets/display.py` (add to existing file)

```python
class NestedWorkflowSetDisplay:
    """
    Interactive display for nested/grouped WorkflowSet results.

    Handles 2D structure: workflows × groups.
    """

    def __init__(self, results):
        """
        Initialize display from nested results.

        Args:
            results: WorkflowSetNestedResults object
        """
        self.results = results
        self.group_col = results.group_col
        self.cognostics = self._compute_cognostics()

    def create_heatmap(
        self,
        metric: str = 'rmse',
        top_n: int = 20,
        groups: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create heatmap: workflows (rows) × groups (columns).

        Color represents metric value. Darker = better (for RMSE/MAE)
        or worse (for R²).
        """
        # Get metrics per group
        metrics = self.results.collect_metrics(by_group=True, split='test')
        metrics = metrics[metrics['metric'] == metric]

        # Filter to top N workflows (by average)
        avg_metrics = metrics.groupby('wflow_id')['value'].mean().nsmallest(top_n)
        top_wflows = avg_metrics.index.tolist()
        metrics = metrics[metrics['wflow_id'].isin(top_wflows)]

        # Filter to selected groups
        if groups:
            metrics = metrics[metrics['group'].isin(groups)]

        # Pivot to matrix
        matrix = metrics.pivot(index='wflow_id', columns='group', values='value')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='RdYlGn_r' if metric in ['rmse', 'mae'] else 'RdYlGn',
            text=matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Workflow: %{y}<br>Group: %{x}<br>%{metric}: %{z:.3f}<extra></extra>'.replace('%{metric}', metric.upper())
        ))

        fig.update_layout(
            title=f'{metric.upper()} by Workflow and Group',
            xaxis_title='Group',
            yaxis_title='Workflow',
            height=max(400, 20 * len(matrix)),
            width=max(600, 50 * len(matrix.columns))
        )

        return fig

    def create_workflow_view(
        self,
        wflow_id: str,
        groups: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Show single workflow across all groups.

        Creates facet grid: one panel per group.
        """
        outputs = self.results.collect_outputs()
        wf_outputs = outputs[outputs['wflow_id'] == wflow_id]

        if groups:
            wf_outputs = wf_outputs[wf_outputs['group'].isin(groups)]

        # Create facet plot
        fig = px.line(
            wf_outputs,
            x='date' if 'date' in wf_outputs.columns else wf_outputs.index,
            y=['actuals', 'fitted'],
            facet_col='group',
            facet_col_wrap=3,
            title=f'Workflow {wflow_id} - Performance by Group',
            labels={'value': 'Value', 'variable': 'Series'}
        )

        # Add metrics annotations
        metrics = self.results.collect_metrics(by_group=True, split='test')
        metrics = metrics[metrics['wflow_id'] == wflow_id]

        for i, group in enumerate(wf_outputs['group'].unique()):
            group_metrics = metrics[metrics['group'] == group]
            rmse = group_metrics[group_metrics['metric'] == 'rmse']['value'].iloc[0] if 'rmse' in group_metrics['metric'].values else None
            mae = group_metrics[group_metrics['metric'] == 'mae']['value'].iloc[0] if 'mae' in group_metrics['metric'].values else None

            annotation = f"RMSE: {rmse:.3f}<br>MAE: {mae:.3f}" if rmse and mae else ""

            # Add annotation to subplot
            fig.add_annotation(
                x=0.5, y=1.05,
                xref='x domain', yref='y domain',
                text=annotation,
                showarrow=False,
                font=dict(size=10),
                xanchor='center',
                row=(i // 3) + 1,
                col=(i % 3) + 1
            )

        return fig
```

---

## Phase 4: Example Notebook (Week 3)

### 4.1 Comprehensive Example

**File:** `examples/22_interactive_workflowset_viewer.ipynb`

```python
# Cell 1: Setup
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest, boost_tree
from py_recipes import recipe
from py_rsample import time_series_cv
from py_yardstick import metric_set, rmse, mae, r_squared
import pandas as pd
import numpy as np

# Cell 2: Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
data = pd.DataFrame({
    'date': dates,
    'y': np.cumsum(np.random.randn(1000)) + 50,
    'x1': np.random.randn(1000),
    'x2': np.random.randn(1000),
    'x3': np.random.randn(1000),
    'country': np.random.choice(['USA', 'Germany', 'Japan'], 1000)
})

# Cell 3: Create multiple preprocessing strategies
formulas = {
    'minimal': 'y ~ x1',
    'base': 'y ~ x1 + x2',
    'full': 'y ~ x1 + x2 + x3',
    'interaction': 'y ~ x1 + x2 + I(x1*x2)'
}

recipes = {
    'normalized': recipe().step_normalize(['x1', 'x2', 'x3']),
    'pca': recipe().step_normalize(['x1', 'x2', 'x3']).step_pca(num_comp=2)
}

# Cell 4: Create workflows
models = [
    linear_reg(),
    rand_forest().set_mode('regression'),
    boost_tree().set_mode('regression')
]

wf_set = WorkflowSet.from_cross(
    preproc={**formulas, **recipes},
    models=models
)

print(f"Created {len(wf_set)} workflows")

# Cell 5: Evaluate workflows
train = data.iloc[:800]
test = data.iloc[800:]

folds = time_series_cv(train, date_column='date', initial='400 days', assess='100 days')
results = wf_set.fit_resamples(folds, metrics=metric_set(rmse, mae, r_squared))

# Cell 6: INTERACTIVE FORECAST VIEW
fig = results.view_interactive(
    panel_type='forecast',
    metric='rmse',
    top_n=12,
    facet_col_wrap=3
)
fig.show()

# Cell 7: METRICS COMPARISON
fig = results.view_interactive(
    panel_type='metrics',
    metric='rmse',
    top_n=10
)
fig.show()

# Cell 8: COMPARE TOP 3 WORKFLOWS
top3 = results.rank_results('rmse', n=3)['wflow_id'].tolist()
fig = results.view_comparison(top3, panel_type='forecast')
fig.show()

# Cell 9: GROUPED/NESTED RESULTS
nested_results = wf_set.fit_nested(train, group_col='country')

# Cell 10: OVERVIEW HEATMAP
fig = nested_results.view_heatmap('rmse', top_n=15)
fig.show()

# Cell 11: DRILL DOWN TO BEST WORKFLOW
best_wf = nested_results.extract_best_workflow('rmse', by_group=False)
print(f"Best workflow: {best_wf}")

fig = nested_results.view_interactive('workflow', wflow_id=best_wf)
fig.show()

# Cell 12: SAVE TO HTML
fig.write_html('workflow_comparison.html')
print("Saved to workflow_comparison.html")
```

---

## Testing Plan

### Unit Tests (15 tests)

1. `test_display_init` - Display initialization
2. `test_cognostics_computation` - Cognostics DataFrame
3. `test_forecast_panel` - Forecast panel generation
4. `test_residuals_panel` - Residuals panel generation
5. `test_metrics_panel` - Metrics panel generation
6. `test_facet_display` - Facet grid creation
7. `test_interactive_controls` - Control addition
8. `test_comparison_display` - Side-by-side comparison
9. `test_nested_heatmap` - Heatmap for nested results
10. `test_workflow_view` - Single workflow across groups
11. `test_group_view` - Single group across workflows
12. `test_empty_results` - Handle empty results gracefully
13. `test_missing_data` - Handle missing metrics/outputs
14. `test_backend_selection` - Backend parameter validation
15. `test_html_export` - Export to HTML file

### Integration Tests (5 tests)

1. `test_full_workflow` - End-to-end with real workflows
2. `test_nested_workflow` - Grouped results pipeline
3. `test_cv_workflow` - CV results visualization
4. `test_large_workflowset` - Performance with 50+ workflows
5. `test_notebook_execution` - Run example notebook

---

## Documentation Updates

### User Guide Section

**File:** `docs/user_guide/interactive_visualization.md`

```markdown
# Interactive Workflow Visualization

## Overview

The `view_interactive()` method provides rich, interactive visualizations
of WorkflowSet results using Plotly. This replaces the static matplotlib
`autoplot()` with fully interactive facet grids that support:

- Hover tooltips with cognostics (summary statistics)
- Zoom and pan
- Export to HTML
- Interactive filtering (coming in Phase 3)

## Basic Usage

### Forecast Comparison

\`\`\`python
# Evaluate workflows
results = wf_set.fit_resamples(folds, metrics=metrics)

# Create interactive forecast view
fig = results.view_interactive('forecast', metric='rmse', top_n=12)
fig.show()
\`\`\`

### Metrics Comparison

\`\`\`python
fig = results.view_interactive('metrics', metric='rmse', top_n=10)
fig.show()
\`\`\`

### Side-by-Side Comparison

\`\`\`python
# Get top 3 workflows
top3 = results.rank_results('rmse', n=3)['wflow_id'].tolist()

# Compare them
fig = results.view_comparison(top3)
fig.show()
\`\`\`

## Grouped Results

### Heatmap Overview

\`\`\`python
# Fit nested models
nested_results = wf_set.fit_nested(data, group_col='country')

# Create heatmap: workflows × countries
fig = nested_results.view_heatmap('rmse', top_n=15)
fig.show()
\`\`\`

### Drill Down to Workflow

\`\`\`python
# Find best workflow
best_wf = nested_results.extract_best_workflow('rmse')

# See how it performs per group
fig = nested_results.view_interactive('workflow', wflow_id=best_wf)
fig.show()
\`\`\`

## Exporting

Save any figure to HTML for sharing:

\`\`\`python
fig.write_html('my_comparison.html')
\`\`\`

The HTML file is self-contained and fully interactive.

## Tips

1. **Start with heatmap** for nested results to identify patterns
2. **Use top_n** to avoid overwhelming displays
3. **Export to HTML** for presentations and reports
4. **Hover over panels** to see detailed cognostics
\`\`\`

---

## Performance Optimization (Week 3)

### Caching Strategy

```python
class WorkflowSetDisplay:
    """Display class with caching."""

    def __init__(self, results):
        self.results = results
        self.panel_cache = {}  # Cache generated panels
        self.cognostics_cache = None

    def _get_or_create_panel(self, wflow_id: str, panel_type: str) -> go.Figure:
        """Get panel from cache or create if not exists."""
        cache_key = f"{wflow_id}_{panel_type}"

        if cache_key not in self.panel_cache:
            if panel_type == 'forecast':
                outputs = self.results.collect_outputs()
                self.panel_cache[cache_key] = self._create_forecast_panel(wflow_id, outputs)
            elif panel_type == 'residuals':
                outputs = self.results.collect_outputs()
                self.panel_cache[cache_key] = self._create_residuals_panel(wflow_id, outputs)
            # ...

        return self.panel_cache[cache_key]
```

### Lazy Loading

```python
def create_facet_display_lazy(
    self,
    panel_type: str = 'forecast',
    top_n: int = 20
) -> go.Figure:
    """
    Create display with lazy loading.

    Only loads visible panels initially. Additional panels
    loaded on user interaction.
    """
    # Load first page of panels
    panels_per_page = 12
    initial_wflows = self._get_top_workflows(top_n)[:panels_per_page]

    # Create figure with placeholders for rest
    # Implementation with Dash callbacks for on-demand loading
```

---

## Success Criteria

### Phase 1 (Week 1)
- ✅ Display class implemented and tested
- ✅ Forecast, residuals, metrics panels working
- ✅ Cognostics computed correctly
- ✅ 10 unit tests passing

### Phase 2 (Week 2)
- ✅ `view_interactive()` added to WorkflowSetResults
- ✅ `view_interactive()` added to WorkflowSetNestedResults
- ✅ Nested displays (heatmap, workflow view) working
- ✅ 15 unit tests passing

### Phase 3 (Week 3)
- ✅ Example notebook complete and tested
- ✅ Documentation updated
- ✅ Integration tests passing
- ✅ Performance acceptable (<3s for 20 workflows)

---

## Future Enhancements (Phase 4+)

### Interactive Filtering (Week 4)
- Add range sliders for metric filtering
- Add model type dropdown
- Add preprocessor type filter

### Dash Dashboard (Week 5-6)
- Convert to full Dash app
- Multi-page layout
- Persistent state
- Real-time updates

### Trelliscope Backend (Future)
- Monitor trelliscope-py development
- Create adapter layer
- Implement backend switching

---

**END OF IMPLEMENTATION PLAN**
