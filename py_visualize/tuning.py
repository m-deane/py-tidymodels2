"""
Hyperparameter tuning visualization with Plotly

Interactive visualizations for hyperparameter optimization results.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_tune_results(
    tune_results,  # TuneResults
    metric: str = "rmse",
    plot_type: str = "auto",
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_best: int = 0
) -> go.Figure:
    """Visualize hyperparameter tuning results.

    Creates interactive visualizations showing how model performance varies
    with hyperparameter values.

    Parameters
    ----------
    tune_results : TuneResults
        Results from tune_grid() or tune_bayes()
    metric : str, default="rmse"
        Metric to visualize. Should match a metric used in tuning.
    plot_type : str, default="auto"
        Type of plot to create:
        - "auto": Automatically choose based on number of parameters
        - "line": Line plot for single parameter (1D)
        - "heatmap": Heatmap for two parameters (2D)
        - "parallel": Parallel coordinates for 3+ parameters
        - "scatter": Scatter plot matrix for 2+ parameters
    title : str, optional
        Plot title. If None, auto-generates title.
    height : int, default=500
        Plot height in pixels
    width : int, optional
        Plot width in pixels. If None, uses Plotly default.
    show_best : int, default=0
        If > 0, highlight the top N best performing configurations

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> from py_workflows import workflow
    >>> from py_parsnip import rand_forest
    >>> from py_tune import tune, tune_grid, grid_regular
    >>> from py_visualize import plot_tune_results
    >>>
    >>> # Create tuning workflow
    >>> wf = workflow().add_formula("sales ~ .").add_model(
    ...     rand_forest(trees=tune(), min_n=tune()).set_mode("regression")
    ... )
    >>>
    >>> # Tune
    >>> results = tune_grid(wf, resamples=cv_splits, grid=10)
    >>>
    >>> # Visualize tuning results
    >>> fig = plot_tune_results(results, metric="rmse", show_best=3)
    >>> fig.show()
    """
    # Get tuning results DataFrame
    results_df = tune_results.results.copy()

    # Identify parameter columns (start with tuned parameters)
    param_cols = [col for col in results_df.columns if col not in ["metric", "value", "split", ".config"]]

    # Filter by metric
    if metric not in results_df["metric"].values:
        available_metrics = results_df["metric"].unique()
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(available_metrics)}")

    metric_data = results_df[results_df["metric"] == metric].copy()

    # Determine plot type automatically if needed
    if plot_type == "auto":
        n_params = len(param_cols)
        if n_params == 1:
            plot_type = "line"
        elif n_params == 2:
            plot_type = "heatmap"
        else:
            plot_type = "parallel"

    # Route to appropriate plot function
    if plot_type == "line":
        return _plot_line_tuning(metric_data, param_cols, metric, title, height, width, show_best)
    elif plot_type == "heatmap":
        return _plot_heatmap_tuning(metric_data, param_cols, metric, title, height, width, show_best)
    elif plot_type == "parallel":
        return _plot_parallel_tuning(metric_data, param_cols, metric, title, height, width, show_best)
    elif plot_type == "scatter":
        return _plot_scatter_tuning(metric_data, param_cols, metric, title, height, width, show_best)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be one of: 'auto', 'line', 'heatmap', 'parallel', 'scatter'")


def _plot_line_tuning(
    metric_data: pd.DataFrame,
    param_cols: List[str],
    metric: str,
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_best: int
) -> go.Figure:
    """Create line plot for single parameter tuning."""
    if len(param_cols) == 0:
        raise ValueError("No tunable parameters found")

    param_col = param_cols[0]

    fig = go.Figure()

    # Group by parameter value and aggregate
    grouped = metric_data.groupby(param_col)["value"].agg(["mean", "std"]).reset_index()
    grouped = grouped.sort_values(param_col)

    # Main line
    fig.add_trace(go.Scatter(
        x=grouped[param_col],
        y=grouped["mean"],
        mode="lines+markers",
        name=metric.upper(),
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=8),
        hovertemplate=f"{param_col}: %{{x}}<br>{metric.upper()}: %{{y:.4f}}<extra></extra>"
    ))

    # Add error bands (if std available)
    if not grouped["std"].isna().all():
        fig.add_trace(go.Scatter(
            x=grouped[param_col],
            y=grouped["mean"] + grouped["std"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=grouped[param_col],
            y=grouped["mean"] - grouped["std"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="±1 Std Dev",
            fillcolor="rgba(31, 119, 180, 0.2)"
        ))

    # Highlight best configurations
    if show_best > 0:
        best_configs = metric_data.nsmallest(show_best, "value")
        fig.add_trace(go.Scatter(
            x=best_configs[param_col],
            y=best_configs["value"],
            mode="markers",
            marker=dict(size=12, color="red", symbol="star"),
            name=f"Top {show_best}",
            hovertemplate=f"{param_col}: %{{x}}<br>{metric.upper()}: %{{y:.4f}}<extra>Best</extra>"
        ))

    # Layout
    layout_config = {
        "title": title or f"{metric.upper()} vs {param_col}",
        "xaxis_title": param_col,
        "yaxis_title": metric.upper(),
        "hovermode": "x",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_heatmap_tuning(
    metric_data: pd.DataFrame,
    param_cols: List[str],
    metric: str,
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_best: int
) -> go.Figure:
    """Create heatmap for two parameter tuning."""
    if len(param_cols) < 2:
        raise ValueError("Heatmap plot requires at least 2 parameters")

    param1, param2 = param_cols[0], param_cols[1]

    # Pivot data for heatmap
    pivot = metric_data.pivot_table(
        index=param2,
        columns=param1,
        values="value",
        aggfunc="mean"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn_r",  # Red (poor) to Green (good) for error metrics
        hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>{metric.upper()}: %{{z:.4f}}<extra></extra>",
        colorbar=dict(title=metric.upper())
    ))

    # Highlight best configurations
    if show_best > 0:
        best_configs = metric_data.nsmallest(show_best, "value")
        fig.add_trace(go.Scatter(
            x=best_configs[param1],
            y=best_configs[param2],
            mode="markers",
            marker=dict(size=15, color="white", symbol="star", line=dict(color="red", width=2)),
            name=f"Top {show_best}",
            hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>{metric.upper()}: Best<extra></extra>"
        ))

    # Layout
    layout_config = {
        "title": title or f"{metric.upper()} Heatmap",
        "xaxis_title": param1,
        "yaxis_title": param2,
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_parallel_tuning(
    metric_data: pd.DataFrame,
    param_cols: List[str],
    metric: str,
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_best: int
) -> go.Figure:
    """Create parallel coordinates plot for 3+ parameters."""
    if len(param_cols) == 0:
        raise ValueError("No tunable parameters found")

    # Prepare dimensions for parallel coordinates
    dimensions = []

    # Add parameter dimensions
    for param in param_cols:
        dimensions.append(dict(
            label=param,
            values=metric_data[param]
        ))

    # Add metric dimension
    dimensions.append(dict(
        label=metric.upper(),
        values=metric_data["value"]
    ))

    # Color by metric value (better = greener)
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=metric_data["value"],
            colorscale="RdYlGn_r",  # Red (poor) to Green (good)
            showscale=True,
            cmin=metric_data["value"].min(),
            cmax=metric_data["value"].max()
        ),
        dimensions=dimensions
    ))

    # Layout
    layout_config = {
        "title": title or f"Hyperparameter Tuning Results ({metric.upper()})",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_scatter_tuning(
    metric_data: pd.DataFrame,
    param_cols: List[str],
    metric: str,
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_best: int
) -> go.Figure:
    """Create scatter plot matrix for parameter exploration."""
    if len(param_cols) < 2:
        raise ValueError("Scatter plot requires at least 2 parameters")

    n_params = len(param_cols)

    # Create subplots
    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    # Create scatter plots for each pair
    for i, param_y in enumerate(param_cols, start=1):
        for j, param_x in enumerate(param_cols, start=1):
            if i == j:
                # Diagonal: histogram
                fig.add_trace(go.Histogram(
                    x=metric_data[param_x],
                    marker_color="rgba(31, 119, 180, 0.6)",
                    showlegend=False
                ), row=i, col=j)
            else:
                # Off-diagonal: scatter plot colored by metric
                fig.add_trace(go.Scatter(
                    x=metric_data[param_x],
                    y=metric_data[param_y],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=metric_data["value"],
                        colorscale="RdYlGn_r",
                        showscale=(i == 1 and j == n_params),  # Show colorbar once
                        colorbar=dict(title=metric.upper(), x=1.1) if (i == 1 and j == n_params) else None
                    ),
                    showlegend=False,
                    hovertemplate=f"{param_x}: %{{x}}<br>{param_y}: %{{y}}<br>{metric.upper()}: %{{marker.color:.4f}}<extra></extra>"
                ), row=i, col=j)

            # Update axes labels
            if i == n_params:
                fig.update_xaxes(title_text=param_x, row=i, col=j)
            if j == 1:
                fig.update_yaxes(title_text=param_y, row=i, col=j)

    # Layout
    layout_config = {
        "title": title or f"Hyperparameter Tuning Scatter Matrix ({metric.upper()})",
        "height": height if height >= 600 else 600,  # Ensure sufficient height
        "showlegend": False
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig
