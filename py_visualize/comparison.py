"""
Model comparison visualization with Plotly

Interactive comparisons of multiple models by metrics.
"""

from typing import List, Union, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_model_comparison(
    stats_list: List[pd.DataFrame],
    model_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    split: str = "test",
    plot_type: str = "bar",
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Compare multiple models by performance metrics.

    Creates interactive visualizations comparing model performance across
    different metrics.

    Parameters
    ----------
    stats_list : list of DataFrame
        List of stats DataFrames from extract_outputs(). Each DataFrame
        should contain metrics for one model.
    model_names : list of str, optional
        Names for each model. If None, uses "Model 1", "Model 2", etc.
    metrics : list of str, optional
        Metrics to compare. If None, uses all available metrics.
        Common metrics: ["rmse", "mae", "r_squared", "mape"]
    split : str, default="test"
        Which data split to compare ("train", "test", or "both")
    plot_type : str, default="bar"
        Type of comparison plot:
        - "bar": Grouped bar chart
        - "heatmap": Heatmap for many models/metrics
        - "radar": Radar chart for normalized metrics
    title : str, optional
        Plot title. If None, auto-generates title.
    height : int, default=500
        Plot height in pixels
    width : int, optional
        Plot width in pixels. If None, uses Plotly default.
    show_legend : bool, default=True
        Whether to show legend

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg, rand_forest
    >>> from py_visualize import plot_model_comparison
    >>>
    >>> # Fit multiple models
    >>> wf1 = workflow().add_formula("sales ~ .").add_model(linear_reg())
    >>> wf2 = workflow().add_formula("sales ~ .").add_model(rand_forest(trees=100).set_mode("regression"))
    >>>
    >>> fit1 = wf1.fit(train).evaluate(test)
    >>> fit2 = wf2.fit(train).evaluate(test)
    >>>
    >>> # Extract stats
    >>> _, _, stats1 = fit1.extract_outputs()
    >>> _, _, stats2 = fit2.extract_outputs()
    >>>
    >>> # Compare models
    >>> fig = plot_model_comparison(
    ...     [stats1, stats2],
    ...     model_names=["Linear Regression", "Random Forest"],
    ...     metrics=["rmse", "mae", "r_squared"]
    ... )
    >>> fig.show()
    """
    # Validate inputs
    if len(stats_list) == 0:
        raise ValueError("stats_list cannot be empty")

    # Generate model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(stats_list))]
    elif len(model_names) != len(stats_list):
        raise ValueError(f"Length of model_names ({len(model_names)}) must match length of stats_list ({len(stats_list)})")

    # Determine metrics to plot
    if metrics is None:
        # Use common metrics found in first stats DataFrame
        common_metrics = ["rmse", "mae", "mape", "smape", "r_squared", "adj_r_squared", "mda"]
        all_metrics = stats_list[0]["metric"].unique()
        metrics = [m for m in common_metrics if m in all_metrics]

        if len(metrics) == 0:
            # Fall back to all metrics except diagnostics and metadata
            exclude_metrics = ["durbin_watson", "shapiro_wilk_stat", "shapiro_wilk_p",
                             "ljung_box_stat", "ljung_box_p", "formula", "model_type",
                             "n_obs_train", "n_obs_test", "lags", "base_model"]
            metrics = [m for m in all_metrics if m not in exclude_metrics]

    # Collect comparison data
    comparison_df = _prepare_comparison_data(stats_list, model_names, metrics, split)

    # Route to appropriate plot function
    if plot_type == "bar":
        return _plot_bar_comparison(comparison_df, metrics, split, title, height, width, show_legend)
    elif plot_type == "heatmap":
        return _plot_heatmap_comparison(comparison_df, metrics, model_names, title, height, width)
    elif plot_type == "radar":
        return _plot_radar_comparison(comparison_df, metrics, model_names, title, height, width, show_legend)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be one of: 'bar', 'heatmap', 'radar'")


def _prepare_comparison_data(
    stats_list: List[pd.DataFrame],
    model_names: List[str],
    metrics: List[str],
    split: str
) -> pd.DataFrame:
    """Prepare comparison DataFrame from stats list."""
    comparison_rows = []

    for model_name, stats_df in zip(model_names, stats_list):
        for metric in metrics:
            # Filter for the metric and split
            if split == "both":
                metric_rows = stats_df[stats_df["metric"] == metric]
            else:
                metric_rows = stats_df[(stats_df["metric"] == metric) & (stats_df["split"] == split)]

            if len(metric_rows) > 0:
                comparison_rows.append({
                    "model": model_name,
                    "metric": metric,
                    "value": metric_rows.iloc[0]["value"],
                    "split": metric_rows.iloc[0]["split"] if "split" in metric_rows.columns else split
                })

    return pd.DataFrame(comparison_rows)


def _plot_bar_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    split: str,
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_legend: bool
) -> go.Figure:
    """Create grouped bar chart comparison."""
    fig = go.Figure()

    # Get unique models
    models = comparison_df["model"].unique()

    # Define colors
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create bar for each model
    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df["model"] == model]

        # Get metric values in order
        metric_values = []
        for metric in metrics:
            metric_row = model_data[model_data["metric"] == metric]
            if len(metric_row) > 0:
                metric_values.append(metric_row.iloc[0]["value"])
            else:
                metric_values.append(None)

        fig.add_trace(go.Bar(
            x=metrics,
            y=metric_values,
            name=model,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"{model}<br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))

    # Layout
    layout_config = {
        "title": title or f"Model Comparison ({split.capitalize()} Set)",
        "xaxis_title": "Metric",
        "yaxis_title": "Value",
        "barmode": "group",
        "height": height,
        "showlegend": show_legend,
        "hovermode": "x"
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_heatmap_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    model_names: List[str],
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create heatmap comparison."""
    # Pivot data for heatmap
    pivot_df = comparison_df.pivot(index="model", columns="metric", values="value")

    # Ensure metrics are in correct order
    pivot_df = pivot_df[metrics] if all(m in pivot_df.columns for m in metrics) else pivot_df

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn_r",  # Red (high error) to Green (low error)
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Value")
    ))

    # Layout
    layout_config = {
        "title": title or "Model Comparison Heatmap",
        "xaxis_title": "Metric",
        "yaxis_title": "Model",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_radar_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    model_names: List[str],
    title: Optional[str],
    height: int,
    width: Optional[int],
    show_legend: bool
) -> go.Figure:
    """Create radar chart comparison with normalized metrics."""
    fig = go.Figure()

    # Normalize metrics to 0-1 scale (higher is better)
    # For error metrics (rmse, mae, mape), invert so lower values appear better
    normalized_df = comparison_df.copy()

    for metric in metrics:
        metric_data = normalized_df[normalized_df["metric"] == metric]["value"]

        if len(metric_data) > 0:
            min_val = metric_data.min()
            max_val = metric_data.max()

            # Normalize to 0-1
            if max_val > min_val:
                normalized = (metric_data - min_val) / (max_val - min_val)

                # Invert for error metrics (so lower error = higher score)
                if metric in ["rmse", "mae", "mape", "smape", "mase"]:
                    normalized = 1 - normalized

                normalized_df.loc[normalized_df["metric"] == metric, "normalized_value"] = normalized
            else:
                normalized_df.loc[normalized_df["metric"] == metric, "normalized_value"] = 0.5

    # Define colors
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create radar trace for each model
    for i, model in enumerate(model_names):
        model_data = normalized_df[normalized_df["model"] == model]

        # Get normalized values in metric order
        r_values = []
        theta_values = metrics.copy()

        for metric in metrics:
            metric_row = model_data[model_data["metric"] == metric]
            if len(metric_row) > 0:
                r_values.append(metric_row.iloc[0]["normalized_value"])
            else:
                r_values.append(0)

        # Close the radar chart
        r_values.append(r_values[0])
        theta_values.append(theta_values[0])

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill="toself",
            name=model,
            line_color=colors[i % len(colors)],
            hovertemplate=f"{model}<br>%{{theta}}: %{{r:.2f}}<extra></extra>"
        ))

    # Layout
    layout_config = {
        "title": title or "Model Comparison Radar Chart (Normalized)",
        "polar": dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        "height": height,
        "showlegend": show_legend
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig
