"""
Residual diagnostic plots with Plotly

Interactive diagnostic visualizations for model validation.
"""

from typing import Union, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats


def plot_residuals(
    fit,  # Union[WorkflowFit, NestedWorkflowFit]
    plot_type: str = "all",
    title: Optional[str] = None,
    height: int = 600,
    width: Optional[int] = None
) -> go.Figure:
    """Create residual diagnostic plots for model validation.

    Provides comprehensive diagnostic plots including residuals vs fitted,
    Q-Q plot, residuals over time, and histogram.

    Parameters
    ----------
    fit : WorkflowFit or NestedWorkflowFit
        Fitted workflow object
    plot_type : str, default="all"
        Type of plot to create:
        - "all": 2x2 grid with all diagnostics
        - "fitted": Residuals vs fitted values
        - "qq": Q-Q plot for normality
        - "time": Residuals vs time
        - "hist": Histogram of residuals
    title : str, optional
        Plot title. If None, auto-generates title.
    height : int, default=600
        Plot height in pixels
    width : int, optional
        Plot width in pixels. If None, uses Plotly default.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg
    >>> from py_visualize import plot_residuals
    >>>
    >>> # Fit model
    >>> wf = workflow().add_formula("sales ~ date").add_model(linear_reg())
    >>> fit = wf.fit(train).evaluate(test)
    >>>
    >>> # Create diagnostic plots
    >>> fig = plot_residuals(fit)
    >>> fig.show()
    >>>
    >>> # Or just Q-Q plot
    >>> fig_qq = plot_residuals(fit, plot_type="qq")
    >>> fig_qq.show()
    """
    # Extract outputs from fit
    outputs, _, _ = fit.extract_outputs()

    # Use only training data for diagnostics
    train_data = outputs[outputs["split"] == "train"].copy()

    if len(train_data) == 0:
        raise ValueError("No training data found in fit outputs")

    # Get residuals and fitted values
    residuals = train_data["residuals"].values
    fitted = train_data["fitted"].values

    # Handle time variable (if present)
    if "date" in train_data.columns:
        time_var = train_data["date"].values
        time_label = "Date"
    else:
        time_var = np.arange(len(residuals))
        time_label = "Observation"

    # Route to appropriate plot function
    if plot_type == "all":
        return _plot_all_diagnostics(residuals, fitted, time_var, time_label, title, height, width)
    elif plot_type == "fitted":
        return _plot_residuals_vs_fitted(residuals, fitted, title, height, width)
    elif plot_type == "qq":
        return _plot_qq(residuals, title, height, width)
    elif plot_type == "time":
        return _plot_residuals_vs_time(residuals, time_var, time_label, title, height, width)
    elif plot_type == "hist":
        return _plot_histogram(residuals, title, height, width)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be one of: 'all', 'fitted', 'qq', 'time', 'hist'")


def _plot_all_diagnostics(
    residuals: np.ndarray,
    fitted: np.ndarray,
    time_var: np.ndarray,
    time_label: str,
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create 2x2 grid with all diagnostic plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals vs Fitted",
            "Q-Q Plot",
            f"Residuals vs {time_label}",
            "Histogram of Residuals"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # 1. Residuals vs Fitted
    fig.add_trace(go.Scatter(
        x=fitted,
        y=residuals,
        mode="markers",
        marker=dict(size=6, color="rgba(31, 119, 180, 0.6)"),
        showlegend=False,
        hovertemplate="Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
    ), row=1, col=1)

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Add LOWESS smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False,
            hoverinfo="skip"
        ), row=1, col=1)
    except ImportError:
        pass  # Skip smoothing if statsmodels not available

    # 2. Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)

    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode="markers",
        marker=dict(size=6, color="rgba(31, 119, 180, 0.6)"),
        showlegend=False,
        hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"
    ), row=1, col=2)

    # Add diagonal reference line
    qq_min = min(theoretical_quantiles.min(), sample_quantiles.min())
    qq_max = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[qq_min, qq_max],
        y=[qq_min, qq_max],
        mode="lines",
        line=dict(color="red", dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ), row=1, col=2)

    # 3. Residuals vs Time
    fig.add_trace(go.Scatter(
        x=time_var,
        y=residuals,
        mode="markers+lines",
        marker=dict(size=6, color="rgba(31, 119, 180, 0.6)"),
        line=dict(color="rgba(31, 119, 180, 0.3)", width=1),
        showlegend=False,
        hovertemplate=f"{time_label}: %{{x}}<br>Residual: %{{y:.2f}}<extra></extra>"
    ), row=2, col=1)

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    # 4. Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color="rgba(31, 119, 180, 0.6)",
        showlegend=False,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
    ), row=2, col=2)

    # Add normal distribution overlay
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
    # Scale to match histogram
    normal_curve_scaled = normal_curve * len(residuals) * (residuals.max() - residuals.min()) / 30

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve_scaled,
        mode="lines",
        line=dict(color="red", width=2),
        showlegend=False,
        hoverinfo="skip"
    ), row=2, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_xaxes(title_text=time_label, row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.update_xaxes(title_text="Residuals", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    # Update layout
    layout_config = {
        "title": title or "Residual Diagnostics",
        "height": height,
        "showlegend": False
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_residuals_vs_fitted(
    residuals: np.ndarray,
    fitted: np.ndarray,
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create residuals vs fitted values plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fitted,
        y=residuals,
        mode="markers",
        marker=dict(size=8, color="rgba(31, 119, 180, 0.6)"),
        hovertemplate="Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
    ))

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    # Add LOWESS smoothing
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode="lines",
            line=dict(color="red", width=2),
            name="LOWESS Smooth"
        ))
    except ImportError:
        pass

    layout_config = {
        "title": title or "Residuals vs Fitted Values",
        "xaxis_title": "Fitted Values",
        "yaxis_title": "Residuals",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_qq(
    residuals: np.ndarray,
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create Q-Q plot for normality check."""
    fig = go.Figure()

    # Calculate quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)

    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode="markers",
        marker=dict(size=8, color="rgba(31, 119, 180, 0.6)"),
        name="Sample Quantiles",
        hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"
    ))

    # Add diagonal reference line
    qq_min = min(theoretical_quantiles.min(), sample_quantiles.min())
    qq_max = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[qq_min, qq_max],
        y=[qq_min, qq_max],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Normal Line"
    ))

    layout_config = {
        "title": title or "Normal Q-Q Plot",
        "xaxis_title": "Theoretical Quantiles",
        "yaxis_title": "Sample Quantiles",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_residuals_vs_time(
    residuals: np.ndarray,
    time_var: np.ndarray,
    time_label: str,
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create residuals vs time plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_var,
        y=residuals,
        mode="markers+lines",
        marker=dict(size=8, color="rgba(31, 119, 180, 0.6)"),
        line=dict(color="rgba(31, 119, 180, 0.3)", width=1),
        hovertemplate=f"{time_label}: %{{x}}<br>Residual: %{{y:.2f}}<extra></extra>"
    ))

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    layout_config = {
        "title": title or f"Residuals vs {time_label}",
        "xaxis_title": time_label,
        "yaxis_title": "Residuals",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_histogram(
    residuals: np.ndarray,
    title: Optional[str],
    height: int,
    width: Optional[int]
) -> go.Figure:
    """Create histogram of residuals."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color="rgba(31, 119, 180, 0.6)",
        name="Residuals",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
    ))

    # Add normal distribution overlay
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
    # Scale to match histogram
    normal_curve_scaled = normal_curve * len(residuals) * (residuals.max() - residuals.min()) / 30

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve_scaled,
        mode="lines",
        line=dict(color="red", width=2),
        name="Normal Distribution"
    ))

    layout_config = {
        "title": title or "Histogram of Residuals",
        "xaxis_title": "Residuals",
        "yaxis_title": "Frequency",
        "height": height
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig
