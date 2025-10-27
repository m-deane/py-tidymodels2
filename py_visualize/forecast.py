"""
Forecast visualization with Plotly

Interactive time series plots with actuals, fitted values, and predictions.
"""

from typing import Union, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_forecast(
    fit,  # Union[WorkflowFit, NestedWorkflowFit]
    prediction_intervals: bool = True,
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Create interactive forecast plot with actual vs predicted values.

    Displays training data, fitted values, and forecasts (if available) with
    optional prediction intervals.

    Parameters
    ----------
    fit : WorkflowFit or NestedWorkflowFit
        Fitted workflow object. For nested fits, will plot all groups.
    prediction_intervals : bool, default=True
        If True, show prediction intervals as shaded regions (if available)
    title : str, optional
        Plot title. If None, auto-generates title based on model type.
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
    >>> from py_parsnip import linear_reg
    >>> from py_visualize import plot_forecast
    >>>
    >>> # Fit model
    >>> wf = workflow().add_formula("sales ~ date").add_model(linear_reg())
    >>> fit = wf.fit(train).evaluate(test)
    >>>
    >>> # Create forecast plot
    >>> fig = plot_forecast(fit)
    >>> fig.show()
    """
    # Extract outputs from fit
    outputs, _, _ = fit.extract_outputs()

    # Check if this is a nested fit (has group column)
    from py_workflows.workflow import NestedWorkflowFit
    is_nested = isinstance(fit, NestedWorkflowFit)

    if is_nested:
        return _plot_forecast_nested(
            outputs,
            group_col=fit.group_col,
            prediction_intervals=prediction_intervals,
            title=title,
            height=height,
            width=width,
            show_legend=show_legend
        )
    else:
        return _plot_forecast_single(
            outputs,
            prediction_intervals=prediction_intervals,
            title=title,
            height=height,
            width=width,
            show_legend=show_legend
        )


def _plot_forecast_single(
    outputs: pd.DataFrame,
    prediction_intervals: bool = True,
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Create forecast plot for a single model."""
    fig = go.Figure()

    # Filter by split
    train_data = outputs[outputs["split"] == "train"].copy()
    test_data = outputs[outputs["split"] == "test"].copy() if "test" in outputs["split"].values else pd.DataFrame()

    # Determine x-axis (date or index)
    x_col = "date" if "date" in outputs.columns else outputs.index

    # Training data (actuals)
    if len(train_data) > 0:
        x_train = train_data["date"] if "date" in train_data.columns else train_data.index
        fig.add_trace(go.Scatter(
            x=x_train,
            y=train_data["actuals"],
            name="Training Data",
            mode="lines",
            line=dict(color="#1f77b4", width=2)
        ))

        # Fitted values (training)
        fig.add_trace(go.Scatter(
            x=x_train,
            y=train_data["fitted"],
            name="Fitted Values",
            mode="lines",
            line=dict(color="#ff7f0e", width=2, dash="dot")
        ))

    # Test data (actuals and forecast)
    if len(test_data) > 0:
        x_test = test_data["date"] if "date" in test_data.columns else test_data.index

        # Test actuals
        fig.add_trace(go.Scatter(
            x=x_test,
            y=test_data["actuals"],
            name="Test Data",
            mode="lines",
            line=dict(color="#2ca02c", width=2)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=x_test,
            y=test_data["forecast"],
            name="Forecast",
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash")
        ))

        # Prediction intervals (if available)
        if prediction_intervals and ".pred_lower" in test_data.columns and ".pred_upper" in test_data.columns:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=x_test,
                y=test_data[".pred_upper"],
                fill=None,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Lower bound (fills to upper)
            fig.add_trace(go.Scatter(
                x=x_test,
                y=test_data[".pred_lower"],
                fill="tonexty",
                mode="lines",
                line=dict(width=0),
                name="95% Prediction Interval",
                fillcolor="rgba(214, 39, 40, 0.2)"
            ))

    # Layout
    layout_config = {
        "title": title or "Forecast Plot",
        "xaxis_title": "Date" if "date" in outputs.columns else "Time",
        "yaxis_title": "Value",
        "hovermode": "x unified",
        "height": height,
        "showlegend": show_legend
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    return fig


def _plot_forecast_nested(
    outputs: pd.DataFrame,
    group_col: str,
    prediction_intervals: bool = True,
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Create forecast plot for nested/grouped models."""
    # Get unique groups
    groups = outputs[group_col].unique()
    n_groups = len(groups)

    # Create subplots (one per group)
    fig = make_subplots(
        rows=n_groups,
        cols=1,
        subplot_titles=[f"{group_col}: {group}" for group in groups],
        vertical_spacing=0.05,
        shared_xaxes=True
    )

    # Plot each group
    for i, group in enumerate(groups, start=1):
        group_data = outputs[outputs[group_col] == group].copy()

        # Filter by split
        train_data = group_data[group_data["split"] == "train"]
        test_data = group_data[group_data["split"] == "test"] if "test" in group_data["split"].values else pd.DataFrame()

        # Determine x-axis
        x_train = train_data["date"] if "date" in train_data.columns else train_data.index
        x_test = test_data["date"] if "date" in test_data.columns and len(test_data) > 0 else (test_data.index if len(test_data) > 0 else [])

        # Training data
        if len(train_data) > 0:
            # Actuals
            fig.add_trace(go.Scatter(
                x=x_train,
                y=train_data["actuals"],
                name=f"Training ({group})",
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                showlegend=(i == 1)  # Only show legend for first group
            ), row=i, col=1)

            # Fitted
            fig.add_trace(go.Scatter(
                x=x_train,
                y=train_data["fitted"],
                name=f"Fitted ({group})",
                mode="lines",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                showlegend=(i == 1)
            ), row=i, col=1)

        # Test data
        if len(test_data) > 0:
            # Actuals
            fig.add_trace(go.Scatter(
                x=x_test,
                y=test_data["actuals"],
                name=f"Test ({group})",
                mode="lines",
                line=dict(color="#2ca02c", width=2),
                showlegend=(i == 1)
            ), row=i, col=1)

            # Forecast
            fig.add_trace(go.Scatter(
                x=x_test,
                y=test_data["forecast"],
                name=f"Forecast ({group})",
                mode="lines",
                line=dict(color="#d62728", width=2, dash="dash"),
                showlegend=(i == 1)
            ), row=i, col=1)

            # Prediction intervals
            if prediction_intervals and ".pred_lower" in test_data.columns and ".pred_upper" in test_data.columns:
                fig.add_trace(go.Scatter(
                    x=x_test,
                    y=test_data[".pred_upper"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ), row=i, col=1)

                fig.add_trace(go.Scatter(
                    x=x_test,
                    y=test_data[".pred_lower"],
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    name="95% PI" if i == 1 else "",
                    fillcolor="rgba(214, 39, 40, 0.2)",
                    showlegend=(i == 1)
                ), row=i, col=1)

    # Update layout
    layout_config = {
        "title": title or f"Forecast Plot by {group_col}",
        "hovermode": "x unified",
        "height": height * n_groups if n_groups > 1 else height,
        "showlegend": show_legend
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    # Update x-axis labels
    fig.update_xaxes(title_text="Date" if "date" in outputs.columns else "Time", row=n_groups, col=1)

    # Update y-axis labels
    for i in range(1, n_groups + 1):
        fig.update_yaxes(title_text="Value", row=i, col=1)

    return fig
