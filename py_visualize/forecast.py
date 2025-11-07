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


def plot_forecast_multi(
    outputs: Union[pd.DataFrame, list],
    model_names: Optional[list] = None,
    group: Optional[str] = None,
    include_residuals: bool = False,
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Create interactive forecast plot comparing multiple models.

    Plots multiple models on the same chart, differentiated by the 'model'
    column in the outputs DataFrame. Supports filtering by group and
    optional residuals subplot.

    Parameters
    ----------
    outputs : DataFrame or list of fits
        Either:
        - Combined outputs DataFrame with 'model' column (from multiple extract_outputs())
        - List of fitted models (WorkflowFit or NestedWorkflowFit objects)
    model_names : list of str, optional
        Custom names for models. If outputs is a DataFrame, maps from
        'model' column values. If outputs is a list, must match list length.
    group : str, optional
        Filter to specific group value (for grouped/nested models).
        If None, uses 'global' or first available group.
    include_residuals : bool, default=False
        If True, add subplot showing residuals for each model.
    title : str, optional
        Plot title. If None, auto-generates title.
    height : int, default=500
        Plot height in pixels (doubled if include_residuals=True)
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
    >>> from py_parsnip import linear_reg, prophet_reg
    >>> from py_visualize import plot_forecast_multi
    >>>
    >>> # Fit multiple models
    >>> wf1 = workflow().add_formula("sales ~ date").add_model(linear_reg().set_engine("statsmodels"))
    >>> wf2 = workflow().add_formula("sales ~ date").add_model(prophet_reg())
    >>>
    >>> fit1 = wf1.fit(train).evaluate(test)
    >>> fit2 = wf2.fit(train).evaluate(test)
    >>>
    >>> # Extract and combine outputs
    >>> outputs1, _, _ = fit1.extract_outputs()
    >>> outputs2, _, _ = fit2.extract_outputs()
    >>> combined = pd.concat([outputs1, outputs2], ignore_index=True)
    >>>
    >>> # Compare models on same plot
    >>> fig = plot_forecast_multi(combined, model_names={"linear_reg": "OLS", "prophet_reg": "Prophet"})
    >>> fig.show()
    >>>
    >>> # Or pass list of fits directly
    >>> fig = plot_forecast_multi([fit1, fit2], model_names=["OLS", "Prophet"])
    >>> fig.show()
    """
    # Convert list of fits to combined DataFrame if needed
    if isinstance(outputs, list):
        outputs_list = []
        for fit_obj in outputs:
            out, _, _ = fit_obj.extract_outputs()
            outputs_list.append(out)
        outputs = pd.concat(outputs_list, ignore_index=True)

    # Check for required columns
    if "model" not in outputs.columns:
        raise ValueError("outputs DataFrame must have 'model' column. Use extract_outputs() from fitted models.")

    # Filter by group if specified
    if group is not None:
        if "group" not in outputs.columns:
            raise ValueError("group parameter specified but outputs has no 'group' column")
        outputs = outputs[outputs["group"] == group].copy()
    elif "group" in outputs.columns:
        # Use first group (often 'global')
        available_groups = outputs["group"].unique()
        outputs = outputs[outputs["group"] == available_groups[0]].copy()

    # Get unique models
    models = outputs["model"].unique()

    # Map model names if provided
    if model_names is not None:
        if isinstance(model_names, dict):
            # Dict mapping model column values to display names
            name_map = model_names
        elif isinstance(model_names, list):
            # List of names matching order of unique models
            if len(model_names) != len(models):
                raise ValueError(f"Length of model_names ({len(model_names)}) must match number of models ({len(models)})")
            name_map = dict(zip(models, model_names))
        else:
            raise TypeError("model_names must be dict or list")
    else:
        # Use model column values as names
        name_map = {m: m for m in models}

    # Color palette for models
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create subplots if including residuals
    if include_residuals:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Forecast Comparison", "Residuals by Model"],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        residuals_row = 2
    else:
        fig = go.Figure()
        residuals_row = None

    # Determine x-axis (date or index)
    x_col = "date" if "date" in outputs.columns else None

    # Plot each model
    for i, model in enumerate(models):
        model_data = outputs[outputs["model"] == model].copy()
        model_name = name_map[model]
        color = colors[i % len(colors)]

        # Split into train and test
        train_data = model_data[model_data["split"] == "train"]
        test_data = model_data[model_data["split"] == "test"] if "test" in model_data["split"].values else pd.DataFrame()

        # Get x-axis values
        x_train = train_data[x_col] if x_col else train_data.index
        x_test = test_data[x_col] if x_col and len(test_data) > 0 else (test_data.index if len(test_data) > 0 else [])

        # Training actuals (only show for first model to avoid clutter)
        if i == 0 and len(train_data) > 0:
            trace = go.Scatter(
                x=x_train,
                y=train_data["actuals"],
                name="Training Data",
                mode="lines",
                line=dict(color="gray", width=1),
                opacity=0.5,
                showlegend=True
            )
            if include_residuals:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        # Fitted values (training)
        if len(train_data) > 0:
            trace = go.Scatter(
                x=x_train,
                y=train_data["fitted"],
                name=f"{model_name} (Train)",
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                showlegend=True
            )
            if include_residuals:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        # Test actuals (only show for first model)
        if i == 0 and len(test_data) > 0:
            trace = go.Scatter(
                x=x_test,
                y=test_data["actuals"],
                name="Test Data",
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=True
            )
            if include_residuals:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        # Forecast (test)
        if len(test_data) > 0:
            trace = go.Scatter(
                x=x_test,
                y=test_data["fitted"],
                name=f"{model_name} (Test)",
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=True
            )
            if include_residuals:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        # Residuals subplot (if requested)
        if include_residuals:
            # Plot residuals for both train and test
            if len(train_data) > 0:
                fig.add_trace(go.Scatter(
                    x=x_train,
                    y=train_data["residuals"],
                    name=f"{model_name} Residuals",
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.6),
                    showlegend=False
                ), row=residuals_row, col=1)

            if len(test_data) > 0:
                fig.add_trace(go.Scatter(
                    x=x_test,
                    y=test_data["residuals"],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.6),
                    showlegend=False
                ), row=residuals_row, col=1)

            # Add zero line for residuals
            if i == 0:
                all_x = pd.concat([
                    pd.Series(x_train) if len(train_data) > 0 else pd.Series([]),
                    pd.Series(x_test) if len(test_data) > 0 else pd.Series([])
                ])
                fig.add_trace(go.Scatter(
                    x=all_x,
                    y=[0] * len(all_x),
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    name="Zero Line",
                    showlegend=False
                ), row=residuals_row, col=1)

    # Layout
    layout_config = {
        "title": title or "Multi-Model Forecast Comparison",
        "hovermode": "x unified",
        "height": height * 2 if include_residuals else height,
        "showlegend": show_legend
    }

    if width is not None:
        layout_config["width"] = width

    fig.update_layout(**layout_config)

    # Axis labels
    if include_residuals:
        fig.update_xaxes(title_text="Date" if x_col else "Time", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Date" if x_col else "Time")
        fig.update_yaxes(title_text="Value")

    return fig
