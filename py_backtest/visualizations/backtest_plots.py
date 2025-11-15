"""
Visualization functions for backtesting results.

Provides matplotlib-based plotting for analyzing backtest performance including:
- Accuracy over time/vintages
- Forecast horizon degradation
- Vintage drift analysis
- Data revision impact
"""

from typing import Optional, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import warnings


def plot_accuracy_over_time(
    backtest_results,
    metric: str = "rmse",
    by_workflow: bool = True,
    workflows: Optional[List[str]] = None,
    show: bool = True,
    figsize: tuple = (12, 6),
    **kwargs
) -> Figure:
    """
    Plot metric performance over time/vintages.

    Shows how model accuracy evolves across different backtest vintages,
    helping identify temporal stability and regime changes.

    Args:
        backtest_results: BacktestResults object from fit_backtests()
        metric: Which metric to plot (rmse, mae, etc.)
        by_workflow: If True, separate lines per workflow; if False, aggregate
        workflows: List of workflow IDs to plot (default None = all workflows)
        show: Whether to display plot immediately
        figsize: Figure size tuple (width, height)
        **kwargs: Additional arguments passed to plt.plot()

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metric not found in results

    Example:
        >>> results = wf_set.fit_backtests(vintage_cv, metrics)
        >>> fig = plot_accuracy_over_time(results, metric="rmse", by_workflow=True)
        >>> fig.savefig("accuracy_over_time.png")
    """
    from py_backtest import BacktestResults

    if not isinstance(backtest_results, BacktestResults):
        raise TypeError("backtest_results must be a BacktestResults object")

    # Collect metrics
    metrics_df = backtest_results.collect_metrics(by_vintage=True, summarize=False)

    # Filter to requested metric
    metric_data = metrics_df[metrics_df["metric"] == metric].copy()

    if len(metric_data) == 0:
        available = metrics_df["metric"].unique()
        raise ValueError(
            f"Metric '{metric}' not found in results. "
            f"Available metrics: {', '.join(available)}"
        )

    # Filter to requested workflows
    if workflows is not None:
        metric_data = metric_data[metric_data["wflow_id"].isin(workflows)]
        if len(metric_data) == 0:
            raise ValueError(f"No data found for workflows: {workflows}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if by_workflow:
        # Plot separate line per workflow
        workflow_ids = metric_data["wflow_id"].unique()

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(workflow_ids)))

        for i, wflow_id in enumerate(workflow_ids):
            wf_data = metric_data[metric_data["wflow_id"] == wflow_id].sort_values("vintage_date")

            # Get default kwargs
            plot_kwargs = {
                "marker": "o",
                "linewidth": 2,
                "markersize": 6,
                "label": wflow_id,
                "color": colors[i],
            }
            plot_kwargs.update(kwargs)

            ax.plot(
                wf_data["vintage_date"],
                wf_data["value"],
                **plot_kwargs
            )
    else:
        # Aggregate across workflows
        agg_data = metric_data.groupby("vintage_date")["value"].agg(
            ["mean", "std"]
        ).reset_index()

        # Get default kwargs
        plot_kwargs = {
            "marker": "o",
            "linewidth": 2,
            "markersize": 6,
            "label": f"Mean {metric.upper()}",
            "color": "steelblue",
        }
        plot_kwargs.update(kwargs)

        # Plot mean
        ax.plot(
            agg_data["vintage_date"],
            agg_data["mean"],
            **plot_kwargs
        )

        # Add confidence band (±1 std)
        ax.fill_between(
            agg_data["vintage_date"],
            agg_data["mean"] - agg_data["std"],
            agg_data["mean"] + agg_data["std"],
            alpha=0.2,
            color=plot_kwargs.get("color", "steelblue"),
            label="±1 Std Dev"
        )

    # Format plot
    ax.set_xlabel("Vintage Date", fontsize=12)
    ax.set_ylabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Backtest Accuracy Over Time ({metric.upper()})", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_horizon_comparison(
    backtest_results,
    metric: str = "rmse",
    workflows: Optional[List[str]] = None,
    show: bool = True,
    figsize: tuple = (12, 6),
    **kwargs
) -> Figure:
    """
    Plot forecast horizon degradation.

    Shows how model accuracy changes with different forecast horizons,
    helping understand short-term vs long-term forecast quality.

    Args:
        backtest_results: BacktestResults object from fit_backtests()
        metric: Which metric to plot (rmse, mae, etc.)
        workflows: List of workflow IDs to plot (default None = all workflows)
        show: Whether to display plot immediately
        figsize: Figure size tuple (width, height)
        **kwargs: Additional arguments passed to plt.bar() or plt.plot()

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metric not found in results

    Example:
        >>> results = wf_set.fit_backtests(vintage_cv, metrics)
        >>> fig = plot_horizon_comparison(results, metric="rmse", workflows=["wf1", "wf2"])
        >>> fig.savefig("horizon_comparison.png")
    """
    from py_backtest import BacktestResults

    if not isinstance(backtest_results, BacktestResults):
        raise TypeError("backtest_results must be a BacktestResults object")

    # Analyze forecast horizon
    horizon_data = backtest_results.analyze_forecast_horizon(metric=metric)

    # Filter to requested workflows
    if workflows is not None:
        horizon_data = horizon_data[horizon_data["wflow_id"].isin(workflows)]
        if len(horizon_data) == 0:
            raise ValueError(f"No data found for workflows: {workflows}")

    # Convert horizon to numeric (days) for plotting
    horizon_data = horizon_data.copy()
    horizon_data["horizon_days"] = horizon_data["horizon"].apply(
        lambda x: x.total_seconds() / 86400 if hasattr(x, "total_seconds") else x
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    workflow_ids = horizon_data["wflow_id"].unique()
    horizons = sorted(horizon_data["horizon_days"].unique())

    if len(workflow_ids) == 1:
        # Single workflow - use bar chart
        wf_data = horizon_data[horizon_data["wflow_id"] == workflow_ids[0]]

        bar_kwargs = {
            "color": "steelblue",
            "alpha": 0.7,
            "edgecolor": "black",
        }
        bar_kwargs.update(kwargs)

        ax.bar(
            wf_data["horizon_days"],
            wf_data[metric],
            width=max(horizons) * 0.05,  # 5% of max horizon
            **bar_kwargs
        )

        ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title(
            f"Forecast Horizon Comparison - {workflow_ids[0]}",
            fontsize=14,
            fontweight="bold"
        )
    else:
        # Multiple workflows - use line plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(workflow_ids)))

        for i, wflow_id in enumerate(workflow_ids):
            wf_data = horizon_data[horizon_data["wflow_id"] == wflow_id].sort_values("horizon_days")

            plot_kwargs = {
                "marker": "o",
                "linewidth": 2,
                "markersize": 8,
                "label": wflow_id,
                "color": colors[i],
            }
            plot_kwargs.update(kwargs)

            ax.plot(
                wf_data["horizon_days"],
                wf_data[metric],
                **plot_kwargs
            )

        ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title(
            f"Forecast Horizon Comparison ({metric.upper()})",
            fontsize=14,
            fontweight="bold"
        )
        ax.legend(loc="best")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_vintage_drift(
    backtest_results,
    metric: str = "rmse",
    workflows: Optional[List[str]] = None,
    show: bool = True,
    figsize: tuple = (12, 6),
    **kwargs
) -> Figure:
    """
    Plot vintage drift analysis.

    Shows how model accuracy changes over vintages relative to the first vintage,
    helping identify model degradation and regime changes.

    Args:
        backtest_results: BacktestResults object from fit_backtests()
        metric: Which metric to plot (rmse, mae, etc.)
        workflows: List of workflow IDs to plot (default None = all workflows)
        show: Whether to display plot immediately
        figsize: Figure size tuple (width, height)
        **kwargs: Additional arguments passed to plt.plot()

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metric not found in results

    Example:
        >>> results = wf_set.fit_backtests(vintage_cv, metrics)
        >>> fig = plot_vintage_drift(results, metric="rmse")
        >>> fig.savefig("vintage_drift.png")
    """
    from py_backtest import BacktestResults

    if not isinstance(backtest_results, BacktestResults):
        raise TypeError("backtest_results must be a BacktestResults object")

    # Analyze vintage drift
    drift_data = backtest_results.analyze_vintage_drift(metric=metric)

    # Filter to requested workflows
    if workflows is not None:
        drift_data = drift_data[drift_data["wflow_id"].isin(workflows)]
        if len(drift_data) == 0:
            raise ValueError(f"No data found for workflows: {workflows}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    workflow_ids = drift_data["wflow_id"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(workflow_ids)))

    # Top plot: Absolute metric values
    for i, wflow_id in enumerate(workflow_ids):
        wf_data = drift_data[drift_data["wflow_id"] == wflow_id].sort_values("vintage_date")

        plot_kwargs = {
            "marker": "o",
            "linewidth": 2,
            "markersize": 6,
            "label": wflow_id,
            "color": colors[i],
        }
        plot_kwargs.update(kwargs)

        ax1.plot(
            wf_data["vintage_date"],
            wf_data["metric_value"],
            **plot_kwargs
        )

    ax1.set_ylabel(f"{metric.upper()}", fontsize=12)
    ax1.set_title("Vintage Drift Analysis - Absolute Values", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Drift from start (percentage)
    for i, wflow_id in enumerate(workflow_ids):
        wf_data = drift_data[drift_data["wflow_id"] == wflow_id].sort_values("vintage_date")

        plot_kwargs = {
            "marker": "s",
            "linewidth": 2,
            "markersize": 6,
            "label": wflow_id,
            "color": colors[i],
        }
        plot_kwargs.update(kwargs)

        ax2.plot(
            wf_data["vintage_date"],
            wf_data["drift_pct"],
            **plot_kwargs
        )

    # Add horizontal line at 0%
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Vintage Date", fontsize=12)
    ax2.set_ylabel("Drift from First Vintage (%)", fontsize=12)
    ax2.set_title("Relative Change from First Vintage", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_revision_impact(
    backtest_results,
    metric: str = "rmse",
    workflows: Optional[List[str]] = None,
    vintage_vs_final_data: Optional[pd.DataFrame] = None,
    show: bool = True,
    figsize: tuple = (10, 6),
    **kwargs
) -> Figure:
    """
    Plot data revision impact on predictions.

    Compares model performance using vintage (point-in-time) data vs final
    revised data, showing how much data revisions affect forecast accuracy.

    Args:
        backtest_results: BacktestResults object from fit_backtests()
        metric: Which metric to plot (rmse, mae, etc.)
        workflows: List of workflow IDs to plot (default None = all workflows)
        vintage_vs_final_data: Optional DataFrame with vintage vs final metrics
            If None, uses compare_vintage_vs_final() method (which currently
            returns placeholder data)
        show: Whether to display plot immediately
        figsize: Figure size tuple (width, height)
        **kwargs: Additional arguments passed to plt.scatter()

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metric not found in results

    Example:
        >>> results = wf_set.fit_backtests(vintage_cv, metrics)
        >>> fig = plot_revision_impact(results, metric="rmse")
        >>> fig.savefig("revision_impact.png")

    Note:
        This function currently uses placeholder data for final metrics.
        To use actual final data metrics, pass vintage_vs_final_data parameter
        with columns: wflow_id, vintage_{metric}, final_{metric}
    """
    from py_backtest import BacktestResults

    if not isinstance(backtest_results, BacktestResults):
        raise TypeError("backtest_results must be a BacktestResults object")

    # Get vintage vs final comparison
    if vintage_vs_final_data is None:
        comparison_data = backtest_results.compare_vintage_vs_final(metric=metric)
    else:
        comparison_data = vintage_vs_final_data.copy()

    # Filter to requested workflows
    if workflows is not None:
        comparison_data = comparison_data[comparison_data["wflow_id"].isin(workflows)]
        if len(comparison_data) == 0:
            raise ValueError(f"No data found for workflows: {workflows}")

    # Check if we have final data
    has_final_data = not comparison_data[f"final_{metric}"].isna().all()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if has_final_data:
        # Scatter plot comparing vintage vs final
        scatter_kwargs = {
            "alpha": 0.6,
            "s": 100,
            "edgecolors": "black",
            "linewidths": 1.5,
        }
        scatter_kwargs.update(kwargs)

        # Color by workflow if multiple workflows
        if len(comparison_data["wflow_id"].unique()) > 1:
            workflow_ids = comparison_data["wflow_id"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(workflow_ids)))
            color_map = dict(zip(workflow_ids, colors))

            for wflow_id in workflow_ids:
                wf_data = comparison_data[comparison_data["wflow_id"] == wflow_id]
                ax.scatter(
                    wf_data[f"vintage_{metric}"],
                    wf_data[f"final_{metric}"],
                    label=wflow_id,
                    color=color_map[wflow_id],
                    **scatter_kwargs
                )

            ax.legend(loc="best")
        else:
            ax.scatter(
                comparison_data[f"vintage_{metric}"],
                comparison_data[f"final_{metric}"],
                color="steelblue",
                **scatter_kwargs
            )

        # Add diagonal line (y=x)
        min_val = min(
            comparison_data[f"vintage_{metric}"].min(),
            comparison_data[f"final_{metric}"].min()
        )
        max_val = max(
            comparison_data[f"vintage_{metric}"].max(),
            comparison_data[f"final_{metric}"].max()
        )
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label="y=x")

        ax.set_xlabel(f"Vintage {metric.upper()}", fontsize=12)
        ax.set_ylabel(f"Final {metric.upper()}", fontsize=12)
        ax.set_title("Data Revision Impact on Forecast Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add text annotation about bias
        mean_diff = (
            comparison_data[f"final_{metric}"] - comparison_data[f"vintage_{metric}"]
        ).mean()
        ax.text(
            0.05, 0.95,
            f"Mean Revision Impact: {mean_diff:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )
    else:
        # No final data available - show warning
        ax.text(
            0.5, 0.5,
            "Final data metrics not available.\n\n"
            "To visualize revision impact:\n"
            "1. Compute metrics on final (fully revised) data\n"
            "2. Pass as vintage_vs_final_data parameter\n\n"
            "See compare_vintage_vs_final() docstring for format.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5)
        )
        ax.set_title("Data Revision Impact - No Final Data Available", fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    return fig
