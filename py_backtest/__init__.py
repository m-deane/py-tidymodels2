"""
Backtesting with data vintages for time series forecasting.

Provides vintage cross-validation and backtest analysis tools for evaluating
forecast accuracy using point-in-time data that simulates production conditions.

Main Components:
    - VintageCV: Create vintage-aware cross-validation splits
    - VintageSplit/VintageRSplit: Vintage-aware train/test splits
    - BacktestResults: Analyze backtest performance across vintages
    - Utility functions: select_vintage, create_vintage_data, validate_vintage_data

Example:
    >>> from py_backtest import VintageCV, create_vintage_data
    >>> from py_workflowsets import WorkflowSet
    >>> from py_yardstick import metric_set, rmse, mae
    >>>
    >>> # Create synthetic vintage data
    >>> vintage_df = create_vintage_data(
    ...     final_data=df,
    ...     date_col="date",
    ...     n_revisions=3,
    ...     revision_std=0.05
    ... )
    >>>
    >>> # Create vintage CV
    >>> vintage_cv = VintageCV(
    ...     data=vintage_df,
    ...     as_of_col="as_of_date",
    ...     date_col="date",
    ...     initial="2 years",
    ...     assess="3 months"
    ... )
    >>>
    >>> # Backtest workflows
    >>> results = wf_set.fit_backtests(
    ...     vintage_cv,
    ...     metrics=metric_set(rmse, mae)
    ... )
    >>>
    >>> # Analyze results
    >>> top_models = results.rank_results("rmse", n=5)
    >>> drift = results.analyze_vintage_drift("rmse")
"""

from py_backtest.vintage_utils import (
    select_vintage,
    create_vintage_data,
    validate_vintage_data,
)

from py_backtest.vintage_split import (
    VintageSplit,
    VintageRSplit,
)

from py_backtest.vintage_cv import (
    VintageCV,
    vintage_cv,
)

from py_backtest.backtest_results import (
    BacktestResults,
)

from py_backtest.visualizations import (
    plot_accuracy_over_time,
    plot_horizon_comparison,
    plot_vintage_drift,
    plot_revision_impact,
)

__all__ = [
    # Utility functions
    "select_vintage",
    "create_vintage_data",
    "validate_vintage_data",
    # Split classes
    "VintageSplit",
    "VintageRSplit",
    # CV classes
    "VintageCV",
    "vintage_cv",
    # Results
    "BacktestResults",
    # Visualizations
    "plot_accuracy_over_time",
    "plot_horizon_comparison",
    "plot_vintage_drift",
    "plot_revision_impact",
]
