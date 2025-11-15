"""
Results from backtesting workflows with data vintages.

Provides BacktestResults class for analyzing and comparing backtest performance
across vintages, including vintage drift and revision impact analysis.
"""

from typing import Union, Optional, List
import pandas as pd
import numpy as np
import warnings


class BacktestResults:
    """
    Results from backtesting workflows with data vintages.

    Similar to WorkflowSetResults but with vintage-specific analysis methods
    for evaluating forecast accuracy with point-in-time data.

    Attributes:
        results: Dictionary mapping workflow IDs to per-fold results
        workflow_ids: List of workflow IDs

    Methods:
        collect_metrics: Collect metrics across vintages
        rank_results: Rank workflows by performance
        extract_best_workflow: Get best workflow ID
        analyze_vintage_drift: Analyze accuracy degradation over time
        analyze_forecast_horizon: Analyze performance by forecast horizon
        compare_vintage_vs_final: Compare vintage vs final data performance
        collect_predictions: Collect predictions from all workflows

    Example:
        >>> results = wf_set.fit_backtests(vintage_cv, metrics)
        >>> top_models = results.rank_results("rmse", n=5)
        >>> drift = results.analyze_vintage_drift("rmse")
    """

    def __init__(self, results: dict):
        """
        Initialize BacktestResults.

        Args:
            results: Dictionary mapping workflow IDs to fold results
                Format: {wflow_id: {"folds": [...], "metrics": DataFrame}}
        """
        self.results = results
        self.workflow_ids = list(results.keys())

    def collect_metrics(
        self,
        by_vintage: bool = True,
        summarize: bool = False
    ) -> pd.DataFrame:
        """
        Collect metrics across vintages.

        Args:
            by_vintage: If True, return per-vintage metrics; if False, average
            summarize: If True, compute mean/std across vintages

        Returns:
            DataFrame with columns:
            - wflow_id: Workflow identifier
            - vintage_date: Vintage date (if by_vintage=True)
            - metric: Metric name (rmse, mae, etc.)
            - value: Metric value
            - n_obs: Number of test observations
            - forecast_horizon: Horizon for this fold (if available)

        Example:
            >>> metrics = results.collect_metrics(by_vintage=True)
            >>> print(metrics.head())
            wflow_id              vintage_date  metric  value   n_obs
            prep_1_linear_reg_1   2023-01-01   rmse    5.2     90
            prep_1_linear_reg_1   2023-01-01   mae     4.1     90
        """
        all_metrics = []

        for wflow_id, wf_results in self.results.items():
            folds = wf_results["folds"]

            for fold in folds:
                fold_metrics = fold["metrics"]
                vintage_info = fold["vintage_info"]

                # Add metadata
                fold_metrics = fold_metrics.copy()
                fold_metrics["wflow_id"] = wflow_id
                fold_metrics["vintage_date"] = vintage_info["vintage_date"]
                fold_metrics["n_obs"] = vintage_info["n_test_obs"]
                fold_metrics["forecast_horizon"] = vintage_info["forecast_horizon"]

                all_metrics.append(fold_metrics)

        # Combine all metrics
        metrics_df = pd.concat(all_metrics, ignore_index=True)

        if summarize:
            # Compute mean and std across vintages
            if by_vintage:
                # Group by workflow and vintage
                summary = metrics_df.groupby(["wflow_id", "vintage_date", "metric"])["value"].agg(
                    ["mean", "std", "count"]
                ).reset_index()
            else:
                # Group by workflow and metric only
                summary = metrics_df.groupby(["wflow_id", "metric"])["value"].agg(
                    ["mean", "std", "count"]
                ).reset_index()
                summary.rename(columns={"count": "n_vintages"}, inplace=True)

            return summary
        elif not by_vintage:
            # Average across vintages
            averaged = metrics_df.groupby(["wflow_id", "metric"])["value"].mean().reset_index()
            return averaged
        else:
            # Return per-vintage metrics
            return metrics_df

    def rank_results(
        self,
        metric: str,
        by_vintage: bool = False,
        n: int = 5
    ) -> pd.DataFrame:
        """
        Rank workflows by performance.

        Args:
            metric: Metric to rank by (e.g., "rmse", "mae")
            by_vintage: If True, rank separately per vintage
            n: Number of top workflows to return

        Returns:
            DataFrame with top N workflows ranked by metric

        Example:
            >>> top_models = results.rank_results("rmse", n=5)
            >>> print(top_models)
            rank  wflow_id              mean_rmse  std_rmse  n_vintages
            1     prep_3_rand_forest_2  4.8        0.3       12
            2     prep_2_linear_reg_2   5.1        0.4       12
        """
        # Collect metrics
        metrics_df = self.collect_metrics(by_vintage=by_vintage, summarize=False)

        # Filter to requested metric
        metric_data = metrics_df[metrics_df["metric"] == metric].copy()

        if len(metric_data) == 0:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {metrics_df['metric'].unique()}"
            )

        if by_vintage:
            # Rank separately per vintage
            ranked = []
            for vintage_date in metric_data["vintage_date"].unique():
                vintage_data = metric_data[metric_data["vintage_date"] == vintage_date]
                vintage_ranked = vintage_data.sort_values("value").head(n).copy()
                vintage_ranked["rank"] = range(1, len(vintage_ranked) + 1)
                vintage_ranked["vintage_date"] = vintage_date
                ranked.append(vintage_ranked[["rank", "wflow_id", "vintage_date", "value"]])

            result = pd.concat(ranked, ignore_index=True)
            result.rename(columns={"value": f"{metric}"}, inplace=True)
        else:
            # Rank by average across vintages
            summary = metric_data.groupby("wflow_id")["value"].agg(
                ["mean", "std", "count"]
            ).reset_index()
            summary.rename(columns={"count": "n_vintages"}, inplace=True)

            # Sort by mean value
            summary = summary.sort_values("mean").head(n).copy()
            summary["rank"] = range(1, len(summary) + 1)

            # Reorder columns
            result = summary[["rank", "wflow_id", "mean", "std", "n_vintages"]]
            result.rename(columns={
                "mean": f"mean_{metric}",
                "std": f"std_{metric}"
            }, inplace=True)

        return result

    def extract_best_workflow(
        self,
        metric: str,
        by_vintage: bool = False
    ) -> Union[str, pd.DataFrame]:
        """
        Extract best workflow ID.

        Args:
            metric: Metric to optimize
            by_vintage: If True, return best per vintage

        Returns:
            If by_vintage=False: str (single best workflow ID)
            If by_vintage=True: DataFrame with best workflow per vintage

        Example:
            >>> best_id = results.extract_best_workflow("rmse")
            >>> print(best_id)
            'prep_3_rand_forest_2'
        """
        ranked = self.rank_results(metric, by_vintage=by_vintage, n=1)

        if by_vintage:
            return ranked[["vintage_date", "wflow_id", metric]]
        else:
            return ranked.iloc[0]["wflow_id"]

    def analyze_vintage_drift(
        self,
        metric: str = "rmse"
    ) -> pd.DataFrame:
        """
        Analyze forecast accuracy degradation over time.

        Shows if models become less accurate as vintages progress
        (e.g., regime changes, non-stationarity).

        Args:
            metric: Metric to analyze (default "rmse")

        Returns:
            DataFrame with:
            - wflow_id: Workflow identifier
            - vintage_date: Vintage date
            - metric_value: Metric value for this vintage
            - drift_from_start: Change from first vintage
            - drift_pct: Percentage change from first vintage

        Example:
            >>> drift = results.analyze_vintage_drift("rmse")
            >>> print(drift[drift["wflow_id"] == "prep_1_linear_reg_1"])
            vintage_date  rmse  drift_from_start  drift_pct
            2023-01-01    4.5   0.0               0.0%
            2023-04-01    4.8   0.3               6.7%
            2023-07-01    5.2   0.7               15.6%
        """
        # Collect per-vintage metrics
        metrics_df = self.collect_metrics(by_vintage=True, summarize=False)

        # Filter to requested metric
        metric_data = metrics_df[metrics_df["metric"] == metric].copy()

        if len(metric_data) == 0:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {metrics_df['metric'].unique()}"
            )

        # Calculate drift per workflow
        drift_results = []

        for wflow_id in metric_data["wflow_id"].unique():
            wf_data = metric_data[metric_data["wflow_id"] == wflow_id].sort_values("vintage_date")

            # Get first vintage value as baseline
            first_value = wf_data.iloc[0]["value"]

            # Calculate drift
            wf_data = wf_data.copy()
            wf_data["metric_value"] = wf_data["value"]
            wf_data["drift_from_start"] = wf_data["value"] - first_value
            wf_data["drift_pct"] = ((wf_data["value"] - first_value) / first_value * 100) if first_value != 0 else 0

            drift_results.append(
                wf_data[["wflow_id", "vintage_date", "metric_value", "drift_from_start", "drift_pct"]]
            )

        result = pd.concat(drift_results, ignore_index=True)
        return result

    def analyze_forecast_horizon(
        self,
        metric: str = "rmse"
    ) -> pd.DataFrame:
        """
        Analyze performance by forecast horizon.

        Shows if accuracy degrades with longer forecast horizons.

        Args:
            metric: Metric to analyze (default "rmse")

        Returns:
            DataFrame with:
            - wflow_id: Workflow identifier
            - horizon: Forecast horizon (Timedelta)
            - metric_value: Average metric for this horizon
            - n_folds: Number of folds with this horizon

        Example:
            >>> horizon_perf = results.analyze_forecast_horizon("rmse")
            >>> print(horizon_perf)
            wflow_id              horizon         rmse   n_folds
            prep_1_linear_reg_1   7 days          4.2    12
            prep_1_linear_reg_1   14 days         4.8    12
        """
        # Collect metrics with horizon info
        metrics_df = self.collect_metrics(by_vintage=True, summarize=False)

        # Filter to requested metric
        metric_data = metrics_df[metrics_df["metric"] == metric].copy()

        if len(metric_data) == 0:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {metrics_df['metric'].unique()}"
            )

        # Group by workflow and horizon
        horizon_summary = metric_data.groupby(["wflow_id", "forecast_horizon"])["value"].agg(
            ["mean", "count"]
        ).reset_index()

        horizon_summary.rename(columns={
            "mean": f"{metric}",
            "count": "n_folds"
        }, inplace=True)

        horizon_summary.rename(columns={"forecast_horizon": "horizon"}, inplace=True)

        return horizon_summary

    def compare_vintage_vs_final(
        self,
        metric: str = "rmse"
    ) -> pd.DataFrame:
        """
        Compare backtest (vintage) vs final revised data performance.

        Shows impact of data revisions on model accuracy.
        "How much does data revision affect my forecasts?"

        Note: This requires access to final data performance, which would need
        to be computed separately. This implementation returns a placeholder.

        Args:
            metric: Metric to compare (default "rmse")

        Returns:
            DataFrame with:
            - wflow_id: Workflow identifier
            - vintage_metric: Performance using vintage data
            - final_metric: Performance using final revised data (placeholder)
            - revision_impact: Difference (final - vintage)
            - revision_impact_pct: Percentage difference

        Example:
            >>> comparison = results.compare_vintage_vs_final("rmse")
            >>> print(comparison)
            wflow_id              vintage_rmse  final_rmse  revision_impact
            prep_1_linear_reg_1   5.2           NaN         NaN
        """
        # Get vintage metrics (average across folds)
        metrics_df = self.collect_metrics(by_vintage=False, summarize=False)
        metric_data = metrics_df[metrics_df["metric"] == metric].copy()

        if len(metric_data) == 0:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {metrics_df['metric'].unique()}"
            )

        result = metric_data[["wflow_id", "value"]].copy()
        result.rename(columns={"value": f"vintage_{metric}"}, inplace=True)

        # Placeholder for final data metrics (would need separate computation)
        result[f"final_{metric}"] = np.nan
        result["revision_impact"] = np.nan
        result["revision_impact_pct"] = np.nan

        warnings.warn(
            "compare_vintage_vs_final() requires final data metrics to be computed separately. "
            "Currently returning vintage metrics only.",
            UserWarning
        )

        return result

    def collect_predictions(
        self,
        vintage_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Collect predictions from all workflows and vintages.

        Args:
            vintage_date: If provided, only return predictions for this vintage

        Returns:
            DataFrame with:
            - wflow_id: Workflow identifier
            - vintage_date: Vintage used for forecast
            - date: Observation date
            - actual: Actual outcome value
            - .pred: Predicted value
            - residual: actual - .pred

        Example:
            >>> preds = results.collect_predictions()
            >>> preds_jan = results.collect_predictions(vintage_date=pd.Timestamp("2023-01-01"))
        """
        all_predictions = []

        for wflow_id, wf_results in self.results.items():
            folds = wf_results["folds"]

            for fold in folds:
                if "predictions" not in fold:
                    continue

                preds = fold["predictions"].copy()
                vintage_info = fold["vintage_info"]

                # Add metadata
                preds["wflow_id"] = wflow_id
                preds["vintage_date"] = vintage_info["vintage_date"]

                # Filter by vintage if requested
                if vintage_date is not None:
                    if vintage_info["vintage_date"] != vintage_date:
                        continue

                all_predictions.append(preds)

        if len(all_predictions) == 0:
            warnings.warn("No predictions found in results.", UserWarning)
            return pd.DataFrame()

        result = pd.concat(all_predictions, ignore_index=True)
        return result

    def plot_accuracy_over_time(
        self,
        metric: str = "rmse",
        by_workflow: bool = True,
        workflows: Optional[List[str]] = None,
        show: bool = True,
        figsize: tuple = (12, 6),
        **kwargs
    ):
        """
        Plot metric performance over time/vintages.

        Delegates to plot_accuracy_over_time() function from py_backtest.visualizations.

        Args:
            metric: Which metric to plot (rmse, mae, etc.)
            by_workflow: If True, separate lines per workflow; if False, aggregate
            workflows: List of workflow IDs to plot (default None = all workflows)
            show: Whether to display plot immediately
            figsize: Figure size tuple (width, height)
            **kwargs: Additional arguments passed to plt.plot()

        Returns:
            matplotlib Figure object

        Example:
            >>> results = wf_set.fit_backtests(vintage_cv, metrics)
            >>> fig = results.plot_accuracy_over_time(metric="rmse", by_workflow=True)
            >>> fig.savefig("accuracy_over_time.png")
        """
        from py_backtest.visualizations.backtest_plots import plot_accuracy_over_time
        return plot_accuracy_over_time(
            self,
            metric=metric,
            by_workflow=by_workflow,
            workflows=workflows,
            show=show,
            figsize=figsize,
            **kwargs
        )

    def plot_horizon_comparison(
        self,
        metric: str = "rmse",
        workflows: Optional[List[str]] = None,
        show: bool = True,
        figsize: tuple = (12, 6),
        **kwargs
    ):
        """
        Plot forecast horizon degradation.

        Delegates to plot_horizon_comparison() function from py_backtest.visualizations.

        Args:
            metric: Which metric to plot (rmse, mae, etc.)
            workflows: List of workflow IDs to plot (default None = all workflows)
            show: Whether to display plot immediately
            figsize: Figure size tuple (width, height)
            **kwargs: Additional arguments passed to plt.bar() or plt.plot()

        Returns:
            matplotlib Figure object

        Example:
            >>> results = wf_set.fit_backtests(vintage_cv, metrics)
            >>> fig = results.plot_horizon_comparison(metric="rmse")
            >>> fig.savefig("horizon_comparison.png")
        """
        from py_backtest.visualizations.backtest_plots import plot_horizon_comparison
        return plot_horizon_comparison(
            self,
            metric=metric,
            workflows=workflows,
            show=show,
            figsize=figsize,
            **kwargs
        )

    def plot_vintage_drift(
        self,
        metric: str = "rmse",
        workflows: Optional[List[str]] = None,
        show: bool = True,
        figsize: tuple = (12, 6),
        **kwargs
    ):
        """
        Plot vintage drift analysis.

        Delegates to plot_vintage_drift() function from py_backtest.visualizations.

        Args:
            metric: Which metric to plot (rmse, mae, etc.)
            workflows: List of workflow IDs to plot (default None = all workflows)
            show: Whether to display plot immediately
            figsize: Figure size tuple (width, height)
            **kwargs: Additional arguments passed to plt.plot()

        Returns:
            matplotlib Figure object

        Example:
            >>> results = wf_set.fit_backtests(vintage_cv, metrics)
            >>> fig = results.plot_vintage_drift(metric="rmse")
            >>> fig.savefig("vintage_drift.png")
        """
        from py_backtest.visualizations.backtest_plots import plot_vintage_drift
        return plot_vintage_drift(
            self,
            metric=metric,
            workflows=workflows,
            show=show,
            figsize=figsize,
            **kwargs
        )

    def plot_revision_impact(
        self,
        metric: str = "rmse",
        workflows: Optional[List[str]] = None,
        vintage_vs_final_data: Optional[pd.DataFrame] = None,
        show: bool = True,
        figsize: tuple = (10, 6),
        **kwargs
    ):
        """
        Plot data revision impact on predictions.

        Delegates to plot_revision_impact() function from py_backtest.visualizations.

        Args:
            metric: Which metric to plot (rmse, mae, etc.)
            workflows: List of workflow IDs to plot (default None = all workflows)
            vintage_vs_final_data: Optional DataFrame with vintage vs final metrics
            show: Whether to display plot immediately
            figsize: Figure size tuple (width, height)
            **kwargs: Additional arguments passed to plt.scatter()

        Returns:
            matplotlib Figure object

        Example:
            >>> results = wf_set.fit_backtests(vintage_cv, metrics)
            >>> fig = results.plot_revision_impact(metric="rmse")
            >>> fig.savefig("revision_impact.png")
        """
        from py_backtest.visualizations.backtest_plots import plot_revision_impact
        return plot_revision_impact(
            self,
            metric=metric,
            workflows=workflows,
            vintage_vs_final_data=vintage_vs_final_data,
            show=show,
            figsize=figsize,
            **kwargs
        )

    def __repr__(self):
        """String representation"""
        n_workflows = len(self.workflow_ids)
        n_vintages = len(self.results[self.workflow_ids[0]]["folds"]) if n_workflows > 0 else 0
        return f"BacktestResults({n_workflows} workflows, {n_vintages} vintages)"
