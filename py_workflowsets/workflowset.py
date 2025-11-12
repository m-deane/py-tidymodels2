"""
WorkflowSets for multi-model comparison.

Provides tools for comparing multiple workflows across different
preprocessing strategies and model specifications.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class WorkflowSet:
    """
    Collection of workflows for multi-model comparison.

    A WorkflowSet allows you to compare multiple workflows, either by
    providing them directly or by creating all combinations of preprocessors
    and models (cross-product).

    Attributes:
        workflows: Dictionary mapping workflow IDs to Workflow objects
        info: Metadata about each workflow (preprocessor type, model type, etc.)
    """
    workflows: Dict[str, Any]
    info: pd.DataFrame

    @classmethod
    def from_workflows(cls, workflows: List[Any], ids: Optional[List[str]] = None):
        """
        Create WorkflowSet from a list of workflows.

        Args:
            workflows: List of Workflow objects OR list of (id, workflow) tuples
            ids: Optional list of IDs for workflows (auto-generated if None)
                 Ignored if workflows is a list of tuples

        Returns:
            WorkflowSet instance

        Examples:
            >>> wf1 = workflow().add_formula("y ~ x1").add_model(linear_reg())
            >>> wf2 = workflow().add_formula("y ~ x1 + x2").add_model(rand_forest())
            >>> # Method 1: Separate workflows and IDs
            >>> wf_set = WorkflowSet.from_workflows([wf1, wf2], ids=["linear", "rf"])
            >>> # Method 2: List of tuples
            >>> wf_set = WorkflowSet.from_workflows([("linear", wf1), ("rf", wf2)])
        """
        # Check if workflows is a list of tuples (id, workflow)
        if workflows and isinstance(workflows[0], tuple) and len(workflows[0]) == 2:
            # Extract IDs and workflows from tuples
            ids = [wf_id for wf_id, _ in workflows]
            workflows = [wf for _, wf in workflows]

        if ids is None:
            ids = [f"workflow_{i+1}" for i in range(len(workflows))]

        if len(ids) != len(workflows):
            raise ValueError("Length of ids must match length of workflows")

        workflow_dict = dict(zip(ids, workflows))

        # Create info DataFrame
        info_data = []
        for wf_id, wf in workflow_dict.items():
            preprocessor_type = "formula" if wf.preprocessor is not None else "none"
            model_type = wf.spec.model_type if wf.spec is not None else "none"

            info_data.append({
                "wflow_id": wf_id,
                "info": f"{preprocessor_type}_{model_type}",
                "option": wf_id,
                "preprocessor": preprocessor_type,
                "model": model_type
            })

        info_df = pd.DataFrame(info_data)

        return cls(workflows=workflow_dict, info=info_df)

    @classmethod
    def from_cross(cls,
                   preproc: List[Union[str, Any]],
                   models: List[Any],
                   ids: Optional[List[str]] = None):
        """
        Create WorkflowSet from cross-product of preprocessors and models.

        Creates all combinations of preprocessors (formulas or recipes) and models.

        Args:
            preproc: List of formulas (strings) or Recipe objects
            models: List of ModelSpec objects
            ids: Optional list of ID prefixes for each preprocessor

        Returns:
            WorkflowSet instance

        Examples:
            >>> formulas = ["y ~ x1", "y ~ x1 + x2"]
            >>> models = [linear_reg(), rand_forest()]
            >>> wf_set = WorkflowSet.from_cross(formulas, models)
            # Creates 4 workflows: 2 formulas × 2 models
        """
        from py_workflows import workflow

        workflows_dict = {}
        info_data = []

        if ids is None:
            ids = [f"prep_{i+1}" for i in range(len(preproc))]

        for i, prep in enumerate(preproc):
            prep_id = ids[i] if i < len(ids) else f"prep_{i+1}"

            for j, model in enumerate(models):
                model_type = model.model_type
                # Make ID unique by including model index
                wf_id = f"{prep_id}_{model_type}_{j+1}"

                # Create workflow
                wf = workflow()

                # Add preprocessor
                if isinstance(prep, str):
                    wf = wf.add_formula(prep)
                    prep_type = "formula"
                else:
                    # Assume it's a recipe
                    wf = wf.add_recipe(prep)
                    prep_type = "recipe"

                # Add model
                wf = wf.add_model(model)

                workflows_dict[wf_id] = wf

                info_data.append({
                    "wflow_id": wf_id,
                    "info": f"{prep_type}_{model_type}",
                    "option": prep_id,
                    "preprocessor": prep_type,
                    "model": model_type
                })

        info_df = pd.DataFrame(info_data)

        return cls(workflows=workflows_dict, info=info_df)

    def __len__(self):
        """Number of workflows in the set"""
        return len(self.workflows)

    def __iter__(self):
        """Iterate over workflow IDs"""
        return iter(self.workflows.keys())

    def __getitem__(self, key):
        """Get workflow by ID"""
        return self.workflows[key]

    def workflow_map(self,
                     fn: str,
                     resamples: Any = None,
                     metrics: Any = None,
                     grid: Any = None,
                     **kwargs) -> "WorkflowSetResults":
        """
        Apply a function to all workflows.

        Args:
            fn: Function name to apply ("fit_resamples" or "tune_grid")
            resamples: Resampling object (from py_rsample)
            metrics: Metric set (from py_yardstick)
            grid: Parameter grid (for tune_grid)
            **kwargs: Additional arguments passed to the function

        Returns:
            WorkflowSetResults object

        Examples:
            >>> results = wf_set.workflow_map(
            ...     "fit_resamples",
            ...     resamples=folds,
            ...     metrics=metric_set(rmse, mae)
            ... )
        """
        if fn == "fit_resamples":
            return self.fit_resamples(resamples=resamples, metrics=metrics, **kwargs)
        elif fn == "tune_grid":
            return self.tune_grid(resamples=resamples, grid=grid, metrics=metrics, **kwargs)
        else:
            raise ValueError(f"Unknown function: {fn}")

    def fit_resamples(self,
                      resamples: Any,
                      metrics: Any = None,
                      control: Optional[Dict[str, Any]] = None) -> "WorkflowSetResults":
        """
        Fit all workflows to all resamples and evaluate.

        Args:
            resamples: Resampling object (VFoldCV, TimeSeriesCV, etc.)
            metrics: Metric set for evaluation
            control: Control parameters (e.g., {'save_pred': True})

        Returns:
            WorkflowSetResults containing metrics and predictions

        Examples:
            >>> folds = vfold_cv(data, v=5)
            >>> metrics = metric_set(rmse, mae, r_squared)
            >>> results = wf_set.fit_resamples(folds, metrics)
            >>> results.rank_results("rmse")
        """
        from py_tune import fit_resamples as fit_resamples_fn

        control = control or {}
        all_results = []

        for wf_id, wf in self.workflows.items():
            print(f"Fitting {wf_id}...")

            # Fit resamples for this workflow
            tune_results = fit_resamples_fn(
                wf,
                resamples,
                metrics=metrics,
                control=control
            )

            # Collect metrics
            metrics_df = tune_results.collect_metrics()
            metrics_df["wflow_id"] = wf_id

            all_results.append({
                "wflow_id": wf_id,
                "tune_results": tune_results,
                "metrics": metrics_df
            })

        return WorkflowSetResults(
            results=all_results,
            workflow_set=self,
            metrics=metrics
        )

    def tune_grid(self,
                  resamples: Any,
                  grid: Any,
                  metrics: Any = None,
                  control: Optional[Dict[str, Any]] = None) -> "WorkflowSetResults":
        """
        Tune all workflows over parameter grids.

        Args:
            resamples: Resampling object
            grid: Parameter grid or dict of grids (one per workflow)
            metrics: Metric set for evaluation
            control: Control parameters

        Returns:
            WorkflowSetResults containing tuning results
        """
        from py_tune import tune_grid as tune_grid_fn

        control = control or {}
        all_results = []

        # Handle single grid vs per-workflow grids
        if isinstance(grid, dict):
            grids = grid
        else:
            # Use same grid for all workflows
            grids = {wf_id: grid for wf_id in self.workflows}

        for wf_id, wf in self.workflows.items():
            print(f"Tuning {wf_id}...")

            # Get grid for this workflow
            wf_grid = grids.get(wf_id, grid)

            # Tune grid for this workflow
            tune_results = tune_grid_fn(
                wf,
                resamples,
                grid=wf_grid,
                metrics=metrics,
                control=control
            )

            # Collect metrics
            metrics_df = tune_results.collect_metrics()
            metrics_df["wflow_id"] = wf_id

            all_results.append({
                "wflow_id": wf_id,
                "tune_results": tune_results,
                "metrics": metrics_df
            })

        return WorkflowSetResults(
            results=all_results,
            workflow_set=self,
            metrics=metrics
        )

    def fit_nested(self,
                   data: pd.DataFrame,
                   group_col: str,
                   per_group_prep: bool = False,
                   min_group_size: int = 30) -> "WorkflowSetNestedResults":
        """
        Fit all workflows across all groups independently (nested/panel modeling).

        Fits a separate model for each group within each workflow. This allows
        different groups to have different model parameters and optionally
        different preprocessing.

        Args:
            data: Training data with group column
            group_col: Column name identifying groups
            per_group_prep: If True, fit separate recipe for each group (default: False)
            min_group_size: Minimum samples for group-specific prep (default: 30)

        Returns:
            WorkflowSetNestedResults with group-aware metrics and outputs

        Examples:
            >>> # Create workflowset
            >>> wf_set = WorkflowSet.from_cross(
            ...     preproc=["y ~ x1", "y ~ x1 + x2"],
            ...     models=[linear_reg(), rand_forest()]
            ... )
            >>>
            >>> # Fit all workflows on all groups
            >>> results = wf_set.fit_nested(train_data, group_col='country')
            >>>
            >>> # Get metrics by workflow and group
            >>> metrics = results.collect_metrics(by_group=True)
            >>>
            >>> # Rank workflows overall (average across groups)
            >>> ranked = results.rank_results('rmse', by_group=False, n=5)
            >>>
            >>> # Get best workflow per group
            >>> best_by_group = results.extract_best_workflow('rmse', by_group=True)
        """
        all_results = []

        for wf_id, wf in self.workflows.items():
            print(f"Fitting {wf_id} across all groups...")

            try:
                # Fit nested workflow
                nested_fit = wf.fit_nested(
                    data,
                    group_col=group_col,
                    per_group_prep=per_group_prep,
                    min_group_size=min_group_size
                )

                # Extract outputs with group column
                outputs, coefs, stats = nested_fit.extract_outputs()

                # Store results
                all_results.append({
                    "wflow_id": wf_id,
                    "nested_fit": nested_fit,
                    "outputs": outputs,
                    "coefs": coefs,
                    "stats": stats
                })

            except Exception as e:
                print(f"  ⚠ Error fitting {wf_id}: {e}")
                # Store error result with NaN metrics
                all_results.append({
                    "wflow_id": wf_id,
                    "nested_fit": None,
                    "outputs": None,
                    "coefs": None,
                    "stats": None,
                    "error": str(e)
                })

        return WorkflowSetNestedResults(
            results=all_results,
            workflow_set=self,
            group_col=group_col
        )

    def fit_global(self,
                   data: pd.DataFrame,
                   group_col: str) -> "WorkflowSetResults":
        """
        Fit all workflows globally with group as a feature.

        Fits a single model per workflow using all groups, with the group
        column included as a predictor. This assumes groups share similar
        patterns and can be modeled together.

        Args:
            data: Training data with group column
            group_col: Column name identifying groups

        Returns:
            WorkflowSetResults containing global fits

        Examples:
            >>> # Create workflowset
            >>> wf_set = WorkflowSet.from_cross(
            ...     preproc=["y ~ x1", "y ~ x1 + x2"],
            ...     models=[linear_reg(), rand_forest()]
            ... )
            >>>
            >>> # Fit all workflows globally
            >>> results = wf_set.fit_global(train_data, group_col='country')
            >>>
            >>> # Collect metrics
            >>> metrics = results.collect_metrics()
        """
        all_results = []

        for wf_id, wf in self.workflows.items():
            print(f"Fitting {wf_id} globally with group feature...")

            try:
                # Fit global workflow
                global_fit = wf.fit_global(data, group_col=group_col)

                # Extract outputs
                outputs, coefs, stats = global_fit.extract_outputs()

                all_results.append({
                    "wflow_id": wf_id,
                    "fit": global_fit,
                    "metrics": stats,
                    "outputs": outputs,
                    "coefs": coefs
                })

            except Exception as e:
                print(f"  ⚠ Error fitting {wf_id}: {e}")
                all_results.append({
                    "wflow_id": wf_id,
                    "fit": None,
                    "metrics": None,
                    "outputs": None,
                    "coefs": None,
                    "error": str(e)
                })

        return WorkflowSetResults(
            results=all_results,
            workflow_set=self,
            metrics=None
        )


@dataclass
class WorkflowSetResults:
    """
    Results from fitting a WorkflowSet.

    Attributes:
        results: List of dictionaries containing results for each workflow
        workflow_set: The original WorkflowSet
        metrics: The metric set used for evaluation
    """
    results: List[Dict[str, Any]]
    workflow_set: WorkflowSet
    metrics: Any

    def collect_metrics(self, summarize: bool = True) -> pd.DataFrame:
        """
        Collect all metrics from all workflows.

        Args:
            summarize: If True, return mean and std across resamples.
                      If False, return raw metrics from each resample.

        Returns:
            DataFrame with metrics for all workflows

        Examples:
            >>> metrics_df = results.collect_metrics()
            >>> metrics_df.head()
        """
        all_metrics = []

        for result in self.results:
            wf_id = result["wflow_id"]
            metrics_df = result["metrics"].copy()

            # Add workflow ID if not already present
            if "wflow_id" not in metrics_df.columns:
                metrics_df["wflow_id"] = wf_id

            all_metrics.append(metrics_df)

        combined = pd.concat(all_metrics, ignore_index=True)

        if summarize:
            # Summarize by taking mean and std across resamples
            summary = combined.groupby(["wflow_id", "metric"])["value"].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("n", "count")
            ]).reset_index()

            # Add workflow info
            summary = summary.merge(
                self.workflow_set.info[["wflow_id", "preprocessor", "model"]],
                on="wflow_id",
                how="left"
            )

            return summary
        else:
            return combined

    def collect_predictions(self) -> pd.DataFrame:
        """
        Collect all predictions from all workflows.

        Returns:
            DataFrame with predictions for all workflows
        """
        all_preds = []

        for result in self.results:
            wf_id = result["wflow_id"]
            tune_results = result["tune_results"]

            preds_df = tune_results.collect_predictions()
            preds_df["wflow_id"] = wf_id

            all_preds.append(preds_df)

        return pd.concat(all_preds, ignore_index=True)

    def rank_results(self,
                     rank_metric: Optional[str] = None,
                     metric: Optional[str] = None,
                     select_best: bool = False,
                     n: int = 10) -> pd.DataFrame:
        """
        Rank workflows by a specific metric.

        Args:
            rank_metric: Metric name to rank by (e.g., "rmse") - deprecated, use 'metric' instead
            metric: Metric name to rank by (e.g., "rmse")
            select_best: If True, return only the best workflow per model type
            n: Number of top workflows to return (if select_best=False)

        Returns:
            DataFrame with ranked workflows (wide format with metric-specific columns)

        Examples:
            >>> # Get top 5 workflows by RMSE
            >>> top5 = results.rank_results(metric="rmse", n=5)
            >>>
            >>> # Get best workflow for each model type
            >>> best = results.rank_results(metric="rmse", select_best=True)
        """
        # Handle both 'metric' and 'rank_metric' parameter names
        if metric is not None:
            rank_metric = metric
        elif rank_metric is None:
            raise ValueError("Must provide either 'metric' or 'rank_metric' parameter")

        # Get summarized metrics
        metrics_df = self.collect_metrics(summarize=True)

        # Pivot to wide format: one column per metric (rmse_mean, mae_mean, etc.)
        wide_metrics = metrics_df.pivot_table(
            index=["wflow_id", "preprocessor", "model"],
            columns="metric",
            values=["mean", "std", "n"]
        )

        # Flatten multi-level columns: (mean, rmse) -> rmse_mean
        wide_metrics.columns = [f"{metric}_{stat}" for stat, metric in wide_metrics.columns]
        wide_metrics = wide_metrics.reset_index()

        # Determine if we want to minimize or maximize
        minimize_metrics = {"rmse", "mae", "mape", "smape", "mse", "log_loss", "brier_score"}
        ascending = rank_metric.lower() in minimize_metrics

        # Sort by the specified metric mean
        sort_col = f"{rank_metric}_mean"
        if sort_col not in wide_metrics.columns:
            raise ValueError(f"Metric '{rank_metric}' not found in results")

        wide_metrics = wide_metrics.sort_values(sort_col, ascending=ascending)

        if select_best:
            # Get best workflow for each model type
            wide_metrics = wide_metrics.groupby("model").first().reset_index()
            wide_metrics = wide_metrics.sort_values(sort_col, ascending=ascending)
        else:
            # Get top n
            wide_metrics = wide_metrics.head(n)

        # Add rank column
        wide_metrics.insert(0, "rank", range(1, len(wide_metrics) + 1))

        return wide_metrics

    def autoplot(self,
                 metric: Optional[str] = None,
                 select_best: bool = False,
                 top_n: int = 15) -> plt.Figure:
        """
        Plot workflow comparison results.

        Args:
            metric: Metric to plot (if None, plots first metric)
            select_best: If True, show only best workflow per model
            top_n: Number of top workflows to show

        Returns:
            matplotlib Figure object

        Examples:
            >>> fig = results.autoplot("rmse", top_n=10)
            >>> plt.show()
        """
        # Get summarized metrics
        metrics_df = self.collect_metrics(summarize=True)

        # Determine metric to plot
        if metric is None:
            metric = metrics_df["metric"].iloc[0]

        # Filter for the specified metric
        metric_data = metrics_df[metrics_df["metric"] == metric].copy()

        # Determine ordering
        minimize_metrics = {"rmse", "mae", "mape", "smape", "mse", "log_loss", "brier_score"}
        ascending = metric.lower() in minimize_metrics

        metric_data = metric_data.sort_values("mean", ascending=ascending)

        if select_best:
            metric_data = metric_data.groupby("model").first().reset_index()
            metric_data = metric_data.sort_values("mean", ascending=ascending)
        else:
            metric_data = metric_data.head(top_n)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(metric_data) * 0.4)))

        # Create bar plot with error bars
        x = np.arange(len(metric_data))
        bars = ax.barh(x, metric_data["mean"], xerr=metric_data["std"],
                       capsize=5, alpha=0.7)

        # Color by model type
        model_types = metric_data["model"].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_types)))
        color_map = dict(zip(model_types, colors))

        for i, (idx, row) in enumerate(metric_data.iterrows()):
            bars[i].set_color(color_map[row["model"]])

        # Set labels
        ax.set_yticks(x)
        ax.set_yticklabels(metric_data["wflow_id"])
        ax.set_xlabel(f"{metric.upper()} (mean ± std)", fontsize=12)
        ax.set_title(f"Workflow Comparison: {metric.upper()}", fontsize=14)
        ax.invert_yaxis()  # Best at top
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[model])
                          for model in model_types]
        ax.legend(legend_elements, model_types, title="Model Type",
                 loc="best", frameon=True)

        plt.tight_layout()
        return fig


@dataclass
class WorkflowSetNestedResults:
    """
    Results from fitting WorkflowSet with fit_nested().

    Provides group-aware metrics and results analysis across all workflows
    and all groups.

    Attributes:
        results: List of nested fit results per workflow
        workflow_set: The original WorkflowSet
        group_col: Grouping column name
    """
    results: List[Dict[str, Any]]
    workflow_set: WorkflowSet
    group_col: str

    def collect_metrics(self, by_group: bool = True, split: str = "all") -> pd.DataFrame:
        """
        Collect metrics across all workflows and groups.

        Args:
            by_group: If True, return metrics per workflow per group.
                     If False, return metrics per workflow (averaged across groups).
            split: Data split to include - "train", "test", or "all" (default: "all")

        Returns:
            DataFrame with metrics

            If by_group=True:
                Columns: wflow_id, group, metric, value, split, preprocessor, model

            If by_group=False:
                Columns: wflow_id, metric, mean, std, n, split, preprocessor, model

        Examples:
            >>> # Get metrics per workflow per group
            >>> metrics = results.collect_metrics(by_group=True)
            >>>
            >>> # Get average metrics per workflow (across groups)
            >>> avg_metrics = results.collect_metrics(by_group=False)
            >>>
            >>> # Get only test metrics
            >>> test_metrics = results.collect_metrics(by_group=True, split='test')
        """
        all_stats = []

        for result in self.results:
            wf_id = result["wflow_id"]
            stats = result.get("stats")

            if stats is None:
                continue  # Skip failed workflows

            stats = stats.copy()
            stats["wflow_id"] = wf_id
            all_stats.append(stats)

        if not all_stats:
            raise ValueError("No valid results to collect metrics from")

        combined = pd.concat(all_stats, ignore_index=True)

        # Filter by split
        if split != "all":
            combined = combined[combined["split"] == split]

        # Filter to only numeric metrics columns before aggregation
        # Select only the columns we need to avoid aggregating metadata columns
        required_cols = ["wflow_id", "group", "metric", "value", "split"]
        available_cols = [col for col in required_cols if col in combined.columns]
        combined_filtered = combined[available_cols].copy()

        if by_group:
            # Return per-group metrics with workflow info
            result_df = combined_filtered[["wflow_id", "group", "metric", "value", "split"]].copy()

            # Add workflow info
            result_df = result_df.merge(
                self.workflow_set.info[["wflow_id", "preprocessor", "model"]],
                on="wflow_id",
                how="left"
            )

            return result_df
        else:
            # Average across groups - ensure value column is numeric
            combined_filtered["value"] = pd.to_numeric(combined_filtered["value"], errors='coerce')

            # Drop any rows where value couldn't be converted
            combined_filtered = combined_filtered.dropna(subset=["value"])

            summary = combined_filtered.groupby(["wflow_id", "metric", "split"])["value"].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("n", "count")
            ]).reset_index()

            # Add workflow info
            summary = summary.merge(
                self.workflow_set.info[["wflow_id", "preprocessor", "model"]],
                on="wflow_id",
                how="left"
            )

            return summary

    def rank_results(self,
                    metric: str,
                    split: str = "test",
                    by_group: bool = False,
                    n: int = 10) -> pd.DataFrame:
        """
        Rank workflows by a specific metric.

        Args:
            metric: Metric to rank by (e.g., 'rmse', 'mae', 'r_squared')
            split: Data split to use - 'train', 'test', or 'all' (default: 'test')
            by_group: If True, rank within each group separately.
                     If False, rank by average across groups.
            n: Number of top workflows to return per group (default: 10)

        Returns:
            DataFrame with ranked workflows

            If by_group=False:
                Columns: rank, wflow_id, mean, std, n, preprocessor, model

            If by_group=True:
                Columns: group, rank, wflow_id, value, preprocessor, model

        Examples:
            >>> # Rank workflows overall (average across groups)
            >>> ranked = results.rank_results('rmse', by_group=False, n=5)
            >>>
            >>> # Rank workflows within each group
            >>> ranked_by_group = results.rank_results('rmse', by_group=True, n=3)
        """
        metrics_df = self.collect_metrics(by_group=True, split=split)
        metrics_df = metrics_df[metrics_df["metric"] == metric]

        if metrics_df.empty:
            raise ValueError(f"No data found for metric '{metric}' and split '{split}'")

        # Determine if we want to minimize or maximize
        minimize_metrics = {"rmse", "mae", "mape", "smape", "mse", "log_loss", "brier_score"}
        ascending = metric.lower() in minimize_metrics

        if by_group:
            # Rank within each group
            ranked = []
            for group in metrics_df["group"].unique():
                group_data = metrics_df[metrics_df["group"] == group].copy()
                group_data = group_data.sort_values("value", ascending=ascending)
                group_data["rank"] = range(1, len(group_data) + 1)
                ranked.append(group_data.head(n))

            result = pd.concat(ranked, ignore_index=True)
            return result[["group", "rank", "wflow_id", "value", "preprocessor", "model"]]
        else:
            # Rank by average across groups
            avg = metrics_df.groupby(["wflow_id", "preprocessor", "model"])["value"].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("n", "count")
            ]).reset_index()

            avg = avg.sort_values("mean", ascending=ascending)
            avg["rank"] = range(1, len(avg) + 1)

            result = avg.head(n)
            return result[["rank", "wflow_id", "mean", "std", "n", "preprocessor", "model"]]

    def extract_best_workflow(self,
                             metric: str,
                             split: str = "test",
                             by_group: bool = False) -> Union[str, pd.DataFrame]:
        """
        Extract best performing workflow(s).

        Args:
            metric: Metric to optimize (e.g., 'rmse', 'mae', 'r_squared')
            split: Data split to use - 'train', 'test', or 'all' (default: 'test')
            by_group: If True, return best workflow per group.
                     If False, return single best workflow overall.

        Returns:
            If by_group=False: workflow_id (str) - single best workflow overall

            If by_group=True: DataFrame with columns:
                group, wflow_id, value, preprocessor, model

        Examples:
            >>> # Get single best workflow overall
            >>> best_wf_id = results.extract_best_workflow('rmse')
            >>> print(best_wf_id)  # 'formula_1_rf_2'
            >>>
            >>> # Get best workflow per group
            >>> best_by_group = results.extract_best_workflow('rmse', by_group=True)
            >>> print(best_by_group)
            >>>   group      wflow_id         value  preprocessor  model
            >>>   Germany    formula_2_rf_2   1.20   formula       rand_forest
            >>>   France     formula_3_xgb_3  1.55   formula       boost_tree
        """
        ranked = self.rank_results(metric, split, by_group=by_group, n=1)

        if by_group:
            return ranked[["group", "wflow_id", "value", "preprocessor", "model"]]
        else:
            return ranked.iloc[0]["wflow_id"]

    def collect_outputs(self) -> pd.DataFrame:
        """
        Collect all outputs (actuals, fitted, forecast, residuals) from all workflows.

        Returns:
            DataFrame with outputs from all workflows and groups
            Columns include: wflow_id, group, actuals, fitted, forecast, residuals, split

        Examples:
            >>> outputs = results.collect_outputs()
            >>> # Filter to specific workflow and group
            >>> wf_group = outputs[
            ...     (outputs['wflow_id'] == 'formula_1_rf_2') &
            ...     (outputs['group'] == 'Germany')
            ... ]
        """
        all_outputs = []

        for result in self.results:
            wf_id = result["wflow_id"]
            outputs = result.get("outputs")

            if outputs is None:
                continue  # Skip failed workflows

            outputs = outputs.copy()
            outputs["wflow_id"] = wf_id
            all_outputs.append(outputs)

        if not all_outputs:
            raise ValueError("No valid results to collect outputs from")

        return pd.concat(all_outputs, ignore_index=True)

    def extract_outputs(self) -> tuple:
        """
        Extract all outputs, coefficients, and stats from all workflows.

        Returns the standard three-DataFrame pattern used throughout py-tidymodels.

        Returns:
            Tuple of (outputs, coefficients, stats) DataFrames:
            - outputs: Observation-level data (actuals, fitted, forecast, residuals)
            - coefficients: Model parameters/importances for all workflows and groups
            - stats: Aggregated metrics for all workflows and groups

        Examples:
            >>> # Get all three DataFrames
            >>> outputs, coefs, stats = results.extract_outputs()
            >>>
            >>> # Filter outputs to specific workflow
            >>> wf_outputs = outputs[outputs['wflow_id'] == 'formula_1_rf_2']
            >>>
            >>> # Filter coefficients to specific group
            >>> group_coefs = coefs[coefs['group'] == 'Germany']
            >>>
            >>> # Get test statistics
            >>> test_stats = stats[stats['split'] == 'test']
        """
        # Collect outputs
        all_outputs = []
        for result in self.results:
            wf_id = result["wflow_id"]
            outputs = result.get("outputs")

            if outputs is None:
                continue

            outputs = outputs.copy()
            outputs["wflow_id"] = wf_id

            # FIX: Ensure generic 'group' column matches specific group column
            # This handles cases where 'group' is 'global' but self.group_col has actual values
            if self.group_col in outputs.columns and 'group' in outputs.columns:
                outputs['group'] = outputs[self.group_col]

            all_outputs.append(outputs)

        if not all_outputs:
            outputs_df = pd.DataFrame()
        else:
            outputs_df = pd.concat(all_outputs, ignore_index=True)

        # Collect coefficients
        all_coefs = []
        for result in self.results:
            wf_id = result["wflow_id"]
            coefs = result.get("coefs")

            if coefs is None:
                continue

            coefs = coefs.copy()
            coefs["wflow_id"] = wf_id

            # FIX: Ensure generic 'group' column matches specific group column
            # This handles cases where 'group' is 'global' but self.group_col has actual values
            if self.group_col in coefs.columns and 'group' in coefs.columns:
                coefs['group'] = coefs[self.group_col]

            all_coefs.append(coefs)

        if not all_coefs:
            coefs_df = pd.DataFrame()
        else:
            coefs_df = pd.concat(all_coefs, ignore_index=True)

        # Collect stats (use existing collect_metrics with all splits)
        try:
            stats_df = self.collect_metrics(by_group=True, split='all')
        except ValueError:
            # No valid results
            stats_df = pd.DataFrame()

        return (outputs_df, coefs_df, stats_df)

    def extract_formulas(self) -> dict:
        """
        Extract formulas from all workflows.

        Returns a dictionary mapping workflow ID to formula string.
        All workflows in a WorkflowSet use the same formula across groups,
        so this extracts one formula per workflow.

        Returns:
            Dictionary with workflow IDs as keys and formula strings as values

        Examples:
            >>> # Get formulas from all workflows
            >>> formulas = results.extract_formulas()
            >>> for wf_id, formula in formulas.items():
            ...     print(f"{wf_id}: {formula}")
            prep_1_linear_reg_1: y ~ x1 + x2
            prep_2_rand_forest_2: y ~ x1 + x2 + x3
        """
        formulas = {}

        for result in self.results:
            wf_id = result["wflow_id"]
            nested_fit = result.get("nested_fit")

            if nested_fit is None:
                continue  # Skip failed workflows

            # Get first group's fit to extract formula
            # (same formula across all groups for a given workflow)
            if nested_fit.group_fits:
                first_group = list(nested_fit.group_fits.keys())[0]
                wf_fit = nested_fit.group_fits[first_group]

                try:
                    formula = wf_fit.extract_formula()
                    formulas[wf_id] = formula
                except Exception:
                    # If extract_formula() fails, skip this workflow
                    continue

        return formulas

    def autoplot(self,
                metric: str,
                split: str = "test",
                by_group: bool = False,
                top_n: int = 10) -> plt.Figure:
        """
        Plot workflow comparison results.

        Args:
            metric: Metric to plot (e.g., 'rmse', 'mae')
            split: Data split to plot - 'train', 'test', or 'all'
            by_group: If True, create separate plot per group.
                     If False, plot average across groups.
            top_n: Number of top workflows to show

        Returns:
            matplotlib Figure object

        Examples:
            >>> # Plot average performance across groups
            >>> fig = results.autoplot('rmse', by_group=False, top_n=10)
            >>> plt.show()
            >>>
            >>> # Plot performance per group
            >>> fig = results.autoplot('rmse', by_group=True, top_n=5)
            >>> plt.show()
        """
        if by_group:
            # Create subplot for each group
            metrics_df = self.collect_metrics(by_group=True, split=split)
            metrics_df = metrics_df[metrics_df["metric"] == metric]

            groups = sorted(metrics_df["group"].unique())
            n_groups = len(groups)

            # Calculate subplot layout
            n_cols = min(3, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            if n_groups == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 else list(axes)

            # Determine if we want to minimize or maximize
            minimize_metrics = {"rmse", "mae", "mape", "smape", "mse", "log_loss", "brier_score"}
            ascending = metric.lower() in minimize_metrics

            for i, group in enumerate(groups):
                ax = axes[i]
                group_data = metrics_df[metrics_df["group"] == group].copy()
                group_data = group_data.sort_values("value", ascending=ascending).head(top_n)

                # Create bar plot
                x = np.arange(len(group_data))
                bars = ax.barh(x, group_data["value"], alpha=0.7)

                # Color by model type
                model_types = group_data["model"].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(model_types)))
                color_map = dict(zip(model_types, colors))

                for j, (idx, row) in enumerate(group_data.iterrows()):
                    bars[j].set_color(color_map[row["model"]])

                # Set labels
                ax.set_yticks(x)
                ax.set_yticklabels(group_data["wflow_id"], fontsize=8)
                ax.set_xlabel(metric.upper(), fontsize=10)
                ax.set_title(f"{group}", fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)

            # Remove empty subplots
            for i in range(n_groups, len(axes)):
                fig.delaxes(axes[i])

            # Add legend to first subplot
            if n_groups > 0:
                legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[model])
                                  for model in model_types]
                axes[0].legend(legend_elements, model_types, title="Model",
                             loc="best", frameon=True, fontsize=8)

            plt.suptitle(f"Workflow Comparison by Group: {metric.upper()} ({split})",
                        fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()

        else:
            # Plot average across groups
            metrics_df = self.collect_metrics(by_group=False, split=split)
            metrics_df = metrics_df[metrics_df["metric"] == metric]

            # Determine if we want to minimize or maximize
            minimize_metrics = {"rmse", "mae", "mape", "smape", "mse", "log_loss", "brier_score"}
            ascending = metric.lower() in minimize_metrics

            metrics_df = metrics_df.sort_values("mean", ascending=ascending).head(top_n)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(metrics_df) * 0.4)))

            # Create bar plot with error bars
            x = np.arange(len(metrics_df))
            bars = ax.barh(x, metrics_df["mean"], xerr=metrics_df["std"],
                          capsize=5, alpha=0.7)

            # Color by model type
            model_types = metrics_df["model"].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_types)))
            color_map = dict(zip(model_types, colors))

            for i, (idx, row) in enumerate(metrics_df.iterrows()):
                bars[i].set_color(color_map[row["model"]])

            # Set labels
            ax.set_yticks(x)
            ax.set_yticklabels(metrics_df["wflow_id"])
            ax.set_xlabel(f"{metric.upper()} (mean ± std)", fontsize=12)
            ax.set_title(f"Workflow Comparison (Avg Across Groups): {metric.upper()} ({split})",
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            # Add legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[model])
                              for model in model_types]
            ax.legend(legend_elements, model_types, title="Model Type",
                     loc="best", frameon=True)

            plt.tight_layout()

        return fig

    def evaluate(self, test_data: pd.DataFrame) -> "WorkflowSetNestedResults":
        """
        Evaluate all workflows on test data.

        Calls evaluate() on each nested workflow fit to compute test metrics.
        Updates the results in place and returns self for method chaining.

        Args:
            test_data: Test data DataFrame with group column

        Returns:
            Self (for method chaining)

        Examples:
            >>> # Fit on training data
            >>> results = wf_set.fit_nested(train_data, group_col='country')
            >>>
            >>> # Evaluate on test data
            >>> results = results.evaluate(test_data)
            >>>
            >>> # Now collect_metrics includes both train and test splits
            >>> metrics = results.collect_metrics(split='test')
        """
        for i, result in enumerate(self.results):
            nested_fit = result.get("nested_fit")

            if nested_fit is None:
                continue  # Skip failed workflows

            try:
                # Evaluate nested workflow on test data
                evaluated_fit = nested_fit.evaluate(test_data)

                # Extract updated outputs with test metrics
                outputs, coefs, stats = evaluated_fit.extract_outputs()

                # Update result with evaluated fit and new outputs
                self.results[i]["nested_fit"] = evaluated_fit
                self.results[i]["outputs"] = outputs
                self.results[i]["coefs"] = coefs
                self.results[i]["stats"] = stats

            except Exception as e:
                print(f"  ⚠ Error evaluating {result['wflow_id']}: {e}")
                # Keep original results if evaluation fails

        return self
