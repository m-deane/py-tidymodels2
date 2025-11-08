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
