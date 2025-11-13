"""
WorkflowSets for multi-model comparison.

Provides tools for comparing multiple workflows across different
preprocessing strategies and model specifications.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple
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
                   preproc: Union[List[Union[str, Any]], Dict[str, Union[str, Any]], List[Tuple[str, Union[str, Any]]]],
                   models: List[Any],
                   ids: Optional[List[str]] = None):
        """
        Create WorkflowSet from cross-product of preprocessors and models.

        Creates all combinations of preprocessors (formulas or recipes) and models.

        Args:
            preproc: Preprocessors in one of three formats:
                - List: [recipe1, recipe2] or ["y ~ x1", "y ~ x2"] (generic IDs: prep_1, prep_2, ...)
                - Dict: {'rec_lags': recipe1, 'rec_pca': recipe2} (uses keys as IDs)
                - List of tuples: [('rec_lags', recipe1), ('rec_pca', recipe2)] (uses names as IDs)
            models: List of ModelSpec objects
            ids: Optional list of ID prefixes for each preprocessor (only used with list format)

        Returns:
            WorkflowSet instance

        Examples:
            >>> # List format with generic IDs
            >>> formulas = ["y ~ x1", "y ~ x1 + x2"]
            >>> models = [linear_reg(), rand_forest()]
            >>> wf_set = WorkflowSet.from_cross(formulas, models)
            # Creates: prep_1_linear_reg_1, prep_1_rand_forest_2, prep_2_linear_reg_1, prep_2_rand_forest_2

            >>> # Dict format with custom names
            >>> rec_lags = recipe().step_lag(['x1'], lag=7)
            >>> rec_pca = recipe().step_pca(num_comp=3)
            >>> wf_set = WorkflowSet.from_cross({'rec_lags': rec_lags, 'rec_pca': rec_pca}, models)
            # Creates: rec_lags_linear_reg_1, rec_lags_rand_forest_2, rec_pca_linear_reg_1, rec_pca_rand_forest_2

            >>> # Tuple format with custom names
            >>> wf_set = WorkflowSet.from_cross([('rec_lags', rec_lags), ('rec_pca', rec_pca)], models)
            # Same result as dict format
        """
        from py_workflows import workflow

        workflows_dict = {}
        info_data = []

        # Normalize preproc to list of (name, preprocessor) tuples
        prep_items = []
        if isinstance(preproc, dict):
            # Dict format: {'rec_lags': recipe1, 'rec_pca': recipe2}
            prep_items = list(preproc.items())
        elif isinstance(preproc, list) and len(preproc) > 0 and isinstance(preproc[0], tuple):
            # Tuple format: [('rec_lags', recipe1), ('rec_pca', recipe2)]
            prep_items = preproc
        else:
            # List format (backward compatible): [recipe1, recipe2] or ["y ~ x1", "y ~ x2"]
            if ids is None:
                ids = [f"prep_{i+1}" for i in range(len(preproc))]
            prep_items = [(ids[i] if i < len(ids) else f"prep_{i+1}", p) for i, p in enumerate(preproc)]

        # Create workflows from cross-product
        for prep_id, prep in prep_items:
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

    def fit_nested_resamples(self,
                            resamples: Dict[str, Any],
                            group_col: str,
                            metrics: Optional[Any] = None,
                            verbose: bool = False) -> "WorkflowSetNestedResamples":
        """
        Fit all workflows per group using pre-defined CV splits.

        Evaluates all workflows on pre-created CV splits for each group.
        This provides robust per-group model evaluation with full control
        over CV strategy.

        Args:
            resamples: Dict mapping group names to CV split objects
                      Example: {'USA': cv_usa, 'Germany': cv_germany, ...}
            group_col: Name of the column identifying groups
            metrics: Metric set for evaluation
            verbose: If True, print detailed progress (workflow, group, fold)

        Returns:
            WorkflowSetNestedResamples with group-aware CV results

        Examples:
            >>> # Create CV splits per group
            >>> from py_rsample import time_series_cv
            >>>
            >>> cv_by_group = {}
            >>> for country in ['USA', 'Germany', 'Japan']:
            ...     country_data = data[data['country'] == country]
            ...     cv_by_group[country] = time_series_cv(
            ...         country_data,
            ...         date_column='date',
            ...         initial='4 years',
            ...         assess='1 year'
            ...     )
            >>>
            >>> # Evaluate all workflows on each group's CV splits
            >>> results = wf_set.fit_nested_resamples(
            ...     resamples=cv_by_group,
            ...     group_col='country',
            ...     metrics=metric_set(rmse, mae),
            ...     verbose=True  # Show detailed progress
            ... )
            >>>
            >>> # Collect and analyze results
            >>> metrics_by_group = results.collect_metrics(by_group=True)
            >>> ranked = results.rank_results('rmse', by_group=False)
            >>> best = results.extract_best_workflow('rmse')
        """
        if metrics is None:
            from py_yardstick import metric_set, rmse, mae, r_squared
            metrics = metric_set(rmse, mae, r_squared)

        groups = list(resamples.keys())
        print(f"Fitting {len(self.workflows)} workflows across {len(groups)} groups with CV...")
        if verbose:
            total_folds = sum(len(cv_splits) for cv_splits in resamples.values())
            print(f"Total evaluations: {len(self.workflows)} workflows × {len(groups)} groups × avg {total_folds // len(groups)} folds")
        print()

        # Store results by (workflow, group)
        all_cv_results = []

        for wf_idx, (wf_id, wf) in enumerate(self.workflows.items(), 1):
            if verbose:
                print(f"\n[{wf_idx}/{len(self.workflows)}] Workflow: {wf_id}")
            else:
                print(f"Evaluating {wf_id} across groups...")

            for group_idx, (group_name, cv_splits) in enumerate(resamples.items(), 1):
                if verbose:
                    print(f"  [{group_idx}/{len(groups)}] Group: {group_name} ({len(cv_splits)} folds)", end="", flush=True)

                try:
                    # Drop group column from CV splits to prevent it from being passed to models
                    # This is critical for supervised feature selection steps
                    from py_rsample import Split, RSplit

                    cv_splits_no_group = []
                    for rsplit in cv_splits:
                        # Get original data and drop group column
                        data_no_group = rsplit._split.data.drop(columns=[group_col])

                        # Create new Split with modified data (same indices)
                        new_split = Split(
                            data=data_no_group,
                            in_id=rsplit._split.in_id,
                            out_id=rsplit._split.out_id,
                            id=rsplit._split.id
                        )

                        # Wrap in RSplit
                        cv_splits_no_group.append(RSplit(new_split))

                    # Evaluate workflow on this group's CV splits (without group column)
                    from py_tune import fit_resamples as tune_fit_resamples
                    cv_results = tune_fit_resamples(wf, resamples=cv_splits_no_group, metrics=metrics)

                    if verbose:
                        print(" ✓")

                    # Extract metrics (fold-level)
                    fold_metrics = cv_results.collect_metrics()

                    # Add metadata columns
                    fold_metrics['wflow_id'] = wf_id
                    fold_metrics['group'] = group_name

                    # Add fold column if not present
                    if 'fold' not in fold_metrics.columns and 'id' in fold_metrics.columns:
                        fold_metrics['fold'] = fold_metrics['id']

                    all_cv_results.append({
                        "wflow_id": wf_id,
                        "group": group_name,
                        "cv_results": cv_results,
                        "fold_metrics": fold_metrics
                    })

                except Exception as e:
                    if verbose:
                        print(f" ✗")
                        print(f"    ⚠ Error: {e}")
                    else:
                        print(f"  ⚠ Error with {wf_id} for {group_name}: {e}")
                    continue

        print("\n✓ CV evaluation complete")

        return WorkflowSetNestedResamples(
            results=all_cv_results,
            workflow_set=self,
            group_col=group_col,
            metrics=metrics
        )

    def fit_global_resamples(self,
                            data: pd.DataFrame,
                            resamples: Dict[str, Any],
                            group_col: str,
                            metrics: Optional[Any] = None) -> "WorkflowSetNestedResamples":
        """
        Fit all workflows globally (with group as feature) using pre-defined per-group CV.

        Evaluates global models (with group as a feature) on pre-created CV splits
        for each group. This shows how a global model performs per-group.

        Args:
            data: Full training data with group column
            resamples: Dict mapping group names to CV split objects
                      Example: {'USA': cv_usa, 'Germany': cv_germany, ...}
            group_col: Column name identifying groups
            metrics: Metric set for evaluation

        Returns:
            WorkflowSetNestedResamples with per-group CV results for global models

        Examples:
            >>> # Create CV splits per group
            >>> from py_rsample import time_series_cv
            >>>
            >>> cv_by_group = {}
            >>> for country in ['USA', 'Germany', 'Japan']:
            ...     country_data = data[data['country'] == country]
            ...     cv_by_group[country] = time_series_cv(
            ...         country_data,
            ...         date_column='date',
            ...         initial='4 years',
            ...         assess='1 year'
            ...     )
            >>>
            >>> # Evaluate global workflows per-group
            >>> results = wf_set.fit_global_resamples(
            ...     data=data,
            ...     resamples=cv_by_group,
            ...     group_col='country',
            ...     metrics=metric_set(rmse, mae)
            ... )
            >>>
            >>> # Compare performance across groups
            >>> metrics_by_group = results.collect_metrics(by_group=True)
        """
        if metrics is None:
            from py_yardstick import metric_set, rmse, mae, r_squared
            metrics = metric_set(rmse, mae, r_squared)

        groups = list(resamples.keys())
        print(f"Fitting {len(self.workflows)} global workflows with per-group CV across {len(groups)} groups...")
        print()

        # Store results by (workflow, group)
        all_cv_results = []

        for wf_id, wf in self.workflows.items():
            print(f"Evaluating {wf_id} globally...")

            for group_name, cv_splits in resamples.items():
                try:
                    # Get group data
                    group_data = data[data[group_col] == group_name].copy()

                    # For each fold, fit global model and evaluate
                    fold_results = []
                    for fold_num, split in enumerate(cv_splits.splits):
                        # Extract train/test indices from RSplit object
                        train_idx = split._split.in_id
                        test_idx = split._split.out_id

                        # Get fold data (with group_col)
                        fold_train = group_data.iloc[train_idx].copy()
                        fold_test = group_data.iloc[test_idx].copy()

                        # Fit global workflow on training fold
                        fold_fit = wf.fit_global(fold_train, group_col=group_col)

                        # Predict on test fold
                        predictions = fold_fit.predict(fold_test)

                        # Calculate metrics
                        from py_yardstick import rmse, mae, r_squared
                        truth = fold_test[fold_fit.extract_formula().split('~')[0].strip()]

                        for metric_fn in [rmse, mae, r_squared]:
                            result_df = metric_fn(truth, predictions['.pred'])
                            metric_name = result_df.iloc[0]['metric']
                            metric_value = result_df.iloc[0]['value']

                            fold_results.append({
                                'wflow_id': wf_id,
                                'group': group_name,
                                'fold': fold_num + 1,
                                'metric': metric_name,
                                'value': metric_value
                            })

                    fold_metrics_df = pd.DataFrame(fold_results)

                    all_cv_results.append({
                        "wflow_id": wf_id,
                        "group": group_name,
                        "cv_results": None,  # Not using standard fit_resamples
                        "fold_metrics": fold_metrics_df
                    })

                except Exception as e:
                    print(f"  ⚠ Error with {wf_id} for {group_name}: {e}")
                    continue

        print("\n✓ Global CV evaluation complete")

        return WorkflowSetNestedResamples(
            results=all_cv_results,
            workflow_set=self,
            group_col=group_col,
            metrics=metrics
        )

    def compare_conformal(
        self,
        data: pd.DataFrame,
        alpha: Union[float, list] = 0.05,
        method: str = 'auto'
    ) -> pd.DataFrame:
        """
        Fit all workflows and compare conformal prediction intervals.

        Useful for understanding which preprocessing strategy or model
        provides tighter or better-calibrated prediction intervals.

        Args:
            data: Data to fit workflows on and get conformal predictions for
            alpha: Confidence level (0.05 for 95% CI) or list of levels
            method: Conformal method ('auto', 'split', 'cv+', 'jackknife+', 'enbpi')

        Returns:
            DataFrame comparing conformal intervals across workflows with columns:
            - wflow_id: Workflow identifier
            - model: Model type
            - preprocessor: Preprocessor type
            - conf_method: Conformal method used
            - avg_interval_width: Average width of prediction intervals
            - median_interval_width: Median width of prediction intervals
            - coverage: Empirical coverage (if actuals available)
            - n_predictions: Number of predictions

        Examples:
            >>> # Create multiple workflows
            >>> wf_set = WorkflowSet.from_cross(
            ...     preproc=["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + I(x1*x2)"],
            ...     models=[linear_reg(), linear_reg(penalty=0.1)]
            ... )
            >>>
            >>> # Compare conformal intervals
            >>> comparison = wf_set.compare_conformal(train_data, alpha=0.05)
            >>> print(comparison.sort_values('avg_interval_width'))
            >>>
            >>> # Find workflow with tightest intervals
            >>> best_wf_id = comparison.loc[comparison['avg_interval_width'].idxmin(), 'wflow_id']
        """
        comparison_results = []

        for wf_id, wf in self.workflows.items():
            try:
                # Fit workflow
                print(f"Fitting {wf_id}...")
                fit = wf.fit(data)

                # Get conformal predictions
                conformal_preds = fit.conformal_predict(
                    data,
                    alpha=alpha,
                    method=method
                )

                # Determine column names based on alpha
                if isinstance(alpha, list):
                    # Multiple alphas - use first one for comparison
                    alpha_val = alpha[0]
                    conf_level = int((1 - alpha_val) * 100)
                    lower_col = f'.pred_lower_{conf_level}'
                    upper_col = f'.pred_upper_{conf_level}'
                else:
                    # Single alpha
                    lower_col = '.pred_lower'
                    upper_col = '.pred_upper'

                # Calculate interval statistics
                interval_widths = (
                    conformal_preds[upper_col] - conformal_preds[lower_col]
                )

                avg_width = interval_widths.mean()
                median_width = interval_widths.median()
                conf_method_used = conformal_preds['.conf_method'].iloc[0]

                # Try to calculate coverage if actuals available
                coverage = np.nan
                try:
                    # Extract formula to get outcome column name
                    formula = fit.extract_formula()
                    outcome_col = formula.split('~')[0].strip()

                    if outcome_col in data.columns:
                        actuals = data[outcome_col].values
                        in_interval = (
                            (actuals >= conformal_preds[lower_col].values) &
                            (actuals <= conformal_preds[upper_col].values)
                        )
                        coverage = in_interval.mean()
                except:
                    # Coverage calculation failed, leave as NaN
                    pass

                # Get workflow info
                wf_info = self.info[self.info['wflow_id'] == wf_id].iloc[0]

                comparison_results.append({
                    'wflow_id': wf_id,
                    'model': wf_info['model'],
                    'preprocessor': wf_info['preprocessor'],
                    'conf_method': conf_method_used,
                    'avg_interval_width': avg_width,
                    'median_interval_width': median_width,
                    'coverage': coverage,
                    'n_predictions': len(conformal_preds)
                })

            except Exception as e:
                print(f"  ⚠ Error with {wf_id}: {e}")
                # Add failed result
                wf_info = self.info[self.info['wflow_id'] == wf_id].iloc[0]
                comparison_results.append({
                    'wflow_id': wf_id,
                    'model': wf_info['model'],
                    'preprocessor': wf_info['preprocessor'],
                    'conf_method': 'failed',
                    'avg_interval_width': np.nan,
                    'median_interval_width': np.nan,
                    'coverage': np.nan,
                    'n_predictions': 0
                })

        comparison_df = pd.DataFrame(comparison_results)

        # Sort by average interval width (tighter is better)
        comparison_df = comparison_df.sort_values('avg_interval_width')

        return comparison_df


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

    def extract_formulas(self) -> pd.DataFrame:
        """
        Extract actual formulas from all workflows for all groups.

        Returns a DataFrame showing the ACTUAL formula used by each workflow
        for each group after all preprocessing. This is critical when using
        supervised feature selection steps (like step_select_corr, step_select_vif),
        which may select different features per group.

        The returned formulas reflect the features that actually made it through
        preprocessing, not the original formula before feature selection.

        Returns:
            DataFrame with columns:
            - wflow_id: Workflow identifier
            - group: Group name
            - formula: Actual formula with features that survived preprocessing
            - n_features: Number of predictor features
            - preprocessor: Preprocessor type (formula or recipe)
            - model: Model type

        Examples:
            >>> # Get formulas from all workflows
            >>> formulas_df = results.extract_formulas()
            >>> print(formulas_df)
               wflow_id   group          formula  n_features preprocessor      model
            0  prep_1...     USA      y ~ x1 + x2           2       recipe  linear_reg
            1  prep_1... Germany          y ~ x1           1       recipe  linear_reg
            2  prep_2...     USA  y ~ x1 + x2 + x3           3       recipe rand_forest

            >>> # Filter to specific workflow
            >>> wf1_formulas = formulas_df[formulas_df['wflow_id'] == 'prep_1_linear_reg_1']

            >>> # Check if formulas differ across groups
            >>> formulas_df.groupby('wflow_id')['formula'].nunique()

            >>> # See which groups have different feature counts
            >>> formulas_df.pivot(index='wflow_id', columns='group', values='n_features')
        """
        all_formulas = []

        for result in self.results:
            wf_id = result["wflow_id"]
            nested_fit = result.get("nested_fit")

            if nested_fit is None:
                continue  # Skip failed workflows

            # Extract ACTUAL formula from each group's fit
            # This shows which features the MODEL actually used (critical for supervised feature selection)
            for group_name, wf_fit in nested_fit.group_fits.items():
                try:
                    # Get original formula for outcome variable name
                    original_formula = wf_fit.extract_formula()
                    outcome_name = original_formula.split('~')[0].strip()

                    # Get ACTUAL feature names from the fitted model
                    # This is more reliable than preprocessing because it shows what the model ACTUALLY used
                    model_fit = wf_fit.fit  # The ModelFit object
                    predictor_cols = None

                    if hasattr(model_fit, 'fit_data'):
                        fit_data = model_fit.fit_data

                        # Try sklearn-based engines first (store X_train DataFrame)
                        if 'X_train' in fit_data:
                            X_train = fit_data['X_train']

                            if isinstance(X_train, pd.DataFrame):
                                # Get column names from the actual training data used by model
                                # This reflects what survived ALL preprocessing including feature selection
                                predictor_cols = [
                                    col for col in X_train.columns
                                    if col != 'Intercept'  # Exclude patsy's Intercept column
                                ]
                            else:
                                # X_train is array, can't get column names
                                n_features = X_train.shape[1] if hasattr(X_train, 'shape') else 0

                        # Try time series engines (store exog_vars list)
                        elif 'exog_vars' in fit_data:
                            exog_vars = fit_data['exog_vars']

                            if isinstance(exog_vars, list):
                                # Time series models store exogenous variable names as list
                                predictor_cols = exog_vars
                            else:
                                # Unexpected format
                                pass

                    # Fallback if we couldn't extract features
                    if predictor_cols is None:
                        formula = original_formula
                        n_features = formula.count('+') + 1 if '+' in formula else 1

                    # Reconstruct formula with actual features
                    if predictor_cols is not None:
                        if len(predictor_cols) == 0:
                            formula = f"{outcome_name} ~ 1"  # Intercept only
                            n_features = 0
                        else:
                            formula = f"{outcome_name} ~ {' + '.join(predictor_cols)}"
                            n_features = len(predictor_cols)

                    all_formulas.append({
                        'wflow_id': wf_id,
                        'group': group_name,
                        'formula': formula,
                        'n_features': n_features
                    })
                except Exception as e:
                    # If extraction fails, skip this group
                    import warnings
                    warnings.warn(f"Failed to extract formula for {wf_id}, {group_name}: {e}")
                    continue

        # Create DataFrame
        formulas_df = pd.DataFrame(all_formulas)

        # Add workflow info from workflow_set
        if not formulas_df.empty:
            formulas_df = formulas_df.merge(
                self.workflow_set.info[['wflow_id', 'preprocessor', 'model']],
                on='wflow_id',
                how='left'
            )

        return formulas_df

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


@dataclass
class WorkflowSetNestedResamples:
    """
    Results from fit_nested_resamples() or fit_global_resamples().

    Contains CV evaluation results for each workflow across each group.

    Attributes:
        results: List of dictionaries with CV results per (workflow, group)
        workflow_set: The original WorkflowSet
        group_col: Name of the group column
        metrics: The metric set used for evaluation
    """
    results: List[Dict[str, Any]]
    workflow_set: WorkflowSet
    group_col: str
    metrics: Any

    def collect_metrics(self, by_group: bool = True, summarize: bool = True) -> pd.DataFrame:
        """
        Collect metrics from CV evaluations.

        Args:
            by_group: If True, return metrics per group; if False, average across groups
            summarize: If True, average across CV folds; if False, return fold-level metrics

        Returns:
            DataFrame with metrics

        Examples:
            >>> # Average metrics per group
            >>> metrics_by_group = results.collect_metrics(by_group=True, summarize=True)
            >>>
            >>> # Average across all groups
            >>> metrics_overall = results.collect_metrics(by_group=False, summarize=True)
            >>>
            >>> # Fold-level metrics per group
            >>> fold_metrics = results.collect_metrics(by_group=True, summarize=False)
        """
        all_metrics = []

        for result in self.results:
            fold_metrics = result['fold_metrics']

            if summarize:
                # Average across folds
                summary = fold_metrics.groupby(['wflow_id', 'group', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()
                summary.columns = ['wflow_id', 'group', 'metric', 'mean', 'std', 'n']
                all_metrics.append(summary)
            else:
                # Keep fold-level detail
                all_metrics.append(fold_metrics)

        if not all_metrics:
            return pd.DataFrame()

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        if not by_group and summarize:
            # Average across groups
            grouped = metrics_df.groupby(['wflow_id', 'metric']).agg({
                'mean': 'mean',
                'std': 'mean',
                'n': 'sum'
            }).reset_index()
            grouped['group'] = 'global'
            metrics_df = grouped

        # Add workflow info
        metrics_df = metrics_df.merge(
            self.workflow_set.info[['wflow_id', 'preprocessor', 'model']],
            on='wflow_id',
            how='left'
        )

        return metrics_df

    def rank_results(self,
                    metric: str = 'rmse',
                    by_group: bool = False,
                    n: int = 10) -> pd.DataFrame:
        """
        Rank workflows by CV performance.

        Args:
            metric: Metric to rank by
            by_group: If True, rank per group; if False, rank overall
            n: Number of top workflows to return

        Returns:
            DataFrame with ranked workflows

        Examples:
            >>> # Top 10 workflows overall
            >>> top_overall = results.rank_results('rmse', by_group=False, n=10)
            >>>
            >>> # Top 5 workflows per group
            >>> top_per_group = results.rank_results('rmse', by_group=True, n=5)
        """
        metrics = self.collect_metrics(by_group=by_group, summarize=True)

        # Filter to ranking metric
        metric_df = metrics[metrics['metric'] == metric].copy()

        if by_group:
            # Rank within each group
            metric_df['rank'] = metric_df.groupby('group')['mean'].rank(method='min')
            ranked = metric_df[metric_df['rank'] <= n].sort_values(['group', 'rank'])
        else:
            # Rank overall
            metric_df['rank'] = metric_df['mean'].rank(method='min')
            ranked = metric_df[metric_df['rank'] <= n].sort_values('rank')

        return ranked[['group', 'rank', 'wflow_id', 'mean', 'std', 'preprocessor', 'model']]

    def extract_best_workflow(self,
                             metric: str = 'rmse',
                             by_group: bool = False) -> Union[str, pd.DataFrame]:
        """
        Extract ID of best workflow.

        Args:
            metric: Metric to use for selection
            by_group: If True, return best per group; if False, return single best

        Returns:
            If by_group=False: workflow ID string
            If by_group=True: DataFrame with best workflow per group

        Examples:
            >>> # Best workflow overall
            >>> best_id = results.extract_best_workflow('rmse', by_group=False)
            >>>
            >>> # Best workflow per group
            >>> best_by_group = results.extract_best_workflow('rmse', by_group=True)
        """
        ranked = self.rank_results(metric=metric, by_group=by_group, n=1)

        if by_group:
            return ranked[['group', 'wflow_id', 'mean', 'preprocessor', 'model']]
        else:
            return ranked.iloc[0]['wflow_id']

    def autoplot(self,
                metric: str = 'rmse',
                by_group: bool = False,
                top_n: int = 10):
        """
        Plot CV performance comparison.

        Args:
            metric: Metric to plot
            by_group: If True, separate subplot per group
            top_n: Number of top workflows to show

        Returns:
            Matplotlib figure

        Examples:
            >>> # Overall comparison with error bars
            >>> fig = results.autoplot('rmse', by_group=False, top_n=10)
            >>> plt.show()
            >>>
            >>> # Per-group comparison
            >>> fig = results.autoplot('rmse', by_group=True, top_n=5)
            >>> plt.show()
        """
        import matplotlib.pyplot as plt

        metrics = self.collect_metrics(by_group=by_group, summarize=True)
        metric_df = metrics[metrics['metric'] == metric].copy()

        if by_group:
            groups = metric_df['group'].unique()
            n_groups = len(groups)
            ncols = min(3, n_groups)
            nrows = (n_groups + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            axes = axes.flatten() if n_groups > 1 else [axes]

            for i, group in enumerate(groups):
                group_df = metric_df[metric_df['group'] == group].sort_values('mean').head(top_n)

                ax = axes[i]
                ax.barh(range(len(group_df)), group_df['mean'], xerr=group_df['std'])
                ax.set_yticks(range(len(group_df)))
                ax.set_yticklabels(group_df['wflow_id'], fontsize=8)
                ax.set_xlabel(metric.upper())
                ax.set_title(f'{group}', fontweight='bold')
                ax.invert_yaxis()

            # Hide unused subplots
            for i in range(n_groups, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
        else:
            # Overall plot
            top_workflows = metric_df.sort_values('mean').head(top_n)

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
            ax.barh(range(len(top_workflows)), top_workflows['mean'], xerr=top_workflows['std'])
            ax.set_yticks(range(len(top_workflows)))
            ax.set_yticklabels(top_workflows['wflow_id'], fontsize=10)
            ax.set_xlabel(f'{metric.upper()} (mean ± std across CV folds & groups)', fontsize=12)
            ax.set_title(f'Top {top_n} Workflows by CV {metric.upper()}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()

        return fig

    def compare_train_cv(self,
                        train_stats: pd.DataFrame,
                        metrics: Optional[list] = None) -> pd.DataFrame:
        """
        Compare training metrics with CV metrics to identify overfitting.

        This helper method combines training performance (from extract_outputs() stats)
        with CV performance (from collect_metrics()) to provide a comprehensive view
        of workflow generalization.

        Args:
            train_stats: DataFrame from extract_outputs()[2] containing training metrics
                        Must have columns: wflow_id, group, and metric columns (rmse, mae, r_squared, etc.)
                        Can be in long format (metric/value columns) or wide format (individual metric columns)
            metrics: List of metrics to compare (default: ['rmse', 'mae', 'r_squared'])

        Returns:
            DataFrame with columns:
                - wflow_id: Workflow identifier
                - group: Group name
                - {metric}_train: Training metric value (optimistic, in-sample)
                - {metric}_cv: CV metric value (realistic, out-of-sample)
                - {metric}_overfit_ratio: Ratio of CV/train (>1.2 indicates overfitting)
                - fit_quality: Status flag ('🟢 Good Generalization', '🟡 Moderate Overfit', etc.)

        Examples:
            >>> # Get training stats from fitted workflows
            >>> outputs, coeffs, wfset_stats = wf_set.extract_outputs()
            >>>
            >>> # Compare with CV results
            >>> comparison = cv_results.compare_train_cv(wfset_stats)
            >>>
            >>> # View workflows with overfitting issues
            >>> overfit = comparison[comparison['rmse_overfit_ratio'] > 1.2]
            >>>
            >>> # Top workflows by CV performance (per group)
            >>> best = comparison.sort_values(['group', 'rmse_cv']).groupby('group').head(5)

        Notes:
            - Training metrics show in-sample fit (optimistic)
            - CV metrics show out-of-sample generalization (realistic)
            - Overfit ratio >1.2 suggests moderate overfitting
            - Overfit ratio >1.5 suggests severe overfitting
            - Good models have ratio close to 1.0 (similar train/CV performance)
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r_squared']

        # Check if train_stats is in long format (metric, value columns) or wide format
        if 'metric' in train_stats.columns and 'value' in train_stats.columns:
            # Long format - pivot to wide
            train_wide = train_stats[train_stats['metric'].isin(metrics)].pivot_table(
                index=['wflow_id', 'group'],
                columns='metric',
                values='value'
            ).reset_index()
        else:
            # Already in wide format
            train_wide = train_stats.copy()

        # Validate required columns exist
        required_cols = ['wflow_id', 'group'] + metrics
        missing_cols = [col for col in required_cols if col not in train_wide.columns]
        if missing_cols:
            raise ValueError(f"train_stats missing required columns: {missing_cols}")

        # Step 1: Aggregate training metrics (in-sample)
        train_summary = train_wide.groupby(['wflow_id', 'group'])[metrics].mean().add_suffix('_train').reset_index()

        # Step 2: Get CV metrics (out-of-sample) - always test metrics from CV folds
        cv_metrics = self.collect_metrics(by_group=True, summarize=True)

        # Pivot CV metrics to wide format
        cv_summary = cv_metrics.pivot_table(
            index=['wflow_id', 'group'],
            columns='metric',
            values='mean'
        ).reset_index()

        # Only keep requested metrics, add _cv suffix
        cv_cols_to_keep = ['wflow_id', 'group'] + [m for m in metrics if m in cv_summary.columns]
        cv_summary = cv_summary[cv_cols_to_keep].copy()

        # Add _cv suffix to metric columns
        rename_dict = {m: f"{m}_cv" for m in metrics if m in cv_summary.columns}
        cv_summary = cv_summary.rename(columns=rename_dict)

        # Step 3: Merge training and CV metrics
        comparison = train_summary.merge(cv_summary, on=['wflow_id', 'group'], how='outer')

        # Step 4: Add overfitting indicators for each metric
        for metric in metrics:
            train_col = f"{metric}_train"
            cv_col = f"{metric}_cv"

            if train_col in comparison.columns and cv_col in comparison.columns:
                # For metrics where lower is better (rmse, mae)
                if metric.lower() in ['rmse', 'mae', 'mape', 'smape', 'mse']:
                    comparison[f"{metric}_overfit_ratio"] = comparison[cv_col] / comparison[train_col]
                # For metrics where higher is better (r_squared)
                elif metric.lower() in ['r_squared', 'r2', 'accuracy']:
                    comparison[f"{metric}_generalization_drop"] = comparison[train_col] - comparison[cv_col]

        # Step 5: Add overall fit quality status (based on RMSE if available)
        if 'rmse_overfit_ratio' in comparison.columns:
            comparison['fit_quality'] = comparison['rmse_overfit_ratio'].apply(lambda x:
                '🔴 Severe Overfit' if pd.notna(x) and x > 1.5 else
                '🟡 Moderate Overfit' if pd.notna(x) and x > 1.2 else
                '🟢 Good Generalization' if pd.notna(x) and x < 1.1 else
                '⚪ Normal' if pd.notna(x) else
                '❓ Unknown'
            )

        # Step 6: Sort by CV performance (most important metric)
        primary_metric = metrics[0]  # Usually rmse
        cv_col = f"{primary_metric}_cv"
        if cv_col in comparison.columns:
            # Sort ascending for metrics where lower is better
            ascending = primary_metric.lower() in ['rmse', 'mae', 'mape', 'smape', 'mse']
            comparison = comparison.sort_values([cv_col], ascending=ascending)

        return comparison
