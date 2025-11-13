"""
Hyperparameter tuning for py-tidymodels

Provides tidymodels-style hyperparameter optimization with grid search and
cross-validation evaluation.
"""

from typing import Optional, Union, List, Dict, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from joblib import Parallel, delayed

from .parallel_utils import (
    validate_n_jobs,
    get_joblib_backend,
    check_windows_compatibility,
    format_parallel_info
)


# ============================================================================
# Parameter Marker
# ============================================================================

class TuneParameter:
    """
    Marker for parameters to be tuned.

    Use tune() to mark model or recipe parameters for hyperparameter optimization.

    Examples:
        >>> from py_parsnip import linear_reg
        >>> from py_tune import tune
        >>>
        >>> # Mark penalty parameter for tuning
        >>> spec = linear_reg(penalty=tune())
    """
    def __init__(self, id: Optional[str] = None):
        # Use __builtins__ to access the built-in id() function
        # since the parameter name 'id' shadows it
        if id is None:
            self.id = f"tune_{__builtins__['id'](self)}"
        else:
            self.id = id

    def __repr__(self):
        return f"tune(id='{self.id}')"


def tune(id: Optional[str] = None) -> TuneParameter:
    """
    Mark a parameter for tuning.

    Args:
        id: Optional identifier for the parameter

    Returns:
        TuneParameter marker object

    Examples:
        >>> spec = linear_reg(penalty=tune(), mixture=tune())
    """
    return TuneParameter(id=id)


# ============================================================================
# Parameter Grid Generation
# ============================================================================

def grid_regular(param_info: Dict[str, Dict[str, Any]], levels: int = 3) -> pd.DataFrame:
    """
    Create a regular grid of parameter values.

    Args:
        param_info: Dictionary mapping parameter names to their specifications
                   Each spec should have 'range' or 'values' key
                   Optionally include 'type': 'int' to convert to integers
        levels: Number of levels for each parameter (default: 3)

    Returns:
        DataFrame with one row per parameter combination

    Examples:
        >>> param_info = {
        ...     'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
        ...     'mixture': {'range': (0, 1)},
        ...     'trees': {'range': (50, 200), 'type': 'int'}
        ... }
        >>> grid = grid_regular(param_info, levels=3)
    """
    # Parameters that should be integers (for automatic detection)
    INT_PARAMS = {'trees', 'tree_depth', 'min_n', 'stop_iter', 'mtry', 'neighbors', 'epochs'}

    param_values = {}

    for param_name, spec in param_info.items():
        if 'values' in spec:
            # Use explicitly provided values
            param_values[param_name] = spec['values']
        elif 'range' in spec:
            # Generate values from range
            min_val, max_val = spec['range']
            trans = spec.get('trans', 'identity')
            param_type = spec.get('type', 'auto')

            if trans == 'log':
                # Log transformation for penalty-like parameters
                values = np.logspace(np.log10(min_val), np.log10(max_val), levels)
            else:
                # Linear spacing
                values = np.linspace(min_val, max_val, levels)

            # Convert to integers if specified or auto-detected
            if param_type == 'int' or (param_type == 'auto' and param_name in INT_PARAMS):
                values = np.round(values).astype(int)

            param_values[param_name] = values
        else:
            raise ValueError(f"Parameter '{param_name}' must have 'range' or 'values' key")

    # Create all combinations
    param_names = list(param_values.keys())
    combinations = list(itertools.product(*[param_values[name] for name in param_names]))

    # Convert to DataFrame
    grid = pd.DataFrame(combinations, columns=param_names)
    grid['.config'] = [f"config_{i+1:03d}" for i in range(len(grid))]

    return grid


def grid_random(param_info: Dict[str, Dict[str, Any]], size: int = 10, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create a random grid of parameter values.

    Args:
        param_info: Dictionary mapping parameter names to their specifications
        size: Number of random combinations to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with one row per parameter combination

    Examples:
        >>> param_info = {
        ...     'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
        ...     'trees': {'range': (10, 1000), 'type': 'int'}
        ... }
        >>> grid = grid_random(param_info, size=20, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    param_values = {}

    for param_name, spec in param_info.items():
        if 'range' in spec:
            min_val, max_val = spec['range']
            trans = spec.get('trans', 'identity')
            param_type = spec.get('type', 'float')

            if trans == 'log':
                # Log-uniform sampling
                values = np.exp(np.random.uniform(np.log(min_val), np.log(max_val), size))
            else:
                # Uniform sampling
                values = np.random.uniform(min_val, max_val, size)

            if param_type == 'int':
                values = np.round(values).astype(int)

            param_values[param_name] = values
        else:
            raise ValueError(f"Parameter '{param_name}' must have 'range' key for random grid")

    # Convert to DataFrame
    grid = pd.DataFrame(param_values)
    grid['.config'] = [f"config_{i+1:03d}" for i in range(len(grid))]

    return grid


# ============================================================================
# Tune Results
# ============================================================================

@dataclass
class TuneResults:
    """
    Results from hyperparameter tuning.

    Stores metrics, predictions, and parameter configurations from tuning.
    """
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    workflow: Any = None
    resamples: Any = None
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)

    def collect_metrics(self) -> pd.DataFrame:
        """
        Collect all metrics from tuning.

        Returns:
            DataFrame with metrics for each configuration
        """
        return self.metrics.copy()

    def collect_predictions(self) -> pd.DataFrame:
        """
        Collect all predictions from tuning.

        Returns:
            DataFrame with predictions for each configuration
        """
        return self.predictions.copy()

    def show_best(self, metric: str, n: int = 5, maximize: bool = True) -> pd.DataFrame:
        """
        Show the best n parameter configurations.

        Args:
            metric: Metric name to rank by
            n: Number of top configurations to show
            maximize: Whether to maximize the metric (False to minimize)

        Returns:
            DataFrame with top n configurations
        """
        # Check if data is in long format (has 'metric' column) or wide format
        if 'metric' in self.metrics.columns:
            # Long format: filter for the specified metric
            metric_data = self.metrics[self.metrics['metric'] == metric]
            # Calculate mean metric across resamples for each config
            summary = metric_data.groupby('.config')['value'].mean().reset_index()
            summary.columns = ['.config', 'mean']
        else:
            # Wide format: metric is already a column
            summary = self.metrics.groupby('.config')[metric].mean().reset_index()
            summary.columns = ['.config', 'mean']

        # Sort and get top n
        summary = summary.sort_values('mean', ascending=not maximize).head(n)

        # Merge with grid to get parameter values
        result = summary.merge(self.grid, on='.config')

        return result

    def select_best(self, metric: str, maximize: bool = True) -> Dict[str, Any]:
        """
        Select the best parameter configuration.

        Args:
            metric: Metric name to rank by
            maximize: Whether to maximize the metric (False to minimize)

        Returns:
            Dictionary of best parameters
        """
        best_config = self.show_best(metric, n=1, maximize=maximize)

        # Extract parameter values
        param_cols = [col for col in best_config.columns if col not in ['.config', 'mean']]
        params = best_config[param_cols].iloc[0].to_dict()

        return params

    def select_by_one_std_err(self, metric: str, maximize: bool = True) -> Dict[str, Any]:
        """
        Select best configuration using the one-standard-error rule.

        Chooses the simplest model within one standard error of the best performance.

        Args:
            metric: Metric name to rank by
            maximize: Whether to maximize the metric (False to minimize)

        Returns:
            Dictionary of selected parameters
        """
        # Check if data is in long format (has 'metric' column) or wide format
        if 'metric' in self.metrics.columns:
            # Long format: filter for the specified metric
            metric_data = self.metrics[self.metrics['metric'] == metric]
            # Calculate mean and std for each config
            summary = metric_data.groupby('.config')['value'].agg(['mean', 'std']).reset_index()
            summary.columns = ['.config', 'mean', 'std']
        else:
            # Wide format: metric is already a column
            summary = self.metrics.groupby('.config')[metric].agg(['mean', 'std']).reset_index()
            summary.columns = ['.config', 'mean', 'std']

        # Find best mean
        best_idx = summary['mean'].idxmax() if maximize else summary['mean'].idxmin()
        best_mean = summary.loc[best_idx, 'mean']
        best_std = summary.loc[best_idx, 'std']

        # Find configs within one std error
        if maximize:
            threshold = best_mean - best_std
            candidates = summary[summary['mean'] >= threshold]
        else:
            threshold = best_mean + best_std
            candidates = summary[summary['mean'] <= threshold]

        # Among candidates, select simplest (assume first parameter is complexity)
        # Merge with grid to get parameter values
        candidates = candidates.merge(self.grid, on='.config')
        param_cols = [col for col in candidates.columns if col not in ['.config', 'mean', 'std']]

        if len(param_cols) > 0:
            # Sort by first parameter (ascending assumes lower = simpler)
            candidates = candidates.sort_values(param_cols[0])

        # Select first (simplest)
        params = candidates[param_cols].iloc[0].to_dict()

        return params


# ============================================================================
# Tuning Functions
# ============================================================================

def fit_resamples(
    workflow,
    resamples,
    metrics=None,
    control: Optional[Dict[str, Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False
) -> TuneResults:
    """
    Fit a workflow to resamples without tuning.

    Evaluates a single workflow configuration across multiple resamples
    (e.g., cross-validation folds).

    Args:
        workflow: Workflow object to fit
        resamples: Resampling object (from py_rsample)
        metrics: Metric set or list of metrics (from py_yardstick)
        control: Optional control parameters
        n_jobs: Number of parallel jobs. None or 1 for sequential execution,
                -1 for all CPU cores, or positive integer for specific number of cores.
        verbose: If True, display progress messages

    Returns:
        TuneResults object with metrics and predictions

    Examples:
        >>> from py_workflows import workflow
        >>> from py_parsnip import linear_reg
        >>> from py_rsample import vfold_cv
        >>> from py_yardstick import metric_set, rmse, mae
        >>>
        >>> wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        >>> folds = vfold_cv(data, v=5)
        >>> my_metrics = metric_set(rmse, mae)
        >>>
        >>> # Sequential execution
        >>> results = fit_resamples(wf, folds, metrics=my_metrics)
        >>>
        >>> # Parallel execution with all cores
        >>> results = fit_resamples(wf, folds, metrics=my_metrics, n_jobs=-1, verbose=True)
    """
    control = control or {}
    save_pred = control.get('save_pred', False)

    # Convert resamples to list for iteration
    resample_splits = list(enumerate(resamples))

    # Validate and resolve n_jobs with warnings
    effective_n_jobs = validate_n_jobs(n_jobs, len(resample_splits), verbose=verbose)

    # Windows compatibility check
    if effective_n_jobs > 1:
        check_windows_compatibility(verbose=verbose and n_jobs is not None)

    # Decide between sequential and parallel execution
    if effective_n_jobs == 1:
        # Sequential execution
        if verbose:
            info = format_parallel_info(1, len(resample_splits), "CV folds")
            print(f"{info}...")

        results = []
        for fold_idx, split in resample_splits:
            result = _fit_single_fold(workflow, split, fold_idx, metrics, save_pred)
            results.append(result)
            if verbose:
                print(f"  Fold {fold_idx+1}/{len(resample_splits)} complete")
    else:
        # Parallel execution
        if verbose:
            info = format_parallel_info(effective_n_jobs, len(resample_splits), "CV folds")
            print(f"{info}...")

        joblib_verbose = 10 if verbose else 0
        backend = get_joblib_backend()
        results = Parallel(n_jobs=effective_n_jobs, verbose=joblib_verbose, backend=backend)(
            delayed(_fit_single_fold)(workflow, split, fold_idx, metrics, save_pred)
            for fold_idx, split in resample_splits
        )

    # Process results
    all_metrics = []
    all_predictions = []
    errors = []

    for metrics_df, predictions_df, error_msg in results:
        if error_msg:
            errors.append(error_msg)

        if not metrics_df.empty:
            all_metrics.append(metrics_df)

        if not predictions_df.empty:
            all_predictions.append(predictions_df)

    # Print errors
    for error in errors:
        print(f"Warning: {error}")

    # Combine results
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = pd.DataFrame(columns=['metric', 'value', '.resample', '.config'])

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()

    # Create grid with single configuration
    grid = pd.DataFrame({'.config': ['config_001']})

    if verbose:
        print(f"✓ Evaluation complete: {len(all_metrics)} successful folds")

    return TuneResults(
        metrics=metrics_df,
        predictions=predictions_df,
        workflow=workflow,
        resamples=resamples,
        grid=grid
    )


def tune_grid(
    workflow,
    resamples,
    grid: Optional[Union[int, pd.DataFrame]] = None,
    metrics=None,
    param_info: Optional[Dict[str, Dict[str, Any]]] = None,
    control: Optional[Dict[str, Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False
) -> TuneResults:
    """
    Tune workflow hyperparameters via grid search.

    Args:
        workflow: Workflow object with tune() placeholders
        resamples: Resampling object
        grid: Either integer (number of levels) or DataFrame of parameter combinations
        metrics: Metric set or list of metrics
        param_info: Parameter information for grid generation (required if grid is int)
        control: Optional control parameters
        n_jobs: Number of parallel jobs. None or 1 for sequential execution,
                -1 for all CPU cores, or positive integer for specific number of cores.
        verbose: If True, display progress messages

    Returns:
        TuneResults object with metrics across all configurations

    Examples:
        >>> from py_tune import tune, tune_grid, grid_regular
        >>>
        >>> # Create workflow with tunable parameters
        >>> spec = linear_reg(penalty=tune(), mixture=tune())
        >>> wf = workflow().add_formula("y ~ x").add_model(spec)
        >>>
        >>> # Define parameter space
        >>> param_info = {
        ...     'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
        ...     'mixture': {'range': (0, 1)}
        ... }
        >>>
        >>> # Run grid search (sequential)
        >>> results = tune_grid(wf, folds, param_info=param_info, grid=5)
        >>>
        >>> # Run grid search (parallel)
        >>> results = tune_grid(wf, folds, param_info=param_info, grid=5, n_jobs=-1, verbose=True)
    """
    control = control or {}
    save_pred = control.get('save_pred', False)

    # Generate grid if not provided
    if grid is None or isinstance(grid, int):
        if param_info is None:
            raise ValueError("param_info required when grid is an integer or None")
        levels = grid if isinstance(grid, int) else 3
        grid_df = grid_regular(param_info, levels=levels)
    else:
        grid_df = grid.copy()
        if '.config' not in grid_df.columns:
            grid_df['.config'] = [f"config_{i+1:03d}" for i in range(len(grid_df))]

    # Create list of all config × fold combinations
    config_fold_combinations = []
    for config_idx, row in grid_df.iterrows():
        config_name = row['.config']
        params = {k: v for k, v in row.items() if k != '.config'}

        for fold_idx, split in enumerate(resamples):
            config_fold_combinations.append((params, config_name, split, fold_idx))

    # Validate and resolve n_jobs with warnings
    effective_n_jobs = validate_n_jobs(n_jobs, len(config_fold_combinations), verbose=verbose)

    # Windows compatibility check
    if effective_n_jobs > 1:
        check_windows_compatibility(verbose=verbose and n_jobs is not None)

    # Decide between sequential and parallel execution
    if effective_n_jobs == 1:
        # Sequential execution
        if verbose:
            print(f"Tuning grid: {len(grid_df)} configs × {len(list(resamples))} folds = {len(config_fold_combinations)} fits (sequential)...")

        results = []
        for i, (params, config_name, split, fold_idx) in enumerate(config_fold_combinations):
            result = _fit_single_config_fold(workflow, params, config_name, split, fold_idx, metrics, save_pred)
            results.append(result)
            if verbose and (i + 1) % len(list(resamples)) == 0:
                completed_configs = (i + 1) // len(list(resamples))
                print(f"  Config {completed_configs}/{len(grid_df)} complete")
    else:
        # Parallel execution
        if verbose:
            info = format_parallel_info(effective_n_jobs, len(config_fold_combinations), "grid fits")
            print(f"Tuning grid: {len(grid_df)} configs × {len(list(resamples))} folds = {info}...")

        joblib_verbose = 10 if verbose else 0
        backend = get_joblib_backend()
        results = Parallel(n_jobs=effective_n_jobs, verbose=joblib_verbose, backend=backend)(
            delayed(_fit_single_config_fold)(workflow, params, config_name, split, fold_idx, metrics, save_pred)
            for params, config_name, split, fold_idx in config_fold_combinations
        )

    # Process results
    all_metrics = []
    all_predictions = []
    errors = []

    for metrics_df, predictions_df, error_msg in results:
        if error_msg:
            errors.append(error_msg)

        if not metrics_df.empty:
            all_metrics.append(metrics_df)

        if not predictions_df.empty:
            all_predictions.append(predictions_df)

    # Print errors
    for error in errors:
        print(f"Warning: {error}")

    # Combine results
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = pd.DataFrame(columns=['metric', 'value', '.resample', '.config'])

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()

    if verbose:
        print(f"✓ Grid search complete: {len(all_metrics)} successful fits")

    return TuneResults(
        metrics=metrics_df,
        predictions=predictions_df,
        workflow=workflow,
        resamples=resamples,
        grid=grid_df
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _fit_single_fold(workflow, split, fold_idx, metrics, save_pred):
    """
    Fit workflow on a single fold.

    Helper function for parallel execution in fit_resamples.

    Args:
        workflow: Workflow to fit
        split: Train/test split
        fold_idx: Fold index
        metrics: Metric function or list
        save_pred: Whether to save predictions

    Returns:
        Tuple of (metrics_df, predictions_df, error_msg)
    """
    from py_rsample import training, testing

    try:
        train_data = training(split)
        test_data = testing(split)

        # Fit workflow
        wf_fit = workflow.fit(train_data)

        # Predict on test set
        predictions = wf_fit.predict(test_data)

        # Calculate metrics
        outcome = None

        if hasattr(workflow, 'preprocessor') and isinstance(workflow.preprocessor, str):
            # Formula-based workflow
            outcome = workflow.preprocessor.split('~')[0].strip()
        else:
            # Recipe-based or other workflow
            from py_recipes import Recipe
            if hasattr(workflow, 'preprocessor') and isinstance(workflow.preprocessor, Recipe):
                blueprint = wf_fit.fit.blueprint

                if hasattr(blueprint, 'outcome_name'):
                    outcome = blueprint.outcome_name
                elif hasattr(blueprint, 'roles') and 'outcome' in blueprint.roles:
                    outcome = blueprint.roles['outcome'][0] if blueprint.roles['outcome'] else None
                elif isinstance(blueprint, dict):
                    outcome = blueprint.get('outcome_name') or blueprint.get('y_name')
                    if outcome is None and 'formula_data' in blueprint:
                        formula_str = str(blueprint.get('formula', ''))
                        if '~' in formula_str:
                            outcome = formula_str.split('~')[0].strip()

        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()

        if outcome and outcome in test_data.columns:
            truth = test_data[outcome]
            estimate = predictions['.pred']

            # Compute metrics
            if metrics is None:
                from py_yardstick import metric_set, rmse, mae, r_squared
                metric_fn = metric_set(rmse, mae, r_squared)
                metric_results = metric_fn(truth, estimate)
            elif callable(metrics):
                metric_results = metrics(truth, estimate)
            else:
                from py_yardstick import metric_set
                metric_fn = metric_set(*metrics)
                metric_results = metric_fn(truth, estimate)

            # Add fold identifier
            metric_results['.resample'] = f"Fold{fold_idx+1:02d}"
            metric_results['.config'] = "config_001"
            metrics_df = metric_results

        # Save predictions if requested
        if save_pred:
            predictions['.resample'] = f"Fold{fold_idx+1:02d}"
            predictions['.config'] = "config_001"
            predictions['.row'] = test_data.index.tolist()
            predictions_df = predictions

        return (metrics_df, predictions_df, None)

    except Exception as e:
        error_msg = f"Fold {fold_idx+1} failed with error: {str(e)}"
        return (pd.DataFrame(), pd.DataFrame(), error_msg)


def _fit_single_config_fold(workflow, params, config_name, split, fold_idx, metrics, save_pred):
    """
    Fit workflow with specific parameters on a single fold.

    Helper function for parallel execution in tune_grid.

    Args:
        workflow: Base workflow
        params: Parameter dictionary
        config_name: Configuration name
        split: Train/test split
        fold_idx: Fold index
        metrics: Metric function or list
        save_pred: Whether to save predictions

    Returns:
        Tuple of (metrics_df, predictions_df, error_msg)
    """
    from py_rsample import training, testing

    try:
        # Update workflow with current parameters
        current_wf = _update_workflow_params(workflow, params)

        train_data = training(split)
        test_data = testing(split)

        # Fit and predict
        wf_fit = current_wf.fit(train_data)
        predictions = wf_fit.predict(test_data)

        # Calculate metrics
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()

        if hasattr(current_wf, 'preprocessor') and isinstance(current_wf.preprocessor, str):
            outcome = current_wf.preprocessor.split('~')[0].strip()
            truth = test_data[outcome]
            estimate = predictions['.pred']

            # Compute metrics
            if metrics is None:
                from py_yardstick import metric_set, rmse, mae, r_squared
                metric_fn = metric_set(rmse, mae, r_squared)
                metric_results = metric_fn(truth, estimate)
            elif callable(metrics):
                metric_results = metrics(truth, estimate)
            else:
                from py_yardstick import metric_set
                metric_fn = metric_set(*metrics)
                metric_results = metric_fn(truth, estimate)

            metric_results['.resample'] = f"Fold{fold_idx+1:02d}"
            metric_results['.config'] = config_name
            metrics_df = metric_results

        # Save predictions
        if save_pred:
            predictions['.resample'] = f"Fold{fold_idx+1:02d}"
            predictions['.config'] = config_name
            predictions['.row'] = test_data.index.tolist()
            predictions_df = predictions

        return (metrics_df, predictions_df, None)

    except Exception as e:
        error_msg = f"Config {config_name}, Fold {fold_idx+1} failed: {str(e)}"
        return (pd.DataFrame(), pd.DataFrame(), error_msg)


def _update_workflow_params(workflow, params: Dict[str, Any]):
    """
    Update workflow parameters.

    Creates a new workflow with updated parameter values.

    Args:
        workflow: Original workflow
        params: Dictionary of parameter values

    Returns:
        Updated workflow
    """
    # Get model spec
    spec = workflow.spec

    # Handle both dictionary and tuple formats for spec.args
    if isinstance(spec.args, dict):
        # Modern format: spec.args is a dictionary
        new_args = spec.args.copy()
        for key, value in params.items():
            new_args[key] = value
    else:
        # Legacy/test format: spec.args is tuple of tuples
        new_args = []
        for key, value in spec.args:
            if key in params:
                new_args.append((key, params[key]))
            else:
                new_args.append((key, value))

        # Add any new parameters
        for key, value in params.items():
            if key not in [k for k, _ in spec.args]:
                new_args.append((key, value))

        new_args = tuple(new_args)

    # Create new spec
    from dataclasses import replace
    new_spec = replace(spec, args=new_args)

    # Create new workflow with updated spec
    new_wf = replace(workflow, spec=new_spec)

    return new_wf


def finalize_workflow(workflow, best_params: Dict[str, Any]):
    """
    Finalize a workflow with the best parameters.

    Args:
        workflow: Original workflow with tune() placeholders
        best_params: Dictionary of best parameter values

    Returns:
        Finalized workflow ready for fitting

    Examples:
        >>> results = tune_grid(wf, folds, grid=grid)
        >>> best = results.select_best('rmse', maximize=False)
        >>> final_wf = finalize_workflow(wf, best)
        >>> final_fit = final_wf.fit(train_data)
    """
    return _update_workflow_params(workflow, best_params)
