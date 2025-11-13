"""
Efficient grid search via racing with ANOVA models.

Racing eliminates poor parameter configurations early during cross-validation
using repeated measures ANOVA to identify statistically inferior configurations.
"""

from typing import Optional, Union, Dict, Any, Callable
import pandas as pd
import numpy as np

from .racing import RaceControl, control_race, test_parameters_anova, restore_rset, randomize_resamples
from .tune import tune_grid, TuneResults


def tune_race_anova(
    workflow,
    resamples,
    grid: Optional[Union[int, pd.DataFrame]] = None,
    metrics: Optional[Callable] = None,
    param_info: Optional[Dict[str, Dict[str, Any]]] = None,
    eval_time: Optional[float] = None,
    control: Optional[RaceControl] = None
) -> TuneResults:
    """
    Efficient grid search via racing with ANOVA statistical tests.

    Racing methods evaluate all parameter configurations on an initial set of
    resamples (burn-in period), then use repeated measures ANOVA to identify
    and eliminate configurations that are statistically worse than the best.

    Algorithm:
    1. Evaluate all configs on burn_in resamples (e.g., first 3 CV folds)
    2. Fit ANOVA model: metric ~ config + (1|resample)
    3. Eliminate configs with p-value <= alpha (significantly worse than best)
    4. Evaluate remaining configs on next resample
    5. Repeat steps 2-4 until one config remains or all resamples used

    Args:
        workflow: Workflow object with tune() parameters
        resamples: Cross-validation resamples (from py_rsample)
        grid: Grid size (int) or explicit grid (DataFrame)
        metrics: Metric set or list of metrics (from py_yardstick)
        param_info: Parameter specifications (required if grid is int)
        eval_time: Evaluation time for dynamic metrics (optional)
        control: Racing control parameters (RaceControl object)

    Returns:
        TuneResults object with metrics and elimination history

    Examples:
        >>> from py_tune import tune_race_anova, control_race
        >>> from py_workflows import workflow
        >>> from py_parsnip import linear_reg
        >>> from py_rsample import vfold_cv
        >>> from py_yardstick import metric_set, rmse, mae
        >>>
        >>> # Create workflow with tunable parameters
        >>> from py_tune import tune
        >>> spec = linear_reg(penalty=tune(), mixture=tune())
        >>> wf = workflow().add_formula("y ~ .").add_model(spec)
        >>>
        >>> # Define parameter space
        >>> param_info = {
        ...     'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
        ...     'mixture': {'range': (0, 1)}
        ... }
        >>>
        >>> # Create resamples
        >>> folds = vfold_cv(train_data, v=10)
        >>>
        >>> # Run racing ANOVA
        >>> ctrl = control_race(burn_in=3, alpha=0.05, verbose_elim=True)
        >>> results = tune_race_anova(
        ...     wf, folds, param_info=param_info, grid=20,
        ...     metrics=metric_set(rmse, mae), control=ctrl
        ... )
        >>>
        >>> # View best configurations (only fully evaluated ones)
        >>> results.show_best('rmse', n=5)

    References:
        Kuhn, M. (2014). "Futility Analysis in the Cross-Validation of Machine
        Learning Models." arXiv:1405.6974.
    """
    # Initialize control
    if control is None:
        control = control_race()

    # Validate inputs
    n_resamples = len(resamples)
    if n_resamples <= control.burn_in:
        raise ValueError(
            f"Number of resamples ({n_resamples}) must be greater than "
            f"burn_in ({control.burn_in})"
        )

    # Randomize resample order if requested
    if control.randomize:
        resamples = randomize_resamples(resamples)

    # Add order column
    if isinstance(resamples, pd.DataFrame):
        resamples = resamples.copy()
        resamples['.order'] = range(len(resamples))

    # Initialize with burn-in resamples
    burn_in_resamples = restore_rset(resamples, list(range(control.burn_in)))

    if control.verbose_elim:
        print(f"ℹ Evaluating all configurations on {control.burn_in} burn-in resamples...")

    # Evaluate all configs on burn-in resamples
    results = tune_grid(
        workflow=workflow,
        resamples=burn_in_resamples,
        grid=grid,
        metrics=metrics,
        param_info=param_info,
        control={'save_pred': control.save_pred}
    )

    # Determine optimization metric
    # Use first metric if multiple provided
    if results.metrics.empty:
        raise ValueError("No metrics calculated - check workflow and data")

    if 'metric' in results.metrics.columns:
        # Long format
        metric_names = results.metrics['metric'].unique()
        opt_metric_name = metric_names[0]
    else:
        # Wide format - use first metric column
        metric_cols = [c for c in results.metrics.columns
                       if c not in ['.config', '.resample', 'id']]
        opt_metric_name = metric_cols[0]

    # Determine if maximizing or minimizing
    # Common minimization metrics
    minimize_metrics = {'rmse', 'mae', 'mape', 'smape', 'rse', 'log_loss', 'brier_score'}
    maximize = opt_metric_name.lower() not in minimize_metrics

    if control.verbose_elim:
        direction = "maximizing" if maximize else "minimizing"
        print(f"ℹ Optimizing metric: {opt_metric_name} ({direction})")

    # Track history
    n_grid = len(results.grid)
    num_ties = 0

    # Racing loop - iterate through remaining resamples
    for resample_idx in range(control.burn_in, n_resamples):
        # Test current results with ANOVA
        filter_results = test_parameters_anova(
            results,
            alpha=control.alpha,
            metric_name=opt_metric_name,
            eval_time=eval_time
        )

        # Check for ties
        n_passing = filter_results['pass'].sum()

        if n_passing == 2:
            num_ties += 1
        else:
            num_ties = 0  # Reset tie counter

        # Determine which configs to continue evaluating
        passing_configs = filter_results[filter_results['pass']]

        if control.verbose_elim:
            n_eliminated = n_grid - len(passing_configs)
            if n_eliminated > 0:
                print(
                    f"✓ Resample {resample_idx + 1}/{n_resamples}: "
                    f"Eliminated {n_eliminated} configs, "
                    f"{len(passing_configs)} remaining"
                )

        # Check if we can stop early
        if len(passing_configs) == 1:
            if control.verbose_elim:
                print(f"✓ Racing complete: 1 configuration remaining")

            # Evaluate final config on all remaining resamples
            final_config = passing_configs['.config'].iloc[0]
            final_grid = results.grid[results.grid['.config'] == final_config]

            remaining_resamples = restore_rset(
                resamples,
                list(range(resample_idx, n_resamples))
            )

            final_results = tune_grid(
                workflow=workflow,
                resamples=remaining_resamples,
                grid=final_grid,
                metrics=metrics,
                param_info=None,  # Grid already has values
                control={'save_pred': control.save_pred}
            )

            # Combine with existing results
            results = _combine_results(results, final_results)
            break

        # Extract grid for passing configs
        new_grid = results.grid[results.grid['.config'].isin(passing_configs['.config'])]

        # Evaluate passing configs on next resample
        next_resample = restore_rset(resamples, [resample_idx])

        next_results = tune_grid(
            workflow=workflow,
            resamples=next_resample,
            grid=new_grid,
            metrics=metrics,
            param_info=None,
            control={'save_pred': control.save_pred}
        )

        # Combine results
        results = _combine_results(results, next_results)

        # Tie-breaking: if stuck with 2 configs for too long, keep both
        if n_passing == 2 and num_ties >= control.num_ties:
            if control.verbose_elim:
                print(
                    f"ℹ Tie detected for {num_ties} iterations - "
                    f"evaluating both configs on remaining resamples"
                )

            # Evaluate both on all remaining resamples
            remaining_resamples = restore_rset(
                resamples,
                list(range(resample_idx + 1, n_resamples))
            )

            if len(remaining_resamples) > 0:
                tie_results = tune_grid(
                    workflow=workflow,
                    resamples=remaining_resamples,
                    grid=new_grid,
                    metrics=metrics,
                    param_info=None,
                    control={'save_pred': control.save_pred}
                )

                results = _combine_results(results, tie_results)

            break

    if control.verbose_elim:
        n_final = len(filter_results[filter_results['pass']])
        print(f"✓ Racing complete: {n_final} configurations fully evaluated")

    # Add racing metadata to results
    results.method = 'race_anova'

    return results


def _combine_results(results1: TuneResults, results2: TuneResults) -> TuneResults:
    """
    Combine two TuneResults objects.

    Args:
        results1: First results object
        results2: Second results object

    Returns:
        Combined TuneResults
    """
    # Combine metrics
    combined_metrics = pd.concat(
        [results1.metrics, results2.metrics],
        ignore_index=True
    )

    # Combine predictions
    if not results1.predictions.empty and not results2.predictions.empty:
        combined_predictions = pd.concat(
            [results1.predictions, results2.predictions],
            ignore_index=True
        )
    elif not results1.predictions.empty:
        combined_predictions = results1.predictions
    else:
        combined_predictions = results2.predictions

    # Grid remains the same (configurations don't change)
    combined_grid = results1.grid

    return TuneResults(
        metrics=combined_metrics,
        predictions=combined_predictions,
        workflow=results1.workflow,
        resamples=results1.resamples,
        grid=combined_grid
    )
