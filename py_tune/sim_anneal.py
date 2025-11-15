"""
Simulated annealing for hyperparameter optimization.

Provides tune_sim_anneal() for sequential parameter search using
simulated annealing with configurable cooling schedules.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
import warnings

from .tune import tune_grid, TuneResults, grid_regular


@dataclass
class SimAnnealControl:
    """
    Control parameters for simulated annealing.

    Parameters
    ----------
    initial_temp : float
        Starting temperature for annealing (default: 1.0)
    cooling_schedule : str
        Cooling schedule: 'exponential', 'linear', 'logarithmic' (default: 'exponential')
    cooling_rate : float
        Rate parameter for cooling schedule (default: 0.95)
        - Exponential: T = T0 * cooling_rate^iteration
        - Linear: T = T0 - cooling_rate * iteration
        - Logarithmic: T = T0 / (1 + cooling_rate * log(1 + iteration))
    max_iter : int
        Maximum number of iterations (default: 50)
    restart_after : Optional[int]
        Restart from best after N iterations without improvement (default: None)
    no_improve : int
        Stop if no improvement after N iterations (default: 20)
    verbose : bool
        Print progress messages (default: False)
    save_pred : bool
        Save predictions (default: False)
    save_workflow : bool
        Save workflow (default: False)

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    initial_temp: float = 1.0
    cooling_schedule: str = 'exponential'
    cooling_rate: float = 0.95
    max_iter: int = 50
    restart_after: Optional[int] = None
    no_improve: int = 20
    verbose: bool = False
    save_pred: bool = False
    save_workflow: bool = False

    def __post_init__(self):
        """Validate control parameters."""
        if self.initial_temp <= 0:
            raise ValueError("initial_temp must be > 0")

        valid_schedules = ['exponential', 'linear', 'logarithmic']
        if self.cooling_schedule not in valid_schedules:
            raise ValueError(f"cooling_schedule must be one of {valid_schedules}")

        if self.cooling_rate <= 0:
            raise ValueError("cooling_rate must be > 0")

        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        if self.no_improve < 1:
            raise ValueError("no_improve must be >= 1")

        if self.restart_after is not None and self.restart_after < 1:
            raise ValueError("restart_after must be >= 1 or None")


def control_sim_anneal(
    initial_temp: float = 1.0,
    cooling_schedule: str = 'exponential',
    cooling_rate: float = 0.95,
    max_iter: int = 50,
    restart_after: Optional[int] = None,
    no_improve: int = 20,
    verbose: bool = False,
    save_pred: bool = False,
    save_workflow: bool = False
) -> SimAnnealControl:
    """
    Create SimAnnealControl object with validation.

    Factory function for creating simulated annealing control parameters.

    Parameters
    ----------
    initial_temp : float
        Starting temperature (default: 1.0)
    cooling_schedule : str
        Cooling schedule type (default: 'exponential')
    cooling_rate : float
        Cooling rate parameter (default: 0.95)
    max_iter : int
        Maximum iterations (default: 50)
    restart_after : Optional[int]
        Restart threshold (default: None)
    no_improve : int
        No improvement stopping criterion (default: 20)
    verbose : bool
        Verbose output (default: False)
    save_pred : bool
        Save predictions (default: False)
    save_workflow : bool
        Save workflow (default: False)

    Returns
    -------
    SimAnnealControl
        Validated control object

    Examples
    --------
    >>> ctrl = control_sim_anneal(initial_temp=2.0, max_iter=100)
    >>> ctrl.cooling_schedule
    'exponential'
    """
    return SimAnnealControl(
        initial_temp=initial_temp,
        cooling_schedule=cooling_schedule,
        cooling_rate=cooling_rate,
        max_iter=max_iter,
        restart_after=restart_after,
        no_improve=no_improve,
        verbose=verbose,
        save_pred=save_pred,
        save_workflow=save_workflow
    )


def _cool_temperature(
    temp: float,
    iteration: int,
    schedule: str,
    rate: float,
    initial_temp: float
) -> float:
    """
    Apply cooling schedule to temperature.

    Parameters
    ----------
    temp : float
        Current temperature
    iteration : int
        Current iteration number
    schedule : str
        Cooling schedule type
    rate : float
        Cooling rate
    initial_temp : float
        Initial temperature

    Returns
    -------
    float
        New temperature
    """
    if schedule == 'exponential':
        return initial_temp * (rate ** iteration)
    elif schedule == 'linear':
        new_temp = initial_temp - (rate * iteration)
        return max(new_temp, 1e-10)  # Prevent negative temperature
    elif schedule == 'logarithmic':
        return initial_temp / (1 + rate * np.log(1 + iteration))
    else:
        return temp


def _acceptance_probability(
    current_value: float,
    new_value: float,
    temperature: float,
    maximize: bool
) -> float:
    """
    Calculate acceptance probability for worse solution.

    Parameters
    ----------
    current_value : float
        Current metric value
    new_value : float
        New metric value
    temperature : float
        Current temperature
    maximize : bool
        Whether higher is better

    Returns
    -------
    float
        Acceptance probability (0 to 1)
    """
    if maximize:
        delta = new_value - current_value
    else:
        delta = current_value - new_value

    if delta > 0:
        # New is better
        return 1.0
    else:
        # New is worse - accept with probability based on temperature
        if temperature <= 0:
            return 0.0
        return np.exp(delta / temperature)


def _generate_neighbor(
    current_params: Dict[str, Any],
    param_info: Dict[str, Dict[str, Any]],
    temperature: float
) -> Dict[str, Any]:
    """
    Generate neighbor configuration by perturbing parameters.

    Parameters
    ----------
    current_params : Dict[str, Any]
        Current parameter values
    param_info : Dict[str, Dict[str, Any]]
        Parameter information with ranges
    temperature : float
        Current temperature (affects perturbation size)

    Returns
    -------
    Dict[str, Any]
        Neighbor parameter configuration
    """
    neighbor = current_params.copy()

    # Choose random parameter to perturb
    param_name = np.random.choice(list(param_info.keys()))
    info = param_info[param_name]

    # Get range
    param_range = info.get('range', (0, 1))
    min_val, max_val = param_range

    # Get transformation
    trans = info.get('trans', 'none')

    # Generate perturbation (scaled by temperature)
    if trans == 'log':
        # Work in log space
        current_val = current_params[param_name]
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_current = np.log10(current_val)

        # Perturbation proportional to temperature and range
        perturbation = np.random.normal(0, temperature * (log_max - log_min) * 0.1)
        new_log_val = log_current + perturbation

        # Clip to bounds
        new_log_val = np.clip(new_log_val, log_min, log_max)
        neighbor[param_name] = 10 ** new_log_val

    else:
        # Linear space
        current_val = current_params[param_name]

        # Perturbation proportional to temperature and range
        perturbation = np.random.normal(0, temperature * (max_val - min_val) * 0.1)
        new_val = current_val + perturbation

        # Clip to bounds
        neighbor[param_name] = np.clip(new_val, min_val, max_val)

    # Handle integer type
    if info.get('type') == 'int':
        neighbor[param_name] = int(round(neighbor[param_name]))

    return neighbor


def tune_sim_anneal(
    workflow: Any,
    resamples: Any,
    param_info: Optional[Dict[str, Dict[str, Any]]] = None,
    initial: Optional[Dict[str, Any]] = None,
    metrics: Optional[Callable] = None,
    control: Optional[SimAnnealControl] = None
) -> TuneResults:
    """
    Hyperparameter tuning via simulated annealing.

    Sequential optimization that explores parameter space using simulated
    annealing. Starts from an initial configuration and iteratively generates
    neighbors, accepting better configurations and worse configurations with
    probability based on temperature.

    Parameters
    ----------
    workflow : Workflow
        Workflow object to tune
    resamples : Any
        Resampling object (e.g., vfold_cv, time_series_cv)
    param_info : Optional[Dict[str, Dict[str, Any]]]
        Parameter information with ranges and transformations
        Example: {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}
    initial : Optional[Dict[str, Any]]
        Initial parameter configuration (default: random)
    metrics : Optional[Callable]
        Metric function or metric_set (default: rmse)
    control : Optional[SimAnnealControl]
        Control parameters (default: SimAnnealControl())

    Returns
    -------
    TuneResults
        Results object with method='sim_anneal'
        - metrics: DataFrame with all evaluated configurations
        - grid: DataFrame with parameter values
        - Use show_best(), select_best() to find optimal configuration

    Examples
    --------
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg
    >>> from py_rsample import vfold_cv
    >>> from py_tune import tune_sim_anneal, control_sim_anneal
    >>>
    >>> wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    >>> folds = vfold_cv(data, v=5)
    >>>
    >>> # Configure simulated annealing
    >>> ctrl = control_sim_anneal(
    ...     initial_temp=2.0,
    ...     max_iter=100,
    ...     cooling_schedule='exponential'
    ... )
    >>>
    >>> # Run optimization
    >>> results = tune_sim_anneal(
    ...     wf, folds,
    ...     param_info={'penalty': {'range': (0.001, 1.0), 'trans': 'log'}},
    ...     control=ctrl
    ... )
    >>>
    >>> # Get best configuration
    >>> best = results.select_best('rmse', maximize=False)
    """
    if control is None:
        control = SimAnnealControl()

    if param_info is None:
        raise ValueError("param_info is required for simulated annealing")

    # Generate initial configuration
    if initial is None:
        # Random initialization
        initial = {}
        for param_name, info in param_info.items():
            param_range = info.get('range', (0, 1))
            min_val, max_val = param_range
            trans = info.get('trans', 'none')

            if trans == 'log':
                log_val = np.random.uniform(np.log10(min_val), np.log10(max_val))
                initial[param_name] = 10 ** log_val
            else:
                initial[param_name] = np.random.uniform(min_val, max_val)

            # Handle integer type
            if info.get('type') == 'int':
                initial[param_name] = int(round(initial[param_name]))

    # Initialize tracking
    current_params = initial.copy()
    best_params = initial.copy()

    all_metrics = []
    all_configs = []

    # Evaluate initial configuration
    config_id = 'config_001'
    initial_grid = pd.DataFrame([{**initial, '.config': config_id}])

    initial_results = tune_grid(
        workflow, resamples,
        grid=initial_grid,
        metrics=metrics
    )

    all_metrics.append(initial_results.metrics)
    all_configs.append(initial_grid)

    # Get initial metric value (use first metric if multiple)
    metric_data = initial_results.metrics

    # Handle long format (has 'metric' column)
    if 'metric' in metric_data.columns:
        # Get first metric
        opt_metric = metric_data.iloc[0]['metric']
        metric_vals = metric_data[metric_data['metric'] == opt_metric]
        current_value = metric_vals['value'].mean()
    else:
        # Wide format - metric is already aggregated
        opt_metric = [col for col in metric_data.columns if col not in ['.config', '.resample']][0]
        current_value = metric_data[opt_metric].mean()

    maximize = False  # Assume lower is better by default
    best_value = current_value

    if control.verbose:
        print(f"Initial configuration: {opt_metric} = {current_value:.4f}")
        print(f"Starting simulated annealing with max_iter={control.max_iter}")

    # Simulated annealing loop
    temperature = control.initial_temp
    iterations_without_improvement = 0
    config_counter = 2

    for iteration in range(1, control.max_iter + 1):
        # Generate neighbor
        neighbor_params = _generate_neighbor(current_params, param_info, temperature)

        # Evaluate neighbor
        config_id = f'config_{config_counter:03d}'
        neighbor_grid = pd.DataFrame([{**neighbor_params, '.config': config_id}])

        neighbor_results = tune_grid(
            workflow, resamples,
            grid=neighbor_grid,
            metrics=metrics
        )

        all_metrics.append(neighbor_results.metrics)
        all_configs.append(neighbor_grid)
        config_counter += 1

        # Get neighbor metric value
        neighbor_data = neighbor_results.metrics
        if 'metric' in neighbor_data.columns:
            neighbor_vals = neighbor_data[neighbor_data['metric'] == opt_metric]
            neighbor_value = neighbor_vals['value'].mean()
        else:
            neighbor_value = neighbor_data[opt_metric].mean()

        # Acceptance decision
        accept_prob = _acceptance_probability(current_value, neighbor_value, temperature, maximize)
        accept = np.random.random() < accept_prob

        if accept:
            current_params = neighbor_params
            current_value = neighbor_value

            # Update best if improved
            if (maximize and neighbor_value > best_value) or \
               (not maximize and neighbor_value < best_value):
                best_params = neighbor_params.copy()
                best_value = neighbor_value
                iterations_without_improvement = 0

                if control.verbose:
                    print(f"Iter {iteration}: New best {opt_metric} = {best_value:.4f} (T={temperature:.4f})")
            else:
                iterations_without_improvement += 1
        else:
            iterations_without_improvement += 1

        # Cool temperature
        temperature = _cool_temperature(
            temperature, iteration,
            control.cooling_schedule,
            control.cooling_rate,
            control.initial_temp
        )

        # Check stopping criteria
        if iterations_without_improvement >= control.no_improve:
            if control.verbose:
                print(f"Stopping: No improvement for {control.no_improve} iterations")
            break

        # Optional restart
        if control.restart_after is not None and \
           iterations_without_improvement >= control.restart_after:
            if control.verbose:
                print(f"Restarting from best configuration")
            current_params = best_params.copy()
            current_value = best_value
            iterations_without_improvement = 0
            temperature = control.initial_temp  # Reset temperature

    if control.verbose:
        print(f"Simulated annealing complete: {opt_metric} = {best_value:.4f}")
        print(f"Evaluated {config_counter - 1} configurations")

    # Combine all results
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_grid = pd.concat(all_configs, ignore_index=True)

    return TuneResults(
        metrics=combined_metrics,
        predictions=pd.DataFrame(),
        workflow=workflow if control.save_workflow else None,
        resamples=resamples,
        grid=combined_grid,
        method='sim_anneal'
    )
