"""
Bayesian optimization for hyperparameter tuning.

Provides tune_bayes() for sequential parameter search using
Gaussian Process surrogate models and acquisition functions.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
import warnings

from .tune import tune_grid, TuneResults, grid_regular


@dataclass
class BayesControl:
    """
    Control parameters for Bayesian optimization.

    Parameters
    ----------
    n_initial : int
        Number of initial random samples (default: 5)
    n_iter : int
        Number of Bayesian optimization iterations (default: 25)
    acquisition : str
        Acquisition function: 'ei' (Expected Improvement), 'pi' (Probability of
        Improvement), 'ucb' (Upper Confidence Bound) (default: 'ei')
    kappa : float
        Exploration parameter for UCB (default: 2.576, 99% confidence)
    xi : float
        Exploration parameter for EI/PI (default: 0.01)
    no_improve : int
        Stop if no improvement after N iterations (default: 10)
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
    n_initial: int = 5
    n_iter: int = 25
    acquisition: str = 'ei'
    kappa: float = 2.576
    xi: float = 0.01
    no_improve: int = 10
    verbose: bool = False
    save_pred: bool = False
    save_workflow: bool = False

    def __post_init__(self):
        """Validate control parameters."""
        if self.n_initial < 1:
            raise ValueError("n_initial must be >= 1")

        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")

        valid_acquisitions = ['ei', 'pi', 'ucb']
        if self.acquisition not in valid_acquisitions:
            raise ValueError(f"acquisition must be one of {valid_acquisitions}")

        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")

        if self.xi < 0:
            raise ValueError("xi must be >= 0")

        if self.no_improve < 1:
            raise ValueError("no_improve must be >= 1")


def control_bayes(
    n_initial: int = 5,
    n_iter: int = 25,
    acquisition: str = 'ei',
    kappa: float = 2.576,
    xi: float = 0.01,
    no_improve: int = 10,
    verbose: bool = False,
    save_pred: bool = False,
    save_workflow: bool = False
) -> BayesControl:
    """
    Create BayesControl object with validation.

    Factory function for creating Bayesian optimization control parameters.

    Parameters
    ----------
    n_initial : int
        Initial random samples (default: 5)
    n_iter : int
        BO iterations (default: 25)
    acquisition : str
        Acquisition function (default: 'ei')
    kappa : float
        UCB exploration parameter (default: 2.576)
    xi : float
        EI/PI exploration parameter (default: 0.01)
    no_improve : int
        No improvement stopping criterion (default: 10)
    verbose : bool
        Verbose output (default: False)
    save_pred : bool
        Save predictions (default: False)
    save_workflow : bool
        Save workflow (default: False)

    Returns
    -------
    BayesControl
        Validated control object

    Examples
    --------
    >>> ctrl = control_bayes(n_initial=10, n_iter=50, acquisition='ucb')
    >>> ctrl.kappa
    2.576
    """
    return BayesControl(
        n_initial=n_initial,
        n_iter=n_iter,
        acquisition=acquisition,
        kappa=kappa,
        xi=xi,
        no_improve=no_improve,
        verbose=verbose,
        save_pred=save_pred,
        save_workflow=save_workflow
    )


def _normalize_params(
    params: Dict[str, Any],
    param_info: Dict[str, Dict[str, Any]]
) -> np.ndarray:
    """
    Normalize parameters to [0, 1] range.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameter values
    param_info : Dict[str, Dict[str, Any]]
        Parameter information with ranges

    Returns
    -------
    np.ndarray
        Normalized parameter vector
    """
    normalized = []
    for param_name in sorted(param_info.keys()):
        value = params[param_name]
        info = param_info[param_name]
        param_range = info.get('range', (0, 1))
        min_val, max_val = param_range
        trans = info.get('trans', 'none')

        if trans == 'log':
            # Normalize in log space
            log_val = np.log10(value)
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            norm_val = (log_val - log_min) / (log_max - log_min)
        else:
            # Linear normalization
            norm_val = (value - min_val) / (max_val - min_val)

        normalized.append(norm_val)

    return np.array(normalized)


def _denormalize_params(
    normalized: np.ndarray,
    param_info: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Denormalize parameters from [0, 1] range.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized parameter vector
    param_info : Dict[str, Dict[str, Any]]
        Parameter information with ranges

    Returns
    -------
    Dict[str, Any]
        Denormalized parameter values
    """
    params = {}
    param_names = sorted(param_info.keys())

    for i, param_name in enumerate(param_names):
        norm_val = np.clip(normalized[i], 0, 1)  # Ensure in bounds
        info = param_info[param_name]
        param_range = info.get('range', (0, 1))
        min_val, max_val = param_range
        trans = info.get('trans', 'none')

        if trans == 'log':
            # Denormalize in log space
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            log_val = log_min + norm_val * (log_max - log_min)
            value = 10 ** log_val
        else:
            # Linear denormalization
            value = min_val + norm_val * (max_val - min_val)

        # Handle integer type
        if info.get('type') == 'int':
            value = int(round(value))

        params[param_name] = value

    return params


def _expected_improvement(
    X: np.ndarray,
    gp_model: Any,
    y_best: float,
    xi: float,
    maximize: bool
) -> np.ndarray:
    """
    Calculate Expected Improvement acquisition function.

    Parameters
    ----------
    X : np.ndarray
        Points to evaluate (n_points, n_params)
    gp_model : GaussianProcessRegressor
        Fitted GP model
    y_best : float
        Best observed value so far
    xi : float
        Exploration parameter
    maximize : bool
        Whether to maximize the metric

    Returns
    -------
    np.ndarray
        EI values for each point
    """
    mu, sigma = gp_model.predict(X, return_std=True)
    mu = mu.flatten()
    sigma = sigma.flatten()

    if maximize:
        improvement = mu - y_best - xi
    else:
        improvement = y_best - mu - xi

    with np.errstate(divide='warn'):
        Z = improvement / sigma
        ei = improvement * _norm_cdf(Z) + sigma * _norm_pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def _probability_of_improvement(
    X: np.ndarray,
    gp_model: Any,
    y_best: float,
    xi: float,
    maximize: bool
) -> np.ndarray:
    """
    Calculate Probability of Improvement acquisition function.

    Parameters
    ----------
    X : np.ndarray
        Points to evaluate
    gp_model : GaussianProcessRegressor
        Fitted GP model
    y_best : float
        Best observed value
    xi : float
        Exploration parameter
    maximize : bool
        Whether to maximize

    Returns
    -------
    np.ndarray
        PI values for each point
    """
    mu, sigma = gp_model.predict(X, return_std=True)

    if maximize:
        Z = (mu - y_best - xi) / (sigma + 1e-9)
    else:
        Z = (y_best - mu - xi) / (sigma + 1e-9)

    return _norm_cdf(Z)


def _upper_confidence_bound(
    X: np.ndarray,
    gp_model: Any,
    kappa: float,
    maximize: bool
) -> np.ndarray:
    """
    Calculate Upper Confidence Bound acquisition function.

    Parameters
    ----------
    X : np.ndarray
        Points to evaluate
    gp_model : GaussianProcessRegressor
        Fitted GP model
    kappa : float
        Exploration parameter
    maximize : bool
        Whether to maximize

    Returns
    -------
    np.ndarray
        UCB values for each point
    """
    mu, sigma = gp_model.predict(X, return_std=True)

    if maximize:
        return mu + kappa * sigma
    else:
        return mu - kappa * sigma


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    from scipy.stats import norm
    return norm.cdf(x)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    from scipy.stats import norm
    return norm.pdf(x)


def _propose_location(
    acquisition_func: Callable,
    gp_model: Any,
    y_best: float,
    bounds: np.ndarray,
    n_restarts: int,
    acquisition: str,
    kappa: float,
    xi: float,
    maximize: bool
) -> np.ndarray:
    """
    Propose next sampling location by optimizing acquisition function.

    Parameters
    ----------
    acquisition_func : Callable
        Acquisition function to optimize
    gp_model : GaussianProcessRegressor
        Fitted GP model
    y_best : float
        Best observed value
    bounds : np.ndarray
        Parameter bounds (n_params, 2)
    n_restarts : int
        Number of random restarts
    acquisition : str
        Acquisition function name
    kappa : float
        UCB parameter
    xi : float
        EI/PI parameter
    maximize : bool
        Whether to maximize

    Returns
    -------
    np.ndarray
        Proposed parameter vector
    """
    from scipy.optimize import minimize

    dim = bounds.shape[0]
    min_val = None
    min_x = None

    def min_obj(X):
        # Minimize negative acquisition (maximize acquisition)
        X = X.reshape(1, -1)
        if acquisition == 'ucb':
            val = -acquisition_func(X, gp_model, kappa, maximize)
        else:
            val = -acquisition_func(X, gp_model, y_best, xi, maximize)
        return val.flatten()[0]

    # Try multiple random starts
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)

        res = minimize(
            min_obj,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if min_val is None or res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x


def tune_bayes(
    workflow: Any,
    resamples: Any,
    param_info: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics: Optional[Callable] = None,
    control: Optional[BayesControl] = None
) -> TuneResults:
    """
    Hyperparameter tuning via Bayesian optimization.

    Sequential optimization using Gaussian Process surrogate models and
    acquisition functions. Efficiently explores parameter space by balancing
    exploration (uncertain regions) and exploitation (promising regions).

    Parameters
    ----------
    workflow : Workflow
        Workflow object to tune
    resamples : Any
        Resampling object (e.g., vfold_cv, time_series_cv)
    param_info : Optional[Dict[str, Dict[str, Any]]]
        Parameter information with ranges and transformations
        Example: {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}
    metrics : Optional[Callable]
        Metric function or metric_set (default: rmse)
    control : Optional[BayesControl]
        Control parameters (default: BayesControl())

    Returns
    -------
    TuneResults
        Results object with method='bayes'
        - metrics: DataFrame with all evaluated configurations
        - grid: DataFrame with parameter values
        - Use show_best(), select_best() to find optimal configuration

    Examples
    --------
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg
    >>> from py_rsample import vfold_cv
    >>> from py_tune import tune_bayes, control_bayes
    >>>
    >>> wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    >>> folds = vfold_cv(data, v=5)
    >>>
    >>> # Configure Bayesian optimization
    >>> ctrl = control_bayes(
    ...     n_initial=10,
    ...     n_iter=40,
    ...     acquisition='ei'
    ... )
    >>>
    >>> # Run optimization
    >>> results = tune_bayes(
    ...     wf, folds,
    ...     param_info={'penalty': {'range': (0.001, 1.0), 'trans': 'log'}},
    ...     control=ctrl
    ... )
    >>>
    >>> # Get best configuration
    >>> best = results.select_best('rmse', maximize=False)
    """
    if control is None:
        control = BayesControl()

    if param_info is None:
        raise ValueError("param_info is required for Bayesian optimization")

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
    except ImportError:
        raise ImportError(
            "Bayesian optimization requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    # Initialize tracking
    all_metrics = []
    all_configs = []
    X_observed = []  # Normalized parameters
    y_observed = []  # Metric values

    n_params = len(param_info)
    param_names = sorted(param_info.keys())
    bounds = np.array([[0, 1]] * n_params)  # Normalized bounds

    # Phase 1: Initial random sampling
    if control.verbose:
        print(f"Phase 1: Evaluating {control.n_initial} initial random samples")

    for i in range(control.n_initial):
        # Generate random configuration
        params = {}
        for param_name in param_names:
            info = param_info[param_name]
            param_range = info.get('range', (0, 1))
            min_val, max_val = param_range
            trans = info.get('trans', 'none')

            if trans == 'log':
                log_val = np.random.uniform(np.log10(min_val), np.log10(max_val))
                params[param_name] = 10 ** log_val
            else:
                params[param_name] = np.random.uniform(min_val, max_val)

            if info.get('type') == 'int':
                params[param_name] = int(round(params[param_name]))

        # Evaluate configuration
        config_id = f'config_{i + 1:03d}'
        grid = pd.DataFrame([{**params, '.config': config_id}])

        results = tune_grid(workflow, resamples, grid=grid, metrics=metrics)

        all_metrics.append(results.metrics)
        all_configs.append(grid)

        # Store observations
        X_observed.append(_normalize_params(params, param_info))

        # Get metric value
        metric_data = results.metrics
        if 'metric' in metric_data.columns:
            opt_metric = metric_data.iloc[0]['metric']
            metric_vals = metric_data[metric_data['metric'] == opt_metric]
            y_val = metric_vals['value'].mean()
        else:
            opt_metric = [col for col in metric_data.columns
                         if col not in ['.config', '.resample']][0]
            y_val = metric_data[opt_metric].mean()

        y_observed.append(y_val)

    X_observed = np.array(X_observed)
    y_observed = np.array(y_observed).reshape(-1, 1)

    maximize = False  # Assume lower is better
    best_value = y_observed.min() if not maximize else y_observed.max()
    best_idx = np.argmin(y_observed) if not maximize else np.argmax(y_observed)

    if control.verbose:
        print(f"Initial best {opt_metric}: {best_value:.4f}")
        print(f"\nPhase 2: Bayesian optimization for {control.n_iter} iterations")

    # Phase 2: Bayesian optimization
    config_counter = control.n_initial + 1
    iterations_without_improvement = 0

    # Select acquisition function
    if control.acquisition == 'ei':
        acq_func = _expected_improvement
    elif control.acquisition == 'pi':
        acq_func = _probability_of_improvement
    else:  # ucb
        acq_func = _upper_confidence_bound

    for iteration in range(control.n_iter):
        # Fit Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=None
        )
        gp.fit(X_observed, y_observed)

        # Find next point by optimizing acquisition function
        current_best = y_observed.min() if not maximize else y_observed.max()

        x_next = _propose_location(
            acq_func, gp, current_best, bounds,
            n_restarts=10,
            acquisition=control.acquisition,
            kappa=control.kappa,
            xi=control.xi,
            maximize=maximize
        )

        # Denormalize and evaluate
        params_next = _denormalize_params(x_next, param_info)

        config_id = f'config_{config_counter:03d}'
        grid = pd.DataFrame([{**params_next, '.config': config_id}])

        results = tune_grid(workflow, resamples, grid=grid, metrics=metrics)

        all_metrics.append(results.metrics)
        all_configs.append(grid)
        config_counter += 1

        # Update observations
        X_observed = np.vstack([X_observed, x_next])

        metric_data = results.metrics
        if 'metric' in metric_data.columns:
            metric_vals = metric_data[metric_data['metric'] == opt_metric]
            y_next = metric_vals['value'].mean()
        else:
            y_next = metric_data[opt_metric].mean()

        y_observed = np.vstack([y_observed, y_next])

        # Check for improvement
        if (not maximize and y_next < best_value) or \
           (maximize and y_next > best_value):
            best_value = y_next
            iterations_without_improvement = 0

            if control.verbose:
                print(f"Iter {iteration + 1}: New best {opt_metric} = {best_value:.4f}")
        else:
            iterations_without_improvement += 1

        # Early stopping
        if iterations_without_improvement >= control.no_improve:
            if control.verbose:
                print(f"Stopping: No improvement for {control.no_improve} iterations")
            break

    if control.verbose:
        print(f"\nBayesian optimization complete: {opt_metric} = {best_value:.4f}")
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
        method='bayes'
    )
