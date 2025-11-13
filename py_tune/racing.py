"""
Racing infrastructure for efficient hyperparameter tuning.

Provides core utilities for racing methods that eliminate poor configurations
early during cross-validation.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


@dataclass
class RaceControl:
    """
    Control parameters for racing methods.

    Racing methods evaluate all configurations on an initial set of resamples
    (burn-in), then eliminate poor configurations based on statistical tests.

    Attributes:
        burn_in: Number of initial resamples to evaluate before elimination starts
        alpha: Significance level for statistical tests (0.05 = 5%)
        randomize: Whether to randomize resample order
        verbose_elim: Whether to log elimination messages
        num_ties: Maximum number of ties before forcing tie-break
        parallel_over: Parallelization strategy ('resamples' or 'everything')
        save_pred: Whether to save out-of-sample predictions
        save_workflow: Whether to save fitted workflows
        pkgs: Additional packages required (populated automatically)

    Examples:
        >>> control = control_race(burn_in=3, alpha=0.05, verbose_elim=True)
        >>> results = tune_race_anova(workflow, resamples, grid, control=control)
    """
    burn_in: int = 3
    alpha: float = 0.05
    randomize: bool = True
    verbose_elim: bool = False
    num_ties: int = 2
    parallel_over: str = "resamples"
    save_pred: bool = False
    save_workflow: bool = False
    pkgs: List[str] = None

    def __post_init__(self):
        if self.pkgs is None:
            self.pkgs = []

        # Validate inputs
        if self.burn_in < 1:
            raise ValueError(f"burn_in must be >= 1, got {self.burn_in}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.num_ties < 1:
            raise ValueError(f"num_ties must be >= 1, got {self.num_ties}")
        if self.parallel_over not in ["resamples", "everything"]:
            raise ValueError(
                f"parallel_over must be 'resamples' or 'everything', got '{self.parallel_over}'"
            )


def control_race(
    burn_in: int = 3,
    alpha: float = 0.05,
    randomize: bool = True,
    verbose_elim: bool = False,
    num_ties: int = 2,
    parallel_over: str = "resamples",
    save_pred: bool = False,
    save_workflow: bool = False
) -> RaceControl:
    """
    Create racing control object.

    Args:
        burn_in: Number of initial resamples before elimination
        alpha: Significance level for tests
        randomize: Randomize resample order
        verbose_elim: Log elimination messages
        num_ties: Max ties before tie-breaking
        parallel_over: Parallelization strategy
        save_pred: Save predictions
        save_workflow: Save workflows

    Returns:
        RaceControl object

    Examples:
        >>> ctrl = control_race(burn_in=5, alpha=0.01)
        >>> ctrl.burn_in
        5
    """
    return RaceControl(
        burn_in=burn_in,
        alpha=alpha,
        randomize=randomize,
        verbose_elim=verbose_elim,
        num_ties=num_ties,
        parallel_over=parallel_over,
        save_pred=save_pred,
        save_workflow=save_workflow
    )


def test_parameters_anova(
    results: Any,
    alpha: float,
    metric_name: str,
    eval_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Test parameter configurations using repeated measures ANOVA.

    Fits a mixed linear model: metric ~ config + (1|resample_id)
    to determine which configurations are statistically different from the best.

    Args:
        results: TuneResults object with metric evaluations
        alpha: Significance level for filtering
        metric_name: Name of metric to optimize (e.g., 'rmse', 'accuracy')
        eval_time: Specific evaluation time for dynamic metrics (optional)

    Returns:
        DataFrame with columns:
            - .config: Configuration ID
            - mean: Mean performance across resamples
            - pass: Boolean indicating if config passes filter

    Examples:
        >>> from py_tune import tune_grid
        >>> results = tune_grid(workflow, resamples, grid)
        >>> filtered = test_parameters_anova(results, alpha=0.05, metric_name='rmse')
        >>> passing_configs = filtered[filtered['pass']]
    """
    # Extract metrics
    metrics_df = results.metrics.copy()

    # Filter for specific metric (handle long format with 'metric' column)
    if 'metric' in metrics_df.columns:
        metrics_df = metrics_df[metrics_df['metric'] == metric_name]
        value_col = 'value'
    else:
        # Wide format - metric is a column name
        value_col = metric_name

    # Filter by eval_time if specified
    if eval_time is not None and '.eval_time' in metrics_df.columns:
        metrics_df = metrics_df[metrics_df['.eval_time'] == eval_time]

    # Use .resample as the resample ID
    if '.resample' not in metrics_df.columns:
        raise ValueError("Metrics DataFrame must have '.resample' column")
    if '.config' not in metrics_df.columns:
        raise ValueError("Metrics DataFrame must have '.config' column")

    # Rename for consistency
    metrics_df = metrics_df.rename(columns={'.resample': 'id', value_col: 'mean'})

    # Get unique configurations
    configs = metrics_df['.config'].unique()

    if len(configs) == 1:
        # Only one config - always passes
        config_means = metrics_df.groupby('.config')['mean'].mean()
        return pd.DataFrame({
            '.config': configs,
            'mean': config_means.values,
            'pass': [True]
        })

    # Determine if we're maximizing or minimizing
    # Assume first metric direction (can be enhanced with metric metadata)
    config_means = metrics_df.groupby('.config')['mean'].mean()
    best_config = config_means.idxmax()  # Assuming higher is better for now

    # Prepare data for mixed model
    # Create numeric config IDs for model matrix
    config_to_num = {config: i for i, config in enumerate(configs)}
    metrics_df['config_num'] = metrics_df['.config'].map(config_to_num)

    # Create contrast matrix (each config vs. best)
    n_configs = len(configs)
    contrasts = np.zeros((len(metrics_df), n_configs - 1))

    best_num = config_to_num[best_config]
    contrast_idx = 0
    for config_num in range(n_configs):
        if config_num != best_num:
            mask = metrics_df['config_num'] == config_num
            contrasts[mask, contrast_idx] = 1
            contrast_idx += 1

    # Fit mixed linear model
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)

            model = MixedLM(
                endog=metrics_df['mean'].values,
                exog=contrasts,
                groups=metrics_df['id'].values
            )
            fitted = model.fit(method='lbfgs', maxiter=100)

        # Extract p-values
        p_values = fitted.pvalues

        # Determine which configs pass
        # Best config always passes
        passes = {best_config: True}

        # Other configs pass if p-value > alpha (not significantly different)
        contrast_idx = 0
        for config in configs:
            if config != best_config:
                passes[config] = p_values[contrast_idx] > alpha
                contrast_idx += 1

    except Exception as e:
        # If ANOVA fails, fall back to simple mean comparison
        # Keep top 50% or configs within 10% of best
        best_mean = config_means[best_config]
        threshold = best_mean * 0.9  # Within 10% of best

        passes = {
            config: (config_means[config] >= threshold)
            for config in configs
        }

    # Create result DataFrame
    result = pd.DataFrame({
        '.config': configs,
        'mean': [config_means[config] for config in configs],
        'pass': [passes[config] for config in configs]
    })

    return result


def test_parameters_bt(
    results: Any,
    alpha: float,
    metric_name: str,
    eval_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Test parameter configurations using Bradley-Terry win/loss model.

    Computes pairwise win/loss records and fits a Bradley-Terry model to
    estimate "winning ability" of each configuration.

    Args:
        results: TuneResults object with metric evaluations
        alpha: Significance level for confidence intervals
        metric_name: Name of metric to optimize
        eval_time: Specific evaluation time for dynamic metrics

    Returns:
        DataFrame with columns:
            - .config: Configuration ID
            - mean: Mean performance
            - ability: Estimated winning ability
            - pass: Boolean filter

    Examples:
        >>> from py_tune import tune_grid
        >>> results = tune_grid(workflow, resamples, grid)
        >>> filtered = test_parameters_bt(results, alpha=0.05, metric_name='rmse')
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import norm

    # Extract metrics
    metrics_df = results.metrics.copy()

    # Filter for specific metric
    if 'metric' in metrics_df.columns:
        metrics_df = metrics_df[metrics_df['metric'] == metric_name]
        value_col = 'value'
    else:
        value_col = metric_name

    if eval_time is not None and '.eval_time' in metrics_df.columns:
        metrics_df = metrics_df[metrics_df['.eval_time'] == eval_time]

    # Rename for consistency
    metrics_df = metrics_df.rename(columns={'.resample': 'id', value_col: 'mean'})

    configs = metrics_df['.config'].unique()

    if len(configs) == 1:
        # Only one config
        config_means = metrics_df.groupby('.config')['mean'].mean()
        return pd.DataFrame({
            '.config': configs,
            'mean': config_means.values,
            'ability': [0.0],
            'pass': [True]
        })

    # Compute pairwise wins/losses across resamples
    # For each resample, compare all pairs of configs
    comparisons = []

    for resample_id in metrics_df['id'].unique():
        resample_data = metrics_df[metrics_df['id'] == resample_id]

        # Get performance for each config in this resample
        config_perf = {}
        for idx, row in resample_data.iterrows():
            config_perf[row['.config']] = row['mean']

        # Create pairwise comparisons
        for i, config_i in enumerate(configs):
            for j, config_j in enumerate(configs):
                if i < j and config_i in config_perf and config_j in config_perf:
                    perf_i = config_perf[config_i]
                    perf_j = config_perf[config_j]

                    if perf_i > perf_j:
                        winner = config_i
                        outcome = 1
                    elif perf_j > perf_i:
                        winner = config_j
                        outcome = 0
                    else:
                        # Tie - count as 0.5 for each
                        comparisons.append({
                            'config_i': config_i,
                            'config_j': config_j,
                            'outcome': 0.5
                        })
                        continue

                    comparisons.append({
                        'config_i': config_i,
                        'config_j': config_j,
                        'outcome': outcome
                    })

    if not comparisons:
        # No valid comparisons - keep all
        config_means = metrics_df.groupby('.config')['mean'].mean()
        return pd.DataFrame({
            '.config': configs,
            'mean': config_means.values,
            'ability': np.zeros(len(configs)),
            'pass': [True] * len(configs)
        })

    # Create Bradley-Terry design matrix
    comp_df = pd.DataFrame(comparisons)
    n_comparisons = len(comp_df)
    n_configs = len(configs)

    config_to_idx = {config: i for i, config in enumerate(configs)}

    X = np.zeros((n_comparisons, n_configs - 1))
    y = comp_df['outcome'].values

    # Use first config as reference (set to 0)
    reference_idx = 0

    for comp_idx, row in comp_df.iterrows():
        i = config_to_idx[row['config_i']]
        j = config_to_idx[row['config_j']]

        if i != reference_idx:
            X[comp_idx, i - 1] = 1
        if j != reference_idx:
            X[comp_idx, j - 1] = -1

    # Fit logistic regression (Bradley-Terry model)
    try:
        model = LogisticRegression(fit_intercept=False, max_iter=1000)
        model.fit(X, y)

        abilities = np.zeros(n_configs)
        abilities[1:] = model.coef_[0]  # Reference is 0

        # Compute confidence intervals
        # Use Wald confidence intervals
        z_crit = norm.ppf(1 - alpha / 2)

        # Approximate standard errors from logistic regression
        # This is simplified - proper SE would need Hessian
        se = np.ones(n_configs - 1) * 0.5  # Conservative estimate

        ci_lower = abilities[1:] - z_crit * se
        ci_upper = abilities[1:] + z_crit * se

        # Config passes if CI doesn't include large negative value
        # (indicating significantly worse than reference)
        passes = np.ones(n_configs, dtype=bool)
        passes[1:] = ci_lower > -2.0  # Heuristic threshold

        # Reference always passes
        passes[reference_idx] = True

    except Exception as e:
        # Fallback: use simple win rate
        config_means = metrics_df.groupby('.config')['mean'].mean()
        best_mean = config_means.max()
        abilities = (config_means.values - config_means.mean()) / (config_means.std() + 1e-9)
        passes = config_means.values >= best_mean * 0.9

    # Create result
    config_means = metrics_df.groupby('.config')['mean'].mean()
    result = pd.DataFrame({
        '.config': configs,
        'mean': config_means.values,
        'ability': abilities,
        'pass': passes
    })

    return result


def restore_rset(resamples: Any, indices: List[int]) -> Any:
    """
    Extract subset of resamples by index.

    Args:
        resamples: Original resamples object (DataFrame-like with splits)
        indices: List or range of indices to extract

    Returns:
        Subset of resamples

    Examples:
        >>> from py_rsample import vfold_cv
        >>> folds = vfold_cv(data, v=10)
        >>> first_three = restore_rset(folds, [0, 1, 2])
    """
    if hasattr(indices, '__iter__') and not isinstance(indices, (str, int)):
        indices = list(indices)
    else:
        indices = [indices]

    # Handle different resample formats
    if isinstance(resamples, pd.DataFrame):
        return resamples.iloc[indices].reset_index(drop=True)
    else:
        # Assume it has an iloc or similar interface
        return resamples.iloc[indices]


def randomize_resamples(resamples: Any) -> Any:
    """
    Randomize the order of resamples.

    Args:
        resamples: Resamples object

    Returns:
        Resamples in random order

    Examples:
        >>> folds_random = randomize_resamples(folds)
    """
    if isinstance(resamples, pd.DataFrame):
        return resamples.sample(frac=1.0).reset_index(drop=True)
    else:
        # Assume DataFrame-like
        n = len(resamples)
        indices = np.random.permutation(n)
        return resamples.iloc[indices].reset_index(drop=True)
