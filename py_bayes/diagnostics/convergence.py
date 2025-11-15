"""
Convergence diagnostics for Bayesian models.

Checks for common MCMC convergence issues:
- Rhat (Gelman-Rubin statistic): Should be < 1.01
- ESS (Effective Sample Size): Should be > 400
- Divergences: Should be 0
"""

from typing import Dict, List, Any
import warnings
import numpy as np


def check_convergence(
    fit,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    warn: bool = True
) -> Dict[str, Any]:
    """
    Check convergence diagnostics for Bayesian model fit.

    Args:
        fit: ModelFit object with Bayesian model
        rhat_threshold: Maximum acceptable Rhat value (default: 1.01)
        ess_threshold: Minimum acceptable ESS value (default: 400)
        warn: Whether to issue warnings (default: True)

    Returns:
        Dictionary with convergence diagnostics:
            - rhat_ok: True if all Rhat < threshold
            - ess_ok: True if all ESS > threshold
            - divergences_ok: True if no divergences
            - max_rhat: Maximum Rhat across parameters
            - min_ess_bulk: Minimum bulk ESS across parameters
            - min_ess_tail: Minimum tail ESS across parameters
            - n_divergences: Number of divergent transitions
            - warnings: List of warning messages

    Example:
        >>> from py_parsnip import linear_reg
        >>> spec = linear_reg().set_engine("pymc")
        >>> fit = spec.fit(data, "y ~ x1 + x2")
        >>> from py_bayes.diagnostics import check_convergence
        >>> diag = check_convergence(fit)
        >>> if not diag['rhat_ok']:
        ...     print("Convergence issue detected!")
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError(
            "ArviZ is required for convergence diagnostics. "
            "Install with: pip install arviz>=0.16.0"
        )

    # Extract posterior samples
    if "posterior_samples" not in fit.fit_data:
        raise ValueError(
            "No posterior samples found in fit. "
            "Only Bayesian models have convergence diagnostics."
        )

    trace = fit.fit_data["posterior_samples"]

    # Compute diagnostics
    rhat = az.rhat(trace)
    ess = az.ess(trace)

    # Extract maximum Rhat (handle xarray Dataset)
    max_rhat = float(np.max(rhat.to_array().values))

    # Extract minimum ESS (both bulk and tail)
    ess_array = ess.to_array().values
    if hasattr(ess, 'data_vars') and len(list(ess.data_vars)) > 1:
        # Separate bulk and tail ESS
        min_ess_bulk = float(np.min(ess_array[0]))
        min_ess_tail = float(np.min(ess_array[1]))
    else:
        min_ess_bulk = float(np.min(ess_array))
        min_ess_tail = min_ess_bulk

    # Count divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        n_divergences = int(trace.sample_stats.diverging.sum().item())
    else:
        n_divergences = 0

    # Check thresholds
    rhat_ok = max_rhat < rhat_threshold
    ess_ok = min_ess_bulk > ess_threshold and min_ess_tail > ess_threshold
    divergences_ok = n_divergences == 0

    # Generate warnings
    warning_messages = []

    if not rhat_ok:
        msg = (
            f"Convergence issue: max Rhat = {max_rhat:.3f} (should be < {rhat_threshold}). "
            f"Consider increasing draws or tune steps."
        )
        warning_messages.append(msg)
        if warn:
            warnings.warn(msg)

    if not ess_ok:
        msg = (
            f"Low effective sample size: min ESS bulk = {min_ess_bulk:.0f}, "
            f"tail = {min_ess_tail:.0f} (should be > {ess_threshold}). "
            f"Consider increasing draws."
        )
        warning_messages.append(msg)
        if warn:
            warnings.warn(msg)

    if not divergences_ok:
        msg = (
            f"{n_divergences} divergent transitions detected. "
            f"Try increasing target_accept or reparameterizing the model."
        )
        warning_messages.append(msg)
        if warn:
            warnings.warn(msg)

    return {
        'rhat_ok': rhat_ok,
        'ess_ok': ess_ok,
        'divergences_ok': divergences_ok,
        'max_rhat': max_rhat,
        'min_ess_bulk': min_ess_bulk,
        'min_ess_tail': min_ess_tail,
        'n_divergences': n_divergences,
        'warnings': warning_messages
    }
