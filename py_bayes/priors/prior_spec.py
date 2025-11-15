"""
Prior Specification Parser for Bayesian Models

Parses string DSL for prior distributions like "normal(0, 10)" into
dictionaries that can be passed to PyMC, Stan, or other Bayesian frameworks.

Supported distributions:
- normal(mu, sigma)
- student_t(nu, mu, sigma)
- half_cauchy(beta)
- exponential(lam)
- gamma(alpha, beta)
- beta(alpha, beta)
- uniform(lower, upper)
"""

import re
from typing import Dict, Any, Union


def parse_prior(prior_spec: str) -> Dict[str, Any]:
    """
    Parse prior specification string into dictionary.

    Args:
        prior_spec: Prior specification string (e.g., "normal(0, 10)")

    Returns:
        Dictionary with distribution name and parameters

    Examples:
        >>> parse_prior("normal(0, 10)")
        {'dist': 'normal', 'mu': 0.0, 'sigma': 10.0}

        >>> parse_prior("student_t(nu=3, mu=0, sigma=10)")
        {'dist': 'student_t', 'nu': 3.0, 'mu': 0.0, 'sigma': 10.0}

        >>> parse_prior("half_cauchy(5)")
        {'dist': 'half_cauchy', 'beta': 5.0}
    """
    # Remove whitespace
    prior_spec = prior_spec.strip()

    # Extract distribution name and parameters
    match = re.match(r'(\w+)\((.*)\)', prior_spec)
    if not match:
        raise ValueError(
            f"Invalid prior specification: '{prior_spec}'. "
            f"Expected format: 'dist_name(param1, param2, ...)'"
        )

    dist_name = match.group(1)
    params_str = match.group(2)

    # Parse parameters
    params = _parse_params(params_str)

    # Validate distribution and map to standard parameter names
    if dist_name == "normal":
        return _parse_normal(params)
    elif dist_name == "student_t":
        return _parse_student_t(params)
    elif dist_name == "half_cauchy":
        return _parse_half_cauchy(params)
    elif dist_name == "exponential":
        return _parse_exponential(params)
    elif dist_name == "gamma":
        return _parse_gamma(params)
    elif dist_name == "beta":
        return _parse_beta(params)
    elif dist_name == "uniform":
        return _parse_uniform(params)
    else:
        raise ValueError(
            f"Unsupported distribution: '{dist_name}'. "
            f"Supported: normal, student_t, half_cauchy, exponential, gamma, beta, uniform"
        )


def _parse_params(params_str: str) -> Dict[str, float]:
    """Parse parameter string into dictionary."""
    if not params_str.strip():
        return {}

    params = {}

    # Split by commas (handling nested parentheses if needed)
    param_parts = [p.strip() for p in params_str.split(',')]

    positional_idx = 0
    for part in param_parts:
        if '=' in part:
            # Named parameter
            name, value = part.split('=', 1)
            params[name.strip()] = float(value.strip())
        else:
            # Positional parameter
            params[f"_pos_{positional_idx}"] = float(part.strip())
            positional_idx += 1

    return params


def _parse_normal(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse normal distribution parameters."""
    if '_pos_0' in params and '_pos_1' in params:
        # Positional: normal(mu, sigma)
        return {
            'dist': 'normal',
            'mu': params['_pos_0'],
            'sigma': params['_pos_1']
        }
    elif 'mu' in params and 'sigma' in params:
        # Named: normal(mu=0, sigma=10)
        return {
            'dist': 'normal',
            'mu': params['mu'],
            'sigma': params['sigma']
        }
    else:
        raise ValueError(
            "Normal distribution requires 2 parameters: mu, sigma. "
            f"Got: {params}"
        )


def _parse_student_t(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse Student-t distribution parameters."""
    if '_pos_0' in params and '_pos_1' in params and '_pos_2' in params:
        # Positional: student_t(nu, mu, sigma)
        return {
            'dist': 'student_t',
            'nu': params['_pos_0'],
            'mu': params['_pos_1'],
            'sigma': params['_pos_2']
        }
    elif 'nu' in params and 'mu' in params and 'sigma' in params:
        # Named: student_t(nu=3, mu=0, sigma=10)
        return {
            'dist': 'student_t',
            'nu': params['nu'],
            'mu': params['mu'],
            'sigma': params['sigma']
        }
    else:
        raise ValueError(
            "Student-t distribution requires 3 parameters: nu, mu, sigma. "
            f"Got: {params}"
        )


def _parse_half_cauchy(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse half-Cauchy distribution parameters."""
    if '_pos_0' in params:
        # Positional: half_cauchy(beta)
        return {
            'dist': 'half_cauchy',
            'beta': params['_pos_0']
        }
    elif 'beta' in params:
        # Named: half_cauchy(beta=5)
        return {
            'dist': 'half_cauchy',
            'beta': params['beta']
        }
    else:
        raise ValueError(
            "Half-Cauchy distribution requires 1 parameter: beta. "
            f"Got: {params}"
        )


def _parse_exponential(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse exponential distribution parameters."""
    if '_pos_0' in params:
        # Positional: exponential(lam)
        return {
            'dist': 'exponential',
            'lam': params['_pos_0']
        }
    elif 'lam' in params:
        # Named: exponential(lam=1)
        return {
            'dist': 'exponential',
            'lam': params['lam']
        }
    else:
        raise ValueError(
            "Exponential distribution requires 1 parameter: lam. "
            f"Got: {params}"
        )


def _parse_gamma(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse gamma distribution parameters."""
    if '_pos_0' in params and '_pos_1' in params:
        # Positional: gamma(alpha, beta)
        return {
            'dist': 'gamma',
            'alpha': params['_pos_0'],
            'beta': params['_pos_1']
        }
    elif 'alpha' in params and 'beta' in params:
        # Named: gamma(alpha=2, beta=1)
        return {
            'dist': 'gamma',
            'alpha': params['alpha'],
            'beta': params['beta']
        }
    else:
        raise ValueError(
            "Gamma distribution requires 2 parameters: alpha, beta. "
            f"Got: {params}"
        )


def _parse_beta(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse beta distribution parameters."""
    if '_pos_0' in params and '_pos_1' in params:
        # Positional: beta(alpha, beta)
        return {
            'dist': 'beta',
            'alpha': params['_pos_0'],
            'beta': params['_pos_1']
        }
    elif 'alpha' in params and 'beta' in params:
        # Named: beta(alpha=2, beta=2)
        return {
            'dist': 'beta',
            'alpha': params['alpha'],
            'beta': params['beta']
        }
    else:
        raise ValueError(
            "Beta distribution requires 2 parameters: alpha, beta. "
            f"Got: {params}"
        )


def _parse_uniform(params: Dict[str, float]) -> Dict[str, Any]:
    """Parse uniform distribution parameters."""
    if '_pos_0' in params and '_pos_1' in params:
        # Positional: uniform(lower, upper)
        return {
            'dist': 'uniform',
            'lower': params['_pos_0'],
            'upper': params['_pos_1']
        }
    elif 'lower' in params and 'upper' in params:
        # Named: uniform(lower=0, upper=10)
        return {
            'dist': 'uniform',
            'lower': params['lower'],
            'upper': params['upper']
        }
    else:
        raise ValueError(
            "Uniform distribution requires 2 parameters: lower, upper. "
            f"Got: {params}"
        )


def get_default_priors() -> Dict[str, str]:
    """
    Get default prior specifications for linear regression.

    Returns:
        Dictionary with default priors for intercept, coefs, and sigma
    """
    return {
        'prior_intercept': 'normal(0, 10)',
        'prior_coefs': 'normal(0, 5)',
        'prior_sigma': 'half_cauchy(5)'
    }
