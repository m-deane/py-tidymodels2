"""
Bayesian Poisson Regression for count data.

Provides Bayesian inference for Poisson GLM using PyMC backend.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from py_parsnip.model_spec import ModelSpec


def poisson_bayes(
    prior_intercept: str = "normal(0, 10)",
    prior_coefs: str = "normal(0, 5)",
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    engine: str = "pymc"
) -> ModelSpec:
    """
    Bayesian Poisson regression for count data.

    Uses log link: log(λ) = α + βX
    where λ is the expected count (Poisson rate parameter).

    Args:
        prior_intercept: Prior specification for intercept (string DSL)
        prior_coefs: Prior specification for coefficients (string DSL or dict)
        draws: Number of MCMC posterior draws per chain
        tune: Number of tuning/warmup steps
        chains: Number of MCMC chains
        target_accept: Target acceptance probability for NUTS sampler
        random_seed: Random seed for reproducibility
        progressbar: Show MCMC sampling progress bar
        engine: Backend engine ("pymc" only currently supported)

    Returns:
        ModelSpec configured for Bayesian Poisson regression

    Examples:
        >>> # Basic Poisson regression
        >>> spec = poisson_bayes()
        >>> fit = spec.fit(data, "count ~ x1 + x2")
        >>> predictions = fit.predict(test_data, type="numeric")

        >>> # Custom priors
        >>> spec = poisson_bayes(
        ...     prior_intercept="normal(0, 5)",
        ...     prior_coefs="student_t(nu=3, mu=0, sigma=2)"
        ... )

        >>> # Per-coefficient priors
        >>> spec = poisson_bayes(
        ...     prior_coefs={
        ...         "x1": "normal(0, 10)",  # Weakly informative
        ...         "x2": "normal(0, 1)"    # More informative
        ...     }
        ... )

        >>> # Prediction types
        >>> fit.predict(test_data, type="numeric")     # Expected counts
        >>> fit.predict(test_data, type="conf_int")    # Credible intervals
        >>> fit.predict(test_data, type="posterior")   # Posterior samples
        >>> fit.predict(test_data, type="predictive")  # Predictive samples with Poisson noise
    """
    args = {
        "prior_intercept": prior_intercept,
        "prior_coefs": prior_coefs,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "progressbar": progressbar
    }

    return ModelSpec(
        model_type="poisson_bayes",
        engine=engine,
        mode="regression",
        args=args
    )
