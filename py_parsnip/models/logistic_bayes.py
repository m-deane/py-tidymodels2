"""
Bayesian Logistic Regression for binary classification.

Provides Bayesian inference for logistic GLM using PyMC backend.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from py_parsnip.model_spec import ModelSpec


def logistic_bayes(
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
    Bayesian logistic regression for binary classification.

    Uses logit link: logit(p) = α + βX
    where p is the probability of the positive class.

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
        ModelSpec configured for Bayesian logistic regression

    Examples:
        >>> # Basic logistic regression
        >>> spec = logistic_bayes()
        >>> fit = spec.fit(data, "outcome ~ x1 + x2")
        >>> predictions = fit.predict(test_data, type="prob")

        >>> # Custom priors
        >>> spec = logistic_bayes(
        ...     prior_intercept="normal(0, 5)",
        ...     prior_coefs="student_t(nu=3, mu=0, sigma=2)"
        ... )

        >>> # Per-coefficient priors
        >>> spec = logistic_bayes(
        ...     prior_coefs={
        ...         "x1": "normal(0, 10)",  # Weakly informative
        ...         "x2": "normal(0, 1)"    # More informative
        ...     }
        ... )

        >>> # Prediction types
        >>> fit.predict(test_data, type="prob")       # P(outcome=1)
        >>> fit.predict(test_data, type="class")      # Binary class (0 or 1)
        >>> fit.predict(test_data, type="conf_int")   # Credible intervals for prob
        >>> fit.predict(test_data, type="posterior")  # Posterior samples of prob
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
        model_type="logistic_bayes",
        engine=engine,
        mode="classification",
        args=args
    )
