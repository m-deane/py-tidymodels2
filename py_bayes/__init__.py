"""
py_bayes: Bayesian Modeling Framework for py-tidymodels

Provides Bayesian inference capabilities through PyMC backend:
- Flexible prior specifications via string DSL
- MCMC sampling with comprehensive diagnostics
- Credible intervals and posterior analysis
- Integration with existing ModelSpec pattern
- Hierarchical models for grouped/panel data
- GLM support (Poisson, Logistic)
- Prior sensitivity analysis

Example:
    >>> from py_parsnip import linear_reg
    >>> from py_bayes.diagnostics import check_convergence
    >>> from py_bayes.analysis import compare_priors
    >>>
    >>> # Standard Bayesian linear regression
    >>> spec = linear_reg().set_engine(
    ...     "pymc",
    ...     prior_intercept="normal(0, 10)",
    ...     prior_coefs="normal(0, 5)",
    ...     prior_sigma="half_cauchy(5)",
    ...     chains=4,
    ...     draws=2000
    ... )
    >>>
    >>> fit = spec.fit(data, "y ~ x1 + x2")
    >>> diag = check_convergence(fit)
    >>> predictions = fit.predict(test_data, type="conf_int")
    >>>
    >>> # Hierarchical model
    >>> spec = linear_reg().set_engine(
    ...     "pymc_hierarchical",
    ...     group_varying_intercept=True,
    ...     group_varying_slopes=["x1"]
    ... )
    >>> fit = spec.fit_global(data, "y ~ x1 + x2", group_col="store_id")
    >>>
    >>> # Prior sensitivity analysis
    >>> priors = {
    ...     "weak": {"prior_coefs": "normal(0, 10)"},
    ...     "strong": {"prior_coefs": "normal(0, 1)"}
    ... }
    >>> results = compare_priors(spec, data, "y ~ x1 + x2", priors)
"""

# Import engines to register them
from py_bayes.engines import (
    PymcLinearRegEngine,
    PymcHierarchicalEngine,
    PymcPoissonEngine,
    PymcLogisticEngine
)
from py_bayes.priors import parse_prior, get_default_priors
from py_bayes.diagnostics import check_convergence
from py_bayes.analysis import compare_priors

__version__ = "0.2.0"

__all__ = [
    'PymcLinearRegEngine',
    'PymcHierarchicalEngine',
    'PymcPoissonEngine',
    'PymcLogisticEngine',
    'parse_prior',
    'get_default_priors',
    'check_convergence',
    'compare_priors',
]
