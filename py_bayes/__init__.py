"""
py_bayes: Bayesian Modeling Framework for py-tidymodels

Provides Bayesian inference capabilities through PyMC backend:
- Flexible prior specifications via string DSL
- MCMC sampling with comprehensive diagnostics
- Credible intervals and posterior analysis
- Integration with existing ModelSpec pattern

Example:
    >>> from py_parsnip import linear_reg
    >>> from py_bayes.diagnostics import check_convergence
    >>>
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
"""

# Import engines to register them
from py_bayes.engines import PymcLinearRegEngine
from py_bayes.priors import parse_prior, get_default_priors
from py_bayes.diagnostics import check_convergence

__version__ = "0.1.0"

__all__ = [
    'PymcLinearRegEngine',
    'parse_prior',
    'get_default_priors',
    'check_convergence',
]
