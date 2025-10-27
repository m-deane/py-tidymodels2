"""
Poisson Regression model specification for count data

Supports engine:
- statsmodels: GLM with Poisson family

Parameters (tidymodels naming):
- penalty: Regularization penalty (if supported)
- mixture: Mix of L1 vs L2 penalty (if supported)

Poisson regression is used for modeling count data and contingency tables.
It assumes the response variable has a Poisson distribution and models
the log of the expected count as a linear function of predictors.
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def poisson_reg(
    penalty: Optional[float] = None,
    mixture: Optional[float] = None,
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create a Poisson regression model specification.

    Poisson regression is a generalized linear model (GLM) for count data.
    It uses a log link function and assumes the response follows a Poisson
    distribution, making it ideal for:
    - Count data (number of events)
    - Rare events
    - Event rates
    - Contingency tables

    Args:
        penalty: Regularization penalty (0 = no penalty, higher = more penalty)
            Note: Regularization support depends on the engine
        mixture: Mix between L1 and L2 penalty (0 to 1)
            - 0 = pure L2 (Ridge)
            - 1 = pure L1 (Lasso)
            Note: May not be supported by all engines
        engine: Computational engine to use (default "statsmodels")

    Returns:
        ModelSpec for Poisson regression

    Examples:
        >>> # Basic Poisson regression
        >>> spec = poisson_reg()

        >>> # With regularization (if supported)
        >>> spec = poisson_reg(penalty=0.1)

        >>> # Fit to count data
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'count': [0, 1, 2, 3, 1, 0, 2, 4],
        ...     'x1': [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 4.5],
        ...     'x2': [0.5, 1.5, 2.5, 3.5, 2.0, 1.0, 3.0, 4.0],
        ... })
        >>> fit = spec.fit(df, 'count ~ x1 + x2')

    References:
        McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
        Chapman and Hall/CRC.
    """
    # Build args dict (only include non-None values)
    args = {}
    if penalty is not None:
        args["penalty"] = penalty
    if mixture is not None:
        args["mixture"] = mixture

    return ModelSpec(
        model_type="poisson_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
