"""
Linear Regression model specification

Supports multiple engines:
- sklearn: LinearRegression, Ridge, Lasso, ElasticNet
- statsmodels: OLS

Parameters (tidymodels naming):
- penalty: Regularization penalty (L1 + L2)
- mixture: Mix of L1 (1.0) vs L2 (0.0) penalty
- intercept: Whether to fit an intercept term (default True)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def linear_reg(
    penalty: Optional[float] = None,
    mixture: Optional[float] = None,
    intercept: bool = True,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a linear regression model specification.

    Args:
        penalty: Regularization penalty (0 = no penalty, higher = more penalty)
            - For sklearn Ridge: maps to 'alpha'
            - For sklearn ElasticNet: maps to 'alpha'
        mixture: Mix between L1 and L2 penalty (0 to 1)
            - 0 = pure L2 (Ridge)
            - 1 = pure L1 (Lasso)
            - 0.5 = equal mix (ElasticNet)
            - For sklearn ElasticNet: maps to 'l1_ratio'
        intercept: Whether to fit an intercept term (default True)
            - True: Fit intercept (standard regression)
            - False: Force regression through origin (no intercept)
            - For sklearn: maps to 'fit_intercept'
            - For statsmodels: modifies formula to include '+0'
            - Note: If formula already contains '+0' or '-1', formula takes precedence
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for linear regression

    Examples:
        >>> # OLS (no penalty)
        >>> spec = linear_reg()

        >>> # Ridge regression (L2 penalty)
        >>> spec = linear_reg(penalty=0.1, mixture=0.0)

        >>> # Lasso regression (L1 penalty)
        >>> spec = linear_reg(penalty=0.1, mixture=1.0)

        >>> # ElasticNet (mix of L1 and L2)
        >>> spec = linear_reg(penalty=0.1, mixture=0.5)

        >>> # Regression through origin (no intercept)
        >>> spec = linear_reg(intercept=False)

        >>> # Change engine
        >>> spec = linear_reg(engine="statsmodels")
    """
    # Build args dict (only include non-None values, always include intercept)
    args = {"intercept": intercept}
    if penalty is not None:
        args["penalty"] = penalty
    if mixture is not None:
        args["mixture"] = mixture

    return ModelSpec(
        model_type="linear_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
