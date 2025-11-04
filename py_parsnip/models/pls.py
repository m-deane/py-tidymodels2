"""
Partial Least Squares (PLS) Regression model specification

Supports sklearn engine with PLSRegression for dimension reduction
and regression on latent components.

Parameters (tidymodels naming):
- num_comp: Number of PLS components to extract (default: 2)
- predictor_prop: Proportion of variance to explain (alternative to num_comp)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def pls(
    num_comp: Optional[int] = None,
    predictor_prop: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a Partial Least Squares regression model specification.

    PLS regression finds latent components that maximize covariance between
    predictors and outcomes, providing dimension reduction while maintaining
    predictive power. Useful when predictors are highly correlated or
    when number of predictors exceeds number of observations.

    Args:
        num_comp: Number of PLS components to extract (default: 2)
            - For sklearn: maps to 'n_components'
            - Should be less than min(n_samples, n_features, n_targets)
        predictor_prop: Proportion of predictor variance to explain (0 to 1)
            - Alternative to num_comp for automatic component selection
            - Not directly supported by sklearn, used for component tuning
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for PLS regression

    Examples:
        >>> # PLS with 2 components
        >>> spec = pls(num_comp=2)

        >>> # PLS with 5 components
        >>> spec = pls(num_comp=5)

        >>> # PLS for high-dimensional data
        >>> spec = pls(num_comp=10)

        >>> # Change engine (if other engines are implemented)
        >>> spec = pls(num_comp=3, engine="sklearn")
    """
    # Build args dict (only include non-None values)
    args = {}
    if num_comp is not None:
        args["num_comp"] = num_comp
    if predictor_prop is not None:
        args["predictor_prop"] = predictor_prop

    return ModelSpec(
        model_type="pls",
        engine=engine,
        mode="regression",
        args=args,
    )
