"""
MARS (Multivariate Adaptive Regression Splines) model specification

Supports engine:
- pyearth: Earth (py-earth implementation)

Parameters (tidymodels naming):
- num_terms: Maximum number of terms (max_terms)
- prod_degree: Maximum degree of interaction (max_degree)
- prune_method: Pruning method ('none', 'forward', 'backward')

MARS creates piecewise linear regression using hinge functions,
allowing for automatic detection of non-linear relationships and interactions.
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def mars(
    num_terms: Optional[int] = None,
    prod_degree: Optional[int] = None,
    prune_method: Optional[str] = None,
    engine: str = "pyearth",
) -> ModelSpec:
    """
    Create a MARS (Multivariate Adaptive Regression Splines) model specification.

    MARS is an adaptive procedure for regression that uses piecewise linear
    basis functions (hinge functions) to model non-linear relationships and
    interactions between predictors.

    Args:
        num_terms: Maximum number of terms in the model (default varies by engine)
            - Controls model complexity
            - Higher values allow more flexible models but risk overfitting
        prod_degree: Maximum degree of interaction between predictors (default 1)
            - 1 = no interactions (additive model)
            - 2 = pairwise interactions
            - Higher values allow more complex interactions
        prune_method: Method for pruning terms (default 'backward')
            - 'none': No pruning
            - 'forward': Forward selection only
            - 'backward': Forward + backward pruning (recommended)
        engine: Computational engine to use (default "pyearth")

    Returns:
        ModelSpec for MARS regression

    Examples:
        >>> # Basic MARS model
        >>> spec = mars()

        >>> # Control complexity
        >>> spec = mars(num_terms=10, prod_degree=1)

        >>> # Allow pairwise interactions
        >>> spec = mars(num_terms=20, prod_degree=2)

        >>> # No pruning (forward selection only)
        >>> spec = mars(prune_method='forward')

    References:
        Friedman, J. H. (1991). Multivariate adaptive regression splines.
        The Annals of Statistics, 19(1), 1-67.
    """
    # Build args dict (only include non-None values)
    args = {}
    if num_terms is not None:
        args["num_terms"] = num_terms
    if prod_degree is not None:
        args["prod_degree"] = prod_degree
    if prune_method is not None:
        args["prune_method"] = prune_method

    return ModelSpec(
        model_type="mars",
        engine=engine,
        mode="regression",
        args=args,
    )
