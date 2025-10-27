"""
Generalized Additive Model (GAM) specification

Supports engine:
- pygam: LinearGAM, GAM

Parameters (tidymodels naming):
- select_features: Enable automatic feature selection (boolean)
- adjust_deg_free: Adjustment to degrees of freedom (controls smoothing)

GAMs fit smooth non-parametric functions to each predictor while maintaining
the interpretability of additive models. They automatically detect and model
non-linear relationships using splines.
"""

from typing import Optional, Union
from py_parsnip.model_spec import ModelSpec


def gen_additive_mod(
    select_features: Optional[bool] = None,
    adjust_deg_free: Optional[Union[float, int]] = None,
    engine: str = "pygam",
) -> ModelSpec:
    """
    Create a Generalized Additive Model (GAM) specification.

    GAMs extend linear models by replacing linear terms with smooth functions,
    allowing automatic detection of non-linear relationships while maintaining
    interpretability. Each predictor is fit with a smooth spline function.

    Key advantages:
    - Automatic non-linearity detection
    - Interpretable smooth functions
    - No need to manually specify transformations
    - Visual assessment of predictor effects

    Args:
        select_features: Enable automatic feature selection (default False)
            - When True, uses regularization to select important features
            - Helpful with many predictors
        adjust_deg_free: Adjustment to degrees of freedom (default 10)
            - Controls smoothness of fitted curves
            - Lower values = smoother (more bias, less variance)
            - Higher values = more flexible (less bias, more variance)
            - Typical range: 3-20
        engine: Computational engine to use (default "pygam")

    Returns:
        ModelSpec for Generalized Additive Model

    Examples:
        >>> # Basic GAM with default smoothing
        >>> spec = gen_additive_mod()

        >>> # More flexible model (more wiggly curves)
        >>> spec = gen_additive_mod(adjust_deg_free=15)

        >>> # Smoother model (less wiggly curves)
        >>> spec = gen_additive_mod(adjust_deg_free=5)

        >>> # With automatic feature selection
        >>> spec = gen_additive_mod(select_features=True)

        >>> # Fit to data
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'y': [1, 4, 9, 16, 25, 20, 15, 10],
        ...     'x': [1, 2, 3, 4, 5, 6, 7, 8],
        ... })
        >>> fit = spec.fit(df, 'y ~ x')

    References:
        Hastie, T., & Tibshirani, R. (1990). Generalized Additive Models.
        Chapman and Hall/CRC.

        Wood, S. N. (2017). Generalized Additive Models: An Introduction with R.
        CRC Press.
    """
    # Build args dict (only include non-None values)
    args = {}
    if select_features is not None:
        args["select_features"] = select_features
    if adjust_deg_free is not None:
        args["adjust_deg_free"] = adjust_deg_free

    return ModelSpec(
        model_type="gen_additive_mod",
        engine=engine,
        mode="regression",
        args=args,
    )
