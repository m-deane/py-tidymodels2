"""
recursive_reg(): Recursive forecasting for time series

This model wraps any sklearn-compatible regression model for multi-step
time series forecasting using the recursive strategy via skforecast.

The recursive strategy uses a model trained on lagged values to predict
one step ahead, then uses that prediction as input for the next prediction.

Key Parameters:
    base_model: ModelSpec - The base regression model (rand_forest, linear_reg, etc.)
    lags: int or list - Number of lags to use as features
        - If int: uses lags 1 through n
        - If list: uses specific lag indices
    differentiation: int - Order of differencing (default None)
    engine: str - Engine to use (default "skforecast")

Example:
    >>> from py_parsnip import recursive_reg, rand_forest
    >>> spec = recursive_reg(
    ...     base_model=rand_forest(trees=100),
    ...     lags=7,  # Use past 7 time steps
    ...     engine="skforecast"
    ... )
"""

from typing import Any, Union, List, Optional
from py_parsnip.model_spec import ModelSpec


def recursive_reg(
    base_model: ModelSpec,
    lags: Union[int, List[int]] = 1,
    differentiation: Optional[int] = None,
    engine: str = "skforecast",
) -> ModelSpec:
    """
    Create recursive forecasting model specification.

    Wraps any sklearn-compatible model for multi-step time series forecasting
    using lagged features and recursive prediction.

    Args:
        base_model: Base regression model (rand_forest, linear_reg, etc.)
        lags: Number of lags or list of specific lag indices
            - If int: uses lags 1 through n (e.g., 7 uses [1,2,3,4,5,6,7])
            - If list: uses specific lags (e.g., [1, 7, 14])
        differentiation: Order of differencing to apply (None, 1, or 2)
        engine: Engine to use (currently only "skforecast")

    Returns:
        ModelSpec configured for recursive forecasting

    Examples:
        >>> # Random Forest with 7 lags
        >>> spec = recursive_reg(
        ...     base_model=rand_forest(trees=100),
        ...     lags=7
        ... )

        >>> # Linear regression with specific lags
        >>> spec = recursive_reg(
        ...     base_model=linear_reg(),
        ...     lags=[1, 7, 14, 28]  # Weekly patterns
        ... )

        >>> # With differencing
        >>> spec = recursive_reg(
        ...     base_model=rand_forest(trees=50),
        ...     lags=14,
        ...     differentiation=1  # Make series stationary
        ... )
    """
    if not isinstance(base_model, ModelSpec):
        raise TypeError(f"base_model must be a ModelSpec, got {type(base_model)}")

    if isinstance(lags, int) and lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    if isinstance(lags, list) and len(lags) == 0:
        raise ValueError("lags list cannot be empty")

    if differentiation is not None and differentiation not in [1, 2]:
        raise ValueError(f"differentiation must be None, 1, or 2, got {differentiation}")

    args = {
        "base_model": base_model,
        "lags": lags,
        "differentiation": differentiation,
    }

    return ModelSpec(
        model_type="recursive_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
