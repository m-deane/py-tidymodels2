"""
VARMAX regression model specification

VARMAX (Vector AutoRegressive Moving Average with eXogenous variables) is a
multivariate time series model that extends ARIMA to multiple dependent variables.

Parameters (tidymodels naming):
- non_seasonal_ar: Number of AR terms (p) for all series
- non_seasonal_ma: Number of MA terms (q) for all series
- trend: Trend specification ('n'=none, 'c'=constant, 't'=linear, 'ct'=both)

Unlike ARIMA, VARMAX models multiple outcomes simultaneously:
- Multiple outcomes influence each other over time
- Can include exogenous predictors
- Provides cross-variable dynamics and forecasting
"""

from typing import Optional, Literal
from py_parsnip.model_spec import ModelSpec


def varmax_reg(
    non_seasonal_ar: int = 1,
    non_seasonal_ma: int = 0,
    trend: Literal["n", "c", "t", "ct"] = "c",
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create a VARMAX model specification.

    VARMAX extends ARIMA to multivariate time series, modeling how multiple
    variables evolve together over time with cross-dependencies.

    Args:
        non_seasonal_ar: AR order (p) - number of past time steps to use
            - Default: 1 (VAR(1) model)
        non_seasonal_ma: MA order (q) - number of past errors to use
            - Default: 0 (no MA terms)
        trend: Trend component to include
            - 'n': No trend
            - 'c': Constant (intercept)
            - 't': Linear time trend
            - 'ct': Both constant and time trend
            - Default: 'c' (constant only)
        engine: Computational engine to use (default "statsmodels")

    Returns:
        ModelSpec for VARMAX regression

    Examples:
        >>> # Basic VAR(1) model (no MA terms)
        >>> spec = varmax_reg(non_seasonal_ar=1, non_seasonal_ma=0)

        >>> # VARMA(2,1) model
        >>> spec = varmax_reg(non_seasonal_ar=2, non_seasonal_ma=1)

        >>> # VAR(2) with time trend
        >>> spec = varmax_reg(
        ...     non_seasonal_ar=2,
        ...     non_seasonal_ma=0,
        ...     trend="ct"
        ... )

    Note:
        - VARMAX requires multiple outcome variables
        - Formula syntax: "y1 + y2 + y3 ~ x1 + x2" (multiple outcomes)
        - All outcomes are modeled jointly with cross-dependencies
        - Seasonal VARMAX not currently supported
    """
    args = {
        "non_seasonal_ar": non_seasonal_ar,
        "non_seasonal_ma": non_seasonal_ma,
        "trend": trend,
    }

    return ModelSpec(
        model_type="varmax_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
