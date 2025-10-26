"""
ARIMA regression model specification

ARIMA (AutoRegressive Integrated Moving Average) is a classic time series model
that handles trend and autocorrelation through differencing, AR, and MA terms.

Parameters (tidymodels naming):
- seasonal_period: Seasonality period (e.g., 12 for monthly data with yearly seasonality)
- non_seasonal_ar: Number of non-seasonal AR terms (p)
- non_seasonal_differences: Number of non-seasonal differences (d)
- non_seasonal_ma: Number of non-seasonal MA terms (q)
- seasonal_ar: Number of seasonal AR terms (P)
- seasonal_differences: Number of seasonal differences (D)
- seasonal_ma: Number of seasonal MA terms (Q)

This creates SARIMA model: ARIMA(p,d,q)(P,D,Q)[m]
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def arima_reg(
    seasonal_period: Optional[int] = None,
    non_seasonal_ar: int = 0,
    non_seasonal_differences: int = 0,
    non_seasonal_ma: int = 0,
    seasonal_ar: int = 0,
    seasonal_differences: int = 0,
    seasonal_ma: int = 0,
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create an ARIMA model specification.

    ARIMA models time series by combining:
    - AR (AutoRegressive): Value depends on past values
    - I (Integrated): Differencing to make series stationary
    - MA (Moving Average): Value depends on past errors

    Args:
        seasonal_period: Number of periods in seasonal cycle (e.g., 12 for monthly)
        non_seasonal_ar: AR order (p) - number of past values to use
        non_seasonal_differences: Differencing order (d) - times to difference
        non_seasonal_ma: MA order (q) - number of past errors to use
        seasonal_ar: Seasonal AR order (P)
        seasonal_differences: Seasonal differencing order (D)
        seasonal_ma: Seasonal MA order (Q)
        engine: Computational engine (default "statsmodels")

    Returns:
        ModelSpec for ARIMA regression

    Examples:
        >>> # ARIMA(1,1,1) - simple ARIMA
        >>> spec = arima_reg(
        ...     non_seasonal_ar=1,
        ...     non_seasonal_differences=1,
        ...     non_seasonal_ma=1
        ... )

        >>> # SARIMA(1,1,1)(1,1,1)[12] - with seasonality
        >>> spec = arima_reg(
        ...     seasonal_period=12,
        ...     non_seasonal_ar=1,
        ...     non_seasonal_differences=1,
        ...     non_seasonal_ma=1,
        ...     seasonal_ar=1,
        ...     seasonal_differences=1,
        ...     seasonal_ma=1
        ... )

        >>> # Auto ARIMA (engine-specific)
        >>> spec = arima_reg().set_engine("auto_arima")

    Note:
        - For non-seasonal data, leave seasonal parameters at 0
        - seasonal_period is required if using seasonal components
        - Differencing (d, D) removes trend/seasonality
    """
    args = {
        "seasonal_period": seasonal_period,
        "non_seasonal_ar": non_seasonal_ar,
        "non_seasonal_differences": non_seasonal_differences,
        "non_seasonal_ma": non_seasonal_ma,
        "seasonal_ar": seasonal_ar,
        "seasonal_differences": seasonal_differences,
        "seasonal_ma": seasonal_ma,
    }

    return ModelSpec(
        model_type="arima_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
