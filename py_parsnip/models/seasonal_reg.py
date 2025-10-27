"""
Seasonal Decomposition model specification

Seasonal regression models use STL (Seasonal-Trend decomposition using LOESS)
to decompose a time series into seasonal, trend, and remainder components,
then fit a forecasting model to the seasonally adjusted series.

STL is robust to outliers and can handle any type of seasonality, including
multiple seasonal periods.

Parameters (tidymodels naming):
- seasonal_period_1: Primary seasonality period (e.g., 7 for daily data with weekly pattern)
- seasonal_period_2: Secondary seasonality period (optional, e.g., 365 for yearly pattern)
- seasonal_period_3: Tertiary seasonality period (optional)

After decomposition, the seasonally adjusted series can be forecast using:
- ETS (Exponential Smoothing)
- ARIMA
- Naive methods

This implementation uses STL + ETS by default.
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def seasonal_reg(
    seasonal_period_1: Optional[int] = None,
    seasonal_period_2: Optional[int] = None,
    seasonal_period_3: Optional[int] = None,
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create a Seasonal Decomposition model specification.

    Uses STL (Seasonal-Trend decomposition using LOESS) to decompose the
    time series, then fits a forecasting model to the seasonally adjusted series.
    The final forecast combines the model prediction with the seasonal component.

    This approach is particularly useful when:
    - You have strong, complex seasonality
    - Multiple seasonal periods exist (daily + weekly + yearly)
    - You want to visualize decomposed components
    - Traditional seasonal models (SARIMA, Holt-Winters) are insufficient

    Args:
        seasonal_period_1: Primary seasonal period (required)
                          Examples: 7 (weekly), 12 (monthly), 24 (hourly)
        seasonal_period_2: Secondary seasonal period (optional)
                          Example: 365 for yearly pattern in daily data
        seasonal_period_3: Tertiary seasonal period (optional)
        engine: Computational engine (default "statsmodels")

    Returns:
        ModelSpec for Seasonal Decomposition

    Examples:
        >>> # Weekly seasonality in daily data
        >>> spec = seasonal_reg(seasonal_period_1=7)

        >>> # Weekly + yearly seasonality in daily data
        >>> spec = seasonal_reg(
        ...     seasonal_period_1=7,
        ...     seasonal_period_2=365
        ... )

        >>> # Hourly + daily + weekly seasonality
        >>> spec = seasonal_reg(
        ...     seasonal_period_1=24,    # daily pattern
        ...     seasonal_period_2=168,   # weekly pattern (24*7)
        ...     seasonal_period_3=8760   # yearly pattern (24*365)
        ... )

        >>> # Monthly seasonality
        >>> spec = seasonal_reg(seasonal_period_1=12)

    Note:
        - At least one seasonal_period must be specified
        - STL requires at least 2 full seasonal cycles
        - The decomposition is additive: y = trend + seasonal + remainder
        - After decomposition, ETS is fit to (trend + remainder)
        - Final forecast = ETS forecast + seasonal component
    """
    if seasonal_period_1 is None:
        raise ValueError("At least seasonal_period_1 must be specified")

    args = {
        "seasonal_period_1": seasonal_period_1,
        "seasonal_period_2": seasonal_period_2,
        "seasonal_period_3": seasonal_period_3,
    }

    return ModelSpec(
        model_type="seasonal_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
