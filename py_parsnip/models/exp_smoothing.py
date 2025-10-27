"""
Exponential Smoothing model specification

Exponential Smoothing (ETS - Error, Trend, Seasonality) models are classic
time series models that forecast by weighted averaging of past observations,
with weights decaying exponentially.

Methods:
- Simple Exponential Smoothing: Level only (no trend, no seasonality)
- Holt's Linear: Level + Trend
- Holt-Winters: Level + Trend + Seasonality

Parameters (tidymodels naming):
- seasonal_period: Seasonality period (e.g., 12 for monthly data with yearly seasonality)
- error: Error type ("additive" or "multiplicative")
- trend: Trend component ("additive", "multiplicative", or None)
- season: Seasonal component ("additive", "multiplicative", or None)
- damping: Damped trend (True/False)

This creates ETS model with flexible component specification.
"""

from typing import Optional, Literal
from py_parsnip.model_spec import ModelSpec


def exp_smoothing(
    seasonal_period: Optional[int] = None,
    error: Optional[Literal["additive", "multiplicative"]] = "additive",
    trend: Optional[Literal["additive", "multiplicative"]] = None,
    season: Optional[Literal["additive", "multiplicative"]] = None,
    damping: bool = False,
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create an Exponential Smoothing model specification.

    Exponential Smoothing models forecast by weighted averaging of past
    observations with exponentially decaying weights. Different variants
    handle trend and seasonality:

    - Simple: Level only (trend=None, season=None)
    - Holt: Level + Trend (trend set, season=None)
    - Holt-Winters: Level + Trend + Seasonality (all components set)

    Args:
        seasonal_period: Number of periods in seasonal cycle (e.g., 12 for monthly).
                        Required if season is not None.
        error: Error type - "additive" or "multiplicative"
        trend: Trend component - "additive", "multiplicative", or None
        season: Seasonal component - "additive", "multiplicative", or None
        damping: Whether to use damped trend (requires trend to be set)
        engine: Computational engine (default "statsmodels")

    Returns:
        ModelSpec for Exponential Smoothing

    Examples:
        >>> # Simple Exponential Smoothing (level only)
        >>> spec = exp_smoothing()

        >>> # Holt's Linear Method (level + trend)
        >>> spec = exp_smoothing(
        ...     trend="additive"
        ... )

        >>> # Holt-Winters (level + trend + seasonality)
        >>> spec = exp_smoothing(
        ...     seasonal_period=12,
        ...     trend="additive",
        ...     season="additive"
        ... )

        >>> # Damped Holt-Winters
        >>> spec = exp_smoothing(
        ...     seasonal_period=12,
        ...     trend="additive",
        ...     season="multiplicative",
        ...     damping=True
        ... )

        >>> # Multiplicative Holt-Winters
        >>> spec = exp_smoothing(
        ...     seasonal_period=12,
        ...     error="multiplicative",
        ...     trend="multiplicative",
        ...     season="multiplicative"
        ... )

    Note:
        - Simple ES: No parameters set (all None/False)
        - Holt: trend set, season=None
        - Holt-Winters: trend and season both set
        - seasonal_period required if season is not None
        - damping requires trend to be set
        - Multiplicative components work best with positive data
    """
    # Validation
    if season is not None and seasonal_period is None:
        raise ValueError("seasonal_period must be specified when season is not None")

    if damping and trend is None:
        raise ValueError("damping requires trend to be set")

    args = {
        "seasonal_period": seasonal_period,
        "error": error,
        "trend": trend,
        "season": season,
        "damping": damping,
    }

    return ModelSpec(
        model_type="exp_smoothing",
        engine=engine,
        mode="regression",
        args=args,
    )
