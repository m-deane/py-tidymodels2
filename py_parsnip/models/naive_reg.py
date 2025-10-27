"""
Naive Forecasting model specification

Essential baseline forecasting methods:
- naive: Last observed value (random walk)
- seasonal_naive: Last value from same season
- drift: Linear trend from first to last observation

These are CRITICAL baselines for time series forecasting.
If your model can't beat these, it's not adding value.

Parameters:
- seasonal_period: Seasonal frequency (for seasonal_naive)
- method: Which naive method to use
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def naive_reg(
    seasonal_period: Optional[int] = None,
    method: str = "naive",
    engine: str = "parsnip",
) -> ModelSpec:
    """
    Create a naive forecasting model specification.

    Naive methods are essential baselines for time series forecasting:

    - **naive**: Predicts last observed value (random walk)
        y_t = y_{t-1}

    - **seasonal_naive** (snaive): Predicts last value from same season
        y_t = y_{t-seasonal_period}

    - **drift**: Linear trend from first to last observation
        y_t = y_{t-1} + (y_T - y_1) / (T - 1)

    Args:
        seasonal_period: Seasonal frequency (required for seasonal_naive)
            - 12 for monthly data with yearly seasonality
            - 7 for daily data with weekly seasonality
            - 24 for hourly data with daily seasonality
        method: Which naive method to use
            - "naive": Last value (default)
            - "seasonal_naive" or "snaive": Last seasonal value
            - "drift": Linear trend
        engine: Computational engine to use (default "parsnip")

    Returns:
        ModelSpec for naive forecasting

    Examples:
        >>> # Naive forecast (last value)
        >>> spec = naive_reg()
        >>>
        >>> # Seasonal naive (e.g., last Monday's value for Monday)
        >>> spec = naive_reg(seasonal_period=7, method="seasonal_naive")
        >>>
        >>> # Drift (linear trend)
        >>> spec = naive_reg(method="drift")

    Notes:
        - These are parameter-free baselines
        - Extremely fast
        - Surprisingly effective for many time series
        - Essential for benchmarking forecasting models
        - If ML model can't beat these, don't use it!
    """
    # Build args dict
    args = {"method": method}

    if seasonal_period is not None:
        args["seasonal_period"] = seasonal_period

    return ModelSpec(
        model_type="naive_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
