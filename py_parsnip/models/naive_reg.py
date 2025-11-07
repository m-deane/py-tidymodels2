"""
Naive Forecasting model specification

Essential baseline forecasting methods:
- naive: Last observed value (random walk)
- seasonal_naive: Last value from same season
- drift: Linear trend from first to last observation
- window: Rolling window average (moving average)

These are CRITICAL baselines for time series forecasting.
If your model can't beat these, it's not adding value.

Parameters:
- strategy: Which naive forecasting strategy to use
- seasonal_period: Seasonal frequency (for seasonal_naive)
- window_size: Window size (for window strategy)
"""

from typing import Optional, Literal
from py_parsnip.model_spec import ModelSpec


def naive_reg(
    strategy: Literal["naive", "seasonal_naive", "drift", "window"] = "naive",
    seasonal_period: Optional[int] = None,
    window_size: Optional[int] = None,
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

    - **window**: Rolling window average (moving average)
        y_t = mean(y_{t-window_size}, ..., y_{t-1})

    Args:
        strategy: Which naive forecasting strategy to use (default "naive")
            - "naive": Last value (random walk)
            - "seasonal_naive" or "snaive": Last seasonal value
            - "drift": Linear trend from first to last value
            - "window": Rolling window average
        seasonal_period: Seasonal frequency (required for seasonal_naive)
            - 12 for monthly data with yearly seasonality
            - 7 for daily data with weekly seasonality
            - 24 for hourly data with daily seasonality
        window_size: Window size for rolling average (required for window strategy)
            - Number of past observations to average
            - Must be >= 1
        engine: Computational engine to use (default "parsnip")

    Returns:
        ModelSpec for naive forecasting

    Examples:
        >>> # Naive forecast (last value)
        >>> spec = naive_reg(strategy="naive")
        >>>
        >>> # Seasonal naive (e.g., last Monday's value for Monday)
        >>> spec = naive_reg(strategy="seasonal_naive", seasonal_period=7)
        >>>
        >>> # Drift (linear trend)
        >>> spec = naive_reg(strategy="drift")
        >>>
        >>> # Window (7-period moving average)
        >>> spec = naive_reg(strategy="window", window_size=7)

    Notes:
        - These are parameter-free baselines (except window)
        - Extremely fast
        - Surprisingly effective for many time series
        - Essential for benchmarking forecasting models
        - If ML model can't beat these, don't use it!
        - Similar to sktime's NaiveForecaster
    """
    # Build args dict
    args = {"strategy": strategy}

    if seasonal_period is not None:
        args["seasonal_period"] = seasonal_period

    if window_size is not None:
        args["window_size"] = window_size

    return ModelSpec(
        model_type="naive_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
