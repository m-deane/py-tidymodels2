"""
Sliding Window Forecasting model specification

Forecasts using rolling window aggregates (mean, median, weighted_mean).
This is different from recursive_reg which uses lagged features + ML models.

Window forecasting is a simple but effective baseline that:
- Uses last N observations (window)
- Applies aggregation function (mean/median/weighted_mean)
- Produces forecast for next period(s)

Parameters:
- window_size: Size of rolling window (default 7)
- method: Aggregation method - "mean", "median", "weighted_mean" (default "mean")
- weights: Optional weights for weighted_mean (default None)
- min_periods: Minimum observations in window (default None)
"""

from typing import Optional, Literal, List
from py_parsnip.model_spec import ModelSpec


def window_reg(
    window_size: int = 7,
    method: Literal["mean", "median", "weighted_mean"] = "mean",
    weights: Optional[List[float]] = None,
    min_periods: Optional[int] = None,
    engine: str = "parsnip",
) -> ModelSpec:
    """
    Create a sliding window forecasting model specification.

    Uses rolling window aggregates for forecasting. Unlike recursive_reg
    (which uses lagged features + ML models), window_reg applies simple
    aggregation functions (mean, median, weighted_mean) to a sliding window
    of past observations.

    Args:
        window_size: Size of rolling window (default 7)
            - Number of past observations to use
            - Must be >= 1
        method: Aggregation method (default "mean")
            - "mean": Simple moving average
            - "median": Median of window
            - "weighted_mean": Weighted moving average (requires weights)
        weights: Optional weights for weighted_mean (default None)
            - Must have length = window_size
            - Should sum to 1.0 (will be normalized if not)
            - Example: [0.5, 0.3, 0.2] gives more weight to recent observations
        min_periods: Minimum observations in window (default None)
            - If None, uses window_size (no partial windows)
            - If < window_size, allows partial windows with fewer observations
        engine: Computational engine to use (default "parsnip")

    Returns:
        ModelSpec for sliding window forecasting

    Examples:
        >>> # Simple 7-day moving average
        >>> spec = window_reg(window_size=7, method="mean")
        >>>
        >>> # Median of last 14 observations
        >>> spec = window_reg(window_size=14, method="median")
        >>>
        >>> # Weighted moving average (more weight to recent)
        >>> spec = window_reg(
        ...     window_size=3,
        ...     method="weighted_mean",
        ...     weights=[0.5, 0.3, 0.2]  # Most recent gets 0.5
        ... )
        >>>
        >>> # Allow partial windows (useful for start of series)
        >>> spec = window_reg(window_size=7, min_periods=3)

    Notes:
        - Extremely simple and interpretable
        - Fast computation
        - Good baseline for smooth time series
        - Works well when recent past is best predictor
        - Weighted mean can emphasize recent observations
        - Different from recursive_reg: no ML model, just aggregation
        - Similar to pandas.DataFrame.rolling().mean()
    """
    # Build args dict
    args = {
        "window_size": window_size,
        "method": method,
    }

    if weights is not None:
        args["weights"] = weights

    if min_periods is not None:
        args["min_periods"] = min_periods

    return ModelSpec(
        model_type="window_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
