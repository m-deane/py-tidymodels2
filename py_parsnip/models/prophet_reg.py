"""
Prophet regression model specification

Prophet is Facebook's time series forecasting model that handles:
- Trend changes
- Multiple seasonality (daily, weekly, yearly)
- Holiday effects
- Missing data

Parameters (tidymodels naming):
- growth: Trend type ('linear' or 'logistic')
- changepoint_prior_scale: Flexibility of trend changes
- seasonality_prior_scale: Flexibility of seasonality
- seasonality_mode: How components combine ('additive' or 'multiplicative')
- n_changepoints: Number of potential changepoints
- changepoint_range: Proportion of history for changepoints
- seasonality_yearly: Toggle yearly seasonality ('auto', True, False)
- seasonality_weekly: Toggle weekly seasonality ('auto', True, False)
- seasonality_daily: Toggle daily seasonality ('auto', True, False)

Note: Prophet requires data with 'ds' (datetime) and 'y' (value) columns.
"""

from typing import Optional, Literal, Union
from py_parsnip.model_spec import ModelSpec


def prophet_reg(
    growth: Literal["linear", "logistic"] = "linear",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Literal["additive", "multiplicative"] = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    seasonality_yearly: Union[Literal["auto"], bool] = "auto",
    seasonality_weekly: Union[Literal["auto"], bool] = "auto",
    seasonality_daily: Union[Literal["auto"], bool] = "auto",
    engine: str = "prophet",
) -> ModelSpec:
    """
    Create a Prophet regression model specification.

    Prophet is designed for forecasting time series data with strong
    seasonal patterns and several seasons of historical data.

    Args:
        growth: Trend model ('linear' or 'logistic')
        changepoint_prior_scale: Controls flexibility of automatic changepoint selection
            - Larger values = more flexible trend
            - Default 0.05
        seasonality_prior_scale: Controls flexibility of seasonality
            - Larger values = more flexible seasonality
            - Default 10.0
        seasonality_mode: How seasonality components are combined
            - 'additive': y = trend + seasonality + holidays + error
            - 'multiplicative': y = trend * (1 + seasonality) * (1 + holidays) + error
        n_changepoints: Number of potential changepoints for trend
        changepoint_range: Proportion of history in which trend changepoints are allowed
        seasonality_yearly: Toggles yearly seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force yearly seasonality on
            - False: Turn yearly seasonality off
        seasonality_weekly: Toggles weekly seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force weekly seasonality on
            - False: Turn weekly seasonality off
        seasonality_daily: Toggles daily seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force daily seasonality on
            - False: Turn daily seasonality off
        engine: Computational engine (default "prophet")

    Returns:
        ModelSpec for Prophet regression

    Examples:
        >>> # Basic Prophet model
        >>> spec = prophet_reg()

        >>> # More flexible trend
        >>> spec = prophet_reg(changepoint_prior_scale=0.1)

        >>> # Multiplicative seasonality (good for exponential growth)
        >>> spec = prophet_reg(seasonality_mode='multiplicative')

        >>> # Logistic growth (requires cap and floor in data)
        >>> spec = prophet_reg(growth='logistic')

        >>> # Turn off yearly seasonality (useful for short time series or hybrid models)
        >>> spec = prophet_reg(seasonality_yearly=False)

        >>> # Turn off all seasonality (useful for hybrid models)
        >>> spec = prophet_reg(
        ...     seasonality_yearly=False,
        ...     seasonality_weekly=False,
        ...     seasonality_daily=False
        ... )

    Note:
        Prophet expects data with specific column names:
        - 'ds': datetime column
        - 'y': value column
        Use formula like "sales ~ date" where 'date' is datetime.

        For hybrid models (e.g., prophet_boost), you may want to turn off
        seasonality components so the boosting model can capture them instead.
    """
    args = {
        "growth": growth,
        "changepoint_prior_scale": changepoint_prior_scale,
        "seasonality_prior_scale": seasonality_prior_scale,
        "seasonality_mode": seasonality_mode,
        "n_changepoints": n_changepoints,
        "changepoint_range": changepoint_range,
        "yearly_seasonality": seasonality_yearly,
        "weekly_seasonality": seasonality_weekly,
        "daily_seasonality": seasonality_daily,
    }

    return ModelSpec(
        model_type="prophet_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
