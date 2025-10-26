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

Note: Prophet requires data with 'ds' (datetime) and 'y' (value) columns.
"""

from typing import Optional, Literal
from py_parsnip.model_spec import ModelSpec


def prophet_reg(
    growth: Literal["linear", "logistic"] = "linear",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Literal["additive", "multiplicative"] = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
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

    Note:
        Prophet expects data with specific column names:
        - 'ds': datetime column
        - 'y': value column
        Use formula like "sales ~ date" where 'date' is datetime.
    """
    args = {
        "growth": growth,
        "changepoint_prior_scale": changepoint_prior_scale,
        "seasonality_prior_scale": seasonality_prior_scale,
        "seasonality_mode": seasonality_mode,
        "n_changepoints": n_changepoints,
        "changepoint_range": changepoint_range,
    }

    return ModelSpec(
        model_type="prophet_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
