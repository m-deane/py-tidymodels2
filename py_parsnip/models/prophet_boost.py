"""
Prophet + XGBoost hybrid model specification

This is a hybrid time series model that combines:
1. Prophet: Captures trend, seasonality, and holiday effects
2. XGBoost: Captures non-linear patterns in the residuals

Strategy:
- Fit Prophet model to capture trend and seasonal patterns
- Fit XGBoost on Prophet residuals to capture non-linear patterns
- Final prediction = Prophet prediction + XGBoost prediction

Parameters:
Prophet parameters:
- growth: Trend type ('linear' or 'logistic')
- changepoint_prior_scale: Flexibility of trend changes
- seasonality_prior_scale: Flexibility of seasonality
- seasonality_mode: How components combine ('additive' or 'multiplicative')
- n_changepoints: Number of potential changepoints
- changepoint_range: Proportion of history for changepoints
- seasonality_yearly: Toggle yearly seasonality ('auto', True, False)
- seasonality_weekly: Toggle weekly seasonality ('auto', True, False)
- seasonality_daily: Toggle daily seasonality ('auto', True, False)

XGBoost parameters:
- trees: Number of boosting iterations (n_estimators)
- tree_depth: Maximum tree depth (max_depth)
- learn_rate: Learning rate (eta/learning_rate)
- min_n: Minimum samples in leaf (min_child_weight)
- loss_reduction: Minimum loss reduction (gamma)
- sample_size: Subsample ratio (subsample)
- mtry: Feature sampling ratio (colsample_bytree)
"""

from typing import Literal, Union
from py_parsnip.model_spec import ModelSpec


def prophet_boost(
    # Prophet parameters
    growth: Literal["linear", "logistic"] = "linear",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Literal["additive", "multiplicative"] = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    seasonality_yearly: Union[Literal["auto"], bool] = "auto",
    seasonality_weekly: Union[Literal["auto"], bool] = "auto",
    seasonality_daily: Union[Literal["auto"], bool] = "auto",
    # XGBoost parameters
    trees: int = 100,
    tree_depth: int = 6,
    learn_rate: float = 0.1,
    min_n: int = 1,
    loss_reduction: float = 0.0,
    sample_size: float = 1.0,
    mtry: float = 1.0,
    engine: str = "hybrid_prophet_xgboost",
) -> ModelSpec:
    """
    Create a Prophet + XGBoost hybrid model specification.

    This hybrid model combines the strengths of both methods:
    - Prophet captures trend, seasonality, and holiday effects
    - XGBoost captures non-linear patterns in the residuals

    The model works in two stages:
    1. Fit Prophet to the time series
    2. Fit XGBoost to the Prophet residuals
    3. Final prediction = Prophet pred + XGBoost pred

    Args:
        growth: Trend model ('linear' or 'logistic')
        changepoint_prior_scale: Controls flexibility of trend changepoints
            - Larger values = more flexible trend (default 0.05)
        seasonality_prior_scale: Controls flexibility of seasonality
            - Larger values = more flexible seasonality (default 10.0)
        seasonality_mode: How components combine
            - 'additive': y = trend + seasonality + error
            - 'multiplicative': y = trend * (1 + seasonality) + error
        n_changepoints: Number of potential changepoints (default 25)
        changepoint_range: Proportion of history for changepoints (default 0.8)
        seasonality_yearly: Toggles yearly seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force yearly seasonality on
            - False: Turn yearly seasonality off (let XGBoost capture it)
        seasonality_weekly: Toggles weekly seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force weekly seasonality on
            - False: Turn weekly seasonality off (let XGBoost capture it)
        seasonality_daily: Toggles daily seasonality component
            - 'auto': Prophet decides based on data (default)
            - True: Force daily seasonality on
            - False: Turn daily seasonality off (let XGBoost capture it)
        trees: Number of boosting trees (default 100)
        tree_depth: Maximum tree depth (default 6)
        learn_rate: XGBoost learning rate (default 0.1)
        min_n: Minimum samples in leaf (default 1)
        loss_reduction: Minimum loss reduction for split (default 0.0)
        sample_size: Row subsample ratio (default 1.0)
        mtry: Feature subsample ratio (default 1.0)
        engine: Computational engine (default "hybrid_prophet_xgboost")

    Returns:
        ModelSpec for Prophet + XGBoost hybrid

    Examples:
        >>> # Basic Prophet + XGBoost
        >>> spec = prophet_boost(
        ...     trees=100,
        ...     tree_depth=3
        ... )

        >>> # More flexible trend with boosting
        >>> spec = prophet_boost(
        ...     changepoint_prior_scale=0.1,
        ...     seasonality_mode='multiplicative',
        ...     trees=200,
        ...     learn_rate=0.05
        ... )

        >>> # Logistic growth with aggressive boosting
        >>> spec = prophet_boost(
        ...     growth='logistic',
        ...     trees=500,
        ...     tree_depth=8,
        ...     learn_rate=0.01
        ... )

        >>> # Let XGBoost capture ALL seasonality (Prophet handles only trend)
        >>> spec = prophet_boost(
        ...     seasonality_yearly=False,
        ...     seasonality_weekly=False,
        ...     seasonality_daily=False,
        ...     trees=200,
        ...     tree_depth=6
        ... )

        >>> # Let XGBoost capture yearly seasonality only
        >>> spec = prophet_boost(
        ...     seasonality_yearly=False,
        ...     trees=150
        ... )

    Note:
        The hybrid approach is particularly effective when:
        - Data has strong seasonality but also non-linear patterns
        - Prophet alone leaves structured residuals
        - You want to capture complex interactions
        - Data has both trend/seasonal and non-linear components

        Turning off Prophet's seasonality components (setting to False) allows
        XGBoost to capture those patterns instead, which can be beneficial when
        seasonality is non-linear or interacts with other features.
    """
    args = {
        # Prophet parameters
        "growth": growth,
        "changepoint_prior_scale": changepoint_prior_scale,
        "seasonality_prior_scale": seasonality_prior_scale,
        "seasonality_mode": seasonality_mode,
        "n_changepoints": n_changepoints,
        "changepoint_range": changepoint_range,
        "yearly_seasonality": seasonality_yearly,
        "weekly_seasonality": seasonality_weekly,
        "daily_seasonality": seasonality_daily,
        # XGBoost parameters
        "trees": trees,
        "tree_depth": tree_depth,
        "learn_rate": learn_rate,
        "min_n": min_n,
        "loss_reduction": loss_reduction,
        "sample_size": sample_size,
        "mtry": mtry,
    }

    return ModelSpec(
        model_type="prophet_boost",
        engine=engine,
        mode="regression",
        args=args,
    )
