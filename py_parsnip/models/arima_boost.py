"""
ARIMA + XGBoost hybrid model specification

This is a hybrid time series model that combines:
1. ARIMA: Captures linear patterns, trend, and autocorrelation
2. XGBoost: Captures non-linear patterns in the residuals

Strategy:
- Fit ARIMA model to capture linear temporal patterns
- Fit XGBoost on ARIMA residuals to capture non-linear patterns
- Final prediction = ARIMA prediction + XGBoost prediction

Parameters:
ARIMA parameters:
- seasonal_period: Seasonality period (e.g., 12 for monthly data)
- non_seasonal_ar: Number of non-seasonal AR terms (p)
- non_seasonal_differences: Number of non-seasonal differences (d)
- non_seasonal_ma: Number of non-seasonal MA terms (q)
- seasonal_ar: Number of seasonal AR terms (P)
- seasonal_differences: Number of seasonal differences (D)
- seasonal_ma: Number of seasonal MA terms (Q)

XGBoost parameters:
- trees: Number of boosting iterations (n_estimators)
- tree_depth: Maximum tree depth (max_depth)
- learn_rate: Learning rate (eta/learning_rate)
- min_n: Minimum samples in leaf (min_child_weight)
- loss_reduction: Minimum loss reduction (gamma)
- sample_size: Subsample ratio (subsample)
- mtry: Feature sampling ratio (colsample_bytree)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def arima_boost(
    # ARIMA parameters
    seasonal_period: Optional[int] = None,
    non_seasonal_ar: int = 0,
    non_seasonal_differences: int = 0,
    non_seasonal_ma: int = 0,
    seasonal_ar: int = 0,
    seasonal_differences: int = 0,
    seasonal_ma: int = 0,
    # XGBoost parameters
    trees: int = 100,
    tree_depth: int = 6,
    learn_rate: float = 0.1,
    min_n: int = 1,
    loss_reduction: float = 0.0,
    sample_size: float = 1.0,
    mtry: float = 1.0,
    engine: str = "hybrid_arima_xgboost",
) -> ModelSpec:
    """
    Create an ARIMA + XGBoost hybrid model specification.

    This hybrid model combines the strengths of both methods:
    - ARIMA captures linear temporal patterns and autocorrelation
    - XGBoost captures non-linear patterns in the residuals

    The model works in two stages:
    1. Fit ARIMA to the time series
    2. Fit XGBoost to the ARIMA residuals
    3. Final prediction = ARIMA pred + XGBoost pred

    Args:
        seasonal_period: Seasonal period for ARIMA (e.g., 12 for monthly)
        non_seasonal_ar: ARIMA AR order (p)
        non_seasonal_differences: ARIMA differencing order (d)
        non_seasonal_ma: ARIMA MA order (q)
        seasonal_ar: Seasonal AR order (P)
        seasonal_differences: Seasonal differencing order (D)
        seasonal_ma: Seasonal MA order (Q)
        trees: Number of boosting trees (default 100)
        tree_depth: Maximum tree depth (default 6)
        learn_rate: XGBoost learning rate (default 0.1)
        min_n: Minimum samples in leaf (default 1)
        loss_reduction: Minimum loss reduction for split (default 0.0)
        sample_size: Row subsample ratio (default 1.0)
        mtry: Feature subsample ratio (default 1.0)
        engine: Computational engine (default "hybrid_arima_xgboost")

    Returns:
        ModelSpec for ARIMA + XGBoost hybrid

    Examples:
        >>> # Simple ARIMA(1,1,1) + XGBoost
        >>> spec = arima_boost(
        ...     non_seasonal_ar=1,
        ...     non_seasonal_differences=1,
        ...     non_seasonal_ma=1,
        ...     trees=100,
        ...     tree_depth=3
        ... )

        >>> # Seasonal ARIMA + XGBoost
        >>> spec = arima_boost(
        ...     seasonal_period=12,
        ...     non_seasonal_ar=1,
        ...     non_seasonal_differences=1,
        ...     non_seasonal_ma=1,
        ...     seasonal_ar=1,
        ...     seasonal_differences=1,
        ...     seasonal_ma=1,
        ...     trees=200,
        ...     learn_rate=0.05
        ... )

    Note:
        The hybrid approach is particularly effective when:
        - Data has both linear and non-linear patterns
        - ARIMA alone leaves structured residuals
        - You want to capture complex interactions
    """
    args = {
        # ARIMA parameters
        "seasonal_period": seasonal_period,
        "non_seasonal_ar": non_seasonal_ar,
        "non_seasonal_differences": non_seasonal_differences,
        "non_seasonal_ma": non_seasonal_ma,
        "seasonal_ar": seasonal_ar,
        "seasonal_differences": seasonal_differences,
        "seasonal_ma": seasonal_ma,
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
        model_type="arima_boost",
        engine=engine,
        mode="regression",
        args=args,
    )
