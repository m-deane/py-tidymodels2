"""
Null Model specification

A baseline model that predicts a constant value (mean, median, or last)
for all observations. Essential for benchmarking other models.

Supports:
- regression: predicts mean, median, or last value
- classification: predicts mode (most frequent class)

Parameters:
- strategy: Strategy for baseline prediction
"""

from typing import Optional, Literal
from py_parsnip.model_spec import ModelSpec


def null_model(
    strategy: Literal["mean", "median", "last"] = "mean",
    engine: str = "parsnip",
) -> ModelSpec:
    """
    Create a null/baseline model specification.

    The null model predicts a constant value for all observations:
    - Regression: mean, median, or last value of training outcomes
    - Classification: mode (most frequent class) of training outcomes

    This is the simplest possible model and serves as a critical
    baseline. If your model can't beat the null model, it's useless.

    Args:
        strategy: Strategy for baseline prediction (default "mean")
            - "mean": Predict the mean of training outcomes (regression)
            - "median": Predict the median of training outcomes (regression)
            - "last": Predict the last observed value (regression, useful for time series)
            For classification mode, always uses "mode" (most frequent class)
        engine: Computational engine to use (default "parsnip")
            - "parsnip": Custom implementation

    Returns:
        ModelSpec for null model

    Examples:
        >>> # Null model for regression (predicts mean)
        >>> spec = null_model(strategy="mean")
        >>> spec = spec.set_mode('regression')
        >>>
        >>> # Null model predicting median
        >>> spec = null_model(strategy="median")
        >>> spec = spec.set_mode('regression')
        >>>
        >>> # Null model predicting last value (time series baseline)
        >>> spec = null_model(strategy="last")
        >>> spec = spec.set_mode('regression')
        >>>
        >>> # Null model for classification (predicts mode)
        >>> spec = null_model()
        >>> spec = spec.set_mode('classification')

    Notes:
        - For regression: strategy parameter controls baseline statistic
        - For classification: always uses mode (most frequent class)
        - No hyperparameters to tune
        - Extremely fast training and prediction
        - Essential baseline for ANY modeling project
        - "last" strategy useful for time series naive baseline
    """
    args = {"strategy": strategy}

    return ModelSpec(
        model_type="null_model",
        engine=engine,
        mode="regression",  # Default mode, can be changed with set_mode()
        args=args,
    )
