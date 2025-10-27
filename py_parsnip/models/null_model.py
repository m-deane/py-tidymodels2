"""
Null Model specification

A baseline model that predicts a constant value (mean, median, or mode)
for all observations. Essential for benchmarking other models.

Supports:
- regression: predicts mean or median
- classification: predicts mode (most frequent class)

Parameters:
None - this is a parameter-free baseline model
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def null_model(
    engine: str = "parsnip",
) -> ModelSpec:
    """
    Create a null/baseline model specification.

    The null model predicts a constant value for all observations:
    - Regression: mean or median of training outcomes
    - Classification: mode (most frequent class) of training outcomes

    This is the simplest possible model and serves as a critical
    baseline. If your model can't beat the null model, it's useless.

    Args:
        engine: Computational engine to use (default "parsnip")
            - "parsnip": Custom implementation (mean for regression)

    Returns:
        ModelSpec for null model

    Examples:
        >>> # Null model for regression (predicts mean)
        >>> spec = null_model()
        >>> spec = spec.set_mode('regression')
        >>>
        >>> # Null model for classification (predicts mode)
        >>> spec = null_model()
        >>> spec = spec.set_mode('classification')

    Notes:
        - For regression: predicts mean by default (can be changed to median)
        - For classification: predicts most frequent class
        - No hyperparameters to tune
        - Extremely fast training and prediction
        - Essential baseline for ANY modeling project
    """
    return ModelSpec(
        model_type="null_model",
        engine=engine,
        mode="regression",  # Default mode, can be changed with set_mode()
        args={},  # No parameters
    )
