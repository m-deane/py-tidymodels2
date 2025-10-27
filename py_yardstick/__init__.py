"""
py-yardstick: Performance metrics for model evaluation

Provides tidymodels-style metric functions for evaluating model performance.
All metrics return standardized DataFrames with columns: metric, value.
"""

from .metrics import (
    # Time Series Metrics
    rmse,
    mae,
    mape,
    smape,
    mase,
    r_squared,
    rsq_trad,

    # Residual Diagnostic Tests
    durbin_watson,
    ljung_box,
    shapiro_wilk,
    adf_test,

    # Classification Metrics
    accuracy,
    precision,
    recall,
    f_meas,
    roc_auc,

    # Additional Regression Metrics
    mda,

    # Metric Set Composer
    metric_set,
)

__all__ = [
    # Time Series Metrics
    "rmse",
    "mae",
    "mape",
    "smape",
    "mase",
    "r_squared",
    "rsq_trad",

    # Residual Diagnostic Tests
    "durbin_watson",
    "ljung_box",
    "shapiro_wilk",
    "adf_test",

    # Classification Metrics
    "accuracy",
    "precision",
    "recall",
    "f_meas",
    "roc_auc",

    # Additional Regression Metrics
    "mda",

    # Metric Set Composer
    "metric_set",
]

__version__ = "0.1.0"
