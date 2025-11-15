"""
py-mlflow: MLflow integration for py-tidymodels

This package provides comprehensive MLflow integration including:
- Custom MLflow flavor for ModelFit/WorkflowFit serialization
- Model persistence with full artifact preservation
- Model signature inference and validation
- Version compatibility checking
- Experiment tracking integration (future)
- Model registry utilities (future)

Core Functions:
    save_model: Save ModelFit/WorkflowFit to MLflow format
    load_model: Load model from MLflow format
    get_model_info: Get model metadata without loading artifacts

Examples:
    >>> from py_mlflow import save_model, load_model
    >>> from py_parsnip import linear_reg
    >>>
    >>> # Train and save model
    >>> spec = linear_reg()
    >>> fit = spec.fit(train_data, "y ~ x1 + x2")
    >>> save_model(fit, "models/my_model", signature="auto")
    >>>
    >>> # Load and use model
    >>> loaded = load_model("models/my_model")
    >>> predictions = loaded.predict(test_data)
"""

from py_mlflow.flavor import save_model as _flavor_save_model
from py_mlflow.flavor import load_model as _flavor_load_model
from py_mlflow.save_load import (
    save_model,
    load_model,
    get_model_info,
    validate_model_exists
)

__version__ = "0.1.0"

__all__ = [
    "save_model",
    "load_model",
    "get_model_info",
    "validate_model_exists",
]
