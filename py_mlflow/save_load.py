"""
Helper functions for model serialization and deserialization.

This module provides convenience functions that wrap the core flavor
save/load functionality with additional validation and utilities.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union, Dict
import pandas as pd

from py_mlflow.flavor import save_model as _save_model, load_model as _load_model
from py_mlflow.utils import get_input_example


def save_model(
    model: Any,
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    signature: Optional[Any] = None,
    input_example: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Save py-tidymodels model to MLflow format (convenience wrapper).

    This is a convenience function that wraps flavor.save_model() with
    additional validation and defaults.

    Args:
        model: ModelFit, WorkflowFit, or NestedWorkflowFit object
        path: Directory path where model will be saved
        conda_env: Conda environment specification (optional)
        signature: Model signature (optional, "auto" to infer)
        input_example: Example input for signature inference (optional)
        registered_model_name: Name for Model Registry (optional)
        metadata: Custom metadata dict (optional)
        **kwargs: Additional arguments passed to flavor.save_model()

    Returns:
        None

    Examples:
        >>> from py_mlflow import save_model
        >>> save_model(fit, "models/my_model", signature="auto")
    """
    # Validate model type
    if not hasattr(model, 'predict'):
        raise ValueError(
            "Model must have a 'predict' method. "
            "Expected ModelFit, WorkflowFit, or NestedWorkflowFit."
        )

    # Call core save function
    _save_model(
        model=model,
        path=path,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        metadata=metadata,
        **kwargs
    )


def load_model(model_uri: str, validate: bool = True) -> Any:
    """
    Load py-tidymodels model from MLflow format (convenience wrapper).

    This is a convenience function that wraps flavor.load_model() with
    optional validation.

    Args:
        model_uri: URI to model location (path, runs:/, models:/)
        validate: Whether to validate loaded model has predict method

    Returns:
        ModelFit, WorkflowFit, or NestedWorkflowFit object

    Raises:
        ValueError: If loaded model is invalid (when validate=True)

    Examples:
        >>> from py_mlflow import load_model
        >>> model = load_model("models/my_model")
        >>> predictions = model.predict(test_data)
    """
    # Load model
    model = _load_model(model_uri)

    # Validate if requested
    if validate:
        if not hasattr(model, 'predict'):
            raise ValueError(
                "Loaded object does not have 'predict' method. "
                "Model file may be corrupted."
            )

    return model


def get_model_info(model_uri: str) -> Dict[str, Any]:
    """
    Get metadata about saved model without loading full artifacts.

    Args:
        model_uri: URI to model location

    Returns:
        Dict with model metadata

    Examples:
        >>> from py_mlflow import get_model_info
        >>> info = get_model_info("models/my_model")
        >>> print(f"Model type: {info['model_type']}")
        >>> print(f"Trained: {info['fit_timestamp']}")
    """
    from mlflow.models import Model
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    from mlflow.models.model import MLMODEL_FILE_NAME

    # Download artifacts if remote
    local_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))

    # Load MLmodel file
    mlmodel_path = local_path / MLMODEL_FILE_NAME
    mlflow_model = Model.load(str(mlmodel_path))

    # Extract py_tidymodels flavor info
    flavor_conf = mlflow_model.flavors.get("py_tidymodels", {})

    return {
        "model_type": flavor_conf.get("model_type", "unknown"),
        "engine": flavor_conf.get("engine", "unknown"),
        "mode": flavor_conf.get("mode", "unknown"),
        "is_workflow": flavor_conf.get("is_workflow", False),
        "is_grouped": flavor_conf.get("is_grouped", False),
        "group_col": flavor_conf.get("group_col"),
        "groups": flavor_conf.get("groups", []),
        "py_tidymodels_version": flavor_conf.get("py_tidymodels_version"),
        "fit_timestamp": flavor_conf.get("fit_timestamp"),
        "metadata": flavor_conf.get("metadata", {}),
    }


def validate_model_exists(path: str) -> bool:
    """
    Check if model exists at given path.

    Args:
        path: Path to model directory

    Returns:
        True if model exists and appears valid

    Examples:
        >>> from py_mlflow import validate_model_exists
        >>> if validate_model_exists("models/my_model"):
        ...     model = load_model("models/my_model")
    """
    from mlflow.models.model import MLMODEL_FILE_NAME

    path = Path(path)
    mlmodel_path = path / MLMODEL_FILE_NAME
    model_pkl_path = path / "model" / "model.pkl"

    return mlmodel_path.exists() and model_pkl_path.exists()
