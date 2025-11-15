"""
Custom MLflow flavor for py-tidymodels models.

This module implements the custom MLflow flavor that enables saving and loading
ModelFit and WorkflowFit objects with full preservation of:
- Model artifacts
- Blueprints (preprocessing metadata)
- Recipes (feature engineering pipelines)
- Hyperparameters
- Training metadata
"""

import os
import platform
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import cloudpickle
import yaml
import pandas as pd

import mlflow
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow import pyfunc

from py_mlflow.utils import (
    check_version_compatibility,
    get_version_metadata,
    infer_model_signature,
    get_input_example
)

FLAVOR_NAME = "py_tidymodels"


def save_model(
    model: Any,  # ModelFit, WorkflowFit, or NestedWorkflowFit
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[Any] = None,
    input_example: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save py-tidymodels model to MLflow format.

    This function serializes a ModelFit or WorkflowFit object into MLflow's
    standard model format, including all necessary artifacts for restoration.

    Args:
        model: ModelFit, WorkflowFit, or NestedWorkflowFit object to save
        path: Directory path where model will be saved
        conda_env: Path to conda.yaml file or dict with conda environment specification.
                  If None, generates default environment.
        mlflow_model: Existing MLflow Model object (optional, for advanced use)
        signature: Model signature for input/output schema validation.
                  If "auto", infers from input_example.
                  If None and input_example provided, infers automatically.
        input_example: Example input DataFrame for signature inference (optional)
        registered_model_name: Name to register model in MLflow Model Registry (optional)
        metadata: Additional custom metadata dict (optional)

    Returns:
        None

    Raises:
        ValueError: If model type is not supported
        FileNotFoundError: If conda_env path doesn't exist

    Examples:
        >>> # Save simple ModelFit
        >>> spec = linear_reg()
        >>> fit = spec.fit(train_data, "y ~ x1 + x2")
        >>> save_model(fit, "models/my_model")
        >>>
        >>> # Save with signature and registry
        >>> save_model(
        ...     fit,
        ...     path="models/my_model",
        ...     input_example=train_data.head(5),
        ...     registered_model_name="MyModel"
        ... )
    """
    import py_parsnip

    # Create model directory
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if mlflow_model is None:
        mlflow_model = Model()

    # Infer signature if requested or if input_example provided
    if signature == "auto" or (signature is None and input_example is not None):
        predictions = model.predict(input_example)
        signature = infer_model_signature(input_example, predictions)

    # Determine model type and extract metadata
    is_workflow = hasattr(model, 'preprocessor')
    is_grouped = hasattr(model, 'group_fits')

    # Get model spec (from direct ModelFit or from WorkflowFit)
    if is_workflow:
        spec = model.spec
    else:
        spec = model.spec if hasattr(model, 'spec') else None

    # Create model data directory
    model_data_path = path / "model"
    model_data_path.mkdir(exist_ok=True)

    # Save model artifact using cloudpickle
    # Note: We use cloudpickle which handles most Python objects better than pickle
    model_pkl_path = model_data_path / "model.pkl"
    try:
        with open(model_pkl_path, "wb") as f:
            cloudpickle.dump(model, f)
    except NotImplementedError as e:
        # Handle patsy objects which don't support pickling
        # For now, we'll use dill which can handle more objects
        import dill
        with open(model_pkl_path, "wb") as f:
            dill.dump(model, f)

    # Create flavor configuration
    flavor_conf = {
        "model_artifact": "model/model.pkl",
        "is_workflow": is_workflow,
        "is_grouped": is_grouped,
    }

    # Add spec metadata if available
    if spec is not None:
        flavor_conf.update({
            "model_type": spec.model_type,
            "engine": spec.engine,
            "mode": spec.mode,
        })

    # Add grouped model metadata
    if is_grouped:
        flavor_conf["group_col"] = model.group_col
        flavor_conf["groups"] = list(model.group_fits.keys())

    # Add version metadata
    version_meta = get_version_metadata(py_parsnip.__version__)
    flavor_conf.update(version_meta)

    # Add fit timestamp
    flavor_conf["fit_timestamp"] = datetime.now().isoformat()

    # Add custom metadata if provided
    if metadata:
        flavor_conf["metadata"] = metadata

    # Add py_tidymodels flavor
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        **flavor_conf
    )

    # Add pyfunc flavor for deployment compatibility
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="py_mlflow.flavor",
        data="model",
        env=conda_env,
        python_version=platform.python_version()
    )

    # Save MLmodel file
    mlflow_model.save(str(path / MLMODEL_FILE_NAME))

    # Add signature if provided
    if signature is not None:
        mlflow_model.signature = signature
        mlflow_model.save(str(path / MLMODEL_FILE_NAME))

    # Save conda environment
    if conda_env is None:
        conda_env = _get_default_conda_env()
    elif isinstance(conda_env, str):
        # Load from file
        with open(conda_env, 'r') as f:
            conda_env = yaml.safe_load(f)

    with open(path / "conda.yaml", "w") as f:
        yaml.dump(conda_env, f, default_flow_style=False)

    # Save input example if provided
    if input_example is not None:
        example_path = path / "input_example.json"
        input_example.head(5).to_json(example_path, orient='split', index=False)

    # Register model if requested
    if registered_model_name is not None:
        mlflow.register_model(f"file://{path.absolute()}", registered_model_name)


def load_model(model_uri: str) -> Any:
    """
    Load py-tidymodels model from MLflow format.

    This function restores a ModelFit or WorkflowFit object from MLflow's
    standard model format, including all artifacts.

    Args:
        model_uri: URI to model location. Can be:
                  - Local path: "models/my_model"
                  - File URI: "file:///path/to/model"
                  - Runs URI: "runs:/<run_id>/model"
                  - Models URI: "models:/<model_name>/<version_or_stage>"

    Returns:
        ModelFit, WorkflowFit, or NestedWorkflowFit object

    Raises:
        ValueError: If model doesn't contain py_tidymodels flavor
        FileNotFoundError: If model artifacts are missing or corrupted

    Examples:
        >>> # Load from local path
        >>> model = load_model("models/my_model")
        >>>
        >>> # Load from MLflow run
        >>> model = load_model("runs:/abc123/model")
        >>>
        >>> # Load from Model Registry
        >>> model = load_model("models:/MyModel/Production")
        >>>
        >>> # Use loaded model
        >>> predictions = model.predict(test_data)
    """
    import py_parsnip

    # Download artifacts if remote URI
    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    local_path = Path(local_path)

    # Load MLmodel metadata
    mlmodel_path = local_path / MLMODEL_FILE_NAME
    if not mlmodel_path.exists():
        raise FileNotFoundError(f"MLmodel file not found at {mlmodel_path}")

    mlflow_model = Model.load(str(mlmodel_path))

    # Get py_tidymodels flavor config
    flavor_conf = mlflow_model.flavors.get(FLAVOR_NAME)
    if flavor_conf is None:
        raise ValueError(
            f"Model does not contain {FLAVOR_NAME} flavor. "
            "This may not be a py-tidymodels model."
        )

    # Check version compatibility
    check_version_compatibility(flavor_conf, py_parsnip.__version__)

    # Load model artifact
    model_artifact = flavor_conf.get("model_artifact", "model/model.pkl")
    model_path = local_path / model_artifact

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Model may be corrupted."
        )

    # Load model using cloudpickle or dill
    try:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
    except Exception:
        # Fallback to dill if cloudpickle fails
        import dill
        with open(model_path, "rb") as f:
            model = dill.load(f)

    return model


def _get_default_conda_env() -> Dict:
    """
    Get default conda environment for py-tidymodels models.

    Returns:
        Dict with conda environment specification
    """
    import py_parsnip
    import pandas
    import numpy
    import sklearn

    return {
        "name": "py_tidymodels_env",
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            f"python={platform.python_version()}",
            {
                "pip": [
                    f"py-tidymodels=={py_parsnip.__version__}",
                    f"pandas=={pandas.__version__}",
                    f"numpy=={numpy.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"mlflow=={mlflow.__version__}",
                    "cloudpickle>=2.2.0",
                    "pyyaml>=6.0",
                ]
            }
        ]
    }


class _TidymodelsPyFuncWrapper(pyfunc.PythonModel):
    """
    Wrapper to make py-tidymodels models compatible with MLflow pyfunc interface.

    This enables deployment to MLflow serving, Databricks, SageMaker, etc.
    """

    def __init__(self, model: Any):
        """
        Initialize wrapper with py-tidymodels model.

        Args:
            model: ModelFit, WorkflowFit, or NestedWorkflowFit object
        """
        self.model = model

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using py-tidymodels model.

        Args:
            context: MLflow context (unused but required by interface)
            model_input: Input data as pandas DataFrame

        Returns:
            Predictions as pandas DataFrame

        Examples:
            >>> wrapper = _TidymodelsPyFuncWrapper(model)
            >>> predictions = wrapper.predict(None, test_data)
        """
        # Call model's predict method
        predictions = self.model.predict(model_input, type="numeric")

        # Ensure output is DataFrame
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame()

        return predictions

    def load_context(self, context: Any) -> None:
        """
        Load model context (optional MLflow pyfunc hook).

        Args:
            context: MLflow context with artifacts
        """
        # No additional context loading needed
        pass


def _load_pyfunc(data_path: str) -> _TidymodelsPyFuncWrapper:
    """
    Load model for pyfunc flavor (called by MLflow).

    This function is called by MLflow when loading the pyfunc flavor.

    Args:
        data_path: Path to model data directory

    Returns:
        _TidymodelsPyFuncWrapper instance
    """
    # Load model from pickle
    model_path = Path(data_path) / "model.pkl"
    try:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
    except Exception:
        # Fallback to dill
        import dill
        with open(model_path, "rb") as f:
            model = dill.load(f)

    return _TidymodelsPyFuncWrapper(model)
