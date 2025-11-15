"""
Model Registry utilities for py-tidymodels.

This module provides helper functions for working with MLflow Model Registry,
including model registration, versioning, stage transitions, and comparison.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.exceptions import MlflowException
except ImportError:
    raise ImportError(
        "MLflow is required for registry utilities. "
        "Install with: pip install mlflow"
    )


def register_model(
    model_fit: Any,
    name: str,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    await_registration: bool = True
) -> ModelVersion:
    """
    Register model in MLflow Model Registry.

    This function saves the model and registers it in the Model Registry,
    making it available for versioning, stage transitions, and deployment.

    Args:
        model_fit: ModelFit, WorkflowFit, or NestedWorkflowFit object
        name: Name for the registered model
        tags: Optional dict of tags to apply to the model version
        description: Optional description for the model version
        await_registration: If True, wait for registration to complete

    Returns:
        ModelVersion object with version information

    Raises:
        MlflowException: If registration fails

    Examples:
        >>> from py_mlflow import register_model
        >>> from py_parsnip import linear_reg
        >>>
        >>> spec = linear_reg()
        >>> fit = spec.fit(train_data, "y ~ x1 + x2")
        >>>
        >>> # Register model
        >>> version = register_model(
        ...     fit,
        ...     name="SalesForecast",
        ...     tags={"team": "data-science", "version": "v1"},
        ...     description="Linear regression model for sales forecasting"
        ... )
        >>> print(f"Registered version: {version.version}")
    """
    from py_mlflow import save_model

    # Get or create active run
    active_run = mlflow.active_run()
    if active_run is None:
        # Ensure an experiment exists
        try:
            experiment = mlflow.get_experiment_by_name("model_registry")
            if experiment is None:
                experiment_id = mlflow.create_experiment("model_registry")
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            # Fallback to default experiment
            experiment_id = "0"

        # Create a temporary run for registration
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(run_name=f"register_{name}") as run:
            # Log model to run
            save_model(
                model_fit,
                path="model",
                registered_model_name=name
            )
            run_id = run.info.run_id
    else:
        # Use existing run
        save_model(
            model_fit,
            path="model",
            registered_model_name=name
        )
        run_id = active_run.info.run_id

    # Get client
    client = MlflowClient()

    # Get the registered model version
    model_uri = f"runs:/{run_id}/model"

    # Register model
    try:
        result = mlflow.register_model(model_uri, name)
        version_number = result.version
    except MlflowException as e:
        # Model might already be registered, get latest version
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            version_number = max([int(v.version) for v in versions])
        else:
            raise e

    # Add tags if provided
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name, version_number, key, str(value))

    # Add description if provided
    if description:
        client.update_model_version(name, version_number, description=description)

    # Get model version object
    model_version = client.get_model_version(name, version_number)

    return model_version


def get_latest_model(
    name: str,
    stage: Optional[str] = None,
    load: bool = True
) -> Union[Any, ModelVersion]:
    """
    Get latest model version from registry.

    Args:
        name: Name of registered model
        stage: Optional stage filter ("Production", "Staging", "Archived", None)
               If None, gets latest version regardless of stage
        load: If True, load and return the model object
              If False, return ModelVersion metadata only

    Returns:
        If load=True: Loaded model object (ModelFit, WorkflowFit, or NestedWorkflowFit)
        If load=False: ModelVersion object

    Raises:
        MlflowException: If model not found or no versions exist

    Examples:
        >>> from py_mlflow import get_latest_model
        >>>
        >>> # Get latest production model
        >>> model = get_latest_model("SalesForecast", stage="Production")
        >>> predictions = model.predict(test_data)
        >>>
        >>> # Get metadata only (fast)
        >>> version_info = get_latest_model("SalesForecast", load=False)
        >>> print(f"Version: {version_info.version}")
    """
    from py_mlflow import load_model

    client = MlflowClient()

    # Build filter query
    if stage:
        filter_string = f"name='{name}'"
        versions = client.search_model_versions(filter_string)
        # Filter by stage
        versions = [v for v in versions if v.current_stage == stage]
    else:
        filter_string = f"name='{name}'"
        versions = client.search_model_versions(filter_string)

    if not versions:
        stage_msg = f" in stage '{stage}'" if stage else ""
        raise MlflowException(
            f"No versions found for model '{name}'{stage_msg}"
        )

    # Get latest version (highest version number)
    latest = max(versions, key=lambda v: int(v.version))

    if load:
        # Load and return model
        model_uri = f"models:/{name}/{latest.version}"
        return load_model(model_uri)
    else:
        # Return version metadata
        return latest


def transition_model_stage(
    name: str,
    version: Union[int, str],
    stage: str,
    archive_existing: bool = True
) -> ModelVersion:
    """
    Transition model to a different stage.

    Args:
        name: Name of registered model
        version: Version number to transition
        stage: Target stage ("Production", "Staging", "Archived", "None")
        archive_existing: If True and transitioning to Production/Staging,
                         archive existing models in that stage

    Returns:
        Updated ModelVersion object

    Raises:
        ValueError: If stage is invalid
        MlflowException: If transition fails

    Examples:
        >>> from py_mlflow import transition_model_stage
        >>>
        >>> # Promote to production
        >>> version = transition_model_stage(
        ...     "SalesForecast",
        ...     version=3,
        ...     stage="Production",
        ...     archive_existing=True
        ... )
        >>>
        >>> # Archive old version
        >>> transition_model_stage("SalesForecast", version=2, stage="Archived")
    """
    valid_stages = ["Production", "Staging", "Archived", "None"]
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of {valid_stages}"
        )

    client = MlflowClient()

    # Archive existing models in target stage if requested
    if archive_existing and stage in ["Production", "Staging"]:
        existing_versions = client.search_model_versions(f"name='{name}'")
        for v in existing_versions:
            if v.current_stage == stage:
                client.transition_model_version_stage(
                    name=name,
                    version=v.version,
                    stage="Archived"
                )

    # Transition target version
    updated_version = client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage
    )

    return updated_version


def compare_model_versions(
    name: str,
    versions: Optional[List[Union[int, str]]] = None,
    test_data: Optional[pd.DataFrame] = None,
    metrics: Optional[Any] = None,
    stages: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare performance of multiple model versions.

    Args:
        name: Name of registered model
        versions: List of version numbers to compare. If None, compares all versions.
        test_data: Test dataset for evaluation (optional)
        metrics: Metric set from py_yardstick (optional, defaults to RMSE/MAE/R²)
        stages: List of stages to filter by (optional)

    Returns:
        DataFrame with version comparison including:
        - version: Version number
        - stage: Current stage
        - created: Creation timestamp
        - tags: Model tags
        - metrics: Performance metrics (if test_data provided)

    Examples:
        >>> from py_mlflow import compare_model_versions
        >>> from py_yardstick import metric_set, rmse, mae, r_squared
        >>>
        >>> # Compare all production versions
        >>> comparison = compare_model_versions(
        ...     "SalesForecast",
        ...     test_data=test_df,
        ...     metrics=metric_set(rmse, mae, r_squared),
        ...     stages=["Production", "Staging"]
        ... )
        >>> print(comparison.sort_values("rmse"))
    """
    from py_mlflow import load_model

    client = MlflowClient()

    # Get all versions
    all_versions = client.search_model_versions(f"name='{name}'")

    # Filter by specified versions
    if versions:
        versions_str = [str(v) for v in versions]
        all_versions = [v for v in all_versions if v.version in versions_str]

    # Filter by stages
    if stages:
        all_versions = [v for v in all_versions if v.current_stage in stages]

    if not all_versions:
        warnings.warn(f"No versions found for model '{name}' with given filters")
        return pd.DataFrame()

    # Build comparison data
    comparison_data = []

    for version in all_versions:
        version_data = {
            "version": int(version.version),
            "stage": version.current_stage,
            "created": pd.to_datetime(version.creation_timestamp, unit='ms'),
            "run_id": version.run_id,
        }

        # Add tags
        if version.tags:
            for key, value in version.tags.items():
                version_data[f"tag_{key}"] = value

        # Evaluate on test data if provided
        if test_data is not None:
            try:
                # Load model
                model_uri = f"models:/{name}/{version.version}"
                model = load_model(model_uri)

                # Get predictions
                predictions = model.predict(test_data)

                # Compute metrics
                if metrics is None:
                    from py_yardstick import metric_set, rmse, mae, r_squared
                    metrics = metric_set(rmse, mae, r_squared)

                # Extract outcome variable
                # Try to get from model metadata
                outcome = None
                if hasattr(model, 'blueprint') and hasattr(model.blueprint, 'outcome_name'):
                    outcome = model.blueprint.outcome_name
                elif hasattr(model, 'fit') and hasattr(model.fit, 'blueprint'):
                    blueprint = model.fit.blueprint
                    if hasattr(blueprint, 'outcome_name'):
                        outcome = blueprint.outcome_name

                # Fallback: use first column not in predictions
                if outcome is None or outcome not in test_data.columns:
                    # Assume first column is outcome
                    outcome = test_data.columns[0]

                # Compute metrics
                truth = test_data[outcome]
                estimate = predictions['.pred']

                metric_results = metrics(truth, estimate)

                # Add metrics to version data
                # Handle both long format (with '.metric' column) and wide format
                if '.metric' in metric_results.columns:
                    for _, row in metric_results.iterrows():
                        metric_name = row['.metric']
                        metric_value = row['value']
                        version_data[metric_name] = metric_value
                elif 'metric' in metric_results.columns:
                    for _, row in metric_results.iterrows():
                        metric_name = row['metric']
                        metric_value = row['value']
                        version_data[metric_name] = metric_value
                else:
                    # Wide format - metrics are columns
                    for col in metric_results.columns:
                        if col not in ['metric', '.metric', 'value']:
                            version_data[col] = metric_results[col].iloc[0]

            except Exception as e:
                warnings.warn(f"Failed to evaluate version {version.version}: {str(e)}")

        comparison_data.append(version_data)

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Sort by version descending
    comparison_df = comparison_df.sort_values("version", ascending=False)

    return comparison_df


def list_registered_models(
    filter_string: Optional[str] = None,
    max_results: int = 100
) -> pd.DataFrame:
    """
    List all registered models with metadata.

    Args:
        filter_string: Optional filter query (e.g., "name LIKE 'Sales%'")
        max_results: Maximum number of models to return

    Returns:
        DataFrame with model information including:
        - name: Model name
        - latest_version: Latest version number
        - latest_stage: Stage of latest version
        - created: Creation timestamp
        - updated: Last update timestamp
        - tags: Model-level tags

    Examples:
        >>> from py_mlflow import list_registered_models
        >>>
        >>> # List all models
        >>> models = list_registered_models()
        >>> print(models[['name', 'latest_version', 'latest_stage']])
        >>>
        >>> # Filter by name pattern
        >>> sales_models = list_registered_models(filter_string="name LIKE 'Sales%'")
    """
    client = MlflowClient()

    # Search for registered models
    if filter_string:
        try:
            models = client.search_registered_models(filter_string=filter_string, max_results=max_results)
        except Exception:
            # Fallback to listing all and filtering manually
            models = client.search_registered_models(max_results=max_results)
            # Simple name-based filtering
            if "LIKE" in filter_string.upper():
                pattern = filter_string.split("'")[1].replace('%', '')
                models = [m for m in models if pattern in m.name]
    else:
        models = client.search_registered_models(max_results=max_results)

    if not models:
        return pd.DataFrame()

    # Build model data
    model_data = []

    for model in models:
        # Get latest version
        latest_versions = model.latest_versions
        if latest_versions:
            latest = max(latest_versions, key=lambda v: int(v.version))
            latest_version = int(latest.version)
            latest_stage = latest.current_stage
            updated = pd.to_datetime(latest.last_updated_timestamp, unit='ms')
        else:
            latest_version = None
            latest_stage = None
            updated = None

        model_info = {
            "name": model.name,
            "latest_version": latest_version,
            "latest_stage": latest_stage,
            "created": pd.to_datetime(model.creation_timestamp, unit='ms') if hasattr(model, 'creation_timestamp') else None,
            "updated": updated,
            "description": model.description if hasattr(model, 'description') else None,
        }

        # Add tags if present
        if hasattr(model, 'tags') and model.tags:
            for key, value in model.tags.items():
                model_info[f"tag_{key}"] = value

        model_data.append(model_info)

    # Convert to DataFrame
    models_df = pd.DataFrame(model_data)

    # Sort by updated timestamp
    if 'updated' in models_df.columns and not models_df['updated'].isna().all():
        models_df = models_df.sort_values("updated", ascending=False)

    return models_df


def delete_model_version(
    name: str,
    version: Union[int, str]
) -> None:
    """
    Delete a specific model version.

    Args:
        name: Name of registered model
        version: Version number to delete

    Raises:
        MlflowException: If deletion fails

    Examples:
        >>> from py_mlflow import delete_model_version
        >>> delete_model_version("SalesForecast", version=1)
    """
    client = MlflowClient()
    client.delete_model_version(name, str(version))


def delete_registered_model(name: str) -> None:
    """
    Delete entire registered model and all versions.

    Args:
        name: Name of registered model

    Raises:
        MlflowException: If deletion fails

    Examples:
        >>> from py_mlflow import delete_registered_model
        >>> delete_registered_model("OldModel")
    """
    client = MlflowClient()
    client.delete_registered_model(name)
