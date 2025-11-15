"""
py-mlflow: MLflow integration for py-tidymodels

This package provides comprehensive MLflow integration including:
- Custom MLflow flavor for ModelFit/WorkflowFit serialization
- Model persistence with full artifact preservation
- Model signature inference and validation
- Version compatibility checking
- Experiment tracking integration
- Model registry utilities
- Deployment helpers
- Performance monitoring

Core Functions:
    save_model: Save ModelFit/WorkflowFit to MLflow format
    load_model: Load model from MLflow format
    get_model_info: Get model metadata without loading artifacts

Registry Functions:
    register_model: Register model in MLflow Model Registry
    get_latest_model: Get latest model version
    transition_model_stage: Transition model stage
    compare_model_versions: Compare model versions
    list_registered_models: List all registered models

Deployment Functions:
    create_deployment_artifact: Create deployment bundle
    validate_deployment: Validate model for deployment
    create_model_card: Generate model documentation
    export_for_serving: Export model for serving

Monitoring Functions:
    log_prediction_batch: Log predictions for monitoring
    detect_data_drift: Detect data distribution drift
    monitor_model_performance: Track performance over time
    create_monitoring_dashboard_data: Create dashboard data

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
    >>>
    >>> # Register and deploy
    >>> from py_mlflow import register_model, transition_model_stage
    >>> version = register_model(fit, "MyModel")
    >>> transition_model_stage("MyModel", version.version, "Production")
"""

from py_mlflow.flavor import save_model as _flavor_save_model
from py_mlflow.flavor import load_model as _flavor_load_model
from py_mlflow.save_load import (
    save_model,
    load_model,
    get_model_info,
    validate_model_exists
)
from py_mlflow.registry import (
    register_model,
    get_latest_model,
    transition_model_stage,
    compare_model_versions,
    list_registered_models,
    delete_model_version,
    delete_registered_model
)
from py_mlflow.deployment import (
    create_deployment_artifact,
    validate_deployment,
    create_model_card,
    export_for_serving
)
from py_mlflow.monitoring import (
    log_prediction_batch,
    detect_data_drift,
    monitor_model_performance,
    create_monitoring_dashboard_data
)

__version__ = "0.2.0"

__all__ = [
    # Save/Load
    "save_model",
    "load_model",
    "get_model_info",
    "validate_model_exists",
    # Registry
    "register_model",
    "get_latest_model",
    "transition_model_stage",
    "compare_model_versions",
    "list_registered_models",
    "delete_model_version",
    "delete_registered_model",
    # Deployment
    "create_deployment_artifact",
    "validate_deployment",
    "create_model_card",
    "export_for_serving",
    # Monitoring
    "log_prediction_batch",
    "detect_data_drift",
    "monitor_model_performance",
    "create_monitoring_dashboard_data",
]
