"""
Deployment utilities for py-tidymodels models.

This module provides helper functions for creating deployment-ready artifacts,
validating models before deployment, and exporting models for serving.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import json
import warnings
import tempfile
import shutil

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError(
        "MLflow is required for deployment utilities. "
        "Install with: pip install mlflow"
    )


def create_deployment_artifact(
    model_fit: Any,
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    include_tests: bool = True
) -> Path:
    """
    Create deployment-ready artifact bundle.

    This function packages the model with configuration, validation tests,
    and deployment metadata into a complete artifact bundle.

    Args:
        model_fit: ModelFit, WorkflowFit, or NestedWorkflowFit object
        config: Deployment configuration dict with keys:
                - serve_port: Port for model serving (optional)
                - batch_size: Batch size for predictions (optional)
                - timeout: Prediction timeout in seconds (optional)
                - environment: Deployment environment name (optional)
                - other custom config
        output_path: Path for output bundle. If None, creates temp directory.
        include_tests: If True, include validation tests in bundle

    Returns:
        Path to deployment artifact bundle

    Examples:
        >>> from py_mlflow import create_deployment_artifact
        >>> from py_parsnip import linear_reg
        >>>
        >>> spec = linear_reg()
        >>> fit = spec.fit(train_data, "y ~ x1 + x2")
        >>>
        >>> config = {
        ...     "serve_port": 5000,
        ...     "batch_size": 100,
        ...     "timeout": 30,
        ...     "environment": "production"
        ... }
        >>>
        >>> bundle_path = create_deployment_artifact(fit, config)
        >>> print(f"Deployment bundle created at: {bundle_path}")
    """
    from py_mlflow import save_model, get_model_info

    # Create output directory
    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="deployment_")
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path)

    # Save model
    model_path = output_path / "model"
    save_model(model_fit, str(model_path))

    # Save deployment configuration
    config_path = output_path / "deployment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Create deployment metadata
    metadata = {
        "artifact_type": "py_tidymodels_deployment",
        "created_at": pd.Timestamp.now().isoformat(),
        "config": config,
        "model_info": get_model_info(str(model_path)),
    }

    metadata_path = output_path / "deployment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create validation tests if requested
    if include_tests:
        tests_path = output_path / "validation_tests.json"
        tests = {
            "tests": [
                {
                    "name": "model_loads",
                    "type": "artifact_check",
                    "required": True
                },
                {
                    "name": "prediction_shape",
                    "type": "prediction_check",
                    "required": True
                },
                {
                    "name": "prediction_no_nan",
                    "type": "prediction_check",
                    "required": True
                }
            ]
        }
        with open(tests_path, 'w') as f:
            json.dump(tests, f, indent=2)

    # Create README
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Deployment Artifact Bundle

## Model Information
- Type: {metadata['model_info']['model_type']}
- Engine: {metadata['model_info']['engine']}
- Mode: {metadata['model_info']['mode']}

## Deployment Configuration
```json
{json.dumps(config, indent=2)}
```

## Files
- `model/`: MLflow model artifacts
- `deployment_config.json`: Deployment configuration
- `deployment_metadata.json`: Artifact metadata
- `validation_tests.json`: Validation test specifications
- `README.md`: This file

## Usage
Load and use the model:
```python
from py_mlflow import load_model

model = load_model("model")
predictions = model.predict(test_data)
```

## Validation
Run validation tests before deployment:
```python
from py_mlflow import validate_deployment

validate_deployment(".", test_data)
```
""")

    return output_path


def validate_deployment(
    model_path: str,
    test_data: pd.DataFrame,
    expected_metrics: Optional[Dict[str, float]] = None,
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Validate model meets deployment criteria.

    This function runs a series of validation checks to ensure the model
    is ready for deployment, including artifact checks, prediction checks,
    and performance validation.

    Args:
        model_path: Path to model or deployment bundle
        test_data: Test dataset for validation
        expected_metrics: Optional dict of expected metric values
                         (e.g., {"rmse": 100, "mae": 80})
        tolerance: Tolerance for metric comparisons (fraction, default 0.1 = 10%)

    Returns:
        Dict with validation results:
        - passed: Boolean indicating if all checks passed
        - checks: List of individual check results
        - metrics: Computed metrics (if applicable)
        - errors: List of errors encountered

    Raises:
        FileNotFoundError: If model artifacts are missing

    Examples:
        >>> from py_mlflow import validate_deployment
        >>>
        >>> # Basic validation
        >>> results = validate_deployment(
        ...     "models/my_model",
        ...     test_data
        ... )
        >>> assert results['passed'], f"Validation failed: {results['errors']}"
        >>>
        >>> # Validation with performance criteria
        >>> expected = {"rmse": 100, "mae": 75}
        >>> results = validate_deployment(
        ...     "models/my_model",
        ...     test_data,
        ...     expected_metrics=expected,
        ...     tolerance=0.1  # Allow 10% deviation
        ... )
    """
    from py_mlflow import load_model, validate_model_exists

    model_path = Path(model_path)
    results = {
        "passed": True,
        "checks": [],
        "metrics": {},
        "errors": []
    }

    # Check 1: Model artifacts exist
    try:
        # Check if it's a deployment bundle
        if (model_path / "model").exists():
            actual_model_path = model_path / "model"
        else:
            actual_model_path = model_path

        if not validate_model_exists(str(actual_model_path)):
            results["passed"] = False
            results["errors"].append("Model artifacts not found or incomplete")
            results["checks"].append({
                "name": "artifacts_exist",
                "passed": False,
                "message": "Missing model artifacts"
            })
            return results
        else:
            results["checks"].append({
                "name": "artifacts_exist",
                "passed": True,
                "message": "Model artifacts found"
            })
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Artifact check failed: {str(e)}")
        results["checks"].append({
            "name": "artifacts_exist",
            "passed": False,
            "message": str(e)
        })
        return results

    # Check 2: Model loads successfully
    try:
        model = load_model(str(actual_model_path))
        results["checks"].append({
            "name": "model_loads",
            "passed": True,
            "message": "Model loaded successfully"
        })
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Model loading failed: {str(e)}")
        results["checks"].append({
            "name": "model_loads",
            "passed": False,
            "message": str(e)
        })
        return results

    # Check 3: Predictions work
    try:
        predictions = model.predict(test_data)
        results["checks"].append({
            "name": "predictions_work",
            "passed": True,
            "message": f"Generated {len(predictions)} predictions"
        })
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Prediction failed: {str(e)}")
        results["checks"].append({
            "name": "predictions_work",
            "passed": False,
            "message": str(e)
        })
        return results

    # Check 4: Predictions have correct shape
    if len(predictions) != len(test_data):
        results["passed"] = False
        results["errors"].append(
            f"Prediction count mismatch: expected {len(test_data)}, got {len(predictions)}"
        )
        results["checks"].append({
            "name": "prediction_shape",
            "passed": False,
            "message": f"Expected {len(test_data)} predictions, got {len(predictions)}"
        })
    else:
        results["checks"].append({
            "name": "prediction_shape",
            "passed": True,
            "message": f"Predictions shape correct: {len(predictions)}"
        })

    # Check 5: No NaN predictions
    if predictions['.pred'].isna().any():
        nan_count = predictions['.pred'].isna().sum()
        results["passed"] = False
        results["errors"].append(f"Found {nan_count} NaN predictions")
        results["checks"].append({
            "name": "no_nan_predictions",
            "passed": False,
            "message": f"Found {nan_count} NaN values"
        })
    else:
        results["checks"].append({
            "name": "no_nan_predictions",
            "passed": True,
            "message": "No NaN predictions"
        })

    # Check 6: Performance metrics (if expected values provided)
    if expected_metrics:
        try:
            from py_yardstick import rmse, mae, r_squared

            # Extract outcome variable
            outcome = None
            if hasattr(model, 'blueprint') and hasattr(model.blueprint, 'outcome_name'):
                outcome = model.blueprint.outcome_name
            elif hasattr(model, 'fit') and hasattr(model.fit, 'blueprint'):
                blueprint = model.fit.blueprint
                if hasattr(blueprint, 'outcome_name'):
                    outcome = blueprint.outcome_name
                elif isinstance(blueprint, dict):
                    outcome = blueprint.get('outcome_name')

            # Fallback: try to extract from test_data
            if outcome is None or outcome not in test_data.columns:
                # Assume 'y' is the outcome, or use first column
                if 'y' in test_data.columns:
                    outcome = 'y'
                else:
                    outcome = test_data.columns[0]

            truth = test_data[outcome].values
            estimate = predictions['.pred'].values

            # Compute metrics
            metric_functions = {
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared
            }

            for metric_name, expected_value in expected_metrics.items():
                if metric_name in metric_functions:
                    try:
                        metric_fn = metric_functions[metric_name]
                        actual_value = metric_fn(truth, estimate).iloc[0]['value']
                        results["metrics"][metric_name] = actual_value

                        # Check if within tolerance
                        if metric_name == 'r_squared':
                            # For R², higher is better
                            threshold = expected_value * (1 - tolerance)
                            passed = actual_value >= threshold
                            message = f"{metric_name}: {actual_value:.4f} (expected >= {threshold:.4f})"
                        else:
                            # For RMSE/MAE, lower is better
                            threshold = expected_value * (1 + tolerance)
                            passed = actual_value <= threshold
                            message = f"{metric_name}: {actual_value:.4f} (expected <= {threshold:.4f})"

                        if not passed:
                            results["passed"] = False
                            results["errors"].append(f"Metric {metric_name} outside tolerance")

                        results["checks"].append({
                            "name": f"metric_{metric_name}",
                            "passed": passed,
                            "message": message,
                            "actual": actual_value,
                            "expected": expected_value,
                            "tolerance": tolerance
                        })
                    except Exception as e:
                        results["errors"].append(f"Failed to compute {metric_name}: {str(e)}")
        except Exception as e:
            results["errors"].append(f"Metric computation failed: {str(e)}")

    return results


def create_model_card(
    model_fit: Any,
    metrics: Optional[Dict[str, float]] = None,
    data_info: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate model card markdown documentation.

    Creates comprehensive model documentation following model card best practices,
    including model details, intended use, performance metrics, and limitations.

    Args:
        model_fit: ModelFit, WorkflowFit, or NestedWorkflowFit object
        metrics: Optional dict of performance metrics
        data_info: Optional dict with training data information:
                   - dataset_name: Name of dataset
                   - n_samples: Number of training samples
                   - n_features: Number of features
                   - date_range: Training data date range
                   - other custom info
        output_path: Path to save model card markdown file (optional)

    Returns:
        Model card content as markdown string

    Examples:
        >>> from py_mlflow import create_model_card
        >>>
        >>> metrics = {"rmse": 95.3, "mae": 72.1, "r_squared": 0.89}
        >>> data_info = {
        ...     "dataset_name": "sales_data",
        ...     "n_samples": 10000,
        ...     "n_features": 15,
        ...     "date_range": "2020-01-01 to 2023-12-31"
        ... }
        >>>
        >>> card = create_model_card(
        ...     fit,
        ...     metrics=metrics,
        ...     data_info=data_info,
        ...     output_path="MODEL_CARD.md"
        ... )
    """
    from datetime import datetime

    # Extract model metadata
    is_workflow = hasattr(model_fit, 'workflow') or (hasattr(model_fit, 'pre') and hasattr(model_fit, 'fit'))
    is_grouped = hasattr(model_fit, 'group_fits')

    # Get model spec
    if is_workflow and not is_grouped:
        spec = model_fit.fit.spec if hasattr(model_fit.fit, 'spec') else None
    elif is_grouped and hasattr(model_fit, 'workflow'):
        # NestedWorkflowFit
        first_group_fit = next(iter(model_fit.group_fits.values()))
        spec = first_group_fit.fit.spec if hasattr(first_group_fit.fit, 'spec') else None
    else:
        spec = model_fit.spec if hasattr(model_fit, 'spec') else None

    # Build model card content
    card_content = f"""# Model Card

## Model Details
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    if spec:
        card_content += f"""**Model Type:** {spec.model_type}
**Engine:** {spec.engine}
**Mode:** {spec.mode}
"""

    if is_workflow:
        card_content += "**Pipeline:** Workflow with preprocessing\n"

    if is_grouped:
        card_content += f"**Grouped Model:** Yes (group column: {model_fit.group_col})\n"
        card_content += f"**Number of Groups:** {len(model_fit.group_fits)}\n"

    card_content += "\n## Intended Use\n"
    card_content += "**Primary Use Case:** [Describe the primary use case]\n\n"
    card_content += "**Users:** [Describe intended users]\n\n"
    card_content += "**Out-of-Scope Uses:** [Describe inappropriate uses]\n\n"

    # Training data section
    card_content += "## Training Data\n"
    if data_info:
        if 'dataset_name' in data_info:
            card_content += f"**Dataset:** {data_info['dataset_name']}\n\n"
        if 'n_samples' in data_info:
            card_content += f"**Training Samples:** {data_info['n_samples']:,}\n\n"
        if 'n_features' in data_info:
            card_content += f"**Features:** {data_info['n_features']}\n\n"
        if 'date_range' in data_info:
            card_content += f"**Date Range:** {data_info['date_range']}\n\n"

        # Add any other custom data info
        for key, value in data_info.items():
            if key not in ['dataset_name', 'n_samples', 'n_features', 'date_range']:
                card_content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    else:
        card_content += "[Provide training data information]\n\n"

    # Performance metrics section
    card_content += "## Performance Metrics\n"
    if metrics:
        card_content += "| Metric | Value |\n"
        card_content += "|--------|-------|\n"
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                card_content += f"| {metric_name.upper()} | {value:.4f} |\n"
            else:
                card_content += f"| {metric_name.upper()} | {value} |\n"
        card_content += "\n"
    else:
        card_content += "[Provide performance metrics]\n\n"

    # Model parameters
    if spec and spec.args:
        card_content += "## Model Parameters\n"
        if isinstance(spec.args, dict):
            for param, value in spec.args.items():
                card_content += f"- **{param}:** {value}\n"
        card_content += "\n"

    # Additional sections
    card_content += """## Ethical Considerations
[Describe any ethical considerations, biases, or fairness concerns]

## Limitations
[Describe model limitations and known failure modes]

## Trade-offs
[Describe trade-offs made during model development]

## Usage Example
```python
from py_mlflow import load_model

# Load model
model = load_model("path/to/model")

# Make predictions
predictions = model.predict(test_data)
```

## Contact
**Maintainer:** [Name/Team]
**Email:** [Contact email]
"""

    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(card_content)

    return card_content


def export_for_serving(
    model_fit: Any,
    path: str,
    format: str = "mlflow",
    include_example: bool = True,
    sample_data: Optional[pd.DataFrame] = None
) -> Path:
    """
    Export model for serving in various formats.

    Args:
        model_fit: ModelFit, WorkflowFit, or NestedWorkflowFit object
        path: Output path for exported model
        format: Export format - "mlflow", "python" (default: "mlflow")
        include_example: If True, include input example in export
        sample_data: Sample data for input example (optional)

    Returns:
        Path to exported model

    Raises:
        ValueError: If format is not supported

    Examples:
        >>> from py_mlflow import export_for_serving
        >>>
        >>> # Export for MLflow serving
        >>> export_path = export_for_serving(
        ...     fit,
        ...     path="exports/my_model",
        ...     format="mlflow",
        ...     sample_data=train_data.head(5)
        ... )
        >>>
        >>> # Serve with: mlflow models serve -m <export_path>
    """
    from py_mlflow import save_model

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if format == "mlflow":
        # Standard MLflow export
        input_example = sample_data.head(5) if (include_example and sample_data is not None) else None

        save_model(
            model_fit,
            path=str(path),
            signature="auto" if sample_data is not None else None,
            input_example=input_example
        )

        # Create serving instructions
        serving_instructions = f"""# Model Serving Instructions

## Start MLflow Serving
```bash
mlflow models serve -m {path.absolute()} -p 5000
```

## Test Endpoint
```bash
curl -X POST http://localhost:5000/invocations \\
  -H 'Content-Type: application/json' \\
  -d '{{"inputs": [[value1, value2, ...]]}}'
```

## Python Client
```python
import requests
import pandas as pd

data = pd.DataFrame({{"x1": [1, 2], "x2": [3, 4]}})
response = requests.post(
    "http://localhost:5000/invocations",
    json={{"inputs": data.to_dict(orient='split')}}
)
predictions = response.json()
```
"""
        with open(path / "SERVING.md", 'w') as f:
            f.write(serving_instructions)

    elif format == "python":
        # Export as standalone Python package
        save_model(model_fit, str(path / "model"))

        # Create wrapper script
        wrapper_script = """
import sys
from pathlib import Path
import pandas as pd
from py_mlflow import load_model

# Load model
model = load_model(str(Path(__file__).parent / "model"))

def predict(data):
    \"\"\"Make predictions on input data.\"\"\"
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    return model.predict(data)

if __name__ == "__main__":
    # Example usage
    example_data = pd.DataFrame({
        "x1": [1, 2, 3],
        "x2": [4, 5, 6]
    })
    predictions = predict(example_data)
    print(predictions)
"""
        with open(path / "predictor.py", 'w') as f:
            f.write(wrapper_script)

    else:
        raise ValueError(f"Unsupported export format: {format}. Use 'mlflow' or 'python'")

    return path
