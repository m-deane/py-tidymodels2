"""
MLflow Model Persistence - Usage Examples

This file demonstrates the py_mlflow API for model persistence.

NOTE: Currently blocked by patsy pickling limitation. These examples will work
once custom blueprint serialization is implemented (see MLFLOW_IMPLEMENTATION_SUMMARY.md).
"""

import pandas as pd
import numpy as np

from py_parsnip import linear_reg, rand_forest
from py_workflows import workflow
from py_recipes import recipe
from py_mlflow import save_model, load_model, get_model_info, validate_model_exists


# ============================================================================
# Example Data
# ============================================================================

def create_sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
        'y': np.random.randn(n)
    })


def create_grouped_data():
    """Create sample data with groups."""
    np.random.seed(42)
    n_per_group = 60

    groups = []
    for group in ['Store_A', 'Store_B', 'Store_C']:
        group_data = pd.DataFrame({
            'store': group,
            'price': np.random.uniform(10, 50, n_per_group),
            'advertising': np.random.uniform(0, 100, n_per_group),
            'sales': np.random.uniform(100, 1000, n_per_group)
        })
        groups.append(group_data)

    return pd.concat(groups, ignore_index=True)


# ============================================================================
# Example 1: Basic ModelFit Save/Load
# ============================================================================

def example_basic_modelfit():
    """Example: Save and load a basic ModelFit."""
    print("\n" + "="*70)
    print("Example 1: Basic ModelFit Save/Load")
    print("="*70)

    # Create data
    data = create_sample_data()
    train = data.iloc[:150]
    test = data.iloc[150:]

    # Fit model
    spec = linear_reg(penalty=0.1, mixture=0.5)
    fit = spec.fit(train, "y ~ x1 + x2 + x3")
    fit = fit.evaluate(test)

    print(f"\nOriginal model fitted: {fit.spec.model_type} ({fit.spec.engine})")

    # Save model
    model_path = "models/basic_model"
    fit.save_mlflow(
        path=model_path,
        signature="auto",
        input_example=train.head(5),
        metadata={"dataset": "sample", "version": "1.0"}
    )

    print(f"Model saved to: {model_path}")

    # Verify model exists
    exists = validate_model_exists(model_path)
    print(f"Model exists: {exists}")

    # Get model info without loading
    info = get_model_info(model_path)
    print(f"\nModel Info:")
    print(f"  Model Type: {info['model_type']}")
    print(f"  Engine: {info['engine']}")
    print(f"  Mode: {info['mode']}")
    print(f"  Version: {info['py_tidymodels_version']}")
    print(f"  Metadata: {info['metadata']}")

    # Load model
    loaded = load_model(model_path)
    print(f"\nModel loaded successfully")

    # Compare predictions
    preds_before = fit.predict(test)
    preds_after = loaded.predict(test)

    print(f"\nPredictions match: {preds_before.equals(preds_after)}")

    return loaded


# ============================================================================
# Example 2: WorkflowFit with Recipe
# ============================================================================

def example_workflow_with_recipe():
    """Example: Save workflow with preprocessing recipe."""
    print("\n" + "="*70)
    print("Example 2: WorkflowFit with Recipe")
    print("="*70)

    # Create data
    data = create_sample_data()

    # Create recipe with preprocessing steps
    rec = (
        recipe()
        .step_normalize()  # Normalize all numeric predictors
        .step_pca(num_comp=3)  # Reduce to 3 PCA components
    )

    # Create workflow
    wf = workflow().add_recipe(rec).add_model(rand_forest(mode='regression', trees=50))

    # Fit workflow
    wf_fit = wf.fit(data)

    print(f"\nWorkflow fitted:")
    print(f"  Model: {wf_fit.spec.model_type}")
    print(f"  Preprocessing: Recipe with {len(rec.steps)} steps")

    # Save workflow (saves both recipe and model)
    model_path = "models/workflow_model"
    wf_fit.save_mlflow(
        path=model_path,
        signature="auto",
        input_example=data.head(5),
        metadata={"recipe_steps": len(rec.steps), "pca_components": 3}
    )

    print(f"Workflow saved to: {model_path}")

    # Load workflow
    loaded = load_model(model_path)

    # Predictions (recipe applied automatically)
    test_data = create_sample_data()
    predictions = loaded.predict(test_data)

    print(f"\nPredictions generated: {len(predictions)} rows")
    print(f"Recipe preprocessing applied automatically during prediction")

    return loaded


# ============================================================================
# Example 3: Grouped/Nested Models
# ============================================================================

def example_grouped_models():
    """Example: Save per-group models."""
    print("\n" + "="*70)
    print("Example 3: Grouped/Nested Models")
    print("="*70)

    # Create grouped data
    data = create_grouped_data()
    train = data.iloc[:150]
    test = data.iloc[150:]

    print(f"\nData with {data['store'].nunique()} stores")

    # Fit separate model for each store
    spec = linear_reg()
    nested_fit = spec.fit_nested(
        train,
        formula="sales ~ price + advertising",
        group_col="store"
    )

    print(f"\nFitted {len(nested_fit.group_fits)} models (one per store)")

    # Save all group models
    model_path = "models/store_models"
    nested_fit.save_mlflow(
        path=model_path,
        metadata={"stores": list(nested_fit.group_fits.keys())}
    )

    print(f"Grouped models saved to: {model_path}")

    # Get model info
    info = get_model_info(model_path)
    print(f"\nModel Info:")
    print(f"  Is Grouped: {info['is_grouped']}")
    print(f"  Group Column: {info['group_col']}")
    print(f"  Groups: {info['groups']}")

    # Load grouped models
    loaded = load_model(model_path)

    # Predict (automatically routes to correct store model)
    predictions = loaded.predict(test)

    print(f"\nPredictions generated for {len(predictions)} rows")
    print(f"Automatic routing to correct store model")

    return loaded


# ============================================================================
# Example 4: Version Compatibility
# ============================================================================

def example_version_compatibility():
    """Example: Version compatibility checking."""
    print("\n" + "="*70)
    print("Example 4: Version Compatibility")
    print("="*70)

    # Create and save model
    data = create_sample_data()
    spec = linear_reg()
    fit = spec.fit(data, "y ~ x1 + x2")

    model_path = "models/version_test"
    fit.save_mlflow(model_path)

    # Get version info
    info = get_model_info(model_path)

    print(f"\nModel Version Info:")
    print(f"  Trained with: {info['py_tidymodels_version']}")
    print(f"  Fit timestamp: {info['fit_timestamp']}")

    # Load model (version compatibility checked automatically)
    loaded = load_model(model_path)

    print(f"\nModel loaded successfully (version compatible)")
    print(f"Compatibility check passed")


# ============================================================================
# Example 5: MLflow Model Registry Integration
# ============================================================================

def example_model_registry():
    """Example: MLflow Model Registry integration."""
    print("\n" + "="*70)
    print("Example 5: MLflow Model Registry Integration")
    print("="*70)

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Set tracking URI
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("py_tidymodels_demo")

        # Fit model
        data = create_sample_data()
        spec = linear_reg(penalty=0.1)
        fit = spec.fit(data, "y ~ x1 + x2 + x3")

        # Save with registry
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "linear_reg")
            mlflow.log_param("penalty", 0.1)

            # Log metrics
            outputs, coeffs, stats = fit.extract_outputs()
            mlflow.log_metric("train_rmse", stats[stats["split"] == "train"]["rmse"].values[0])

            # Save model to registry
            fit.save_mlflow(
                "models/registry_model",
                registered_model_name="SalesForecaster"
            )

            print(f"\nModel registered as: SalesForecaster")

        # Transition to production
        client = MlflowClient()
        client.transition_model_version_stage(
            name="SalesForecaster",
            version=1,
            stage="Production"
        )

        print(f"Model transitioned to Production stage")

        # Load from registry
        prod_model = mlflow.pyfunc.load_model("models:/SalesForecaster/Production")

        print(f"Production model loaded from registry")

    except ImportError:
        print("\nMLflow not configured - skipping registry example")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("py_mlflow - Usage Examples")
    print("="*70)
    print("\nNOTE: These examples are currently blocked by patsy pickling limitation.")
    print("They demonstrate the API design and will work once custom serialization")
    print("is implemented. See MLFLOW_IMPLEMENTATION_SUMMARY.md for details.")
    print("="*70)

    # Run examples
    try:
        # Example 1: Basic save/load
        example_basic_modelfit()

        # Example 2: Workflow with recipe
        example_workflow_with_recipe()

        # Example 3: Grouped models
        example_grouped_models()

        # Example 4: Version compatibility
        example_version_compatibility()

        # Example 5: Model registry
        example_model_registry()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n\nError: {type(e).__name__}: {e}")
        print("\nThis is expected due to patsy pickling limitation.")
        print("See MLFLOW_IMPLEMENTATION_SUMMARY.md for solution.")
