"""
Tests for MLflow deployment utilities.

Tests deployment artifact creation, validation, model cards, and export.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from py_parsnip import linear_reg
from py_mlflow import (
    create_deployment_artifact,
    validate_deployment,
    create_model_card,
    export_for_serving,
    save_model
)


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'y': 2 * np.random.randn(n) + 10  # Mean around 10
    })


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model."""
    spec = linear_reg()
    train_data = sample_data.iloc[:80]
    fit = spec.fit(train_data, "y ~ x1 + x2")
    return fit


@pytest.fixture
def temp_dir():
    """Create temporary directory for artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestDeploymentArtifact:
    """Test deployment artifact creation."""

    def test_create_basic_artifact(self, trained_model, temp_dir):
        """Test creating basic deployment artifact."""
        config = {
            "serve_port": 5000,
            "batch_size": 100
        }

        bundle_path = create_deployment_artifact(
            trained_model,
            config=config,
            output_path=str(temp_dir / "deploy_bundle")
        )

        assert bundle_path.exists()
        assert (bundle_path / "model").exists()
        assert (bundle_path / "deployment_config.json").exists()
        assert (bundle_path / "deployment_metadata.json").exists()
        assert (bundle_path / "README.md").exists()

    def test_artifact_with_validation_tests(self, trained_model, temp_dir):
        """Test that validation tests are included in artifact."""
        config = {"environment": "production"}

        bundle_path = create_deployment_artifact(
            trained_model,
            config=config,
            output_path=str(temp_dir / "bundle_with_tests"),
            include_tests=True
        )

        assert (bundle_path / "validation_tests.json").exists()

    def test_artifact_without_tests(self, trained_model, temp_dir):
        """Test creating artifact without validation tests."""
        config = {"environment": "staging"}

        bundle_path = create_deployment_artifact(
            trained_model,
            config=config,
            output_path=str(temp_dir / "bundle_no_tests"),
            include_tests=False
        )

        assert not (bundle_path / "validation_tests.json").exists()

    def test_artifact_temp_directory(self, trained_model):
        """Test creating artifact in temporary directory."""
        config = {"timeout": 30}

        # Don't specify output_path - should create temp dir
        bundle_path = create_deployment_artifact(
            trained_model,
            config=config
        )

        try:
            assert bundle_path.exists()
            assert (bundle_path / "model").exists()
        finally:
            # Clean up temp directory
            if bundle_path.exists():
                shutil.rmtree(bundle_path)


class TestDeploymentValidation:
    """Test deployment validation."""

    def test_validate_basic_deployment(self, trained_model, sample_data, temp_dir):
        """Test basic deployment validation."""
        # Save model
        model_path = temp_dir / "test_model"
        save_model(trained_model, str(model_path))

        # Validate
        test_data = sample_data.iloc[80:]
        results = validate_deployment(
            str(model_path),
            test_data
        )

        assert results["passed"] is True
        assert len(results["checks"]) > 0
        assert len(results["errors"]) == 0

    def test_validate_with_metrics(self, trained_model, sample_data, temp_dir):
        """Test validation with expected metrics."""
        # Save model
        model_path = temp_dir / "metrics_model"
        save_model(trained_model, str(model_path))

        # Validate with loose metrics (should pass)
        test_data = sample_data.iloc[80:]
        expected_metrics = {
            "rmse": 10.0,  # High threshold, should pass
            "mae": 10.0
        }

        results = validate_deployment(
            str(model_path),
            test_data,
            expected_metrics=expected_metrics,
            tolerance=0.5  # 50% tolerance
        )

        assert "rmse" in results["metrics"]
        assert "mae" in results["metrics"]

    def test_validate_deployment_bundle(self, trained_model, sample_data, temp_dir):
        """Test validating a deployment bundle."""
        # Create bundle
        config = {"serve_port": 5000}
        bundle_path = create_deployment_artifact(
            trained_model,
            config=config,
            output_path=str(temp_dir / "validate_bundle")
        )

        # Validate the bundle
        test_data = sample_data.iloc[80:]
        results = validate_deployment(
            str(bundle_path),
            test_data
        )

        assert results["passed"] is True

    def test_validate_missing_model(self, sample_data, temp_dir):
        """Test validation fails for missing model."""
        nonexistent_path = temp_dir / "nonexistent"

        results = validate_deployment(
            str(nonexistent_path),
            sample_data
        )

        assert results["passed"] is False
        assert len(results["errors"]) > 0

    def test_validate_checks_structure(self, trained_model, sample_data, temp_dir):
        """Test that validation returns proper check structure."""
        model_path = temp_dir / "check_model"
        save_model(trained_model, str(model_path))

        results = validate_deployment(str(model_path), sample_data)

        # Check structure
        assert "passed" in results
        assert "checks" in results
        assert "metrics" in results
        assert "errors" in results

        # Verify checks have required fields
        for check in results["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "message" in check


class TestModelCard:
    """Test model card generation."""

    def test_create_basic_model_card(self, trained_model):
        """Test creating basic model card."""
        card = create_model_card(trained_model)

        assert "Model Card" in card
        assert "Model Details" in card
        assert "linear_reg" in card

    def test_model_card_with_metrics(self, trained_model):
        """Test model card with performance metrics."""
        metrics = {
            "rmse": 2.5,
            "mae": 1.8,
            "r_squared": 0.85
        }

        card = create_model_card(
            trained_model,
            metrics=metrics
        )

        assert "2.5" in card or "2.500" in card
        assert "Performance Metrics" in card

    def test_model_card_with_data_info(self, trained_model):
        """Test model card with data information."""
        data_info = {
            "dataset_name": "sales_data",
            "n_samples": 10000,
            "n_features": 15,
            "date_range": "2020-01 to 2023-12"
        }

        card = create_model_card(
            trained_model,
            data_info=data_info
        )

        assert "sales_data" in card
        assert "10,000" in card or "10000" in card
        assert "Training Data" in card

    def test_model_card_saved_to_file(self, trained_model, temp_dir):
        """Test saving model card to file."""
        output_path = temp_dir / "MODEL_CARD.md"

        card = create_model_card(
            trained_model,
            output_path=str(output_path)
        )

        assert output_path.exists()

        # Verify file content matches returned content
        with open(output_path) as f:
            file_content = f.read()

        assert file_content == card


class TestExportForServing:
    """Test model export for serving."""

    def test_export_mlflow_format(self, trained_model, sample_data, temp_dir):
        """Test exporting model in MLflow format."""
        export_path = export_for_serving(
            trained_model,
            path=str(temp_dir / "mlflow_export"),
            format="mlflow",
            sample_data=sample_data
        )

        assert export_path.exists()
        assert (export_path / "MLmodel").exists()
        assert (export_path / "SERVING.md").exists()

    def test_export_python_format(self, trained_model, temp_dir):
        """Test exporting model in Python format."""
        export_path = export_for_serving(
            trained_model,
            path=str(temp_dir / "python_export"),
            format="python"
        )

        assert export_path.exists()
        assert (export_path / "model").exists()
        assert (export_path / "predictor.py").exists()

    def test_export_with_input_example(self, trained_model, sample_data, temp_dir):
        """Test exporting with input example."""
        export_path = export_for_serving(
            trained_model,
            path=str(temp_dir / "example_export"),
            format="mlflow",
            include_example=True,
            sample_data=sample_data
        )

        assert (export_path / "input_example.json").exists()

    def test_export_without_example(self, trained_model, temp_dir):
        """Test exporting without input example."""
        export_path = export_for_serving(
            trained_model,
            path=str(temp_dir / "no_example_export"),
            format="mlflow",
            include_example=False
        )

        assert not (export_path / "input_example.json").exists()

    def test_export_invalid_format(self, trained_model, temp_dir):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_for_serving(
                trained_model,
                path=str(temp_dir / "invalid_export"),
                format="invalid_format"
            )
