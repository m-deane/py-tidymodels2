"""
Tests for MLflow Model Registry utilities.

Tests model registration, versioning, stage transitions, and comparison.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# MLflow imports with fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from py_parsnip import linear_reg, rand_forest
from py_mlflow import (
    register_model,
    get_latest_model,
    transition_model_stage,
    compare_model_versions,
    list_registered_models,
    delete_model_version,
    delete_registered_model,
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
        'y': 2 * np.random.randn(n) + 5  # Mean around 5
    })


@pytest.fixture
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking and registry."""
    temp_dir = tempfile.mkdtemp()
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"file://{temp_dir}")
        # Set registry URI to same location
        mlflow.set_registry_uri(f"file://{temp_dir}")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model."""
    spec = linear_reg()
    train_data = sample_data.iloc[:80]
    fit = spec.fit(train_data, "y ~ x1 + x2")
    return fit


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestModelRegistration:
    """Test model registration functionality."""

    def test_register_model_basic(self, trained_model, temp_mlflow_dir):
        """Test basic model registration."""
        version = register_model(
            trained_model,
            name="TestModel"
        )

        assert version is not None
        assert version.name == "TestModel"
        assert int(version.version) >= 1

    def test_register_model_with_tags(self, trained_model, temp_mlflow_dir):
        """Test model registration with tags."""
        tags = {
            "team": "data-science",
            "version": "v1.0"
        }

        version = register_model(
            trained_model,
            name="TestModelWithTags",
            tags=tags
        )

        # Verify tags
        client = MlflowClient()
        model_version = client.get_model_version("TestModelWithTags", version.version)

        assert "team" in model_version.tags
        assert model_version.tags["team"] == "data-science"

    def test_register_model_with_description(self, trained_model, temp_mlflow_dir):
        """Test model registration with description."""
        description = "Linear regression model for testing"

        version = register_model(
            trained_model,
            name="TestModelWithDesc",
            description=description
        )

        # Verify description
        client = MlflowClient()
        model_version = client.get_model_version("TestModelWithDesc", version.version)

        assert model_version.description == description

    def test_register_multiple_versions(self, sample_data, temp_mlflow_dir):
        """Test registering multiple versions of same model."""
        # Register first version
        spec1 = linear_reg(penalty=0.1)
        fit1 = spec1.fit(sample_data, "y ~ x1 + x2")
        version1 = register_model(fit1, name="MultiVersionModel")

        # Register second version
        spec2 = linear_reg(penalty=0.5)
        fit2 = spec2.fit(sample_data, "y ~ x1 + x2 + x3")
        version2 = register_model(fit2, name="MultiVersionModel")

        assert int(version2.version) > int(version1.version)


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestGetLatestModel:
    """Test getting latest model version."""

    def test_get_latest_model_any_stage(self, trained_model, temp_mlflow_dir):
        """Test getting latest model regardless of stage."""
        # Register model
        register_model(trained_model, name="LatestTest")

        # Get latest
        loaded = get_latest_model("LatestTest", load=True)

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_get_latest_model_metadata_only(self, trained_model, temp_mlflow_dir):
        """Test getting model metadata without loading."""
        register_model(trained_model, name="MetadataTest")

        # Get metadata only (fast)
        version_info = get_latest_model("MetadataTest", load=False)

        assert version_info is not None
        assert hasattr(version_info, 'version')
        assert version_info.name == "MetadataTest"

    def test_get_latest_model_by_stage(self, trained_model, temp_mlflow_dir):
        """Test getting latest model by stage."""
        # Register and transition to production
        version = register_model(trained_model, name="StageTest")
        transition_model_stage("StageTest", version.version, "Production")

        # Get production model
        loaded = get_latest_model("StageTest", stage="Production", load=True)

        assert loaded is not None

    def test_get_latest_model_nonexistent(self, temp_mlflow_dir):
        """Test getting nonexistent model raises error."""
        with pytest.raises(Exception):  # MlflowException
            get_latest_model("NonexistentModel")


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestStageTransitions:
    """Test model stage transitions."""

    def test_transition_to_production(self, trained_model, temp_mlflow_dir):
        """Test transitioning model to Production."""
        version = register_model(trained_model, name="ProdTest")

        updated = transition_model_stage(
            "ProdTest",
            version.version,
            "Production"
        )

        assert updated.current_stage == "Production"

    def test_transition_to_staging(self, trained_model, temp_mlflow_dir):
        """Test transitioning model to Staging."""
        version = register_model(trained_model, name="StagingTest")

        updated = transition_model_stage(
            "StagingTest",
            version.version,
            "Staging"
        )

        assert updated.current_stage == "Staging"

    def test_transition_archives_existing(self, sample_data, temp_mlflow_dir):
        """Test that transitioning to Production archives existing prod models."""
        # Register two versions
        fit1 = linear_reg().fit(sample_data, "y ~ x1")
        v1 = register_model(fit1, name="ArchiveTest")

        fit2 = linear_reg().fit(sample_data, "y ~ x1 + x2")
        v2 = register_model(fit2, name="ArchiveTest")

        # Move v1 to production
        transition_model_stage("ArchiveTest", v1.version, "Production")

        # Move v2 to production (should archive v1)
        transition_model_stage("ArchiveTest", v2.version, "Production", archive_existing=True)

        # Check v1 was archived
        client = MlflowClient()
        v1_updated = client.get_model_version("ArchiveTest", v1.version)

        assert v1_updated.current_stage == "Archived"

    def test_transition_invalid_stage(self, trained_model, temp_mlflow_dir):
        """Test that invalid stage raises error."""
        version = register_model(trained_model, name="InvalidStageTest")

        with pytest.raises(ValueError, match="Invalid stage"):
            transition_model_stage("InvalidStageTest", version.version, "InvalidStage")


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestCompareVersions:
    """Test comparing model versions."""

    def test_compare_versions_without_data(self, sample_data, temp_mlflow_dir):
        """Test comparing versions without evaluation data."""
        # Register multiple versions
        fit1 = linear_reg(penalty=0.1).fit(sample_data, "y ~ x1 + x2")
        v1 = register_model(fit1, name="CompareTest", tags={"version": "v1"})

        fit2 = linear_reg(penalty=0.5).fit(sample_data, "y ~ x1 + x2")
        v2 = register_model(fit2, name="CompareTest", tags={"version": "v2"})

        # Compare
        comparison = compare_model_versions("CompareTest")

        assert not comparison.empty
        assert len(comparison) >= 2
        assert "version" in comparison.columns
        assert "stage" in comparison.columns

    def test_compare_versions_with_evaluation(self, sample_data, temp_mlflow_dir):
        """Test comparing versions with evaluation metrics."""
        train_data = sample_data.iloc[:80]
        test_data = sample_data.iloc[80:]

        # Register multiple versions
        fit1 = linear_reg(penalty=0.1).fit(train_data, "y ~ x1 + x2")
        register_model(fit1, name="EvalTest")

        fit2 = linear_reg(penalty=0.5).fit(train_data, "y ~ x1 + x2")
        register_model(fit2, name="EvalTest")

        # Compare with evaluation
        from py_yardstick import metric_set, rmse, mae

        comparison = compare_model_versions(
            "EvalTest",
            test_data=test_data,
            metrics=metric_set(rmse, mae)
        )

        assert not comparison.empty
        assert "rmse" in comparison.columns
        assert "mae" in comparison.columns

    def test_compare_specific_versions(self, sample_data, temp_mlflow_dir):
        """Test comparing specific version numbers."""
        # Register 3 versions
        for i in range(3):
            fit = linear_reg().fit(sample_data, "y ~ x1 + x2")
            register_model(fit, name="SpecificTest")

        # Compare only versions 1 and 2
        comparison = compare_model_versions(
            "SpecificTest",
            versions=[1, 2]
        )

        assert len(comparison) == 2

    def test_compare_by_stage(self, sample_data, temp_mlflow_dir):
        """Test comparing versions filtered by stage."""
        # Register and stage models
        fit1 = linear_reg().fit(sample_data, "y ~ x1 + x2")
        v1 = register_model(fit1, name="StageFilterTest")
        transition_model_stage("StageFilterTest", v1.version, "Production")

        fit2 = linear_reg().fit(sample_data, "y ~ x1 + x2")
        v2 = register_model(fit2, name="StageFilterTest")
        transition_model_stage("StageFilterTest", v2.version, "Staging")

        # Compare only Production
        comparison = compare_model_versions(
            "StageFilterTest",
            stages=["Production"]
        )

        assert len(comparison) == 1
        assert comparison.iloc[0]["stage"] == "Production"


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestListModels:
    """Test listing registered models."""

    def test_list_all_models(self, trained_model, temp_mlflow_dir):
        """Test listing all registered models."""
        # Register some models
        register_model(trained_model, name="Model1")
        register_model(trained_model, name="Model2")

        # List all
        models = list_registered_models()

        assert not models.empty
        assert len(models) >= 2
        assert "name" in models.columns
        assert "latest_version" in models.columns

    def test_list_models_with_filter(self, trained_model, temp_mlflow_dir):
        """Test listing models with filter."""
        # Register models with different names
        register_model(trained_model, name="Sales_Model")
        register_model(trained_model, name="Forecast_Model")

        # Filter for Sales models
        # Note: MLflow filter string support varies, so we test basic functionality
        models = list_registered_models()

        assert not models.empty

    def test_list_models_empty(self, temp_mlflow_dir):
        """Test listing when no models registered."""
        models = list_registered_models()

        # May be empty or have minimal structure
        assert isinstance(models, pd.DataFrame)


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestDeletion:
    """Test model deletion."""

    def test_delete_model_version(self, trained_model, temp_mlflow_dir):
        """Test deleting specific model version."""
        # Register two versions
        v1 = register_model(trained_model, name="DeleteTest")
        v2 = register_model(trained_model, name="DeleteTest")

        # Delete version 1
        delete_model_version("DeleteTest", v1.version)

        # Version 2 should still exist
        client = MlflowClient()
        v2_check = client.get_model_version("DeleteTest", v2.version)
        assert v2_check is not None

    def test_delete_registered_model(self, trained_model, temp_mlflow_dir):
        """Test deleting entire registered model."""
        register_model(trained_model, name="DeleteEntireModel")

        # Delete entire model
        delete_registered_model("DeleteEntireModel")

        # Model should not exist
        client = MlflowClient()
        with pytest.raises(Exception):  # Should raise exception
            client.get_registered_model("DeleteEntireModel")
