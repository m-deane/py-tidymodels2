"""
Tests for MLflow flavor save/load functionality.

Tests comprehensive save/load round-trips for all model types.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from py_parsnip import linear_reg, rand_forest, decision_tree
from py_mlflow import save_model, load_model, get_model_info, validate_model_exists


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'y': np.random.randn(n)
    })


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestBasicSaveLoad:
    """Test basic save/load functionality."""

    def test_save_load_linear_reg(self, sample_data, temp_model_dir):
        """Test save/load round-trip for linear regression."""
        # Fit model
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2 + x3")

        # Get predictions before saving
        preds_before = fit.predict(sample_data)

        # Save model
        model_path = temp_model_dir / "linear_model"
        save_model(fit, str(model_path))

        # Verify artifacts exist
        assert (model_path / "MLmodel").exists()
        assert (model_path / "conda.yaml").exists()
        assert (model_path / "model" / "model.pkl").exists()

        # Load model
        loaded = load_model(str(model_path))

        # Get predictions after loading
        preds_after = loaded.predict(sample_data)

        # Predictions should match
        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_save_load_rand_forest(self, sample_data, temp_model_dir):
        """Test save/load for random forest."""
        spec = rand_forest(mode='regression', trees=10)
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        preds_before = fit.predict(sample_data)

        model_path = temp_model_dir / "rf_model"
        save_model(fit, str(model_path))

        loaded = load_model(str(model_path))
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_save_load_decision_tree(self, sample_data, temp_model_dir):
        """Test save/load for decision tree."""
        spec = decision_tree(tree_depth=5).set_mode('regression')
        fit = spec.fit(sample_data, "y ~ x1 + x2 + x3")

        preds_before = fit.predict(sample_data)

        model_path = temp_model_dir / "dt_model"
        save_model(fit, str(model_path))

        loaded = load_model(str(model_path))
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)


class TestSignatureInference:
    """Test model signature inference."""

    def test_signature_auto(self, sample_data, temp_model_dir):
        """Test automatic signature inference."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "sig_model"
        save_model(
            fit,
            str(model_path),
            signature="auto",
            input_example=sample_data.head(5)
        )

        # Verify model saved
        assert (model_path / "MLmodel").exists()

        # Load should work
        loaded = load_model(str(model_path))
        assert loaded is not None

    def test_input_example_saved(self, sample_data, temp_model_dir):
        """Test that input example is saved."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "example_model"
        save_model(
            fit,
            str(model_path),
            input_example=sample_data.head(5)
        )

        # Check input example was saved
        assert (model_path / "input_example.json").exists()


class TestModelFitMethod:
    """Test save_mlflow() method on ModelFit objects."""

    def test_modelfit_save_mlflow(self, sample_data, temp_model_dir):
        """Test ModelFit.save_mlflow() method."""
        spec = linear_reg(penalty=0.1)
        fit = spec.fit(sample_data, "y ~ x1 + x2 + x3")

        model_path = temp_model_dir / "fit_method_model"
        fit.save_mlflow(str(model_path))

        # Load and verify
        loaded = load_model(str(model_path))
        preds_before = fit.predict(sample_data)
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_modelfit_save_with_signature(self, sample_data, temp_model_dir):
        """Test save_mlflow with signature parameter."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "sig_fit_model"
        fit.save_mlflow(
            str(model_path),
            signature="auto",
            input_example=sample_data.head(3)
        )

        loaded = load_model(str(model_path))
        assert loaded is not None


class TestExtractOutputs:
    """Test that extract_outputs() works after loading."""

    def test_extract_outputs_after_load(self, sample_data, temp_model_dir):
        """Test extract_outputs() on loaded model."""
        # Fit and evaluate
        spec = linear_reg()
        fit = spec.fit(sample_data.iloc[:80], "y ~ x1 + x2")
        fit = fit.evaluate(sample_data.iloc[80:])

        # Get outputs before saving
        outputs_before, coeffs_before, stats_before = fit.extract_outputs()

        # Save and load
        model_path = temp_model_dir / "extract_model"
        save_model(fit, str(model_path))
        loaded = load_model(str(model_path))

        # Get outputs after loading
        outputs_after, coeffs_after, stats_after = loaded.extract_outputs()

        # Outputs should match
        pd.testing.assert_frame_equal(outputs_before, outputs_after)
        pd.testing.assert_frame_equal(coeffs_before, coeffs_after)
        pd.testing.assert_frame_equal(stats_before, stats_after)


class TestModelInfo:
    """Test get_model_info() function."""

    def test_get_model_info(self, sample_data, temp_model_dir):
        """Test extracting model metadata."""
        spec = linear_reg(penalty=0.1, mixture=0.5)
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "info_model"
        save_model(
            fit,
            str(model_path),
            metadata={"dataset": "sample", "version": "1.0"}
        )

        # Get info without loading full model
        info = get_model_info(str(model_path))

        assert info["model_type"] == "linear_reg"
        assert info["engine"] == "sklearn"
        assert info["mode"] == "regression"
        assert info["is_workflow"] is False
        assert info["is_grouped"] is False
        assert info["metadata"]["dataset"] == "sample"
        assert info["metadata"]["version"] == "1.0"

    def test_validate_model_exists(self, sample_data, temp_model_dir):
        """Test validate_model_exists() function."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "exist_model"

        # Model doesn't exist yet
        assert not validate_model_exists(str(model_path))

        # Save model
        save_model(fit, str(model_path))

        # Now it exists
        assert validate_model_exists(str(model_path))


class TestErrorHandling:
    """Test error handling."""

    def test_load_nonexistent_model(self, temp_model_dir):
        """Test loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            load_model(str(temp_model_dir / "nonexistent"))

    def test_load_corrupted_mlmodel(self, temp_model_dir):
        """Test loading corrupted MLmodel file."""
        model_path = temp_model_dir / "corrupt_model"
        model_path.mkdir()

        # Create empty MLmodel file
        (model_path / "MLmodel").touch()

        with pytest.raises(Exception):  # Will raise yaml parse error
            load_model(str(model_path))

    def test_save_invalid_model(self, temp_model_dir):
        """Test saving invalid model raises error."""
        # Try to save a non-model object
        invalid_model = {"not": "a model"}

        with pytest.raises(ValueError, match="predict"):
            save_model(invalid_model, str(temp_model_dir / "invalid"))
