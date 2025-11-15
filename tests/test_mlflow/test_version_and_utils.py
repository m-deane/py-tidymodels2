"""
Tests for version compatibility checking and utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import warnings

from py_parsnip import linear_reg
from py_mlflow import save_model, load_model, validate_model_exists
from py_mlflow.utils import (
    check_version_compatibility,
    get_version_metadata,
    infer_model_signature,
    get_input_example,
    should_compress,
    get_artifact_size_mb
)


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randn(100)
    })


@pytest.fixture
def temp_model_dir():
    """Create temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestVersionCompatibility:
    """Test version compatibility checking."""

    def test_get_version_metadata(self):
        """Test get_version_metadata() function."""
        meta = get_version_metadata("0.1.0")

        assert "py_tidymodels_version" in meta
        assert "min_py_tidymodels_version" in meta
        assert "max_py_tidymodels_version" in meta
        assert meta["py_tidymodels_version"] == "0.1.0"
        assert meta["min_py_tidymodels_version"] == "0.1.0"

    def test_version_compatibility_same_version(self):
        """Test compatibility check with same version."""
        flavor_conf = {
            "py_tidymodels_version": "0.1.0",
            "min_py_tidymodels_version": "0.1.0",
            "max_py_tidymodels_version": "0.2.0"
        }

        # Should not raise or warn
        check_version_compatibility(flavor_conf, "0.1.0")

    def test_version_compatibility_minor_version_diff(self):
        """Test compatibility check with minor version difference."""
        flavor_conf = {
            "py_tidymodels_version": "0.1.0",
            "min_py_tidymodels_version": "0.1.0",
            "max_py_tidymodels_version": "0.2.0"
        }

        # Should work (within max version)
        check_version_compatibility(flavor_conf, "0.1.5")

    def test_version_compatibility_major_version_mismatch(self):
        """Test compatibility check with major version mismatch."""
        flavor_conf = {
            "py_tidymodels_version": "0.1.0",
            "min_py_tidymodels_version": "0.1.0",
            "max_py_tidymodels_version": "0.2.0"
        }

        # Should warn about major version mismatch
        with pytest.warns(UserWarning, match="Major version mismatch"):
            check_version_compatibility(flavor_conf, "1.0.0")

    def test_version_below_minimum(self):
        """Test error when current version below minimum."""
        flavor_conf = {
            "py_tidymodels_version": "0.2.0",
            "min_py_tidymodels_version": "0.2.0",
            "max_py_tidymodels_version": "0.3.0"
        }

        # Should raise error
        with pytest.raises(ValueError, match="requires py-tidymodels"):
            check_version_compatibility(flavor_conf, "0.1.0")

    def test_version_above_maximum(self):
        """Test warning when current version above maximum."""
        flavor_conf = {
            "py_tidymodels_version": "0.1.0",
            "min_py_tidymodels_version": "0.1.0",
            "max_py_tidymodels_version": "0.2.0"
        }

        # Should warn
        with pytest.warns(UserWarning, match="Compatibility not guaranteed"):
            check_version_compatibility(flavor_conf, "0.3.0")


class TestSignatureUtils:
    """Test signature-related utilities."""

    def test_infer_model_signature(self, sample_data):
        """Test signature inference."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        input_data = sample_data[['x1', 'x2']]
        output_data = fit.predict(sample_data)

        signature = infer_model_signature(input_data, output_data)

        assert signature is not None
        assert signature.inputs is not None
        assert signature.outputs is not None

    def test_get_input_example(self, sample_data):
        """Test get_input_example() function."""
        example = get_input_example(sample_data, n_rows=5)

        assert len(example) == 5
        pd.testing.assert_frame_equal(example, sample_data.head(5))

    def test_get_input_example_default(self, sample_data):
        """Test get_input_example() with default n_rows."""
        example = get_input_example(sample_data)
        assert len(example) == 5


class TestArtifactUtils:
    """Test artifact handling utilities."""

    def test_should_compress_small_file(self, temp_model_dir, sample_data):
        """Test should_compress() for small files."""
        # Save a small model
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "small_model"
        save_model(fit, str(model_path))

        pkl_path = model_path / "model" / "model.pkl"

        # Small file shouldn't need compression (default threshold 100MB)
        assert not should_compress(pkl_path)

    def test_should_compress_custom_threshold(self, temp_model_dir, sample_data):
        """Test should_compress() with custom threshold."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "threshold_model"
        save_model(fit, str(model_path))

        pkl_path = model_path / "model" / "model.pkl"

        # With very low threshold, should compress
        assert should_compress(pkl_path, threshold_mb=0.001)

    def test_get_artifact_size_mb(self, temp_model_dir, sample_data):
        """Test get_artifact_size_mb() function."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "size_model"
        save_model(fit, str(model_path))

        pkl_path = model_path / "model" / "model.pkl"

        size_mb = get_artifact_size_mb(pkl_path)

        assert size_mb > 0
        assert size_mb < 10  # Should be small

    def test_get_artifact_size_nonexistent(self, temp_model_dir):
        """Test get_artifact_size_mb() for nonexistent file."""
        fake_path = temp_model_dir / "nonexistent.pkl"
        size = get_artifact_size_mb(fake_path)
        assert size == 0.0


class TestValidateModelExists:
    """Test validate_model_exists() function."""

    def test_validate_nonexistent_model(self, temp_model_dir):
        """Test validation of nonexistent model."""
        assert not validate_model_exists(str(temp_model_dir / "nonexistent"))

    def test_validate_existing_model(self, sample_data, temp_model_dir):
        """Test validation of existing model."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "exist_model"
        save_model(fit, str(model_path))

        assert validate_model_exists(str(model_path))

    def test_validate_incomplete_model(self, temp_model_dir):
        """Test validation of incomplete model (missing artifacts)."""
        # Create directory with MLmodel but no model.pkl
        model_path = temp_model_dir / "incomplete"
        model_path.mkdir()
        (model_path / "MLmodel").touch()

        assert not validate_model_exists(str(model_path))


class TestEndToEndVersioning:
    """Test end-to-end versioning workflow."""

    def test_save_and_check_version_info(self, sample_data, temp_model_dir):
        """Test that version info is properly saved and retrieved."""
        from py_mlflow import get_model_info

        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "version_model"
        save_model(fit, str(model_path))

        # Get model info
        info = get_model_info(str(model_path))

        # Version should be set
        assert "py_tidymodels_version" in info
        assert info["py_tidymodels_version"] is not None
        assert len(info["py_tidymodels_version"]) > 0

    def test_load_checks_version(self, sample_data, temp_model_dir):
        """Test that load_model() checks version compatibility."""
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1 + x2")

        model_path = temp_model_dir / "check_version"
        save_model(fit, str(model_path))

        # Loading should work (same version)
        loaded = load_model(str(model_path))
        assert loaded is not None
