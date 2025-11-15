"""
Tests for MLflow workflow save/load functionality.

Tests save/load for WorkflowFit objects with recipes and formulas.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg, rand_forest
from py_mlflow import save_model, load_model, get_model_info


@pytest.fixture
def sample_data():
    """Create sample data with multiple features."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
        'x5': np.random.randn(n),
        'y': np.random.randn(n)
    })


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestWorkflowWithFormula:
    """Test workflow with formula preprocessing."""

    def test_save_load_workflow_formula(self, sample_data, temp_model_dir):
        """Test save/load workflow with formula."""
        # Create and fit workflow
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(linear_reg())
        )
        wf_fit = wf.fit(sample_data)

        # Get predictions before saving
        preds_before = wf_fit.predict(sample_data)

        # Save workflow
        model_path = temp_model_dir / "wf_formula"
        save_model(wf_fit, str(model_path))

        # Load workflow
        loaded = load_model(str(model_path))

        # Get predictions after loading
        preds_after = loaded.predict(sample_data)

        # Predictions should match
        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_workflow_extract_outputs(self, sample_data, temp_model_dir):
        """Test extract_outputs() after loading workflow."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        wf_fit = wf.fit(sample_data.iloc[:100])
        wf_fit = wf_fit.evaluate(sample_data.iloc[100:])

        # Get outputs before saving
        outputs_before, coeffs_before, stats_before = wf_fit.extract_outputs()

        # Save and load
        model_path = temp_model_dir / "wf_outputs"
        save_model(wf_fit, str(model_path))
        loaded = load_model(str(model_path))

        # Get outputs after loading
        outputs_after, coeffs_after, stats_after = loaded.extract_outputs()

        # Should match
        pd.testing.assert_frame_equal(outputs_before, outputs_after)
        pd.testing.assert_frame_equal(coeffs_before, coeffs_after)
        pd.testing.assert_frame_equal(stats_before, stats_after)


class TestWorkflowWithRecipe:
    """Test workflow with recipe preprocessing."""

    def test_save_load_workflow_recipe(self, sample_data, temp_model_dir):
        """Test save/load workflow with recipe."""
        # Create workflow with recipe
        rec = (
            recipe()
            .step_normalize()
            .step_select_corr(threshold=0.9)
        )
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        wf_fit = wf.fit(sample_data)

        # Get predictions before saving
        preds_before = wf_fit.predict(sample_data)

        # Save workflow
        model_path = temp_model_dir / "wf_recipe"
        save_model(wf_fit, str(model_path))

        # Load workflow
        loaded = load_model(str(model_path))

        # Predictions should match
        preds_after = loaded.predict(sample_data)
        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_save_load_recipe_pca(self, sample_data, temp_model_dir):
        """Test workflow with PCA recipe."""
        rec = recipe().step_normalize().step_pca(num_comp=3)
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        wf_fit = wf.fit(sample_data)

        preds_before = wf_fit.predict(sample_data)

        model_path = temp_model_dir / "wf_pca"
        save_model(wf_fit, str(model_path))

        loaded = load_model(str(model_path))
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_workflow_recipe_multiple_steps(self, sample_data, temp_model_dir):
        """Test workflow with multiple recipe steps."""
        rec = (
            recipe()
            .step_normalize()
            .step_pca(num_comp=3)
            .step_select_corr(threshold=0.95)
        )
        wf = workflow().add_recipe(rec).add_model(rand_forest(mode='regression', trees=10))
        wf_fit = wf.fit(sample_data)

        preds_before = wf_fit.predict(sample_data)

        model_path = temp_model_dir / "wf_multi_step"
        wf_fit.save_mlflow(str(model_path))

        loaded = load_model(str(model_path))
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)


class TestWorkflowFitMethod:
    """Test save_mlflow() method on WorkflowFit."""

    def test_workflowfit_save_mlflow(self, sample_data, temp_model_dir):
        """Test WorkflowFit.save_mlflow() method."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        wf_fit = wf.fit(sample_data)

        model_path = temp_model_dir / "wf_method"
        wf_fit.save_mlflow(str(model_path))

        loaded = load_model(str(model_path))
        preds_before = wf_fit.predict(sample_data)
        preds_after = loaded.predict(sample_data)

        pd.testing.assert_frame_equal(preds_before, preds_after)

    def test_workflowfit_save_with_metadata(self, sample_data, temp_model_dir):
        """Test save with custom metadata."""
        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        wf_fit = wf.fit(sample_data)

        model_path = temp_model_dir / "wf_metadata"
        wf_fit.save_mlflow(
            str(model_path),
            metadata={"team": "data_science", "experiment": "test_001"}
        )

        # Verify metadata
        info = get_model_info(str(model_path))
        assert info["metadata"]["team"] == "data_science"
        assert info["metadata"]["experiment"] == "test_001"


class TestWorkflowInfo:
    """Test model info for workflows."""

    def test_workflow_is_workflow_flag(self, sample_data, temp_model_dir):
        """Test that is_workflow flag is set correctly."""
        wf = workflow().add_formula("y ~ x1").add_model(linear_reg())
        wf_fit = wf.fit(sample_data)

        model_path = temp_model_dir / "wf_flag"
        save_model(wf_fit, str(model_path))

        info = get_model_info(str(model_path))
        assert info["is_workflow"] is True
        assert info["model_type"] == "linear_reg"

    def test_workflow_vs_modelfit_info(self, sample_data, temp_model_dir):
        """Test that workflow and modelfit have different is_workflow flag."""
        # Save ModelFit
        spec = linear_reg()
        fit = spec.fit(sample_data, "y ~ x1")
        model_path_fit = temp_model_dir / "modelfit"
        save_model(fit, str(model_path_fit))

        # Save WorkflowFit
        wf = workflow().add_formula("y ~ x1").add_model(linear_reg())
        wf_fit = wf.fit(sample_data)
        model_path_wf = temp_model_dir / "workflowfit"
        save_model(wf_fit, str(model_path_wf))

        # Check flags
        info_fit = get_model_info(str(model_path_fit))
        info_wf = get_model_info(str(model_path_wf))

        assert info_fit["is_workflow"] is False
        assert info_wf["is_workflow"] is True
