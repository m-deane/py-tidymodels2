"""
Tests for MLflow grouped/nested model save/load functionality.

Tests save/load for NestedModelFit and NestedWorkflowFit objects.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from py_parsnip import linear_reg, rand_forest
from py_workflows import workflow
from py_mlflow import save_model, load_model, get_model_info


@pytest.fixture
def grouped_data():
    """Create sample data with groups."""
    np.random.seed(42)
    n_per_group = 50

    groups = []
    for group in ['A', 'B', 'C']:
        group_data = pd.DataFrame({
            'group': group,
            'x1': np.random.randn(n_per_group),
            'x2': np.random.randn(n_per_group),
            'y': np.random.randn(n_per_group)
        })
        groups.append(group_data)

    return pd.concat(groups, ignore_index=True)


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestNestedModelFit:
    """Test NestedModelFit save/load."""

    def test_save_load_nested_modelfit(self, grouped_data, temp_model_dir):
        """Test save/load for NestedModelFit."""
        # Fit nested model
        spec = linear_reg()
        nested_fit = spec.fit_nested(
            grouped_data,
            formula="y ~ x1 + x2",
            group_col="group"
        )

        # Get predictions before saving
        preds_before = nested_fit.predict(grouped_data)

        # Save model
        model_path = temp_model_dir / "nested_model"
        save_model(nested_fit, str(model_path))

        # Load model
        loaded = load_model(str(model_path))

        # Get predictions after loading
        preds_after = loaded.predict(grouped_data)

        # Predictions should match
        pd.testing.assert_frame_equal(
            preds_before.sort_values('group').reset_index(drop=True),
            preds_after.sort_values('group').reset_index(drop=True)
        )

    def test_nested_modelfit_save_mlflow_method(self, grouped_data, temp_model_dir):
        """Test NestedModelFit.save_mlflow() method."""
        spec = linear_reg(penalty=0.1)
        nested_fit = spec.fit_nested(
            grouped_data,
            formula="y ~ x1 + x2",
            group_col="group"
        )

        model_path = temp_model_dir / "nested_method"
        nested_fit.save_mlflow(str(model_path))

        # Load and verify
        loaded = load_model(str(model_path))
        preds_before = nested_fit.predict(grouped_data)
        preds_after = loaded.predict(grouped_data)

        pd.testing.assert_frame_equal(
            preds_before.sort_values('group').reset_index(drop=True),
            preds_after.sort_values('group').reset_index(drop=True)
        )

    def test_nested_extract_outputs(self, grouped_data, temp_model_dir):
        """Test extract_outputs() after loading nested model."""
        spec = linear_reg()
        nested_fit = spec.fit_nested(
            grouped_data.iloc[:120],  # Train on part of data
            formula="y ~ x1 + x2",
            group_col="group"
        )

        # Evaluate on test data
        nested_fit = nested_fit.evaluate(grouped_data.iloc[120:])

        # Get outputs before saving
        outputs_before, coeffs_before, stats_before = nested_fit.extract_outputs()

        # Save and load
        model_path = temp_model_dir / "nested_outputs"
        save_model(nested_fit, str(model_path))
        loaded = load_model(str(model_path))

        # Get outputs after loading
        outputs_after, coeffs_after, stats_after = loaded.extract_outputs()

        # Should match (sort by group for consistent comparison)
        pd.testing.assert_frame_equal(
            outputs_before.sort_values(['group', 'split']).reset_index(drop=True),
            outputs_after.sort_values(['group', 'split']).reset_index(drop=True)
        )


class TestNestedWorkflowFit:
    """Test NestedWorkflowFit save/load."""

    def test_save_load_nested_workflow(self, grouped_data, temp_model_dir):
        """Test save/load for NestedWorkflowFit."""
        # Create and fit nested workflow
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        nested_wf_fit = wf.fit_nested(grouped_data, group_col="group")

        # Get predictions before saving
        preds_before = nested_wf_fit.predict(grouped_data)

        # Save workflow
        model_path = temp_model_dir / "nested_wf"
        save_model(nested_wf_fit, str(model_path))

        # Load workflow
        loaded = load_model(str(model_path))

        # Predictions should match
        preds_after = loaded.predict(grouped_data)
        pd.testing.assert_frame_equal(
            preds_before.sort_values('group').reset_index(drop=True),
            preds_after.sort_values('group').reset_index(drop=True)
        )

    def test_nested_workflow_save_mlflow_method(self, grouped_data, temp_model_dir):
        """Test NestedWorkflowFit.save_mlflow() method."""
        wf = workflow().add_formula("y ~ x1").add_model(linear_reg())
        nested_wf_fit = wf.fit_nested(grouped_data, group_col="group")

        model_path = temp_model_dir / "nested_wf_method"
        nested_wf_fit.save_mlflow(str(model_path))

        loaded = load_model(str(model_path))
        preds_before = nested_wf_fit.predict(grouped_data)
        preds_after = loaded.predict(grouped_data)

        pd.testing.assert_frame_equal(
            preds_before.sort_values('group').reset_index(drop=True),
            preds_after.sort_values('group').reset_index(drop=True)
        )

    def test_nested_workflow_extract_outputs(self, grouped_data, temp_model_dir):
        """Test extract_outputs() for nested workflow."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(rand_forest(trees=10).set_mode('regression'))
        nested_wf_fit = wf.fit_nested(
            grouped_data.iloc[:120],
            group_col="group"
        )

        # Evaluate
        nested_wf_fit = nested_wf_fit.evaluate(grouped_data.iloc[120:])

        # Get outputs before saving
        outputs_before, coeffs_before, stats_before = nested_wf_fit.extract_outputs()

        # Save and load
        model_path = temp_model_dir / "nested_wf_outputs"
        save_model(nested_wf_fit, str(model_path))
        loaded = load_model(str(model_path))

        # Get outputs after loading
        outputs_after, coeffs_after, stats_after = loaded.extract_outputs()

        # Should match
        pd.testing.assert_frame_equal(
            outputs_before.sort_values(['group', 'split']).reset_index(drop=True),
            outputs_after.sort_values(['group', 'split']).reset_index(drop=True)
        )


class TestGroupedModelInfo:
    """Test model info for grouped models."""

    def test_nested_model_info(self, grouped_data, temp_model_dir):
        """Test get_model_info() for nested models."""
        spec = linear_reg()
        nested_fit = spec.fit_nested(
            grouped_data,
            formula="y ~ x1 + x2",
            group_col="group"
        )

        model_path = temp_model_dir / "nested_info"
        save_model(nested_fit, str(model_path))

        info = get_model_info(str(model_path))
        assert info["is_grouped"] is True
        assert info["group_col"] == "group"
        assert set(info["groups"]) == {'A', 'B', 'C'}
        assert info["model_type"] == "linear_reg"

    def test_nested_workflow_info(self, grouped_data, temp_model_dir):
        """Test get_model_info() for nested workflows."""
        wf = workflow().add_formula("y ~ x1").add_model(linear_reg())
        nested_wf_fit = wf.fit_nested(grouped_data, group_col="group")

        model_path = temp_model_dir / "nested_wf_info"
        save_model(nested_wf_fit, str(model_path))

        info = get_model_info(str(model_path))
        assert info["is_workflow"] is True
        assert info["is_grouped"] is True
        assert info["group_col"] == "group"
        assert set(info["groups"]) == {'A', 'B', 'C'}


class TestGroupedModelPrediction:
    """Test that predictions work correctly after loading grouped models."""

    def test_predict_subset_of_groups(self, grouped_data, temp_model_dir):
        """Test predicting on subset of groups."""
        spec = linear_reg()
        nested_fit = spec.fit_nested(
            grouped_data,
            formula="y ~ x1 + x2",
            group_col="group"
        )

        # Save and load
        model_path = temp_model_dir / "nested_subset"
        save_model(nested_fit, str(model_path))
        loaded = load_model(str(model_path))

        # Predict on subset (only groups A and B)
        subset_data = grouped_data[grouped_data['group'].isin(['A', 'B'])]
        preds = loaded.predict(subset_data)

        # Should have predictions for both groups
        assert set(preds['group'].unique()) == {'A', 'B'}
        assert len(preds) == len(subset_data)

    def test_predict_single_group(self, grouped_data, temp_model_dir):
        """Test predicting on single group."""
        spec = linear_reg()
        nested_fit = spec.fit_nested(
            grouped_data,
            formula="y ~ x1 + x2",
            group_col="group"
        )

        model_path = temp_model_dir / "nested_single"
        save_model(nested_fit, str(model_path))
        loaded = load_model(str(model_path))

        # Predict on single group
        single_group_data = grouped_data[grouped_data['group'] == 'A']
        preds = loaded.predict(single_group_data)

        assert set(preds['group'].unique()) == {'A'}
        assert len(preds) == len(single_group_data)
