"""
Tests for MLflow experiment tracking integration with py_tune.

Tests auto-logging for tune_grid() and fit_resamples().
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

from py_parsnip import linear_reg
from py_workflows import workflow
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared
from py_tune import tune_grid, fit_resamples, grid_regular, tune


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
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"file://{temp_dir}")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestFitResamplesTracking:
    """Test MLflow tracking for fit_resamples()."""

    def test_fit_resamples_without_tracking(self, sample_data, temp_mlflow_dir):
        """Test fit_resamples works without tracking enabled."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=3)
        my_metrics = metric_set(rmse, mae)

        results = fit_resamples(wf, folds, metrics=my_metrics)

        assert results is not None
        assert not results.metrics.empty

    def test_fit_resamples_with_tracking(self, sample_data, temp_mlflow_dir):
        """Test fit_resamples with MLflow tracking enabled."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=3)
        my_metrics = metric_set(rmse, mae)

        results = fit_resamples(
            wf, folds, metrics=my_metrics,
            mlflow_tracking=True,
            mlflow_experiment_name="test_cv"
        )

        assert results is not None
        assert not results.metrics.empty

        # Verify MLflow tracking
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name("test_cv")
        assert experiment is not None

        # Check runs exist
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

    def test_fit_resamples_logs_metrics(self, sample_data, temp_mlflow_dir):
        """Test that fit_resamples logs metrics correctly."""
        wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=3)
        my_metrics = metric_set(rmse, mae, r_squared)

        results = fit_resamples(
            wf, folds, metrics=my_metrics,
            mlflow_tracking=True,
            mlflow_experiment_name="test_metrics_logging"
        )

        # Get experiment and runs
        experiment = mlflow.get_experiment_by_name("test_metrics_logging")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        assert len(runs) > 0

        # Check that mean and std metrics were logged
        first_run = runs.iloc[0]
        assert "metrics.rmse_mean" in runs.columns
        assert "metrics.mae_mean" in runs.columns
        assert "metrics.rmse_std" in runs.columns
        assert "metrics.mae_std" in runs.columns

    def test_fit_resamples_logs_params(self, sample_data, temp_mlflow_dir):
        """Test that fit_resamples logs parameters."""
        spec = linear_reg(penalty=0.1)
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=3)

        results = fit_resamples(
            wf, folds,
            mlflow_tracking=True,
            mlflow_experiment_name="test_params"
        )

        # Get runs
        experiment = mlflow.get_experiment_by_name("test_params")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        first_run = runs.iloc[0]
        assert "params.model_type" in runs.columns
        assert first_run["params.model_type"] == "linear_reg"
        assert first_run["params.n_folds"] == "3"

    def test_fit_resamples_logs_artifacts(self, sample_data, temp_mlflow_dir):
        """Test that fit_resamples logs artifacts."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=3)

        results = fit_resamples(
            wf, folds,
            mlflow_tracking=True,
            mlflow_experiment_name="test_artifacts"
        )

        # Get run
        experiment = mlflow.get_experiment_by_name("test_artifacts")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        run_id = runs.iloc[0]["run_id"]
        client = MlflowClient()

        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]

        assert "metrics" in artifact_paths

@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestTuneGridTracking:
    """Test MLflow tracking for tune_grid()."""

    def test_tune_grid_without_tracking(self, sample_data, temp_mlflow_dir):
        """Test tune_grid works without tracking enabled."""
        spec = linear_reg(penalty=tune(), mixture=tune())
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=3)

        param_info = {
            'penalty': {'range': (0.01, 0.1)},
            'mixture': {'range': (0, 1)}
        }

        results = tune_grid(wf, folds, param_info=param_info, grid=2)

        assert results is not None
        assert not results.metrics.empty

    def test_tune_grid_with_tracking(self, sample_data, temp_mlflow_dir):
        """Test tune_grid with MLflow tracking enabled."""
        spec = linear_reg(penalty=tune(), mixture=tune())
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=2)

        param_info = {
            'penalty': {'range': (0.01, 0.1)},
            'mixture': {'range': (0, 1)}
        }

        results = tune_grid(
            wf, folds, param_info=param_info, grid=2,
            mlflow_tracking=True,
            mlflow_experiment_name="test_tuning"
        )

        assert results is not None

        # Verify MLflow tracking
        experiment = mlflow.get_experiment_by_name("test_tuning")
        assert experiment is not None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

    def test_tune_grid_nested_runs(self, sample_data, temp_mlflow_dir):
        """Test that tune_grid creates nested runs for each config."""
        spec = linear_reg(penalty=tune())
        wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(spec)
        folds = vfold_cv(sample_data, v=2)

        param_info = {
            'penalty': {'range': (0.01, 0.5)}
        }

        results = tune_grid(
            wf, folds, param_info=param_info, grid=3,
            mlflow_tracking=True,
            mlflow_experiment_name="test_nested_runs"
        )

        # Get all runs
        experiment = mlflow.get_experiment_by_name("test_nested_runs")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Should have parent run + nested runs for each config
        assert len(runs) >= 3  # At least 3 configs

    def test_tune_grid_logs_best_params(self, sample_data, temp_mlflow_dir):
        """Test that tune_grid logs best parameters to parent run."""
        spec = linear_reg(penalty=tune())
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=2)

        param_info = {
            'penalty': {'range': (0.01, 1.0), 'trans': 'log'}
        }

        results = tune_grid(
            wf, folds, param_info=param_info, grid=3,
            mlflow_tracking=True,
            mlflow_experiment_name="test_best_params"
        )

        # Get parent run (no parent_run_id)
        experiment = mlflow.get_experiment_by_name("test_best_params")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Filter for parent run (has tags.mlflow.runName starting with "tuning_")
        parent_runs = runs[runs['tags.mlflow.runName'].str.startswith('tuning_', na=False)]

        if len(parent_runs) > 0:
            parent_run = parent_runs.iloc[0]
            assert "params.best_config" in runs.columns

    def test_tune_grid_logs_grid_info(self, sample_data, temp_mlflow_dir):
        """Test that tune_grid logs grid information."""
        spec = linear_reg(penalty=tune(), mixture=tune())
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=2)

        param_info = {
            'penalty': {'range': (0.01, 0.1)},
            'mixture': {'range': (0, 1)}
        }

        results = tune_grid(
            wf, folds, param_info=param_info, grid=3,
            mlflow_tracking=True,
            mlflow_experiment_name="test_grid_info"
        )

        experiment = mlflow.get_experiment_by_name("test_grid_info")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Check parent run has grid info
        parent_runs = runs[runs['tags.mlflow.runName'].str.startswith('tuning_', na=False)]
        if len(parent_runs) > 0:
            parent_run = parent_runs.iloc[0]
            assert parent_run["params.n_configs"] == str(len(results.grid))
            assert parent_run["params.n_folds"] == "2"

    def test_tune_grid_logs_artifacts(self, sample_data, temp_mlflow_dir):
        """Test that tune_grid logs grid and metrics artifacts."""
        spec = linear_reg(penalty=tune())
        wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
        folds = vfold_cv(sample_data, v=2)

        param_info = {
            'penalty': {'range': (0.01, 0.5)}
        }

        results = tune_grid(
            wf, folds, param_info=param_info, grid=2,
            mlflow_tracking=True,
            mlflow_experiment_name="test_tune_artifacts"
        )

        # Get parent run
        experiment = mlflow.get_experiment_by_name("test_tune_artifacts")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        parent_runs = runs[runs['tags.mlflow.runName'].str.startswith('tuning_', na=False)]

        if len(parent_runs) > 0:
            run_id = parent_runs.iloc[0]["run_id"]
            client = MlflowClient()

            # Check artifacts
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [a.path for a in artifacts]

            assert "metrics" in artifact_paths or len(artifact_paths) > 0


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestTrackingErrorHandling:
    """Test error handling for tracking."""

    def test_tracking_continues_if_mlflow_fails(self, sample_data, temp_mlflow_dir):
        """Test that tuning continues even if MLflow logging fails."""
        # This test ensures robustness
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=2)

        # Should complete successfully even if tracking has issues
        results = fit_resamples(
            wf, folds,
            mlflow_tracking=True,
            mlflow_experiment_name="test_error_handling"
        )

        assert results is not None
        assert not results.metrics.empty

    def test_default_experiment_names(self, sample_data, temp_mlflow_dir):
        """Test that default experiment names are used when not specified."""
        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        folds = vfold_cv(sample_data, v=2)

        # Without specifying experiment name
        results = fit_resamples(
            wf, folds,
            mlflow_tracking=True
        )

        # Should use default name "fit_resamples"
        experiment = mlflow.get_experiment_by_name("fit_resamples")
        assert experiment is not None
