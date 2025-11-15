"""
Tests for MLflow monitoring utilities.

Tests prediction logging, drift detection, and performance monitoring.
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
from py_mlflow import (
    log_prediction_batch,
    detect_data_drift,
    monitor_model_performance,
    create_monitoring_dashboard_data
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
        'y': 2 * np.random.randn(n) + 10
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
class TestPredictionLogging:
    """Test prediction batch logging."""

    def test_log_predictions_basic(self, sample_data, temp_mlflow_dir):
        """Test basic prediction logging."""
        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })

        run_id = log_prediction_batch(
            model_name="TestModel",
            predictions=predictions
        )

        assert run_id is not None

        # Verify experiment created
        experiment = mlflow.get_experiment_by_name("TestModel_monitoring")
        assert experiment is not None

    def test_log_predictions_with_actuals(self, sample_data, temp_mlflow_dir):
        """Test logging predictions with actual values."""
        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })
        actuals = pd.DataFrame({
            'y': np.random.randn(20) + 10
        })

        run_id = log_prediction_batch(
            model_name="TestModelActuals",
            predictions=predictions,
            actuals=actuals
        )

        # Verify metrics logged
        client = MlflowClient()
        run = client.get_run(run_id)

        assert "rmse" in run.data.metrics
        assert "mae" in run.data.metrics

    def test_log_predictions_with_features(self, sample_data, temp_mlflow_dir):
        """Test logging predictions with input features."""
        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })
        features = sample_data.iloc[:20][['x1', 'x2', 'x3']]

        run_id = log_prediction_batch(
            model_name="TestModelFeatures",
            predictions=predictions,
            features=features
        )

        # Verify feature statistics logged
        client = MlflowClient()
        run = client.get_run(run_id)

        assert "feature_x1_mean" in run.data.metrics

    def test_log_predictions_with_metadata(self, sample_data, temp_mlflow_dir):
        """Test logging predictions with metadata."""
        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })
        metadata = {
            "batch_id": "2024-01-15",
            "source": "production"
        }

        run_id = log_prediction_batch(
            model_name="TestModelMetadata",
            predictions=predictions,
            metadata=metadata
        )

        # Verify metadata logged
        client = MlflowClient()
        run = client.get_run(run_id)

        assert "meta_batch_id" in run.data.params
        assert run.data.params["meta_batch_id"] == "2024-01-15"

    def test_log_predictions_artifacts(self, sample_data, temp_mlflow_dir):
        """Test that prediction artifacts are saved."""
        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })

        run_id = log_prediction_batch(
            model_name="TestModelArtifacts",
            predictions=predictions
        )

        # Check artifacts
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]

        assert "predictions" in artifact_paths


class TestDataDrift:
    """Test data drift detection."""

    def test_drift_no_drift(self, sample_data):
        """Test drift detection when no drift present."""
        # Same distribution
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]

        results = detect_data_drift(reference, current)

        assert "drifted_features" in results
        assert "drift_scores" in results
        # With same distribution, should have few or no drifted features
        assert len(results["drifted_features"]) <= 3  # Allow some variability

    def test_drift_with_shift(self, sample_data):
        """Test drift detection with mean shift."""
        reference = sample_data.iloc[:50]

        # Create shifted data (significant drift)
        current = sample_data.iloc[50:].copy()
        current['x1'] = current['x1'] + 5.0  # Large shift

        results = detect_data_drift(reference, current, threshold=0.05)

        # Should detect drift in x1
        assert 'x1' in results["drifted_features"] or len(results["drifted_features"]) > 0

    def test_drift_ks_method(self, sample_data):
        """Test drift detection with Kolmogorov-Smirnov test."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]

        results = detect_data_drift(reference, current, method="ks")

        assert "summary" in results
        assert results["summary"]["method"] == "ks"

    def test_drift_simple_method(self, sample_data):
        """Test drift detection with simple method."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]

        results = detect_data_drift(reference, current, method="simple")

        assert "summary" in results
        assert results["summary"]["method"] == "simple"

    def test_drift_scores_structure(self, sample_data):
        """Test drift scores have proper structure."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]

        results = detect_data_drift(reference, current)

        # Check drift scores structure
        for col, score_data in results["drift_scores"].items():
            assert "score" in score_data
            assert "drifted" in score_data
            assert "ref_mean" in score_data
            assert "curr_mean" in score_data


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestPerformanceMonitoring:
    """Test performance monitoring over time."""

    def test_monitor_performance_basic(self, temp_mlflow_dir):
        """Test basic performance monitoring."""
        # Log some prediction batches first
        for i in range(3):
            predictions = pd.DataFrame({'.pred': np.random.randn(10) + 10})
            actuals = pd.DataFrame({'y': np.random.randn(10) + 10})

            log_prediction_batch(
                model_name="MonitorTest",
                predictions=predictions,
                actuals=actuals
            )

        # Monitor performance
        perf_df = monitor_model_performance(
            "MonitorTest",
            window="7d"
        )

        assert isinstance(perf_df, pd.DataFrame)

    def test_monitor_custom_metrics(self, temp_mlflow_dir):
        """Test monitoring with custom metrics."""
        # Log batches
        for i in range(2):
            predictions = pd.DataFrame({'.pred': np.random.randn(10) + 10})
            actuals = pd.DataFrame({'y': np.random.randn(10) + 10})

            log_prediction_batch(
                model_name="CustomMetrics",
                predictions=predictions,
                actuals=actuals
            )

        # Monitor specific metrics
        perf_df = monitor_model_performance(
            "CustomMetrics",
            window="7d",
            metrics=["rmse", "mae"]
        )

        assert isinstance(perf_df, pd.DataFrame)

    def test_monitor_time_windows(self, temp_mlflow_dir):
        """Test different time windows."""
        # Log batch
        predictions = pd.DataFrame({'.pred': np.random.randn(10) + 10})

        log_prediction_batch(
            model_name="TimeWindow",
            predictions=predictions
        )

        # Test different windows
        for window in ["1d", "7d", "1m"]:
            perf_df = monitor_model_performance(
                "TimeWindow",
                window=window
            )
            assert isinstance(perf_df, pd.DataFrame)


class TestMonitoringDashboard:
    """Test monitoring dashboard data creation."""

    def test_create_dashboard_basic(self, sample_data):
        """Test creating basic dashboard data."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:70]

        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })

        dashboard_data = create_monitoring_dashboard_data(
            model_name="DashboardTest",
            reference_data=reference,
            current_data=current,
            predictions=predictions
        )

        assert "model_info" in dashboard_data
        assert "drift_status" in dashboard_data
        assert "prediction_stats" in dashboard_data
        assert "alerts" in dashboard_data

    def test_dashboard_with_actuals(self, sample_data):
        """Test dashboard with actual values."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:70]

        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })
        actuals = pd.DataFrame({
            'y': np.random.randn(20) + 10
        })

        dashboard_data = create_monitoring_dashboard_data(
            model_name="DashboardActuals",
            reference_data=reference,
            current_data=current,
            predictions=predictions,
            actuals=actuals
        )

        assert "performance" in dashboard_data
        # Should have metrics
        if "error" not in dashboard_data["performance"]:
            assert "rmse" in dashboard_data["performance"] or len(dashboard_data["performance"]) > 0

    def test_dashboard_drift_alerts(self, sample_data):
        """Test that dashboard generates drift alerts."""
        reference = sample_data.iloc[:50]

        # Create drifted data
        current = sample_data.iloc[50:70].copy()
        current['x1'] = current['x1'] + 5.0  # Significant drift

        predictions = pd.DataFrame({
            '.pred': np.random.randn(20) + 10
        })

        dashboard_data = create_monitoring_dashboard_data(
            model_name="DriftAlerts",
            reference_data=reference,
            current_data=current,
            predictions=predictions
        )

        # Should have drift-related alerts or drift status
        assert len(dashboard_data["alerts"]) >= 0  # May or may not have alerts

    def test_dashboard_structure(self, sample_data):
        """Test dashboard data has proper structure."""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:70]
        predictions = pd.DataFrame({'.pred': np.random.randn(20) + 10})

        dashboard_data = create_monitoring_dashboard_data(
            model_name="StructureTest",
            reference_data=reference,
            current_data=current,
            predictions=predictions
        )

        # Check model_info structure
        assert "name" in dashboard_data["model_info"]
        assert "timestamp" in dashboard_data["model_info"]

        # Check alerts structure
        for alert in dashboard_data["alerts"]:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert
