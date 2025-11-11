"""
Tests for time series transformation steps (Phase 2).
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.timeseries_transformations import (
    StepCleanAnomalies,
    StepStationary,
    StepDeseasonalize,
    StepDetrend,
    StepHStat,
    StepBestLag,
)


@pytest.fixture
def sample_ts_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Create time series with trend, seasonality, and noise
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
    noise = np.random.normal(0, 2, n)

    data = pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise,
        'x1': np.random.normal(50, 10, n),
        'x2': np.random.normal(100, 20, n),
        'target': np.random.normal(0, 1, n)
    })

    # Make target correlated with features
    data['target'] = 0.3 * data['x1'] + 0.2 * data['x2'] + np.random.normal(0, 5, n)

    # Add some anomalies
    data.loc[10, 'value'] = 200  # Outlier
    data.loc[50, 'value'] = -50  # Outlier

    return data


@pytest.fixture
def sample_stationary_data():
    """Create sample data for stationarity testing."""
    np.random.seed(42)
    n = 100

    # Non-stationary: random walk with drift
    data = pd.DataFrame({
        'non_stationary': np.cumsum(np.random.normal(0.5, 1, n)),
        'stationary': np.random.normal(0, 1, n),
        'target': np.random.normal(0, 1, n)
    })

    return data


class TestStepCleanAnomalies:
    """Tests for StepCleanAnomalies."""

    def test_basic_prep_bake(self, sample_ts_data):
        """Test basic prep and bake functionality."""
        rec = recipe().step_clean_anomalies(
            date_column='date',
            value_columns=['value'],
            method='stl'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert 'value' in result.columns
        assert result.shape[0] == sample_ts_data.shape[0]
        # Anomalies should be reduced
        assert result['value'].max() < sample_ts_data['value'].max()

    def test_twitter_method(self, sample_ts_data):
        """Test Twitter anomaly detection method."""
        rec = recipe().step_clean_anomalies(
            date_column='date',
            value_columns=['value'],
            method='twitter'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape == sample_ts_data.shape

    def test_recipe_method(self, sample_ts_data):
        """Test recipe convenience method."""
        rec = recipe().step_clean_anomalies(
            date_column='date',
            method='stl'
        )
        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]


class TestStepStationary:
    """Tests for StepStationary."""

    def test_basic_differencing(self, sample_stationary_data):
        """Test basic differencing."""
        rec = recipe().step_stationary(
            columns=['non_stationary'],
            max_diff=2,
            test='adf'
        )

        prepped = rec.prep(sample_stationary_data)
        result = prepped.bake(sample_stationary_data)

        assert 'non_stationary' in result.columns
        assert result.shape[0] == sample_stationary_data.shape[0]
        # After differencing, should have some NaN at the beginning
        assert result['non_stationary'].isna().sum() > 0

    def test_already_stationary(self, sample_stationary_data):
        """Test with already stationary data."""
        rec = recipe().step_stationary(
            columns=['stationary'],
            max_diff=2,
            test='adf'
        )

        prepped = rec.prep(sample_stationary_data)
        result = prepped.bake(sample_stationary_data)

        # Should apply minimal or no differencing
        assert 'stationary' in result.columns

    def test_recipe_method(self, sample_stationary_data):
        """Test recipe convenience method."""
        rec = recipe().step_stationary(max_diff=1, test='adf')
        prepped = rec.prep(sample_stationary_data)
        result = prepped.bake(sample_stationary_data)

        assert result.shape[1] == sample_stationary_data.shape[1]


class TestStepDeseasonalize:
    """Tests for StepDeseasonalize."""

    def test_stl_method(self, sample_ts_data):
        """Test STL decomposition."""
        rec = recipe().step_deseasonalize(
            columns=['value'],
            period=30,  # Monthly seasonality
            method='stl'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert 'value' in result.columns
        assert result.shape == sample_ts_data.shape
        # Deseasonalized should have different values
        assert not np.allclose(result['value'].dropna(), sample_ts_data['value'].dropna())

    def test_classical_method(self, sample_ts_data):
        """Test classical decomposition."""
        rec = recipe().step_deseasonalize(
            columns=['value'],
            period=30,
            method='classical',
            model='additive'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape == sample_ts_data.shape

    def test_recipe_method(self, sample_ts_data):
        """Test recipe convenience method."""
        rec = recipe().step_deseasonalize(period=30, method='stl')
        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]


class TestStepDetrend:
    """Tests for StepDetrend."""

    def test_linear_detrending(self, sample_ts_data):
        """Test linear detrending."""
        rec = recipe().step_detrend(
            columns=['value'],
            method='linear'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert 'value' in result.columns
        assert result.shape == sample_ts_data.shape
        # Detrended mean should be near zero
        assert abs(result['value'].mean()) < abs(sample_ts_data['value'].mean())

    def test_constant_detrending(self, sample_ts_data):
        """Test constant detrending (mean removal)."""
        rec = recipe().step_detrend(
            columns=['value'],
            method='constant'
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape == sample_ts_data.shape
        # Mean should be approximately zero
        assert abs(result['value'].mean()) < 1e-10

    def test_recipe_method(self, sample_ts_data):
        """Test recipe convenience method."""
        rec = recipe().step_detrend(method='linear')
        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]


class TestStepHStat:
    """Tests for StepHStat."""

    def test_interaction_detection(self, sample_ts_data):
        """Test interaction detection."""
        rec = recipe().step_h_stat(
            outcome='target',
            columns=['x1', 'x2'],
            top_n=1,
            n_estimators=50
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        # Should create interaction feature
        assert result.shape[1] > sample_ts_data.shape[1]
        # Check for interaction column
        interaction_cols = [c for c in result.columns if '_x_' in c]
        assert len(interaction_cols) > 0

    def test_threshold_selection(self, sample_ts_data):
        """Test threshold-based selection."""
        rec = recipe().step_h_stat(
            outcome='target',
            columns=['x1', 'x2'],
            threshold=0.0,  # Very low threshold
            n_estimators=50
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[1] >= sample_ts_data.shape[1]

    def test_recipe_method(self, sample_ts_data):
        """Test recipe convenience method."""
        rec = recipe().step_h_stat(
            outcome='target',
            top_n=1,
            n_estimators=50
        )
        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]


class TestStepBestLag:
    """Tests for StepBestLag."""

    def test_lag_creation(self, sample_ts_data):
        """Test lag feature creation."""
        # Create autocorrelated data for better Granger results
        data = sample_ts_data.copy()
        for i in range(1, len(data)):
            data.loc[i, 'x1'] = 0.7 * data.loc[i-1, 'x1'] + np.random.normal(0, 1)
            data.loc[i, 'target'] = 0.5 * data.loc[i-1, 'x1'] + np.random.normal(0, 1)

        rec = recipe().step_best_lag(
            outcome='target',
            columns=['x1'],
            max_lag=3,
            alpha=0.2  # More lenient for test
        )

        prepped = rec.prep(data)
        result = prepped.bake(data)

        # May or may not find significant lags depending on data
        assert result.shape[0] == data.shape[0]
        # At minimum, original columns should be present
        assert 'x1' in result.columns
        assert 'target' in result.columns

    def test_no_significant_lags(self, sample_ts_data):
        """Test when no significant lags found."""
        rec = recipe().step_best_lag(
            outcome='target',
            columns=['x1'],
            max_lag=2,
            alpha=0.001  # Very strict
        )

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        # Should not create lag features if none significant
        assert result.shape[1] >= sample_ts_data.shape[1]

    def test_recipe_method(self, sample_ts_data):
        """Test recipe convenience method."""
        rec = recipe().step_best_lag(
            outcome='target',
            max_lag=2,
            alpha=0.2
        )
        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]


class TestIntegration:
    """Integration tests for time series transformation steps."""

    def test_multiple_steps_chain(self, sample_ts_data):
        """Test chaining multiple time series steps."""
        rec = (recipe()
               .step_detrend(columns=['value'], method='linear')
               .step_h_stat(outcome='target', top_n=1, n_estimators=50))

        prepped = rec.prep(sample_ts_data)
        result = prepped.bake(sample_ts_data)

        assert result.shape[0] == sample_ts_data.shape[0]
        # Should have detrended value and interaction features
        assert 'value' in result.columns

    def test_import_from_steps(self):
        """Test that steps can be imported from py_recipes.steps."""
        from py_recipes.steps import (
            StepCleanAnomalies,
            StepStationary,
            StepDeseasonalize,
            StepDetrend,
            StepHStat,
            StepBestLag,
        )

        assert StepCleanAnomalies is not None
        assert StepStationary is not None
        assert StepDeseasonalize is not None
        assert StepDetrend is not None
        assert StepHStat is not None
        assert StepBestLag is not None
