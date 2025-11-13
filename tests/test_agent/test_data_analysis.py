"""
Tests for data analysis tools.

Tests the core temporal pattern detection and analysis functions
that power the ForecastAgent's recommendations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_agent.tools.data_analysis import (
    analyze_temporal_patterns,
    detect_seasonality,
    detect_trend,
    calculate_autocorrelation,
    _detect_frequency,
    _detect_outliers
)


class TestAnalyzeTemporalPatterns:
    """Tests for the main analysis function."""

    def test_daily_data_analysis(self):
        """Test analysis of daily time series data."""
        # Create daily data with weekly seasonality
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        # Add weekly pattern + trend + noise
        values = (
            np.sin(np.arange(365) * 2 * np.pi / 7) * 10 +  # Weekly seasonality
            np.arange(365) * 0.1 +  # Trend
            np.random.randn(365) * 2  # Noise
        )

        df = pd.DataFrame({'date': dates, 'value': values})

        result = analyze_temporal_patterns(df, 'date', 'value')

        # Check structure
        assert 'frequency' in result
        assert 'seasonality' in result
        assert 'trend' in result
        assert 'autocorrelation' in result
        assert 'missing_rate' in result
        assert 'outlier_rate' in result

        # Check values
        assert result['frequency'] == 'daily'
        assert result['seasonality']['detected'] == True
        assert result['seasonality']['period'] == 7
        assert result['trend']['direction'] == 'increasing'
        assert result['n_observations'] == 365

    def test_monthly_data_analysis(self):
        """Test analysis of monthly time series data."""
        dates = pd.date_range('2015-01-01', periods=60, freq='MS')
        # Add yearly seasonality
        values = (
            np.sin(np.arange(60) * 2 * np.pi / 12) * 20 +
            np.random.randn(60) * 5 + 100
        )

        df = pd.DataFrame({'date': dates, 'value': values})

        result = analyze_temporal_patterns(df, 'date', 'value')

        assert result['frequency'] == 'monthly'
        assert result['seasonality']['detected'] == True
        assert result['seasonality']['period'] == 12

    def test_with_missing_values(self):
        """Test analysis with missing data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) + 50
        # Introduce missing values
        values[10:20] = np.nan

        df = pd.DataFrame({'date': dates, 'value': values})

        result = analyze_temporal_patterns(df, 'date', 'value')

        assert result['missing_rate'] == 0.1  # 10%

    def test_with_outliers(self):
        """Test analysis with outliers."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) * 2 + 50
        # Add outliers
        values[5] = 1000  # Extreme outlier
        values[20] = -500

        df = pd.DataFrame({'date': dates, 'value': values})

        result = analyze_temporal_patterns(df, 'date', 'value')

        assert result['outlier_rate'] > 0.01  # At least 2 outliers


class TestDetectSeasonality:
    """Tests for seasonality detection."""

    def test_detect_strong_seasonality(self):
        """Test detection of strong seasonal pattern."""
        # Create strong weekly pattern
        n = 100
        series = np.sin(np.arange(n) * 2 * np.pi / 7) * 10 + 50

        result = detect_seasonality(series, frequency='daily', period=7)

        assert result['detected'] == True
        assert result['period'] == 7
        assert result['strength'] > 0.5  # Strong seasonality

    def test_detect_no_seasonality(self):
        """Test when no seasonality present."""
        # Random walk (no seasonality)
        series = np.cumsum(np.random.randn(100))

        result = detect_seasonality(series, frequency='daily', period=7)

        assert result['detected'] == False or result['strength'] < 0.3

    def test_insufficient_data(self):
        """Test with insufficient data for period."""
        series = np.random.randn(10)  # Only 10 points for period=7

        result = detect_seasonality(series, frequency='daily', period=7)

        assert result['detected'] == False
        assert 'reason' in result

    def test_monthly_seasonality(self):
        """Test monthly data with yearly seasonality."""
        # 5 years of monthly data
        series = np.sin(np.arange(60) * 2 * np.pi / 12) * 15 + 100

        result = detect_seasonality(series, frequency='monthly', period=12)

        assert result['detected'] == True
        assert result['period'] == 12


class TestDetectTrend:
    """Tests for trend detection."""

    def test_detect_increasing_trend(self):
        """Test detection of increasing trend."""
        series = np.arange(100) + np.random.randn(100) * 5

        result = detect_trend(series)

        assert result['direction'] == 'increasing'
        assert result['significant'] == True
        assert result['slope'] > 0
        assert result['p_value'] < 0.05

    def test_detect_decreasing_trend(self):
        """Test detection of decreasing trend."""
        series = -np.arange(100) + np.random.randn(100) * 5

        result = detect_trend(series)

        assert result['direction'] == 'decreasing'
        assert result['significant'] == True
        assert result['slope'] < 0

    def test_detect_stable_trend(self):
        """Test detection of stable (no trend)."""
        series = np.random.randn(100) + 50  # Just noise around mean

        result = detect_trend(series)

        assert result['direction'] == 'stable'
        assert result['significant'] == False

    def test_trend_with_nans(self):
        """Test trend detection with missing values."""
        series = np.arange(100).astype(float)
        series[10:15] = np.nan

        result = detect_trend(series)

        # Should still detect increasing trend after imputation
        assert result['direction'] == 'increasing'


class TestCalculateAutocorrelation:
    """Tests for autocorrelation calculation."""

    def test_high_autocorrelation(self):
        """Test with high autocorrelation (AR process)."""
        # AR(1) process with phi=0.9
        series = [0]
        for _ in range(99):
            series.append(0.9 * series[-1] + np.random.randn())
        series = np.array(series)

        result = calculate_autocorrelation(series, lags=[1, 5, 10])

        assert result['lag_1'] > 0.7  # High at lag 1
        assert 'lag_5' in result
        assert 'lag_10' in result

    def test_low_autocorrelation(self):
        """Test with white noise (no autocorrelation)."""
        series = np.random.randn(100)

        result = calculate_autocorrelation(series, lags=[1, 7])

        assert abs(result['lag_1']) < 0.3  # Low autocorrelation
        assert abs(result['lag_7']) < 0.3

    def test_custom_lags(self):
        """Test with custom lag values."""
        series = np.random.randn(100)

        result = calculate_autocorrelation(series, lags=[2, 3, 5, 7])

        assert len(result) == 4
        assert 'lag_2' in result
        assert 'lag_3' in result
        assert 'lag_5' in result
        assert 'lag_7' in result

    def test_insufficient_data_for_lag(self):
        """Test when series is shorter than lag."""
        series = np.random.randn(5)

        result = calculate_autocorrelation(series, lags=[10, 20])

        # Should return 0 for lags longer than series
        assert result['lag_10'] == 0.0
        assert result['lag_20'] == 0.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_detect_daily_frequency(self):
        """Test frequency detection for daily data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        freq = _detect_frequency(dates)

        assert freq == 'daily'

    def test_detect_weekly_frequency(self):
        """Test frequency detection for weekly data."""
        dates = pd.date_range('2020-01-01', periods=52, freq='W')

        freq = _detect_frequency(dates)

        assert freq == 'weekly'

    def test_detect_monthly_frequency(self):
        """Test frequency detection for monthly data."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')

        freq = _detect_frequency(dates)

        assert freq == 'monthly'

    def test_detect_outliers_normal_data(self):
        """Test outlier detection with normal data."""
        series = np.random.randn(100)

        outlier_rate = _detect_outliers(series, threshold=3.0)

        assert outlier_rate < 0.05  # Should be very few outliers

    def test_detect_outliers_with_extremes(self):
        """Test outlier detection with extreme values."""
        series = np.random.randn(100)
        series[5] = 100  # Extreme outlier
        series[20] = -100

        outlier_rate = _detect_outliers(series, threshold=3.0)

        assert outlier_rate >= 0.02  # At least 2 outliers out of 100


def test_integration_full_analysis():
    """Integration test: full analysis pipeline."""
    # Create realistic time series
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=730, freq='D')  # 2 years

    # Components:
    trend = np.arange(730) * 0.05
    seasonal = np.sin(np.arange(730) * 2 * np.pi / 7) * 10  # Weekly
    noise = np.random.randn(730) * 3
    values = 100 + trend + seasonal + noise

    # Add some missing and outliers
    values[100:110] = np.nan
    values[200] = values[200] + 100  # Outlier

    df = pd.DataFrame({'date': dates, 'sales': values})

    # Run full analysis
    result = analyze_temporal_patterns(df, 'date', 'sales')

    # Verify comprehensive results
    assert result['frequency'] == 'daily'
    assert result['seasonality']['detected'] == True
    assert result['trend']['significant'] == True
    assert result['missing_rate'] > 0
    assert result['outlier_rate'] > 0
    assert 'lag_1' in result['autocorrelation']
    assert result['n_observations'] == 730
    assert 'date_range' in result
