"""
Time series tests for conformal prediction intervals.

These tests verify:
- EnbPI method for time series models
- Block bootstrap configuration
- Temporal calibration splitting
- Seasonal period detection
- Time series auto-detection
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import linear_reg
from py_parsnip.utils.conformal_utils import (
    is_time_series_model,
    estimate_seasonal_period,
    split_calibration_time_series,
    create_block_bootstrap
)


@pytest.fixture
def time_series_data():
    """Generate simple time series data with trend and seasonality."""
    np.random.seed(42)
    n = 500

    # Create date range (daily data)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Generate features
    X1 = np.sin(np.arange(n) * 2 * np.pi / 7)  # Weekly seasonality
    X2 = np.arange(n) / 100  # Trend
    X3 = np.random.randn(n) * 0.1  # Noise feature

    # Target: trend + seasonality + noise
    y = 10 + 0.5 * X2 + 2 * X1 + np.random.randn(n) * 0.5

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'x1': X1,
        'x2': X2,
        'x3': X3,
        'y': y
    })

    # Split into train/test (chronological)
    train = df.iloc[:400]
    test = df.iloc[400:]

    return train, test


def test_is_time_series_model():
    """Test time series model detection."""
    # Time series models
    assert is_time_series_model('prophet_reg')
    assert is_time_series_model('arima_reg')
    assert is_time_series_model('seasonal_reg')
    assert is_time_series_model('exp_smoothing')
    assert is_time_series_model('varmax_reg')
    assert is_time_series_model('arima_boost')
    assert is_time_series_model('prophet_boost')
    assert is_time_series_model('recursive_reg')

    # Non-time series models
    assert not is_time_series_model('linear_reg')
    assert not is_time_series_model('rand_forest')
    assert not is_time_series_model('decision_tree')
    assert not is_time_series_model('svm_rbf')


def test_estimate_seasonal_period():
    """Test seasonal period estimation from date frequency."""
    # Daily data (weekly seasonality)
    daily_dates = pd.date_range('2020-01-01', periods=100, freq='D')
    daily_df = pd.DataFrame({'date': daily_dates, 'y': range(100)})
    period = estimate_seasonal_period(daily_df, 'date')
    assert period == 7, f"Expected period 7 for daily data, got {period}"

    # Monthly data (annual seasonality)
    monthly_dates = pd.date_range('2020-01-01', periods=100, freq='MS')
    monthly_df = pd.DataFrame({'date': monthly_dates, 'y': range(100)})
    period = estimate_seasonal_period(monthly_df, 'date')
    assert period == 12, f"Expected period 12 for monthly data, got {period}"

    # Weekly data (annual seasonality)
    weekly_dates = pd.date_range('2020-01-01', periods=100, freq='W')
    weekly_df = pd.DataFrame({'date': weekly_dates, 'y': range(100)})
    period = estimate_seasonal_period(weekly_df, 'date')
    assert period == 52, f"Expected period 52 for weekly data, got {period}"


def test_split_calibration_time_series():
    """Test temporal splitting preserves ordering."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'x': range(100),
        'y': range(100)
    })

    train, cal = split_calibration_time_series(df, 'date', calibration_size=0.2)

    # Check sizes
    assert len(train) == 80
    assert len(cal) == 20

    # Check ordering (calibration should be AFTER training)
    assert train['date'].max() < cal['date'].min()

    # Check values
    assert train['x'].max() == 79
    assert cal['x'].min() == 80


def test_create_block_bootstrap():
    """Test block bootstrap creation."""
    from mapie.subsample import BlockBootstrap

    # Create block bootstrap
    bb = create_block_bootstrap(
        n_blocks=7,
        n_resamplings=10,
        overlapping=False,
        random_state=42
    )

    # Check type
    assert isinstance(bb, BlockBootstrap)

    # Check configuration
    assert bb.n_resamplings == 10
    assert bb.n_blocks == 7
    assert bb.overlapping == False


def test_conformal_predict_with_time_series_data(time_series_data):
    """
    Test conformal prediction with time series data.

    Note: This uses linear_reg with time series data structure.
    EnbPI is for actual time series models (prophet, ARIMA), but we test
    that the temporal splitting works correctly.
    """
    train, test = time_series_data

    # Fit linear model on time series data
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Get conformal predictions
    # linear_reg is not a time series model, so should NOT select enbpi
    preds = fit.conformal_predict(test, alpha=0.05, method='auto')

    # Check output
    assert isinstance(preds, pd.DataFrame)
    assert '.pred' in preds.columns
    assert '.pred_lower' in preds.columns
    assert '.pred_upper' in preds.columns

    # Verify intervals are valid
    assert (preds['.pred_lower'] <= preds['.pred']).all()
    assert (preds['.pred'] <= preds['.pred_upper']).all()

    # Check method (should be jackknife+ or cv+, NOT enbpi for linear_reg)
    method = preds['.conf_method'].iloc[0]
    assert method in ['jackknife+', 'cv+', 'split']
    assert method != 'enbpi', "linear_reg should not use enbpi"


def test_conformal_predict_explicit_enbpi_on_linear():
    """
    Test that we can explicitly request EnbPI even for non-time-series models.

    This is useful for testing EnbPI functionality without requiring
    prophet or ARIMA dependencies.
    """
    np.random.seed(42)
    n = 300

    # Create time series structure
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    X = np.random.randn(n, 2)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n) * 0.5

    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['date'] = dates
    df['y'] = y

    # Split
    train = df.iloc[:250]
    test = df.iloc[250:]

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2')

    # Explicitly request EnbPI
    # This tests the EnbPI configuration without needing time series models
    preds = fit.conformal_predict(
        test,
        alpha=0.05,
        method='enbpi',
        n_resamplings=5  # Small number for faster testing
    )

    # Check method was used
    assert preds['.conf_method'].iloc[0] == 'enbpi'

    # Check intervals are valid
    assert len(preds) == len(test)
    assert (preds['.pred_lower'] <= preds['.pred']).all()
    assert (preds['.pred'] <= preds['.pred_upper']).all()


def test_temporal_splitting_preserves_test_distribution(time_series_data):
    """
    Test that temporal splitting for calibration doesn't use future data.

    This is critical for time series: calibration must use past data only.
    """
    train, test = time_series_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # The internal calibration split should use PAST data
    # We can't directly observe this, but we can verify that
    # predictions on test data (future) are reasonable

    # Get conformal predictions
    preds = fit.conformal_predict(test, alpha=0.05, method='split')

    # Calculate coverage on test set
    y_test = test['y'].values
    in_interval = (
        (y_test >= preds['.pred_lower'].values) &
        (y_test <= preds['.pred_upper'].values)
    )
    coverage = in_interval.mean()

    # Should have reasonable coverage even with temporal structure
    assert 0.80 <= coverage <= 1.0, f"Coverage {coverage:.2%} too low for time series"


def test_conformal_predict_with_cv_method_on_time_series(time_series_data):
    """Test CV+ method on time series data."""
    train, test = time_series_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Explicitly use CV+
    preds = fit.conformal_predict(test, alpha=0.05, method='cv+', cv=5)

    # Check method
    assert preds['.conf_method'].iloc[0] == 'cv+'

    # Check intervals
    assert (preds['.pred_lower'] <= preds['.pred']).all()
    assert (preds['.pred'] <= preds['.pred_upper']).all()


def test_seasonal_period_detection_with_hourly_data():
    """Test seasonal period detection with hourly data."""
    # Hourly data (daily seasonality)
    hourly_dates = pd.date_range('2020-01-01', periods=100, freq='H')
    hourly_df = pd.DataFrame({'date': hourly_dates, 'y': range(100)})
    period = estimate_seasonal_period(hourly_df, 'date')
    assert period == 24, f"Expected period 24 for hourly data, got {period}"


def test_seasonal_period_detection_with_quarterly_data():
    """Test seasonal period detection with quarterly data."""
    # Quarterly data (annual seasonality)
    quarterly_dates = pd.date_range('2020-01-01', periods=50, freq='Q')
    quarterly_df = pd.DataFrame({'date': quarterly_dates, 'y': range(50)})
    period = estimate_seasonal_period(quarterly_df, 'date')
    assert period == 4, f"Expected period 4 for quarterly data, got {period}"


def test_seasonal_period_fallback_for_unknown_frequency():
    """Test that unknown frequencies fall back to default."""
    # Create dates with irregular spacing (no clear frequency)
    irregular_dates = pd.to_datetime(['2020-01-01', '2020-01-05', '2020-01-12'])
    irregular_df = pd.DataFrame({'date': irregular_dates, 'y': [1, 2, 3]})
    period = estimate_seasonal_period(irregular_df, 'date')
    # Should return default (10) when frequency cannot be inferred
    assert period == 10, f"Expected default period 10 for irregular data, got {period}"
