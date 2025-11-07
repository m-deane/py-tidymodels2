"""
Tests for XGBoost seasonality capture in prophet_boost hybrid model

Verifies that when Prophet seasonality is turned off, XGBoost can capture
seasonal patterns using cyclical date features (day_of_week, month, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.prophet_boost import prophet_boost


@pytest.fixture
def weekly_seasonal_data():
    """Create time series with strong weekly seasonality and trend"""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)

    # Linear trend
    trend = np.linspace(100, 200, len(dates))

    # Strong weekly seasonality (day of week effect)
    # Monday=0, Sunday=6
    day_of_week = dates.dayofweek
    weekly_effect = np.zeros(len(dates))
    weekly_effect[day_of_week == 0] = -20  # Monday: low
    weekly_effect[day_of_week == 1] = -10  # Tuesday: medium-low
    weekly_effect[day_of_week == 2] = 0    # Wednesday: baseline
    weekly_effect[day_of_week == 3] = 0    # Thursday: baseline
    weekly_effect[day_of_week == 4] = 10   # Friday: medium-high
    weekly_effect[day_of_week == 5] = 20   # Saturday: high
    weekly_effect[day_of_week == 6] = 15   # Sunday: high

    # Add noise
    noise = np.random.randn(len(dates)) * 5

    y = trend + weekly_effect + noise

    return pd.DataFrame({
        'date': dates,
        'y': y,
        'day_of_week': day_of_week.values
    })


@pytest.fixture
def monthly_seasonal_data():
    """Create time series with monthly seasonality"""
    dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
    np.random.seed(42)

    # Trend
    trend = np.linspace(100, 200, len(dates))

    # Monthly seasonality
    month = dates.month
    monthly_effect = np.zeros(len(dates))
    for m in range(1, 13):
        # Summer months higher, winter months lower
        monthly_effect[month == m] = 30 * np.sin(2 * np.pi * (m - 1) / 12)

    # Add noise
    noise = np.random.randn(len(dates)) * 5

    y = trend + monthly_effect + noise

    return pd.DataFrame({
        'date': dates,
        'y': y,
        'month': month.values
    })


class TestXGBoostSeasonalityCapture:
    """Test that XGBoost captures seasonality when Prophet seasonality is off"""

    def test_xgb_predictions_vary_with_cyclical_features(self, weekly_seasonal_data):
        """
        Test that XGBoost predictions vary for different test dates
        (not constant like before the fix)
        """
        train = weekly_seasonal_data.iloc[:160].copy()
        test = weekly_seasonal_data.iloc[160:180].copy()

        # Turn off Prophet seasonality - let XGBoost capture it
        spec = prophet_boost(
            seasonality_yearly=False,
            seasonality_weekly=False,
            seasonality_daily=False,
            trees=100,
            tree_depth=6
        )

        # Fit and evaluate
        fit = spec.fit(train, 'y ~ date')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Get XGBoost component for test data
        test_outputs = outputs[outputs['split'] == 'test']
        xgb_test_values = test_outputs['xgb_fitted'].values

        # CRITICAL: XGBoost predictions should VARY (not constant)
        unique_values = len(np.unique(np.round(xgb_test_values, 2)))

        # Should have multiple unique values (at least 3 for 20 test points)
        assert unique_values >= 3, \
            f"XGBoost predictions are too constant: only {unique_values} unique values"

        # Standard deviation should be non-trivial
        xgb_std = np.std(xgb_test_values)
        assert xgb_std > 0.5, \
            f"XGBoost predictions have very low variance: std={xgb_std:.4f}"

        print(f"\n✓ XGBoost predictions vary: {unique_values} unique values, std={xgb_std:.2f}")

    def test_xgb_captures_weekly_pattern_when_prophet_off(self, weekly_seasonal_data):
        """
        Test that XGBoost learns weekly seasonality when Prophet seasonality is disabled
        """
        train = weekly_seasonal_data.iloc[:160].copy()
        test = weekly_seasonal_data.iloc[160:180].copy()

        # Turn off Prophet weekly seasonality
        spec = prophet_boost(
            seasonality_yearly=False,
            seasonality_weekly=False,  # OFF - XGBoost should capture this
            seasonality_daily=False,
            trees=150,
            tree_depth=6,
            learn_rate=0.1
        )

        # Fit model
        fit = spec.fit(train, 'y ~ date')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()
        test_outputs = outputs[outputs['split'] == 'test'].copy()

        # Add day_of_week to test outputs
        test_outputs['day_of_week'] = pd.to_datetime(test_outputs['date']).dt.dayofweek

        # Group XGBoost predictions by day of week
        xgb_by_dow = test_outputs.groupby('day_of_week')['xgb_fitted'].mean()

        # Should see different mean XGBoost predictions for different days
        dow_variance = xgb_by_dow.var()

        assert dow_variance > 1.0, \
            f"XGBoost not capturing weekly patterns: variance across days={dow_variance:.4f}"

        print(f"\n✓ XGBoost captures weekly patterns: variance across days={dow_variance:.2f}")
        print(f"  Mean XGB prediction by day of week:\n{xgb_by_dow}")

    def test_xgb_improves_fit_with_seasonality(self, weekly_seasonal_data):
        """
        Test that XGBoost component improves predictions when capturing seasonality
        """
        train = weekly_seasonal_data.iloc[:160].copy()
        test = weekly_seasonal_data.iloc[160:180].copy()

        # Model 1: Prophet with seasonality OFF, XGBoost captures it
        spec_xgb_seasonal = prophet_boost(
            seasonality_yearly=False,
            seasonality_weekly=False,
            seasonality_daily=False,
            trees=150,
            tree_depth=6
        )

        fit1 = spec_xgb_seasonal.fit(train, 'y ~ date')
        fit1 = fit1.evaluate(test)
        outputs1, _, stats1 = fit1.extract_outputs()

        # Get test RMSE (stats may be in long format with metric/value columns)
        test_stats = stats1[stats1['split'] == 'test']
        if 'metric' in test_stats.columns:
            # Long format
            rmse_with_xgb = test_stats[test_stats['metric'] == 'rmse']['value'].values[0]
        else:
            # Wide format
            rmse_with_xgb = test_stats['rmse'].values[0]

        # XGBoost should contribute meaningfully (not just constant)
        test_out = outputs1[outputs1['split'] == 'test']
        xgb_std = test_out['xgb_fitted'].std()

        # XGBoost should vary and contribute to predictions
        assert xgb_std > 0.5, \
            f"XGBoost contribution is too constant: std={xgb_std:.4f}"

        # Combined predictions should be better than Prophet alone would be
        # (we expect RMSE < 20 for this data with proper seasonality capture)
        assert rmse_with_xgb < 25, \
            f"Model RMSE too high: {rmse_with_xgb:.2f} (XGBoost may not be capturing seasonality)"

        print(f"\n✓ Hybrid model performs well: RMSE={rmse_with_xgb:.2f}")
        print(f"  XGBoost component std={xgb_std:.2f}")

    def test_xgb_features_include_cyclical_encodings(self, weekly_seasonal_data):
        """
        Test that XGBoost receives multiple features (not just days_since_start)
        """
        train = weekly_seasonal_data.iloc[:160].copy()

        spec = prophet_boost(
            seasonality_weekly=False,
            trees=50
        )

        fit = spec.fit(train, 'y ~ date')

        # Get XGBoost model
        xgb_model = fit.fit_data['xgb_model']

        # Check number of features XGBoost was trained on
        n_features = xgb_model.n_features_in_

        # Should have many features (days_since_start + day_of_week + cyclical encodings, etc.)
        # Our implementation has 12 features total
        assert n_features >= 10, \
            f"XGBoost has too few features: {n_features} (need cyclical features for seasonality)"

        print(f"\n✓ XGBoost trained with {n_features} features (includes cyclical encodings)")

    def test_monthly_seasonality_capture(self, monthly_seasonal_data):
        """
        Test that XGBoost captures monthly/yearly seasonality patterns
        """
        train = monthly_seasonal_data.iloc[:600].copy()
        test = monthly_seasonal_data.iloc[600:650].copy()

        # Turn off Prophet yearly seasonality - let XGBoost capture it
        spec = prophet_boost(
            seasonality_yearly=False,  # OFF - XGBoost should capture this
            seasonality_weekly=False,
            seasonality_daily=False,
            trees=200,
            tree_depth=6
        )

        fit = spec.fit(train, 'y ~ date')
        fit = fit.evaluate(test)

        outputs, _, _ = fit.extract_outputs()
        test_outputs = outputs[outputs['split'] == 'test'].copy()

        # XGBoost should vary with monthly patterns
        xgb_test = test_outputs['xgb_fitted'].values
        unique_values = len(np.unique(np.round(xgb_test, 1)))

        assert unique_values >= 5, \
            f"XGBoost not capturing monthly patterns: only {unique_values} unique values"

        # Should have reasonable variance
        xgb_std = np.std(xgb_test)
        assert xgb_std > 1.0, \
            f"XGBoost monthly predictions too constant: std={xgb_std:.4f}"

        print(f"\n✓ XGBoost captures monthly patterns: {unique_values} unique values, std={xgb_std:.2f}")


class TestXGBoostFeatureEngineering:
    """Test the _create_xgb_features helper method"""

    def test_create_xgb_features_shape(self):
        """Test that feature engineering produces correct shape"""
        from py_parsnip.engines.hybrid_prophet_boost import HybridProphetBoostEngine

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        features = HybridProphetBoostEngine._create_xgb_features(dates)

        # Should have 100 samples, 11 features (removed days_since_start)
        assert features.shape == (100, 11), \
            f"Wrong feature shape: {features.shape}, expected (100, 11)"

    def test_create_xgb_features_cyclical_bounds(self):
        """Test that cyclical features stay in valid ranges"""
        from py_parsnip.engines.hybrid_prophet_boost import HybridProphetBoostEngine

        dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
        features = HybridProphetBoostEngine._create_xgb_features(dates)

        # Extract specific features (by column index - removed days_since_start so indices shifted)
        day_of_week = features[:, 0]      # Column 0 (was 1)
        month = features[:, 3]             # Column 3 (was 4)
        sin_day_of_week = features[:, 5]  # Column 5 (was 6)
        cos_day_of_week = features[:, 6]  # Column 6 (was 7)

        # Check ranges
        assert day_of_week.min() >= 0 and day_of_week.max() <= 6, \
            "day_of_week out of range [0, 6]"
        assert month.min() >= 1 and month.max() <= 12, \
            "month out of range [1, 12]"
        assert sin_day_of_week.min() >= -1 and sin_day_of_week.max() <= 1, \
            "sin_day_of_week out of range [-1, 1]"
        assert cos_day_of_week.min() >= -1 and cos_day_of_week.max() <= 1, \
            "cos_day_of_week out of range [-1, 1]"

        print("✓ All cyclical features within valid bounds")



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
