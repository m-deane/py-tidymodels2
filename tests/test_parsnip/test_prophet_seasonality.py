"""
Tests for Prophet seasonality control parameters

Verifies that seasonality_yearly, seasonality_weekly, and seasonality_daily
parameters work correctly for both prophet_reg() and prophet_boost().
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.prophet_reg import prophet_reg
from py_parsnip.models.prophet_boost import prophet_boost


@pytest.fixture
def daily_ts_data():
    """Create daily time series data with multiple seasonality patterns"""
    dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
    np.random.seed(42)

    # Add trend
    trend = np.linspace(100, 200, len(dates))

    # Add yearly seasonality
    yearly = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)

    # Add weekly seasonality
    weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)

    # Add noise
    noise = np.random.randn(len(dates)) * 5

    y = trend + yearly + weekly + noise

    return pd.DataFrame({
        'date': dates,
        'y': y
    })


class TestProphetRegSeasonality:
    """Test seasonality parameters for prophet_reg()"""

    def test_default_seasonality_auto(self, daily_ts_data):
        """Test default seasonality='auto' behavior"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Default spec should use 'auto'
        spec = prophet_reg()

        # Verify args contain default 'auto' values
        assert spec.args['yearly_seasonality'] == 'auto'
        assert spec.args['weekly_seasonality'] == 'auto'
        assert spec.args['daily_seasonality'] == 'auto'

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)
        assert '.pred' in predictions.columns

    def test_seasonality_yearly_false(self, daily_ts_data):
        """Test turning off yearly seasonality"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Turn off yearly seasonality
        spec = prophet_reg(seasonality_yearly=False)

        # Verify parameter is set
        assert spec.args['yearly_seasonality'] == False

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)

    def test_seasonality_all_false(self, daily_ts_data):
        """Test turning off all seasonality (trend only)"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Turn off all seasonality
        spec = prophet_reg(
            seasonality_yearly=False,
            seasonality_weekly=False,
            seasonality_daily=False
        )

        # Verify parameters are set
        assert spec.args['yearly_seasonality'] == False
        assert spec.args['weekly_seasonality'] == False
        assert spec.args['daily_seasonality'] == False

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)

    def test_seasonality_explicit_true(self, daily_ts_data):
        """Test explicitly setting seasonality to True"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Explicitly enable seasonality
        spec = prophet_reg(
            seasonality_yearly=True,
            seasonality_weekly=True
        )

        # Verify parameters are set
        assert spec.args['yearly_seasonality'] == True
        assert spec.args['weekly_seasonality'] == True

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)


class TestProphetBoostSeasonality:
    """Test seasonality parameters for prophet_boost()"""

    def test_default_seasonality_auto(self, daily_ts_data):
        """Test default seasonality='auto' behavior"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Default spec should use 'auto'
        spec = prophet_boost(trees=50)

        # Verify args contain default 'auto' values
        assert spec.args['yearly_seasonality'] == 'auto'
        assert spec.args['weekly_seasonality'] == 'auto'
        assert spec.args['daily_seasonality'] == 'auto'

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)
        assert '.pred' in predictions.columns

    def test_seasonality_all_false_xgboost_captures(self, daily_ts_data):
        """Test turning off Prophet seasonality so XGBoost can capture it"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Turn off all Prophet seasonality - let XGBoost handle it
        spec = prophet_boost(
            seasonality_yearly=False,
            seasonality_weekly=False,
            seasonality_daily=False,
            trees=100,
            tree_depth=6
        )

        # Verify parameters are set
        assert spec.args['yearly_seasonality'] == False
        assert spec.args['weekly_seasonality'] == False
        assert spec.args['daily_seasonality'] == False

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)

        # Extract outputs to verify XGBoost component exists
        outputs, _, _ = fit.extract_outputs()
        assert 'prophet_fitted' in outputs.columns
        assert 'xgb_fitted' in outputs.columns
        assert 'fitted' in outputs.columns

        # Verify XGBoost is contributing (not all zeros/NaN)
        xgb_values = outputs['xgb_fitted'].dropna()
        assert len(xgb_values) > 0
        assert not np.allclose(xgb_values, 0, atol=0.1)

    def test_seasonality_yearly_false_only(self, daily_ts_data):
        """Test turning off only yearly seasonality"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Turn off only yearly seasonality
        spec = prophet_boost(
            seasonality_yearly=False,
            trees=75
        )

        # Verify parameter is set
        assert spec.args['yearly_seasonality'] == False
        assert spec.args['weekly_seasonality'] == 'auto'
        assert spec.args['daily_seasonality'] == 'auto'

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)

    def test_seasonality_mixed_settings(self, daily_ts_data):
        """Test mixed seasonality settings"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # Mix of True, False, and 'auto'
        spec = prophet_boost(
            seasonality_yearly=True,
            seasonality_weekly=False,
            seasonality_daily='auto',
            trees=50
        )

        # Verify parameters are set
        assert spec.args['yearly_seasonality'] == True
        assert spec.args['weekly_seasonality'] == False
        assert spec.args['daily_seasonality'] == 'auto'

        # Fit and predict should work
        fit = spec.fit(train, 'y ~ date')
        predictions = fit.predict(test)

        assert predictions is not None
        assert len(predictions) == len(test)


class TestSeasonalityComparison:
    """Compare models with different seasonality settings"""

    def test_prophet_with_vs_without_seasonality(self, daily_ts_data):
        """Compare Prophet with and without seasonality"""
        train = daily_ts_data.iloc[:600].copy()
        test = daily_ts_data.iloc[600:650].copy()

        # With seasonality (auto)
        spec_with = prophet_reg()
        fit_with = spec_with.fit(train, 'y ~ date')
        fit_with = fit_with.evaluate(test)
        outputs_with, _, _ = fit_with.extract_outputs()

        # Without seasonality
        spec_without = prophet_reg(
            seasonality_yearly=False,
            seasonality_weekly=False,
            seasonality_daily=False
        )
        fit_without = spec_without.fit(train, 'y ~ date')
        fit_without = fit_without.evaluate(test)
        outputs_without, _, _ = fit_without.extract_outputs()

        # Both should produce outputs
        assert len(outputs_with) > 0
        assert len(outputs_without) > 0

        # Predictions should be different (seasonality makes a difference)
        test_preds_with = outputs_with[outputs_with['split'] == 'test']['fitted'].values
        test_preds_without = outputs_without[outputs_without['split'] == 'test']['fitted'].values

        # Should be different (not identical)
        assert not np.allclose(test_preds_with, test_preds_without, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
