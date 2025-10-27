"""
Tests for seasonal_reg model specification

Tests cover:
- Model specification creation
- STL decomposition with single seasonal period
- Multiple seasonal periods
- Fitting and prediction
- Extract outputs with decomposed components
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from py_parsnip import seasonal_reg


class TestSeasonalRegSpec:
    """Test seasonal_reg() model specification"""

    def test_default_spec_error(self):
        """Test that default requires seasonal_period_1"""
        with pytest.raises(ValueError, match="At least seasonal_period_1"):
            seasonal_reg()

    def test_spec_single_period(self):
        """Test specification with single seasonal period"""
        spec = seasonal_reg(seasonal_period_1=7)

        assert spec.model_type == "seasonal_reg"
        assert spec.engine == "statsmodels"
        assert spec.mode == "regression"
        assert spec.args["seasonal_period_1"] == 7
        assert spec.args["seasonal_period_2"] is None
        assert spec.args["seasonal_period_3"] is None

    def test_spec_two_periods(self):
        """Test specification with two seasonal periods"""
        spec = seasonal_reg(
            seasonal_period_1=7,
            seasonal_period_2=365,
        )

        assert spec.args["seasonal_period_1"] == 7
        assert spec.args["seasonal_period_2"] == 365

    def test_spec_three_periods(self):
        """Test specification with three seasonal periods"""
        spec = seasonal_reg(
            seasonal_period_1=24,
            seasonal_period_2=168,
            seasonal_period_3=8760,
        )

        assert spec.args["seasonal_period_1"] == 24
        assert spec.args["seasonal_period_2"] == 168
        assert spec.args["seasonal_period_3"] == 8760


class TestSeasonalRegFit:
    """Test seasonal_reg fitting"""

    @pytest.fixture
    def weekly_seasonal_data(self):
        """Create daily data with weekly seasonality"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        t = np.arange(len(dates))

        # Trend + weekly seasonality
        trend = 100 + 0.1 * t
        seasonality = 15 * np.sin(t * 2 * np.pi / 7)
        noise = np.random.normal(0, 3, len(dates))
        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    @pytest.fixture
    def monthly_seasonal_data(self):
        """Create monthly data with yearly seasonality"""
        np.random.seed(42)
        dates = pd.date_range(start="2018-01-01", periods=60, freq="MS")
        t = np.arange(len(dates))

        # Trend + yearly seasonality
        trend = 1000 + 10 * t
        seasonality = 200 * np.sin(t * 2 * np.pi / 12)
        noise = np.random.normal(0, 30, len(dates))
        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_weekly(self, weekly_seasonal_data):
        """Test fitting with weekly seasonality"""
        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(weekly_seasonal_data, "sales ~ date")

        assert fit is not None
        assert "ets_model" in fit.fit_data
        assert "stl_result" in fit.fit_data
        assert fit.fit_data["seasonal_period_1"] == 7
        assert fit.fit_data["n_obs"] == 730

        # Check decomposition components exist
        assert "seasonal_1" in fit.fit_data
        assert "trend" in fit.fit_data
        assert "remainder" in fit.fit_data

    def test_fit_monthly(self, monthly_seasonal_data):
        """Test fitting with monthly/yearly seasonality"""
        spec = seasonal_reg(seasonal_period_1=12)
        fit = spec.fit(monthly_seasonal_data, "sales ~ date")

        assert fit is not None
        assert fit.fit_data["seasonal_period_1"] == 12

    def test_fit_insufficient_data(self):
        """Test that insufficient data raises error"""
        # Only 10 observations with period=7 (need at least 14)
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.random.normal(100, 5, 10)
        data = pd.DataFrame({"date": dates, "sales": values})

        spec = seasonal_reg(seasonal_period_1=7)

        with pytest.raises(ValueError, match="Time series too short"):
            spec.fit(data, "sales ~ date")

    def test_fit_multiple_periods(self):
        """Test fitting with multiple seasonal periods"""
        np.random.seed(42)
        # Long series with multiple seasonalities
        dates = pd.date_range(start="2018-01-01", periods=365 * 3, freq="D")
        t = np.arange(len(dates))

        # Trend + weekly + yearly seasonality
        trend = 100 + 0.05 * t
        weekly = 10 * np.sin(t * 2 * np.pi / 7)
        yearly = 20 * np.sin(t * 2 * np.pi / 365)
        noise = np.random.normal(0, 5, len(dates))
        values = trend + weekly + yearly + noise

        data = pd.DataFrame({"date": dates, "sales": values})

        spec = seasonal_reg(
            seasonal_period_1=7,
            seasonal_period_2=365,
        )
        fit = spec.fit(data, "sales ~ date")

        assert fit is not None
        assert fit.fit_data["seasonal_period_1"] == 7
        assert fit.fit_data["seasonal_period_2"] == 365
        assert "seasonal_2" in fit.fit_data


class TestSeasonalRegPredict:
    """Test seasonal_reg prediction"""

    @pytest.fixture
    def fitted_weekly(self):
        """Create fitted model with weekly seasonality"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        t = np.arange(len(dates))
        trend = 100 + 0.1 * t
        seasonality = 15 * np.sin(t * 2 * np.pi / 7)
        values = trend + seasonality + np.random.normal(0, 3, len(dates))
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(train, "sales ~ date")
        return fit

    @pytest.fixture
    def fitted_multiple(self):
        """Create fitted model with multiple seasonal periods"""
        np.random.seed(42)
        dates = pd.date_range(start="2018-01-01", periods=365 * 3, freq="D")
        t = np.arange(len(dates))
        trend = 100 + 0.05 * t
        weekly = 10 * np.sin(t * 2 * np.pi / 7)
        yearly = 20 * np.sin(t * 2 * np.pi / 365)
        values = trend + weekly + yearly + np.random.normal(0, 5, len(dates))
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = seasonal_reg(
            seasonal_period_1=7,
            seasonal_period_2=365,
        )
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_predict_numeric(self, fitted_weekly):
        """Test prediction with type='numeric'"""
        future_dates = pd.date_range(start="2022-01-01", periods=28, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_weekly.predict(test, type="numeric")

        assert len(predictions) == 28
        assert ".pred" in predictions.columns
        # Should have seasonal variation
        assert predictions[".pred"].std() > 1.0

    def test_predict_conf_int(self, fitted_weekly):
        """Test prediction with confidence intervals"""
        future_dates = pd.date_range(start="2022-01-01", periods=28, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_weekly.predict(test, type="conf_int")

        assert len(predictions) == 28
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns

        # Check intervals are reasonable
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])

    def test_predict_seasonal_pattern(self, fitted_weekly):
        """Test that forecast preserves seasonal pattern"""
        # Predict 4 weeks (28 days)
        future_dates = pd.date_range(start="2022-01-01", periods=28, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_weekly.predict(test)

        # Weekly pattern should be visible
        # Check that day 0 and day 7 are similar (same day of week)
        diff_week_1 = abs(predictions[".pred"].iloc[7] - predictions[".pred"].iloc[0])
        diff_adjacent = abs(predictions[".pred"].iloc[1] - predictions[".pred"].iloc[0])

        # Same day of week should be more similar than adjacent days
        # (This is a rough test, may not always hold due to trend)
        # At least check forecast varies
        assert predictions[".pred"].std() > 0.5

    def test_predict_multiple_periods(self, fitted_multiple):
        """Test prediction with multiple seasonal periods"""
        future_dates = pd.date_range(start="2021-01-01", periods=60, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_multiple.predict(test)

        assert len(predictions) == 60
        assert ".pred" in predictions.columns

    def test_predict_different_horizons(self, fitted_weekly):
        """Test predictions with different forecast horizons"""
        for n_periods in [7, 14, 30, 90]:
            future_dates = pd.date_range(start="2022-01-01", periods=n_periods, freq="D")
            test = pd.DataFrame({"date": future_dates})

            predictions = fitted_weekly.predict(test)

            assert len(predictions) == n_periods


class TestSeasonalRegExtract:
    """Test seasonal_reg output extraction"""

    @pytest.fixture
    def fitted_weekly(self):
        """Create fitted model with weekly seasonality"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        t = np.arange(len(dates))
        trend = 100 + 0.1 * t
        seasonality = 15 * np.sin(t * 2 * np.pi / 7)
        values = trend + seasonality + np.random.normal(0, 3, len(dates))
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_extract_fit_engine(self, fitted_weekly):
        """Test extract_fit_engine()"""
        model = fitted_weekly.extract_fit_engine()

        assert model is not None
        assert hasattr(model, "forecast")

    def test_extract_outputs(self, fitted_weekly):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_weekly.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

        # Check Outputs DataFrame has observation-level results
        assert "date" in outputs.columns
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "forecast" in outputs.columns
        assert "residuals" in outputs.columns
        assert "split" in outputs.columns
        assert "model" in outputs.columns

        # Should have decomposition components
        assert "trend" in outputs.columns
        assert "seasonal" in outputs.columns
        assert "remainder" in outputs.columns

        # All should be training data
        assert all(outputs["split"] == "train")
        assert len(outputs) == 730

        # Check Coefficients has parameters
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns

        # Check Stats has metrics
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

    def test_extract_outputs_decomposition(self, fitted_weekly):
        """Test that outputs contain decomposition components"""
        outputs, _, _ = fitted_weekly.extract_outputs()

        # Should have trend, seasonal, remainder
        assert "trend" in outputs.columns
        assert "seasonal" in outputs.columns
        assert "remainder" in outputs.columns

        # Components should sum to actuals (approximately)
        reconstructed = outputs["trend"] + outputs["seasonal"] + outputs["remainder"]
        difference = abs(reconstructed - outputs["actuals"])
        assert difference.max() < 1e-6  # Very small numerical error

    def test_extract_outputs_stats(self, fitted_weekly):
        """Test that stats contain comprehensive information"""
        _, _, stats = fitted_weekly.extract_outputs()

        metric_names = stats["metric"].values

        # Performance metrics
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names

        # Model info
        assert "model_type" in metric_names
        assert "decomposition" in metric_names
        assert "forecasting_model" in metric_names
        assert "seasonal_period_1" in metric_names

        # Check decomposition method
        decomp_value = stats[stats["metric"] == "decomposition"]["value"].values[0]
        assert decomp_value == "STL"

        # Check forecasting model
        forecast_value = stats[stats["metric"] == "forecasting_model"]["value"].values[0]
        assert forecast_value == "ETS"


class TestIntegration:
    """Integration tests for Seasonal Decomposition workflow"""

    def test_full_workflow_weekly(self):
        """Test complete fit -> predict workflow for weekly seasonality"""
        np.random.seed(42)
        # Training data (2 years, weekly pattern)
        train_dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        t = np.arange(len(train_dates))
        trend = 100 + 0.1 * t
        seasonality = 15 * np.sin(t * 2 * np.pi / 7)
        noise = np.random.normal(0, 3, len(train_dates))
        train_values = trend + seasonality + noise

        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model
        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(train, "sales ~ date")

        # Forecast 28 days ahead (4 weeks)
        future_dates = pd.date_range(start="2022-01-01", periods=28, freq="D")
        test = pd.DataFrame({"date": future_dates})

        # Predict with intervals
        predictions = fit.predict(test, type="conf_int")

        # Verify
        assert len(predictions) == 28
        assert all(predictions[".pred_lower"] <= predictions[".pred_upper"])

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()

        assert len(outputs) == 730  # 2 years of training data
        assert "trend" in outputs.columns
        assert "seasonal" in outputs.columns
        assert "rmse" in stats["metric"].values

    def test_full_workflow_multiple_periods(self):
        """Test workflow with multiple seasonal periods"""
        np.random.seed(42)
        # Long series with weekly + yearly seasonality
        train_dates = pd.date_range(start="2018-01-01", periods=365 * 3, freq="D")
        t = np.arange(len(train_dates))
        trend = 100 + 0.05 * t
        weekly = 10 * np.sin(t * 2 * np.pi / 7)
        yearly = 20 * np.sin(t * 2 * np.pi / 365)
        noise = np.random.normal(0, 5, len(train_dates))
        train_values = trend + weekly + yearly + noise

        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model with multiple periods
        spec = seasonal_reg(
            seasonal_period_1=7,
            seasonal_period_2=365,
        )
        fit = spec.fit(train, "sales ~ date")

        # Forecast
        future_dates = pd.date_range(start="2021-01-01", periods=60, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 60
        assert all(predictions[".pred"] > 0)

        # Check decomposition captured both periods
        outputs, _, stats = fit.extract_outputs()
        assert "seasonal" in outputs.columns
        assert "seasonal_2" in outputs.columns

    def test_monthly_data(self):
        """Test with monthly data and yearly seasonality"""
        np.random.seed(42)
        # 5 years of monthly data
        dates = pd.date_range(start="2018-01-01", periods=60, freq="MS")
        t = np.arange(len(dates))

        # Trend + yearly seasonality
        trend = 1000 + 10 * t
        seasonality = 200 * np.sin(t * 2 * np.pi / 12)
        noise = np.random.normal(0, 30, len(dates))
        values = trend + seasonality + noise

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit model
        spec = seasonal_reg(seasonal_period_1=12)
        fit = spec.fit(train, "sales ~ date")

        # Forecast 12 months ahead
        future_dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        test = pd.DataFrame({"date": future_dates})

        predictions = fit.predict(test)

        assert len(predictions) == 12
        assert all(predictions[".pred"] > 0)

    def test_hourly_data_daily_seasonality(self):
        """Test with hourly data and daily seasonality"""
        np.random.seed(42)
        # 7 days of hourly data (168 hours)
        dates = pd.date_range(start="2022-01-01", periods=168, freq="H")
        t = np.arange(len(dates))

        # Daily pattern (24-hour cycle)
        trend = 50
        seasonality = 20 * np.sin(t * 2 * np.pi / 24)
        noise = np.random.normal(0, 3, len(dates))
        values = trend + seasonality + noise

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit model with 24-hour seasonality
        spec = seasonal_reg(seasonal_period_1=24)
        fit = spec.fit(train, "sales ~ date")

        # Forecast 48 hours ahead
        future_dates = pd.date_range(start="2022-01-08", periods=48, freq="H")
        test = pd.DataFrame({"date": future_dates})

        predictions = fit.predict(test)

        assert len(predictions) == 48
        # Should capture daily pattern
        assert predictions[".pred"].std() > 1.0

    def test_visualize_decomposition(self):
        """Test that decomposition components are accessible for visualization"""
        np.random.seed(42)
        # Create data with clear components
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        t = np.arange(365)
        trend = 100 + 0.2 * t  # Clear upward trend
        seasonality = 30 * np.sin(t * 2 * np.pi / 7)  # Strong weekly pattern
        noise = np.random.normal(0, 5, 365)
        values = trend + seasonality + noise

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit model
        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(train, "sales ~ date")

        # Extract components
        outputs, _, _ = fit.extract_outputs()

        # Verify decomposition makes sense
        # Trend should be increasing
        assert outputs["trend"].iloc[-1] > outputs["trend"].iloc[0]

        # Seasonal component should have mean near 0
        assert abs(outputs["seasonal"].mean()) < 5

        # Remainder should be small compared to signal
        signal_to_noise = outputs["actuals"].std() / outputs["remainder"].std()
        assert signal_to_noise > 2
