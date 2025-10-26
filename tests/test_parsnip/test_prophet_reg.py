"""
Tests for prophet_reg model specification

Tests cover:
- Model specification creation
- Fitting with datetime data
- Prediction with intervals
- Extract outputs
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from py_parsnip import prophet_reg


class TestProphetRegSpec:
    """Test prophet_reg() model specification"""

    def test_default_spec(self):
        """Test default prophet_reg specification"""
        spec = prophet_reg()

        assert spec.model_type == "prophet_reg"
        assert spec.engine == "prophet"
        assert spec.mode == "regression"
        assert spec.args["growth"] == "linear"
        assert spec.args["seasonality_mode"] == "additive"

    def test_spec_with_custom_params(self):
        """Test prophet_reg with custom parameters"""
        spec = prophet_reg(
            growth="logistic",
            changepoint_prior_scale=0.1,
            seasonality_mode="multiplicative",
        )

        assert spec.args["growth"] == "logistic"
        assert spec.args["changepoint_prior_scale"] == 0.1
        assert spec.args["seasonality_mode"] == "multiplicative"


class TestProphetRegFit:
    """Test prophet_reg fitting"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data"""
        # Create daily data for 2 years
        dates = pd.date_range(start="2022-01-01", periods=730, freq="D")

        # Simple trend + seasonality
        trend = np.linspace(100, 200, 730)
        seasonality = 10 * np.sin(np.arange(730) * 2 * np.pi / 365)
        noise = np.random.RandomState(42).normal(0, 5, 730)

        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_basic(self, time_series_data):
        """Test basic fitting"""
        spec = prophet_reg()
        fit = spec.fit(time_series_data, "sales ~ date")

        assert fit is not None
        assert "model" in fit.fit_data
        assert fit.fit_data["n_obs"] == 730

    def test_fit_with_flexible_trend(self, time_series_data):
        """Test fitting with more flexible trend"""
        spec = prophet_reg(changepoint_prior_scale=0.1)
        fit = spec.fit(time_series_data, "sales ~ date")

        assert fit.spec.args["changepoint_prior_scale"] == 0.1


class TestProphetRegPredict:
    """Test prophet_reg prediction"""

    @pytest.fixture
    def fitted_prophet(self):
        """Create fitted Prophet model"""
        # Simple time series
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        values = np.linspace(100, 200, 365) + np.random.RandomState(42).normal(
            0, 5, 365
        )

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = prophet_reg()
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_predict_numeric(self, fitted_prophet):
        """Test prediction with type='numeric'"""
        # Future dates
        future_dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_prophet.predict(test, type="numeric")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)  # Sales should be positive

    def test_predict_conf_int(self, fitted_prophet):
        """Test prediction with confidence intervals"""
        future_dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_prophet.predict(test, type="conf_int")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns

        # Check intervals are reasonable
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])


class TestProphetRegExtract:
    """Test prophet_reg output extraction"""

    @pytest.fixture
    def fitted_prophet(self):
        """Create fitted Prophet model"""
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        values = np.linspace(100, 200, 365) + np.random.RandomState(42).normal(
            0, 5, 365
        )

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = prophet_reg()
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_extract_fit_engine(self, fitted_prophet):
        """Test extract_fit_engine()"""
        prophet_model = fitted_prophet.extract_fit_engine()

        assert prophet_model is not None
        assert hasattr(prophet_model, "predict")

    def test_extract_outputs(self, fitted_prophet):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_prophet.extract_outputs()

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

        # All should be training data
        assert all(outputs["split"] == "train")

        # Check Coefficients has hyperparameters
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns

        # Check Stats has metrics
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns


class TestIntegration:
    """Integration tests for Prophet workflow"""

    def test_full_workflow(self):
        """Test complete fit → predict workflow"""
        # Training data (2 years)
        train_dates = pd.date_range(start="2020-01-01", periods=730, freq="D")
        train_values = (
            np.linspace(100, 200, 730)
            + 20 * np.sin(np.arange(730) * 2 * np.pi / 365)
            + np.random.RandomState(42).normal(0, 5, 730)
        )
        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model
        spec = prophet_reg(
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0
        )
        fit = spec.fit(train, "sales ~ date")

        # Forecast 90 days ahead
        future_dates = pd.date_range(start="2022-01-01", periods=90, freq="D")
        test = pd.DataFrame({"date": future_dates})

        # Predict with intervals
        predictions = fit.predict(test, type="conf_int")

        # Verify
        assert len(predictions) == 90
        assert all(predictions[".pred"] > 0)
        assert all(predictions[".pred_lower"] <= predictions[".pred_upper"])

    def test_multiplicative_seasonality(self):
        """Test with multiplicative seasonality"""
        # Create data with multiplicative pattern
        dates = pd.date_range(start="2020-01-01", periods=730, freq="D")
        trend = np.exp(np.linspace(4, 5, 730))  # Exponential growth
        seasonality = 1 + 0.2 * np.sin(np.arange(730) * 2 * np.pi / 365)
        values = trend * seasonality

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit with multiplicative seasonality
        spec = prophet_reg(seasonality_mode="multiplicative")
        fit = spec.fit(train, "sales ~ date")

        # Predict
        future_dates = pd.date_range(start="2022-01-01", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fit.predict(test)

        assert len(predictions) == 30
        assert all(predictions[".pred"] > 0)
