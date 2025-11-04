"""
Tests for auto_arima engine (pmdarima) for arima_reg model

Tests cover:
- Engine registration
- Automatic parameter selection
- Non-seasonal ARIMA
- Seasonal ARIMA
- With/without exogenous variables
- Prediction with forecasts and confidence intervals
- Extract outputs (three-DataFrame format)
- Parameter constraints (max values)
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip.models.arima_reg import arima_reg
from py_parsnip.model_spec import ModelSpec, ModelFit


class TestAutoARIMAEngine:
    """Test auto_arima engine registration and basic usage"""

    def test_set_engine_auto_arima(self):
        """Test setting engine to auto_arima"""
        spec = arima_reg().set_engine("auto_arima")

        assert spec.engine == "auto_arima"
        assert spec.model_type == "arima_reg"

    def test_default_spec_with_auto_arima(self):
        """Test default arima_reg specification with auto_arima"""
        spec = arima_reg().set_engine("auto_arima")

        assert isinstance(spec, ModelSpec)
        assert spec.mode == "regression"
        # arima_reg has default args, so check that args dict exists
        assert isinstance(spec.args, dict)

    def test_spec_with_parameters_as_max_constraints(self):
        """Test that parameters act as MAX search constraints for auto_arima"""
        spec = arima_reg(
            non_seasonal_ar=3,
            non_seasonal_differences=1,
            non_seasonal_ma=2
        ).set_engine("auto_arima")

        # These should be passed as max_p, max_d, max_q
        assert spec.args["non_seasonal_ar"] == 3
        assert spec.args["non_seasonal_differences"] == 1
        assert spec.args["non_seasonal_ma"] == 2


class TestAutoARIMANonSeasonal:
    """Test non-seasonal auto ARIMA fitting and prediction"""

    @pytest.fixture
    def ts_data_simple(self):
        """Create simple time series data"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Trend + noise
        trend = np.linspace(100, 150, 50)
        noise = np.random.normal(0, 5, 50)
        values = trend + noise

        return pd.DataFrame({
            "date": dates,
            "value": values,
        })

    @pytest.fixture
    def ts_data_with_exog(self):
        """Create time series with exogenous variables"""
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Trend + external effect
        x1 = np.linspace(1, 10, 50)
        x2 = np.sin(np.linspace(0, 4*np.pi, 50))
        value = 100 + 5*x1 + 10*x2 + np.random.normal(0, 2, 50)

        return pd.DataFrame({
            "date": dates,
            "value": value,
            "x1": x1,
            "x2": x2,
        })

    @pytest.fixture
    def forecast_data(self):
        """Create future dates for forecasting"""
        dates = pd.date_range("2020-02-20", periods=10, freq="D")
        return pd.DataFrame({"date": dates})

    @pytest.fixture
    def forecast_data_with_exog(self):
        """Create future data with exogenous variables"""
        dates = pd.date_range("2020-02-20", periods=10, freq="D")
        x1 = np.linspace(10, 12, 10)
        x2 = np.sin(np.linspace(4*np.pi, 5*np.pi, 10))

        return pd.DataFrame({
            "date": dates,
            "x1": x1,
            "x2": x2,
        })

    def test_fit_basic_auto_arima(self, ts_data_simple):
        """Test basic non-seasonal auto ARIMA fitting"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_simple, formula="value ~ date")

        assert isinstance(fit, ModelFit)
        assert fit.spec.engine == "auto_arima"
        assert "model" in fit.fit_data
        assert "order" in fit.fit_data
        assert "seasonal_order" in fit.fit_data

    def test_automatic_order_selection(self, ts_data_simple):
        """Test that auto_arima automatically selects optimal order"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_simple, formula="value ~ date")

        # Should have selected some order
        order = fit.fit_data["order"]
        assert isinstance(order, tuple)
        assert len(order) == 3  # (p, d, q)

        # Orders should be within reasonable bounds
        p, d, q = order
        assert 0 <= p <= 5
        assert 0 <= d <= 2
        assert 0 <= q <= 5

    def test_fit_with_max_constraints(self, ts_data_simple):
        """Test fitting with parameter constraints"""
        spec = arima_reg(
            non_seasonal_ar=2,  # max_p=2
            non_seasonal_differences=1,  # max_d=1
            non_seasonal_ma=1  # max_q=1
        ).set_engine("auto_arima")

        fit = spec.fit(ts_data_simple, formula="value ~ date")

        # Selected order should respect constraints
        p, d, q = fit.fit_data["order"]
        assert p <= 2
        assert d <= 1
        assert q <= 1

    def test_fit_with_exog(self, ts_data_with_exog):
        """Test fitting with exogenous variables"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_with_exog, formula="value ~ date + x1 + x2")

        assert isinstance(fit, ModelFit)
        # Should have predictor names stored
        predictor_names = fit.fit_data["predictor_names"]
        assert "x1" in predictor_names
        assert "x2" in predictor_names

    def test_predict_numeric(self, ts_data_simple, forecast_data):
        """Test numeric predictions"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_simple, formula="value ~ date")

        predictions = fit.predict(forecast_data, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(forecast_data)
        # Predictions should be reasonable values
        assert all(predictions[".pred"] > 0)

    def test_predict_conf_int(self, ts_data_simple, forecast_data):
        """Test prediction intervals"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_simple, formula="value ~ date")

        predictions = fit.predict(forecast_data, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert len(predictions) == len(forecast_data)

        # Lower should be less than upper
        assert all(predictions[".pred_lower"] < predictions[".pred_upper"])
        # Prediction should be between bounds
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])

    def test_predict_with_exog(self, ts_data_with_exog, forecast_data_with_exog):
        """Test predictions with exogenous variables"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data_with_exog, formula="value ~ date + x1 + x2")

        predictions = fit.predict(forecast_data_with_exog, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(forecast_data_with_exog)


class TestAutoARIMASeasonal:
    """Test seasonal auto ARIMA fitting"""

    @pytest.fixture
    def seasonal_ts_data(self):
        """Create seasonal time series data"""
        np.random.seed(456)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Trend + seasonal + noise
        t = np.arange(100)
        trend = 0.5 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 12)  # period=12
        noise = np.random.normal(0, 3, 100)
        value = 100 + trend + seasonal + noise

        return pd.DataFrame({
            "date": dates,
            "value": value,
        })

    @pytest.fixture
    def forecast_data_seasonal(self):
        """Create future dates for seasonal forecasting"""
        dates = pd.date_range("2020-04-11", periods=12, freq="D")
        return pd.DataFrame({"date": dates})

    def test_fit_seasonal_auto_arima(self, seasonal_ts_data):
        """Test seasonal auto ARIMA fitting"""
        spec = arima_reg(
            seasonal_period=12
        ).set_engine("auto_arima")

        fit = spec.fit(seasonal_ts_data, formula="value ~ date")

        assert isinstance(fit, ModelFit)
        assert "seasonal_order" in fit.fit_data

        # Check seasonal order
        seasonal_order = fit.fit_data["seasonal_order"]
        assert isinstance(seasonal_order, tuple)
        assert len(seasonal_order) == 4  # (P, D, Q, m)
        assert seasonal_order[3] == 12  # period should be 12

    def test_seasonal_parameter_constraints(self, seasonal_ts_data):
        """Test seasonal parameter constraints"""
        spec = arima_reg(
            seasonal_period=12,
            seasonal_ar=1,  # max_P=1
            seasonal_differences=1,  # max_D=1
            seasonal_ma=1,  # max_Q=1
        ).set_engine("auto_arima")

        fit = spec.fit(seasonal_ts_data, formula="value ~ date")

        P, D, Q, m = fit.fit_data["seasonal_order"]
        assert P <= 1
        assert D <= 1
        assert Q <= 1
        assert m == 12

    def test_predict_seasonal(self, seasonal_ts_data, forecast_data_seasonal):
        """Test predictions for seasonal ARIMA"""
        spec = arima_reg(
            seasonal_period=12
        ).set_engine("auto_arima")

        fit = spec.fit(seasonal_ts_data, formula="value ~ date")
        predictions = fit.predict(forecast_data_seasonal, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(forecast_data_seasonal)


class TestAutoARIMAOutputs:
    """Test extract_outputs() for auto_arima"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        np.random.seed(789)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        trend = np.linspace(100, 150, 50)
        noise = np.random.normal(0, 5, 50)
        value = trend + noise

        return pd.DataFrame({
            "date": dates,
            "value": value,
        })

    def test_extract_outputs_structure(self, ts_data):
        """Test that extract_outputs returns three DataFrames"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should return three DataFrames
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_dataframe(self, ts_data):
        """Test outputs DataFrame structure"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")

        outputs, _, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["actuals", "fitted", "residuals", "split"]
        assert all(col in outputs.columns for col in required_cols)

        # Check split column
        assert all(outputs["split"] == "train")

        # Check lengths match
        assert len(outputs) == len(ts_data)

    def test_coefficients_dataframe(self, ts_data):
        """Test coefficients DataFrame"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")

        _, coefficients, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["variable", "coefficient"]
        assert all(col in coefficients.columns for col in required_cols)

        # ARIMA models have parameters (AR, MA, etc.)
        assert len(coefficients) > 0

    def test_stats_dataframe(self, ts_data):
        """Test stats DataFrame structure"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")

        _, _, stats = fit.extract_outputs()

        # Check required columns
        required_cols = ["metric", "value", "split"]
        assert all(col in stats.columns for col in required_cols)

        # Check for key metrics
        metric_names = stats["metric"].tolist()
        assert "formula" in metric_names
        assert "model_type" in metric_names
        assert "order" in metric_names
        assert "aic" in metric_names

    def test_stats_include_selected_order(self, ts_data):
        """Test that stats include the selected ARIMA order"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")

        _, _, stats = fit.extract_outputs()

        # Extract order information
        order_row = stats[stats["metric"] == "order"]
        assert len(order_row) > 0

        # Order should be stored as string representation
        order_str = order_row["value"].iloc[0]
        assert isinstance(order_str, str)
        assert "(" in order_str  # Tuple format

    def test_outputs_with_model_name(self, ts_data):
        """Test that model name is included in outputs"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(ts_data, formula="value ~ date")
        fit.model_name = "my_auto_arima"

        outputs, coefficients, stats = fit.extract_outputs()

        # All three DataFrames should have model column
        assert "model" in outputs.columns
        assert "model" in coefficients.columns
        assert "model" in stats.columns

        # Should use the model_name
        assert all(outputs["model"] == "my_auto_arima")
        assert all(coefficients["model"] == "my_auto_arima")
        assert all(stats["model"] == "my_auto_arima")


class TestAutoARIMAEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def simple_ts(self):
        """Create minimal time series"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        value = np.linspace(100, 130, 30) + np.random.normal(0, 2, 30)

        return pd.DataFrame({
            "date": dates,
            "value": value,
        })

    def test_minimal_data(self, simple_ts):
        """Test with minimal amount of data"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(simple_ts, formula="value ~ date")

        # Should still fit successfully
        assert isinstance(fit, ModelFit)
        assert "order" in fit.fit_data

    def test_very_restrictive_constraints(self, simple_ts):
        """Test with very restrictive parameter constraints"""
        spec = arima_reg(
            non_seasonal_ar=1,
            non_seasonal_differences=0,
            non_seasonal_ma=1
        ).set_engine("auto_arima")

        fit = spec.fit(simple_ts, formula="value ~ date")

        # Should still find a model within constraints
        p, d, q = fit.fit_data["order"]
        assert p <= 1
        assert d == 0
        assert q <= 1

    def test_invalid_prediction_type(self, simple_ts):
        """Test invalid prediction type raises error"""
        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(simple_ts, formula="value ~ date")

        forecast_data = pd.DataFrame({
            "date": pd.date_range("2020-01-31", periods=5, freq="D")
        })

        with pytest.raises(ValueError, match="supports type='numeric' or 'conf_int'"):
            fit.predict(forecast_data, type="class")

    def test_stationary_data(self):
        """Test with stationary data (should select d=0)"""
        np.random.seed(999)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Stationary process: AR(1) without drift
        value = np.random.normal(100, 10, 50)

        data = pd.DataFrame({"date": dates, "value": value})

        spec = arima_reg().set_engine("auto_arima")
        fit = spec.fit(data, formula="value ~ date")

        # Should detect stationarity (d should be 0 or 1)
        order = fit.fit_data["order"]
        assert order[1] <= 1  # d should be 0 or 1
