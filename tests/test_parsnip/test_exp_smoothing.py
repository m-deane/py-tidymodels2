"""
Tests for exp_smoothing model specification

Tests cover:
- Model specification creation with different parameters
- Simple Exponential Smoothing
- Holt's Linear Method (trend)
- Holt-Winters Seasonal Method
- Fitting and prediction
- Extract outputs with three-DataFrame structure
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from py_parsnip import exp_smoothing


class TestExpSmoothingSpec:
    """Test exp_smoothing() model specification"""

    def test_default_spec(self):
        """Test default exp_smoothing specification (Simple ES)"""
        spec = exp_smoothing()

        assert spec.model_type == "exp_smoothing"
        assert spec.engine == "statsmodels"
        assert spec.mode == "regression"
        assert spec.args["error"] == "additive"
        assert spec.args["trend"] is None
        assert spec.args["season"] is None
        assert spec.args["damping"] is False

    def test_spec_simple(self):
        """Test Simple Exponential Smoothing specification"""
        spec = exp_smoothing(
            error="additive",
            trend=None,
            season=None,
        )

        assert spec.args["trend"] is None
        assert spec.args["season"] is None

    def test_spec_holt(self):
        """Test Holt's Linear Method specification"""
        spec = exp_smoothing(
            trend="additive",
            season=None,
        )

        assert spec.args["trend"] == "additive"
        assert spec.args["season"] is None

    def test_spec_holt_winters(self):
        """Test Holt-Winters specification"""
        spec = exp_smoothing(
            seasonal_period=12,
            trend="additive",
            season="additive",
        )

        assert spec.args["seasonal_period"] == 12
        assert spec.args["trend"] == "additive"
        assert spec.args["season"] == "additive"

    def test_spec_damped(self):
        """Test damped trend specification"""
        spec = exp_smoothing(
            trend="additive",
            damping=True,
        )

        assert spec.args["trend"] == "additive"
        assert spec.args["damping"] is True

    def test_spec_multiplicative(self):
        """Test multiplicative components"""
        spec = exp_smoothing(
            seasonal_period=12,
            error="multiplicative",
            trend="multiplicative",
            season="multiplicative",
        )

        assert spec.args["error"] == "multiplicative"
        assert spec.args["trend"] == "multiplicative"
        assert spec.args["season"] == "multiplicative"

    def test_validation_season_requires_period(self):
        """Test that season requires seasonal_period"""
        with pytest.raises(ValueError, match="seasonal_period must be specified"):
            exp_smoothing(season="additive")

    def test_validation_damping_requires_trend(self):
        """Test that damping requires trend"""
        with pytest.raises(ValueError, match="damping requires trend"):
            exp_smoothing(damping=True)


class TestExpSmoothingFit:
    """Test exp_smoothing fitting"""

    @pytest.fixture
    def simple_data(self):
        """Create simple time series data (no trend, no seasonality)"""
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        # Constant level with noise
        values = 100 + np.random.normal(0, 5, 100)

        return pd.DataFrame({"date": dates, "sales": values})

    @pytest.fixture
    def trend_data(self):
        """Create time series with trend"""
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        # Linear trend
        trend = np.linspace(100, 200, 100)
        noise = np.random.normal(0, 5, 100)
        values = trend + noise

        return pd.DataFrame({"date": dates, "sales": values})

    @pytest.fixture
    def seasonal_data(self):
        """Create time series with trend and seasonality"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        # Trend + weekly seasonality
        t = np.arange(len(dates))
        trend = 100 + 0.1 * t
        seasonality = 10 * np.sin(t * 2 * np.pi / 7)  # Weekly pattern
        noise = np.random.normal(0, 3, len(dates))
        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_simple(self, simple_data):
        """Test Simple Exponential Smoothing"""
        spec = exp_smoothing()
        fit = spec.fit(simple_data, "sales ~ date")

        assert fit is not None
        assert "model" in fit.fit_data
        assert fit.fit_data["method"] == "simple"
        assert fit.fit_data["n_obs"] == 100

    def test_fit_holt(self, trend_data):
        """Test Holt's Linear Method"""
        spec = exp_smoothing(trend="additive")
        fit = spec.fit(trend_data, "sales ~ date")

        assert fit is not None
        assert fit.fit_data["method"] == "holt"
        assert fit.spec.args["trend"] == "additive"

    def test_fit_holt_winters(self, seasonal_data):
        """Test Holt-Winters Seasonal Method"""
        spec = exp_smoothing(
            seasonal_period=7,
            trend="additive",
            season="additive",
        )
        fit = spec.fit(seasonal_data, "sales ~ date")

        assert fit is not None
        assert fit.fit_data["method"] == "holt-winters"
        assert fit.fit_data["seasonal_period"] == 7

    def test_fit_damped(self, trend_data):
        """Test damped trend model"""
        spec = exp_smoothing(trend="additive", damping=True)
        fit = spec.fit(trend_data, "sales ~ date")

        assert fit is not None
        assert fit.spec.args["damping"] is True


class TestExpSmoothingPredict:
    """Test exp_smoothing prediction"""

    @pytest.fixture
    def fitted_simple(self):
        """Create fitted Simple ES model"""
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        values = 100 + np.random.normal(0, 5, 100)
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = exp_smoothing()
        fit = spec.fit(train, "sales ~ date")
        return fit

    @pytest.fixture
    def fitted_holt(self):
        """Create fitted Holt model"""
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        trend = np.linspace(100, 200, 100)
        values = trend + np.random.normal(0, 5, 100)
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = exp_smoothing(trend="additive")
        fit = spec.fit(train, "sales ~ date")
        return fit

    @pytest.fixture
    def fitted_hw(self):
        """Create fitted Holt-Winters model"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        t = np.arange(365)
        trend = 100 + 0.1 * t
        seasonality = 10 * np.sin(t * 2 * np.pi / 7)
        values = trend + seasonality + np.random.normal(0, 3, 365)
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = exp_smoothing(
            seasonal_period=7,
            trend="additive",
            season="additive",
        )
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_predict_numeric_simple(self, fitted_simple):
        """Test prediction with Simple ES"""
        future_dates = pd.date_range(start="2022-04-11", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_simple.predict(test, type="numeric")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        # Simple ES produces flat forecast
        assert predictions[".pred"].std() < 1.0  # Nearly constant

    def test_predict_numeric_holt(self, fitted_holt):
        """Test prediction with Holt's method"""
        future_dates = pd.date_range(start="2022-04-11", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_holt.predict(test, type="numeric")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        # Holt produces trending forecast
        assert predictions[".pred"].iloc[-1] > predictions[".pred"].iloc[0]

    def test_predict_numeric_hw(self, fitted_hw):
        """Test prediction with Holt-Winters"""
        future_dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_hw.predict(test, type="numeric")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        # Holt-Winters produces seasonal forecast
        assert predictions[".pred"].std() > 1.0  # Has variation

    def test_predict_conf_int(self, fitted_holt):
        """Test prediction with confidence intervals"""
        future_dates = pd.date_range(start="2022-04-11", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_holt.predict(test, type="conf_int")

        assert len(predictions) == 30
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns

        # Check intervals are reasonable
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])

    def test_predict_different_horizons(self, fitted_holt):
        """Test predictions with different forecast horizons"""
        for n_periods in [1, 7, 30, 90]:
            future_dates = pd.date_range(start="2022-04-11", periods=n_periods, freq="D")
            test = pd.DataFrame({"date": future_dates})

            predictions = fitted_holt.predict(test)

            assert len(predictions) == n_periods


class TestExpSmoothingExtract:
    """Test exp_smoothing output extraction"""

    @pytest.fixture
    def fitted_hw(self):
        """Create fitted Holt-Winters model"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        t = np.arange(365)
        trend = 100 + 0.1 * t
        seasonality = 10 * np.sin(t * 2 * np.pi / 7)
        values = trend + seasonality + np.random.normal(0, 3, 365)
        train = pd.DataFrame({"date": dates, "sales": values})

        spec = exp_smoothing(
            seasonal_period=7,
            trend="additive",
            season="additive",
        )
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_extract_fit_engine(self, fitted_hw):
        """Test extract_fit_engine()"""
        model = fitted_hw.extract_fit_engine()

        assert model is not None
        assert hasattr(model, "forecast")

    def test_extract_outputs(self, fitted_hw):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_hw.extract_outputs()

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
        assert len(outputs) == 365

        # Check Coefficients has smoothing parameters
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns
        # Should have alpha, beta, gamma
        assert len(coefs) >= 2  # At least alpha and beta

        # Check Stats has metrics
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Should have standard metrics
        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names

    def test_extract_outputs_coefficients(self, fitted_hw):
        """Test that coefficients contain smoothing parameters"""
        _, coefs, _ = fitted_hw.extract_outputs()

        # Extract variable names
        variables = coefs["variable"].values

        # Should contain alpha (level), beta (trend), gamma (seasonal)
        has_alpha = any("alpha" in str(v).lower() for v in variables)
        has_beta = any("beta" in str(v).lower() for v in variables)
        has_gamma = any("gamma" in str(v).lower() for v in variables)

        assert has_alpha, "Should have alpha (level smoothing parameter)"
        assert has_beta, "Should have beta (trend smoothing parameter)"
        assert has_gamma, "Should have gamma (seasonal smoothing parameter)"

    def test_extract_outputs_stats(self, fitted_hw):
        """Test that stats contain comprehensive information"""
        _, _, stats = fitted_hw.extract_outputs()

        metric_names = stats["metric"].values

        # Performance metrics
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "mape" in metric_names
        assert "r_squared" in metric_names

        # Model info
        assert "model_type" in metric_names
        assert "method" in metric_names
        assert "seasonal_period" in metric_names

        # Check method is holt-winters
        method_value = stats[stats["metric"] == "method"]["value"].values[0]
        assert method_value == "holt-winters"


class TestIntegration:
    """Integration tests for Exponential Smoothing workflow"""

    def test_full_workflow_simple(self):
        """Test complete fit -> predict workflow for Simple ES"""
        np.random.seed(42)
        # Training data
        train_dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        train_values = 100 + np.random.normal(0, 5, 100)
        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model
        spec = exp_smoothing()
        fit = spec.fit(train, "sales ~ date")

        # Forecast 30 days ahead
        future_dates = pd.date_range(start="2022-04-11", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        # Predict with intervals
        predictions = fit.predict(test, type="conf_int")

        # Verify
        assert len(predictions) == 30
        assert all(predictions[".pred_lower"] <= predictions[".pred_upper"])

    def test_full_workflow_holt_winters(self):
        """Test complete workflow for Holt-Winters"""
        np.random.seed(42)
        # Training data (2 years, weekly seasonality)
        train_dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        t = np.arange(len(train_dates))
        trend = 100 + 0.05 * t
        seasonality = 15 * np.sin(t * 2 * np.pi / 7)
        noise = np.random.normal(0, 5, len(train_dates))
        train_values = trend + seasonality + noise

        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit Holt-Winters model
        spec = exp_smoothing(
            seasonal_period=7,
            trend="additive",
            season="additive",
        )
        fit = spec.fit(train, "sales ~ date")

        # Forecast 28 days ahead (4 weeks)
        future_dates = pd.date_range(start="2022-01-01", periods=28, freq="D")
        test = pd.DataFrame({"date": future_dates})

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 28
        assert all(predictions[".pred"] > 0)

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()

        assert len(outputs) == 730  # 2 years of training data
        assert len(coefs) >= 3  # alpha, beta, gamma
        assert "rmse" in stats["metric"].values

    def test_comparison_simple_vs_holt(self):
        """Compare Simple ES vs Holt on trending data"""
        np.random.seed(42)
        # Data with clear trend
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        trend = np.linspace(100, 200, 100)
        values = trend + np.random.normal(0, 5, 100)
        train = pd.DataFrame({"date": dates, "sales": values})

        # Simple ES
        spec_simple = exp_smoothing()
        fit_simple = spec_simple.fit(train, "sales ~ date")

        # Holt
        spec_holt = exp_smoothing(trend="additive")
        fit_holt = spec_holt.fit(train, "sales ~ date")

        # Forecast
        future_dates = pd.date_range(start="2022-04-11", periods=30, freq="D")
        test = pd.DataFrame({"date": future_dates})

        pred_simple = fit_simple.predict(test)
        pred_holt = fit_holt.predict(test)

        # Holt should have higher final predictions (captures trend)
        assert pred_holt[".pred"].iloc[-1] > pred_simple[".pred"].iloc[-1]

    def test_multiplicative_components(self):
        """Test multiplicative Holt-Winters"""
        np.random.seed(42)
        # Data with multiplicative seasonality (amplitude grows with level)
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        t = np.arange(365)
        trend = np.exp(np.linspace(4, 5, 365))  # Exponential trend
        seasonality = 1 + 0.2 * np.sin(t * 2 * np.pi / 7)  # Multiplicative
        values = trend * seasonality

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit with multiplicative components
        spec = exp_smoothing(
            seasonal_period=7,
            error="multiplicative",
            trend="multiplicative",
            season="multiplicative",
        )
        fit = spec.fit(train, "sales ~ date")

        # Predict
        future_dates = pd.date_range(start="2021-01-01", periods=14, freq="D")
        test = pd.DataFrame({"date": future_dates})

        predictions = fit.predict(test)

        assert len(predictions) == 14
        assert all(predictions[".pred"] > 0)
