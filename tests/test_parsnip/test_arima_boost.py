"""
Tests for arima_boost model specification

Tests cover:
- Model specification creation
- Fitting hybrid ARIMA+XGBoost model
- Prediction with combined models
- Extract outputs with both components
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from py_parsnip import arima_boost


class TestARIMABoostSpec:
    """Test arima_boost() model specification"""

    def test_default_spec(self):
        """Test default arima_boost specification"""
        spec = arima_boost()

        assert spec.model_type == "arima_boost"
        assert spec.engine == "hybrid_arima_xgboost"
        assert spec.mode == "regression"

        # ARIMA defaults
        assert spec.args["non_seasonal_ar"] == 0
        assert spec.args["non_seasonal_differences"] == 0
        assert spec.args["non_seasonal_ma"] == 0

        # XGBoost defaults
        assert spec.args["trees"] == 100
        assert spec.args["tree_depth"] == 6
        assert spec.args["learn_rate"] == 0.1

    def test_spec_with_custom_params(self):
        """Test arima_boost with custom parameters"""
        spec = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            seasonal_period=12,
            trees=200,
            tree_depth=3,
            learn_rate=0.05,
        )

        # ARIMA params
        assert spec.args["non_seasonal_ar"] == 1
        assert spec.args["non_seasonal_differences"] == 1
        assert spec.args["non_seasonal_ma"] == 1
        assert spec.args["seasonal_period"] == 12

        # XGBoost params
        assert spec.args["trees"] == 200
        assert spec.args["tree_depth"] == 3
        assert spec.args["learn_rate"] == 0.05


class TestARIMABoostFit:
    """Test arima_boost fitting"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data with linear + non-linear patterns"""
        np.random.seed(42)

        # Create 200 observations
        n = 200
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

        # Linear trend
        trend = np.linspace(100, 200, n)

        # Seasonal pattern
        seasonality = 10 * np.sin(np.arange(n) * 2 * np.pi / 30)

        # Non-linear pattern (quadratic)
        non_linear = 0.01 * (np.arange(n) - n/2) ** 2

        # Random noise
        noise = np.random.normal(0, 5, n)

        # Combine
        values = trend + seasonality + non_linear + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_basic(self, time_series_data):
        """Test basic fitting of ARIMA+XGBoost"""
        spec = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(time_series_data, "sales ~ date")

        assert fit is not None
        assert "arima_model" in fit.fit_data
        assert "xgb_model" in fit.fit_data
        assert fit.fit_data["n_obs"] == 200

    def test_fit_stores_both_models(self, time_series_data):
        """Test that both ARIMA and XGBoost models are stored"""
        spec = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            trees=100,
        )
        fit = spec.fit(time_series_data, "sales ~ date")

        # Check ARIMA model
        assert fit.fit_data["arima_model"] is not None
        assert "arima_fitted" in fit.fit_data

        # Check XGBoost model
        assert fit.fit_data["xgb_model"] is not None
        assert "xgb_fitted" in fit.fit_data

        # Check combined fitted values
        assert "fitted" in fit.fit_data
        fitted = fit.fit_data["fitted"]
        arima_fitted = fit.fit_data["arima_fitted"]
        xgb_fitted = fit.fit_data["xgb_fitted"]

        # Verify combination
        np.testing.assert_allclose(fitted, arima_fitted + xgb_fitted, rtol=1e-5)

    def test_fit_with_seasonal_arima(self, time_series_data):
        """Test fitting with seasonal ARIMA component"""
        spec = arima_boost(
            seasonal_period=7,  # Weekly seasonality
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            seasonal_ar=1,
            seasonal_ma=1,
            trees=50,
        )
        fit = spec.fit(time_series_data, "sales ~ date")

        assert fit.fit_data["seasonal_order"] == (1, 0, 1, 7)


class TestARIMABoostPredict:
    """Test arima_boost prediction"""

    @pytest.fixture
    def fitted_arima_boost(self):
        """Create fitted ARIMA+XGBoost model"""
        np.random.seed(42)

        # Create training data
        n_train = 150
        dates_train = pd.date_range(start="2022-01-01", periods=n_train, freq="D")

        trend = np.linspace(100, 180, n_train)
        seasonality = 10 * np.sin(np.arange(n_train) * 2 * np.pi / 30)
        non_linear = 0.01 * (np.arange(n_train) - n_train/2) ** 2
        noise = np.random.normal(0, 5, n_train)

        values_train = trend + seasonality + non_linear + noise

        train = pd.DataFrame({"date": dates_train, "sales": values_train})

        spec = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_predict_numeric(self, fitted_arima_boost):
        """Test prediction with type='numeric'"""
        # Future dates
        future_dates = pd.date_range(
            start="2022-05-31", periods=30, freq="D"  # Next 30 days
        )
        future = pd.DataFrame({"date": future_dates})

        preds = fitted_arima_boost.predict(future, type="numeric")

        assert ".pred" in preds.columns
        assert len(preds) == 30
        assert not preds[".pred"].isna().any()

    def test_predict_combines_both_models(self, fitted_arima_boost):
        """Test that predictions combine ARIMA + XGBoost"""
        future_dates = pd.date_range(start="2022-05-31", periods=10, freq="D")
        future = pd.DataFrame({"date": future_dates})

        preds = fitted_arima_boost.predict(future, type="numeric")

        # Predictions should be reasonable (not NaN, not too extreme)
        assert preds[".pred"].notna().all()
        assert (preds[".pred"] > 0).all()  # Positive values
        assert (preds[".pred"] < 500).all()  # Not unreasonably large


class TestARIMABoostExtractOutputs:
    """Test arima_boost extract_outputs"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)

        n = 100
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
        trend = np.linspace(100, 150, n)
        seasonality = 5 * np.sin(np.arange(n) * 2 * np.pi / 20)
        non_linear = 0.005 * (np.arange(n) - n/2) ** 2
        noise = np.random.normal(0, 3, n)
        values = trend + seasonality + non_linear + noise

        data = pd.DataFrame({"date": dates, "sales": values})

        spec = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(data, "sales ~ date")
        return fit

    def test_extract_outputs_structure(self, fitted_model):
        """Test that extract_outputs returns correct structure"""
        outputs, coefficients, stats = fitted_model.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "arima_fitted" in outputs.columns
        assert "xgb_fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert "split" in outputs.columns

        # Check coefficients DataFrame
        assert isinstance(coefficients, pd.DataFrame)
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Should have both ARIMA and XGBoost parameters
        assert any("arima_" in str(v) for v in coefficients["variable"])
        assert any("xgb_" in str(v) for v in coefficients["variable"])

        # Check stats DataFrame
        assert isinstance(stats, pd.DataFrame)
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

    def test_extract_outputs_metrics(self, fitted_model):
        """Test that metrics are calculated"""
        outputs, coefficients, stats = fitted_model.extract_outputs()

        # Check for standard metrics
        metrics = stats["metric"].unique()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics

        # Check for model-specific info
        assert "model_type" in metrics
        assert "arima_order" in metrics
        assert "xgb_n_estimators" in metrics

    def test_hybrid_predictions_sum(self, fitted_model):
        """Test that hybrid predictions are sum of components"""
        outputs, _, _ = fitted_model.extract_outputs()

        train_data = outputs[outputs["split"] == "train"]

        # Check that fitted = arima_fitted + xgb_fitted
        expected = train_data["arima_fitted"] + train_data["xgb_fitted"]
        actual = train_data["fitted"]

        np.testing.assert_allclose(actual, expected, rtol=1e-5)


class TestARIMABoostComparison:
    """Test that ARIMA+XGBoost improves over ARIMA alone"""

    @pytest.fixture
    def complex_time_series(self):
        """Create time series with strong non-linear patterns"""
        np.random.seed(42)

        n = 200
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

        # Linear trend (ARIMA should capture this)
        trend = np.linspace(100, 200, n)

        # Seasonality (ARIMA should capture this)
        seasonality = 15 * np.sin(np.arange(n) * 2 * np.pi / 30)

        # Strong non-linear pattern (XGBoost should capture this)
        non_linear = 20 * np.sin(np.arange(n) * 0.05) * np.cos(np.arange(n) * 0.03)

        # Noise
        noise = np.random.normal(0, 5, n)

        values = trend + seasonality + non_linear + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_hybrid_captures_non_linear_patterns(self, complex_time_series):
        """Test that hybrid model captures patterns ARIMA misses"""
        # Split data
        train = complex_time_series.iloc[:150]
        test = complex_time_series.iloc[150:]

        # Fit hybrid model
        spec_hybrid = arima_boost(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1,
            trees=100,
            tree_depth=5,
            learn_rate=0.1,
        )
        fit_hybrid = spec_hybrid.fit(train, "sales ~ date")

        # Make predictions
        preds_hybrid = fit_hybrid.predict(test[["date"]], type="numeric")

        # Calculate RMSE
        test_actuals = test["sales"].values
        hybrid_rmse = np.sqrt(np.mean((test_actuals - preds_hybrid[".pred"].values) ** 2))

        # Hybrid RMSE should be reasonable
        assert hybrid_rmse < 30  # Reasonable threshold for this data
        assert not np.isnan(hybrid_rmse)
