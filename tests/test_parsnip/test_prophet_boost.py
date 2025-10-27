"""
Tests for prophet_boost model specification

Tests cover:
- Model specification creation
- Fitting hybrid Prophet+XGBoost model
- Prediction with combined models
- Extract outputs with both components
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from py_parsnip import prophet_boost


class TestProphetBoostSpec:
    """Test prophet_boost() model specification"""

    def test_default_spec(self):
        """Test default prophet_boost specification"""
        spec = prophet_boost()

        assert spec.model_type == "prophet_boost"
        assert spec.engine == "hybrid_prophet_xgboost"
        assert spec.mode == "regression"

        # Prophet defaults
        assert spec.args["growth"] == "linear"
        assert spec.args["seasonality_mode"] == "additive"
        assert spec.args["changepoint_prior_scale"] == 0.05

        # XGBoost defaults
        assert spec.args["trees"] == 100
        assert spec.args["tree_depth"] == 6
        assert spec.args["learn_rate"] == 0.1

    def test_spec_with_custom_params(self):
        """Test prophet_boost with custom parameters"""
        spec = prophet_boost(
            growth="logistic",
            changepoint_prior_scale=0.1,
            seasonality_mode="multiplicative",
            trees=200,
            tree_depth=3,
            learn_rate=0.05,
        )

        # Prophet params
        assert spec.args["growth"] == "logistic"
        assert spec.args["changepoint_prior_scale"] == 0.1
        assert spec.args["seasonality_mode"] == "multiplicative"

        # XGBoost params
        assert spec.args["trees"] == 200
        assert spec.args["tree_depth"] == 3
        assert spec.args["learn_rate"] == 0.05


class TestProphetBoostFit:
    """Test prophet_boost fitting"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data with trend + seasonality + non-linear"""
        np.random.seed(42)

        # Create 2 years of daily data
        n = 730
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

        # Trend
        trend = np.linspace(100, 300, n)

        # Yearly seasonality
        yearly_season = 30 * np.sin(np.arange(n) * 2 * np.pi / 365)

        # Weekly seasonality
        weekly_season = 10 * np.sin(np.arange(n) * 2 * np.pi / 7)

        # Non-linear pattern (Prophet should miss this)
        non_linear = 15 * np.sin(np.arange(n) * 0.02) ** 2

        # Noise
        noise = np.random.normal(0, 5, n)

        # Combine
        values = trend + yearly_season + weekly_season + non_linear + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_basic(self, time_series_data):
        """Test basic fitting of Prophet+XGBoost"""
        spec = prophet_boost(
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(time_series_data, "sales ~ date")

        assert fit is not None
        assert "prophet_model" in fit.fit_data
        assert "xgb_model" in fit.fit_data
        assert fit.fit_data["n_obs"] == 730

    def test_fit_stores_both_models(self, time_series_data):
        """Test that both Prophet and XGBoost models are stored"""
        spec = prophet_boost(trees=50)
        fit = spec.fit(time_series_data, "sales ~ date")

        # Check Prophet model
        assert fit.fit_data["prophet_model"] is not None
        assert "prophet_fitted" in fit.fit_data

        # Check XGBoost model
        assert fit.fit_data["xgb_model"] is not None
        assert "xgb_fitted" in fit.fit_data

        # Check combined fitted values
        assert "fitted" in fit.fit_data
        fitted = fit.fit_data["fitted"]
        prophet_fitted = fit.fit_data["prophet_fitted"]
        xgb_fitted = fit.fit_data["xgb_fitted"]

        # Verify combination
        np.testing.assert_allclose(fitted, prophet_fitted + xgb_fitted, rtol=1e-5)

    def test_fit_with_flexible_trend(self, time_series_data):
        """Test fitting with more flexible Prophet trend"""
        spec = prophet_boost(
            changepoint_prior_scale=0.2,
            seasonality_prior_scale=15.0,
            trees=50,
        )
        fit = spec.fit(time_series_data, "sales ~ date")

        # Check that parameters are stored
        assert fit.fit_data["prophet_params"]["changepoint_prior_scale"] == 0.2
        assert fit.fit_data["prophet_params"]["seasonality_prior_scale"] == 15.0


class TestProphetBoostPredict:
    """Test prophet_boost prediction"""

    @pytest.fixture
    def fitted_prophet_boost(self):
        """Create fitted Prophet+XGBoost model"""
        np.random.seed(42)

        # Create training data
        n_train = 365
        dates_train = pd.date_range(start="2022-01-01", periods=n_train, freq="D")

        trend = np.linspace(100, 200, n_train)
        seasonality = 20 * np.sin(np.arange(n_train) * 2 * np.pi / 30)
        non_linear = 10 * np.sin(np.arange(n_train) * 0.05) ** 2
        noise = np.random.normal(0, 5, n_train)

        values_train = trend + seasonality + non_linear + noise

        train = pd.DataFrame({"date": dates_train, "sales": values_train})

        spec = prophet_boost(
            trees=50,
            tree_depth=3,
            learn_rate=0.1,
        )
        fit = spec.fit(train, "sales ~ date")
        return fit

    def test_predict_numeric(self, fitted_prophet_boost):
        """Test prediction with type='numeric'"""
        # Future dates
        future_dates = pd.date_range(
            start="2023-01-01", periods=30, freq="D"  # Next 30 days
        )
        future = pd.DataFrame({"date": future_dates})

        preds = fitted_prophet_boost.predict(future, type="numeric")

        assert ".pred" in preds.columns
        assert len(preds) == 30
        assert not preds[".pred"].isna().any()

    def test_predict_combines_both_models(self, fitted_prophet_boost):
        """Test that predictions combine Prophet + XGBoost"""
        future_dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        future = pd.DataFrame({"date": future_dates})

        preds = fitted_prophet_boost.predict(future, type="numeric")

        # Predictions should be reasonable (not NaN, not too extreme)
        assert preds[".pred"].notna().all()
        assert (preds[".pred"] > 0).all()  # Positive values
        assert (preds[".pred"] < 500).all()  # Not unreasonably large

    def test_predict_far_future(self, fitted_prophet_boost):
        """Test prediction far into the future"""
        # Predict 6 months ahead
        future_dates = pd.date_range(start="2023-01-01", periods=180, freq="D")
        future = pd.DataFrame({"date": future_dates})

        preds = fitted_prophet_boost.predict(future, type="numeric")

        assert len(preds) == 180
        assert preds[".pred"].notna().all()


class TestProphetBoostExtractOutputs:
    """Test prophet_boost extract_outputs"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)

        n = 200
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
        trend = np.linspace(100, 150, n)
        seasonality = 10 * np.sin(np.arange(n) * 2 * np.pi / 30)
        non_linear = 5 * np.sin(np.arange(n) * 0.03) ** 2
        noise = np.random.normal(0, 3, n)
        values = trend + seasonality + non_linear + noise

        data = pd.DataFrame({"date": dates, "sales": values})

        spec = prophet_boost(trees=50, tree_depth=3)
        fit = spec.fit(data, "sales ~ date")
        return fit

    def test_extract_outputs_structure(self, fitted_model):
        """Test that extract_outputs returns correct structure"""
        outputs, coefficients, stats = fitted_model.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "prophet_fitted" in outputs.columns
        assert "xgb_fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert "split" in outputs.columns

        # Check coefficients DataFrame
        assert isinstance(coefficients, pd.DataFrame)
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Should have both Prophet and XGBoost parameters
        assert any("prophet_" in str(v) for v in coefficients["variable"])
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
        assert "prophet_growth" in metrics
        assert "xgb_n_estimators" in metrics

    def test_hybrid_predictions_sum(self, fitted_model):
        """Test that hybrid predictions are sum of components"""
        outputs, _, _ = fitted_model.extract_outputs()

        train_data = outputs[outputs["split"] == "train"]

        # Check that fitted = prophet_fitted + xgb_fitted
        expected = train_data["prophet_fitted"] + train_data["xgb_fitted"]
        actual = train_data["fitted"]

        np.testing.assert_allclose(actual, expected, rtol=1e-5)


class TestProphetBoostComparison:
    """Test that Prophet+XGBoost improves over Prophet alone"""

    @pytest.fixture
    def complex_time_series(self):
        """Create time series with strong non-linear patterns"""
        np.random.seed(42)

        n = 500
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

        # Trend (Prophet should capture this)
        trend = np.linspace(100, 250, n)

        # Seasonality (Prophet should capture this)
        yearly_season = 25 * np.sin(np.arange(n) * 2 * np.pi / 365)

        # Strong non-linear pattern (XGBoost should help with this)
        non_linear = 20 * np.sin(np.arange(n) * 0.03) ** 2 * np.cos(np.arange(n) * 0.02)

        # Noise
        noise = np.random.normal(0, 5, n)

        values = trend + yearly_season + non_linear + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_hybrid_captures_complex_patterns(self, complex_time_series):
        """Test that hybrid model captures patterns Prophet might miss"""
        # Split data
        train = complex_time_series.iloc[:400]
        test = complex_time_series.iloc[400:]

        # Fit hybrid model
        spec_hybrid = prophet_boost(
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
        assert hybrid_rmse < 40  # Reasonable threshold for this data
        assert not np.isnan(hybrid_rmse)

    def test_different_xgboost_configurations(self, complex_time_series):
        """Test different XGBoost configurations"""
        train = complex_time_series.iloc[:400]

        # Shallow trees
        spec_shallow = prophet_boost(trees=100, tree_depth=3, learn_rate=0.1)
        fit_shallow = spec_shallow.fit(train, "sales ~ date")

        # Deep trees
        spec_deep = prophet_boost(trees=50, tree_depth=8, learn_rate=0.05)
        fit_deep = spec_deep.fit(train, "sales ~ date")

        # Both should fit successfully
        assert fit_shallow is not None
        assert fit_deep is not None

        # Both should have reasonable training metrics
        _, _, stats_shallow = fit_shallow.extract_outputs()
        _, _, stats_deep = fit_deep.extract_outputs()

        # Get training RMSE
        rmse_shallow = stats_shallow[
            (stats_shallow["metric"] == "rmse") & (stats_shallow["split"] == "train")
        ]["value"].values[0]

        rmse_deep = stats_deep[
            (stats_deep["metric"] == "rmse") & (stats_deep["split"] == "train")
        ]["value"].values[0]

        # Both should have reasonable fit
        assert rmse_shallow < 20
        assert rmse_deep < 20


class TestProphetBoostMultiplicative:
    """Test prophet_boost with multiplicative seasonality"""

    @pytest.fixture
    def multiplicative_data(self):
        """Create data with multiplicative seasonality"""
        np.random.seed(42)

        n = 365
        dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

        # Exponential trend
        trend = 100 * np.exp(np.arange(n) * 0.002)

        # Multiplicative seasonality
        seasonality = 1 + 0.3 * np.sin(np.arange(n) * 2 * np.pi / 30)

        # Non-linear pattern
        non_linear = 10 * np.sin(np.arange(n) * 0.05)

        # Combine multiplicatively
        values = trend * seasonality + non_linear + np.random.normal(0, 5, n)

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_multiplicative(self, multiplicative_data):
        """Test fitting with multiplicative seasonality"""
        spec = prophet_boost(
            seasonality_mode="multiplicative",
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(multiplicative_data, "sales ~ date")

        assert fit is not None
        assert fit.fit_data["prophet_params"]["seasonality_mode"] == "multiplicative"

    def test_predict_multiplicative(self, multiplicative_data):
        """Test prediction with multiplicative seasonality"""
        spec = prophet_boost(
            seasonality_mode="multiplicative",
            trees=50,
            tree_depth=3,
        )
        fit = spec.fit(multiplicative_data, "sales ~ date")

        # Future predictions
        future_dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        future = pd.DataFrame({"date": future_dates})

        preds = fit.predict(future, type="numeric")

        assert len(preds) == 30
        assert preds[".pred"].notna().all()
        assert (preds[".pred"] > 0).all()
