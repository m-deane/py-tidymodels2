"""
Tests for nhits_reg (Neural Hierarchical Interpolation for Time Series) model.

Tests cover:
- Model specification creation and parameter validation
- Fitting with univariate and multivariate data
- Predictions (numeric and confidence intervals)
- Device management (CPU, CUDA, MPS, auto)
- Extract outputs (three-DataFrame pattern)
- Integration workflows
- Error handling and edge cases

NHITS is a deep learning model that uses hierarchical interpolation
for multi-scale temporal pattern capture. It supports exogenous variables.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Check if NeuralForecast is available
try:
    import neuralforecast
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False

from py_parsnip import nhits_reg
from py_parsnip.model_spec import ModelSpec, ModelFit


# Skip all tests if NeuralForecast not available
pytestmark = pytest.mark.skipif(
    not NEURALFORECAST_AVAILABLE,
    reason="NeuralForecast not installed. Install with: pip install neuralforecast"
)


class TestNHITSSpec:
    """Test nhits_reg() model specification"""

    def test_default_spec(self):
        """Test default NHITS specification"""
        spec = nhits_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "nhits_reg"
        assert spec.engine == "neuralforecast"
        assert spec.mode == "regression"
        assert spec.args["h"] == 1  # Default horizon
        assert spec.args["input_size"] is None  # Auto-calculated
        assert spec.args["device"] == "auto"

    def test_spec_with_custom_horizon(self):
        """Test NHITS with custom horizon"""
        spec = nhits_reg(horizon=7, input_size=28)

        assert spec.args["h"] == 7
        assert spec.args["input_size"] == 28

    def test_spec_with_architecture_params(self):
        """Test NHITS with custom architecture"""
        spec = nhits_reg(
            horizon=7,
            n_freq_downsample=[16, 8, 4, 1],
            n_blocks=[2, 2, 2, 2],
            mlp_units=[[1024, 512], [512, 512], [512, 512], [512, 512]]
        )

        assert spec.args["n_freq_downsample"] == [16, 8, 4, 1]
        assert spec.args["n_blocks"] == [2, 2, 2, 2]
        assert len(spec.args["mlp_units"]) == 4

    def test_spec_with_training_params(self):
        """Test NHITS with training parameters"""
        spec = nhits_reg(
            max_steps=2000,
            learning_rate=5e-4,
            batch_size=64,
            early_stop_patience_steps=100
        )

        assert spec.args["max_steps"] == 2000
        assert spec.args["learning_rate"] == 5e-4
        assert spec.args["batch_size"] == 64
        assert spec.args["early_stop_patience_steps"] == 100

    def test_architecture_consistency_validation(self):
        """Test that architecture parameters must match in length"""
        # Mismatched n_blocks length
        with pytest.raises(ValueError, match="n_blocks length"):
            nhits_reg(
                n_freq_downsample=[8, 4, 1],
                n_blocks=[1, 1]  # Wrong length
            )

        # Mismatched mlp_units length
        with pytest.raises(ValueError, match="mlp_units length"):
            nhits_reg(
                n_freq_downsample=[8, 4, 1],
                mlp_units=[[512, 512], [512, 512]]  # Wrong length
            )

    def test_invalid_horizon(self):
        """Test that invalid horizon raises error"""
        with pytest.raises(ValueError, match="horizon must be positive"):
            nhits_reg(horizon=0)

        with pytest.raises(ValueError, match="horizon must be positive"):
            nhits_reg(horizon=-1)

    def test_invalid_validation_split(self):
        """Test that invalid validation_split raises error"""
        with pytest.raises(ValueError, match="validation_split must be between 0 and 1"):
            nhits_reg(validation_split=1.5)

        with pytest.raises(ValueError, match="validation_split must be between 0 and 1"):
            nhits_reg(validation_split=0.0)


class TestNHITSFit:
    """Test nhits_reg fitting"""

    @pytest.fixture
    def daily_univariate_data(self):
        """Create daily univariate time series (300 observations)"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")

        # Trend + seasonality + noise
        t = np.arange(300)
        trend = 100 + 0.5 * t
        seasonality = 20 * np.sin(2 * np.pi * t / 7)  # Weekly
        noise = np.random.RandomState(42).normal(0, 5, 300)

        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    @pytest.fixture
    def daily_multivariate_data(self):
        """Create daily multivariate time series with exogenous variables"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")

        # Generate features
        t = np.arange(300)
        price = 10 + np.random.RandomState(42).normal(0, 1, 300)
        promo = np.random.RandomState(43).choice([0, 1], 300, p=[0.7, 0.3])

        # Sales depends on price and promo
        trend = 100 + 0.5 * t
        price_effect = -2 * price
        promo_effect = 15 * promo
        seasonality = 20 * np.sin(2 * np.pi * t / 7)
        noise = np.random.RandomState(44).normal(0, 5, 300)

        sales = trend + price_effect + promo_effect + seasonality + noise

        return pd.DataFrame({
            "date": dates,
            "sales": sales,
            "price": price,
            "promo": promo
        })

    def test_fit_univariate(self, daily_univariate_data):
        """Test basic univariate fitting"""
        spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=50,  # Quick training for tests
            device='cpu'  # Force CPU for CI/CD
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        assert isinstance(fit, ModelFit)
        assert fit.spec.model_type == "nhits_reg"
        assert "model" in fit.fit_data
        assert fit.fit_data["n_obs"] == 300
        assert fit.fit_data["horizon"] == 7

    def test_fit_with_exogenous_variables(self, daily_multivariate_data):
        """Test fitting with exogenous variables"""
        spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=50,
            device='cpu'
        )

        fit = spec.fit(daily_multivariate_data, "sales ~ price + promo + date")

        assert fit.fit_data["exog_vars"] == ["price", "promo"]
        assert "model" in fit.fit_data

    def test_fit_insufficient_data(self):
        """Test that insufficient data raises error"""
        # Create very small dataset
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(10)
        })

        spec = nhits_reg(horizon=7, input_size=28, device='cpu')

        with pytest.raises(ValueError, match="Insufficient data"):
            spec.fit(data, "sales ~ date")

    def test_fit_with_validation_split(self, daily_univariate_data):
        """Test fitting with validation split"""
        spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=50,
            validation_split=0.2,  # 20% validation
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        # Should complete without error
        assert fit is not None
        assert "model" in fit.fit_data

    def test_fit_with_early_stopping(self, daily_univariate_data):
        """Test fitting with early stopping enabled"""
        spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=500,  # High max_steps
            early_stop_patience_steps=20,  # Should stop early
            validation_split=0.2,
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        # Should complete (possibly early stopped)
        assert fit is not None


class TestNHITSPredict:
    """Test nhits_reg prediction"""

    @pytest.fixture
    def fitted_nhits(self):
        """Create fitted NHITS model"""
        # Generate training data
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        values = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + \
                 np.random.RandomState(42).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=50,
            device='cpu'
        )

        return spec.fit(train, "sales ~ date")

    def test_predict_numeric(self, fitted_nhits):
        """Test prediction with type='numeric'"""
        # Future dates (7 days ahead)
        future_dates = pd.date_range(
            start="2023-10-28",  # After training end
            periods=7,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_nhits.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())

    def test_predict_conf_int(self, fitted_nhits):
        """Test prediction with confidence intervals"""
        future_dates = pd.date_range(
            start="2023-10-28",
            periods=7,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_nhits.predict(test, type="conf_int")

        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert len(predictions) == 7

        # Check intervals are reasonable
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])

    def test_predict_with_exogenous(self):
        """Test prediction with exogenous variables"""
        # Training data with exogenous
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        price = 10 + np.random.RandomState(42).normal(0, 1, 300)
        sales = 100 + 0.5 * t - 2 * price + np.random.RandomState(43).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": sales, "price": price})

        spec = nhits_reg(horizon=7, input_size=21, max_steps=50, device='cpu')
        fit = spec.fit(train, "sales ~ price + date")

        # Test data with exogenous
        future_dates = pd.date_range(start="2023-10-28", periods=7, freq="D")
        future_price = 10 + np.random.RandomState(44).normal(0, 1, 7)
        test = pd.DataFrame({"date": future_dates, "price": future_price})

        predictions = fit.predict(test, type="numeric")

        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())

    def test_forecast_beyond_training_dates(self, fitted_nhits):
        """Test forecasting into future beyond training data"""
        # Forecast 30 days into future
        future_dates = pd.date_range(
            start="2023-11-01",  # Well beyond training
            periods=30,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        # NHITS forecast horizon is fixed at fit time
        # This should use the last 7 days as forecast
        predictions = fitted_nhits.predict(test, type="numeric")

        # Should return horizon predictions (7 days)
        assert len(predictions) == fitted_nhits.fit_data["horizon"]


class TestNHITSExtract:
    """Test nhits_reg output extraction"""

    @pytest.fixture
    def fitted_nhits(self):
        """Create fitted NHITS model"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        values = 100 + 0.5 * t + np.random.RandomState(42).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nhits_reg(horizon=7, input_size=21, max_steps=50, device='cpu')
        return spec.fit(train, "sales ~ date")

    def test_extract_fit_engine(self, fitted_nhits):
        """Test extract_fit_engine() returns NeuralForecast model"""
        engine_model = fitted_nhits.extract_fit_engine()

        assert engine_model is not None
        # Should be NHITS model
        assert hasattr(engine_model, 'predict')

    def test_extract_outputs_format(self, fitted_nhits):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefficients, stats = fitted_nhits.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_columns(self, fitted_nhits):
        """Test outputs DataFrame has correct columns"""
        outputs, _, _ = fitted_nhits.extract_outputs()

        required_cols = ["actuals", "fitted", "forecast", "residuals", "dates", "split", "model"]
        for col in required_cols:
            assert col in outputs.columns

        # All training data
        assert all(outputs["split"] == "train")
        assert len(outputs) == 300

    def test_coefficients_hyperparameters(self, fitted_nhits):
        """Test coefficients DataFrame contains hyperparameters"""
        _, coefficients, _ = fitted_nhits.extract_outputs()

        assert "term" in coefficients.columns
        assert "estimate" in coefficients.columns

        # Should have hyperparameters as "coefficients"
        terms = coefficients["term"].tolist()
        assert "horizon" in terms
        assert "input_size" in terms
        assert "learning_rate" in terms

    def test_stats_metrics(self, fitted_nhits):
        """Test stats DataFrame contains performance metrics"""
        _, _, stats = fitted_nhits.extract_outputs()

        assert "rmse" in stats.columns
        assert "mae" in stats.columns
        assert "r_squared" in stats.columns
        assert "train_time" in stats.columns
        assert "device" in stats.columns

        # Check split column
        assert stats["split"].iloc[0] == "train"


class TestNHITSDevice:
    """Test device management for NHITS"""

    def test_device_auto_detection(self):
        """Test automatic device selection"""
        spec = nhits_reg(device='auto')

        assert spec.args["device"] == 'auto'
        # Device will be selected during fit

    def test_device_cpu(self):
        """Test CPU device specification"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(100) + 100
        })

        spec = nhits_reg(horizon=7, input_size=21, max_steps=20, device='cpu')
        fit = spec.fit(data, "sales ~ date")

        assert fit.fit_data["device"] == 'cpu'

    @pytest.mark.gpu
    def test_device_cuda(self):
        """Test CUDA device (requires GPU)"""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(100) + 100
        })

        spec = nhits_reg(horizon=7, input_size=21, max_steps=20, device='cuda')
        fit = spec.fit(data, "sales ~ date")

        assert fit.fit_data["device"] == 'cuda'

    def test_device_validation(self):
        """Test that invalid device is handled gracefully"""
        # Invalid device string should fall back to CPU
        # (validated in engine, not spec)
        spec = nhits_reg(device='invalid_device')

        # Spec accepts any string, validation happens at fit time
        assert spec.args["device"] == 'invalid_device'


class TestNHITSIntegration:
    """Integration tests for NHITS workflows"""

    def test_full_workflow(self):
        """Test complete fit → predict → extract workflow"""
        # Training data (6 months daily)
        train_dates = pd.date_range(start="2023-01-01", periods=180, freq="D")
        t = np.arange(180)
        train_values = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + \
                      np.random.RandomState(42).normal(0, 5, 180)
        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model
        spec = nhits_reg(
            horizon=7,
            input_size=28,
            max_steps=100,
            learning_rate=1e-3,
            device='cpu'
        )
        fit = spec.fit(train, "sales ~ date")

        # Predict
        test_dates = pd.date_range(start="2023-06-30", periods=7, freq="D")
        test = pd.DataFrame({"date": test_dates})
        predictions = fit.predict(test, type="conf_int")

        # Extract outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Verify all components
        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())
        assert len(outputs) == 180
        assert len(coefficients) > 0
        assert len(stats) > 0

    def test_multivariate_workflow(self):
        """Test workflow with exogenous variables"""
        # Generate data with exogenous effects
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        t = np.arange(200)
        price = 10 + np.random.RandomState(42).normal(0, 1, 200)
        promo = np.random.RandomState(43).choice([0, 1], 200, p=[0.7, 0.3])

        sales = 100 + 0.5 * t - 2 * price + 15 * promo + \
                np.random.RandomState(44).normal(0, 5, 200)

        train = pd.DataFrame({
            "date": dates,
            "sales": sales,
            "price": price,
            "promo": promo
        })

        # Fit with exogenous
        spec = nhits_reg(horizon=7, input_size=21, max_steps=50, device='cpu')
        fit = spec.fit(train, "sales ~ price + promo + date")

        # Predict with exogenous
        test_dates = pd.date_range(start="2023-07-20", periods=7, freq="D")
        test_price = 10 + np.random.RandomState(45).normal(0, 1, 7)
        test_promo = np.random.RandomState(46).choice([0, 1], 7)
        test = pd.DataFrame({
            "date": test_dates,
            "price": test_price,
            "promo": test_promo
        })

        predictions = fit.predict(test, type="numeric")

        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())

    def test_with_different_frequencies(self):
        """Test NHITS with different time series frequencies"""
        # Weekly data
        weekly_dates = pd.date_range(start="2023-01-01", periods=100, freq="W")
        weekly_data = pd.DataFrame({
            "date": weekly_dates,
            "sales": np.random.randn(100) * 10 + 100
        })

        spec = nhits_reg(horizon=4, input_size=12, max_steps=30, device='cpu')
        fit = spec.fit(weekly_data, "sales ~ date")

        # Should infer weekly frequency
        assert fit.fit_data["freq"] in ['W', 'W-SUN']

    def test_long_horizon_forecasting(self):
        """Test NHITS with long forecast horizon"""
        dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
        t = np.arange(500)
        values = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + \
                 np.random.RandomState(42).normal(0, 5, 500)

        train = pd.DataFrame({"date": dates, "sales": values})

        # 28-day horizon
        spec = nhits_reg(
            horizon=28,
            input_size=84,  # 3x horizon
            max_steps=100,
            device='cpu'
        )
        fit = spec.fit(train, "sales ~ date")

        # Predict 28 days ahead
        test_dates = pd.date_range(start="2024-05-15", periods=28, freq="D")
        test = pd.DataFrame({"date": test_dates})
        predictions = fit.predict(test, type="numeric")

        assert len(predictions) == 28
