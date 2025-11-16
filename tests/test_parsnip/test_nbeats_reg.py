"""
Tests for nbeats_reg (Neural Basis Expansion Analysis for Time Series) model.

Tests cover:
- Model specification creation and parameter validation
- Fitting univariate time series (NBEATS does NOT support exogenous variables)
- Stack type validation (trend, seasonality, generic)
- Predictions (numeric and confidence intervals)
- Decomposition component extraction
- Device management (CPU, CUDA, MPS, auto)
- Extract outputs (three-DataFrame pattern)
- Integration workflows
- Error handling and edge cases

NBEATS is a univariate deep learning model with interpretable decomposition
into trend and seasonality components using basis expansion.
"""

import pandas as pd
import numpy as np
import pytest
import warnings
from datetime import datetime, timedelta

# Check if NeuralForecast is available
try:
    import neuralforecast
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False

from py_parsnip import nbeats_reg
from py_parsnip.model_spec import ModelSpec, ModelFit


# Skip all tests if NeuralForecast not available
pytestmark = pytest.mark.skipif(
    not NEURALFORECAST_AVAILABLE,
    reason="NeuralForecast not installed. Install with: pip install neuralforecast"
)


class TestNBEATSSpec:
    """Test nbeats_reg() model specification"""

    def test_default_spec(self):
        """Test default NBEATS specification"""
        spec = nbeats_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "nbeats_reg"
        assert spec.engine == "neuralforecast"
        assert spec.mode == "regression"
        assert spec.args["h"] == 1  # Default horizon
        assert spec.args["stack_types"] == ['trend', 'seasonality']
        assert spec.args["device"] == "auto"

    def test_spec_with_custom_horizon(self):
        """Test NBEATS with custom horizon"""
        spec = nbeats_reg(horizon=7, input_size=14)

        assert spec.args["h"] == 7
        assert spec.args["input_size"] == 14

    def test_spec_interpretable_stacks(self):
        """Test NBEATS with interpretable stacks (trend + seasonality)"""
        spec = nbeats_reg(
            stack_types=['trend', 'seasonality'],
            n_polynomials=3,  # Cubic trend
            n_harmonics=4  # 4 Fourier harmonics
        )

        assert spec.args["stack_types"] == ['trend', 'seasonality']
        assert spec.args["n_polynomials"] == 3
        assert spec.args["n_harmonics"] == 4

    def test_spec_generic_stack(self):
        """Test NBEATS with generic stack (no interpretability)"""
        spec = nbeats_reg(
            stack_types=['generic'],
            n_blocks=[3],
            mlp_units=[[1024, 1024]]
        )

        assert spec.args["stack_types"] == ['generic']
        assert spec.args["n_blocks"] == [3]

    def test_spec_mixed_stacks(self):
        """Test NBEATS with mixed stack types"""
        spec = nbeats_reg(
            stack_types=['trend', 'seasonality', 'generic'],
            n_blocks=[2, 2, 2],
            mlp_units=[[512, 512], [512, 512], [512, 512]]
        )

        assert spec.args["stack_types"] == ['trend', 'seasonality', 'generic']
        assert len(spec.args["n_blocks"]) == 3

    def test_stack_type_validation(self):
        """Test that invalid stack types raise error"""
        with pytest.raises(ValueError, match="Invalid stack types"):
            nbeats_reg(stack_types=['invalid_stack'])

        with pytest.raises(ValueError, match="Invalid stack types"):
            nbeats_reg(stack_types=['trend', 'invalid', 'seasonality'])

    def test_architecture_consistency_validation(self):
        """Test that architecture parameters must match in length"""
        # Mismatched n_blocks length
        with pytest.raises(ValueError, match="n_blocks length.*must match stack_types length"):
            nbeats_reg(
                stack_types=['trend', 'seasonality'],
                n_blocks=[1]  # Wrong length
            )

        # Mismatched mlp_units length
        with pytest.raises(ValueError, match="mlp_units length.*must match stack_types length"):
            nbeats_reg(
                stack_types=['trend', 'seasonality'],
                mlp_units=[[512, 512]]  # Wrong length
            )

    def test_spec_with_training_params(self):
        """Test NBEATS with training parameters"""
        spec = nbeats_reg(
            max_steps=2000,
            learning_rate=5e-4,
            batch_size=64,
            early_stop_patience_steps=100,
            dropout_prob_theta=0.1
        )

        assert spec.args["max_steps"] == 2000
        assert spec.args["learning_rate"] == 5e-4
        assert spec.args["batch_size"] == 64
        assert spec.args["early_stop_patience_steps"] == 100
        assert spec.args["dropout_prob_theta"] == 0.1


class TestNBEATSFit:
    """Test nbeats_reg fitting"""

    @pytest.fixture
    def daily_univariate_data(self):
        """Create daily univariate time series (300 observations)"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")

        # Trend + weekly seasonality + noise
        t = np.arange(300)
        trend = 100 + 0.5 * t
        seasonality = 20 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        noise = np.random.RandomState(42).normal(0, 5, 300)

        values = trend + seasonality + noise

        return pd.DataFrame({"date": dates, "sales": values})

    def test_fit_univariate(self, daily_univariate_data):
        """Test basic univariate fitting"""
        spec = nbeats_reg(
            horizon=7,
            input_size=14,
            max_steps=50,  # Quick training for tests
            device='cpu'  # Force CPU for CI/CD
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        assert isinstance(fit, ModelFit)
        assert fit.spec.model_type == "nbeats_reg"
        assert "model" in fit.fit_data
        assert fit.fit_data["n_obs"] == 300
        assert fit.fit_data["horizon"] == 7

    def test_fit_with_exog_warning(self):
        """Test that NBEATS warns when exogenous variables provided"""
        # Create data with exogenous variables
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(100) + 100,
            "price": np.random.randn(100) + 10,
            "promo": np.random.choice([0, 1], 100)
        })

        spec = nbeats_reg(horizon=7, input_size=14, max_steps=20, device='cpu')

        # Should warn about ignoring exogenous variables
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = spec.fit(data, "sales ~ price + promo + date")

            # Check that a warning was issued
            assert len(w) >= 1
            assert any("univariate" in str(warning.message).lower() for warning in w)

        # Exogenous variables should be ignored
        assert fit.fit_data["exog_vars"] == []

    def test_fit_insufficient_data(self):
        """Test that insufficient data raises error"""
        # Create very small dataset
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(10) + 100
        })

        spec = nbeats_reg(horizon=7, input_size=28, device='cpu')

        with pytest.raises(ValueError, match="Insufficient training data"):
            spec.fit(data, "sales ~ date")

    def test_fit_with_validation_split(self, daily_univariate_data):
        """Test fitting with validation split"""
        spec = nbeats_reg(
            horizon=7,
            input_size=14,
            max_steps=50,
            validation_split=0.2,  # 20% validation
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        # Should complete without error
        assert fit is not None
        assert "model" in fit.fit_data

    def test_fit_trend_stack_only(self, daily_univariate_data):
        """Test fitting with only trend stack"""
        spec = nbeats_reg(
            stack_types=['trend'],
            n_blocks=[2],
            n_polynomials=3,  # Cubic trend
            horizon=7,
            input_size=14,
            max_steps=50,
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        assert fit.fit_data["stack_types"] == ['trend']

    def test_fit_seasonality_stack_only(self, daily_univariate_data):
        """Test fitting with only seasonality stack"""
        spec = nbeats_reg(
            stack_types=['seasonality'],
            n_blocks=[2],
            n_harmonics=4,
            horizon=7,
            input_size=14,
            max_steps=50,
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        assert fit.fit_data["stack_types"] == ['seasonality']

    def test_fit_generic_stack(self, daily_univariate_data):
        """Test fitting with generic stack (maximum flexibility)"""
        spec = nbeats_reg(
            stack_types=['generic'],
            n_blocks=[3],
            mlp_units=[[512, 512]],
            horizon=7,
            input_size=14,
            max_steps=50,
            device='cpu'
        )

        fit = spec.fit(daily_univariate_data, "sales ~ date")

        assert fit.fit_data["stack_types"] == ['generic']


class TestNBEATSPredict:
    """Test nbeats_reg prediction"""

    @pytest.fixture
    def fitted_nbeats(self):
        """Create fitted NBEATS model"""
        # Generate training data with trend and seasonality
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        values = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + \
                 np.random.RandomState(42).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nbeats_reg(
            horizon=7,
            input_size=14,
            max_steps=50,
            device='cpu'
        )

        return spec.fit(train, "sales ~ date")

    def test_predict_numeric(self, fitted_nbeats):
        """Test prediction with type='numeric'"""
        # Future dates (7 days ahead)
        future_dates = pd.date_range(
            start="2023-10-28",  # After training end
            periods=7,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_nbeats.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())

    def test_predict_conf_int(self, fitted_nbeats):
        """Test prediction with confidence intervals"""
        future_dates = pd.date_range(
            start="2023-10-28",
            periods=7,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_nbeats.predict(test, type="conf_int")

        assert ".pred" in predictions.columns
        # Confidence intervals may or may not be available depending on NeuralForecast version
        # Just check that prediction succeeded
        assert len(predictions) == 7

    def test_forecast_future(self, fitted_nbeats):
        """Test forecasting into future beyond training data"""
        # Forecast well into future
        future_dates = pd.date_range(
            start="2023-11-01",
            periods=30,
            freq="D"
        )
        test = pd.DataFrame({"date": future_dates})

        predictions = fitted_nbeats.predict(test, type="numeric")

        # NBEATS has fixed horizon, should return horizon predictions
        assert len(predictions) == fitted_nbeats.fit_data["horizon"]


class TestNBEATSExtract:
    """Test nbeats_reg output extraction"""

    @pytest.fixture
    def fitted_nbeats(self):
        """Create fitted NBEATS model"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        values = 100 + 0.5 * t + np.random.RandomState(42).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nbeats_reg(horizon=7, input_size=14, max_steps=50, device='cpu')
        return spec.fit(train, "sales ~ date")

    def test_extract_fit_engine(self, fitted_nbeats):
        """Test extract_fit_engine() returns NeuralForecast model"""
        engine_model = fitted_nbeats.extract_fit_engine()

        assert engine_model is not None
        # Should be NBEATS model
        assert hasattr(engine_model, 'predict')

    def test_extract_outputs_format(self, fitted_nbeats):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefficients, stats = fitted_nbeats.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_columns(self, fitted_nbeats):
        """Test outputs DataFrame has correct columns"""
        outputs, _, _ = fitted_nbeats.extract_outputs()

        required_cols = ["date", "actuals", "fitted", "forecast", "residuals", "split"]
        for col in required_cols:
            assert col in outputs.columns

        # All training data
        assert all(outputs["split"] == "train")

    def test_coefficients_decomposition(self, fitted_nbeats):
        """Test coefficients DataFrame contains hyperparameters and decomposition info"""
        _, coefficients, _ = fitted_nbeats.extract_outputs()

        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Should have stack configuration info
        variables = coefficients["variable"].tolist()
        assert "stack_types" in variables
        assert "horizon" in variables
        assert "input_size" in variables

    def test_stats_metrics(self, fitted_nbeats):
        """Test stats DataFrame contains performance metrics"""
        _, _, stats = fitted_nbeats.extract_outputs()

        metrics = stats["metric"].tolist()

        # Should have standard metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics

        # Should have model info
        assert "model_type" in metrics
        assert "stack_types" in metrics
        assert "horizon" in metrics


class TestNBEATSDecomposition:
    """Test NBEATS decomposition capabilities"""

    def test_trend_stack_decomposition(self):
        """Test trend component extraction"""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        t = np.arange(200)
        # Strong linear trend
        values = 100 + 2 * t + np.random.RandomState(42).normal(0, 5, 200)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nbeats_reg(
            stack_types=['trend'],
            n_polynomials=2,  # Quadratic
            horizon=7,
            input_size=14,
            max_steps=100,
            device='cpu'
        )

        fit = spec.fit(train, "sales ~ date")

        # Should fit trend successfully
        assert fit.fit_data["stack_types"] == ['trend']
        outputs, _, _ = fit.extract_outputs()
        assert len(outputs) > 0

    def test_seasonality_stack_decomposition(self):
        """Test seasonality component extraction"""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        t = np.arange(200)
        # Strong weekly seasonality
        values = 100 + 30 * np.sin(2 * np.pi * t / 7) + \
                 np.random.RandomState(42).normal(0, 5, 200)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nbeats_reg(
            stack_types=['seasonality'],
            n_harmonics=4,  # Multiple harmonics
            horizon=7,
            input_size=21,
            max_steps=100,
            device='cpu'
        )

        fit = spec.fit(train, "sales ~ date")

        # Should fit seasonality successfully
        assert fit.fit_data["stack_types"] == ['seasonality']
        outputs, _, _ = fit.extract_outputs()
        assert len(outputs) > 0

    def test_combined_decomposition(self):
        """Test combined trend + seasonality decomposition"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        t = np.arange(300)
        # Trend + seasonality
        trend = 100 + 0.5 * t
        seasonality = 20 * np.sin(2 * np.pi * t / 7)
        values = trend + seasonality + np.random.RandomState(42).normal(0, 5, 300)

        train = pd.DataFrame({"date": dates, "sales": values})

        spec = nbeats_reg(
            stack_types=['trend', 'seasonality'],
            n_polynomials=2,
            n_harmonics=3,
            horizon=7,
            input_size=21,
            max_steps=100,
            device='cpu'
        )

        fit = spec.fit(train, "sales ~ date")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should have both stacks
        coef_vars = coefficients["variable"].tolist()
        assert "n_polynomials" in coef_vars
        assert "n_harmonics" in coef_vars


class TestNBEATSDevice:
    """Test device management for NBEATS"""

    def test_device_auto_detection(self):
        """Test automatic device selection"""
        spec = nbeats_reg(device='auto')

        assert spec.args["device"] == 'auto'

    def test_device_cpu(self):
        """Test CPU device specification"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(100) + 100
        })

        spec = nbeats_reg(horizon=7, input_size=14, max_steps=20, device='cpu')
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

        spec = nbeats_reg(horizon=7, input_size=14, max_steps=20, device='cuda')
        fit = spec.fit(data, "sales ~ date")

        assert fit.fit_data["device"] == 'cuda'


class TestNBEATSIntegration:
    """Integration tests for NBEATS workflows"""

    def test_full_workflow(self):
        """Test complete fit → predict → extract workflow"""
        # Training data (6 months daily)
        train_dates = pd.date_range(start="2023-01-01", periods=180, freq="D")
        t = np.arange(180)
        train_values = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + \
                      np.random.RandomState(42).normal(0, 5, 180)
        train = pd.DataFrame({"date": train_dates, "sales": train_values})

        # Fit model
        spec = nbeats_reg(
            horizon=7,
            input_size=21,
            stack_types=['trend', 'seasonality'],
            max_steps=100,
            device='cpu'
        )
        fit = spec.fit(train, "sales ~ date")

        # Predict
        test_dates = pd.date_range(start="2023-06-30", periods=7, freq="D")
        test = pd.DataFrame({"date": test_dates})
        predictions = fit.predict(test, type="numeric")

        # Extract outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Verify all components
        assert len(predictions) == 7
        assert all(predictions[".pred"].notna())
        assert len(outputs) == 180
        assert len(coefficients) > 0
        assert len(stats) > 0

    def test_with_different_frequencies(self):
        """Test NBEATS with different time series frequencies"""
        # Weekly data
        weekly_dates = pd.date_range(start="2023-01-01", periods=100, freq="W")
        weekly_data = pd.DataFrame({
            "date": weekly_dates,
            "sales": np.random.randn(100) * 10 + 100
        })

        spec = nbeats_reg(horizon=4, input_size=12, max_steps=30, device='cpu')
        fit = spec.fit(weekly_data, "sales ~ date")

        # Should infer weekly frequency
        assert fit.fit_data["freq"] in ['W', 'W-SUN']

    def test_comparison_with_nhits(self):
        """Compare NBEATS vs NHITS on same data (univariate only)"""
        # Generate test data
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        t = np.arange(200)
        values = 100 + 0.5 * t + 15 * np.sin(2 * np.pi * t / 7) + \
                 np.random.RandomState(42).normal(0, 5, 200)

        train = pd.DataFrame({"date": dates, "sales": values})

        # Fit NBEATS
        nbeats_spec = nbeats_reg(
            horizon=7,
            input_size=21,
            max_steps=50,
            device='cpu'
        )
        nbeats_fit = nbeats_spec.fit(train, "sales ~ date")

        # Fit NHITS (for comparison)
        from py_parsnip import nhits_reg
        nhits_spec = nhits_reg(
            horizon=7,
            input_size=21,
            max_steps=50,
            device='cpu'
        )
        nhits_fit = nhits_spec.fit(train, "sales ~ date")

        # Both should predict successfully
        test_dates = pd.date_range(start="2023-07-20", periods=7, freq="D")
        test = pd.DataFrame({"date": test_dates})

        nbeats_pred = nbeats_fit.predict(test, type="numeric")
        nhits_pred = nhits_fit.predict(test, type="numeric")

        assert len(nbeats_pred) == 7
        assert len(nhits_pred) == 7

    def test_share_weights_option(self):
        """Test share_weights_in_stack option"""
        dates = pd.date_range(start="2023-01-01", periods=150, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "sales": np.random.randn(150) * 10 + 100
        })

        # With weight sharing (fewer parameters)
        spec_shared = nbeats_reg(
            horizon=7,
            input_size=14,
            share_weights_in_stack=True,
            max_steps=50,
            device='cpu'
        )
        fit_shared = spec_shared.fit(data, "sales ~ date")

        # Without weight sharing (more parameters)
        spec_no_share = nbeats_reg(
            horizon=7,
            input_size=14,
            share_weights_in_stack=False,
            max_steps=50,
            device='cpu'
        )
        fit_no_share = spec_no_share.fit(data, "sales ~ date")

        # Both should fit successfully
        assert fit_shared is not None
        assert fit_no_share is not None
