"""
Tests for svm_linear model specification and sklearn engine

Tests cover:
- Model specification creation
- Parameter translation (cost, margin)
- Fitting with formula
- Prediction
- Extract outputs (linear coefficients, stats)
- Evaluate() method
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import svm_linear, ModelSpec, ModelFit


class TestSVMLinearSpec:
    """Test svm_linear() model specification"""

    def test_default_spec(self):
        """Test default svm_linear specification"""
        spec = svm_linear()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "svm_linear"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_cost(self):
        """Test svm_linear with cost parameter"""
        spec = svm_linear(cost=10.0)

        assert spec.args == {"cost": 10.0}

    def test_spec_with_margin(self):
        """Test svm_linear with margin parameter"""
        spec = svm_linear(margin=0.1)

        assert spec.args == {"margin": 0.1}

    def test_spec_with_all_parameters(self):
        """Test svm_linear with all parameters"""
        spec = svm_linear(cost=5.0, margin=0.05)

        assert spec.args == {"cost": 5.0, "margin": 0.05}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = svm_linear(cost=1.0)
        spec2 = spec1.set_args(cost=10.0)

        assert spec1.args == {"cost": 1.0}
        assert spec2.args == {"cost": 10.0}


class TestSVMLinearFit:
    """Test svm_linear fitting"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = svm_linear()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data

    def test_fit_model_class(self, train_data):
        """Test correct sklearn model class"""
        spec = svm_linear()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "LinearSVR"

    def test_fit_with_cost(self, train_data):
        """Test fitting with cost parameter"""
        spec = svm_linear(cost=10.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.C == 10.0

    def test_fit_with_margin(self, train_data):
        """Test fitting with margin parameter"""
        spec = svm_linear(margin=0.1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.epsilon == 0.1

    def test_fit_has_coefficients(self, train_data):
        """Test that linear SVM has coefficients"""
        spec = svm_linear()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert hasattr(model, "coef_")
        assert len(model.coef_) == 2  # Two features


class TestSVMLinearPredict:
    """Test svm_linear prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_linear()
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_predict_basic(self, fitted_model):
        """Test basic prediction"""
        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fitted_model.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_predict_values_reasonable(self, fitted_model):
        """Test that predictions are in reasonable range"""
        test = pd.DataFrame({
            "x1": [15],
            "x2": [7],
        })

        predictions = fitted_model.predict(test)

        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_predict_invalid_type(self, fitted_model):
        """Test prediction with invalid type raises error"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model.predict(test, type="prob")


class TestSVMLinearExtractOutputs:
    """Test svm_linear output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_linear()
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_extract_outputs_returns_three_dataframes(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_model.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_model_outputs(self, fitted_model):
        """Test Outputs DataFrame structure"""
        outputs, _, _ = fitted_model.extract_outputs()

        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert len(outputs) == 10

    def test_extract_outputs_coefficients(self, fitted_model):
        """Test Coefficients DataFrame has linear coefficients"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Linear SVM has coefficients
        assert len(coefs) == 2  # Two features
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame structure"""
        _, _, stats = fitted_model.extract_outputs()

        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "r_squared" in metric_names
        assert "C" in metric_names


class TestSVMLinearParameterTranslation:
    """Test parameter translation"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

    def test_param_translation_cost(self, train_data):
        """Test cost parameter maps to C"""
        spec = svm_linear(cost=5.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.C == 5.0

    def test_param_translation_margin(self, train_data):
        """Test margin parameter maps to epsilon"""
        spec = svm_linear(margin=0.1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.epsilon == 0.1


class TestSVMLinearIntegration:
    """Integration tests"""

    def test_full_workflow(self):
        """Test complete workflow"""
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_linear(cost=5.0, margin=0.05)
        fit = spec.fit(train, "sales ~ price + advertising")

        test = pd.DataFrame({
            "price": [12, 22, 28],
            "advertising": [6, 11, 14],
        })

        predictions = fit.predict(test)

        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()
        assert len(outputs) == 10
        assert len(coefs) == 2  # Linear coefficients
        assert len(stats) > 0
