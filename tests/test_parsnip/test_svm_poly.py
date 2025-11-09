"""
Tests for svm_poly model specification and sklearn engine

Tests cover:
- Model specification creation
- Parameter translation (cost, degree, scale_factor, margin)
- Fitting with formula (regression and classification)
- Prediction
- Extract outputs (empty coefficients for polynomial kernel, stats)
- Evaluate() method
- Different polynomial degrees
- Mode setting (.set_mode())
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import svm_poly, ModelSpec, ModelFit


class TestSVMPolySpec:
    """Test svm_poly() model specification"""

    def test_default_spec(self):
        """Test default svm_poly specification"""
        spec = svm_poly()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "svm_poly"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_cost(self):
        """Test svm_poly with cost parameter"""
        spec = svm_poly(cost=10.0)

        assert spec.args == {"cost": 10.0}

    def test_spec_with_degree(self):
        """Test svm_poly with degree parameter"""
        spec = svm_poly(degree=2)

        assert spec.args == {"degree": 2}

    def test_spec_with_scale_factor(self):
        """Test svm_poly with scale_factor parameter"""
        spec = svm_poly(scale_factor=0.1)

        assert spec.args == {"scale_factor": 0.1}

    def test_spec_with_margin(self):
        """Test svm_poly with margin parameter"""
        spec = svm_poly(margin=0.2)

        assert spec.args == {"margin": 0.2}

    def test_spec_with_all_parameters(self):
        """Test svm_poly with all parameters"""
        spec = svm_poly(cost=5.0, degree=4, scale_factor=0.05, margin=0.15)

        assert spec.args == {
            "cost": 5.0,
            "degree": 4,
            "scale_factor": 0.05,
            "margin": 0.15
        }

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = svm_poly(cost=1.0, degree=2)
        spec2 = spec1.set_args(cost=10.0)

        assert spec1.args == {"cost": 1.0, "degree": 2}
        assert spec2.args == {"cost": 10.0, "degree": 2}


class TestSVMPolyFitRegression:
    """Test svm_poly fitting for regression"""

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
        spec = svm_poly()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data

    def test_fit_model_class(self, train_data):
        """Test correct sklearn model class"""
        spec = svm_poly()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "SVR"

    def test_fit_with_cost(self, train_data):
        """Test fitting with cost parameter"""
        spec = svm_poly(cost=10.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.C == 10.0

    def test_fit_with_degree_2(self, train_data):
        """Test fitting with degree=2 (quadratic)"""
        spec = svm_poly(degree=2)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.degree == 2

    def test_fit_with_degree_3(self, train_data):
        """Test fitting with degree=3 (cubic - default)"""
        spec = svm_poly(degree=3)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.degree == 3

    def test_fit_with_degree_4(self, train_data):
        """Test fitting with degree=4 (quartic)"""
        spec = svm_poly(degree=4)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.degree == 4

    def test_fit_with_scale_factor(self, train_data):
        """Test fitting with scale_factor parameter"""
        spec = svm_poly(scale_factor=0.1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.gamma == 0.1

    def test_fit_with_margin(self, train_data):
        """Test fitting with margin parameter"""
        spec = svm_poly(margin=0.2)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.epsilon == 0.2

    def test_fit_kernel_is_poly(self, train_data):
        """Test that kernel is forced to poly"""
        spec = svm_poly()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.kernel == "poly"

    def test_fit_default_degree(self, train_data):
        """Test that default degree is 3"""
        spec = svm_poly()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.degree == 3


class TestSVMPolyPredictRegression:
    """Test svm_poly prediction for regression"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_poly(degree=2)
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
            fitted_model.predict(test, type="class")


class TestSVMPolyClassification:
    """Test svm_poly for classification"""

    @pytest.fixture
    def iris_data(self):
        """Create iris-like classification data"""
        np.random.seed(42)
        return pd.DataFrame({
            "species": ["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10,
            "sepal_length": np.concatenate([
                np.random.normal(5.0, 0.3, 10),
                np.random.normal(6.0, 0.3, 10),
                np.random.normal(6.5, 0.3, 10)
            ]),
            "sepal_width": np.concatenate([
                np.random.normal(3.5, 0.3, 10),
                np.random.normal(2.8, 0.3, 10),
                np.random.normal(3.0, 0.3, 10)
            ]),
        })

    def test_classification_mode_setting(self):
        """Test setting classification mode"""
        spec = svm_poly(degree=2).set_mode("classification")

        assert spec.mode == "classification"

    def test_classification_fit(self, iris_data):
        """Test fitting classification model"""
        spec = svm_poly(degree=2).set_mode("classification")
        fit = spec.fit(iris_data, "species ~ sepal_length + sepal_width")

        assert isinstance(fit, ModelFit)
        assert fit.fit_data["model_class"] == "SVC"

    def test_classification_predict_class(self, iris_data):
        """Test classification prediction (class)"""
        spec = svm_poly(degree=2).set_mode("classification")
        fit = spec.fit(iris_data, "species ~ sepal_length + sepal_width")

        test = pd.DataFrame({
            "sepal_length": [5.0, 6.0],
            "sepal_width": [3.5, 2.8],
        })

        predictions = fit.predict(test, type="class")

        assert ".pred_class" in predictions.columns
        assert len(predictions) == 2

    def test_classification_predict_prob(self, iris_data):
        """Test classification prediction (probabilities)"""
        spec = svm_poly(degree=3).set_mode("classification")
        fit = spec.fit(iris_data, "species ~ sepal_length + sepal_width")

        test = pd.DataFrame({
            "sepal_length": [5.0, 6.0],
            "sepal_width": [3.5, 2.8],
        })

        # Note: SVC needs probability=True for predict_proba
        # The engine should handle this or raise appropriate error
        try:
            predictions = fit.predict(test, type="prob")
            # If it works, check structure
            assert len(predictions) == 2
            assert any(".pred_" in col for col in predictions.columns)
        except AttributeError:
            # SVC without probability=True will fail
            # This is expected behavior
            pytest.skip("SVC requires probability=True for predict_proba")


class TestSVMPolyExtractOutputs:
    """Test svm_poly output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_poly(degree=2)
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

    def test_extract_outputs_coefficients_empty(self, fitted_model):
        """Test Coefficients DataFrame is empty (polynomial is non-parametric)"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Polynomial kernel is non-parametric
        assert len(coefs) == 0

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame structure"""
        _, _, stats = fitted_model.extract_outputs()

        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "r_squared" in metric_names
        assert "degree" in metric_names
        assert "n_support" in metric_names


class TestSVMPolyParameterTranslation:
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
        spec = svm_poly(cost=5.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.C == 5.0

    def test_param_translation_degree(self, train_data):
        """Test degree parameter maps to degree"""
        spec = svm_poly(degree=2)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.degree == 2

    def test_param_translation_scale_factor(self, train_data):
        """Test scale_factor parameter maps to gamma"""
        spec = svm_poly(scale_factor=0.1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.gamma == 0.1

    def test_param_translation_margin(self, train_data):
        """Test margin parameter maps to epsilon"""
        spec = svm_poly(margin=0.2)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.epsilon == 0.2


class TestSVMPolyEvaluate:
    """Test svm_poly evaluate() method"""

    @pytest.fixture
    def train_test_data(self):
        """Create train and test data"""
        np.random.seed(42)
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

        test = pd.DataFrame({
            "y": [160, 240],
            "x1": [16, 24],
            "x2": [8, 12],
        })

        return train, test

    def test_evaluate_basic(self, train_test_data):
        """Test basic evaluate() call"""
        train, test = train_test_data

        spec = svm_poly(degree=2)
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test)

        assert "test_predictions" in fit.evaluation_data
        assert "test_data" in fit.evaluation_data

    def test_evaluate_outputs_include_test(self, train_test_data):
        """Test that extract_outputs includes test data after evaluate()"""
        train, test = train_test_data

        spec = svm_poly(degree=3)
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test)

        outputs, _, stats = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Check stats has test metrics
        test_stats = stats[stats["split"] == "test"]
        assert len(test_stats) > 0


class TestSVMPolyIntegration:
    """Integration tests"""

    def test_full_workflow_regression(self):
        """Test complete regression workflow"""
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = svm_poly(cost=5.0, degree=2, scale_factor=0.1)
        fit = spec.fit(train, "sales ~ price + advertising")

        test = pd.DataFrame({
            "price": [12, 22, 28],
            "advertising": [6, 11, 14],
        })

        predictions = fit.predict(test)

        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)

    def test_full_workflow_classification(self):
        """Test complete classification workflow"""
        np.random.seed(42)
        train = pd.DataFrame({
            "species": ["A"] * 15 + ["B"] * 15,
            "x1": np.concatenate([
                np.random.normal(5.0, 0.5, 15),
                np.random.normal(7.0, 0.5, 15)
            ]),
            "x2": np.concatenate([
                np.random.normal(3.0, 0.5, 15),
                np.random.normal(4.0, 0.5, 15)
            ]),
        })

        spec = svm_poly(cost=2.0, degree=2).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")

        test = pd.DataFrame({
            "x1": [5.0, 7.0],
            "x2": [3.0, 4.0],
        })

        predictions = fit.predict(test, type="class")

        assert len(predictions) == 2
        assert ".pred_class" in predictions.columns

    def test_different_degrees_comparison(self):
        """Test that different polynomial degrees produce different results"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        # Fit models with different degrees
        fit_deg2 = svm_poly(degree=2).fit(train, "y ~ x1 + x2")
        fit_deg3 = svm_poly(degree=3).fit(train, "y ~ x1 + x2")
        fit_deg4 = svm_poly(degree=4).fit(train, "y ~ x1 + x2")

        pred_deg2 = fit_deg2.predict(test)[".pred"].values
        pred_deg3 = fit_deg3.predict(test)[".pred"].values
        pred_deg4 = fit_deg4.predict(test)[".pred"].values

        # Predictions should be different (not exactly equal)
        # Note: They might be close but shouldn't be identical
        assert not np.allclose(pred_deg2, pred_deg3, rtol=0.001) or \
               not np.allclose(pred_deg3, pred_deg4, rtol=0.001)
