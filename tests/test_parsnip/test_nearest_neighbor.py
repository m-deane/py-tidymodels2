"""
Tests for nearest_neighbor model specification and sklearn engine

Tests cover:
- Model specification creation
- Parameter translation
- Fitting with formula
- Prediction
- Extract outputs (outputs, empty coefficients, stats)
- Evaluate() method with test data
- Different distance metrics and weighting schemes
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import nearest_neighbor, ModelSpec, ModelFit


class TestNearestNeighborSpec:
    """Test nearest_neighbor() model specification"""

    def test_default_spec(self):
        """Test default nearest_neighbor specification"""
        spec = nearest_neighbor()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "nearest_neighbor"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_neighbors(self):
        """Test nearest_neighbor with neighbors parameter"""
        spec = nearest_neighbor(neighbors=10)

        assert spec.args == {"neighbors": 10}

    def test_spec_with_weight_func(self):
        """Test nearest_neighbor with weight_func parameter"""
        spec = nearest_neighbor(weight_func="distance")

        assert spec.args == {"weight_func": "distance"}

    def test_spec_with_dist_power(self):
        """Test nearest_neighbor with dist_power parameter"""
        spec = nearest_neighbor(dist_power=1)

        assert spec.args == {"dist_power": 1}

    def test_spec_with_all_parameters(self):
        """Test nearest_neighbor with all parameters"""
        spec = nearest_neighbor(neighbors=7, weight_func="distance", dist_power=2)

        assert spec.args == {"neighbors": 7, "weight_func": "distance", "dist_power": 2}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = nearest_neighbor(neighbors=5)
        spec2 = spec1.set_args(neighbors=10)

        # Original spec should be unchanged
        assert spec1.args == {"neighbors": 5}
        # New spec should have new value
        assert spec2.args == {"neighbors": 10}


class TestNearestNeighborFit:
    """Test nearest_neighbor fitting"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = nearest_neighbor()
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_model_class(self, train_data):
        """Test correct sklearn model class"""
        spec = nearest_neighbor()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "KNeighborsRegressor"

    def test_fit_with_neighbors(self, train_data):
        """Test fitting with neighbors parameter"""
        spec = nearest_neighbor(neighbors=3)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        model = fit.fit_data["model"]
        assert model.n_neighbors == 3

    def test_fit_with_weight_func(self, train_data):
        """Test fitting with weight_func parameter"""
        spec = nearest_neighbor(weight_func="distance")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.weights == "distance"

    def test_fit_with_dist_power(self, train_data):
        """Test fitting with dist_power parameter (Minkowski p)"""
        spec = nearest_neighbor(dist_power=1)  # Manhattan distance
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.p == 1

    def test_fit_stores_residuals(self, train_data):
        """Test that fit stores residuals"""
        spec = nearest_neighbor()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert "residuals" in fit.fit_data
        assert fit.fit_data["residuals"] is not None
        assert len(fit.fit_data["residuals"]) == len(train_data)


class TestNearestNeighborPredict:
    """Test nearest_neighbor prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = nearest_neighbor(neighbors=5)
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

        # KNN should predict based on nearest neighbors
        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_predict_invalid_type(self, fitted_model):
        """Test prediction with invalid type raises error"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model.predict(test, type="class")


class TestNearestNeighborExtractOutputs:
    """Test nearest_neighbor output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

        spec = nearest_neighbor(neighbors=5)
        fit = spec.fit(train, "y ~ x1 + x2 + x3")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        sklearn_model = fitted_model.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "n_neighbors")
        assert hasattr(sklearn_model, "weights")

    def test_extract_outputs_returns_three_dataframes(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_model.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_model_outputs(self, fitted_model):
        """Test Outputs DataFrame structure"""
        outputs, _, _ = fitted_model.extract_outputs()

        # Check for observation-level columns
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "forecast" in outputs.columns
        assert "residuals" in outputs.columns
        assert "split" in outputs.columns
        assert "model" in outputs.columns
        # All training data should have split='train'
        assert all(outputs["split"] == "train")
        # Should have same number of rows as training data
        assert len(outputs) == 10

    def test_extract_outputs_coefficients_empty(self, fitted_model):
        """Test Coefficients DataFrame is empty (KNN is non-parametric)"""
        _, coefs, _ = fitted_model.extract_outputs()

        # KNN is non-parametric, so no coefficients
        assert len(coefs) == 0

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame structure"""
        _, _, stats = fitted_model.extract_outputs()

        # Check that stats has metrics
        assert len(stats) > 0
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check for specific metrics
        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names
        assert "n_neighbors" in metric_names
        assert "weights" in metric_names


class TestNearestNeighborEvaluate:
    """Test nearest_neighbor evaluate() method"""

    @pytest.fixture
    def train_test_data(self):
        """Create train/test split"""
        np.random.seed(42)
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })
        test = pd.DataFrame({
            "y": [160, 240, 200],
            "x1": [16, 24, 20],
            "x2": [8, 12, 10],
        })
        return train, test

    def test_evaluate(self, train_test_data):
        """Test evaluate() method"""
        train, test = train_test_data

        spec = nearest_neighbor(neighbors=3)
        fit = spec.fit(train, "y ~ x1 + x2")

        # Evaluate on test data
        fit = fit.evaluate(test, "y")

        # Check that evaluation data is stored
        assert "test_predictions" in fit.evaluation_data
        assert "test_data" in fit.evaluation_data
        assert "outcome_col" in fit.evaluation_data

    def test_evaluate_outputs(self, train_test_data):
        """Test that evaluate() includes test data in outputs"""
        train, test = train_test_data

        spec = nearest_neighbor()
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        # Test split should have 3 observations
        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == 3


class TestNearestNeighborParameterTranslation:
    """Test parameter translation from tidymodels to sklearn"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

    def test_param_translation_neighbors(self, train_data):
        """Test neighbors parameter maps to n_neighbors"""
        spec = nearest_neighbor(neighbors=7)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.n_neighbors == 7

    def test_param_translation_weight_func(self, train_data):
        """Test weight_func parameter maps to weights"""
        spec = nearest_neighbor(weight_func="distance")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.weights == "distance"

    def test_param_translation_dist_power(self, train_data):
        """Test dist_power parameter maps to p"""
        spec = nearest_neighbor(dist_power=1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.p == 1


class TestNearestNeighborIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self):
        """Test complete workflow"""
        # Training data
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        # Create spec and fit
        spec = nearest_neighbor(neighbors=5, weight_func="distance")
        fit = spec.fit(train, "sales ~ price + advertising")

        # Test data
        test = pd.DataFrame({
            "price": [12, 22, 28],
            "advertising": [6, 11, 14],
        })

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)  # Sales should be positive

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()
        assert len(outputs) == 10  # Training observations
        assert len(coefs) == 0  # No coefficients for KNN
        assert len(stats) > 0
