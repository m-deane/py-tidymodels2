"""
Tests for pls model specification and sklearn engine

Tests cover:
- Model specification creation
- Engine registration
- Fitting with formula
- Prediction
- Extract outputs
- Component handling
- High-dimensional data
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip.models.pls import pls
from py_parsnip.model_spec import ModelSpec, ModelFit


class TestPLSSpec:
    """Test pls() model specification"""

    def test_default_spec(self):
        """Test default pls specification"""
        spec = pls()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "pls"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_num_comp(self):
        """Test pls with num_comp parameter"""
        spec = pls(num_comp=3)

        assert spec.args == {"num_comp": 3}

    def test_spec_with_predictor_prop(self):
        """Test pls with predictor_prop parameter"""
        spec = pls(predictor_prop=0.8)

        assert spec.args == {"predictor_prop": 0.8}

    def test_spec_with_both_params(self):
        """Test pls with both num_comp and predictor_prop"""
        spec = pls(num_comp=5, predictor_prop=0.9)

        assert spec.args == {"num_comp": 5, "predictor_prop": 0.9}

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = pls()
        spec = spec.set_engine("sklearn")

        assert spec.engine == "sklearn"

    def test_set_args(self):
        """Test set_args() method"""
        spec = pls()
        spec = spec.set_args(num_comp=4)

        assert spec.args == {"num_comp": 4}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = pls(num_comp=2)
        spec2 = spec1.set_args(num_comp=5)

        # Original spec should be unchanged
        assert spec1.args == {"num_comp": 2}
        # New spec should have new value
        assert spec2.args == {"num_comp": 5}


class TestPLSFit:
    """Test pls fitting with sklearn engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5],
        })

    @pytest.fixture
    def high_dim_data(self):
        """Create high-dimensional data for PLS"""
        np.random.seed(123)
        n = 50
        p = 20  # More features than typically useful

        # Create correlated predictors
        X = np.random.randn(n, p)
        # Add correlation structure
        for i in range(1, p):
            X[:, i] = 0.7 * X[:, i-1] + 0.3 * X[:, i]

        # Create response with relationship to some features
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5

        data = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
        data["y"] = y

        return data

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_default_components(self, train_data):
        """Test PLS with default 2 components"""
        spec = pls()
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        # Should use 2 components by default
        assert fit.fit_data["model_class"] == "PLSRegression"
        assert fit.fit_data["n_components"] == 2

    def test_fit_custom_components(self, train_data):
        """Test PLS with custom number of components"""
        spec = pls(num_comp=3)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        # Should use 3 components as specified
        assert fit.fit_data["n_components"] == 3

    def test_fit_components_limited_by_features(self, train_data):
        """Test that components are limited by number of features"""
        # Request more components than features
        spec = pls(num_comp=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        # Should be limited by min(n_samples, n_features)
        # With 8 samples and 3 features (plus intercept=4), should be limited
        assert fit.fit_data["n_components"] <= 8

    def test_predict(self, train_data):
        """Test prediction"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        predictions = fit.predict(train_data)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(train_data)

    def test_predict_new_data(self, train_data):
        """Test prediction on new data"""
        train = train_data[:6]
        test = train_data[6:]

        spec = pls(num_comp=2)
        fit = spec.fit(train, "y ~ x1 + x2 + x3")

        test_predictions = fit.predict(test)

        assert len(test_predictions) == len(test)
        assert ".pred" in test_predictions.columns

    def test_extract_fit_engine(self, train_data):
        """Test extracting underlying sklearn model"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        model = fit.extract_fit_engine()

        # Should be sklearn PLSRegression
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")

    def test_extract_outputs(self, train_data):
        """Test extract_outputs returns three DataFrames"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert "split" in outputs.columns
        assert all(outputs["split"] == "train")

        # Check coefficients
        assert isinstance(coefficients, pd.DataFrame)
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns
        assert len(coefficients) == 4  # Intercept, x1, x2, x3

        # Check stats
        assert isinstance(stats, pd.DataFrame)
        assert "metric" in stats.columns
        assert "value" in stats.columns

        # Check for PLS-specific metrics
        metrics = stats["metric"].tolist()
        assert "n_components" in metrics
        assert "rmse" in metrics
        assert "r_squared" in metrics

    def test_evaluate_with_test_data(self, train_data):
        """Test evaluate() method with test data"""
        train = train_data[:6]
        test = train_data[6:]

        spec = pls(num_comp=2)
        fit = spec.fit(train, "y ~ x1 + x2 + x3")
        fit = fit.evaluate(test)

        outputs, coefficients, stats = fit.extract_outputs()

        # Should have both train and test splits
        assert set(outputs["split"].unique()) == {"train", "test"}

        # Check test metrics exist
        test_stats = stats[stats["split"] == "test"]
        assert len(test_stats) > 0
        assert "rmse" in test_stats["metric"].values

    def test_high_dimensional_data(self, high_dim_data):
        """Test PLS on high-dimensional data"""
        # Create formula with all 20 predictors
        predictors = " + ".join([f"x{i}" for i in range(20)])
        formula = f"y ~ {predictors}"

        spec = pls(num_comp=5)  # Use 5 components for dimension reduction
        fit = spec.fit(high_dim_data, formula)

        assert fit.fit_data["n_components"] == 5
        # With intercept included in features, n_features = 21
        assert fit.fit_data["n_features"] == 21

        # Check predictions work
        predictions = fit.predict(high_dim_data)
        assert len(predictions) == len(high_dim_data)

        # Check coefficients include intercept
        outputs, coefficients, stats = fit.extract_outputs()
        assert len(coefficients) == 21  # Intercept + 20 predictors

    def test_multicollinearity_handling(self, high_dim_data):
        """Test PLS handles multicollinearity well"""
        # high_dim_data has correlated features by design
        predictors = " + ".join([f"x{i}" for i in range(20)])
        formula = f"y ~ {predictors}"

        spec = pls(num_comp=3)
        fit = spec.fit(high_dim_data, formula)

        outputs, coefficients, stats = fit.extract_outputs()

        # Should successfully fit despite multicollinearity
        # Includes intercept + 20 predictors
        assert len(coefficients) == 21
        # Check that we get reasonable R-squared
        r_sq_row = stats[(stats["metric"] == "r_squared") & (stats["split"] == "train")]
        if len(r_sq_row) > 0:
            r_squared = r_sq_row["value"].iloc[0]
            assert not np.isnan(r_squared)
            assert r_squared > 0  # Should capture some variance

    def test_small_components_underfitting(self, train_data):
        """Test PLS with very few components"""
        spec = pls(num_comp=1)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should still work, but might have lower R-squared
        assert fit.fit_data["n_components"] == 1
        assert len(coefficients) == 4  # Intercept + x1, x2, x3

    def test_prediction_accuracy(self, train_data):
        """Test that PLS predictions are reasonable"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        predictions = fit.predict(train_data)
        actuals = train_data["y"].values

        # Calculate R-squared manually
        residuals = actuals - predictions[".pred"].values
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Should have decent fit (R² > 0.5 at least)
        assert r_squared > 0.5

    def test_residuals_calculation(self, train_data):
        """Test that residuals are correctly calculated"""
        spec = pls(num_comp=2)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, _, _ = fit.extract_outputs()

        # Check residuals = actuals - fitted
        calculated_residuals = outputs["actuals"] - outputs["fitted"]
        np.testing.assert_array_almost_equal(
            outputs["residuals"].values,
            calculated_residuals.values,
            decimal=10
        )
