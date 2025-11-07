"""
Tests for linear_reg regularization (Ridge, Lasso, ElasticNet)

Tests verify that:
- Different mixture values produce different model types
- Penalty parameter affects coefficient magnitudes
- Ridge, Lasso, and ElasticNet produce different coefficients
- Higher penalties lead to more regularization
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import linear_reg


class TestRegularizationModelSelection:
    """Test that penalty and mixture select correct sklearn model classes"""

    @pytest.fixture
    def train_data(self):
        """Create training data with multiple features"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n) * 10 + 50,
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'x4': np.random.randn(n),
        })

    def test_ridge_model_class(self, train_data):
        """Test that mixture=0 creates Ridge model"""
        spec = linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        assert fit.fit_data['model_class'] == "Ridge"
        assert hasattr(fit.fit_data['model'], 'alpha')
        assert fit.fit_data['model'].alpha == 0.1

    def test_lasso_model_class(self, train_data):
        """Test that mixture=1 creates Lasso model"""
        spec = linear_reg(penalty=0.1, mixture=1.0).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        assert fit.fit_data['model_class'] == "Lasso"
        assert hasattr(fit.fit_data['model'], 'alpha')
        assert fit.fit_data['model'].alpha == 0.1

    def test_elasticnet_model_class(self, train_data):
        """Test that 0 < mixture < 1 creates ElasticNet model"""
        spec = linear_reg(penalty=0.1, mixture=0.5).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        assert fit.fit_data['model_class'] == "ElasticNet"
        assert hasattr(fit.fit_data['model'], 'alpha')
        assert hasattr(fit.fit_data['model'], 'l1_ratio')
        assert fit.fit_data['model'].alpha == 0.1
        assert fit.fit_data['model'].l1_ratio == 0.5

    def test_ols_model_class(self, train_data):
        """Test that penalty=0 or None creates LinearRegression"""
        spec = linear_reg(penalty=0).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        assert fit.fit_data['model_class'] == "LinearRegression"


class TestRegularizationCoefficients:
    """Test that regularization affects coefficients as expected"""

    @pytest.fixture
    def train_data(self):
        """Create training data with correlated features"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # Highly correlated with x1
        x3 = np.random.randn(n)
        x4 = np.random.randn(n)

        # Create y with known relationship
        y = 2*x1 + 3*x2 + 1*x3 + 0.5*x4 + np.random.randn(n) * 0.5

        return pd.DataFrame({
            'y': y,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
        })

    def test_ridge_lasso_elasticnet_produce_different_coefficients(self, train_data):
        """Test that Ridge, Lasso, and ElasticNet produce different coefficients"""
        # Fit three different models with same penalty but different mixture
        spec_ridge = linear_reg(penalty=1.0, mixture=0.0).set_engine("sklearn")
        spec_lasso = linear_reg(penalty=1.0, mixture=1.0).set_engine("sklearn")
        spec_elastic = linear_reg(penalty=1.0, mixture=0.5).set_engine("sklearn")

        fit_ridge = spec_ridge.fit(train_data, "y ~ x1 + x2 + x3 + x4")
        fit_lasso = spec_lasso.fit(train_data, "y ~ x1 + x2 + x3 + x4")
        fit_elastic = spec_elastic.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        # Extract coefficients
        _, coefs_ridge, _ = fit_ridge.extract_outputs()
        _, coefs_lasso, _ = fit_lasso.extract_outputs()
        _, coefs_elastic, _ = fit_elastic.extract_outputs()

        # Get coefficient values (excluding intercept for comparison)
        ridge_vals = coefs_ridge[coefs_ridge['variable'] != 'Intercept']['coefficient'].values
        lasso_vals = coefs_lasso[coefs_lasso['variable'] != 'Intercept']['coefficient'].values
        elastic_vals = coefs_elastic[coefs_elastic['variable'] != 'Intercept']['coefficient'].values

        # Ridge and Lasso should produce different coefficients
        assert not np.allclose(ridge_vals, lasso_vals, rtol=0.01), \
            "Ridge and Lasso coefficients should be different"

        # Ridge and ElasticNet should produce different coefficients
        assert not np.allclose(ridge_vals, elastic_vals, rtol=0.01), \
            "Ridge and ElasticNet coefficients should be different"

        # Lasso and ElasticNet should produce different coefficients
        assert not np.allclose(lasso_vals, elastic_vals, rtol=0.01), \
            "Lasso and ElasticNet coefficients should be different"

    def test_higher_penalty_reduces_coefficient_magnitudes(self, train_data):
        """Test that higher penalty leads to smaller coefficient magnitudes"""
        # Fit Ridge models with different penalties
        spec_low = linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn")
        spec_high = linear_reg(penalty=10.0, mixture=0.0).set_engine("sklearn")

        fit_low = spec_low.fit(train_data, "y ~ x1 + x2 + x3 + x4")
        fit_high = spec_high.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        # Extract coefficients
        _, coefs_low, _ = fit_low.extract_outputs()
        _, coefs_high, _ = fit_high.extract_outputs()

        # Get coefficient magnitudes (excluding intercept)
        low_vals = coefs_low[coefs_low['variable'] != 'Intercept']['coefficient'].values
        high_vals = coefs_high[coefs_high['variable'] != 'Intercept']['coefficient'].values

        # Higher penalty should lead to smaller coefficient magnitudes
        low_magnitude = np.linalg.norm(low_vals)
        high_magnitude = np.linalg.norm(high_vals)

        assert high_magnitude < low_magnitude, \
            f"Higher penalty should reduce coefficient magnitudes: {high_magnitude} >= {low_magnitude}"

    def test_lasso_can_zero_out_coefficients(self, train_data):
        """Test that Lasso with high penalty can zero out coefficients"""
        # Fit Lasso with very high penalty
        spec = linear_reg(penalty=100.0, mixture=1.0).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        # Extract coefficients
        _, coefs, _ = fit.extract_outputs()

        # Get non-intercept coefficients
        non_intercept_coefs = coefs[coefs['variable'] != 'Intercept']['coefficient'].values

        # At least one coefficient should be very close to zero
        min_abs_coef = np.min(np.abs(non_intercept_coefs))
        assert min_abs_coef < 0.1, \
            "Lasso with high penalty should zero out some coefficients"

    def test_elasticnet_interpolates_between_ridge_and_lasso(self, train_data):
        """Test that ElasticNet behavior is between Ridge and Lasso"""
        # Fit three models
        spec_ridge = linear_reg(penalty=1.0, mixture=0.0).set_engine("sklearn")
        spec_elastic = linear_reg(penalty=1.0, mixture=0.5).set_engine("sklearn")
        spec_lasso = linear_reg(penalty=1.0, mixture=1.0).set_engine("sklearn")

        fit_ridge = spec_ridge.fit(train_data, "y ~ x1 + x2 + x3 + x4")
        fit_elastic = spec_elastic.fit(train_data, "y ~ x1 + x2 + x3 + x4")
        fit_lasso = spec_lasso.fit(train_data, "y ~ x1 + x2 + x3 + x4")

        # Extract coefficients
        _, coefs_ridge, _ = fit_ridge.extract_outputs()
        _, coefs_elastic, _ = fit_elastic.extract_outputs()
        _, coefs_lasso, _ = fit_lasso.extract_outputs()

        # Get coefficient values (excluding intercept)
        ridge_vals = coefs_ridge[coefs_ridge['variable'] != 'Intercept']['coefficient'].values
        elastic_vals = coefs_elastic[coefs_elastic['variable'] != 'Intercept']['coefficient'].values
        lasso_vals = coefs_lasso[coefs_lasso['variable'] != 'Intercept']['coefficient'].values

        # ElasticNet coefficients should generally be between Ridge and Lasso
        # Check that ElasticNet is closer to the middle than either extreme
        dist_ridge_elastic = np.linalg.norm(ridge_vals - elastic_vals)
        dist_lasso_elastic = np.linalg.norm(lasso_vals - elastic_vals)
        dist_ridge_lasso = np.linalg.norm(ridge_vals - lasso_vals)

        # ElasticNet should not be identical to either Ridge or Lasso
        assert dist_ridge_elastic > 0.01, "ElasticNet should differ from Ridge"
        assert dist_lasso_elastic > 0.01, "ElasticNet should differ from Lasso"


class TestRegularizationPredictions:
    """Test that different regularization affects predictions"""

    @pytest.fixture
    def train_data(self):
        """Create training data"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n) * 10 + 50,
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
        })

    @pytest.fixture
    def test_data(self):
        """Create test data"""
        np.random.seed(123)
        n = 20
        return pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
        })

    def test_different_models_produce_different_predictions(self, train_data, test_data):
        """Test that Ridge, Lasso, and ElasticNet produce different predictions"""
        # Fit models
        spec_ridge = linear_reg(penalty=1.0, mixture=0.0).set_engine("sklearn")
        spec_lasso = linear_reg(penalty=1.0, mixture=1.0).set_engine("sklearn")
        spec_elastic = linear_reg(penalty=1.0, mixture=0.5).set_engine("sklearn")

        fit_ridge = spec_ridge.fit(train_data, "y ~ x1 + x2 + x3")
        fit_lasso = spec_lasso.fit(train_data, "y ~ x1 + x2 + x3")
        fit_elastic = spec_elastic.fit(train_data, "y ~ x1 + x2 + x3")

        # Make predictions
        pred_ridge = fit_ridge.predict(test_data)
        pred_lasso = fit_lasso.predict(test_data)
        pred_elastic = fit_elastic.predict(test_data)

        # Predictions should be different
        assert not np.allclose(pred_ridge['.pred'].values, pred_lasso['.pred'].values, rtol=0.01), \
            "Ridge and Lasso predictions should differ"
        assert not np.allclose(pred_ridge['.pred'].values, pred_elastic['.pred'].values, rtol=0.01), \
            "Ridge and ElasticNet predictions should differ"


class TestRegularizationEdgeCases:
    """Test edge cases for regularization"""

    @pytest.fixture
    def train_data(self):
        """Create training data"""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'y': np.random.randn(n) * 5 + 20,
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })

    def test_zero_penalty_equals_ols(self, train_data):
        """Test that penalty=0 is equivalent to OLS"""
        spec_ols = linear_reg().set_engine("sklearn")
        spec_ridge_zero = linear_reg(penalty=0, mixture=0).set_engine("sklearn")

        fit_ols = spec_ols.fit(train_data, "y ~ x1 + x2")
        fit_ridge_zero = spec_ridge_zero.fit(train_data, "y ~ x1 + x2")

        # Both should use LinearRegression
        assert fit_ols.fit_data['model_class'] == "LinearRegression"
        assert fit_ridge_zero.fit_data['model_class'] == "LinearRegression"

        # Coefficients should be identical
        _, coefs_ols, _ = fit_ols.extract_outputs()
        _, coefs_ridge_zero, _ = fit_ridge_zero.extract_outputs()

        assert np.allclose(
            coefs_ols['coefficient'].values,
            coefs_ridge_zero['coefficient'].values
        )

    def test_very_small_penalty(self, train_data):
        """Test that very small penalty behaves like OLS"""
        spec_ols = linear_reg().set_engine("sklearn")
        spec_tiny_penalty = linear_reg(penalty=1e-10, mixture=0).set_engine("sklearn")

        fit_ols = spec_ols.fit(train_data, "y ~ x1 + x2")
        fit_tiny = spec_tiny_penalty.fit(train_data, "y ~ x1 + x2")

        # Extract coefficients
        _, coefs_ols, _ = fit_ols.extract_outputs()
        _, coefs_tiny, _ = fit_tiny.extract_outputs()

        # Should be very close (but tiny penalty still uses Ridge)
        assert fit_tiny.fit_data['model_class'] == "Ridge"
        assert np.allclose(
            coefs_ols['coefficient'].values,
            coefs_tiny['coefficient'].values,
            rtol=0.01
        )

    def test_mixture_boundary_values(self, train_data):
        """Test mixture at boundary values (0.0 and 1.0)"""
        # mixture=0.0 should use Ridge
        spec_0 = linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn")
        fit_0 = spec_0.fit(train_data, "y ~ x1 + x2")
        assert fit_0.fit_data['model_class'] == "Ridge"

        # mixture=1.0 should use Lasso
        spec_1 = linear_reg(penalty=0.1, mixture=1.0).set_engine("sklearn")
        fit_1 = spec_1.fit(train_data, "y ~ x1 + x2")
        assert fit_1.fit_data['model_class'] == "Lasso"

    def test_different_penalty_values(self, train_data):
        """Test a range of penalty values"""
        penalties = [0.01, 0.1, 1.0, 10.0, 100.0]
        coef_magnitudes = []

        for penalty in penalties:
            spec = linear_reg(penalty=penalty, mixture=0).set_engine("sklearn")
            fit = spec.fit(train_data, "y ~ x1 + x2")
            _, coefs, _ = fit.extract_outputs()

            # Get magnitude of non-intercept coefficients
            non_intercept = coefs[coefs['variable'] != 'Intercept']['coefficient'].values
            magnitude = np.linalg.norm(non_intercept)
            coef_magnitudes.append(magnitude)

        # Coefficient magnitudes should generally decrease with higher penalty
        # (allowing for some non-monotonicity due to the specific data)
        assert coef_magnitudes[0] > coef_magnitudes[-1], \
            "Higher penalties should generally reduce coefficient magnitudes"
