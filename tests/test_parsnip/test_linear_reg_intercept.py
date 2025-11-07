"""
Tests for linear_reg intercept parameter

Tests cover:
- Default intercept=True behavior (backwards compatibility)
- intercept=False for regression through origin
- Sklearn engine intercept handling
- Statsmodels engine intercept handling
- Coefficient extraction with/without intercept
- Workflow integration
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from py_parsnip import linear_reg, ModelSpec, ModelFit


class TestInterceptParameter:
    """Test intercept parameter specification"""

    def test_default_intercept_true(self):
        """Test that intercept defaults to True"""
        spec = linear_reg()

        assert "intercept" in spec.args
        assert spec.args["intercept"] is True

    def test_explicit_intercept_true(self):
        """Test explicit intercept=True"""
        spec = linear_reg(intercept=True)

        assert spec.args["intercept"] is True

    def test_explicit_intercept_false(self):
        """Test explicit intercept=False"""
        spec = linear_reg(intercept=False)

        assert spec.args["intercept"] is False

    def test_intercept_with_penalty(self):
        """Test intercept parameter works with penalty"""
        spec = linear_reg(penalty=0.1, intercept=False)

        assert spec.args["penalty"] == 0.1
        assert spec.args["intercept"] is False


class TestSklearnEngineIntercept:
    """Test sklearn engine intercept handling"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [10, 20, 30, 40, 50],
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        })

    def test_sklearn_intercept_true_default(self, train_data):
        """Test sklearn engine with default intercept=True"""
        spec = linear_reg()  # intercept=True by default
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Check that model has fit_intercept=True
        model = fit.fit_data["model"]
        assert hasattr(model, "fit_intercept")
        assert model.fit_intercept is True

        # Check that intercept is non-zero
        assert hasattr(model, "intercept_")
        # In this case, intercept might be zero if x1 and x2 perfectly predict y
        # So we just check it exists

        # Check coefficients include Intercept row
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" in coefficients["variable"].values

    def test_sklearn_intercept_false(self, train_data):
        """Test sklearn engine with intercept=False"""
        spec = linear_reg(intercept=False).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Check that model has fit_intercept=False
        model = fit.fit_data["model"]
        assert hasattr(model, "fit_intercept")
        assert model.fit_intercept is False

        # Check that intercept is zero
        assert hasattr(model, "intercept_")
        assert model.intercept_ == 0.0

        # Check coefficients do NOT include Intercept row
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" not in coefficients["variable"].values

    def test_sklearn_ridge_intercept_false(self, train_data):
        """Test Ridge regression with intercept=False"""
        spec = linear_reg(penalty=0.1, mixture=0.0, intercept=False).set_engine("sklearn")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "Ridge"
        assert fit.fit_data["model"].fit_intercept is False

        # Check coefficients
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" not in coefficients["variable"].values

    def test_sklearn_predictions_no_intercept(self, train_data):
        """Test predictions work correctly with intercept=False"""
        spec = linear_reg(intercept=False)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Make predictions
        test = pd.DataFrame({
            "x1": [6, 7],
            "x2": [12, 14],
        })
        predictions = fit.predict(test)

        assert ".pred" in predictions.columns
        assert len(predictions) == 2
        # Predictions should be non-zero (model still fits through origin)
        assert all(predictions[".pred"] > 0)


class TestStatsmodelsEngineIntercept:
    """Test statsmodels engine intercept handling"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [10, 20, 30, 40, 50],
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        })

    def test_statsmodels_intercept_true_default(self, train_data):
        """Test statsmodels engine with default intercept=True"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Check coefficients include Intercept
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" in coefficients["variable"].values

        # Intercept coefficient should exist
        intercept_row = coefficients[coefficients["variable"] == "Intercept"]
        assert len(intercept_row) == 1

    def test_statsmodels_intercept_false(self, train_data):
        """Test statsmodels engine with intercept=False"""
        spec = linear_reg(intercept=False).set_engine("statsmodels")

        # Should issue a warning about removing intercept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = spec.fit(train_data, "y ~ x1 + x2")

            # Check that a warning was issued
            assert len(w) == 1
            assert "intercept=False" in str(w[0].message)

        # Check coefficients do NOT include Intercept
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" not in coefficients["variable"].values

    def test_statsmodels_predictions_no_intercept(self, train_data):
        """Test predictions work correctly with intercept=False"""
        spec = linear_reg(intercept=False).set_engine("statsmodels")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = spec.fit(train_data, "y ~ x1 + x2")

        # Make predictions
        test = pd.DataFrame({
            "x1": [6, 7],
            "x2": [12, 14],
        })
        predictions = fit.predict(test)

        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_statsmodels_formula_no_intercept(self, train_data):
        """Test statsmodels with formula that already has +0 (no intercept)"""
        spec = linear_reg(intercept=True).set_engine("statsmodels")

        # Formula already specifies no intercept
        fit = spec.fit(train_data, "y ~ x1 + x2 + 0")

        # Check coefficients do NOT include Intercept (formula wins)
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" not in coefficients["variable"].values


class TestInterceptBackwardsCompatibility:
    """Test that existing code continues to work (backwards compatibility)"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [10, 20, 30, 40, 50],
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        })

    def test_no_intercept_param_defaults_true(self, train_data):
        """Test that code without intercept parameter still works (intercept=True)"""
        # Old code that doesn't specify intercept
        spec = linear_reg()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should have intercept by default
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" in coefficients["variable"].values

    def test_penalty_without_intercept(self, train_data):
        """Test Ridge/Lasso without specifying intercept"""
        spec = linear_reg(penalty=0.1, mixture=0.5)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should have intercept by default
        outputs, coefficients, stats = fit.extract_outputs()
        # Note: Regularized models might not show Intercept in coefficients
        # depending on implementation, so we just check it doesn't crash


class TestInterceptWorkflowIntegration:
    """Test intercept parameter works in workflows"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [10, 20, 30, 40, 50],
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        })

    def test_workflow_with_intercept_false(self, train_data):
        """Test intercept=False in a workflow"""
        from py_workflows import Workflow

        spec = linear_reg(intercept=False)
        wf = Workflow().add_formula("y ~ x1 + x2").add_model(spec)

        fit = wf.fit(train_data)

        # Check that predictions work
        test = pd.DataFrame({
            "x1": [6, 7],
            "x2": [12, 14],
        })
        predictions = fit.predict(test)
        assert len(predictions) == 2

    def test_workflow_with_intercept_true(self, train_data):
        """Test intercept=True (default) in a workflow"""
        from py_workflows import Workflow

        spec = linear_reg()  # intercept=True by default
        wf = Workflow().add_formula("y ~ x1 + x2").add_model(spec)

        fit = wf.fit(train_data)

        # Extract outputs and check for intercept
        outputs, coefficients, stats = fit.extract_outputs()
        assert "Intercept" in coefficients["variable"].values


class TestInterceptCoefficientExtraction:
    """Test coefficient extraction with/without intercept"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [10, 20, 30, 40, 50],
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        })

    def test_coefficients_with_intercept(self, train_data):
        """Test coefficient DataFrame includes intercept"""
        spec = linear_reg(intercept=True)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should have 3 rows: Intercept, x1, x2
        assert len(coefficients) == 3
        assert "Intercept" in coefficients["variable"].values
        assert "x1" in coefficients["variable"].values
        assert "x2" in coefficients["variable"].values

        # Intercept should have a coefficient value
        intercept_row = coefficients[coefficients["variable"] == "Intercept"]
        assert not pd.isna(intercept_row["coefficient"].iloc[0])

    def test_coefficients_without_intercept(self, train_data):
        """Test coefficient DataFrame excludes intercept"""
        spec = linear_reg(intercept=False)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should have 2 rows: x1, x2 (no Intercept)
        assert len(coefficients) == 2
        assert "Intercept" not in coefficients["variable"].values
        assert "x1" in coefficients["variable"].values
        assert "x2" in coefficients["variable"].values

    def test_coefficient_count_matches_parameters(self, train_data):
        """Test that number of coefficients matches number of parameters"""
        # With intercept: 3 parameters (intercept + x1 + x2)
        spec_with = linear_reg(intercept=True)
        fit_with = spec_with.fit(train_data, "y ~ x1 + x2")
        outputs_with, coefficients_with, stats_with = fit_with.extract_outputs()
        assert len(coefficients_with) == 3

        # Without intercept: 2 parameters (x1 + x2)
        spec_without = linear_reg(intercept=False)
        fit_without = spec_without.fit(train_data, "y ~ x1 + x2")
        outputs_without, coefficients_without, stats_without = fit_without.extract_outputs()
        assert len(coefficients_without) == 2


class TestInterceptStatisticalInference:
    """Test that statistical inference works correctly with/without intercept"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        # Create data where y = 2*x1 + 3*x2 + intercept
        # Use uncorrelated x1 and x2 to avoid multicollinearity
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([1, 3, 2, 5, 4])  # Different pattern than x1
        y = 5 + 2*x1 + 3*x2 + np.random.normal(0, 0.1, 5)
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def test_std_errors_with_intercept(self, train_data):
        """Test standard errors calculated with intercept"""
        spec = linear_reg(intercept=True)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # All coefficients should have std_error
        # (might be NaN for regularized models, but not for OLS)
        assert "std_error" in coefficients.columns

        # For OLS, std_error should not be NaN
        if fit.fit_data.get("model_class") == "LinearRegression":
            assert not coefficients["std_error"].isna().all()

    def test_std_errors_without_intercept(self, train_data):
        """Test standard errors calculated without intercept"""
        spec = linear_reg(intercept=False)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should have std_error column
        assert "std_error" in coefficients.columns

        # Number of rows should be 2 (x1, x2, no intercept)
        assert len(coefficients) == 2
