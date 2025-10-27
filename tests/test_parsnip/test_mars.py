"""
Tests for MARS (Multivariate Adaptive Regression Splines) model

Tests cover:
- Model specification creation
- Engine registration
- Fitting with formula
- Prediction
- Extract outputs (basis functions)
- Parameter translation
- Non-linear relationship detection
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import mars, ModelSpec, ModelFit


class TestMARSSpec:
    """Test mars() model specification"""

    def test_default_spec(self):
        """Test default MARS specification"""
        spec = mars()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "mars"
        assert spec.engine == "pyearth"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_num_terms(self):
        """Test MARS with num_terms parameter"""
        spec = mars(num_terms=10)

        assert spec.args == {"num_terms": 10}

    def test_spec_with_prod_degree(self):
        """Test MARS with interaction degree"""
        spec = mars(prod_degree=2)

        assert spec.args == {"prod_degree": 2}

    def test_spec_with_all_params(self):
        """Test MARS with all parameters"""
        spec = mars(num_terms=15, prod_degree=2, prune_method="forward")

        assert spec.args == {
            "num_terms": 15,
            "prod_degree": 2,
            "prune_method": "forward"
        }

    def test_set_args(self):
        """Test set_args() method"""
        spec = mars()
        spec = spec.set_args(num_terms=20, prod_degree=3)

        assert spec.args == {"num_terms": 20, "prod_degree": 3}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = mars(num_terms=10)
        spec2 = spec1.set_args(num_terms=20)

        # Original spec should be unchanged
        assert spec1.args == {"num_terms": 10}
        # New spec should have new value
        assert spec2.args == {"num_terms": 20}


class TestMARSFit:
    """Test MARS fitting with pyearth engine"""

    @pytest.fixture
    def linear_data(self):
        """Create simple linear training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

    @pytest.fixture
    def nonlinear_data(self):
        """Create non-linear training data (quadratic)"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = x ** 2 + np.random.normal(0, 5, 50)
        return pd.DataFrame({"y": y, "x": x})

    def test_fit_with_formula(self, linear_data):
        """Test fitting with formula"""
        spec = mars()
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_default(self, linear_data):
        """Test default MARS fit"""
        spec = mars()
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        # Should have fitted model
        assert "model" in fit.fit_data
        assert "n_terms" in fit.fit_data

    def test_fit_with_num_terms(self, linear_data):
        """Test MARS with limited terms"""
        spec = mars(num_terms=5)
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        # Should respect num_terms
        assert fit.fit_data["n_terms"] <= 5

    def test_fit_with_prod_degree(self, linear_data):
        """Test MARS with interaction terms"""
        spec = mars(prod_degree=2)
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        assert "model" in fit.fit_data

    def test_fit_nonlinear(self, nonlinear_data):
        """Test MARS can fit non-linear relationships"""
        spec = mars(num_terms=10)
        fit = spec.fit(nonlinear_data, "y ~ x")

        # MARS should detect the quadratic relationship
        assert isinstance(fit, ModelFit)
        assert fit.fit_data["n_terms"] > 1  # Should use multiple terms


class TestMARSPredict:
    """Test MARS prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

        spec = mars()
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_predict_basic(self, fitted_model):
        """Test basic prediction"""
        test = pd.DataFrame({
            "x1": [12, 22, 26],
            "x2": [6, 11, 13],
        })

        predictions = fitted_model.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3

    def test_predict_values_reasonable(self, fitted_model):
        """Test that predictions are in reasonable range"""
        test = pd.DataFrame({
            "x1": [15],
            "x2": [7],
        })

        predictions = fitted_model.predict(test)

        # Should be in reasonable range based on training data
        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_predict_type_numeric(self, fitted_model):
        """Test numeric prediction type"""
        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fitted_model.predict(test, type="numeric")
        assert ".pred" in predictions.columns

    def test_predict_invalid_type(self, fitted_model):
        """Test that invalid prediction type raises error"""
        test = pd.DataFrame({
            "x1": [12],
            "x2": [6],
        })

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model.predict(test, type="prob")


class TestMARSExtract:
    """Test MARS output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

        spec = mars()
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        earth_model = fitted_model.extract_fit_engine()

        assert earth_model is not None
        # Check for Earth model attributes
        assert hasattr(earth_model, "basis_") or hasattr(earth_model, "coef_")

    def test_extract_outputs(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, basis_funcs, stats = fitted_model.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(basis_funcs, pd.DataFrame)
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
        assert len(outputs) == 8

    def test_extract_outputs_basis_functions(self, fitted_model):
        """Test Basis Functions DataFrame structure"""
        _, basis_funcs, _ = fitted_model.extract_outputs()

        # Check for basis function columns
        assert "basis_id" in basis_funcs.columns or "description" in basis_funcs.columns
        # Should have at least one basis function
        assert len(basis_funcs) >= 1

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame structure"""
        _, _, stats = fitted_model.extract_outputs()

        # Check for stats columns
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check for key metrics
        stat_names = stats["metric"].tolist()
        assert "rmse" in stat_names
        assert "r_squared" in stat_names
        assert "n_terms" in stat_names


class TestMARSNonlinearity:
    """Test MARS ability to detect non-linear relationships"""

    def test_quadratic_relationship(self):
        """Test MARS fits quadratic relationship"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = x ** 2 + np.random.normal(0, 5, 50)
        data = pd.DataFrame({"y": y, "x": x})

        spec = mars(num_terms=10)
        fit = spec.fit(data, "y ~ x")

        # Predict
        test = pd.DataFrame({"x": [3, 5, 7]})
        predictions = fit.predict(test)

        # Check predictions roughly follow quadratic
        # Expected: 9, 25, 49 (with some noise tolerance)
        assert predictions[".pred"].iloc[0] > 5  # ~9
        assert predictions[".pred"].iloc[1] > 15  # ~25
        assert predictions[".pred"].iloc[2] > 35  # ~49

    def test_interaction_detection(self):
        """Test MARS with interaction terms"""
        np.random.seed(42)
        x1 = np.random.uniform(0, 10, 40)
        x2 = np.random.uniform(0, 10, 40)
        # Create interaction effect
        y = 2 * x1 + 3 * x2 + 0.5 * x1 * x2 + np.random.normal(0, 2, 40)
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        # Allow interactions
        spec = mars(prod_degree=2, num_terms=15)
        fit = spec.fit(data, "y ~ x1 + x2")

        # Model should capture interaction
        outputs, _, stats = fit.extract_outputs()

        # Check R-squared is decent
        r2_row = stats[stats["metric"] == "r_squared"]
        if not r2_row.empty:
            r2 = r2_row["value"].iloc[0]
            assert r2 > 0.5  # Should explain a good portion of variance


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self):
        """Test complete fit → predict → extract workflow"""
        np.random.seed(42)
        # Training data with non-linearity
        x = np.linspace(0, 10, 60)
        y = 2 * x + 0.5 * x ** 2 + np.random.normal(0, 3, 60)
        train = pd.DataFrame({"y": y, "x": x})

        # Create spec and fit
        spec = mars(num_terms=10)
        fit = spec.fit(train, "y ~ x")

        # Test data
        test = pd.DataFrame({"x": [2, 5, 8]})

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)

        # Extract outputs
        outputs, basis_funcs, stats = fit.extract_outputs()
        assert len(outputs) == 60  # Training data size
        assert len(basis_funcs) >= 1
        assert len(stats) > 0

    def test_evaluate_and_extract(self):
        """Test evaluate() and extract_outputs() with test data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 70)
        y = x ** 2 + np.random.normal(0, 5, 70)
        data = pd.DataFrame({"y": y, "x": x})

        # Split data
        train = data.iloc[:50]
        test = data.iloc[50:]

        spec = mars(num_terms=10)
        fit = spec.fit(train, "y ~ x")
        fit = fit.evaluate(test)

        outputs, basis_funcs, stats = fit.extract_outputs()

        # Outputs should have both train and test splits
        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert len(outputs[outputs["split"] == "train"]) == 50
        assert len(outputs[outputs["split"] == "test"]) == 20

        # Stats should have both train and test metrics
        train_stats = stats[stats["split"] == "train"]
        test_stats = stats[stats["split"] == "test"]
        assert len(train_stats) > 0
        assert len(test_stats) > 0

        # Both should have RMSE
        assert "rmse" in train_stats["metric"].values
        assert "rmse" in test_stats["metric"].values
