"""
Tests for Poisson regression model

Tests cover:
- Model specification creation
- Engine registration
- Fitting with formula
- Prediction (numeric and conf_int)
- Extract outputs
- Count data validation
- GLM statistics
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import poisson_reg, ModelSpec, ModelFit


class TestPoissonRegSpec:
    """Test poisson_reg() model specification"""

    def test_default_spec(self):
        """Test default Poisson regression specification"""
        spec = poisson_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "poisson_reg"
        assert spec.engine == "statsmodels"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_penalty(self):
        """Test Poisson with penalty parameter"""
        spec = poisson_reg(penalty=0.1)

        assert spec.args == {"penalty": 0.1}

    def test_spec_with_penalty_and_mixture(self):
        """Test Poisson with penalty and mixture"""
        spec = poisson_reg(penalty=0.1, mixture=0.5)

        assert spec.args == {"penalty": 0.1, "mixture": 0.5}

    def test_set_args(self):
        """Test set_args() method"""
        spec = poisson_reg()
        spec = spec.set_args(penalty=0.2)

        assert spec.args == {"penalty": 0.2}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = poisson_reg(penalty=0.1)
        spec2 = spec1.set_args(penalty=0.2)

        # Original spec should be unchanged
        assert spec1.args == {"penalty": 0.1}
        # New spec should have new value
        assert spec2.args == {"penalty": 0.2}


class TestPoissonRegFit:
    """Test Poisson regression fitting with statsmodels engine"""

    @pytest.fixture
    def count_data(self):
        """Create count data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            "count": [0, 1, 2, 3, 1, 0, 2, 4, 2, 1, 3, 5, 2, 1, 0],
            "x1": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 4.5, 3.0, 2.0, 4.0, 5.0, 3.2, 2.2, 1.8],
            "x2": [0.5, 1.5, 2.5, 3.5, 2.0, 1.0, 3.0, 4.0, 2.8, 1.8, 3.8, 4.5, 3.0, 2.0, 1.2],
        })

    def test_fit_with_formula(self, count_data):
        """Test fitting with formula"""
        spec = poisson_reg()
        fit = spec.fit(count_data, "count ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert "results" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_basic(self, count_data):
        """Test basic Poisson fit"""
        spec = poisson_reg()
        fit = spec.fit(count_data, "count ~ x1 + x2")

        # Should have fitted GLM results
        assert "results" in fit.fit_data
        assert "aic" in fit.fit_data
        assert "bic" in fit.fit_data

    def test_fit_rejects_penalty(self, count_data):
        """Test that statsmodels rejects regularization"""
        spec = poisson_reg(penalty=0.1)

        with pytest.raises(ValueError, match="does not support regularization"):
            spec.fit(count_data, "count ~ x1 + x2")

    def test_fit_rejects_negative_counts(self):
        """Test that Poisson requires non-negative outcomes"""
        bad_data = pd.DataFrame({
            "count": [-1, 0, 1, 2],
            "x": [1, 2, 3, 4],
        })

        spec = poisson_reg()

        with pytest.raises(ValueError, match="non-negative count outcomes"):
            spec.fit(bad_data, "count ~ x")


class TestPoissonRegPredict:
    """Test Poisson regression prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        train = pd.DataFrame({
            "count": [0, 1, 2, 3, 1, 0, 2, 4, 2, 1, 3, 5],
            "x1": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 4.5, 3.0, 2.0, 4.0, 5.0],
            "x2": [0.5, 1.5, 2.5, 3.5, 2.0, 1.0, 3.0, 4.0, 2.8, 1.8, 3.8, 4.5],
        })

        spec = poisson_reg()
        fit = spec.fit(train, "count ~ x1 + x2")
        return fit

    def test_predict_basic(self, fitted_model):
        """Test basic prediction"""
        test = pd.DataFrame({
            "x1": [2.5, 3.5, 4.5],
            "x2": [2.0, 3.0, 4.0],
        })

        predictions = fitted_model.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3
        # Predictions should be non-negative
        assert all(predictions[".pred"] >= 0)

    def test_predict_type_numeric(self, fitted_model):
        """Test numeric prediction type"""
        test = pd.DataFrame({
            "x1": [2.5, 3.5],
            "x2": [2.0, 3.0],
        })

        predictions = fitted_model.predict(test, type="numeric")
        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_predict_type_conf_int(self, fitted_model):
        """Test predictions with confidence intervals"""
        test = pd.DataFrame({
            "x1": [2.5, 3.5],
            "x2": [2.0, 3.0],
        })

        predictions = fitted_model.predict(test, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert len(predictions) == 2
        # Lower bound should be less than prediction
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        # Upper bound should be greater than prediction
        assert all(predictions[".pred_upper"] >= predictions[".pred"])

    def test_predict_invalid_type(self, fitted_model):
        """Test that invalid prediction type raises error"""
        test = pd.DataFrame({
            "x1": [2.5],
            "x2": [2.0],
        })

        with pytest.raises(ValueError, match="supports type='numeric' or 'conf_int'"):
            fitted_model.predict(test, type="prob")


class TestPoissonRegExtract:
    """Test Poisson regression output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        train = pd.DataFrame({
            "count": [0, 1, 2, 3, 1, 0, 2, 4, 2, 1, 3, 5],
            "x1": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 4.5, 3.0, 2.0, 4.0, 5.0],
            "x2": [0.5, 1.5, 2.5, 3.5, 2.0, 1.0, 3.0, 4.0, 2.8, 1.8, 3.8, 4.5],
        })

        spec = poisson_reg()
        fit = spec.fit(train, "count ~ x1 + x2")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        sm_model = fitted_model.extract_fit_engine()

        assert sm_model is not None
        # Check for statsmodels GLM model attributes
        # extract_fit_engine returns the model, not results
        assert hasattr(sm_model, "fit") or hasattr(sm_model, "family")

    def test_extract_outputs(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_model.extract_outputs()

        # Check all three DataFrames exist
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
        assert len(outputs) == 12

        # Check for Poisson-specific residuals
        assert "pearson_resid" in outputs.columns
        assert "deviance_resid" in outputs.columns

    def test_extract_outputs_coefficients(self, fitted_model):
        """Test Coefficients DataFrame structure"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Check for coefficient columns
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns
        assert "std_error" in coefs.columns
        assert "z_stat" in coefs.columns  # GLM uses z-statistic
        assert "p_value" in coefs.columns
        assert "ci_0.025" in coefs.columns
        assert "ci_0.975" in coefs.columns

        # Should have intercept + x1 + x2 = 3 coefficients
        assert len(coefs) == 3
        assert "Intercept" in coefs["variable"].values

        # All inference should be non-NaN (statsmodels provides them)
        assert all(coefs["std_error"].notna())
        assert all(coefs["z_stat"].notna())
        assert all(coefs["p_value"].notna())

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame includes GLM statistics"""
        _, _, stats = fitted_model.extract_outputs()

        # Check for stats columns
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check for key metrics
        stat_names = stats["metric"].tolist()
        assert "rmse" in stat_names
        assert "mae" in stat_names
        assert "poisson_deviance" in stat_names or "deviance" in stat_names

        # Check for GLM-specific stats
        assert "aic" in stat_names
        assert "bic" in stat_names
        assert "deviance" in stat_names
        assert "log_likelihood" in stat_names


class TestPoissonRegCountData:
    """Test Poisson regression with realistic count data"""

    def test_basic_count_model(self):
        """Test Poisson regression on count outcomes"""
        np.random.seed(42)
        # Simulate count data (e.g., number of events)
        x = np.random.uniform(0, 5, 30)
        # Generate counts from Poisson distribution
        lambda_true = np.exp(0.5 + 0.3 * x)
        counts = np.random.poisson(lambda_true)

        data = pd.DataFrame({"count": counts, "x": x})

        spec = poisson_reg()
        fit = spec.fit(data, "count ~ x")

        # Predict
        test = pd.DataFrame({"x": [1.0, 2.5, 4.0]})
        predictions = fit.predict(test)

        # Predictions should be positive and increasing with x
        assert all(predictions[".pred"] > 0)
        assert predictions[".pred"].iloc[0] < predictions[".pred"].iloc[2]

    def test_zero_inflated_data(self):
        """Test Poisson with many zeros"""
        np.random.seed(42)
        data = pd.DataFrame({
            "count": [0, 0, 0, 1, 0, 2, 0, 1, 0, 3, 1, 0],
            "x": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5],
        })

        spec = poisson_reg()
        fit = spec.fit(data, "count ~ x")

        # Should handle zeros gracefully
        assert isinstance(fit, ModelFit)

        # Extract stats
        _, _, stats = fit.extract_outputs()
        assert len(stats) > 0


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self):
        """Test complete fit → predict → extract workflow"""
        np.random.seed(42)
        # Create count data
        x1 = np.random.uniform(0, 5, 40)
        x2 = np.random.uniform(0, 3, 40)
        lambda_true = np.exp(0.2 + 0.3 * x1 + 0.5 * x2)
        counts = np.random.poisson(lambda_true)
        train = pd.DataFrame({"count": counts, "x1": x1, "x2": x2})

        # Create spec and fit
        spec = poisson_reg()
        fit = spec.fit(train, "count ~ x1 + x2")

        # Test data
        test = pd.DataFrame({
            "x1": [1.5, 3.0, 4.5],
            "x2": [1.0, 2.0, 2.5],
        })

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] >= 0)

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()
        assert len(outputs) == 40  # Training data size
        assert len(coefs) == 3  # Intercept + x1 + x2
        assert len(stats) > 0

    def test_evaluate_and_extract(self):
        """Test evaluate() and extract_outputs() with test data"""
        np.random.seed(42)
        x = np.random.uniform(0, 5, 50)
        lambda_true = np.exp(0.5 + 0.4 * x)
        counts = np.random.poisson(lambda_true)
        data = pd.DataFrame({"count": counts, "x": x})

        # Split data
        train = data.iloc[:35]
        test = data.iloc[35:]

        spec = poisson_reg()
        fit = spec.fit(train, "count ~ x")
        fit = fit.evaluate(test)

        outputs, coefs, stats = fit.extract_outputs()

        # Outputs should have both train and test splits
        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert len(outputs[outputs["split"] == "train"]) == 35
        assert len(outputs[outputs["split"] == "test"]) == 15

        # Stats should have both train and test metrics
        train_stats = stats[stats["split"] == "train"]
        test_stats = stats[stats["split"] == "test"]
        assert len(train_stats) > 0
        assert len(test_stats) > 0

        # Both should have RMSE
        assert "rmse" in train_stats["metric"].values
        assert "rmse" in test_stats["metric"].values
