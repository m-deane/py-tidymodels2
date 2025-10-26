"""
Tests for linear_reg model specification and sklearn engine

Tests cover:
- Model specification creation
- Engine registration
- Fitting with formula
- Prediction
- Extract outputs
- Parameter translation
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import linear_reg, ModelSpec, ModelFit


class TestLinearRegSpec:
    """Test linear_reg() model specification"""

    def test_default_spec(self):
        """Test default linear_reg specification"""
        spec = linear_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "linear_reg"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_penalty(self):
        """Test linear_reg with penalty (Ridge)"""
        spec = linear_reg(penalty=0.1)

        assert spec.args == {"penalty": 0.1}

    def test_spec_with_penalty_and_mixture(self):
        """Test linear_reg with penalty and mixture (ElasticNet)"""
        spec = linear_reg(penalty=0.1, mixture=0.5)

        assert spec.args == {"penalty": 0.1, "mixture": 0.5}

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = linear_reg()
        spec = spec.set_engine("statsmodels")

        assert spec.engine == "statsmodels"

    def test_set_args(self):
        """Test set_args() method"""
        spec = linear_reg()
        spec = spec.set_args(penalty=0.2, mixture=0.3)

        assert spec.args == {"penalty": 0.2, "mixture": 0.3}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = linear_reg(penalty=0.1)
        spec2 = spec1.set_args(penalty=0.2)

        # Original spec should be unchanged
        assert spec1.args == {"penalty": 0.1}
        # New spec should have new value
        assert spec2.args == {"penalty": 0.2}


class TestLinearRegFit:
    """Test linear_reg fitting with sklearn engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250],
            "x1": [10, 20, 15, 30, 25],
            "x2": [5, 10, 7, 15, 12],
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = linear_reg()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_ols(self, train_data):
        """Test OLS (no penalty)"""
        spec = linear_reg()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should use LinearRegression
        assert fit.fit_data["model_class"] == "LinearRegression"

    def test_fit_ridge(self, train_data):
        """Test Ridge regression (L2 penalty)"""
        spec = linear_reg(penalty=0.1, mixture=0.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should use Ridge
        assert fit.fit_data["model_class"] == "Ridge"

    def test_fit_lasso(self, train_data):
        """Test Lasso regression (L1 penalty)"""
        spec = linear_reg(penalty=0.1, mixture=1.0)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should use Lasso
        assert fit.fit_data["model_class"] == "Lasso"

    def test_fit_elasticnet(self, train_data):
        """Test ElasticNet (mixed penalty)"""
        spec = linear_reg(penalty=0.1, mixture=0.5)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should use ElasticNet
        assert fit.fit_data["model_class"] == "ElasticNet"


class TestLinearRegPredict:
    """Test linear_reg prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250],
            "x1": [10, 20, 15, 30, 25],
            "x2": [5, 10, 7, 15, 12],
        })

        spec = linear_reg()
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

        # For x1=15, x2=7 (middle values), prediction should be ~150
        # Allow wide range since this is a small sample
        assert 50 < predictions[".pred"].iloc[0] < 350


class TestLinearRegExtract:
    """Test linear_reg output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250],
            "x1": [10, 20, 15, 30, 25],
            "x2": [5, 10, 7, 15, 12],
        })

        spec = linear_reg()
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        sklearn_model = fitted_model.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "coef_")
        assert hasattr(sklearn_model, "intercept_")

    def test_extract_outputs(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_model.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_model_outputs(self, fitted_model):
        """Test Outputs DataFrame structure (observation-level results)"""
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
        assert len(outputs) == 5

    def test_extract_outputs_coefficients(self, fitted_model):
        """Test Coefficients DataFrame structure"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Check for new column names
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns
        assert "std_error" in coefs.columns
        assert "p_value" in coefs.columns
        assert "vif" in coefs.columns
        # Should have intercept + x1 + x2 = 3 coefficients
        assert len(coefs) == 3
        assert "Intercept" in coefs["variable"].values


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self):
        """Test complete mold → fit → forge → predict workflow"""
        # Training data
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180],
            "price": [10, 20, 15, 30, 25, 18],
            "advertising": [5, 10, 7, 15, 12, 9],
        })

        # Create spec and fit
        spec = linear_reg(penalty=0.1)
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

    def test_categorical_variables(self):
        """Test with categorical predictors"""
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180],
            "price": [10, 20, 15, 30, 25, 18],
            "region": ["A", "B", "A", "C", "B", "C"],
        })

        spec = linear_reg()
        fit = spec.fit(train, "sales ~ price + region")

        # Test with valid regions
        test = pd.DataFrame({
            "price": [12, 22],
            "region": ["A", "B"],
        })

        predictions = fit.predict(test)
        assert len(predictions) == 2


class TestStatsmodelsEngine:
    """Test linear_reg with statsmodels engine (OLS)"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220],
            "x1": [10, 20, 15, 30, 25, 18, 22],
            "x2": [5, 10, 7, 15, 12, 9, 11],
        })

    def test_fit_with_statsmodels(self, train_data):
        """Test fitting with statsmodels engine"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec.engine == "statsmodels"
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_statsmodels_rejects_penalty(self, train_data):
        """Test that statsmodels rejects regularization"""
        spec = linear_reg(penalty=0.1).set_engine("statsmodels")

        with pytest.raises(ValueError, match="does not support regularization"):
            spec.fit(train_data, "y ~ x1 + x2")

    def test_predict_numeric(self, train_data):
        """Test numeric predictions"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        test = pd.DataFrame({
            "x1": [12, 22, 28],
            "x2": [6, 11, 14],
        })

        predictions = fit.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3
        assert all(predictions[".pred"].notna())

    def test_predict_conf_int(self, train_data):
        """Test predictions with confidence intervals"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fit.predict(test, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert len(predictions) == 2
        # Lower bound should be less than prediction
        assert all(predictions[".pred_lower"] < predictions[".pred"])
        # Upper bound should be greater than prediction
        assert all(predictions[".pred_upper"] > predictions[".pred"])

    def test_extract_outputs_with_full_inference(self, train_data):
        """Test extract_outputs returns full statistical inference"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert len(outputs) == 7  # Same as training data

        # Check coefficients DataFrame has full inference
        assert isinstance(coefficients, pd.DataFrame)
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns
        assert "std_error" in coefficients.columns
        assert "t_stat" in coefficients.columns
        assert "p_value" in coefficients.columns
        assert "ci_0.025" in coefficients.columns
        assert "ci_0.975" in coefficients.columns
        assert "vif" in coefficients.columns

        # All statistical inference should be non-NaN (statsmodels provides them)
        assert all(coefficients["std_error"].notna())
        assert all(coefficients["t_stat"].notna())
        assert all(coefficients["p_value"].notna())
        assert all(coefficients["ci_0.025"].notna())
        assert all(coefficients["ci_0.975"].notna())

        # Check stats DataFrame includes statsmodels-specific metrics
        assert isinstance(stats, pd.DataFrame)
        assert "metric" in stats.columns
        assert "value" in stats.columns

        # Check for statsmodels-specific stats
        stat_names = stats["metric"].tolist()
        assert "aic" in stat_names
        assert "bic" in stat_names
        assert "f_statistic" in stat_names
        assert "f_pvalue" in stat_names
        assert "log_likelihood" in stat_names
        assert "condition_number" in stat_names

    def test_residual_diagnostics(self, train_data):
        """Test that statsmodels engine provides enhanced residual diagnostics"""
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        _, _, stats = fit.extract_outputs()

        stat_names = stats["metric"].tolist()

        # Should include enhanced diagnostics from statsmodels
        assert "durbin_watson" in stat_names
        assert "shapiro_wilk_stat" in stat_names
        assert "shapiro_wilk_p" in stat_names
        assert "ljung_box_stat" in stat_names
        assert "ljung_box_p" in stat_names
        assert "breusch_pagan_stat" in stat_names
        assert "breusch_pagan_p" in stat_names

    def test_evaluate_and_extract(self, train_data):
        """Test evaluate() and extract_outputs() with test data"""
        # Split data
        train = train_data.iloc[:5]
        test = train_data.iloc[5:]

        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test)

        outputs, coefficients, stats = fit.extract_outputs()

        # Outputs should have both train and test splits
        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert len(outputs[outputs["split"] == "train"]) == 5
        assert len(outputs[outputs["split"] == "test"]) == 2

        # Stats should have both train and test metrics
        train_stats = stats[stats["split"] == "train"]
        test_stats = stats[stats["split"] == "test"]
        assert len(train_stats) > 0
        assert len(test_stats) > 0

        # Check that both have RMSE
        assert "rmse" in train_stats["metric"].values
        assert "rmse" in test_stats["metric"].values
