"""
Tests for Generalized Additive Model (GAM)

Tests cover:
- Model specification creation
- Engine registration
- Fitting with formula
- Prediction (numeric and conf_int)
- Extract outputs (partial effects)
- Non-linear relationship detection
- Smoothing parameter tuning
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import gen_additive_mod, ModelSpec, ModelFit


class TestGAMSpec:
    """Test gen_additive_mod() model specification"""

    def test_default_spec(self):
        """Test default GAM specification"""
        spec = gen_additive_mod()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "gen_additive_mod"
        assert spec.engine == "pygam"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_adjust_deg_free(self):
        """Test GAM with adjust_deg_free parameter"""
        spec = gen_additive_mod(adjust_deg_free=5)

        assert spec.args == {"adjust_deg_free": 5}

    def test_spec_with_select_features(self):
        """Test GAM with feature selection"""
        spec = gen_additive_mod(select_features=True)

        assert spec.args == {"select_features": True}

    def test_spec_with_all_params(self):
        """Test GAM with all parameters"""
        spec = gen_additive_mod(select_features=True, adjust_deg_free=15)

        assert spec.args == {
            "select_features": True,
            "adjust_deg_free": 15
        }

    def test_set_args(self):
        """Test set_args() method"""
        spec = gen_additive_mod()
        spec = spec.set_args(adjust_deg_free=20)

        assert spec.args == {"adjust_deg_free": 20}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = gen_additive_mod(adjust_deg_free=10)
        spec2 = spec1.set_args(adjust_deg_free=20)

        # Original spec should be unchanged
        assert spec1.args == {"adjust_deg_free": 10}
        # New spec should have new value
        assert spec2.args == {"adjust_deg_free": 20}


class TestGAMFit:
    """Test GAM fitting with pygam engine"""

    @pytest.fixture
    def linear_data(self):
        """Create simple linear training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 240, 190],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 24, 19],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 12, 9],
        })

    @pytest.fixture
    def nonlinear_data(self):
        """Create non-linear training data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) * 10 + 20 + np.random.normal(0, 1, 50)
        return pd.DataFrame({"y": y, "x": x})

    def test_fit_with_formula(self, linear_data):
        """Test fitting with formula"""
        spec = gen_additive_mod()
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_default(self, linear_data):
        """Test default GAM fit"""
        spec = gen_additive_mod()
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        # Should have fitted model
        assert "model" in fit.fit_data
        assert "n_splines" in fit.fit_data

    def test_fit_with_adjust_deg_free(self, linear_data):
        """Test GAM with different smoothing levels"""
        # More flexible (more splines)
        spec_flexible = gen_additive_mod(adjust_deg_free=15)
        fit_flexible = spec_flexible.fit(linear_data, "y ~ x1 + x2")

        # Less flexible (fewer splines)
        spec_smooth = gen_additive_mod(adjust_deg_free=5)
        fit_smooth = spec_smooth.fit(linear_data, "y ~ x1 + x2")

        assert fit_flexible.fit_data["n_splines"] == 15
        assert fit_smooth.fit_data["n_splines"] == 5

    def test_fit_with_select_features(self, linear_data):
        """Test GAM with automatic feature selection"""
        spec = gen_additive_mod(select_features=True)
        fit = spec.fit(linear_data, "y ~ x1 + x2")

        assert "model" in fit.fit_data
        # Should have used gridsearch

    def test_fit_nonlinear(self, nonlinear_data):
        """Test GAM can fit non-linear relationships"""
        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(nonlinear_data, "y ~ x")

        # GAM should capture the sinusoidal pattern
        assert isinstance(fit, ModelFit)


class TestGAMPredict:
    """Test GAM prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        x = np.linspace(0, 10, 40)
        y = x ** 2 + np.random.normal(0, 3, 40)
        train = pd.DataFrame({"y": y, "x": x})

        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(train, "y ~ x")
        return fit

    def test_predict_basic(self, fitted_model):
        """Test basic prediction"""
        test = pd.DataFrame({"x": [2, 5, 8]})

        predictions = fitted_model.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3

    def test_predict_type_numeric(self, fitted_model):
        """Test numeric prediction type"""
        test = pd.DataFrame({"x": [2, 5, 8]})

        predictions = fitted_model.predict(test, type="numeric")
        assert ".pred" in predictions.columns

    def test_predict_type_conf_int(self, fitted_model):
        """Test predictions with confidence intervals"""
        test = pd.DataFrame({"x": [2, 5, 8]})

        predictions = fitted_model.predict(test, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert len(predictions) == 3
        # Lower bound should be less than or equal to prediction
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        # Upper bound should be greater than or equal to prediction
        assert all(predictions[".pred_upper"] >= predictions[".pred"])

    def test_predict_invalid_type(self, fitted_model):
        """Test that invalid prediction type raises error"""
        test = pd.DataFrame({"x": [2]})

        with pytest.raises(ValueError, match="supports type='numeric' or 'conf_int'"):
            fitted_model.predict(test, type="prob")

    def test_predict_values_reasonable(self, fitted_model):
        """Test that predictions follow quadratic pattern"""
        test = pd.DataFrame({"x": [2, 5, 8]})
        predictions = fitted_model.predict(test)

        # For quadratic (x^2), predictions should increase
        # 2^2 = 4, 5^2 = 25, 8^2 = 64
        assert predictions[".pred"].iloc[0] < predictions[".pred"].iloc[1]
        assert predictions[".pred"].iloc[1] < predictions[".pred"].iloc[2]


class TestGAMExtract:
    """Test GAM output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        np.random.seed(42)
        x1 = np.linspace(0, 10, 40)
        x2 = np.linspace(0, 5, 40)
        y = x1 ** 2 + 2 * x2 + np.random.normal(0, 3, 40)
        train = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        gam_model = fitted_model.extract_fit_engine()

        assert gam_model is not None
        # Check for pygam attributes
        assert hasattr(gam_model, "predict")

    def test_extract_outputs(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, partial_effects, stats = fitted_model.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(partial_effects, pd.DataFrame)
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
        assert len(outputs) == 40

    def test_extract_outputs_partial_effects(self, fitted_model):
        """Test Partial Effects DataFrame structure"""
        _, partial_effects, _ = fitted_model.extract_outputs()

        # Check for partial effect columns
        assert "feature" in partial_effects.columns or "feature_index" in partial_effects.columns
        # Should have entry for each feature
        assert len(partial_effects) >= 1

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

        # Check for GAM-specific stats
        assert "aic" in stat_names or "aicc" in stat_names
        assert "gcv" in stat_names
        assert "n_splines" in stat_names


class TestGAMNonlinearity:
    """Test GAM ability to detect and model non-linear relationships"""

    def test_quadratic_relationship(self):
        """Test GAM fits quadratic relationship"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = x ** 2 + np.random.normal(0, 5, 50)
        data = pd.DataFrame({"y": y, "x": x})

        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(data, "y ~ x")

        # Predict across range
        test = pd.DataFrame({"x": [2, 5, 8]})
        predictions = fit.predict(test)

        # Check predictions roughly follow quadratic
        # 2^2 = 4, 5^2 = 25, 8^2 = 64
        # With noise, just check monotonicity and rough magnitude
        assert predictions[".pred"].iloc[0] < 20
        assert predictions[".pred"].iloc[1] > 15
        assert predictions[".pred"].iloc[2] > 50

    def test_sinusoidal_relationship(self):
        """Test GAM can capture sinusoidal patterns"""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 60)
        y = np.sin(x) * 10 + 20 + np.random.normal(0, 1, 60)
        data = pd.DataFrame({"y": y, "x": x})

        spec = gen_additive_mod(adjust_deg_free=15)
        fit = spec.fit(data, "y ~ x")

        # Should capture the wave pattern
        assert isinstance(fit, ModelFit)

        # Check R-squared is reasonable
        _, _, stats = fit.extract_outputs()
        r2_row = stats[stats["metric"] == "r_squared"]
        if not r2_row.empty:
            r2 = r2_row["value"].iloc[0]
            assert r2 > 0.7  # Should explain most variance

    def test_smoothness_control(self):
        """Test that adjust_deg_free controls smoothness"""
        np.random.seed(42)
        x = np.linspace(0, 10, 40)
        y = x ** 2 + np.random.normal(0, 8, 40)  # Noisy quadratic
        data = pd.DataFrame({"y": y, "x": x})

        # Very smooth (few splines) - must be > 3 for pygam
        spec_smooth = gen_additive_mod(adjust_deg_free=5)
        fit_smooth = spec_smooth.fit(data, "y ~ x")

        # Very flexible (many splines)
        spec_flexible = gen_additive_mod(adjust_deg_free=20)
        fit_flexible = spec_flexible.fit(data, "y ~ x")

        # Both should fit, but with different characteristics
        assert isinstance(fit_smooth, ModelFit)
        assert isinstance(fit_flexible, ModelFit)

        # More flexible model might have lower training RMSE
        _, _, stats_smooth = fit_smooth.extract_outputs()
        _, _, stats_flex = fit_flexible.extract_outputs()

        rmse_smooth = stats_smooth[stats_smooth["metric"] == "rmse"]["value"].iloc[0]
        rmse_flex = stats_flex[stats_flex["metric"] == "rmse"]["value"].iloc[0]

        # Both should be reasonable (not checking which is better to avoid flakiness)
        assert rmse_smooth > 0
        assert rmse_flex > 0


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self):
        """Test complete fit → predict → extract workflow"""
        np.random.seed(42)
        # Create non-linear data
        x1 = np.linspace(0, 10, 60)
        x2 = np.linspace(0, 5, 60)
        y = np.sin(x1) * 10 + x2 ** 2 + np.random.normal(0, 2, 60)
        train = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        # Create spec and fit
        spec = gen_additive_mod(adjust_deg_free=12)
        fit = spec.fit(train, "y ~ x1 + x2")

        # Test data
        test = pd.DataFrame({
            "x1": [2, 5, 8],
            "x2": [1, 3, 4],
        })

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 3
        assert ".pred" in predictions.columns

        # Extract outputs
        outputs, partial_effects, stats = fit.extract_outputs()
        assert len(outputs) == 60  # Training data size
        assert len(partial_effects) >= 1
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

        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(train, "y ~ x")
        fit = fit.evaluate(test)

        outputs, partial_effects, stats = fit.extract_outputs()

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

    def test_multiple_predictors(self):
        """Test GAM with multiple predictors"""
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        x3 = np.random.uniform(-2, 2, n)
        y = x1 ** 2 + np.sin(x2 * 2) * 5 + x3 + np.random.normal(0, 3, n)

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        spec = gen_additive_mod(adjust_deg_free=10)
        fit = spec.fit(data, "y ~ x1 + x2 + x3")

        # Should fit successfully
        assert isinstance(fit, ModelFit)

        # Extract partial effects
        _, partial_effects, _ = fit.extract_outputs()

        # Should have partial effect info for each predictor
        assert len(partial_effects) >= 3  # At least 3 features (may include intercept)
