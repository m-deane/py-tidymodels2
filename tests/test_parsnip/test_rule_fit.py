"""
Tests for rule_fit model specification and imodels engine

Tests cover:
- Model specification creation
- Parameter translation
- Fitting with formula (regression)
- Fitting with formula (classification)
- Prediction (numeric, class, prob)
- Extract outputs (outputs, coefficients as rules, stats)
- Rule extraction and interpretability
- Evaluate() method with test data
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import rule_fit, ModelSpec, ModelFit


class TestRuleFitSpec:
    """Test rule_fit() model specification"""

    def test_default_spec(self):
        """Test default rule_fit specification"""
        spec = rule_fit()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "rule_fit"
        assert spec.engine == "imodels"
        assert spec.mode == "unknown"
        assert spec.args == {}

    def test_spec_with_max_rules(self):
        """Test rule_fit with max_rules parameter"""
        spec = rule_fit(max_rules=20)

        assert spec.args == {"max_rules": 20}

    def test_spec_with_tree_depth(self):
        """Test rule_fit with tree_depth parameter"""
        spec = rule_fit(tree_depth=5)

        assert spec.args == {"tree_depth": 5}

    def test_spec_with_penalty(self):
        """Test rule_fit with penalty parameter"""
        spec = rule_fit(penalty=0.01)

        assert spec.args == {"penalty": 0.01}

    def test_spec_with_tree_generator(self):
        """Test rule_fit with tree_generator parameter"""
        spec = rule_fit(tree_generator="boosting")

        assert spec.args == {"tree_generator": "boosting"}

    def test_spec_with_all_parameters(self):
        """Test rule_fit with all parameters"""
        spec = rule_fit(max_rules=15, tree_depth=4, penalty=0.001, tree_generator="rf")

        assert spec.args == {
            "max_rules": 15,
            "tree_depth": 4,
            "penalty": 0.001,
            "tree_generator": "rf"
        }

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = rule_fit()
        spec = spec.set_engine("imodels")

        assert spec.engine == "imodels"

    def test_set_mode_regression(self):
        """Test set_mode() for regression"""
        spec = rule_fit()
        spec = spec.set_mode("regression")

        assert spec.mode == "regression"

    def test_set_mode_classification(self):
        """Test set_mode() for classification"""
        spec = rule_fit()
        spec = spec.set_mode("classification")

        assert spec.mode == "classification"

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = rule_fit(max_rules=10)
        spec2 = spec1.set_args(max_rules=20)

        # Original spec should be unchanged
        assert spec1.args == {"max_rules": 10}
        # New spec should have new value
        assert spec2.args == {"max_rules": 20}


class TestRuleFitFitRegression:
    """Test rule_fit fitting for regression"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data for regression"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "y": np.random.randn(n) * 10 + 50,
            "x1": np.random.randn(n) * 5,
            "x2": np.random.randn(n) * 3,
            "x3": np.random.randn(n) * 2,
            "x4": np.random.choice([1, 2, 3], n),
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec.mode == "regression"
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_model_class(self, train_data):
        """Test fitted model class"""
        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "RuleFitRegressor"

    def test_fit_stores_training_data(self, train_data):
        """Test that fit stores training data"""
        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert "X_train" in fit.fit_data
        assert "y_train" in fit.fit_data
        assert "fitted" in fit.fit_data
        assert "residuals" in fit.fit_data

    def test_fit_with_max_rules(self, train_data):
        """Test fitting with max_rules parameter"""
        spec = rule_fit(max_rules=15).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert fit.fit_data["model"].max_rules == 15

    def test_fit_with_tree_depth(self, train_data):
        """Test fitting with tree_depth parameter"""
        spec = rule_fit(tree_depth=5).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model"].tree_size == 5

    def test_fit_with_penalty(self, train_data):
        """Test fitting with penalty parameter"""
        spec = rule_fit(penalty=0.01).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model"].alpha == 0.01


class TestRuleFitFitClassification:
    """Test rule_fit fitting for classification"""

    @pytest.fixture
    def train_data_class(self):
        """Create sample training data for classification"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "y": np.random.choice([0, 1], n),
            "x1": np.random.randn(n) * 5,
            "x2": np.random.randn(n) * 3,
            "x3": np.random.randn(n) * 2,
        })

    def test_fit_classification(self, train_data_class):
        """Test fitting for classification"""
        spec = rule_fit().set_mode("classification")
        fit = spec.fit(train_data_class, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec.mode == "classification"
        assert fit.fit_data["model_class"] == "RuleFitClassifier"

    def test_fit_classification_stores_data(self, train_data_class):
        """Test that classification fit stores data"""
        spec = rule_fit().set_mode("classification")
        fit = spec.fit(train_data_class, "y ~ x1 + x2")

        assert "X_train" in fit.fit_data
        assert "y_train" in fit.fit_data
        assert "fitted" in fit.fit_data
        # Classification doesn't have residuals
        assert fit.fit_data["residuals"] is None


class TestRuleFitPredict:
    """Test rule_fit predictions"""

    @pytest.fixture
    def fitted_regression(self):
        """Create fitted regression model"""
        np.random.seed(42)
        n = 100
        train_data = pd.DataFrame({
            "y": np.random.randn(n) * 10 + 50,
            "x1": np.random.randn(n) * 5,
            "x2": np.random.randn(n) * 3,
        })
        spec = rule_fit().set_mode("regression")
        return spec.fit(train_data, "y ~ x1 + x2")

    @pytest.fixture
    def fitted_classification(self):
        """Create fitted classification model"""
        np.random.seed(42)
        n = 100
        train_data = pd.DataFrame({
            "y": np.random.choice([0, 1], n),
            "x1": np.random.randn(n) * 5,
            "x2": np.random.randn(n) * 3,
        })
        spec = rule_fit().set_mode("classification")
        return spec.fit(train_data, "y ~ x1 + x2")

    def test_predict_regression_numeric(self, fitted_regression):
        """Test numeric predictions for regression"""
        np.random.seed(43)
        test_data = pd.DataFrame({
            "x1": np.random.randn(10) * 5,
            "x2": np.random.randn(10) * 3,
        })

        predictions = fitted_regression.predict(test_data, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 10
        assert predictions[".pred"].dtype in [np.float64, np.float32]

    def test_predict_classification_class(self, fitted_classification):
        """Test class predictions for classification"""
        np.random.seed(43)
        test_data = pd.DataFrame({
            "x1": np.random.randn(10) * 5,
            "x2": np.random.randn(10) * 3,
        })

        predictions = fitted_classification.predict(test_data, type="class")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred_class" in predictions.columns
        assert len(predictions) == 10
        assert all(predictions[".pred_class"].isin([0, 1]))

    def test_predict_classification_prob(self, fitted_classification):
        """Test probability predictions for classification"""
        np.random.seed(43)
        test_data = pd.DataFrame({
            "x1": np.random.randn(10) * 5,
            "x2": np.random.randn(10) * 3,
        })

        predictions = fitted_classification.predict(test_data, type="prob")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred_0" in predictions.columns
        assert ".pred_1" in predictions.columns
        assert len(predictions) == 10
        # Probabilities should sum to 1
        assert np.allclose(predictions[[".pred_0", ".pred_1"]].sum(axis=1), 1.0)

    def test_predict_invalid_type_regression(self, fitted_regression):
        """Test error for invalid prediction type on regression"""
        test_data = pd.DataFrame({"x1": [1.0], "x2": [2.0]})

        with pytest.raises(ValueError, match="only valid for classification"):
            fitted_regression.predict(test_data, type="class")

    def test_predict_invalid_type_classification(self, fitted_classification):
        """Test error for invalid prediction type on classification"""
        test_data = pd.DataFrame({"x1": [1.0], "x2": [2.0]})

        with pytest.raises(ValueError, match="only valid for regression"):
            fitted_classification.predict(test_data, type="numeric")

    def test_predict_conf_int_not_supported(self, fitted_regression):
        """Test that confidence intervals are not supported"""
        test_data = pd.DataFrame({"x1": [1.0], "x2": [2.0]})

        with pytest.raises(ValueError, match="does not support confidence intervals"):
            fitted_regression.predict(test_data, type="conf_int")


class TestRuleFitExtractOutputs:
    """Test rule_fit extract_outputs"""

    @pytest.fixture
    def fitted_regression(self):
        """Create fitted regression model"""
        np.random.seed(42)
        n = 100
        train_data = pd.DataFrame({
            "y": np.random.randn(n) * 10 + 50,
            "x1": np.random.randn(n) * 5,
            "x2": np.random.randn(n) * 3,
            "x3": np.random.randn(n) * 2,
        })
        spec = rule_fit().set_mode("regression")
        return spec.fit(train_data, "y ~ x1 + x2 + x3")

    def test_extract_outputs_returns_three_dataframes(self, fitted_regression):
        """Test that extract_outputs returns 3 DataFrames"""
        outputs, coefficients, stats = fitted_regression.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_has_required_columns(self, fitted_regression):
        """Test outputs DataFrame has required columns"""
        outputs, _, _ = fitted_regression.extract_outputs()

        required_cols = ["actuals", "fitted", "forecast", "residuals", "split", "model", "model_group_name", "group"]
        for col in required_cols:
            assert col in outputs.columns

    def test_outputs_has_train_data(self, fitted_regression):
        """Test outputs has training data"""
        outputs, _, _ = fitted_regression.extract_outputs()

        train_outputs = outputs[outputs["split"] == "train"]
        assert len(train_outputs) > 0
        assert train_outputs["actuals"].notna().all()
        assert train_outputs["fitted"].notna().all()

    def test_coefficients_has_rules(self, fitted_regression):
        """Test coefficients DataFrame has rules"""
        _, coefficients, _ = fitted_regression.extract_outputs()

        # Should have variable and coefficient columns
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns
        assert "importance" in coefficients.columns

        # Should have at least some rules or features
        assert len(coefficients) > 0

    def test_stats_has_metrics(self, fitted_regression):
        """Test stats DataFrame has metrics"""
        _, _, stats = fitted_regression.extract_outputs()

        # Should have metric and value columns
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Should have regression metrics
        metrics = stats["metric"].unique()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics

    def test_stats_has_rule_count(self, fitted_regression):
        """Test stats includes rule count"""
        _, _, stats = fitted_regression.extract_outputs()

        # Should have n_rules metric
        n_rules_row = stats[stats["metric"] == "n_rules"]
        assert len(n_rules_row) > 0

    def test_stats_has_model_info(self, fitted_regression):
        """Test stats includes model information"""
        _, _, stats = fitted_regression.extract_outputs()

        metrics = stats["metric"].unique()
        assert "model_type" in metrics
        assert "mode" in metrics
        assert "n_features" in metrics
        assert "max_rules" in metrics


class TestRuleFitEvaluate:
    """Test rule_fit evaluate method"""

    @pytest.fixture
    def train_test_data(self):
        """Create train and test data"""
        np.random.seed(42)
        n_train = 80
        n_test = 20

        train = pd.DataFrame({
            "y": np.random.randn(n_train) * 10 + 50,
            "x1": np.random.randn(n_train) * 5,
            "x2": np.random.randn(n_train) * 3,
        })

        test = pd.DataFrame({
            "y": np.random.randn(n_test) * 10 + 50,
            "x1": np.random.randn(n_test) * 5,
            "x2": np.random.randn(n_test) * 3,
        })

        return train, test

    def test_evaluate_with_test_data(self, train_test_data):
        """Test evaluate() method with test data"""
        train, test = train_test_data

        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        evaluated = fit.evaluate(test)

        assert isinstance(evaluated, ModelFit)
        assert "test_predictions" in evaluated.evaluation_data

    def test_evaluate_extract_outputs_has_test(self, train_test_data):
        """Test that extract_outputs includes test data after evaluate"""
        train, test = train_test_data

        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        evaluated = fit.evaluate(test)

        outputs, _, _ = evaluated.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == len(test)

    def test_evaluate_stats_has_test_metrics(self, train_test_data):
        """Test that stats includes test metrics after evaluate"""
        train, test = train_test_data

        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        evaluated = fit.evaluate(test)

        _, _, stats = evaluated.extract_outputs()

        # Should have test split metrics
        test_stats = stats[stats["split"] == "test"]
        assert len(test_stats) > 0

        # Should have test RMSE
        test_rmse = stats[(stats["metric"] == "rmse") & (stats["split"] == "test")]
        assert len(test_rmse) == 1
        assert test_rmse["value"].iloc[0] > 0


class TestRuleFitEdgeCases:
    """Test edge cases and error handling"""

    def test_fit_without_mode_raises_error(self):
        """Test that fitting without mode raises error"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.randn(20),
            "x1": np.random.randn(20),
        })

        spec = rule_fit()  # No mode set

        with pytest.raises(ValueError, match="Unsupported mode"):
            spec.fit(train_data, "y ~ x1")

    def test_fit_with_single_predictor(self):
        """Test fitting with single predictor"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.randn(50) * 10,
            "x1": np.random.randn(50) * 5,
        })

        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1")

        assert isinstance(fit, ModelFit)
        assert fit.fit_data["n_features"] == 1

    def test_predict_with_new_data_different_size(self):
        """Test prediction with different sized test data"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.randn(100) * 10,
            "x1": np.random.randn(100) * 5,
            "x2": np.random.randn(100) * 3,
        })

        test_data = pd.DataFrame({
            "x1": np.random.randn(25) * 5,
            "x2": np.random.randn(25) * 3,
        })

        spec = rule_fit().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")
        predictions = fit.predict(test_data, type="numeric")

        assert len(predictions) == 25

    def test_zero_penalty(self):
        """Test with zero penalty (no regularization)"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.randn(50) * 10,
            "x1": np.random.randn(50) * 5,
            "x2": np.random.randn(50) * 3,
        })

        spec = rule_fit(penalty=0.0).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # For regression, alpha should be 0.0
        assert fit.fit_data["model"].alpha == 0.0

    def test_zero_penalty_classification(self):
        """Test with zero penalty for classification (uses small value to avoid div by zero)"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.choice([0, 1], 50),
            "x1": np.random.randn(50) * 5,
            "x2": np.random.randn(50) * 3,
        })

        spec = rule_fit(penalty=0.0).set_mode("classification")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # For classification, alpha=0.0 would cause div by zero, so use small value
        assert fit.fit_data["model"].alpha == 1e-10

    def test_large_max_rules(self):
        """Test with large max_rules value"""
        np.random.seed(42)
        train_data = pd.DataFrame({
            "y": np.random.randn(100) * 10,
            "x1": np.random.randn(100) * 5,
            "x2": np.random.randn(100) * 3,
        })

        spec = rule_fit(max_rules=50).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model"].max_rules == 50
