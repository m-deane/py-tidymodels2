"""
Tests for decision_tree model specification and sklearn engine

Tests cover:
- Model specification creation
- Parameter translation
- Fitting with formula
- Prediction
- Extract outputs (outputs, coefficients as feature importances, stats)
- Evaluate() method with test data
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import decision_tree, ModelSpec, ModelFit


class TestDecisionTreeSpec:
    """Test decision_tree() model specification"""

    def test_default_spec(self):
        """Test default decision_tree specification"""
        spec = decision_tree()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "decision_tree"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_tree_depth(self):
        """Test decision_tree with tree_depth parameter"""
        spec = decision_tree(tree_depth=5)

        assert spec.args == {"tree_depth": 5}

    def test_spec_with_min_n(self):
        """Test decision_tree with min_n parameter"""
        spec = decision_tree(min_n=10)

        assert spec.args == {"min_n": 10}

    def test_spec_with_cost_complexity(self):
        """Test decision_tree with cost_complexity parameter"""
        spec = decision_tree(cost_complexity=0.01)

        assert spec.args == {"cost_complexity": 0.01}

    def test_spec_with_all_parameters(self):
        """Test decision_tree with all parameters"""
        spec = decision_tree(tree_depth=10, min_n=5, cost_complexity=0.001)

        assert spec.args == {"tree_depth": 10, "min_n": 5, "cost_complexity": 0.001}

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = decision_tree()
        spec = spec.set_engine("sklearn")

        assert spec.engine == "sklearn"

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = decision_tree(tree_depth=5)
        spec2 = spec1.set_args(tree_depth=10)

        # Original spec should be unchanged
        assert spec1.args == {"tree_depth": 5}
        # New spec should have new value
        assert spec2.args == {"tree_depth": 10}


class TestDecisionTreeFit:
    """Test decision_tree fitting"""

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
        spec = decision_tree()
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_model_class(self, train_data):
        """Test correct sklearn model class"""
        spec = decision_tree()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "DecisionTreeRegressor"

    def test_fit_with_tree_depth(self, train_data):
        """Test fitting with tree_depth parameter"""
        spec = decision_tree(tree_depth=3)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        model = fit.fit_data["model"]
        assert model.max_depth == 3

    def test_fit_with_min_n(self, train_data):
        """Test fitting with min_n parameter"""
        spec = decision_tree(min_n=3)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.min_samples_split == 3

    def test_fit_with_cost_complexity(self, train_data):
        """Test fitting with cost_complexity parameter"""
        spec = decision_tree(cost_complexity=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.ccp_alpha == 0.01

    def test_fit_stores_residuals(self, train_data):
        """Test that fit stores residuals"""
        spec = decision_tree()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert "residuals" in fit.fit_data
        assert fit.fit_data["residuals"] is not None
        assert len(fit.fit_data["residuals"]) == len(train_data)


class TestDecisionTreePredict:
    """Test decision_tree prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = decision_tree(tree_depth=5)
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
        # Allow wide range for tree-based model
        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_predict_invalid_type(self, fitted_model):
        """Test prediction with invalid type raises error"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model.predict(test, type="class")


class TestDecisionTreeExtractOutputs:
    """Test decision_tree output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

        spec = decision_tree(tree_depth=5)
        fit = spec.fit(train, "y ~ x1 + x2 + x3")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        sklearn_model = fitted_model.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "feature_importances_")
        assert hasattr(sklearn_model, "max_depth")

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

    def test_extract_outputs_coefficients_feature_importances(self, fitted_model):
        """Test Coefficients DataFrame contains feature importances"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Check for column names
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns  # Contains feature importances
        # Should have 3 features (x1, x2, x3)
        assert len(coefs) == 3
        # Feature importances should be non-negative
        assert all(coefs["coefficient"] >= 0)
        # Feature importances should sum to 1
        assert np.isclose(coefs["coefficient"].sum(), 1.0)

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
        assert "max_depth" in metric_names
        assert "n_leaves" in metric_names


class TestDecisionTreeEvaluate:
    """Test decision_tree evaluate() method"""

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

        spec = decision_tree(tree_depth=5)
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

        spec = decision_tree()
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        # Test split should have 3 observations
        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == 3

    def test_evaluate_stats(self, train_test_data):
        """Test that evaluate() includes test metrics in stats"""
        train, test = train_test_data

        spec = decision_tree()
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        _, _, stats = fit.extract_outputs()

        # Should have metrics for both train and test
        assert "train" in stats["split"].values
        assert "test" in stats["split"].values
        # Check for test RMSE
        test_stats = stats[stats["split"] == "test"]
        assert "rmse" in test_stats["metric"].values
        assert "mae" in test_stats["metric"].values


class TestDecisionTreeParameterTranslation:
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

    def test_param_translation_tree_depth(self, train_data):
        """Test tree_depth parameter maps to max_depth"""
        spec = decision_tree(tree_depth=5)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.max_depth == 5

    def test_param_translation_min_n(self, train_data):
        """Test min_n parameter maps to min_samples_split"""
        spec = decision_tree(min_n=4)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.min_samples_split == 4

    def test_param_translation_cost_complexity(self, train_data):
        """Test cost_complexity parameter maps to ccp_alpha"""
        spec = decision_tree(cost_complexity=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.ccp_alpha == 0.01

    def test_param_translation_all(self, train_data):
        """Test all parameter translations together"""
        spec = decision_tree(tree_depth=5, min_n=3, cost_complexity=0.001)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.max_depth == 5
        assert model.min_samples_split == 3
        assert model.ccp_alpha == 0.001


class TestDecisionTreeIntegration:
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
        spec = decision_tree(tree_depth=5)
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
        assert len(coefs) == 2  # price, advertising
        assert len(stats) > 0

    def test_workflow_with_evaluate(self):
        """Test workflow with evaluate()"""
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

        spec = decision_tree(tree_depth=5)
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        # Extract outputs should include both train and test
        outputs, coefs, stats = fit.extract_outputs()

        assert len(outputs) == 11  # 8 train + 3 test
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Stats should include test metrics
        test_stats = stats[stats["split"] == "test"]
        assert len(test_stats) > 0
        assert "rmse" in test_stats["metric"].values
