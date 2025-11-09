"""
Tests for bag_tree model specification and sklearn engine

Tests cover:
- Model specification creation
- Engine registration
- Regression mode fitting
- Classification mode fitting
- Prediction types
- Extract outputs (three-DataFrame format)
- Feature importance
- Parameter handling
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip.models.bag_tree import bag_tree
from py_parsnip.model_spec import ModelSpec, ModelFit


class TestBagTreeSpec:
    """Test bag_tree() model specification"""

    def test_default_spec(self):
        """Test default bag_tree specification"""
        spec = bag_tree()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "bag_tree"
        assert spec.engine == "sklearn"
        assert spec.mode == "unknown"  # Must be set explicitly
        assert spec.args == {}

    def test_spec_with_trees(self):
        """Test bag_tree with trees parameter"""
        spec = bag_tree(trees=50)

        assert spec.args == {"trees": 50}

    def test_spec_with_min_n(self):
        """Test bag_tree with min_n parameter"""
        spec = bag_tree(min_n=10)

        assert spec.args == {"min_n": 10}

    def test_spec_with_cost_complexity(self):
        """Test bag_tree with cost_complexity parameter"""
        spec = bag_tree(cost_complexity=0.01)

        assert spec.args == {"cost_complexity": 0.01}

    def test_spec_with_tree_depth(self):
        """Test bag_tree with tree_depth parameter"""
        spec = bag_tree(tree_depth=5)

        assert spec.args == {"tree_depth": 5}

    def test_spec_with_all_params(self):
        """Test bag_tree with all parameters"""
        spec = bag_tree(trees=30, min_n=5, cost_complexity=0.02, tree_depth=8)

        assert spec.args == {
            "trees": 30,
            "min_n": 5,
            "cost_complexity": 0.02,
            "tree_depth": 8,
        }

    def test_set_mode_regression(self):
        """Test set_mode() for regression"""
        spec = bag_tree()
        spec = spec.set_mode("regression")

        assert spec.mode == "regression"

    def test_set_mode_classification(self):
        """Test set_mode() for classification"""
        spec = bag_tree()
        spec = spec.set_mode("classification")

        assert spec.mode == "classification"

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = bag_tree()
        spec = spec.set_engine("sklearn")

        assert spec.engine == "sklearn"

    def test_set_args(self):
        """Test set_args() method"""
        spec = bag_tree()
        spec = spec.set_args(trees=100)

        assert spec.args == {"trees": 100}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = bag_tree(trees=25)
        spec2 = spec1.set_args(trees=50)

        # Original spec should be unchanged
        assert spec1.args == {"trees": 25}
        # New spec should have new value
        assert spec2.args == {"trees": 50}


class TestBagTreeRegression:
    """Test bag_tree fitting in regression mode"""

    @pytest.fixture
    def train_data(self):
        """Create sample regression training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 190, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 19, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 9.5, 12.5],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.8, 5.2],
        })

    @pytest.fixture
    def test_data(self):
        """Create sample test data"""
        return pd.DataFrame({
            "x1": [12, 26, 21],
            "x2": [6, 13, 10.5],
            "x3": [2.5, 5.2, 4.2],
        })

    def test_fit_basic_regression(self, train_data):
        """Test basic regression model fitting"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec.model_type == "bag_tree"
        assert fit.spec.mode == "regression"
        assert "model" in fit.fit_data
        assert fit.fit_data["n_estimators"] == 10

    def test_fit_with_default_trees(self, train_data):
        """Test fitting with default number of trees (25)"""
        spec = bag_tree().set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        # Default should be 25 trees
        assert fit.fit_data["n_estimators"] == 25

    def test_fit_with_all_parameters(self, train_data):
        """Test fitting with all parameters"""
        spec = bag_tree(
            trees=15,
            min_n=3,
            cost_complexity=0.01,
            tree_depth=4
        ).set_mode("regression")

        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        assert fit.fit_data["n_estimators"] == 15
        # Base estimator parameters are stored in the model

    def test_predict_regression(self, train_data, test_data):
        """Test regression predictions"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        predictions = fit.predict(test_data)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test_data)
        # Predictions should be numeric
        assert predictions[".pred"].dtype in [np.float64, np.float32, np.int64]

    def test_feature_importance(self, train_data):
        """Test that feature importance is available"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        # Feature names should be stored
        assert "feature_names" in fit.fit_data
        assert len(fit.fit_data["feature_names"]) == 3
        assert set(fit.fit_data["feature_names"]) == {"x1", "x2", "x3"}

    def test_fitted_values(self, train_data):
        """Test that fitted values are calculated"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        assert "fitted" in fit.fit_data
        assert len(fit.fit_data["fitted"]) == len(train_data)

    def test_residuals(self, train_data):
        """Test that residuals are calculated for regression"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2 + x3")

        assert "residuals" in fit.fit_data
        assert fit.fit_data["residuals"] is not None
        assert len(fit.fit_data["residuals"]) == len(train_data)


class TestBagTreeClassification:
    """Test bag_tree fitting in classification mode"""

    @pytest.fixture
    def train_data_binary(self):
        """Create sample binary classification data"""
        np.random.seed(123)
        return pd.DataFrame({
            "y": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "x1": [1, 5, 2, 6, 1.5, 5.5, 2.5, 6.5, 1.2, 5.2],
            "x2": [2, 8, 3, 9, 2.5, 8.5, 3.5, 9.5, 2.2, 8.2],
        })

    @pytest.fixture
    def train_data_multiclass(self):
        """Create sample multiclass classification data"""
        np.random.seed(456)
        return pd.DataFrame({
            "y": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "x1": [1, 5, 9, 1.5, 5.5, 9.5, 1.2, 5.2, 9.2, 1.8, 5.8, 9.8],
            "x2": [2, 6, 10, 2.5, 6.5, 10.5, 2.2, 6.2, 10.2, 2.8, 6.8, 10.8],
        })

    @pytest.fixture
    def test_data_class(self):
        """Create sample test data for classification"""
        return pd.DataFrame({
            "x1": [1.3, 5.3, 9.3],
            "x2": [2.3, 6.3, 10.3],
        })

    def test_fit_binary_classification(self, train_data_binary):
        """Test binary classification model fitting"""
        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data_binary, formula="y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec.mode == "classification"
        assert "model" in fit.fit_data

    def test_fit_multiclass_classification(self, train_data_multiclass):
        """Test multiclass classification model fitting"""
        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data_multiclass, formula="y ~ x1 + x2")

        assert isinstance(fit, ModelFit)
        assert fit.spec.mode == "classification"

    def test_predict_class(self, train_data_binary, test_data_class):
        """Test class predictions"""
        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data_binary, formula="y ~ x1 + x2")

        predictions = fit.predict(test_data_class, type="class")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred_class" in predictions.columns
        assert len(predictions) == len(test_data_class)
        # Predictions should be one of the original classes
        assert all(p in ["A", "B"] for p in predictions[".pred_class"])

    def test_predict_prob(self, train_data_binary, test_data_class):
        """Test probability predictions"""
        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data_binary, formula="y ~ x1 + x2")

        predictions = fit.predict(test_data_class, type="prob")

        assert isinstance(predictions, pd.DataFrame)
        # Should have probability columns for each class
        assert ".pred_A" in predictions.columns
        assert ".pred_B" in predictions.columns
        assert len(predictions) == len(test_data_class)
        # Probabilities should sum to 1
        prob_sum = predictions[[".pred_A", ".pred_B"]].sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sum, np.ones(len(test_data_class)))

    def test_predict_prob_multiclass(self, train_data_multiclass, test_data_class):
        """Test probability predictions for multiclass"""
        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data_multiclass, formula="y ~ x1 + x2")

        predictions = fit.predict(test_data_class, type="prob")

        # Should have probability columns for each class
        assert ".pred_A" in predictions.columns
        assert ".pred_B" in predictions.columns
        assert ".pred_C" in predictions.columns
        # Probabilities should sum to 1
        prob_sum = predictions[[".pred_A", ".pred_B", ".pred_C"]].sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sum, np.ones(len(test_data_class)))


class TestBagTreeOutputs:
    """Test extract_outputs() for bag_tree"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(789)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

    def test_extract_outputs_structure(self, train_data):
        """Test that extract_outputs returns three DataFrames"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should return three DataFrames
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_dataframe(self, train_data):
        """Test outputs DataFrame structure"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")

        outputs, _, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["actuals", "fitted", "forecast", "residuals", "split"]
        assert all(col in outputs.columns for col in required_cols)

        # Check split column
        assert all(outputs["split"] == "train")

        # Check lengths match
        assert len(outputs) == len(train_data)

    def test_coefficients_dataframe(self, train_data):
        """Test coefficients DataFrame (feature importance)"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")

        _, coefficients, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["variable", "coefficient"]
        assert all(col in coefficients.columns for col in required_cols)

        # Should have one row per feature
        assert len(coefficients) == 2  # x1, x2
        assert set(coefficients["variable"]) == {"x1", "x2"}

        # Feature importance (coefficient) should be non-negative
        assert all(coefficients["coefficient"] >= 0)

    def test_stats_dataframe(self, train_data):
        """Test stats DataFrame structure"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")

        _, _, stats = fit.extract_outputs()

        # Check required columns
        required_cols = ["metric", "value", "split"]
        assert all(col in stats.columns for col in required_cols)

        # Check for key metrics
        metric_names = stats["metric"].tolist()
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names
        assert "n_estimators" in metric_names
        assert "model_type" in metric_names

    def test_stats_model_info(self, train_data):
        """Test that stats include model information"""
        spec = bag_tree(trees=15).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")

        _, _, stats = fit.extract_outputs()

        # Extract specific metrics
        n_estimators = stats[stats["metric"] == "n_estimators"]["value"].iloc[0]
        model_type = stats[stats["metric"] == "model_type"]["value"].iloc[0]

        assert n_estimators == 15
        assert model_type == "bag_tree"

    def test_outputs_with_model_name(self, train_data):
        """Test that model name is included in outputs"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1 + x2")
        fit.model_name = "my_bag_tree"

        outputs, coefficients, stats = fit.extract_outputs()

        # All three DataFrames should have model column
        assert "model" in outputs.columns
        assert "model" in coefficients.columns
        assert "model" in stats.columns

        # Should use the model_name
        assert all(outputs["model"] == "my_bag_tree")
        assert all(coefficients["model"] == "my_bag_tree")
        assert all(stats["model"] == "my_bag_tree")


class TestBagTreeErrors:
    """Test error handling for bag_tree"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        return pd.DataFrame({
            "y": [1, 2, 3, 4, 5],
            "x1": [1, 2, 3, 4, 5],
        })

    def test_fit_without_mode_set(self, train_data):
        """Test that fitting without setting mode raises error"""
        spec = bag_tree(trees=10)

        # Mode is "unknown", should raise error during fit
        with pytest.raises(ValueError, match="mode must be"):
            spec.fit(train_data, formula="y ~ x1")

    def test_invalid_prediction_type_regression(self, train_data):
        """Test invalid prediction type for regression"""
        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train_data, formula="y ~ x1")

        test_data = pd.DataFrame({"x1": [6, 7]})

        with pytest.raises(ValueError, match="type must be"):
            fit.predict(test_data, type="class")

    def test_invalid_prediction_type_classification(self):
        """Test invalid prediction type for classification"""
        train_data = pd.DataFrame({
            "y": ["A", "B", "A", "B", "A"],
            "x1": [1, 2, 3, 4, 5],
        })

        spec = bag_tree(trees=10).set_mode("classification")
        fit = spec.fit(train_data, formula="y ~ x1")

        test_data = pd.DataFrame({"x1": [6, 7]})

        with pytest.raises(ValueError, match="type must be"):
            fit.predict(test_data, type="numeric")


class TestBagTreeEvaluate:
    """Test bag_tree evaluate() method"""

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

    def test_evaluate_basic(self, train_test_data):
        """Test evaluate() method"""
        train, test = train_test_data

        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")

        # Evaluate on test data
        fit = fit.evaluate(test, "y")

        # Check that evaluation data is stored
        assert "test_predictions" in fit.evaluation_data
        assert "test_data" in fit.evaluation_data
        assert "outcome_col" in fit.evaluation_data

    def test_evaluate_outputs_include_test(self, train_test_data):
        """Test that evaluate() includes test data in outputs"""
        train, test = train_test_data

        spec = bag_tree(trees=10).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        # Test split should have 3 observations
        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == 3

    def test_evaluate_stats_include_test(self, train_test_data):
        """Test that evaluate() includes test metrics in stats"""
        train, test = train_test_data

        spec = bag_tree(trees=10).set_mode("regression")
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


class TestBagTreeParameterTranslation:
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

    def test_param_translation_trees(self, train_data):
        """Test trees parameter maps to n_estimators"""
        spec = bag_tree(trees=30).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["n_estimators"] == 30

    def test_param_translation_tree_depth(self, train_data):
        """Test tree_depth parameter maps to max_depth in base estimator"""
        spec = bag_tree(tree_depth=5, trees=10).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Check base estimator has correct max_depth
        model = fit.extract_fit_engine()
        base_estimator = model.estimators_[0]
        assert base_estimator.max_depth == 5

    def test_param_translation_min_n(self, train_data):
        """Test min_n parameter maps to min_samples_split in base estimator"""
        spec = bag_tree(min_n=4, trees=10).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Check base estimator has correct min_samples_split
        model = fit.extract_fit_engine()
        base_estimator = model.estimators_[0]
        assert base_estimator.min_samples_split == 4


class TestBagTreeIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow_regression(self):
        """Test complete regression workflow"""
        # Training data
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        # Create spec and fit
        spec = bag_tree(trees=15, tree_depth=5).set_mode("regression")
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

    def test_full_workflow_classification(self):
        """Test complete classification workflow"""
        # Training data
        train = pd.DataFrame({
            "species": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "sepal_length": [5.1, 7.0, 4.9, 6.4, 5.0, 6.9, 5.4, 6.5, 4.6, 6.8],
            "sepal_width": [3.5, 3.2, 3.0, 3.2, 3.6, 3.1, 3.9, 2.8, 3.1, 2.8],
        })

        # Create spec and fit
        spec = bag_tree(trees=15).set_mode("classification")
        fit = spec.fit(train, "species ~ sepal_length + sepal_width")

        # Test data
        test = pd.DataFrame({
            "sepal_length": [5.2, 6.7, 4.8],
            "sepal_width": [3.4, 3.0, 3.2],
        })

        # Predict class
        predictions = fit.predict(test, type="class")
        assert ".pred_class" in predictions.columns
        assert all(p in ["A", "B"] for p in predictions[".pred_class"])

        # Predict probabilities
        probs = fit.predict(test, type="prob")
        assert ".pred_A" in probs.columns
        assert ".pred_B" in probs.columns
        # Probabilities should sum to 1
        prob_sum = probs[[".pred_A", ".pred_B"]].sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sum, np.ones(3))

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

        spec = bag_tree(trees=15, tree_depth=5).set_mode("regression")
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

    def test_feature_importance_ordering(self):
        """Test that feature importance correctly identifies most important features"""
        # Create data where x1 is clearly more predictive than x2
        np.random.seed(42)
        train = pd.DataFrame({
            "y": np.array([10, 20, 15, 30, 25, 18, 22, 28, 16, 24]) * 10,  # Strong relationship with x1
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": np.random.randn(10),  # Random noise
        })

        spec = bag_tree(trees=20).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")

        _, coefs, _ = fit.extract_outputs()

        # x1 should have higher importance than x2
        x1_importance = coefs[coefs["variable"] == "x1"]["coefficient"].iloc[0]
        x2_importance = coefs[coefs["variable"] == "x2"]["coefficient"].iloc[0]

        assert x1_importance > x2_importance
