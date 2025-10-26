"""
Tests for rand_forest model specification and sklearn engine

Tests cover:
- Model specification creation (default, with parameters)
- Setting mode (regression vs classification)
- Immutability of ModelSpec
- Fitting with regression mode
- Fitting with classification mode
- Prediction (numeric for regression, class/prob for classification)
- Extract outputs (outputs, coefficients as feature importances, stats)
- Evaluate() method with test data
- Parameter translation (mtry→max_features, trees→n_estimators, min_n→min_samples_split)
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import rand_forest, ModelSpec, ModelFit


class TestRandForestSpec:
    """Test rand_forest() model specification"""

    def test_default_spec(self):
        """Test default rand_forest specification"""
        spec = rand_forest()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "rand_forest"
        assert spec.engine == "sklearn"
        assert spec.mode == "unknown"  # Must be set via set_mode()
        assert spec.args == {}

    def test_spec_with_mtry(self):
        """Test rand_forest with mtry parameter"""
        spec = rand_forest(mtry=5)

        assert spec.args == {"mtry": 5}

    def test_spec_with_trees(self):
        """Test rand_forest with trees parameter"""
        spec = rand_forest(trees=1000)

        assert spec.args == {"trees": 1000}

    def test_spec_with_min_n(self):
        """Test rand_forest with min_n parameter"""
        spec = rand_forest(min_n=10)

        assert spec.args == {"min_n": 10}

    def test_spec_with_all_parameters(self):
        """Test rand_forest with all parameters"""
        spec = rand_forest(mtry=5, trees=1000, min_n=10)

        assert spec.args == {"mtry": 5, "trees": 1000, "min_n": 10}

    def test_set_mode_regression(self):
        """Test set_mode() for regression"""
        spec = rand_forest()
        spec = spec.set_mode("regression")

        assert spec.mode == "regression"

    def test_set_mode_classification(self):
        """Test set_mode() for classification"""
        spec = rand_forest()
        spec = spec.set_mode("classification")

        assert spec.mode == "classification"

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = rand_forest()
        spec = spec.set_engine("sklearn")

        assert spec.engine == "sklearn"

    def test_set_args(self):
        """Test set_args() method"""
        spec = rand_forest()
        spec = spec.set_args(mtry=3, trees=500, min_n=5)

        assert spec.args == {"mtry": 3, "trees": 500, "min_n": 5}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = rand_forest(trees=100)
        spec2 = spec1.set_args(trees=500)

        # Original spec should be unchanged
        assert spec1.args == {"trees": 100}
        # New spec should have new value
        assert spec2.args == {"trees": 500}

    def test_immutability_set_mode(self):
        """Test that set_mode() returns new instance"""
        spec1 = rand_forest()
        spec2 = spec1.set_mode("regression")

        # Original spec should be unchanged
        assert spec1.mode == "unknown"
        # New spec should have new mode
        assert spec2.mode == "regression"


class TestRandForestFitRegression:
    """Test rand_forest fitting with regression mode"""

    @pytest.fixture
    def train_data_regression(self):
        """Create sample regression training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

    def test_fit_requires_mode(self, train_data_regression):
        """Test that fitting requires mode to be set"""
        spec = rand_forest()  # mode="unknown" by default

        with pytest.raises(ValueError, match="mode must be 'regression' or 'classification'"):
            spec.fit(train_data_regression, "y ~ x1 + x2 + x3")

    def test_fit_regression_with_formula(self, train_data_regression):
        """Test fitting with formula in regression mode"""
        spec = rand_forest().set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_regression_default_trees(self, train_data_regression):
        """Test regression with default trees (500)"""
        spec = rand_forest().set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2")

        # Should use RandomForestRegressor
        assert fit.fit_data["model_class"] == "RandomForestRegressor"
        # Check default n_estimators = 500
        model = fit.fit_data["model"]
        assert model.n_estimators == 500

    def test_fit_regression_custom_trees(self, train_data_regression):
        """Test regression with custom trees parameter"""
        spec = rand_forest(trees=100).set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.n_estimators == 100

    def test_fit_regression_custom_mtry(self, train_data_regression):
        """Test regression with mtry (maps to max_features)"""
        spec = rand_forest(mtry=2).set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2 + x3")

        model = fit.fit_data["model"]
        assert model.max_features == 2

    def test_fit_regression_custom_min_n(self, train_data_regression):
        """Test regression with min_n (maps to min_samples_split)"""
        spec = rand_forest(min_n=5).set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.min_samples_split == 5

    def test_fit_regression_all_params(self, train_data_regression):
        """Test regression with all parameters"""
        spec = rand_forest(mtry=2, trees=200, min_n=3).set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2 + x3")

        model = fit.fit_data["model"]
        assert model.max_features == 2
        assert model.n_estimators == 200
        assert model.min_samples_split == 3

    def test_fit_regression_has_residuals(self, train_data_regression):
        """Test that regression fit calculates residuals"""
        spec = rand_forest(trees=50).set_mode("regression")
        fit = spec.fit(train_data_regression, "y ~ x1 + x2")

        assert "residuals" in fit.fit_data
        assert fit.fit_data["residuals"] is not None
        assert len(fit.fit_data["residuals"]) == len(train_data_regression)


class TestRandForestFitClassification:
    """Test rand_forest fitting with classification mode"""

    @pytest.fixture
    def train_data_classification(self):
        """Create sample classification training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21, 29, 13, 19, 27],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5, 14.5, 6.5, 9.5, 13.5],
            "x3": [2, 4, 2.2, 6, 4.2, 5.8, 2.1, 4.1, 5.9, 2.3, 3.9, 5.7],
        })

    def test_fit_classification_with_formula(self, train_data_classification):
        """Test fitting with formula in classification mode"""
        spec = rand_forest().set_mode("classification")
        fit = spec.fit(train_data_classification, "species ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_classification_model_class(self, train_data_classification):
        """Test classification uses RandomForestClassifier"""
        spec = rand_forest().set_mode("classification")
        fit = spec.fit(train_data_classification, "species ~ x1 + x2")

        # Should use RandomForestClassifier
        assert fit.fit_data["model_class"] == "RandomForestClassifier"

    def test_fit_classification_custom_params(self, train_data_classification):
        """Test classification with custom parameters"""
        spec = rand_forest(mtry=2, trees=100, min_n=2).set_mode("classification")
        fit = spec.fit(train_data_classification, "species ~ x1 + x2 + x3")

        model = fit.fit_data["model"]
        assert model.max_features == 2
        assert model.n_estimators == 100
        assert model.min_samples_split == 2

    def test_fit_classification_no_residuals(self, train_data_classification):
        """Test that classification fit does not calculate meaningful residuals"""
        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train_data_classification, "species ~ x1 + x2")

        # Residuals should be None for classification
        assert fit.fit_data["residuals"] is None


class TestRandForestPredict:
    """Test rand_forest prediction"""

    @pytest.fixture
    def fitted_model_regression(self):
        """Create fitted regression model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = rand_forest(trees=50).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    @pytest.fixture
    def fitted_model_classification(self):
        """Create fitted classification model for testing"""
        train = pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21, 29, 13, 19, 27],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5, 14.5, 6.5, 9.5, 13.5],
        })

        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")
        return fit

    def test_predict_regression_numeric(self, fitted_model_regression):
        """Test regression prediction with type='numeric'"""
        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fitted_model_regression.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_predict_regression_default_numeric(self, fitted_model_regression):
        """Test regression prediction defaults to numeric"""
        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fitted_model_regression.predict(test)

        assert ".pred" in predictions.columns

    def test_predict_regression_values_reasonable(self, fitted_model_regression):
        """Test that regression predictions are in reasonable range"""
        test = pd.DataFrame({
            "x1": [15],
            "x2": [7],
        })

        predictions = fitted_model_regression.predict(test)

        # For x1=15, x2=7 (middle values), prediction should be ~150
        # Allow wide range for tree-based model
        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_predict_regression_invalid_type(self, fitted_model_regression):
        """Test regression raises error for invalid prediction types"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model_regression.predict(test, type="class")

    def test_predict_classification_class(self, fitted_model_classification):
        """Test classification prediction with type='class'"""
        test = pd.DataFrame({
            "x1": [12, 22, 28],
            "x2": [6, 11, 14],
        })

        predictions = fitted_model_classification.predict(test, type="class")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred_class" in predictions.columns
        assert len(predictions) == 3
        # Check predictions are valid classes
        assert all(pred in ["A", "B", "C"] for pred in predictions[".pred_class"])

    def test_predict_classification_prob(self, fitted_model_classification):
        """Test classification prediction with type='prob'"""
        test = pd.DataFrame({
            "x1": [12, 22, 28],
            "x2": [6, 11, 14],
        })

        predictions = fitted_model_classification.predict(test, type="prob")

        assert isinstance(predictions, pd.DataFrame)
        # Should have columns for each class
        assert ".pred_A" in predictions.columns
        assert ".pred_B" in predictions.columns
        assert ".pred_C" in predictions.columns
        assert len(predictions) == 3
        # Probabilities should sum to 1 for each row
        assert all(np.isclose(predictions.sum(axis=1), 1.0))

    def test_predict_classification_invalid_type(self, fitted_model_classification):
        """Test classification raises error for invalid prediction types"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="supports type='class' or 'prob'"):
            fitted_model_classification.predict(test, type="numeric")


class TestRandForestExtractOutputs:
    """Test rand_forest output extraction"""

    @pytest.fixture
    def fitted_model_regression(self):
        """Create fitted regression model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

        spec = rand_forest(trees=50).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2 + x3")
        return fit

    @pytest.fixture
    def fitted_model_classification(self):
        """Create fitted classification model for testing"""
        train = pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21, 29, 13, 19, 27],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5, 14.5, 6.5, 9.5, 13.5],
        })

        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")
        return fit

    def test_extract_fit_engine_regression(self, fitted_model_regression):
        """Test extract_fit_engine() for regression"""
        sklearn_model = fitted_model_regression.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "feature_importances_")
        assert hasattr(sklearn_model, "n_estimators")

    def test_extract_fit_engine_classification(self, fitted_model_classification):
        """Test extract_fit_engine() for classification"""
        sklearn_model = fitted_model_classification.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "feature_importances_")
        assert hasattr(sklearn_model, "classes_")

    def test_extract_outputs_regression(self, fitted_model_regression):
        """Test extract_outputs() returns three DataFrames for regression"""
        outputs, coefs, stats = fitted_model_regression.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_classification(self, fitted_model_classification):
        """Test extract_outputs() returns three DataFrames for classification"""
        outputs, coefs, stats = fitted_model_classification.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_model_outputs_regression(self, fitted_model_regression):
        """Test Outputs DataFrame structure for regression"""
        outputs, _, _ = fitted_model_regression.extract_outputs()

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

    def test_extract_outputs_model_outputs_classification(self, fitted_model_classification):
        """Test Outputs DataFrame structure for classification"""
        outputs, _, _ = fitted_model_classification.extract_outputs()

        # Check for observation-level columns
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "forecast" in outputs.columns
        assert "residuals" in outputs.columns  # Should be NaN for classification
        assert "split" in outputs.columns
        assert "model" in outputs.columns
        # All training data should have split='train'
        assert all(outputs["split"] == "train")
        # Residuals should be NaN for classification
        assert all(outputs["residuals"].isna())

    def test_extract_outputs_coefficients_feature_importances(self, fitted_model_regression):
        """Test Coefficients DataFrame contains feature importances"""
        _, coefs, _ = fitted_model_regression.extract_outputs()

        # Check for column names
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns  # Contains feature importances
        assert "std_error" in coefs.columns
        assert "p_value" in coefs.columns
        assert "vif" in coefs.columns
        # Should have 3 features (x1, x2, x3)
        assert len(coefs) == 3
        # Feature importances (coefficient column) should be non-negative
        assert all(coefs["coefficient"] >= 0)
        # Feature importances should sum to 1
        assert np.isclose(coefs["coefficient"].sum(), 1.0)

    def test_extract_outputs_coefficients_no_inference_stats(self, fitted_model_regression):
        """Test that statistical inference columns are NaN for tree-based models"""
        _, coefs, _ = fitted_model_regression.extract_outputs()

        # For Random Forest, these statistical inference metrics are not applicable
        assert all(coefs["std_error"].isna())
        assert all(coefs["p_value"].isna())
        assert all(coefs["vif"].isna())

    def test_extract_outputs_stats_regression(self, fitted_model_regression):
        """Test Stats DataFrame structure for regression"""
        _, _, stats = fitted_model_regression.extract_outputs()

        # Check that stats has metrics
        assert len(stats) > 0
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check for specific regression metrics
        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names
        assert "n_trees" in metric_names
        assert "n_features" in metric_names

    def test_extract_outputs_stats_classification(self, fitted_model_classification):
        """Test Stats DataFrame structure for classification"""
        _, _, stats = fitted_model_classification.extract_outputs()

        # Check that stats has model info
        assert len(stats) > 0
        assert "metric" in stats.columns
        assert "value" in stats.columns

        # Check for model information
        metric_names = stats["metric"].values
        assert "n_trees" in metric_names
        assert "n_features" in metric_names
        assert "mode" in metric_names


class TestRandForestEvaluate:
    """Test rand_forest evaluate() method"""

    @pytest.fixture
    def train_test_data_regression(self):
        """Create train/test split for regression"""
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

    @pytest.fixture
    def train_test_data_classification(self):
        """Create train/test split for classification"""
        np.random.seed(42)
        train = pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5],
        })
        test = pd.DataFrame({
            "species": ["C", "A", "B"],
            "x1": [29, 13, 19],
            "x2": [14.5, 6.5, 9.5],
        })
        return train, test

    def test_evaluate_regression(self, train_test_data_regression):
        """Test evaluate() method for regression"""
        train, test = train_test_data_regression

        spec = rand_forest(trees=50).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")

        # Evaluate on test data
        fit = fit.evaluate(test, "y")

        # Check that evaluation data is stored
        assert "test_predictions" in fit.evaluation_data
        assert "test_data" in fit.evaluation_data
        assert "outcome_col" in fit.evaluation_data

    def test_evaluate_regression_outputs(self, train_test_data_regression):
        """Test that evaluate() includes test data in outputs"""
        train, test = train_test_data_regression

        spec = rand_forest(trees=50).set_mode("regression")
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        # Test split should have 3 observations
        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == 3

    def test_evaluate_regression_stats(self, train_test_data_regression):
        """Test that evaluate() includes test metrics in stats"""
        train, test = train_test_data_regression

        spec = rand_forest(trees=50).set_mode("regression")
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

    def test_evaluate_classification(self, train_test_data_classification):
        """Test evaluate() method for classification"""
        train, test = train_test_data_classification

        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")

        # Evaluate on test data
        fit = fit.evaluate(test, "species")

        # Check that evaluation data is stored
        assert "test_predictions" in fit.evaluation_data
        assert "test_data" in fit.evaluation_data
        assert "outcome_col" in fit.evaluation_data

    def test_evaluate_classification_outputs(self, train_test_data_classification):
        """Test that evaluate() includes test data in outputs for classification"""
        train, test = train_test_data_classification

        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")
        fit = fit.evaluate(test, "species")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        # Test split should have 3 observations
        test_outputs = outputs[outputs["split"] == "test"]
        assert len(test_outputs) == 3


class TestRandForestParameterTranslation:
    """Test parameter translation from tidymodels to sklearn"""

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

    def test_param_translation_mtry_to_max_features(self, train_data):
        """Test mtry parameter maps to max_features"""
        spec = rand_forest(mtry=2).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        model = fit.extract_fit_engine()
        assert model.max_features == 2

    def test_param_translation_trees_to_n_estimators(self, train_data):
        """Test trees parameter maps to n_estimators"""
        spec = rand_forest(trees=150).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.n_estimators == 150

    def test_param_translation_min_n_to_min_samples_split(self, train_data):
        """Test min_n parameter maps to min_samples_split"""
        spec = rand_forest(min_n=4).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.min_samples_split == 4

    def test_param_translation_all_parameters(self, train_data):
        """Test all parameter translations together"""
        spec = rand_forest(mtry=2, trees=200, min_n=3).set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        model = fit.extract_fit_engine()
        assert model.max_features == 2
        assert model.n_estimators == 200
        assert model.min_samples_split == 3

    def test_default_parameters_regression(self, train_data):
        """Test default parameters for regression"""
        spec = rand_forest().set_mode("regression")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        # Default trees should be 500
        assert model.n_estimators == 500
        # Default min_samples_split should be 2
        assert model.min_samples_split == 2

    def test_default_parameters_classification(self):
        """Test default parameters for classification"""
        train = pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21, 29, 13],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5, 14.5, 6.5],
        })

        spec = rand_forest().set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")

        model = fit.extract_fit_engine()
        # Default trees should be 500
        assert model.n_estimators == 500
        # Default min_samples_split should be 2
        assert model.min_samples_split == 2


class TestIntegration:
    """Integration tests for full rand_forest workflow"""

    def test_full_workflow_regression(self):
        """Test complete regression workflow"""
        # Training data
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        # Create spec and fit
        spec = rand_forest(trees=100).set_mode("regression")
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
            "species": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
            "sepal_length": [5.1, 7.0, 4.9, 6.3, 6.8, 5.8, 5.0, 6.5, 6.2, 5.2, 6.9, 5.9],
            "sepal_width": [3.5, 3.2, 3.0, 3.3, 2.8, 2.7, 3.6, 2.9, 2.9, 3.4, 3.1, 3.0],
        })

        # Create spec and fit
        spec = rand_forest(trees=100).set_mode("classification")
        fit = spec.fit(train, "species ~ sepal_length + sepal_width")

        # Test data
        test = pd.DataFrame({
            "sepal_length": [5.3, 6.7, 6.0],
            "sepal_width": [3.3, 3.0, 2.8],
        })

        # Predict classes
        pred_class = fit.predict(test, type="class")
        assert len(pred_class) == 3
        assert ".pred_class" in pred_class.columns

        # Predict probabilities
        pred_prob = fit.predict(test, type="prob")
        assert len(pred_prob) == 3
        assert ".pred_A" in pred_prob.columns
        assert ".pred_B" in pred_prob.columns
        assert ".pred_C" in pred_prob.columns

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()
        assert len(outputs) == 12  # Training observations
        assert len(coefs) == 2  # sepal_length, sepal_width

    def test_workflow_with_evaluate_regression(self):
        """Test regression workflow with evaluate()"""
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

        spec = rand_forest(trees=50).set_mode("regression")
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

    def test_workflow_with_evaluate_classification(self):
        """Test classification workflow with evaluate()"""
        train = pd.DataFrame({
            "species": ["A", "B", "A", "C", "B", "C", "A", "B"],
            "x1": [10, 20, 12, 30, 22, 28, 11, 21],
            "x2": [5, 10, 6, 15, 11, 14, 5.5, 10.5],
        })

        test = pd.DataFrame({
            "species": ["C", "A", "B"],
            "x1": [29, 13, 19],
            "x2": [14.5, 6.5, 9.5],
        })

        spec = rand_forest(trees=50).set_mode("classification")
        fit = spec.fit(train, "species ~ x1 + x2")
        fit = fit.evaluate(test, "species")

        # Extract outputs should include both train and test
        outputs, coefs, stats = fit.extract_outputs()

        assert len(outputs) == 11  # 8 train + 3 test
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
