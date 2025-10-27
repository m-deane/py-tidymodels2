"""
Tests for boost_tree model specification and gradient boosting engines

Tests cover:
- Model specification creation
- Engine registration (XGBoost, LightGBM, CatBoost)
- Fitting with formula
- Prediction
- Extract outputs
- Parameter translation
- Feature importance
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import boost_tree, ModelSpec, ModelFit


class TestBoostTreeSpec:
    """Test boost_tree() model specification"""

    def test_default_spec(self):
        """Test default boost_tree specification"""
        spec = boost_tree()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "boost_tree"
        assert spec.engine == "xgboost"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_trees(self):
        """Test boost_tree with trees parameter"""
        spec = boost_tree(trees=100)

        assert spec.args == {"trees": 100}

    def test_spec_with_multiple_params(self):
        """Test boost_tree with multiple parameters"""
        spec = boost_tree(
            trees=100,
            tree_depth=6,
            learn_rate=0.1,
            mtry=5,
            min_n=10
        )

        assert spec.args == {
            "trees": 100,
            "tree_depth": 6,
            "learn_rate": 0.1,
            "mtry": 5,
            "min_n": 10,
        }

    def test_set_engine_xgboost(self):
        """Test set_engine() method with xgboost"""
        spec = boost_tree()
        spec = spec.set_engine("xgboost")

        assert spec.engine == "xgboost"

    def test_set_engine_lightgbm(self):
        """Test set_engine() method with lightgbm"""
        spec = boost_tree()
        spec = spec.set_engine("lightgbm")

        assert spec.engine == "lightgbm"

    def test_set_engine_catboost(self):
        """Test set_engine() method with catboost"""
        spec = boost_tree()
        spec = spec.set_engine("catboost")

        assert spec.engine == "catboost"

    def test_set_args(self):
        """Test set_args() method"""
        spec = boost_tree()
        spec = spec.set_args(trees=200, learn_rate=0.05)

        assert spec.args == {"trees": 200, "learn_rate": 0.05}

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = boost_tree(trees=100)
        spec2 = spec1.set_args(trees=200)

        # Original spec should be unchanged
        assert spec1.args == {"trees": 100}
        # New spec should have new value
        assert spec2.args == {"trees": 200}


class TestXGBoostEngine:
    """Test boost_tree with XGBoost engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 190, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 19, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 9.5, 12.5],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.8, 5],
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data
        assert fit.blueprint is not None

    def test_fit_xgboost(self, train_data):
        """Test XGBoost fitting"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        # Should use XGBRegressor
        assert fit.fit_data["model_class"] == "XGBRegressor"

    def test_fit_with_all_params(self, train_data):
        """Test fitting with all parameters"""
        spec = boost_tree(
            trees=20,
            tree_depth=4,
            learn_rate=0.1,
            mtry=2,
            min_n=5,
            loss_reduction=0.01,
            sample_size=0.8
        )
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert fit.fit_data["model_class"] == "XGBRegressor"
        # n_features includes intercept, so 3 predictors + 1 intercept = 4
        assert fit.fit_data["n_features"] == 4

    def test_predict_basic(self, train_data):
        """Test basic prediction"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
            "x3": [2.5, 4.5],
        })

        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_predict_values_reasonable(self, train_data):
        """Test that predictions are in reasonable range"""
        spec = boost_tree(trees=50, learn_rate=0.1)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        test = pd.DataFrame({
            "x1": [15],
            "x2": [7],
            "x3": [3],
        })

        predictions = fit.predict(test)

        # For middle values, prediction should be in reasonable range
        assert 50 < predictions[".pred"].iloc[0] < 350

    def test_extract_outputs(self, train_data):
        """Test extract_outputs() returns three DataFrames"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, importance, stats = fit.extract_outputs()

        # Check all three DataFrames exist
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(importance, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_structure(self, train_data):
        """Test Outputs DataFrame structure"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, _, _ = fit.extract_outputs()

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

    def test_extract_feature_importance(self, train_data):
        """Test Feature Importance DataFrame"""
        spec = boost_tree(trees=20)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        _, importance, _ = fit.extract_outputs()

        # Check structure
        assert "variable" in importance.columns
        assert "importance" in importance.columns
        # Should have 4 features (including intercept)
        assert len(importance) == 4
        # All importance values should be non-negative
        assert all(importance["importance"] >= 0)

    def test_extract_stats(self, train_data):
        """Test Stats DataFrame"""
        spec = boost_tree(trees=10)
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        _, _, stats = fit.extract_outputs()

        # Check structure
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check for key metrics
        metrics = stats["metric"].tolist()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics
        assert "model_type" in metrics
        assert "n_trees" in metrics

    def test_evaluate_and_extract(self, train_data):
        """Test evaluate() and extract_outputs() with test data"""
        # Split data
        train = train_data.iloc[:7]
        test = train_data.iloc[7:]

        spec = boost_tree(trees=10)
        fit = spec.fit(train, "y ~ x1 + x2 + x3")
        fit = fit.evaluate(test)

        outputs, importance, stats = fit.extract_outputs()

        # Outputs should have both train and test splits
        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert len(outputs[outputs["split"] == "train"]) == 7
        assert len(outputs[outputs["split"] == "test"]) == 3

        # Stats should have both train and test metrics
        train_stats = stats[stats["split"] == "train"]
        test_stats = stats[stats["split"] == "test"]
        assert len(train_stats) > 0
        assert len(test_stats) > 0

        # Check that both have RMSE
        assert "rmse" in train_stats["metric"].values
        assert "rmse" in test_stats["metric"].values


class TestLightGBMEngine:
    """Test boost_tree with LightGBM engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 190, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 19, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 9.5, 12.5],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.8, 5],
        })

    def test_fit_with_lightgbm(self, train_data):
        """Test fitting with LightGBM engine"""
        spec = boost_tree(trees=10).set_engine("lightgbm")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec.engine == "lightgbm"
        assert fit.fit_data["model_class"] == "LGBMRegressor"

    def test_fit_with_all_params(self, train_data):
        """Test fitting with all parameters"""
        spec = boost_tree(
            trees=20,
            tree_depth=4,
            learn_rate=0.1,
            mtry=2,
            min_n=5,
            loss_reduction=0.01,
            sample_size=0.8
        ).set_engine("lightgbm")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert fit.fit_data["model_class"] == "LGBMRegressor"
        # n_features includes intercept, so 3 predictors + 1 intercept = 4
        assert fit.fit_data["n_features"] == 4

    def test_predict_numeric(self, train_data):
        """Test numeric predictions"""
        spec = boost_tree(trees=10).set_engine("lightgbm")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        test = pd.DataFrame({
            "x1": [12, 22, 28],
            "x2": [6, 11, 14],
            "x3": [2.5, 4.5, 5.5],
        })

        predictions = fit.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3
        assert all(predictions[".pred"].notna())

    def test_extract_outputs_lightgbm(self, train_data):
        """Test extract_outputs with LightGBM"""
        spec = boost_tree(trees=10).set_engine("lightgbm")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, importance, stats = fit.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert len(outputs) == 10

        # Check feature importance
        assert isinstance(importance, pd.DataFrame)
        assert "variable" in importance.columns
        assert "importance" in importance.columns
        # Should have 4 features (including intercept)
        assert len(importance) == 4

        # Check stats
        assert isinstance(stats, pd.DataFrame)
        metrics = stats["metric"].tolist()
        assert "rmse" in metrics
        assert "model_type" in metrics


class TestCatBoostEngine:
    """Test boost_tree with CatBoost engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 190, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 19, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 9.5, 12.5],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.8, 5],
        })

    def test_fit_with_catboost(self, train_data):
        """Test fitting with CatBoost engine"""
        spec = boost_tree(trees=10).set_engine("catboost")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec.engine == "catboost"
        assert fit.fit_data["model_class"] == "CatBoostRegressor"

    def test_fit_with_all_params(self, train_data):
        """Test fitting with all parameters (except loss_reduction)"""
        spec = boost_tree(
            trees=20,
            tree_depth=4,
            learn_rate=0.1,
            mtry=2,
            min_n=5,
            sample_size=0.8
        ).set_engine("catboost")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert fit.fit_data["model_class"] == "CatBoostRegressor"
        # n_features includes intercept, so 3 predictors + 1 intercept = 4
        assert fit.fit_data["n_features"] == 4

    def test_predict_numeric(self, train_data):
        """Test numeric predictions"""
        spec = boost_tree(trees=10).set_engine("catboost")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        test = pd.DataFrame({
            "x1": [12, 22, 28],
            "x2": [6, 11, 14],
            "x3": [2.5, 4.5, 5.5],
        })

        predictions = fit.predict(test, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 3
        assert all(predictions[".pred"].notna())

    def test_extract_outputs_catboost(self, train_data):
        """Test extract_outputs with CatBoost"""
        spec = boost_tree(trees=10).set_engine("catboost")
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        outputs, importance, stats = fit.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert len(outputs) == 10

        # Check feature importance
        assert isinstance(importance, pd.DataFrame)
        assert "variable" in importance.columns
        assert "importance" in importance.columns
        # Should have 4 features (including intercept)
        assert len(importance) == 4

        # Check stats
        assert isinstance(stats, pd.DataFrame)
        metrics = stats["metric"].tolist()
        assert "rmse" in metrics
        assert "model_type" in metrics


class TestIntegration:
    """Integration tests for full workflow"""

    @pytest.fixture
    def train_data(self):
        """Create larger training dataset"""
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(10, 30, n)
        x2 = np.random.uniform(5, 15, n)
        x3 = np.random.uniform(2, 6, n)
        # Create target with some non-linear relationships
        y = 50 + 5 * x1 + 3 * x2 + 2 * x3 + 0.1 * x1 * x2 + np.random.normal(0, 10, n)

        return pd.DataFrame({
            "sales": y,
            "price": x1,
            "advertising": x2,
            "competition": x3,
        })

    def test_full_workflow_xgboost(self, train_data):
        """Test complete workflow with XGBoost"""
        # Split data
        train = train_data.iloc[:40]
        test = train_data.iloc[40:]

        # Create spec and fit
        spec = boost_tree(trees=50, learn_rate=0.1)
        fit = spec.fit(train, "sales ~ price + advertising + competition")

        # Predict
        predictions = fit.predict(test)

        # Verify
        assert len(predictions) == 10
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)

        # Evaluate and extract
        fit = fit.evaluate(test)
        outputs, importance, stats = fit.extract_outputs()

        # Check outputs has both splits
        assert len(outputs[outputs["split"] == "train"]) == 40
        assert len(outputs[outputs["split"] == "test"]) == 10

        # Check feature importance is sorted
        assert importance["importance"].iloc[0] >= importance["importance"].iloc[1]

    def test_full_workflow_lightgbm(self, train_data):
        """Test complete workflow with LightGBM"""
        train = train_data.iloc[:40]
        test = train_data.iloc[40:]

        spec = boost_tree(trees=50, learn_rate=0.1).set_engine("lightgbm")
        fit = spec.fit(train, "sales ~ price + advertising + competition")

        predictions = fit.predict(test)
        assert len(predictions) == 10
        assert ".pred" in predictions.columns

    def test_full_workflow_catboost(self, train_data):
        """Test complete workflow with CatBoost"""
        train = train_data.iloc[:40]
        test = train_data.iloc[40:]

        spec = boost_tree(trees=50, learn_rate=0.1).set_engine("catboost")
        fit = spec.fit(train, "sales ~ price + advertising + competition")

        predictions = fit.predict(test)
        assert len(predictions) == 10
        assert ".pred" in predictions.columns

    def test_categorical_variables(self, train_data):
        """Test with categorical predictors"""
        # Add categorical variable
        train_data["region"] = np.random.choice(["A", "B", "C"], len(train_data))

        train = train_data.iloc[:40]
        test = train_data.iloc[40:]

        # Test with XGBoost
        spec = boost_tree(trees=20)
        fit = spec.fit(train, "sales ~ price + advertising + region")

        predictions = fit.predict(test)
        assert len(predictions) == 10

    def test_compare_engines(self, train_data):
        """Compare predictions across all three engines"""
        train = train_data.iloc[:40]
        test = train_data.iloc[40:]

        # Fit with all three engines
        xgb_fit = boost_tree(trees=50, learn_rate=0.1).fit(
            train, "sales ~ price + advertising + competition"
        )
        lgb_fit = boost_tree(trees=50, learn_rate=0.1).set_engine("lightgbm").fit(
            train, "sales ~ price + advertising + competition"
        )
        cat_fit = boost_tree(trees=50, learn_rate=0.1).set_engine("catboost").fit(
            train, "sales ~ price + advertising + competition"
        )

        # Predict with all three
        xgb_pred = xgb_fit.predict(test)
        lgb_pred = lgb_fit.predict(test)
        cat_pred = cat_fit.predict(test)

        # All should return same structure
        assert xgb_pred.shape == lgb_pred.shape == cat_pred.shape
        assert all(xgb_pred.columns == lgb_pred.columns)
        assert all(lgb_pred.columns == cat_pred.columns)

        # All predictions should be in similar range (not testing exact equality)
        assert xgb_pred[".pred"].mean() > 0
        assert lgb_pred[".pred"].mean() > 0
        assert cat_pred[".pred"].mean() > 0


class TestParameterMapping:
    """Test parameter mapping for each engine"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })

    def test_mtry_as_integer(self, train_data):
        """Test mtry parameter with integer (count)"""
        spec = boost_tree(trees=10, mtry=1)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should fit successfully
        assert fit.fit_data["model_class"] == "XGBRegressor"

    def test_mtry_as_fraction(self, train_data):
        """Test mtry parameter with fraction"""
        spec = boost_tree(trees=10, mtry=0.5)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should fit successfully
        assert fit.fit_data["model_class"] == "XGBRegressor"

    def test_sample_size(self, train_data):
        """Test sample_size parameter"""
        spec = boost_tree(trees=10, sample_size=0.7)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        # Should fit successfully
        assert fit.fit_data["model_class"] == "XGBRegressor"

    def test_early_stopping(self, train_data):
        """Test early stopping parameter"""
        # Need more data for early stopping
        np.random.seed(42)
        large_data = pd.DataFrame({
            "y": np.random.randn(100) * 50 + 200,
            "x1": np.random.randn(100) * 10 + 20,
            "x2": np.random.randn(100) * 5 + 10,
        })

        spec = boost_tree(trees=100, stop_iter=10)
        fit = spec.fit(large_data, "y ~ x1 + x2")

        # Should fit successfully with early stopping
        assert fit.fit_data["model_class"] == "XGBRegressor"


class TestErrorHandling:
    """Test error handling"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        return pd.DataFrame({
            "y": [100, 200, 150],
            "x1": [10, 20, 15],
        })

    def test_invalid_prediction_type(self, train_data):
        """Test that invalid prediction type raises error"""
        spec = boost_tree(trees=5)
        fit = spec.fit(train_data, "y ~ x1")

        test = pd.DataFrame({"x1": [12]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fit.predict(test, type="class")

    def test_predict_without_fit(self):
        """Test that prediction without fitting raises appropriate error"""
        spec = boost_tree(trees=5)
        test = pd.DataFrame({"x1": [12]})

        # This should raise an error when trying to predict
        with pytest.raises(AttributeError):
            spec.predict(test)  # spec doesn't have predict, only fit does
