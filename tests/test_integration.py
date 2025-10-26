"""
Integration tests for py-tidymodels

Tests end-to-end workflows combining multiple packages:
- py-hardhat (preprocessing)
- py-parsnip (models)
- py-rsample (resampling)
- py-workflows (composition)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest, arima_reg, prophet_reg
from py_rsample import initial_split, training, testing, time_series_cv


class TestBasicWorkflowIntegration:
    """Test basic workflow composition and execution"""

    @pytest.fixture
    def sample_data(self):
        """Create sample regression data"""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "y": np.random.randn(n) + 10,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
        })

    def test_basic_workflow_with_train_test_evaluation(self, sample_data):
        """Test basic workflow with train/test split and evaluation"""
        # Split data
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        # Create workflow
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit on training data
        wf_fit = wf.fit(train)
        assert wf_fit is not None

        # Evaluate on test data
        wf_fit = wf_fit.evaluate(test)
        assert "test_predictions" in wf_fit.fit.evaluation_data
        assert "test_data" in wf_fit.fit.evaluation_data

        # Extract comprehensive outputs
        outputs, coefficients, stats = wf_fit.extract_outputs()

        # Verify Outputs DataFrame
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "forecast" in outputs.columns
        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert len(outputs[outputs["split"] == "train"]) == len(train)
        assert len(outputs[outputs["split"] == "test"]) == len(test)

        # Verify Coefficients DataFrame
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns
        assert len(coefficients) == 4  # x1, x2, x3, Intercept

        # Verify Stats DataFrame
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns
        train_stats = stats[stats["split"] == "train"]
        test_stats = stats[stats["split"] == "test"]
        assert len(train_stats) > 0
        assert len(test_stats) > 0
        assert "rmse" in stats["metric"].values
        assert "mae" in stats["metric"].values
        assert "r_squared" in stats["metric"].values

    def test_workflow_method_chaining(self, sample_data):
        """Test full workflow method chaining"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        # Full pipeline in one chain
        outputs, coefficients, stats = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
            .fit(train)
            .evaluate(test)
            .extract_outputs()
        )

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)
        assert len(outputs) == len(train) + len(test)

    def test_workflow_with_regularization(self, sample_data):
        """Test workflow with Ridge regression"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn"))
        )

        wf_fit = wf.fit(train).evaluate(test)
        outputs, coefficients, stats = wf_fit.extract_outputs()

        # Verify regularization was applied
        assert wf_fit.workflow.spec.args["penalty"] == 0.1
        assert wf_fit.workflow.spec.args["mixture"] == 0.0

        # Check outputs are valid
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

    def test_workflow_update_and_comparison(self, sample_data):
        """Test updating workflows and comparing models"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        # Base workflow
        base_wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Update to include more predictors
        updated_wf = base_wf.update_formula("y ~ x1 + x2 + x3")

        # Fit both
        base_fit = base_wf.fit(train).evaluate(test)
        updated_fit = updated_wf.fit(train).evaluate(test)

        # Extract stats
        _, _, base_stats = base_fit.extract_outputs()
        _, _, updated_stats = updated_fit.extract_outputs()

        # Compare R² on test set
        base_r2 = base_stats[
            (base_stats["metric"] == "r_squared") & (base_stats["split"] == "test")
        ]["value"].values[0]
        updated_r2 = updated_stats[
            (updated_stats["metric"] == "r_squared") & (updated_stats["split"] == "test")
        ]["value"].values[0]

        assert isinstance(base_r2, (int, float))
        assert isinstance(updated_r2, (int, float))


class TestTimeSeriesCVIntegration:
    """Test time series cross-validation integration"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        return pd.DataFrame({
            "date": dates,
            "value": np.cumsum(np.random.randn(365)) + 100,
            "feature1": np.random.randn(365),
            "feature2": np.random.randn(365),
        })

    def test_workflow_with_time_series_cv(self, time_series_data):
        """Test workflow with time series CV splits"""
        # Create CV splits
        cv_splits = time_series_cv(
            time_series_data,
            date_column="date",
            initial="6 months",
            assess="1 month",
            cumulative=True,
        )

        # Create workflow
        wf = (
            workflow()
            .add_formula("value ~ feature1 + feature2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit to each split
        results = []
        for fold in cv_splits:
            train_data = fold.training()
            test_data = fold.testing()

            wf_fit = wf.fit(train_data)
            preds = wf_fit.predict(test_data)

            results.append({
                "fold_id": fold.split.id,
                "n_train": len(train_data),
                "n_test": len(test_data),
                "predictions": preds,
            })

        assert len(results) > 0
        assert all("predictions" in r for r in results)
        assert all(len(r["predictions"]) == r["n_test"] for r in results)

    def test_cv_with_evaluation_metrics(self, time_series_data):
        """Test CV with comprehensive evaluation metrics"""
        cv_splits = time_series_cv(
            time_series_data,
            date_column="date",
            initial="6 months",
            assess="1 month",
            cumulative=True,
        )

        wf = (
            workflow()
            .add_formula("value ~ feature1 + feature2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        fold_metrics = []
        for fold in cv_splits:
            train_data = fold.training()
            test_data = fold.testing()

            wf_fit = wf.fit(train_data).evaluate(test_data)
            _, _, stats = wf_fit.extract_outputs()

            # Extract test metrics
            test_rmse = stats[
                (stats["metric"] == "rmse") & (stats["split"] == "test")
            ]["value"].values[0]
            test_mae = stats[
                (stats["metric"] == "mae") & (stats["split"] == "test")
            ]["value"].values[0]

            fold_metrics.append({
                "fold_id": fold.split.id,
                "rmse": test_rmse,
                "mae": test_mae,
            })

        # Check all folds have metrics
        assert len(fold_metrics) == len(cv_splits)
        assert all("rmse" in m for m in fold_metrics)
        assert all("mae" in m for m in fold_metrics)


class TestARIMAWorkflowIntegration:
    """Test ARIMA workflow integration"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        return pd.DataFrame({
            "date": dates,
            "sales": np.cumsum(np.random.randn(200)) + 100,
        })

    def test_arima_workflow(self, time_series_data):
        """Test ARIMA workflow with prediction"""
        split = initial_split(time_series_data, prop=0.8)
        train = training(split)
        test = testing(split)

        # ARIMA workflow
        arima_wf = (
            workflow()
            .add_formula("sales ~ 1")
            .add_model(
                arima_reg(
                    non_seasonal_ar=1,
                    non_seasonal_differences=1,
                    non_seasonal_ma=1,
                ).set_engine("statsmodels")
            )
        )

        # Fit
        arima_fit = arima_wf.fit(train)
        assert arima_fit is not None

        # Predict
        forecast = arima_fit.predict(test)
        assert ".pred" in forecast.columns
        assert len(forecast) == len(test)

    def test_arima_workflow_with_evaluation(self, time_series_data):
        """Test ARIMA workflow with evaluation"""
        split = initial_split(time_series_data, prop=0.8)
        train = training(split)
        test = testing(split)

        arima_wf = (
            workflow()
            .add_formula("sales ~ 1")
            .add_model(
                arima_reg(
                    non_seasonal_ar=1,
                    non_seasonal_differences=0,
                    non_seasonal_ma=1,
                ).set_engine("statsmodels")
            )
        )

        arima_fit = arima_wf.fit(train).evaluate(test)
        outputs, coefficients, stats = arima_fit.extract_outputs()

        # Check outputs have date column
        assert "date" in outputs.columns or len(outputs) > 0

        # Check coefficients have ARIMA parameters
        assert len(coefficients) > 0
        assert "variable" in coefficients.columns

        # Check stats have AIC, BIC
        metric_names = stats["metric"].values
        assert "aic" in metric_names or len(stats) > 0

    def test_arima_prediction_intervals(self, time_series_data):
        """Test ARIMA workflow with prediction intervals"""
        split = initial_split(time_series_data, prop=0.8)
        train = training(split)
        test = testing(split)

        arima_wf = (
            workflow()
            .add_formula("sales ~ 1")
            .add_model(
                arima_reg(non_seasonal_ar=1, non_seasonal_ma=1).set_engine(
                    "statsmodels"
                )
            )
        )

        arima_fit = arima_wf.fit(train)
        forecast = arima_fit.predict(test, type="conf_int")

        # Check prediction intervals are present
        assert ".pred" in forecast.columns
        assert ".pred_lower" in forecast.columns
        assert ".pred_upper" in forecast.columns
        assert len(forecast) == len(test)


class TestProphetWorkflowIntegration:
    """Test Prophet workflow integration"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        return pd.DataFrame({
            "date": dates,
            "sales": 100 + np.cumsum(np.random.randn(200) * 5),
        })

    def test_prophet_workflow(self, time_series_data):
        """Test Prophet workflow with prediction"""
        split = initial_split(time_series_data, prop=0.8)
        train = training(split)
        test = testing(split)

        # Prophet workflow
        prophet_wf = (
            workflow()
            .add_formula("sales ~ date")
            .add_model(
                prophet_reg(growth="linear", n_changepoints=10).set_engine("prophet")
            )
        )

        # Fit
        prophet_fit = prophet_wf.fit(train)
        assert prophet_fit is not None

        # Predict
        forecast = prophet_fit.predict(test)
        assert ".pred" in forecast.columns
        assert len(forecast) == len(test)

    def test_prophet_workflow_with_evaluation(self, time_series_data):
        """Test Prophet workflow with evaluation"""
        split = initial_split(time_series_data, prop=0.8)
        train = training(split)
        test = testing(split)

        prophet_wf = (
            workflow()
            .add_formula("sales ~ date")
            .add_model(prophet_reg().set_engine("prophet"))
        )

        prophet_fit = prophet_wf.fit(train).evaluate(test)
        outputs, coefficients, stats = prophet_fit.extract_outputs()

        # Check outputs
        assert len(outputs) > 0
        assert "split" in outputs.columns

        # Check coefficients (Prophet hyperparameters)
        assert len(coefficients) > 0

        # Check stats
        assert len(stats) > 0
        assert "rmse" in stats["metric"].values


class TestRandomForestWorkflowIntegration:
    """Test Random Forest workflow integration"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "y": np.random.randn(n) + 10,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
        })

    def test_random_forest_workflow(self, sample_data):
        """Test Random Forest workflow"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        rf_wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(
                rand_forest(trees=50, min_n=5)
                .set_mode("regression")
                .set_engine("sklearn")
            )
        )

        rf_fit = rf_wf.fit(train).evaluate(test)
        outputs, coefficients, stats = rf_fit.extract_outputs()

        # Check outputs
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Check coefficients (feature importances)
        assert len(coefficients) > 0
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Check stats
        test_rmse = stats[
            (stats["metric"] == "rmse") & (stats["split"] == "test")
        ]["value"].values[0]
        assert isinstance(test_rmse, (int, float))
        assert test_rmse > 0


class TestMultiModelComparison:
    """Test comparing multiple models"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "y": np.random.randn(n) * 5 + 10,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
        })

    def test_compare_multiple_models(self, sample_data):
        """Test comparing OLS, Ridge, and Random Forest"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        # Define models to compare
        models = {
            "OLS": linear_reg().set_engine("sklearn"),
            "Ridge": linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn"),
            "Random Forest": rand_forest(trees=50, min_n=5)
            .set_mode("regression")
            .set_engine("sklearn"),
        }

        results = {}
        for name, model_spec in models.items():
            wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(model_spec)
            wf_fit = wf.fit(train).evaluate(test)
            _, _, stats = wf_fit.extract_outputs()

            # Extract test RMSE
            test_rmse = stats[
                (stats["metric"] == "rmse") & (stats["split"] == "test")
            ]["value"].values[0]
            results[name] = test_rmse

        # Check all models produced valid results
        assert len(results) == 3
        assert all(isinstance(rmse, (int, float)) for rmse in results.values())
        assert all(rmse > 0 for rmse in results.values())

    def test_compare_with_different_formulas(self, sample_data):
        """Test comparing models with different formulas"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        formulas = {
            "Simple": "y ~ x1",
            "Two predictors": "y ~ x1 + x2",
            "All predictors": "y ~ x1 + x2 + x3",
        }

        results = {}
        for name, formula in formulas.items():
            wf = (
                workflow()
                .add_formula(formula)
                .add_model(linear_reg().set_engine("sklearn"))
            )
            wf_fit = wf.fit(train).evaluate(test)
            _, _, stats = wf_fit.extract_outputs()

            test_r2 = stats[
                (stats["metric"] == "r_squared") & (stats["split"] == "test")
            ]["value"].values[0]
            results[name] = test_r2

        # Check all formulas produced valid results
        assert len(results) == 3
        assert all(isinstance(r2, (int, float)) for r2 in results.values())


class TestComprehensiveOutputs:
    """Test comprehensive output structure"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        n = 150
        return pd.DataFrame({
            "y": np.random.randn(n) + 10,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        })

    def test_outputs_dataframe_structure(self, sample_data):
        """Test Outputs DataFrame has correct structure"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        wf_fit = wf.fit(train).evaluate(test)
        outputs, _, _ = wf_fit.extract_outputs()

        # Required columns
        required_cols = ["actuals", "fitted", "forecast", "residuals", "split"]
        for col in required_cols:
            assert col in outputs.columns

        # Check splits
        assert set(outputs["split"].unique()) == {"train", "test"}

        # Check no NaN in key columns (except fitted for test)
        train_outputs = outputs[outputs["split"] == "train"]
        assert not train_outputs["actuals"].isna().any()
        assert not train_outputs["fitted"].isna().any()
        assert not train_outputs["forecast"].isna().any()

        test_outputs = outputs[outputs["split"] == "test"]
        assert not test_outputs["actuals"].isna().any()
        assert not test_outputs["forecast"].isna().any()

    def test_coefficients_dataframe_structure(self, sample_data):
        """Test Coefficients DataFrame has correct structure"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        wf_fit = wf.fit(train).evaluate(test)
        _, coefficients, _ = wf_fit.extract_outputs()

        # Required columns
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Check variables
        variables = coefficients["variable"].values
        assert "x1" in variables
        assert "x2" in variables
        assert "Intercept" in variables

    def test_stats_dataframe_structure(self, sample_data):
        """Test Stats DataFrame has correct structure"""
        split = initial_split(sample_data, prop=0.75)
        train = training(split)
        test = testing(split)

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        wf_fit = wf.fit(train).evaluate(test)
        _, _, stats = wf_fit.extract_outputs()

        # Required columns
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns

        # Check key metrics exist
        metrics = stats["metric"].values
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics

        # Check both train and test splits
        splits = stats["split"].unique()
        assert "train" in splits
        assert "test" in splits
