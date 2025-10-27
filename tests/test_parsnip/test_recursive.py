"""
Tests for recursive_reg model with skforecast engine

Tests cover:
- Basic model fitting and prediction
- Lag specifications (int and list)
- Different base models (linear, rand_forest)
- Differentiation parameter
- Three-DataFrame output structure
- Prediction intervals
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import recursive_reg, linear_reg, rand_forest


class TestRecursiveRegSpec:
    """Test recursive_reg model specification"""

    def test_basic_spec(self):
        """Test basic model specification creation"""
        spec = recursive_reg(base_model=linear_reg(), lags=7)

        assert spec.model_type == "recursive_reg"
        assert spec.engine == "skforecast"
        assert spec.mode == "regression"
        assert spec.args["lags"] == 7
        assert isinstance(spec.args["base_model"], type(linear_reg()))

    def test_spec_with_list_lags(self):
        """Test specification with list of specific lags"""
        spec = recursive_reg(base_model=rand_forest(), lags=[1, 7, 14])

        assert spec.args["lags"] == [1, 7, 14]

    def test_spec_with_differentiation(self):
        """Test specification with differentiation"""
        spec = recursive_reg(
            base_model=linear_reg(), lags=5, differentiation=1
        )

        assert spec.args["differentiation"] == 1

    def test_invalid_base_model(self):
        """Test that invalid base_model raises error"""
        with pytest.raises(TypeError, match="base_model must be a ModelSpec"):
            recursive_reg(base_model="not_a_model_spec", lags=7)

    def test_invalid_lags_int(self):
        """Test that invalid lags integer raises error"""
        with pytest.raises(ValueError, match="lags must be >= 1"):
            recursive_reg(base_model=linear_reg(), lags=0)

    def test_invalid_lags_list(self):
        """Test that empty lags list raises error"""
        with pytest.raises(ValueError, match="lags list cannot be empty"):
            recursive_reg(base_model=linear_reg(), lags=[])

    def test_invalid_differentiation(self):
        """Test that invalid differentiation raises error"""
        with pytest.raises(ValueError, match="differentiation must be None, 1, or 2"):
            recursive_reg(base_model=linear_reg(), lags=5, differentiation=3)


class TestRecursiveRegFit:
    """Test recursive_reg model fitting"""

    def test_fit_with_linear_model(self):
        """Test fitting recursive model with linear base model"""
        # Create time series data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit model
        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        assert fit is not None
        assert "forecaster" in fit.fit_data
        assert "y_train" in fit.fit_data
        assert len(fit.fit_data["y_train"]) == 100

    def test_fit_with_rand_forest(self):
        """Test fitting recursive model with random forest base model"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit model
        spec = recursive_reg(base_model=rand_forest(trees=50), lags=7)
        fit = spec.fit(data, "value ~ .")

        assert fit is not None
        assert "forecaster" in fit.fit_data

    def test_fit_with_specific_lags(self):
        """Test fitting with specific lag list"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit model with weekly patterns
        spec = recursive_reg(base_model=linear_reg(), lags=[1, 7, 14])
        fit = spec.fit(data, "value ~ .")

        assert fit is not None
        assert fit.fit_data["lags"] == [1, 7, 14]

    def test_fit_with_differentiation(self):
        """Test fitting with differentiation"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50  # Non-stationary

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit model with differencing
        spec = recursive_reg(
            base_model=linear_reg(), lags=7, differentiation=1
        )
        fit = spec.fit(data, "value ~ .")

        assert fit is not None
        assert fit.fit_data["differentiation"] == 1


class TestRecursiveRegPredict:
    """Test recursive_reg predictions"""

    def test_predict_numeric(self):
        """Test numeric predictions"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit and predict
        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        # Predict 14 days ahead
        future_dates = pd.date_range("2023-04-11", periods=14, freq="D")
        future_data = pd.DataFrame(index=future_dates)

        preds = fit.predict(future_data)

        assert preds is not None
        assert ".pred" in preds.columns
        assert len(preds) == 14
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_predict_intervals(self):
        """Test prediction with intervals"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit and predict
        spec = recursive_reg(base_model=rand_forest(trees=50), lags=7)
        fit = spec.fit(data, "value ~ .")

        # Predict with intervals
        future_dates = pd.date_range("2023-04-11", periods=14, freq="D")
        future_data = pd.DataFrame(index=future_dates)

        preds = fit.predict(future_data, type="pred_int")

        assert preds is not None
        assert ".pred" in preds.columns
        assert ".pred_lower" in preds.columns
        assert ".pred_upper" in preds.columns
        assert len(preds) == 14

        # Check intervals are properly ordered
        assert (preds[".pred_lower"] <= preds[".pred"]).all()
        assert (preds[".pred"] <= preds[".pred_upper"]).all()


class TestRecursiveRegOutputs:
    """Test recursive_reg three-DataFrame outputs"""

    def test_extract_outputs_structure(self):
        """Test that extract_outputs returns three DataFrames"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Fit model
        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        # Extract outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Check structure
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_dataframe_columns(self):
        """Test outputs DataFrame has required columns"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        outputs, _, _ = fit.extract_outputs()

        required_cols = ["date", "actuals", "fitted", "forecast", "residuals", "split"]
        for col in required_cols:
            assert col in outputs.columns, f"Missing column: {col}"

        assert (outputs["split"] == "train").all()

    def test_coefficients_dataframe(self):
        """Test coefficients DataFrame structure"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        _, coefficients, _ = fit.extract_outputs()

        # Should have coefficient for each lag
        assert len(coefficients) >= 7  # At least 7 lags
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Check lag variables are present
        lag_vars = [v for v in coefficients["variable"] if "lag_" in str(v)]
        assert len(lag_vars) >= 7

    def test_stats_dataframe(self):
        """Test stats DataFrame has metrics"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        spec = recursive_reg(base_model=linear_reg(), lags=7)
        fit = spec.fit(data, "value ~ .")

        _, _, stats = fit.extract_outputs()

        # Check for key metrics
        metrics = stats["metric"].values
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics
        assert "lags" in metrics
        assert "base_model" in metrics


class TestRecursiveRegIntegration:
    """Integration tests for recursive_reg"""

    def test_full_workflow_train_test(self):
        """Test complete train/test workflow"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        y = np.cumsum(np.random.randn(120)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        # Split into train/test
        train = data.iloc[:100]
        test = data.iloc[100:]

        # Fit on train
        spec = recursive_reg(base_model=rand_forest(trees=50), lags=7)
        fit = spec.fit(train, "value ~ .")

        # Predict on test
        preds = fit.predict(test)

        assert len(preds) == 20
        assert ".pred" in preds.columns

        # Evaluate
        fit = fit.evaluate(test)
        assert "test_predictions" in fit.evaluation_data

        # Extract full outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Should have both train and test splits
        splits = outputs["split"].unique()
        assert "train" in splits
        assert "test" in splits

    def test_rand_forest_feature_importances(self):
        """Test that Random Forest base model reports feature importances"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y = np.cumsum(np.random.randn(100)) + 50

        data = pd.DataFrame({"date": dates, "value": y})
        data = data.set_index("date")

        spec = recursive_reg(base_model=rand_forest(trees=50), lags=7)
        fit = spec.fit(data, "value ~ .")

        _, coefficients, _ = fit.extract_outputs()

        # Random Forest should report importances
        assert len(coefficients) > 0
        assert "coefficient" in coefficients.columns
        assert all(coefficients["coefficient"] >= 0)  # Importances are non-negative
