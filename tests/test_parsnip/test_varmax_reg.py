"""
Tests for varmax_reg model specification and statsmodels engine

Tests cover:
- Model specification creation
- Engine registration
- Multivariate time series fitting (multiple outcomes)
- With/without exogenous variables
- Prediction for all outcome variables
- Extract outputs (three-DataFrame format with multiple outcomes)
- Parameter handling (AR, MA, trend)
- Error handling for insufficient outcome variables
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip.models.varmax_reg import varmax_reg
from py_parsnip.model_spec import ModelSpec, ModelFit


class TestVARMAXSpec:
    """Test varmax_reg() model specification"""

    def test_default_spec(self):
        """Test default varmax_reg specification"""
        spec = varmax_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "varmax_reg"
        assert spec.engine == "statsmodels"
        assert spec.mode == "regression"
        # Check default parameters
        assert spec.args["non_seasonal_ar"] == 1
        assert spec.args["non_seasonal_ma"] == 0
        assert spec.args["trend"] == "c"

    def test_spec_with_ar_parameter(self):
        """Test varmax_reg with AR parameter"""
        spec = varmax_reg(non_seasonal_ar=2)

        assert spec.args["non_seasonal_ar"] == 2

    def test_spec_with_ma_parameter(self):
        """Test varmax_reg with MA parameter"""
        spec = varmax_reg(non_seasonal_ma=1)

        assert spec.args["non_seasonal_ma"] == 1

    def test_spec_with_trend(self):
        """Test varmax_reg with different trend options"""
        # No trend
        spec_n = varmax_reg(trend="n")
        assert spec_n.args["trend"] == "n"

        # Constant only
        spec_c = varmax_reg(trend="c")
        assert spec_c.args["trend"] == "c"

        # Linear trend
        spec_t = varmax_reg(trend="t")
        assert spec_t.args["trend"] == "t"

        # Both constant and trend
        spec_ct = varmax_reg(trend="ct")
        assert spec_ct.args["trend"] == "ct"

    def test_spec_with_all_parameters(self):
        """Test varmax_reg with all parameters"""
        spec = varmax_reg(
            non_seasonal_ar=2,
            non_seasonal_ma=1,
            trend="ct"
        )

        assert spec.args["non_seasonal_ar"] == 2
        assert spec.args["non_seasonal_ma"] == 1
        assert spec.args["trend"] == "ct"

    def test_set_engine(self):
        """Test set_engine() method"""
        spec = varmax_reg()
        spec = spec.set_engine("statsmodels")

        assert spec.engine == "statsmodels"

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = varmax_reg(non_seasonal_ar=1)
        spec2 = spec1.set_args(non_seasonal_ar=2)

        # Original spec should be unchanged
        assert spec1.args["non_seasonal_ar"] == 1
        # New spec should have new value
        assert spec2.args["non_seasonal_ar"] == 2


class TestVARMAXFit:
    """Test varmax_reg fitting with multiple outcomes"""

    @pytest.fixture
    def bivariate_data(self):
        """Create bivariate time series data"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        # Create correlated time series
        t = np.arange(50)
        y1 = 100 + 0.5*t + np.random.normal(0, 5, 50)
        y2 = 50 + 0.3*t + 0.5*y1 + np.random.normal(0, 3, 50)

        return pd.DataFrame({
            "date": dates,
            "y1": y1,
            "y2": y2,
        })

    @pytest.fixture
    def trivariate_data(self):
        """Create trivariate time series data"""
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        t = np.arange(50)
        y1 = 100 + 0.5*t + np.random.normal(0, 5, 50)
        y2 = 50 + 0.3*y1 + np.random.normal(0, 3, 50)
        y3 = 75 + 0.2*y1 + 0.3*y2 + np.random.normal(0, 4, 50)

        return pd.DataFrame({
            "date": dates,
            "y1": y1,
            "y2": y2,
            "y3": y3,
        })

    @pytest.fixture
    def data_with_exog(self):
        """Create bivariate data with exogenous variables"""
        np.random.seed(456)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        x1 = np.linspace(1, 10, 50)
        t = np.arange(50)
        y1 = 100 + 2*x1 + 0.5*t + np.random.normal(0, 5, 50)
        y2 = 50 + 1.5*x1 + 0.3*t + np.random.normal(0, 3, 50)

        return pd.DataFrame({
            "date": dates,
            "y1": y1,
            "y2": y2,
            "x1": x1,
        })

    @pytest.fixture
    def forecast_data(self):
        """Create future dates for forecasting"""
        dates = pd.date_range("2020-02-20", periods=10, freq="D")
        return pd.DataFrame({"date": dates})

    @pytest.fixture
    def forecast_data_with_exog(self):
        """Create future data with exogenous variables"""
        dates = pd.date_range("2020-02-20", periods=10, freq="D")
        x1 = np.linspace(10, 12, 10)

        return pd.DataFrame({
            "date": dates,
            "x1": x1,
        })

    def test_fit_bivariate(self, bivariate_data):
        """Test fitting bivariate VARMAX model"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        assert isinstance(fit, ModelFit)
        assert fit.spec.model_type == "varmax_reg"
        assert "model" in fit.fit_data
        assert fit.fit_data["n_outcomes"] == 2
        assert fit.fit_data["outcome_names"] == ["y1", "y2"]

    def test_fit_trivariate(self, trivariate_data):
        """Test fitting trivariate VARMAX model"""
        spec = varmax_reg()
        fit = spec.fit(trivariate_data, formula="y1 + y2 + y3 ~ date")

        assert isinstance(fit, ModelFit)
        assert fit.fit_data["n_outcomes"] == 3
        assert fit.fit_data["outcome_names"] == ["y1", "y2", "y3"]

    def test_fit_with_parameters(self, bivariate_data):
        """Test fitting with specified AR and MA parameters"""
        spec = varmax_reg(non_seasonal_ar=2, non_seasonal_ma=1)
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        assert fit.fit_data["order"] == (2, 1)

    def test_fit_with_trend(self, bivariate_data):
        """Test fitting with different trend specifications"""
        spec = varmax_reg(trend="ct")
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        assert fit.fit_data["trend"] == "ct"

    def test_fit_with_exog(self, data_with_exog):
        """Test fitting with exogenous variables"""
        spec = varmax_reg()
        fit = spec.fit(data_with_exog, formula="y1 + y2 ~ date + x1")

        assert isinstance(fit, ModelFit)
        assert "x1" in fit.fit_data["predictor_names"]

    def test_predict_numeric(self, bivariate_data, forecast_data):
        """Test numeric predictions for all outcomes"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        predictions = fit.predict(forecast_data, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        # Should have predictions for both outcomes
        assert ".pred_y1" in predictions.columns
        assert ".pred_y2" in predictions.columns
        assert len(predictions) == len(forecast_data)

    def test_predict_conf_int(self, bivariate_data, forecast_data):
        """Test prediction intervals for all outcomes"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        predictions = fit.predict(forecast_data, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        # Should have predictions and intervals for both outcomes
        for outcome in ["y1", "y2"]:
            assert f".pred_{outcome}" in predictions.columns
            assert f".pred_{outcome}_lower" in predictions.columns
            assert f".pred_{outcome}_upper" in predictions.columns

        # Lower should be less than upper
        assert all(predictions[".pred_y1_lower"] < predictions[".pred_y1_upper"])
        assert all(predictions[".pred_y2_lower"] < predictions[".pred_y2_upper"])

    def test_predict_with_exog(self, data_with_exog, forecast_data_with_exog):
        """Test predictions with exogenous variables"""
        spec = varmax_reg()
        fit = spec.fit(data_with_exog, formula="y1 + y2 ~ date + x1")

        predictions = fit.predict(forecast_data_with_exog, type="numeric")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred_y1" in predictions.columns
        assert ".pred_y2" in predictions.columns
        assert len(predictions) == len(forecast_data_with_exog)


class TestVARMAXOutputs:
    """Test extract_outputs() for VARMAX"""

    @pytest.fixture
    def bivariate_data(self):
        """Create bivariate time series data"""
        np.random.seed(789)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        t = np.arange(50)
        y1 = 100 + 0.5*t + np.random.normal(0, 5, 50)
        y2 = 50 + 0.3*t + np.random.normal(0, 3, 50)

        return pd.DataFrame({
            "date": dates,
            "y1": y1,
            "y2": y2,
        })

    def test_extract_outputs_structure(self, bivariate_data):
        """Test that extract_outputs returns three DataFrames"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        outputs, coefficients, stats = fit.extract_outputs()

        # Should return three DataFrames
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_dataframe_multiple_outcomes(self, bivariate_data):
        """Test outputs DataFrame has separate rows for each outcome"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        outputs, _, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["outcome_variable", "actuals", "fitted", "residuals", "split"]
        assert all(col in outputs.columns for col in required_cols)

        # Should have separate rows for each outcome
        outcome_vars = outputs["outcome_variable"].unique()
        assert set(outcome_vars) == {"y1", "y2"}

        # Each outcome should have same number of rows
        y1_rows = outputs[outputs["outcome_variable"] == "y1"]
        y2_rows = outputs[outputs["outcome_variable"] == "y2"]
        assert len(y1_rows) == len(y2_rows) == len(bivariate_data)

    def test_coefficients_dataframe(self, bivariate_data):
        """Test coefficients DataFrame"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        _, coefficients, _ = fit.extract_outputs()

        # Check required columns
        required_cols = ["variable", "coefficient"]
        assert all(col in coefficients.columns for col in required_cols)

        # VARMAX models have parameters (AR, MA, intercept, etc.)
        assert len(coefficients) > 0

    def test_stats_dataframe(self, bivariate_data):
        """Test stats DataFrame structure"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        _, _, stats = fit.extract_outputs()

        # Check required columns
        required_cols = ["metric", "value", "split"]
        assert all(col in stats.columns for col in required_cols)

        # Check for key metrics
        metric_names = stats["metric"].tolist()
        assert "formula" in metric_names
        assert "model_type" in metric_names
        assert "order" in metric_names
        assert "n_outcomes" in metric_names

    def test_stats_include_n_outcomes(self, bivariate_data):
        """Test that stats include number of outcomes"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")

        _, _, stats = fit.extract_outputs()

        # Extract n_outcomes
        n_outcomes_row = stats[stats["metric"] == "n_outcomes"]
        assert len(n_outcomes_row) > 0
        assert n_outcomes_row["value"].iloc[0] == 2

    def test_outputs_with_model_name(self, bivariate_data):
        """Test that model name is included in outputs"""
        spec = varmax_reg()
        fit = spec.fit(bivariate_data, formula="y1 + y2 ~ date")
        fit.model_name = "my_varmax"

        outputs, coefficients, stats = fit.extract_outputs()

        # All three DataFrames should have model column
        assert "model" in outputs.columns
        assert "model" in coefficients.columns
        assert "model" in stats.columns

        # Should use the model_name
        assert all(outputs["model"] == "my_varmax")
        assert all(coefficients["model"] == "my_varmax")
        assert all(stats["model"] == "my_varmax")


class TestVARMAXErrors:
    """Test error handling for VARMAX"""

    @pytest.fixture
    def univariate_data(self):
        """Create univariate data (invalid for VARMAX)"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = np.linspace(100, 130, 30) + np.random.normal(0, 2, 30)

        return pd.DataFrame({
            "date": dates,
            "y": y,
        })

    def test_error_single_outcome(self, univariate_data):
        """Test that VARMAX requires at least 2 outcomes"""
        spec = varmax_reg()

        # Should raise error with single outcome
        with pytest.raises(ValueError, match="VARMAX requires at least 2 outcome variables"):
            spec.fit(univariate_data, formula="y ~ date")

    def test_invalid_prediction_type(self, ):
        """Test invalid prediction type raises error"""
        np.random.seed(999)
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y1 = np.random.normal(100, 10, 30)
        y2 = np.random.normal(50, 5, 30)

        data = pd.DataFrame({"date": dates, "y1": y1, "y2": y2})

        spec = varmax_reg()
        fit = spec.fit(data, formula="y1 + y2 ~ date")

        forecast_data = pd.DataFrame({
            "date": pd.date_range("2020-01-31", periods=5, freq="D")
        })

        with pytest.raises(ValueError, match="supports type='numeric' or 'conf_int'"):
            fit.predict(forecast_data, type="class")
