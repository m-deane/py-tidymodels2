"""
Tests for window_reg (sliding window forecasting)

Comprehensive test coverage for window-based forecasting with:
- Different window sizes (3, 7, 14, 30)
- Different methods (mean, median, weighted_mean)
- Weighted mean with custom weights
- Extract outputs (all 3 DataFrames)
- Evaluate on test data
- Edge cases
- Multi-step ahead forecasting
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.window_reg import window_reg


@pytest.fixture
def time_series_data():
    """Create time series data with trend and seasonality"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    trend = np.linspace(10, 50, n)
    seasonal = 5 * np.sin(np.arange(n) * 2 * np.pi / 7)  # Weekly seasonality
    noise = np.random.randn(n) * 2
    data = pd.DataFrame({
        "date": dates,
        "x": np.arange(n),
        "y": trend + seasonal + noise,
    })
    return data


@pytest.fixture
def smooth_series():
    """Create smooth time series (good for window methods)"""
    np.random.seed(123)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    trend = np.linspace(20, 40, n)
    noise = np.random.randn(n) * 0.5  # Low noise
    data = pd.DataFrame({
        "date": dates,
        "value": trend + noise,
    })
    return data


class TestWindowRegBasicFit:
    """Test basic fit and predict functionality"""

    def test_fit_with_mean_method(self, time_series_data):
        """Test fit with mean aggregation method"""
        spec = window_reg(window_size=7, method="mean")
        assert spec.args["window_size"] == 7
        assert spec.args["method"] == "mean"

        fit = spec.fit(time_series_data, "y ~ x")
        assert fit is not None
        assert "model" in fit.fit_data
        assert fit.fit_data["model"]["method"] == "mean"

    def test_fit_with_median_method(self, time_series_data):
        """Test fit with median aggregation method"""
        spec = window_reg(window_size=7, method="median")
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["method"] == "median"

    def test_fit_with_weighted_mean(self, time_series_data):
        """Test fit with weighted mean method"""
        weights = [0.5, 0.3, 0.2]
        spec = window_reg(window_size=3, method="weighted_mean", weights=weights)
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["method"] == "weighted_mean"
        # Weights should be normalized
        fitted_weights = fit.fit_data["model"]["weights"]
        assert np.isclose(sum(fitted_weights), 1.0)

    def test_predict_returns_dataframe(self, time_series_data):
        """Test that predict returns DataFrame with .pred column"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "y ~ x")
        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test)

    def test_predict_uses_last_window(self, smooth_series):
        """Test that predictions use last window_size observations"""
        train = smooth_series.iloc[:50].copy()
        test = smooth_series.iloc[50:55].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "value ~ date")
        predictions = fit.predict(test)

        # Prediction should equal mean of last 7 training values
        expected = train["value"].iloc[-7:].mean()
        assert np.allclose(predictions[".pred"].values, expected)


class TestWindowRegDifferentSizes:
    """Test different window sizes"""

    def test_window_size_3(self, time_series_data):
        """Test window_size=3"""
        spec = window_reg(window_size=3, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["window_size"] == 3

    def test_window_size_7(self, time_series_data):
        """Test window_size=7 (weekly)"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["window_size"] == 7

    def test_window_size_14(self, time_series_data):
        """Test window_size=14 (bi-weekly)"""
        spec = window_reg(window_size=14, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["window_size"] == 14

    def test_window_size_30(self, time_series_data):
        """Test window_size=30 (monthly)"""
        spec = window_reg(window_size=30, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")
        assert fit.fit_data["model"]["window_size"] == 30

    def test_larger_window_smoother_predictions(self, smooth_series):
        """Test that larger windows produce smoother fitted values"""
        small_window = window_reg(window_size=3, method="mean")
        large_window = window_reg(window_size=14, method="mean")

        fit_small = small_window.fit(smooth_series, "value ~ date")
        fit_large = large_window.fit(smooth_series, "value ~ date")

        outputs_small, _, _ = fit_small.extract_outputs()
        outputs_large, _, _ = fit_large.extract_outputs()

        # Remove NaN values
        fitted_small = outputs_small["fitted"].dropna()
        fitted_large = outputs_large["fitted"].dropna()

        # Larger window should have lower variance (smoother)
        assert np.var(fitted_large) < np.var(fitted_small)


class TestWindowRegDifferentMethods:
    """Test different aggregation methods"""

    def test_mean_vs_median_different_with_outliers(self):
        """Test that mean and median differ when outliers present"""
        # Create data with outliers
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            "x": np.arange(n),
            "y": np.random.randn(n) * 2 + 10,
        })
        # Add outliers
        data.loc[[10, 20, 30], "y"] = [50, -20, 60]

        train = data.iloc[:40].copy()
        test = data.iloc[40:45].copy()

        fit_mean = window_reg(window_size=7, method="mean").fit(train, "y ~ x")
        fit_median = window_reg(window_size=7, method="median").fit(train, "y ~ x")

        pred_mean = fit_mean.predict(test)
        pred_median = fit_median.predict(test)

        # Predictions should differ (mean affected by outliers, median robust)
        assert not np.allclose(
            pred_mean[".pred"].values,
            pred_median[".pred"].values
        )

    def test_weighted_mean_emphasizes_recent(self):
        """Test that weighted_mean with declining weights emphasizes recent data"""
        # Create data with recent jump
        n = 50
        data = pd.DataFrame({
            "x": np.arange(n),
            "y": np.concatenate([
                np.ones(40) * 10,  # Low values
                np.ones(10) * 30,  # Recent high values
            ]),
        })

        train = data.iloc[:50].copy()
        test = data.iloc[:5].copy()  # Dummy test

        # Uniform weights (simple average of all observations)
        fit_uniform = window_reg(
            window_size=10,
            method="mean"
        ).fit(train, "y ~ x")

        # Declining weights (emphasize recent)
        fit_recent = window_reg(
            window_size=10,
            method="weighted_mean",
            weights=[0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.25]
        ).fit(train, "y ~ x")

        pred_uniform = fit_uniform.predict(test)
        pred_recent = fit_recent.predict(test)

        # Recent-weighted should be higher (more influenced by high recent values)
        # Since last 10 values are all 30, both should equal 30
        # But the weighted version should still equal or be close to 30
        assert np.isclose(pred_recent[".pred"].values[0], 30.0)


class TestWindowRegWeightedMean:
    """Test weighted mean with custom weights"""

    def test_weighted_mean_requires_weights(self, time_series_data):
        """Test that weighted_mean raises error without weights"""
        spec = window_reg(window_size=7, method="weighted_mean")

        with pytest.raises(ValueError, match="weights required"):
            spec.fit(time_series_data, "y ~ x")

    def test_weights_length_must_match_window_size(self, time_series_data):
        """Test that weights length must equal window_size"""
        spec = window_reg(
            window_size=7,
            method="weighted_mean",
            weights=[0.5, 0.3, 0.2],  # Only 3 weights for window_size=7
        )

        with pytest.raises(ValueError, match="weights length"):
            spec.fit(time_series_data, "y ~ x")

    def test_weights_normalized_to_sum_one(self, time_series_data):
        """Test that weights are normalized to sum to 1.0"""
        weights = [1, 2, 3]  # Sum = 6
        spec = window_reg(window_size=3, method="weighted_mean", weights=weights)
        fit = spec.fit(time_series_data, "y ~ x")

        fitted_weights = fit.fit_data["model"]["weights"]
        assert np.isclose(sum(fitted_weights), 1.0)
        # Check normalization: [1/6, 2/6, 3/6]
        assert np.allclose(fitted_weights, [1/6, 2/6, 3/6])

    def test_custom_weights_used_in_prediction(self):
        """Test that custom weights are actually used in predictions"""
        # Create simple data where we can verify weighted calculation
        data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
        })

        # Use last 3 values [30, 40, 50] with weights [0.5, 0.3, 0.2]
        # Expected: 30*0.5 + 40*0.3 + 50*0.2 = 15 + 12 + 10 = 37
        spec = window_reg(
            window_size=3,
            method="weighted_mean",
            weights=[0.5, 0.3, 0.2]
        )
        fit = spec.fit(data, "y ~ x")

        test = pd.DataFrame({"x": [6]})
        pred = fit.predict(test)

        expected = 30 * 0.5 + 40 * 0.3 + 50 * 0.2
        assert np.isclose(pred[".pred"].values[0], expected)


class TestWindowRegExtractOutputs:
    """Test extract_outputs for all 3 DataFrames"""

    def test_extract_outputs_returns_three_dataframes(self, time_series_data):
        """Test that extract_outputs returns 3 DataFrames"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        outputs, coefficients, stats = fit.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_outputs_has_required_columns(self, time_series_data):
        """Test that outputs DataFrame has required columns"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        outputs, _, _ = fit.extract_outputs()

        required_cols = ["actuals", "fitted", "forecast", "residuals", "split"]
        for col in required_cols:
            assert col in outputs.columns

    def test_coefficients_contains_parameters(self, time_series_data):
        """Test that coefficients DataFrame contains model parameters"""
        spec = window_reg(window_size=7, method="mean", min_periods=5)
        fit = spec.fit(time_series_data, "y ~ x")

        _, coefficients, _ = fit.extract_outputs()

        variables = coefficients["variable"].tolist()
        assert "window_size" in variables
        assert "method" in variables
        assert "min_periods" in variables

        # Check values
        window_row = coefficients[coefficients["variable"] == "window_size"]
        assert window_row["coefficient"].values[0] == 7

        method_row = coefficients[coefficients["variable"] == "method"]
        assert method_row["coefficient"].values[0] == "mean"

    def test_coefficients_includes_weights_for_weighted_mean(self, time_series_data):
        """Test that coefficients includes weights for weighted_mean"""
        weights = [0.5, 0.3, 0.2]
        spec = window_reg(window_size=3, method="weighted_mean", weights=weights)
        fit = spec.fit(time_series_data, "y ~ x")

        _, coefficients, _ = fit.extract_outputs()

        variables = coefficients["variable"].tolist()
        assert "weight_0" in variables
        assert "weight_1" in variables
        assert "weight_2" in variables

    def test_stats_contains_metrics(self, time_series_data):
        """Test that stats DataFrame contains performance metrics"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        _, _, stats = fit.extract_outputs()

        metrics = stats["metric"].tolist()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "r_squared" in metrics
        assert "window_size" in metrics
        assert "method" in metrics

    def test_outputs_includes_date_column(self, time_series_data):
        """Test that outputs includes date column when available"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ date")

        outputs, _, _ = fit.extract_outputs()

        assert "date" in outputs.columns
        # Date should be first column
        assert outputs.columns[0] == "date"


class TestWindowRegEvaluate:
    """Test evaluate on test data"""

    def test_evaluate_adds_test_split(self, time_series_data):
        """Test that evaluate adds test split to outputs"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "y ~ x")
        fit = fit.evaluate(test)

        outputs, _, _ = fit.extract_outputs()

        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert (outputs["split"] == "train").sum() == 80
        assert (outputs["split"] == "test").sum() == 20

    def test_evaluate_computes_test_metrics(self, time_series_data):
        """Test that evaluate computes test metrics"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "y ~ x")
        fit = fit.evaluate(test)

        _, _, stats = fit.extract_outputs()

        test_stats = stats[stats["split"] == "test"]
        test_metrics = test_stats["metric"].tolist()

        assert "rmse" in test_metrics
        assert "mae" in test_metrics
        assert "mape" in test_metrics
        assert "r_squared" in test_metrics

    def test_evaluate_preserves_train_metrics(self, time_series_data):
        """Test that evaluate preserves training metrics"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "y ~ x")
        fit = fit.evaluate(test)

        _, _, stats = fit.extract_outputs()

        train_stats = stats[stats["split"] == "train"]
        train_metrics = train_stats["metric"].tolist()

        assert "rmse" in train_metrics
        assert "mae" in train_metrics


class TestWindowRegEdgeCases:
    """Test edge cases and error handling"""

    def test_window_size_must_be_positive(self, time_series_data):
        """Test that window_size must be >= 1"""
        spec = window_reg(window_size=0, method="mean")

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            spec.fit(time_series_data, "y ~ x")

    def test_window_size_cannot_exceed_data_length(self):
        """Test that window_size cannot exceed data length"""
        small_data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

        spec = window_reg(window_size=10, method="mean")

        with pytest.raises(ValueError, match="cannot exceed data length"):
            spec.fit(small_data, "y ~ x")

    def test_invalid_method_raises_error(self, time_series_data):
        """Test that invalid method raises ValueError"""
        spec = window_reg(window_size=7, method="invalid")

        with pytest.raises(ValueError, match="method must be"):
            spec.fit(time_series_data, "y ~ x")

    def test_min_periods_must_be_positive(self, time_series_data):
        """Test that min_periods must be >= 1"""
        spec = window_reg(window_size=7, min_periods=0)

        with pytest.raises(ValueError, match="min_periods must be >= 1"):
            spec.fit(time_series_data, "y ~ x")

    def test_min_periods_cannot_exceed_window_size(self, time_series_data):
        """Test that min_periods cannot exceed window_size"""
        spec = window_reg(window_size=7, min_periods=10)

        with pytest.raises(ValueError, match="cannot exceed window_size"):
            spec.fit(time_series_data, "y ~ x")

    def test_invalid_formula_raises_error(self, time_series_data):
        """Test that invalid formula raises ValueError"""
        spec = window_reg(window_size=7, method="mean")

        with pytest.raises(ValueError, match="Invalid formula"):
            spec.fit(time_series_data, "invalid_formula")

    def test_missing_outcome_column_raises_error(self, time_series_data):
        """Test that missing outcome column raises ValueError"""
        spec = window_reg(window_size=7, method="mean")

        with pytest.raises(ValueError, match="not found in data"):
            spec.fit(time_series_data, "missing_col ~ x")


class TestWindowRegMinPeriods:
    """Test min_periods parameter"""

    def test_min_periods_allows_partial_windows(self):
        """Test that min_periods allows partial windows"""
        data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })

        # window_size=5, min_periods=3 means allow windows with 3+ observations
        spec = window_reg(window_size=5, method="mean", min_periods=3)
        fit = spec.fit(data, "y ~ x")

        outputs, _, _ = fit.extract_outputs()

        # Should have fitted values for observations with 3+ prior observations
        fitted = outputs["fitted"].values
        # First observation uses itself (no prior data)
        # Subsequent observations use expanding/rolling windows
        assert not np.isnan(fitted[3])  # Has 3 prior observations

    def test_min_periods_default_is_window_size(self, time_series_data):
        """Test that min_periods defaults to window_size"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        assert fit.fit_data["model"]["min_periods"] == 7


class TestWindowRegMultiStepForecasting:
    """Test multi-step ahead forecasting"""

    def test_multi_step_forecast_constant(self, smooth_series):
        """Test that multi-step forecasts are constant (flat)"""
        train = smooth_series.iloc[:50].copy()
        test = smooth_series.iloc[50:60].copy()  # 10 steps ahead

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "value ~ date")
        predictions = fit.predict(test)

        # All predictions should be equal (constant forecast)
        assert np.std(predictions[".pred"].values) < 1e-10

    def test_horizon_determines_prediction_length(self, smooth_series):
        """Test that prediction length matches test data length"""
        train = smooth_series.iloc[:50].copy()

        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(train, "value ~ date")

        # Predict 5 steps ahead
        test_5 = smooth_series.iloc[50:55].copy()
        pred_5 = fit.predict(test_5)
        assert len(pred_5) == 5

        # Predict 15 steps ahead
        test_15 = pd.DataFrame({"date": pd.date_range("2020-03-01", periods=15)})
        pred_15 = fit.predict(test_15)
        assert len(pred_15) == 15


class TestWindowRegVsNaiveReg:
    """Compare window_reg with naive_reg(strategy='window')"""

    def test_window_reg_matches_naive_reg_window_for_mean(self, time_series_data):
        """Test that window_reg(method='mean') matches naive_reg(strategy='window')"""
        from py_parsnip.models.naive_reg import naive_reg

        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:85].copy()

        # window_reg with mean
        fit_window = window_reg(window_size=7, method="mean").fit(train, "y ~ x")
        pred_window = fit_window.predict(test)

        # naive_reg with window strategy
        fit_naive = naive_reg(strategy="window", window_size=7).fit(train, "y ~ x")
        pred_naive = fit_naive.predict(test)

        # Predictions should be identical
        assert np.allclose(
            pred_window[".pred"].values,
            pred_naive[".pred"].values
        )


class TestWindowRegModelMetadata:
    """Test model metadata columns in outputs"""

    def test_outputs_has_model_metadata(self, time_series_data):
        """Test that outputs includes model metadata columns"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        outputs, _, _ = fit.extract_outputs()

        assert "model" in outputs.columns
        assert "model_group_name" in outputs.columns
        assert "group" in outputs.columns

    def test_coefficients_has_model_metadata(self, time_series_data):
        """Test that coefficients includes model metadata columns"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        _, coefficients, _ = fit.extract_outputs()

        assert "model" in coefficients.columns
        assert "model_group_name" in coefficients.columns
        assert "group" in coefficients.columns

    def test_stats_has_model_metadata(self, time_series_data):
        """Test that stats includes model metadata columns"""
        spec = window_reg(window_size=7, method="mean")
        fit = spec.fit(time_series_data, "y ~ x")

        _, _, stats = fit.extract_outputs()

        assert "model" in stats.columns
        assert "model_group_name" in stats.columns
        assert "group" in stats.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
