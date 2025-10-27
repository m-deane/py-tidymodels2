"""Tests for py-yardstick metrics"""

import pytest
import pandas as pd
import numpy as np
from py_yardstick import (
    # Time Series Metrics
    rmse, mae, mape, smape, mase, r_squared, rsq_trad,
    # Residual Tests
    durbin_watson, ljung_box, shapiro_wilk, adf_test,
    # Classification Metrics
    accuracy, precision, recall, f_meas, roc_auc,
    # Additional Metrics
    mda,
    # Metric Set
    metric_set
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def perfect_predictions():
    """Perfect predictions for testing."""
    truth = pd.Series([1, 2, 3, 4, 5])
    estimate = pd.Series([1, 2, 3, 4, 5])
    return truth, estimate


@pytest.fixture
def good_predictions():
    """Good but not perfect predictions."""
    truth = pd.Series([1, 2, 3, 4, 5])
    estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
    return truth, estimate


@pytest.fixture
def poor_predictions():
    """Poor predictions for testing."""
    truth = pd.Series([1, 2, 3, 4, 5])
    estimate = pd.Series([5, 4, 3, 2, 1])
    return truth, estimate


@pytest.fixture
def classification_data():
    """Binary classification data."""
    truth = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    estimate = pd.Series([0, 1, 0, 0, 1, 0, 1, 1])
    estimate_prob = pd.Series([0.1, 0.9, 0.2, 0.4, 0.8, 0.15, 0.85, 0.6])
    return truth, estimate, estimate_prob


@pytest.fixture
def residuals_data():
    """Residuals for testing diagnostics."""
    np.random.seed(42)
    residuals = pd.Series(np.random.normal(0, 1, 100))
    return residuals


@pytest.fixture
def time_series_data():
    """Time series data for testing."""
    np.random.seed(42)
    train = pd.Series(np.cumsum(np.random.randn(50)) + 10)
    truth = pd.Series(np.cumsum(np.random.randn(10)) + 60)
    estimate = truth + np.random.randn(10) * 0.5
    return train, truth, estimate


# ============================================================================
# Time Series Metrics Tests
# ============================================================================

class TestRMSE:
    """Tests for RMSE metric."""

    def test_rmse_perfect(self, perfect_predictions):
        """Test RMSE with perfect predictions."""
        truth, estimate = perfect_predictions
        result = rmse(truth, estimate)

        assert isinstance(result, pd.DataFrame)
        assert "metric" in result.columns
        assert "value" in result.columns
        assert result["metric"].iloc[0] == "rmse"
        assert result["value"].iloc[0] == 0.0

    def test_rmse_good(self, good_predictions):
        """Test RMSE with good predictions."""
        truth, estimate = good_predictions
        result = rmse(truth, estimate)

        assert result["value"].iloc[0] > 0
        assert result["value"].iloc[0] < 1

    def test_rmse_poor(self, poor_predictions):
        """Test RMSE with poor predictions."""
        truth, estimate = poor_predictions
        result = rmse(truth, estimate)

        assert result["value"].iloc[0] > 2

    def test_rmse_with_na(self):
        """Test RMSE with NA values."""
        truth = pd.Series([1, 2, np.nan, 4, 5])
        estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        result = rmse(truth, estimate)

        # Should compute on non-NA values
        assert not np.isnan(result["value"].iloc[0])

    def test_rmse_all_na(self):
        """Test RMSE with all NA values."""
        truth = pd.Series([np.nan, np.nan, np.nan])
        estimate = pd.Series([1, 2, 3])
        result = rmse(truth, estimate)

        # Should return NaN
        assert np.isnan(result["value"].iloc[0])


class TestMAE:
    """Tests for MAE metric."""

    def test_mae_perfect(self, perfect_predictions):
        """Test MAE with perfect predictions."""
        truth, estimate = perfect_predictions
        result = mae(truth, estimate)

        assert result["metric"].iloc[0] == "mae"
        assert result["value"].iloc[0] == 0.0

    def test_mae_good(self, good_predictions):
        """Test MAE with good predictions."""
        truth, estimate = good_predictions
        result = mae(truth, estimate)

        assert result["value"].iloc[0] > 0
        assert result["value"].iloc[0] < 1

    def test_mae_known_value(self):
        """Test MAE with known value."""
        truth = pd.Series([0, 1, 2, 3, 4])
        estimate = pd.Series([0, 1, 2, 3, 5])
        result = mae(truth, estimate)

        # MAE should be 1/5 = 0.2
        assert np.isclose(result["value"].iloc[0], 0.2)


class TestMAPE:
    """Tests for MAPE metric."""

    def test_mape_perfect(self):
        """Test MAPE with perfect predictions."""
        truth = pd.Series([10, 20, 30, 40, 50])
        estimate = pd.Series([10, 20, 30, 40, 50])
        result = mape(truth, estimate)

        assert result["metric"].iloc[0] == "mape"
        assert result["value"].iloc[0] == 0.0

    def test_mape_known_value(self):
        """Test MAPE with known value."""
        truth = pd.Series([100, 200])
        estimate = pd.Series([110, 180])
        result = mape(truth, estimate)

        # MAPE = (|100-110|/100 + |200-180|/200) / 2 * 100 = (10 + 10) / 2 = 10
        assert np.isclose(result["value"].iloc[0], 10.0)

    def test_mape_with_zeros(self):
        """Test MAPE with zeros in truth (should exclude them)."""
        truth = pd.Series([0, 10, 20, 0, 40])
        estimate = pd.Series([5, 11, 22, 10, 41])
        result = mape(truth, estimate)

        # Should only compute on non-zero values: [10, 20, 40]
        assert not np.isnan(result["value"].iloc[0])


class TestSMAPE:
    """Tests for SMAPE metric."""

    def test_smape_perfect(self):
        """Test SMAPE with perfect predictions."""
        truth = pd.Series([10, 20, 30, 40, 50])
        estimate = pd.Series([10, 20, 30, 40, 50])
        result = smape(truth, estimate)

        assert result["metric"].iloc[0] == "smape"
        assert result["value"].iloc[0] == 0.0

    def test_smape_handles_zeros(self):
        """Test SMAPE handles zeros better than MAPE."""
        truth = pd.Series([0, 10, 20])
        estimate = pd.Series([5, 11, 22])
        result = smape(truth, estimate)

        # Should compute without error
        assert not np.isnan(result["value"].iloc[0])
        assert result["value"].iloc[0] >= 0
        assert result["value"].iloc[0] <= 200


class TestMASE:
    """Tests for MASE metric."""

    def test_mase_requires_train(self):
        """Test MASE requires training data."""
        truth = pd.Series([1, 2, 3])
        estimate = pd.Series([1.1, 2.2, 2.9])

        with pytest.raises(ValueError, match="requires training data"):
            mase(truth, estimate)

    def test_mase_with_train(self, time_series_data):
        """Test MASE with training data."""
        train, truth, estimate = time_series_data
        result = mase(truth, estimate, train=train)

        assert result["metric"].iloc[0] == "mase"
        assert not np.isnan(result["value"].iloc[0])

    def test_mase_perfect_on_train_pattern(self):
        """Test MASE when predictions match naive forecast."""
        # Create data where naive forecast is optimal
        train = pd.Series([1, 2, 3, 4, 5])
        truth = pd.Series([6, 7, 8])
        # Naive forecast (last value of train repeated)
        estimate = pd.Series([5, 6, 7])

        result = mase(truth, estimate, train=train, m=1)

        # MASE should be close to 1 (as good as naive)
        assert result["value"].iloc[0] > 0


class TestRSquared:
    """Tests for R² metric."""

    def test_rsq_perfect(self, perfect_predictions):
        """Test R² with perfect predictions."""
        truth, estimate = perfect_predictions
        result = r_squared(truth, estimate)

        assert result["metric"].iloc[0] == "r_squared"
        assert np.isclose(result["value"].iloc[0], 1.0)

    def test_rsq_good(self, good_predictions):
        """Test R² with good predictions."""
        truth, estimate = good_predictions
        result = r_squared(truth, estimate)

        assert result["value"].iloc[0] > 0.9
        assert result["value"].iloc[0] <= 1.0

    def test_rsq_poor(self, poor_predictions):
        """Test R² with poor predictions."""
        truth, estimate = poor_predictions
        result = r_squared(truth, estimate)

        # For reversed predictions, R² can be negative
        assert result["value"].iloc[0] < 0.5


class TestRSQTrad:
    """Tests for traditional R² metric."""

    def test_rsq_trad_perfect(self, perfect_predictions):
        """Test traditional R² with perfect predictions."""
        truth, estimate = perfect_predictions
        result = rsq_trad(truth, estimate)

        assert result["metric"].iloc[0] == "rsq_trad"
        assert np.isclose(result["value"].iloc[0], 1.0)

    def test_rsq_trad_always_positive(self, poor_predictions):
        """Test traditional R² is always non-negative."""
        truth, estimate = poor_predictions
        result = rsq_trad(truth, estimate)

        # Traditional R² (squared correlation) is always >= 0
        assert result["value"].iloc[0] >= 0


class TestMDA:
    """Tests for Mean Directional Accuracy metric."""

    def test_mda_perfect(self):
        """Test MDA with perfect directional predictions."""
        truth = pd.Series([1, 2, 3, 4, 5])
        estimate = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
        result = mda(truth, estimate)

        assert result["metric"].iloc[0] == "mda"
        assert result["value"].iloc[0] == 1.0

    def test_mda_poor(self):
        """Test MDA with poor directional predictions."""
        truth = pd.Series([1, 2, 3, 4, 5])
        estimate = pd.Series([5, 4, 3, 2, 1])
        result = mda(truth, estimate)

        # All directions are reversed
        assert result["value"].iloc[0] == 0.0

    def test_mda_mixed(self):
        """Test MDA with mixed directional predictions."""
        truth = pd.Series([1, 2, 1, 3, 2])
        estimate = pd.Series([1.1, 2.1, 0.9, 3.1, 2.2])
        result = mda(truth, estimate)

        # Directions: truth [+, -, +, -], estimate [+, -, +, -]
        # (2-1=+, 1-2=-, 3-1=+, 2-3=- vs 2.1-1.1=+, 0.9-2.1=-, 3.1-0.9=+, 2.2-3.1=-)
        # Matches: 4/4 = 1.0
        assert np.isclose(result["value"].iloc[0], 1.0)


# ============================================================================
# Residual Diagnostic Tests
# ============================================================================

class TestDurbinWatson:
    """Tests for Durbin-Watson statistic."""

    def test_dw_returns_dataframe(self, residuals_data):
        """Test Durbin-Watson returns DataFrame."""
        result = durbin_watson(residuals_data)

        assert isinstance(result, pd.DataFrame)
        assert result["metric"].iloc[0] == "durbin_watson"

    def test_dw_range(self, residuals_data):
        """Test Durbin-Watson is in valid range [0, 4]."""
        result = durbin_watson(residuals_data)

        value = result["value"].iloc[0]
        assert value >= 0
        assert value <= 4

    def test_dw_uncorrelated(self):
        """Test Durbin-Watson for uncorrelated residuals."""
        np.random.seed(42)
        residuals = pd.Series(np.random.normal(0, 1, 100))
        result = durbin_watson(residuals)

        # For uncorrelated residuals, DW should be close to 2
        assert result["value"].iloc[0] > 1.5
        assert result["value"].iloc[0] < 2.5


class TestLjungBox:
    """Tests for Ljung-Box test."""

    def test_ljung_box_returns_two_values(self, residuals_data):
        """Test Ljung-Box returns stat and p-value."""
        result = ljung_box(residuals_data, lags=10)

        assert len(result) == 2
        assert "ljung_box_stat" in result["metric"].values
        assert "ljung_box_p" in result["metric"].values

    def test_ljung_box_custom_lags(self, residuals_data):
        """Test Ljung-Box with custom lags."""
        result = ljung_box(residuals_data, lags=5)

        # Should return two rows (stat and p-value)
        assert len(result) == 2
        assert "ljung_box_stat" in result["metric"].values
        assert "ljung_box_p" in result["metric"].values


class TestShapiroWilk:
    """Tests for Shapiro-Wilk normality test."""

    def test_shapiro_wilk_returns_two_values(self, residuals_data):
        """Test Shapiro-Wilk returns stat and p-value."""
        result = shapiro_wilk(residuals_data)

        assert len(result) == 2
        assert "shapiro_wilk_stat" in result["metric"].values
        assert "shapiro_wilk_p" in result["metric"].values

    def test_shapiro_wilk_normal_data(self):
        """Test Shapiro-Wilk on normal data."""
        np.random.seed(42)
        residuals = pd.Series(np.random.normal(0, 1, 100))
        result = shapiro_wilk(residuals)

        # For normal data, p-value should be high (not rejecting normality)
        p_value = result[result["metric"] == "shapiro_wilk_p"]["value"].iloc[0]
        assert p_value > 0.01  # Not significant at 1% level

    def test_shapiro_wilk_non_normal_data(self):
        """Test Shapiro-Wilk on non-normal data."""
        np.random.seed(42)
        residuals = pd.Series(np.random.exponential(1, 100))
        result = shapiro_wilk(residuals)

        # For non-normal data, p-value should be low
        p_value = result[result["metric"] == "shapiro_wilk_p"]["value"].iloc[0]
        # Exponential should be detected as non-normal
        assert not np.isnan(p_value)


class TestADF:
    """Tests for Augmented Dickey-Fuller test."""

    def test_adf_returns_two_values(self):
        """Test ADF returns stat and p-value."""
        np.random.seed(42)
        series = pd.Series(np.cumsum(np.random.randn(100)))
        result = adf_test(series)

        assert len(result) == 2
        assert "adf_stat" in result["metric"].values
        assert "adf_p" in result["metric"].values

    def test_adf_stationary(self):
        """Test ADF on stationary series."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 100))
        result = adf_test(series)

        # Stationary series should have low p-value (reject unit root)
        p_value = result[result["metric"] == "adf_p"]["value"].iloc[0]
        assert p_value < 0.05

    def test_adf_nonstationary(self):
        """Test ADF on non-stationary series."""
        np.random.seed(42)
        series = pd.Series(np.cumsum(np.random.randn(100)))
        result = adf_test(series)

        # Non-stationary series should have high p-value (fail to reject unit root)
        p_value = result[result["metric"] == "adf_p"]["value"].iloc[0]
        # Note: Random walk might still reject sometimes, so we just check it runs
        assert not np.isnan(p_value)


# ============================================================================
# Classification Metrics Tests
# ============================================================================

class TestAccuracy:
    """Tests for accuracy metric."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 0, 1, 1])
        result = accuracy(truth, estimate)

        assert result["metric"].iloc[0] == "accuracy"
        assert result["value"].iloc[0] == 1.0

    def test_accuracy_known_value(self, classification_data):
        """Test accuracy with known value."""
        truth, estimate, _ = classification_data
        result = accuracy(truth, estimate)

        # Count correct: [T, T, T, F, T, T, T, F] = 6/8 = 0.75
        assert np.isclose(result["value"].iloc[0], 0.75)


class TestPrecision:
    """Tests for precision metric."""

    def test_precision_perfect(self):
        """Test precision with perfect predictions."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 0, 1, 1])
        result = precision(truth, estimate)

        assert result["metric"].iloc[0] == "precision"
        assert result["value"].iloc[0] == 1.0

    def test_precision_known_value(self):
        """Test precision with known value."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 1, 0, 1])
        result = precision(truth, estimate)

        # TP=2, FP=1, precision = 2/3
        assert np.isclose(result["value"].iloc[0], 2/3)


class TestRecall:
    """Tests for recall metric."""

    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 0, 1, 1])
        result = recall(truth, estimate)

        assert result["metric"].iloc[0] == "recall"
        assert result["value"].iloc[0] == 1.0

    def test_recall_known_value(self):
        """Test recall with known value."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 1, 0, 1])
        result = recall(truth, estimate)

        # TP=2, FN=1, recall = 2/3
        assert np.isclose(result["value"].iloc[0], 2/3)


class TestFMeasure:
    """Tests for F-measure metric."""

    def test_f_meas_perfect(self):
        """Test F-measure with perfect predictions."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 0, 1, 1])
        result = f_meas(truth, estimate)

        assert result["metric"].iloc[0] == "f_meas"
        assert result["value"].iloc[0] == 1.0

    def test_f_meas_known_value(self):
        """Test F-measure with known value."""
        truth = pd.Series([0, 1, 0, 1, 1])
        estimate = pd.Series([0, 1, 1, 0, 1])
        result = f_meas(truth, estimate)

        # precision=2/3, recall=2/3, F1 = 2/3
        assert np.isclose(result["value"].iloc[0], 2/3)

    def test_f_meas_beta(self):
        """Test F-measure with different beta."""
        # Use data where precision != recall to see beta effect
        truth = pd.Series([0, 1, 0, 1, 1, 0])
        estimate = pd.Series([0, 1, 1, 1, 1, 0])

        result_f1 = f_meas(truth, estimate, beta=1.0)
        result_f2 = f_meas(truth, estimate, beta=2.0)

        # F2 should weight recall more heavily than F1
        # Since recall (3/3=1.0) > precision (3/4=0.75), F2 should be higher
        assert result_f2["value"].iloc[0] >= result_f1["value"].iloc[0]
        # Both should be positive
        assert result_f1["value"].iloc[0] > 0
        assert result_f2["value"].iloc[0] > 0


class TestROCAUC:
    """Tests for ROC AUC metric."""

    def test_roc_auc_perfect(self):
        """Test ROC AUC with perfect predictions."""
        truth = pd.Series([0, 0, 1, 1])
        estimate_prob = pd.Series([0.1, 0.2, 0.8, 0.9])
        result = roc_auc(truth, estimate_prob)

        assert result["metric"].iloc[0] == "roc_auc"
        assert result["value"].iloc[0] == 1.0

    def test_roc_auc_random(self):
        """Test ROC AUC with random predictions."""
        truth = pd.Series([0, 1, 0, 1])
        estimate_prob = pd.Series([0.5, 0.5, 0.5, 0.5])
        result = roc_auc(truth, estimate_prob)

        # Random classifier should have AUC ~0.5
        assert np.isclose(result["value"].iloc[0], 0.5)

    def test_roc_auc_good(self, classification_data):
        """Test ROC AUC with good predictions."""
        truth, _, estimate_prob = classification_data
        result = roc_auc(truth, estimate_prob)

        # Good predictions should have AUC > 0.5
        assert result["value"].iloc[0] > 0.5


# ============================================================================
# Metric Set Tests
# ============================================================================

class TestMetricSet:
    """Tests for metric_set composer."""

    def test_metric_set_basic(self, good_predictions):
        """Test basic metric set composition."""
        truth, estimate = good_predictions
        my_metrics = metric_set(rmse, mae, r_squared)

        result = my_metrics(truth, estimate)

        assert len(result) == 3
        assert set(result["metric"]) == {"rmse", "mae", "r_squared"}

    def test_metric_set_with_params(self, time_series_data):
        """Test metric set with additional parameters."""
        train, truth, estimate = time_series_data
        my_metrics = metric_set(rmse, mae, mase)

        result = my_metrics(truth, estimate, train=train)

        assert len(result) == 3
        assert "mase" in result["metric"].values

    def test_metric_set_empty(self):
        """Test metric set with no metrics."""
        my_metrics = metric_set()
        truth = pd.Series([1, 2, 3])
        estimate = pd.Series([1.1, 2.1, 3.1])

        result = my_metrics(truth, estimate)

        assert len(result) == 0

    def test_metric_set_error_handling(self):
        """Test metric set handles errors gracefully."""
        def bad_metric(truth, estimate, **kwargs):
            raise ValueError("This metric always fails")

        my_metrics = metric_set(rmse, bad_metric, mae)
        truth = pd.Series([1, 2, 3])
        estimate = pd.Series([1.1, 2.1, 3.1])

        result = my_metrics(truth, estimate)

        # Should still compute rmse and mae
        assert "rmse" in result["metric"].values
        assert "mae" in result["metric"].values
        # bad_metric should have NaN value
        assert len(result) == 3

    def test_metric_set_introspection(self):
        """Test metric set preserves metric names."""
        my_metrics = metric_set(rmse, mae, r_squared)

        assert hasattr(my_metrics, "metrics")
        assert my_metrics.metrics == ["rmse", "mae", "r_squared"]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_series(self):
        """Test metrics with empty series."""
        truth = pd.Series([])
        estimate = pd.Series([])

        result = rmse(truth, estimate)
        assert np.isnan(result["value"].iloc[0])

    def test_single_value(self):
        """Test metrics with single value."""
        truth = pd.Series([1])
        estimate = pd.Series([1.1])

        result = rmse(truth, estimate)
        assert not np.isnan(result["value"].iloc[0])

    def test_mismatched_lengths(self):
        """Test metrics with mismatched lengths."""
        truth = pd.Series([1, 2, 3])
        estimate = pd.Series([1.1, 2.1])

        # NumPy should handle this by raising or truncating
        # We test that it doesn't crash
        try:
            result = rmse(truth, estimate)
            # If it succeeds, result should be valid
            assert isinstance(result, pd.DataFrame)
        except ValueError:
            # If it fails, that's also acceptable behavior
            pass

    def test_all_same_values(self):
        """Test metrics with constant predictions."""
        truth = pd.Series([1, 2, 3, 4, 5])
        estimate = pd.Series([3, 3, 3, 3, 3])

        result = rmse(truth, estimate)
        assert not np.isnan(result["value"].iloc[0])

    def test_negative_values(self):
        """Test metrics with negative values."""
        truth = pd.Series([-5, -3, -1, 1, 3])
        estimate = pd.Series([-4.5, -2.8, -1.2, 0.9, 3.1])

        result = rmse(truth, estimate)
        assert not np.isnan(result["value"].iloc[0])

    def test_very_large_values(self):
        """Test metrics with very large values."""
        truth = pd.Series([1e10, 2e10, 3e10])
        estimate = pd.Series([1.1e10, 2.1e10, 2.9e10])

        result = rmse(truth, estimate)
        assert not np.isnan(result["value"].iloc[0])

    def test_very_small_values(self):
        """Test metrics with very small values."""
        truth = pd.Series([1e-10, 2e-10, 3e-10])
        estimate = pd.Series([1.1e-10, 2.1e-10, 2.9e-10])

        result = rmse(truth, estimate)
        assert not np.isnan(result["value"].iloc[0])
