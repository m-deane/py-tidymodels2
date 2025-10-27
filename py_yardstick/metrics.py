"""
Performance metrics for model evaluation

Provides tidymodels-style metric functions that return standardized DataFrames.
Includes time series metrics, residual diagnostics, and classification/regression metrics.
"""

from typing import Union, Callable, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


# ============================================================================
# Helper Functions
# ============================================================================

def _safe_isnan(arr):
    """
    Safely check for NaN values, handling both numeric and non-numeric arrays.

    Args:
        arr: NumPy array

    Returns:
        Boolean array indicating NaN values
    """
    try:
        # Try numeric isnan first
        return np.isnan(arr)
    except (TypeError, ValueError):
        # For non-numeric types, use pandas isna
        return pd.isna(arr)


# ============================================================================
# Time Series Metrics (Priority)
# ============================================================================

def rmse(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Root mean squared error.

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([1, 2, 3, 4, 5])
        >>> estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> rmse(truth, estimate)
           metric     value
        0    rmse  0.141421
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        mse = np.mean((truth_clean - estimate_clean) ** 2)
        value = np.sqrt(mse)

    return pd.DataFrame({
        "metric": ["rmse"],
        "value": [value]
    })


def mae(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Mean absolute error.

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([1, 2, 3, 4, 5])
        >>> estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> mae(truth, estimate)
          metric  value
        0    mae    0.16
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        value = np.mean(np.abs(truth_clean - estimate_clean))

    return pd.DataFrame({
        "metric": ["mae"],
        "value": [value]
    })


def mape(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Mean absolute percentage error.

    Note: MAPE is undefined when truth contains zeros. These values are excluded.

    Args:
        truth: Actual values (should not contain zeros)
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value (percentage, 0-100)

    Examples:
        >>> truth = pd.Series([10, 20, 30, 40, 50])
        >>> estimate = pd.Series([11, 22, 29, 41, 48])
        >>> mape(truth, estimate)
          metric  value
        0   mape    5.5
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values and zeros in truth
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr)) & (truth_arr != 0)
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        value = np.mean(np.abs((truth_clean - estimate_clean) / truth_clean)) * 100

    return pd.DataFrame({
        "metric": ["mape"],
        "value": [value]
    })


def smape(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Symmetric mean absolute percentage error.

    SMAPE is bounded between 0 and 200, and handles zeros better than MAPE.

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value (percentage, 0-200)

    Examples:
        >>> truth = pd.Series([10, 20, 30, 40, 50])
        >>> estimate = pd.Series([11, 22, 29, 41, 48])
        >>> smape(truth, estimate)
          metric     value
        0  smape  5.384615
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        numerator = np.abs(truth_clean - estimate_clean)
        denominator = (np.abs(truth_clean) + np.abs(estimate_clean)) / 2

        # Handle case where both truth and estimate are zero
        mask_nonzero = denominator != 0
        if not np.any(mask_nonzero):
            value = 0.0
        else:
            value = np.mean(numerator[mask_nonzero] / denominator[mask_nonzero]) * 100

    return pd.DataFrame({
        "metric": ["smape"],
        "value": [value]
    })


def mase(
    truth: pd.Series,
    estimate: pd.Series,
    train: Optional[pd.Series] = None,
    m: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Mean absolute scaled error.

    MASE scales MAE by the naive forecast MAE on the training data.
    Requires training data to compute the scaling factor.

    Args:
        truth: Actual values (test data)
        estimate: Predicted values (test data)
        train: Training data for computing scaling factor (required)
        m: Seasonal period for naive forecast (default: 1 for non-seasonal)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> train = pd.Series([10, 12, 11, 13, 12, 14])
        >>> truth = pd.Series([15, 16, 17])
        >>> estimate = pd.Series([15.5, 15.8, 16.9])
        >>> mase(truth, estimate, train=train)
          metric  value
        0   mase    0.5
    """
    if train is None:
        raise ValueError("MASE requires training data via 'train' parameter")

    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)
    train_arr = np.asarray(train)

    # Remove NaN values from test data
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    # Compute MAE on test data
    if len(truth_clean) == 0:
        value = np.nan
    else:
        test_mae = np.mean(np.abs(truth_clean - estimate_clean))

        # Compute naive forecast MAE on training data
        # Naive forecast: y_t+m = y_t
        if len(train_arr) <= m:
            # Not enough training data
            value = np.nan
        else:
            naive_errors = np.abs(train_arr[m:] - train_arr[:-m])
            naive_mae = np.mean(naive_errors)

            if naive_mae == 0:
                value = np.nan  # Cannot scale by zero
            else:
                value = test_mae / naive_mae

    return pd.DataFrame({
        "metric": ["mase"],
        "value": [value]
    })


def r_squared(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Coefficient of determination (R²).

    Standard R² using total sum of squares.

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([1, 2, 3, 4, 5])
        >>> estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> r_squared(truth, estimate)
              metric     value
        0  r_squared  0.989796
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        ss_res = np.sum((truth_clean - estimate_clean) ** 2)
        ss_tot = np.sum((truth_clean - np.mean(truth_clean)) ** 2)

        if ss_tot == 0:
            value = np.nan
        else:
            value = 1 - (ss_res / ss_tot)

    return pd.DataFrame({
        "metric": ["r_squared"],
        "value": [value]
    })


def rsq_trad(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Traditional R² (squared correlation).

    Computes R² as the square of the correlation between truth and estimate.
    Always non-negative, unlike standard R².

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([1, 2, 3, 4, 5])
        >>> estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> rsq_trad(truth, estimate)
            metric     value
        0  rsq_trad  0.994898
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) < 2:
        value = np.nan
    else:
        corr = np.corrcoef(truth_clean, estimate_clean)[0, 1]
        value = corr ** 2

    return pd.DataFrame({
        "metric": ["rsq_trad"],
        "value": [value]
    })


# ============================================================================
# Residual Diagnostic Tests (Time Series)
# ============================================================================

def durbin_watson(residuals: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Durbin-Watson test statistic for autocorrelation.

    Tests for first-order autocorrelation in residuals.
    Values range from 0 to 4, with 2 indicating no autocorrelation.

    Args:
        residuals: Model residuals
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> residuals = pd.Series([0.1, -0.2, 0.15, -0.1, 0.05])
        >>> durbin_watson(residuals)
                 metric  value
        0  durbin_watson    2.5
    """
    from statsmodels.stats.stattools import durbin_watson as dw_test

    residuals_arr = np.asarray(residuals)

    # Remove NaN values
    residuals_clean = residuals_arr[~_safe_isnan(residuals_arr)]

    if len(residuals_clean) < 2:
        value = np.nan
    else:
        value = dw_test(residuals_clean)

    return pd.DataFrame({
        "metric": ["durbin_watson"],
        "value": [value]
    })


def ljung_box(residuals: pd.Series, lags: int = 10, **kwargs) -> pd.DataFrame:
    """
    Ljung-Box test for autocorrelation.

    Tests for autocorrelation in residuals up to specified lag.
    Returns both test statistic and p-value.

    Args:
        residuals: Model residuals
        lags: Number of lags to test (default: 10)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value
        Two rows: ljung_box_stat and ljung_box_p

    Examples:
        >>> residuals = pd.Series([0.1, -0.2, 0.15, -0.1, 0.05])
        >>> ljung_box(residuals, lags=2)
                 metric     value
        0  ljung_box_stat  0.454545
        1    ljung_box_p  0.797089
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals_arr = np.asarray(residuals)

    # Remove NaN values
    residuals_clean = residuals_arr[~_safe_isnan(residuals_arr)]

    if len(residuals_clean) <= lags:
        stat_value = np.nan
        p_value = np.nan
    else:
        try:
            result = acorr_ljungbox(residuals_clean, lags=lags, return_df=False)
            # Result is (lb_stat, lb_pvalue) for the last lag
            stat_value = result[0][-1] if hasattr(result[0], '__len__') else result[0]
            p_value = result[1][-1] if hasattr(result[1], '__len__') else result[1]
        except Exception:
            stat_value = np.nan
            p_value = np.nan

    return pd.DataFrame({
        "metric": ["ljung_box_stat", "ljung_box_p"],
        "value": [stat_value, p_value]
    })


def shapiro_wilk(residuals: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Shapiro-Wilk test for normality of residuals.

    Tests whether residuals are normally distributed.
    Returns both test statistic and p-value.

    Args:
        residuals: Model residuals
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value
        Two rows: shapiro_wilk_stat and shapiro_wilk_p

    Examples:
        >>> residuals = pd.Series(np.random.normal(0, 1, 100))
        >>> shapiro_wilk(residuals)
                    metric     value
        0  shapiro_wilk_stat  0.989012
        1    shapiro_wilk_p  0.543210
    """
    residuals_arr = np.asarray(residuals)

    # Remove NaN values
    residuals_clean = residuals_arr[~_safe_isnan(residuals_arr)]

    if len(residuals_clean) < 3:
        stat_value = np.nan
        p_value = np.nan
    else:
        try:
            stat_value, p_value = stats.shapiro(residuals_clean)
        except Exception:
            stat_value = np.nan
            p_value = np.nan

    return pd.DataFrame({
        "metric": ["shapiro_wilk_stat", "shapiro_wilk_p"],
        "value": [stat_value, p_value]
    })


def adf_test(series: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests the null hypothesis that a unit root is present (non-stationary).
    Returns test statistic and p-value.

    Args:
        series: Time series to test
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value
        Two rows: adf_stat and adf_p

    Examples:
        >>> series = pd.Series(np.cumsum(np.random.randn(100)))
        >>> adf_test(series)
               metric     value
        0    adf_stat -1.234567
        1      adf_p  0.654321
    """
    from statsmodels.tsa.stattools import adfuller

    series_arr = np.asarray(series)

    # Remove NaN values
    series_clean = series_arr[~_safe_isnan(series_arr)]

    if len(series_clean) < 12:  # ADF requires minimum data
        stat_value = np.nan
        p_value = np.nan
    else:
        try:
            result = adfuller(series_clean)
            stat_value = result[0]
            p_value = result[1]
        except Exception:
            stat_value = np.nan
            p_value = np.nan

    return pd.DataFrame({
        "metric": ["adf_stat", "adf_p"],
        "value": [stat_value, p_value]
    })


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Classification accuracy.

    Proportion of correctly classified observations.

    Args:
        truth: True class labels
        estimate: Predicted class labels
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([0, 1, 0, 1, 1])
        >>> estimate = pd.Series([0, 1, 0, 0, 1])
        >>> accuracy(truth, estimate)
            metric  value
        0  accuracy    0.8
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        value = np.mean(truth_clean == estimate_clean)

    return pd.DataFrame({
        "metric": ["accuracy"],
        "value": [value]
    })


def precision(truth: pd.Series, estimate: pd.Series, positive_class=1, **kwargs) -> pd.DataFrame:
    """
    Precision (positive predictive value).

    Proportion of positive predictions that are correct.

    Args:
        truth: True class labels
        estimate: Predicted class labels
        positive_class: Label for positive class (default: 1)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([0, 1, 0, 1, 1])
        >>> estimate = pd.Series([0, 1, 1, 0, 1])
        >>> precision(truth, estimate)
             metric     value
        0  precision  0.666667
    """
    from sklearn.metrics import precision_score

    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        try:
            value = precision_score(truth_clean, estimate_clean, pos_label=positive_class, zero_division=0)
        except Exception:
            value = np.nan

    return pd.DataFrame({
        "metric": ["precision"],
        "value": [value]
    })


def recall(truth: pd.Series, estimate: pd.Series, positive_class=1, **kwargs) -> pd.DataFrame:
    """
    Recall (sensitivity, true positive rate).

    Proportion of actual positives that are correctly predicted.

    Args:
        truth: True class labels
        estimate: Predicted class labels
        positive_class: Label for positive class (default: 1)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([0, 1, 0, 1, 1])
        >>> estimate = pd.Series([0, 1, 1, 0, 1])
        >>> recall(truth, estimate)
          metric     value
        0  recall  0.666667
    """
    from sklearn.metrics import recall_score

    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        try:
            value = recall_score(truth_clean, estimate_clean, pos_label=positive_class, zero_division=0)
        except Exception:
            value = np.nan

    return pd.DataFrame({
        "metric": ["recall"],
        "value": [value]
    })


def f_meas(truth: pd.Series, estimate: pd.Series, positive_class=1, beta: float = 1.0, **kwargs) -> pd.DataFrame:
    """
    F-measure (F-score, F1 score).

    Harmonic mean of precision and recall.

    Args:
        truth: True class labels
        estimate: Predicted class labels
        positive_class: Label for positive class (default: 1)
        beta: Weight of recall vs precision (default: 1.0 for F1)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([0, 1, 0, 1, 1])
        >>> estimate = pd.Series([0, 1, 1, 0, 1])
        >>> f_meas(truth, estimate)
          metric     value
        0  f_meas  0.666667
    """
    from sklearn.metrics import fbeta_score

    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0:
        value = np.nan
    else:
        try:
            value = fbeta_score(truth_clean, estimate_clean, beta=beta, pos_label=positive_class, zero_division=0)
        except Exception:
            value = np.nan

    return pd.DataFrame({
        "metric": ["f_meas"],
        "value": [value]
    })


def roc_auc(truth: pd.Series, estimate_prob: pd.Series, positive_class=1, **kwargs) -> pd.DataFrame:
    """
    Area under the ROC curve.

    Requires predicted probabilities (not class labels).

    Args:
        truth: True class labels
        estimate_prob: Predicted probabilities for positive class
        positive_class: Label for positive class (default: 1)
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([0, 1, 0, 1, 1])
        >>> estimate_prob = pd.Series([0.1, 0.8, 0.3, 0.7, 0.9])
        >>> roc_auc(truth, estimate_prob)
          metric  value
        0  roc_auc    1.0
    """
    from sklearn.metrics import roc_auc_score

    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate_prob)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) == 0 or len(np.unique(truth_clean)) < 2:
        value = np.nan
    else:
        try:
            # Convert truth to binary if needed
            truth_binary = (truth_clean == positive_class).astype(int)
            value = roc_auc_score(truth_binary, estimate_clean)
        except Exception:
            value = np.nan

    return pd.DataFrame({
        "metric": ["roc_auc"],
        "value": [value]
    })


# ============================================================================
# Additional Regression Metrics
# ============================================================================

def mda(truth: pd.Series, estimate: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Mean directional accuracy.

    Proportion of times the predicted change direction matches actual change direction.
    Useful for time series forecasting.

    Args:
        truth: Actual values
        estimate: Predicted values
        **kwargs: Additional arguments (for consistency with metric_set)

    Returns:
        DataFrame with columns: metric, value

    Examples:
        >>> truth = pd.Series([1, 2, 3, 2, 3])
        >>> estimate = pd.Series([1.1, 2.1, 2.9, 2.2, 2.8])
        >>> mda(truth, estimate)
          metric  value
        0    mda    0.75
    """
    truth_arr = np.asarray(truth)
    estimate_arr = np.asarray(estimate)

    # Remove NaN values
    mask = ~(_safe_isnan(truth_arr) | _safe_isnan(estimate_arr))
    truth_clean = truth_arr[mask]
    estimate_clean = estimate_arr[mask]

    if len(truth_clean) < 2:
        value = np.nan
    else:
        # Compute direction of change
        truth_direction = np.sign(np.diff(truth_clean))
        estimate_direction = np.sign(np.diff(estimate_clean))

        # Proportion of correct directions
        value = np.mean(truth_direction == estimate_direction)

    return pd.DataFrame({
        "metric": ["mda"],
        "value": [value]
    })


# ============================================================================
# Metric Set Composer
# ============================================================================

def metric_set(*metrics: Callable) -> Callable:
    """
    Create a metric set that computes multiple metrics at once.

    Args:
        *metrics: Metric functions to include

    Returns:
        Function that computes all metrics and returns combined DataFrame

    Examples:
        >>> truth = pd.Series([1, 2, 3, 4, 5])
        >>> estimate = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> my_metrics = metric_set(rmse, mae, r_squared)
        >>> my_metrics(truth, estimate)
              metric     value
        0       rmse  0.141421
        1        mae  0.160000
        2  r_squared  0.989796
    """
    def compute(truth, estimate, **kwargs):
        """Compute all metrics in the set."""
        results = []
        for metric_fn in metrics:
            try:
                result = metric_fn(truth, estimate, **kwargs)
                results.append(result)
            except Exception as e:
                # If a metric fails, add NaN result
                metric_name = metric_fn.__name__
                results.append(pd.DataFrame({
                    "metric": [metric_name],
                    "value": [np.nan]
                }))

        if len(results) == 0:
            return pd.DataFrame(columns=["metric", "value"])

        return pd.concat(results, ignore_index=True)

    # Preserve metric names for introspection
    compute.metrics = [m.__name__ for m in metrics]

    return compute
