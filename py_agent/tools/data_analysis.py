"""
Data analysis tools for temporal pattern detection.

These tools analyze time series data to identify:
- Frequency (daily, weekly, monthly)
- Seasonality patterns and strength
- Trend direction and strength
- Autocorrelation at various lags
- Data quality issues (missing values, outliers)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy import stats


def analyze_temporal_patterns(
    data: pd.DataFrame,
    date_col: str,
    value_col: str
) -> Dict:
    """
    Analyze temporal patterns in time series data.

    This is the primary analysis function that combines multiple
    detection methods to provide a comprehensive view of the data.

    Args:
        data: DataFrame containing time series data
        date_col: Name of the date/time column
        value_col: Name of the value column to analyze

    Returns:
        Dictionary containing:
        - frequency: 'daily', 'weekly', 'monthly', etc.
        - seasonality: Detection results and strength
        - trend: Direction and strength
        - autocorrelation: Values at key lags
        - missing_rate: Proportion of missing values
        - outlier_rate: Proportion of outliers

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=365, freq='D'),
        ...     'sales': np.random.randn(365) + 100
        ... })
        >>> results = analyze_temporal_patterns(df, 'date', 'sales')
        >>> print(results['frequency'])
        'daily'
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])

    # Sort by date
    data = data.sort_values(date_col)

    # Detect frequency
    frequency = _detect_frequency(data[date_col])

    # Detect seasonality
    seasonality_info = detect_seasonality(
        data[value_col].values,
        frequency=frequency
    )

    # Detect trend
    trend_info = detect_trend(data[value_col].values)

    # Calculate autocorrelation
    autocorr_info = calculate_autocorrelation(
        data[value_col].values,
        lags=[1, 7, 30]
    )

    # Calculate data quality metrics
    missing_rate = data[value_col].isnull().mean()
    outlier_rate = _detect_outliers(data[value_col].values)

    return {
        'frequency': frequency,
        'seasonality': seasonality_info,
        'trend': trend_info,
        'autocorrelation': autocorr_info,
        'missing_rate': float(missing_rate),
        'outlier_rate': float(outlier_rate),
        'n_observations': len(data),
        'date_range': {
            'start': data[date_col].min().isoformat(),
            'end': data[date_col].max().isoformat()
        }
    }


def detect_seasonality(
    series: np.ndarray,
    frequency: str = 'daily',
    period: Optional[int] = None
) -> Dict:
    """
    Detect seasonality in a time series.

    Uses seasonal decomposition to identify periodic patterns
    and measure their strength.

    Args:
        series: Time series values as numpy array
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        period: Explicit seasonal period (overrides frequency-based detection)

    Returns:
        Dictionary containing:
        - detected: Whether seasonality was found
        - period: Seasonal period (e.g., 7 for weekly in daily data)
        - strength: Strength metric (0-1)
        - component: Seasonal component values

    Example:
        >>> sales = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100)*0.1 + 50
        >>> result = detect_seasonality(sales, frequency='daily', period=7)
        >>> result['detected']
        True
    """
    # Determine period if not provided
    if period is None:
        period = _get_default_period(frequency)

    # Need at least 2 full periods
    if len(series) < 2 * period:
        return {
            'detected': False,
            'period': period,
            'strength': 0.0,
            'reason': f'Insufficient data: need at least {2*period} observations, have {len(series)}'
        }

    try:
        # Remove NaNs for decomposition
        clean_series = pd.Series(series).fillna(method='ffill').fillna(method='bfill')

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            clean_series,
            model='additive',
            period=period,
            extrapolate_trend='freq'
        )

        # Calculate seasonality strength
        # Strength = Var(seasonal) / Var(seasonal + residual)
        seasonal_var = np.var(decomposition.seasonal)
        residual_var = np.var(decomposition.resid)

        if seasonal_var + residual_var > 0:
            strength = seasonal_var / (seasonal_var + residual_var)
        else:
            strength = 0.0

        # Consider seasonality detected if strength > 0.3
        detected = strength > 0.3

        return {
            'detected': detected,
            'period': period,
            'strength': float(strength),
            'seasonal_component': decomposition.seasonal.tolist()[:period]  # One cycle
        }

    except Exception as e:
        return {
            'detected': False,
            'period': period,
            'strength': 0.0,
            'error': str(e)
        }


def detect_trend(series: np.ndarray) -> Dict:
    """
    Detect trend in a time series.

    Uses linear regression to identify upward, downward, or stable trends
    and measure their strength.

    Args:
        series: Time series values as numpy array

    Returns:
        Dictionary containing:
        - direction: 'increasing', 'decreasing', or 'stable'
        - strength: Strength metric (0-1)
        - slope: Regression slope
        - p_value: Statistical significance of trend

    Example:
        >>> trend_series = np.arange(100) + np.random.randn(100)*5
        >>> result = detect_trend(trend_series)
        >>> result['direction']
        'increasing'
    """
    # Remove NaNs
    clean_series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values

    # Fit linear regression
    x = np.arange(len(clean_series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_series)

    # Determine direction
    if p_value > 0.05:  # Not statistically significant
        direction = 'stable'
    elif slope > 0:
        direction = 'increasing'
    else:
        direction = 'decreasing'

    # Strength is R-squared value
    strength = float(r_value ** 2)

    return {
        'direction': direction,
        'strength': strength,
        'slope': float(slope),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def calculate_autocorrelation(
    series: np.ndarray,
    lags: list[int] = [1, 7, 30]
) -> Dict:
    """
    Calculate autocorrelation at specified lags.

    Measures how strongly the series correlates with itself
    at different time lags.

    Args:
        series: Time series values as numpy array
        lags: List of lag values to compute

    Returns:
        Dictionary mapping lag to correlation value

    Example:
        >>> ar_series = np.concatenate([[0], np.cumsum(np.random.randn(99))])
        >>> result = calculate_autocorrelation(ar_series, lags=[1, 5, 10])
        >>> result['lag_1'] > 0.5  # High autocorrelation at lag 1
        True
    """
    # Remove NaNs
    clean_series = pd.Series(series).dropna().values

    if len(clean_series) < max(lags) + 1:
        return {f'lag_{lag}': 0.0 for lag in lags}

    try:
        # Calculate ACF
        max_lag = max(lags)
        acf_values = acf(clean_series, nlags=max_lag, fft=True)

        # Extract requested lags
        result = {}
        for lag in lags:
            if lag < len(acf_values):
                result[f'lag_{lag}'] = float(acf_values[lag])
            else:
                result[f'lag_{lag}'] = 0.0

        return result

    except Exception as e:
        return {f'lag_{lag}': 0.0 for lag in lags}


# Helper functions

def _detect_frequency(date_series: pd.Series) -> str:
    """
    Detect the frequency of a datetime series.

    Args:
        date_series: Series of datetime values

    Returns:
        Frequency string: 'daily', 'weekly', 'monthly', etc.
    """
    # Try pandas infer_freq
    freq = pd.infer_freq(date_series)

    if freq is None:
        # Calculate most common time delta
        diffs = date_series.diff().dropna()
        if len(diffs) == 0:
            return 'unknown'

        most_common_diff = diffs.value_counts().index[0]
        days = most_common_diff.total_seconds() / 86400

        if days < 1.5:
            return 'daily'
        elif days < 8:
            return 'weekly'
        elif days < 32:
            return 'monthly'
        elif days < 100:
            return 'quarterly'
        else:
            return 'yearly'

    # Map pandas frequency codes to readable names
    freq_map = {
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'MS': 'monthly',
        'Q': 'quarterly',
        'QS': 'quarterly',
        'Y': 'yearly',
        'YS': 'yearly',
        'H': 'hourly',
        'T': 'minutely',
        'S': 'secondly'
    }

    for code, name in freq_map.items():
        if freq.startswith(code):
            return name

    return 'unknown'


def _get_default_period(frequency: str) -> int:
    """
    Get default seasonal period for a given frequency.

    Args:
        frequency: Data frequency string

    Returns:
        Period as integer (e.g., 7 for daily data with weekly seasonality)
    """
    period_map = {
        'hourly': 24,  # Daily seasonality
        'daily': 7,  # Weekly seasonality
        'weekly': 52,  # Yearly seasonality
        'monthly': 12,  # Yearly seasonality
        'quarterly': 4,  # Yearly seasonality
        'yearly': 1  # No seasonality
    }

    return period_map.get(frequency, 7)


def _detect_outliers(series: np.ndarray, threshold: float = 3.0) -> float:
    """
    Detect outliers using IQR method.

    Args:
        series: Time series values
        threshold: IQR multiplier for outlier detection

    Returns:
        Proportion of values that are outliers
    """
    clean_series = pd.Series(series).dropna()

    if len(clean_series) == 0:
        return 0.0

    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outliers = (clean_series < lower_bound) | (clean_series > upper_bound)

    return float(outliers.mean())
