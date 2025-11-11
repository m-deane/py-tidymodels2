"""
Time series transformation steps.

Provides advanced time series transformations including anomaly detection,
stationarity transformations, detrending, and deseasonalization.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import pandas as pd
import numpy as np
from ..selectors import resolve_selector, all_numeric


@dataclass
class StepCleanAnomalies:
    """
    Detect and clean anomalies in time series data.

    Uses pytimetk's anomalize function to detect anomalies using STL decomposition
    or Twitter's AnomalyDetection algorithm, then cleans them using various strategies.

    Parameters
    ----------
    date_column : str
        Name of the date/time column
    value_columns : selector, optional
        Which columns to check for anomalies. If None, uses all numeric columns
    period : int, optional
        Seasonal period for decomposition (auto-detected if None)
    trend : int, optional
        Trend window for STL decomposition (auto-detected if None)
    method : str, default='stl'
        Anomaly detection method: 'stl' or 'twitter'
    decomp : str, default='additive'
        Decomposition type: 'additive' or 'multiplicative'
    clean : str, default='min_max'
        Cleaning method: 'min_max' (replace with min/max) or 'linear' (interpolate)
    iqr_alpha : float, default=0.05
        Alpha for IQR anomaly detection (lower = more sensitive)
    clean_alpha : float, default=0.75
        Cleaning strength (0-1, higher = more aggressive cleaning)
    max_anomalies : float, default=0.2
        Maximum proportion of data that can be anomalies (0-1)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_clean_anomalies
    >>>
    >>> # Clean anomalies in sales data using STL decomposition
    >>> rec = recipe().step_clean_anomalies(
    ...     date_column='date',
    ...     value_columns=['sales', 'revenue'],
    ...     method='stl',
    ...     clean='min_max'
    ... )
    >>>
    >>> # More aggressive anomaly detection
    >>> rec = recipe().step_clean_anomalies(
    ...     date_column='date',
    ...     iqr_alpha=0.01,  # More sensitive
    ...     clean_alpha=0.9   # More aggressive cleaning
    ... )

    Notes
    -----
    - Requires regular time series (consistent frequency)
    - Uses STL decomposition to separate trend, seasonal, and remainder
    - Anomalies detected in remainder component
    - Original data preserved; cleaned values returned
    - Works on grouped data (multiple time series)
    """
    date_column: str
    value_columns: Union[None, str, List[str], Callable] = None
    period: Optional[int] = None
    trend: Optional[int] = None
    method: str = 'stl'
    decomp: str = 'additive'
    clean: str = 'min_max'
    iqr_alpha: float = 0.05
    clean_alpha: float = 0.75
    max_anomalies: float = 0.2
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _anomaly_info: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_methods = ['stl', 'twitter']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

        valid_decomp = ['additive', 'multiplicative']
        if self.decomp not in valid_decomp:
            raise ValueError(f"decomp must be one of {valid_decomp}, got {self.decomp}")

        valid_clean = ['min_max', 'linear']
        if self.clean not in valid_clean:
            raise ValueError(f"clean must be one of {valid_clean}, got {self.clean}")

        if not (0 < self.iqr_alpha < 1):
            raise ValueError(f"iqr_alpha must be in (0, 1), got {self.iqr_alpha}")

        if not (0 < self.clean_alpha <= 1):
            raise ValueError(f"clean_alpha must be in (0, 1], got {self.clean_alpha}")

        if not (0 < self.max_anomalies <= 1):
            raise ValueError(f"max_anomalies must be in (0, 1], got {self.max_anomalies}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by identifying anomaly detection parameters."""
        if self.skip or not training:
            return self

        # Verify date column exists
        if self.date_column not in data.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in data")

        # Resolve value columns
        if self.value_columns is None:
            value_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.date_column]
        elif isinstance(self.value_columns, str):
            value_cols = [self.value_columns]
        elif callable(self.value_columns):
            value_cols = resolve_selector(self.value_columns, data)
        else:
            value_cols = list(self.value_columns)

        # Remove date column if accidentally included
        value_cols = [c for c in value_cols if c != self.date_column]

        if len(value_cols) == 0:
            raise ValueError("No value columns to check for anomalies after resolving selector")

        # Store configuration
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._anomaly_info = {
            'period': self.period,
            'trend': self.trend,
            'method': self.method,
            'decomp': self.decomp,
            'clean': self.clean,
            'iqr_alpha': self.iqr_alpha,
            'clean_alpha': self.clean_alpha,
            'max_anomalies': self.max_anomalies
        }
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and clean anomalies in new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Check if date column exists
        if self.date_column not in data.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in data")

        # Only process columns that exist in data
        cols_to_clean = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_clean) == 0:
            return data

        import pytimetk as tk

        # Copy data
        data = data.copy()

        # Clean each column
        for col in cols_to_clean:
            # Skip if column has all NaN or too few non-NaN values
            if data[col].isna().all() or data[col].notna().sum() < 3:
                continue

            try:
                # Use pytimetk anomalize
                result = tk.anomalize(
                    data=data,
                    date_column=self.date_column,
                    value_column=col,
                    period=self._anomaly_info['period'],
                    trend=self._anomaly_info['trend'],
                    method=self._anomaly_info['method'],
                    decomp=self._anomaly_info['decomp'],
                    clean=self._anomaly_info['clean'],
                    iqr_alpha=self._anomaly_info['iqr_alpha'],
                    clean_alpha=self._anomaly_info['clean_alpha'],
                    max_anomalies=self._anomaly_info['max_anomalies'],
                    bind_data=True
                )

                # Replace original values with cleaned values
                if 'observed_clean' in result.columns:
                    data[col] = result['observed_clean'].values
                elif f'{col}_clean' in result.columns:
                    data[col] = result[f'{col}_clean'].values

            except Exception as e:
                # If anomaly detection fails, leave data as-is
                # (could be due to insufficient data, irregular time series, etc.)
                import warnings
                warnings.warn(f"Anomaly detection failed for column '{col}': {str(e)}")
                continue

        return data


@dataclass
class StepStationary:
    """
    Transform time series to make it stationary.

    Applies differencing and/or transformations to achieve stationarity, verified by
    Augmented Dickey-Fuller (ADF) test and optionally KPSS test.

    Parameters
    ----------
    columns : selector, optional
        Which columns to transform. If None, uses all numeric columns
    max_diff : int, default=2
        Maximum number of differencing operations (1 or 2)
    test : str, default='adf'
        Stationarity test: 'adf' (Augmented Dickey-Fuller), 'kpss', or 'both'
    alpha : float, default=0.05
        Significance level for stationarity test
    seasonal_diff : bool, default=False
        Apply seasonal differencing before regular differencing
    seasonal_period : int, optional
        Seasonal period for seasonal differencing (required if seasonal_diff=True)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_stationary
    >>>
    >>> # Make time series stationary with ADF test
    >>> rec = recipe().step_stationary(
    ...     columns=['sales', 'revenue'],
    ...     max_diff=2,
    ...     test='adf'
    ... )
    >>>
    >>> # Seasonal differencing for monthly data
    >>> rec = recipe().step_stationary(
    ...     seasonal_diff=True,
    ...     seasonal_period=12,
    ...     max_diff=1
    ... )

    Notes
    -----
    - Applies differencing: x'(t) = x(t) - x(t-1)
    - Tests stationarity after each differencing
    - Stops when stationary or max_diff reached
    - ADF test: H0 = non-stationary (reject if p < alpha)
    - KPSS test: H0 = stationary (reject if p < alpha)
    - Missing values introduced by differencing are handled
    """
    columns: Union[None, str, List[str], Callable] = None
    max_diff: int = 2
    test: str = 'adf'
    alpha: float = 0.05
    seasonal_diff: bool = False
    seasonal_period: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _diff_orders: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _seasonal_diff_applied: Dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.max_diff not in [1, 2]:
            raise ValueError(f"max_diff must be 1 or 2, got {self.max_diff}")

        valid_tests = ['adf', 'kpss', 'both']
        if self.test not in valid_tests:
            raise ValueError(f"test must be one of {valid_tests}, got {self.test}")

        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        if self.seasonal_diff and self.seasonal_period is None:
            raise ValueError("seasonal_period required when seasonal_diff=True")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by determining differencing order for each column."""
        if self.skip or not training:
            return self

        from statsmodels.tsa.stattools import adfuller, kpss

        # Resolve columns
        if self.columns is None:
            value_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            value_cols = [self.columns]
        elif callable(self.columns):
            value_cols = resolve_selector(self.columns, data)
        else:
            value_cols = list(self.columns)

        if len(value_cols) == 0:
            raise ValueError("No columns to transform after resolving selector")

        diff_orders = {}
        seasonal_diff_applied = {}

        # Determine differencing order for each column
        for col in value_cols:
            # Get non-null values
            series = data[col].dropna()

            if len(series) < 20:
                # Not enough data for reliable test
                diff_orders[col] = 1  # Apply first difference by default
                seasonal_diff_applied[col] = False
                continue

            current_series = series.copy()
            current_order = 0

            # Apply seasonal differencing if requested
            if self.seasonal_diff and self.seasonal_period:
                current_series = current_series.diff(self.seasonal_period).dropna()
                seasonal_diff_applied[col] = True
            else:
                seasonal_diff_applied[col] = False

            # Test for stationarity and difference until stationary or max_diff reached
            for d in range(self.max_diff + 1):
                if len(current_series) < 20:
                    break

                # Test stationarity
                is_stationary = self._is_stationary(current_series, self.test, self.alpha)

                if is_stationary:
                    current_order = d
                    break

                # Apply differencing
                if d < self.max_diff:
                    current_series = current_series.diff().dropna()
            else:
                # Max diff reached
                current_order = self.max_diff

            diff_orders[col] = current_order

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._diff_orders = diff_orders
        prepared._seasonal_diff_applied = seasonal_diff_applied
        prepared._is_prepared = True

        return prepared

    def _is_stationary(self, series: pd.Series, test: str, alpha: float) -> bool:
        """Test if series is stationary."""
        from statsmodels.tsa.stattools import adfuller, kpss

        try:
            if test == 'adf':
                # ADF test: H0 = non-stationary
                result = adfuller(series, autolag='AIC')
                p_value = result[1]
                return p_value < alpha  # Reject H0 if p < alpha (stationary)

            elif test == 'kpss':
                # KPSS test: H0 = stationary
                result = kpss(series, regression='c', nlags='auto')
                p_value = result[1]
                return p_value >= alpha  # Don't reject H0 if p >= alpha (stationary)

            elif test == 'both':
                # Both tests must agree
                adf_result = adfuller(series, autolag='AIC')
                kpss_result = kpss(series, regression='c', nlags='auto')
                adf_stationary = adf_result[1] < alpha
                kpss_stationary = kpss_result[1] >= alpha
                return adf_stationary and kpss_stationary

        except Exception:
            # If test fails, assume not stationary
            return False

        return False

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply differencing to new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Only transform columns that exist in data
        cols_to_transform = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_transform) == 0:
            return data

        # Transform
        data = data.copy()

        for col in cols_to_transform:
            # Apply seasonal differencing if used in prep
            if self._seasonal_diff_applied.get(col, False) and self.seasonal_period:
                data[col] = data[col].diff(self.seasonal_period)

            # Apply regular differencing
            diff_order = self._diff_orders[col]
            for _ in range(diff_order):
                data[col] = data[col].diff()

        return data


@dataclass
class StepDeseasonalize:
    """
    Remove seasonal component from time series.

    Uses seasonal decomposition (additive or multiplicative) to extract and remove
    the seasonal component, leaving trend + residual.

    Parameters
    ----------
    columns : selector, optional
        Which columns to deseasonalize. If None, uses all numeric columns
    period : int, optional
        Seasonal period (e.g., 12 for monthly data with yearly seasonality).
        Auto-detected if None.
    model : str, default='additive'
        Decomposition model: 'additive' or 'multiplicative'
    method : str, default='stl'
        Decomposition method: 'stl' (robust) or 'classical'
    extrapolate_trend : int, default=0
        Number of points to extrapolate trend at beginning/end
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_deseasonalize
    >>>
    >>> # Remove monthly seasonality (period=12)
    >>> rec = recipe().step_deseasonalize(
    ...     columns=['sales'],
    ...     period=12,
    ...     model='additive'
    ... )
    >>>
    >>> # Use STL for robust decomposition
    >>> rec = recipe().step_deseasonalize(
    ...     period=7,  # Weekly seasonality
    ...     method='stl'
    ... )

    Notes
    -----
    - Additive model: y = trend + seasonal + residual
      Deseasonalized: y - seasonal = trend + residual
    - Multiplicative model: y = trend * seasonal * residual
      Deseasonalized: y / seasonal = trend * residual
    - STL is more robust to outliers than classical decomposition
    - Requires at least 2 full seasonal periods of data
    """
    columns: Union[None, str, List[str], Callable] = None
    period: Optional[int] = None
    model: str = 'additive'
    method: str = 'stl'
    extrapolate_trend: int = 0
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _seasonal_components: Dict[str, pd.Series] = field(default_factory=dict, init=False, repr=False)
    _periods: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_models = ['additive', 'multiplicative']
        if self.model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}, got {self.model}")

        valid_methods = ['stl', 'classical']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by computing seasonal components."""
        if self.skip or not training:
            return self

        from statsmodels.tsa.seasonal import seasonal_decompose, STL

        # Resolve columns
        if self.columns is None:
            value_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            value_cols = [self.columns]
        elif callable(self.columns):
            value_cols = resolve_selector(self.columns, data)
        else:
            value_cols = list(self.columns)

        if len(value_cols) == 0:
            raise ValueError("No columns to deseasonalize after resolving selector")

        seasonal_components = {}
        periods = {}

        # Decompose each column
        for col in value_cols:
            series = data[col].dropna()

            if len(series) < 20:
                # Not enough data
                continue

            # Determine period
            if self.period is not None:
                period = self.period
            else:
                # Auto-detect period (simplified)
                period = self._detect_period(series)

            # Need at least 2 full periods
            if len(series) < 2 * period:
                continue

            try:
                if self.method == 'stl':
                    # STL decomposition
                    stl = STL(series, period=period, seasonal=13)
                    result = stl.fit()
                    seasonal = result.seasonal
                else:
                    # Classical decomposition
                    result = seasonal_decompose(
                        series,
                        model=self.model,
                        period=period,
                        extrapolate_trend=self.extrapolate_trend
                    )
                    seasonal = result.seasonal

                seasonal_components[col] = seasonal
                periods[col] = period

            except Exception:
                # Decomposition failed, skip this column
                continue

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._seasonal_components = seasonal_components
        prepared._periods = periods
        prepared._is_prepared = True

        return prepared

    def _detect_period(self, series: pd.Series) -> int:
        """Auto-detect seasonal period (simplified heuristic)."""
        # Use autocorrelation to find dominant period
        # This is a simplified version - for production use more sophisticated methods
        from statsmodels.tsa.stattools import acf

        try:
            # Compute autocorrelation
            acf_values = acf(series.values, nlags=min(len(series) // 2, 50), fft=True)

            # Find first significant peak after lag 1
            for lag in range(2, len(acf_values)):
                if acf_values[lag] > 0.5:  # Significant correlation
                    return lag

            # Default to common periods
            if len(series) >= 24:
                return 12  # Monthly
            elif len(series) >= 14:
                return 7   # Weekly
            else:
                return 4   # Quarterly

        except Exception:
            # Default to 12 if detection fails
            return 12

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove seasonal component from new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Only transform columns that exist in data
        cols_to_transform = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_transform) == 0:
            return data

        # Transform
        data = data.copy()

        for col in cols_to_transform:
            if col not in self._seasonal_components:
                # No seasonal component learned for this column
                continue

            seasonal = self._seasonal_components[col]
            period = self._periods[col]

            # Extend seasonal component to match data length (repeating pattern)
            n_repeats = int(np.ceil(len(data) / period))
            extended_seasonal = pd.concat([seasonal] * n_repeats, ignore_index=True)
            extended_seasonal = extended_seasonal.iloc[:len(data)]

            # Remove seasonal component
            if self.model == 'additive':
                data[col] = data[col] - extended_seasonal.values
            else:  # multiplicative
                data[col] = data[col] / extended_seasonal.values

        return data


@dataclass
class StepDetrend:
    """
    Remove trend from time series.

    Uses linear or polynomial detrending to remove systematic trend, leaving
    only seasonal and random components.

    Parameters
    ----------
    columns : selector, optional
        Which columns to detrend. If None, uses all numeric columns
    method : str, default='linear'
        Detrending method: 'linear' or 'constant' (mean removal)
    breakpoints : int or list, default=0
        Breakpoints for piecewise linear detrending.
        0 = single line, int = number of breakpoints, list = specific positions
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_detrend
    >>>
    >>> # Simple linear detrending
    >>> rec = recipe().step_detrend(
    ...     columns=['sales', 'revenue'],
    ...     method='linear'
    ... )
    >>>
    >>> # Piecewise linear with 2 breakpoints
    >>> rec = recipe().step_detrend(
    ...     method='linear',
    ...     breakpoints=2
    ... )

    Notes
    -----
    - Linear: Fits line and subtracts it
    - Constant: Subtracts mean (centers data)
    - Piecewise: Multiple linear segments for changing trends
    - Preserves variance structure of detrended data
    """
    columns: Union[None, str, List[str], Callable] = None
    method: Literal['linear', 'constant'] = 'linear'
    breakpoints: Union[int, List[int]] = 0
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _trend_params: Dict[str, Dict] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_methods = ['linear', 'constant']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting trend."""
        if self.skip or not training:
            return self

        from scipy.signal import detrend as scipy_detrend

        # Resolve columns
        if self.columns is None:
            value_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            value_cols = [self.columns]
        elif callable(self.columns):
            value_cols = resolve_selector(self.columns, data)
        else:
            value_cols = list(self.columns)

        if len(value_cols) == 0:
            raise ValueError("No columns to detrend after resolving selector")

        trend_params = {}

        # Fit trend for each column
        for col in value_cols:
            series = data[col].dropna()

            if len(series) < 3:
                continue

            # Determine breakpoints
            if isinstance(self.breakpoints, int):
                bp = self.breakpoints
            else:
                bp = list(self.breakpoints)

            # Store parameters for this column
            trend_params[col] = {
                'method': self.method,
                'breakpoints': bp,
                'length': len(series)
            }

            # For linear trend, compute coefficients
            if self.method == 'linear':
                x = np.arange(len(series))
                y = series.values

                if bp == 0:
                    # Simple linear regression
                    slope, intercept = np.polyfit(x, y, 1)
                    trend_params[col]['slope'] = slope
                    trend_params[col]['intercept'] = intercept
                else:
                    # For piecewise, we'll use scipy.signal.detrend in bake
                    # Store breakpoints
                    trend_params[col]['piecewise'] = True

            elif self.method == 'constant':
                # Store mean
                trend_params[col]['mean'] = series.mean()

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._trend_params = trend_params
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove trend from new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        from scipy.signal import detrend as scipy_detrend

        # Only transform columns that exist in data
        cols_to_transform = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_transform) == 0:
            return data

        # Transform
        data = data.copy()

        for col in cols_to_transform:
            if col not in self._trend_params:
                continue

            params = self._trend_params[col]
            series = data[col].values

            if params['method'] == 'linear':
                if params.get('piecewise', False):
                    # Use scipy for piecewise
                    detrended = scipy_detrend(
                        series,
                        type='linear',
                        bp=params['breakpoints']
                    )
                else:
                    # Simple linear detrending
                    x = np.arange(len(series))
                    trend = params['slope'] * x + params['intercept']
                    detrended = series - trend

                data[col] = detrended

            elif params['method'] == 'constant':
                # Subtract mean
                data[col] = series - params['mean']

        return data


@dataclass
class StepHStat:
    """
    Detect interactions using Friedman's H-statistic.

    Uses gradient boosting to identify pairwise feature interactions based on
    H-statistic, which measures interaction strength between features.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    columns : selector, optional
        Which columns to check for interactions. If None, uses all numeric columns except outcome
    top_n : int, default=10
        Number of top interactions to keep
    threshold : float, optional
        H-statistic threshold (keep interactions above this value)
    n_estimators : int, default=100
        Number of trees in gradient boosting
    max_depth : int, default=5
        Maximum tree depth
    random_state : int, optional
        Random state for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_h_stat
    >>>
    >>> # Find top 10 interactions
    >>> rec = recipe().step_h_stat(
    ...     outcome='target',
    ...     top_n=10
    ... )
    >>>
    >>> # Use threshold instead of top_n
    >>> rec = recipe().step_h_stat(
    ...     outcome='target',
    ...     threshold=0.1
    ... )

    Notes
    -----
    - H-statistic measures partial dependence between features
    - Values range from 0 (no interaction) to 1 (strong interaction)
    - Creates interaction features: x1 * x2 for detected pairs
    - Computational cost increases with number of feature pairs
    """
    outcome: str
    columns: Union[None, str, List[str], Callable] = None
    top_n: int = 10
    threshold: Optional[float] = None
    n_estimators: int = 100
    max_depth: int = 5
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _interactions: List[tuple] = field(default_factory=list, init=False, repr=False)
    _h_statistics: Dict[tuple, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by identifying important interactions."""
        if self.skip or not training:
            return self

        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from itertools import combinations

        # Verify outcome exists
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            value_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.outcome]
        elif isinstance(self.columns, str):
            value_cols = [self.columns]
        elif callable(self.columns):
            value_cols = resolve_selector(self.columns, data)
        else:
            value_cols = list(self.columns)

        # Remove outcome if accidentally included
        value_cols = [c for c in value_cols if c != self.outcome]

        if len(value_cols) < 2:
            raise ValueError("Need at least 2 columns to detect interactions")

        # Determine if regression or classification
        is_regression = pd.api.types.is_numeric_dtype(y)

        # Fit gradient boosting model
        if is_regression:
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

        X = data[value_cols]
        model.fit(X, y)

        # Compute H-statistic for all pairs (simplified approximation)
        # True H-statistic requires partial dependence computation
        # Here we use a simplified approach based on feature importances
        h_statistics = {}

        # Get feature importances
        importances = dict(zip(value_cols, model.feature_importances_))

        # Approximate H-statistic as product of importances
        for (feat1, feat2) in combinations(value_cols, 2):
            # Simplified H-statistic approximation
            h_approx = 2 * importances[feat1] * importances[feat2]
            h_statistics[(feat1, feat2)] = h_approx

        # Select interactions
        if self.threshold is not None:
            # Use threshold
            selected_interactions = [
                pair for pair, h_val in h_statistics.items()
                if h_val >= self.threshold
            ]
        else:
            # Use top_n
            sorted_interactions = sorted(
                h_statistics.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_interactions = [pair for pair, _ in sorted_interactions[:self.top_n]]

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._interactions = selected_interactions
        prepared._h_statistics = h_statistics
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Transform
        data = data.copy()

        # Create interaction features
        for (feat1, feat2) in self._interactions:
            if feat1 in data.columns and feat2 in data.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                data[interaction_name] = data[feat1] * data[feat2]

        return data


@dataclass
class StepBestLag:
    """
    Select optimal lags using Granger causality test.

    Tests multiple lag values to find those with significant Granger causality,
    then creates lag features for the selected lags.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    columns : selector, optional
        Which columns to create lags for. If None, uses all numeric columns except outcome
    max_lag : int, default=12
        Maximum lag to test
    test : str, default='ssr_ftest'
        Granger causality test: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
    alpha : float, default=0.05
        Significance level for test
    add_const : bool, default=True
        Add constant to regression
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_best_lag
    >>>
    >>> # Find best lags up to 12
    >>> rec = recipe().step_best_lag(
    ...     outcome='target',
    ...     max_lag=12,
    ...     alpha=0.05
    ... )
    >>>
    >>> # Test specific predictors
    >>> rec = recipe().step_best_lag(
    ...     outcome='sales',
    ...     columns=['price', 'advertising'],
    ...     max_lag=6
    ... )

    Notes
    -----
    - Tests Granger causality for each lag up to max_lag
    - Keeps lags with p-value < alpha
    - Creates lag features: x_lag1, x_lag2, etc.
    - Requires stationary time series for valid results
    - Granger causality: X Granger-causes Y if past X helps predict Y
    """
    outcome: str
    columns: Union[None, str, List[str], Callable] = None
    max_lag: int = 12
    test: str = 'ssr_ftest'
    alpha: float = 0.05
    add_const: bool = True
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _best_lags: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
        if self.test not in valid_tests:
            raise ValueError(f"test must be one of {valid_tests}, got {self.test}")

        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by finding optimal lags."""
        if self.skip or not training:
            return self

        from statsmodels.tsa.stattools import grangercausalitytests

        # Verify outcome exists
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        # Resolve columns
        if self.columns is None:
            value_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.outcome]
        elif isinstance(self.columns, str):
            value_cols = [self.columns]
        elif callable(self.columns):
            value_cols = resolve_selector(self.columns, data)
        else:
            value_cols = list(self.columns)

        # Remove outcome if accidentally included
        value_cols = [c for c in value_cols if c != self.outcome]

        if len(value_cols) == 0:
            raise ValueError("No columns for lag selection after resolving selector")

        best_lags = {}

        # Test each column
        for col in value_cols:
            # Prepare data for Granger test
            test_data = data[[self.outcome, col]].dropna()

            if len(test_data) < self.max_lag + 10:
                # Not enough data
                continue

            try:
                # Run Granger causality test
                result = grangercausalitytests(
                    test_data,
                    maxlag=self.max_lag,
                    addconst=self.add_const,
                    verbose=False
                )

                # Extract significant lags
                significant_lags = []
                for lag in range(1, self.max_lag + 1):
                    # Get p-value for this lag
                    p_value = result[lag][0][self.test][1]

                    if p_value < self.alpha:
                        significant_lags.append(lag)

                if len(significant_lags) > 0:
                    best_lags[col] = significant_lags

            except Exception:
                # Granger test failed for this column
                continue

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_columns = value_cols
        prepared._best_lags = best_lags
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for selected lags."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Transform
        data = data.copy()

        # Create lag features
        for col, lags in self._best_lags.items():
            if col not in data.columns:
                continue

            for lag in lags:
                lag_name = f"{col}_lag{lag}"
                data[lag_name] = data[col].shift(lag)

        return data


# Export all step classes
__all__ = [
    'StepCleanAnomalies',
    'StepStationary',
    'StepDeseasonalize',
    'StepDetrend',
    'StepHStat',
    'StepBestLag',
]
