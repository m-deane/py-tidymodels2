"""
Financial machine learning preprocessing steps.

Provides advanced financial ML transformations including fractional differentiation,
volatility estimation, entropy features, and sample weighting.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Literal
import pandas as pd
import numpy as np
from itertools import permutations

from py_recipes.selectors import resolve_selector, all_numeric


@dataclass
class StepFractionalDiff:
    """
    Apply fractional differentiation to achieve stationarity while preserving memory.

    Uses Fixed-Width Window Fractional Differentiation (FFD) method to transform
    time series to be stationary while preserving long-term memory. This is an
    alternative to integer differencing that can preserve more information.

    Attributes:
        columns: Columns to differentiate (selector function, column names, or None for all_numeric())
        d: Fractional differentiation order (typically 0-1, default: 0.5). If None, auto-calculates optimal d.
        auto_d: If True and d is None, automatically finds optimal d using stationarity tests (default: False)
        d_range: Range of d values to test when auto_d=True (default: [0.1, 0.2, ..., 0.9])
        stationarity_test: Test to use for stationarity ('adf' or 'kpss', default: 'adf')
        alpha: Significance level for stationarity test (default: 0.05)
        threshold: Threshold for determining optimal window (default: 1e-5)
        window: Window size for FFD method (default: None, auto-determined)
        date_col: Optional date/time column for ordering
        group_col: Optional grouping column for panel data
        prefix: Prefix for created columns (default: 'frac_diff_')
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    d: Optional[float] = 0.5
    auto_d: bool = False
    d_range: Optional[List[float]] = None
    stationarity_test: str = 'adf'
    alpha: float = 0.05
    threshold: float = 1e-5
    window: Optional[int] = None
    date_col: Optional[str] = None
    group_col: Optional[str] = None
    prefix: str = "frac_diff_"

    def __post_init__(self):
        """Validate parameters."""
        if self.d is not None and not (0 <= self.d <= 1):
            raise ValueError(f"d must be between 0 and 1, got {self.d}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.stationarity_test not in ['adf', 'kpss']:
            raise ValueError(f"stationarity_test must be 'adf' or 'kpss', got {self.stationarity_test}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.d_range is not None:
            if not all(0 <= d_val <= 1 for d_val in self.d_range):
                raise ValueError(f"All values in d_range must be between 0 and 1")

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepFractionalDiff":
        """
        Prepare fractional differentiation step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepFractionalDiff ready to apply fractional differentiation
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        if len(cols) == 0:
            raise ValueError("No columns to transform after resolving selector")

        # Determine optimal d if auto_d is enabled
        d = self.d
        if (d is None or self.auto_d) and training:
            d = self._find_optimal_d(data, cols)

        # If d is still None, use default
        if d is None:
            d = 0.5

        # Determine window size if not provided
        window = self.window
        if window is None:
            # Auto-determine window based on threshold
            # Find window where weights decay below threshold
            window = self._find_optimal_window(d, self.threshold)
            # Cap window to data length to avoid all-NaN results
            if self.group_col:
                # For grouped data, use minimum group size
                min_group_size = data.groupby(self.group_col).size().min()
                window = min(window, min_group_size)
            else:
                window = min(window, len(data))

        # Validate date column if provided
        if self.date_col and self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        # Validate group column if provided
        if self.group_col and self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        return PreparedStepFractionalDiff(
            columns=cols,
            d=d,
            threshold=self.threshold,
            window=window,
            date_col=self.date_col,
            group_col=self.group_col,
            prefix=self.prefix,
        )

    def _find_optimal_d(self, data: pd.DataFrame, cols: List[str]) -> float:
        """
        Find optimal d value that makes the series stationary while preserving maximum memory.
        
        Tests different d values and selects the smallest d that makes the series stationary.
        This maximizes memory preservation while achieving stationarity.
        
        Args:
            data: Training data
            cols: Columns to test
            
        Returns:
            Optimal d value
        """
        # Determine d_range to test
        if self.d_range is None:
            d_range = [round(0.1 * i, 1) for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
        else:
            d_range = sorted(self.d_range)
        
        # Use first column for testing (or average across columns)
        test_col = cols[0]
        
        # Get series, handling grouped data
        if self.group_col:
            # For grouped data, use first group
            groups = data.groupby(self.group_col)
            first_group = next(iter(groups))[1]
            if self.date_col and self.date_col in first_group.columns:
                first_group = first_group.sort_values(by=self.date_col)
            series = first_group[test_col].copy()
        else:
            series = data[test_col].copy()
            # Sort by date if date column provided
            if self.date_col and self.date_col in data.columns:
                data_sorted = data.sort_values(by=self.date_col)
                series = data_sorted[test_col].copy()
        
        # Remove NaN values
        series = series.dropna()
        
        if len(series) < 20:  # Need minimum data for testing
            return 0.5  # Default fallback
        
        # Test each d value
        optimal_d = None
        for d_test in d_range:
            # Apply fractional differentiation with this d
            try:
                diff_series = self._apply_fractional_diff_to_series(series, d_test)
                
                # Test for stationarity
                if self._is_stationary(diff_series, self.stationarity_test, self.alpha):
                    optimal_d = d_test
                    break
            except Exception:
                # If differentiation fails, skip this d
                continue
        
        # If no d makes it stationary, return the largest d (most differentiation)
        if optimal_d is None:
            optimal_d = d_range[-1]
        
        return optimal_d
    
    def _apply_fractional_diff_to_series(self, series: pd.Series, d: float) -> pd.Series:
        """Apply fractional differentiation to a single series for testing."""
        # Determine window
        window = self._find_optimal_window(d, self.threshold)
        window = min(window, len(series))
        
        # Calculate weights
        weights = np.zeros(window)
        for k in range(window):
            weights[k] = self._fractional_weight(k, d)
        
        # Apply fractional differentiation
        series_arr = series.values
        frac_diff = np.zeros_like(series_arr)
        
        for i in range(len(series_arr)):
            for k in range(min(i + 1, len(weights))):
                if i - k >= 0:
                    frac_diff[i] += weights[k] * series_arr[i - k]
        
        # Center around zero
        if len(frac_diff) > 0 and not np.isnan(frac_diff[0]):
            frac_diff = frac_diff - frac_diff[0]
        
        # Remove NaN values
        result = pd.Series(frac_diff, index=series.index)
        result = result.dropna()
        
        return result
    
    def _is_stationary(self, series: pd.Series, test: str, alpha: float) -> bool:
        """Test if series is stationary using ADF or KPSS test."""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        try:
            series_clean = series.dropna()
            if len(series_clean) < 12:  # Minimum data required
                return False
            
            if test == 'adf':
                # ADF test: H0 = non-stationary (reject if p < alpha)
                result = adfuller(series_clean, autolag='AIC')
                p_value = result[1]
                return p_value < alpha
            
            elif test == 'kpss':
                # KPSS test: H0 = stationary (reject if p < alpha means non-stationary)
                result = kpss(series_clean, regression='c', nlags='auto')
                p_value = result[1]
                return p_value >= alpha  # Don't reject H0 if p >= alpha (stationary)
        
        except Exception:
            # If test fails, assume not stationary
            return False
        
        return False

    def _find_optimal_window(self, d: float, threshold: float) -> int:
        """Find optimal window size where weights decay below threshold."""
        # Calculate weights for fractional differentiation
        # Weight at lag k: w_k = (-1)^k * binom(d, k)
        # We find the window where |w_k| < threshold
        window = 100  # Start with reasonable default
        max_window = 1000  # Maximum window to check

        for w in range(1, max_window):
            # Calculate weight at lag w
            weight = self._fractional_weight(w, d)
            if abs(weight) < threshold:
                window = w
                break

        return min(window, max_window)

    @staticmethod
    def _fractional_weight(k: int, d: float) -> float:
        """Calculate fractional differentiation weight at lag k."""
        if k == 0:
            return 1.0
        # w_k = (-1)^k * binom(d, k)
        # binom(d, k) = d * (d-1) * ... * (d-k+1) / k!
        sign = (-1) ** k
        binom = 1.0
        for i in range(k):
            binom *= (d - i) / (i + 1)
        return sign * binom


@dataclass
class PreparedStepFractionalDiff:
    """
    Fitted fractional differentiation step.

    Attributes:
        columns: Columns to differentiate
        d: Fractional differentiation order
        threshold: Threshold for weight decay
        window: Window size for FFD
        date_col: Optional date column
        group_col: Optional group column
        prefix: Column name prefix
    """

    columns: List[str]
    d: float
    threshold: float
    window: int
    date_col: Optional[str]
    group_col: Optional[str]
    prefix: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fractional differentiation to new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with fractionally differentiated columns added
        """
        if len(data) == 0:
            return data.copy()

        result = data.copy()

        # Sort by date if date column provided
        if self.date_col:
            result = result.sort_values(by=self.date_col)

        # Calculate weights once
        weights = self._calculate_weights()

        # Apply fractional differentiation
        if self.group_col:
            # Grouped application
            result = result.groupby(self.group_col, group_keys=False).apply(
                lambda group: self._apply_fractional_diff(group, weights)
            )
        else:
            # Single series
            result = self._apply_fractional_diff(result, weights)

        return result

    def _calculate_weights(self) -> np.ndarray:
        """Calculate fractional differentiation weights."""
        weights = np.zeros(self.window)
        for k in range(self.window):
            weights[k] = StepFractionalDiff._fractional_weight(k, self.d)
        return weights

    def _apply_fractional_diff(
        self, data: pd.DataFrame, weights: np.ndarray
    ) -> pd.DataFrame:
        """Apply fractional differentiation to a single group."""
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            # Skip if column is date or group column
            if col == self.date_col or col == self.group_col:
                continue

            series = result[col].values

            # Handle insufficient data
            if len(series) < self.window:
                # Not enough data for fractional differentiation
                new_col = f"{self.prefix}{col}"
                result[new_col] = np.nan
                continue

            # Handle all-NaN columns
            if pd.isna(series).all():
                new_col = f"{self.prefix}{col}"
                result[new_col] = np.nan
                continue

            frac_diff = np.zeros_like(series)

            # Apply fractional differentiation using convolution
            # For FFD: (1-L)^d * X_t = sum_{k=0}^{window} w_k * X_{t-k}
            # This produces a stationary series
            for i in range(len(series)):
                # Sum weighted lags
                for k in range(min(i + 1, len(weights))):
                    if i - k >= 0:
                        frac_diff[i] += weights[k] * series[i - k]

            # Center the series around zero by subtracting the first value
            # This improves visualization and makes the stationary nature more apparent
            # The first value is just the original price (weight k=0 = 1.0)
            if len(frac_diff) > 0 and not np.isnan(frac_diff[0]):
                frac_diff = frac_diff - frac_diff[0]

            # Create new column name
            new_col = f"{self.prefix}{col}"
            result[new_col] = frac_diff

        return result


@dataclass
class StepVolatilityEWM:
    """
    Calculate exponentially weighted moving volatility.

    Computes volatility using exponentially weighted moving average of squared returns,
    giving more weight to recent observations. This is the standard approach in
    financial machine learning.

    Attributes:
        return_col: Column containing returns (or price if use_returns=False)
        span: Span for exponential weighting (default: 20)
        use_returns: Whether input column is returns (True) or prices (False)
        date_col: Optional date/time column for ordering
        group_col: Optional grouping column for panel data
        prefix: Prefix for created columns (default: 'vol_ewm_')
    """

    return_col: str
    span: int = 20
    use_returns: bool = True
    date_col: Optional[str] = None
    group_col: Optional[str] = None
    prefix: str = "vol_ewm_"

    def __post_init__(self):
        """Validate parameters."""
        if self.span <= 0:
            raise ValueError(f"span must be positive, got {self.span}")

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepVolatilityEWM":
        """
        Prepare volatility EWM step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepVolatilityEWM ready to calculate volatility
        """
        # Validate return column exists
        if self.return_col not in data.columns:
            raise ValueError(f"Return column '{self.return_col}' not found in data")

        # Validate date column if provided
        if self.date_col and self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        # Validate group column if provided
        if self.group_col and self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        return PreparedStepVolatilityEWM(
            return_col=self.return_col,
            span=self.span,
            use_returns=self.use_returns,
            date_col=self.date_col,
            group_col=self.group_col,
            prefix=self.prefix,
        )


@dataclass
class PreparedStepVolatilityEWM:
    """
    Fitted volatility EWM step.

    Attributes:
        return_col: Return column name
        span: EWM span
        use_returns: Whether input is returns
        date_col: Optional date column
        group_col: Optional group column
        prefix: Column name prefix
    """

    return_col: str
    span: int
    use_returns: bool
    date_col: Optional[str]
    group_col: Optional[str]
    prefix: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving volatility.

        Args:
            data: Data to transform

        Returns:
            DataFrame with volatility column added
        """
        if len(data) == 0:
            return data.copy()

        result = data.copy()

        # Sort by date if date column provided
        if self.date_col:
            result = result.sort_values(by=self.date_col)

        # Calculate volatility
        if self.group_col:
            # Grouped application
            result = result.groupby(self.group_col, group_keys=False).apply(
                self._calculate_volatility
            )
        else:
            # Single series
            result = self._calculate_volatility(result)

        return result

    def _calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility for a single group."""
        result = data.copy()

        if self.return_col not in result.columns:
            return result

        if self.use_returns:
            # Input is already returns
            returns = result[self.return_col]
        else:
            # Input is prices, calculate returns
            returns = result[self.return_col].pct_change()

        # Handle all-NaN or insufficient data
        if returns.isna().all() or len(returns.dropna()) < self.span:
            vol_col = f"{self.prefix}{self.return_col}"
            result[vol_col] = np.nan
            return result

        # Calculate squared returns
        squared_returns = returns ** 2

        # Calculate EWM of squared returns
        ewm_var = squared_returns.ewm(span=self.span, adjust=False).mean()

        # Volatility is square root of variance
        vol = np.sqrt(ewm_var)

        # Create new column name
        vol_col = f"{self.prefix}{self.return_col}"
        result[vol_col] = vol

        return result


@dataclass
class StepPermutationEntropy:
    """
    Calculate permutation entropy for time series.

    Measures complexity and predictability of time series by analyzing the
    ordinal patterns in the data. More robust to noise than Shannon entropy.

    Attributes:
        columns: Columns to calculate entropy for (selector function, column names, or None for all_numeric())
        window: Rolling window for entropy calculation (default: None, uses full series)
        order: Order of permutations (default: 3)
        normalize: Whether to normalize entropy (default: True)
        date_col: Optional date/time column for ordering
        group_col: Optional grouping column for panel data
        prefix: Prefix for created columns (default: 'perm_entropy_')
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    window: Optional[int] = None
    order: int = 3
    normalize: bool = True
    date_col: Optional[str] = None
    group_col: Optional[str] = None
    prefix: str = "perm_entropy_"

    def __post_init__(self):
        """Validate parameters."""
        if self.order < 2:
            raise ValueError(f"order must be >= 2, got {self.order}")
        if self.order > 10:
            raise ValueError(f"order must be <= 10, got {self.order} (computationally expensive)")

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPermutationEntropy":
        """
        Prepare permutation entropy step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPermutationEntropy ready to calculate entropy
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        if len(cols) == 0:
            raise ValueError("No columns to transform after resolving selector")

        # Validate date column if provided
        if self.date_col and self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        # Validate group column if provided
        if self.group_col and self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        return PreparedStepPermutationEntropy(
            columns=cols,
            window=self.window,
            order=self.order,
            normalize=self.normalize,
            date_col=self.date_col,
            group_col=self.group_col,
            prefix=self.prefix,
        )


@dataclass
class PreparedStepPermutationEntropy:
    """
    Fitted permutation entropy step.

    Attributes:
        columns: Columns to calculate entropy for
        window: Rolling window
        order: Order of permutations
        normalize: Whether to normalize
        date_col: Optional date column
        group_col: Optional group column
        prefix: Column name prefix
    """

    columns: List[str]
    window: Optional[int]
    order: int
    normalize: bool
    date_col: Optional[str]
    group_col: Optional[str]
    prefix: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate permutation entropy for new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with permutation entropy columns added
        """
        if len(data) == 0:
            return data.copy()

        result = data.copy()

        # Sort by date if date column provided
        if self.date_col:
            result = result.sort_values(by=self.date_col)

        # Calculate entropy
        if self.group_col:
            # Grouped application
            result = result.groupby(self.group_col, group_keys=False).apply(
                self._calculate_entropy
            )
        else:
            # Single series
            result = self._calculate_entropy(result)

        return result

    def _calculate_entropy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate permutation entropy for a single group."""
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            # Skip if column is date or group column
            if col == self.date_col or col == self.group_col:
                continue

            series = result[col].dropna().values

            # Handle insufficient data or all-NaN
            if len(series) < self.order:
                # Not enough data
                entropy_values = np.full(len(result), np.nan)
            elif len(series) == 0:
                # All NaN values
                entropy_values = np.full(len(result), np.nan)
            elif len(np.unique(series)) == 1:
                # Constant series - entropy is 0
                entropy_values = np.zeros(len(result))
            else:
                if self.window is None:
                    # Calculate single entropy value for entire series
                    entropy = self._permutation_entropy(series)
                    entropy_values = np.full(len(result), entropy)
                else:
                    # Rolling window entropy
                    entropy_values = np.full(len(result), np.nan)
                    for i in range(self.window - 1, len(series)):
                        window_data = series[i - self.window + 1 : i + 1]
                        entropy_values[i] = self._permutation_entropy(window_data)

            # Create new column name
            new_col = f"{self.prefix}{col}"
            result[new_col] = entropy_values

        return result

    def _permutation_entropy(self, series: np.ndarray) -> float:
        """Calculate permutation entropy for a series."""
        n = len(series)
        if n < self.order:
            return np.nan

        # Generate all possible ordinal patterns
        patterns = list(permutations(range(self.order)))
        pattern_counts = {pattern: 0 for pattern in patterns}

        # Count occurrences of each pattern
        for i in range(n - self.order + 1):
            window = series[i : i + self.order]
            # Get ordinal pattern (ranks)
            ranks = tuple(np.argsort(window))
            if ranks in pattern_counts:
                pattern_counts[ranks] += 1

        # Calculate probabilities
        total = sum(pattern_counts.values())
        if total == 0:
            return 0.0

        probabilities = [count / total for count in pattern_counts.values() if count > 0]

        # Calculate Shannon entropy
        entropy = -sum(p * np.log(p) for p in probabilities)

        # Normalize if requested
        if self.normalize:
            max_entropy = np.log(len(patterns))
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return entropy


@dataclass
class StepWeightTimeDecay:
    """
    Apply time decay weighting to observations.

    Assigns higher weights to more recent observations, reflecting the adaptive
    nature of financial markets. Can use linear, exponential, or piecewise decay.

    Attributes:
        date_col: Date/time column for determining observation age
        decay_rate: Decay rate (higher = faster decay, default: 0.1)
        method: Decay method - 'linear', 'exponential', or 'piecewise' (default: 'exponential')
        base_weight: Base weight for oldest observations (default: 1.0)
        weight_col_name: Name for the weight column (default: 'weight')
        group_col: Optional grouping column for panel data
    """

    date_col: str
    decay_rate: float = 0.1
    method: Literal["linear", "exponential", "piecewise"] = "exponential"
    base_weight: float = 1.0
    weight_col_name: str = "weight"
    group_col: Optional[str] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")
        valid_methods = ["linear", "exponential", "piecewise"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        if self.base_weight <= 0:
            raise ValueError(f"base_weight must be positive, got {self.base_weight}")

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepWeightTimeDecay":
        """
        Prepare time decay weighting step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepWeightTimeDecay ready to calculate weights
        """
        # Validate date column exists
        if self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        # Validate group column if provided
        if self.group_col and self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        # Store reference date for normalization
        if training:
            # Use most recent date as reference
            dates = pd.to_datetime(data[self.date_col])
            reference_date = dates.max()
        else:
            # For prediction, use the max date from training (will be set during prep)
            reference_date = None

        return PreparedStepWeightTimeDecay(
            date_col=self.date_col,
            decay_rate=self.decay_rate,
            method=self.method,
            base_weight=self.base_weight,
            weight_col_name=self.weight_col_name,
            group_col=self.group_col,
            reference_date=reference_date,
        )


@dataclass
class PreparedStepWeightTimeDecay:
    """
    Fitted time decay weighting step.

    Attributes:
        date_col: Date column name
        decay_rate: Decay rate
        method: Decay method
        base_weight: Base weight
        weight_col_name: Weight column name
        group_col: Optional group column
        reference_date: Reference date for calculating age
    """

    date_col: str
    decay_rate: float
    method: str
    base_weight: float
    weight_col_name: str
    group_col: Optional[str]
    reference_date: Optional[pd.Timestamp]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time decay weights for new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with weight column added
        """
        if len(data) == 0:
            return data.copy()

        result = data.copy()

        # Sort by date
        result = result.sort_values(by=self.date_col)

        # Calculate weights
        if self.group_col:
            # Grouped application
            result = result.groupby(self.group_col, group_keys=False).apply(
                self._calculate_weights
            )
        else:
            # Single series
            result = self._calculate_weights(result)

        return result

    def _calculate_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate time decay weights for a single group."""
        result = data.copy()

        # Convert dates to datetime
        dates = pd.to_datetime(result[self.date_col])

        # Determine reference date
        if self.reference_date is None:
            # Use most recent date in this group
            reference_date = dates.max()
        else:
            reference_date = self.reference_date

        # Calculate age (days since reference)
        age_days = (reference_date - dates).dt.total_seconds() / (24 * 3600)
        max_age = age_days.max()

        if len(result) == 0:
            # Empty data
            weights = np.array([])
        elif max_age == 0:
            # All dates are the same (single row or identical dates)
            weights = np.full(len(result), self.base_weight)
        else:
            # Normalize age to [0, 1]
            normalized_age = age_days / max_age

            # Calculate weights based on method
            if self.method == "linear":
                # Linear decay: weight = base_weight * (1 - decay_rate * normalized_age)
                weights = self.base_weight * (1 - self.decay_rate * normalized_age)
                weights = np.maximum(weights, 0)  # Ensure non-negative

            elif self.method == "exponential":
                # Exponential decay: weight = base_weight * exp(-decay_rate * normalized_age)
                weights = self.base_weight * np.exp(-self.decay_rate * normalized_age)

            else:  # piecewise
                # Piecewise linear: faster decay initially, then slower
                # weight = base_weight * (1 - decay_rate * normalized_age^2)
                weights = self.base_weight * (1 - self.decay_rate * normalized_age ** 2)
                weights = np.maximum(weights, 0)  # Ensure non-negative

        # Store weights
        result[self.weight_col_name] = weights

        return result

