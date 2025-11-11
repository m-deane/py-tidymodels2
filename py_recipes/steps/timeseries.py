"""
Time series preprocessing steps

Provides lag features, differencing, percent changes, and rolling statistics.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union, Callable
import pandas as pd
import numpy as np

from py_recipes.selectors import resolve_selector, all_numeric


@dataclass
class StepLag:
    """
    Create lag features for time series data.

    Creates lagged versions of specified columns to capture temporal patterns.

    Attributes:
        columns: Columns to create lags for (selector function, column names, or None for all_numeric())
        lags: List of lag periods (e.g., [1, 2, 7] for 1-day, 2-day, 7-day lags)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    lags: List[int] = None

    def __post_init__(self):
        if self.lags is None:
            self.lags = [1]

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepLag":
        """
        Prepare lag step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepLag ready to create lag features
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepLag(
            columns=cols,
            lags=self.lags
        )


@dataclass
class PreparedStepLag:
    """
    Fitted lag step.

    Attributes:
        columns: Columns to lag
        lags: Lag periods
    """

    columns: List[str]
    lags: List[int]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply lag transformation to new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with lag features added
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                for lag in self.lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    result[lag_col_name] = result[col].shift(lag)

        return result


@dataclass
class StepDiff:
    """
    Create differenced features.

    Computes differences between consecutive observations to make
    time series stationary.

    Attributes:
        columns: Columns to difference (selector function, column names, or None for all_numeric())
        lag: Period for differencing (default 1)
        differences: Number of times to difference (default 1)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    lag: int = 1
    differences: int = 1

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepDiff":
        """
        Prepare differencing step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepDiff ready to difference data
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepDiff(
            columns=cols,
            lag=self.lag,
            differences=self.differences
        )


@dataclass
class PreparedStepDiff:
    """
    Fitted differencing step.

    Attributes:
        columns: Columns to difference
        lag: Differencing lag
        differences: Number of differences
    """

    columns: List[str]
    lag: int
    differences: int

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply differencing to new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with differenced columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                diff_col = result[col]
                for _ in range(self.differences):
                    diff_col = diff_col.diff(periods=self.lag)

                # Include "lag" in column name for clarity
                if self.differences == 1:
                    diff_col_name = f"{col}_diff_lag_{self.lag}"
                else:
                    diff_col_name = f"{col}_diff_lag_{self.lag}_order_{self.differences}"
                result[diff_col_name] = diff_col

        return result


@dataclass
class StepPctChange:
    """
    Create percent change features.

    Computes percentage changes between consecutive observations.

    Attributes:
        columns: Columns to compute percent changes for (selector function, column names, or None for all_numeric())
        periods: Number of periods for change calculation (default 1)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    periods: int = 1

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPctChange":
        """
        Prepare percent change step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPctChange ready to compute percent changes
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepPctChange(
            columns=cols,
            periods=self.periods
        )


@dataclass
class PreparedStepPctChange:
    """
    Fitted percent change step.

    Attributes:
        columns: Columns to transform
        periods: Periods for calculation
    """

    columns: List[str]
    periods: int

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply percent change transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with percent change features
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                pct_col_name = f"{col}_pct_change_{self.periods}"
                result[pct_col_name] = result[col].pct_change(periods=self.periods)

        return result


@dataclass
class StepRolling:
    """
    Create rolling window statistics.

    Computes statistics over rolling windows (mean, std, min, max, sum).

    Attributes:
        columns: Columns to compute rolling stats for (selector function, column names, or None for all_numeric())
        window: Size of rolling window
        stats: Statistics to compute (mean, std, min, max, sum)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    window: int = 1
    stats: List[str] = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = ["mean"]

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepRolling":
        """
        Prepare rolling window step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepRolling ready to compute rolling stats
        """
        # Use resolve_selector for consistent column selection
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepRolling(
            columns=cols,
            window=self.window,
            stats=self.stats
        )


@dataclass
class PreparedStepRolling:
    """
    Fitted rolling window step.

    Attributes:
        columns: Columns to transform
        window: Window size
        stats: Statistics to compute
    """

    columns: List[str]
    window: int
    stats: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling window statistics.

        Args:
            data: Data to transform

        Returns:
            DataFrame with rolling statistics
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                rolling = result[col].rolling(window=self.window)

                for stat in self.stats:
                    stat_col_name = f"{col}_rolling_{self.window}_{stat}"

                    if stat == "mean":
                        result[stat_col_name] = rolling.mean()
                    elif stat == "std":
                        result[stat_col_name] = rolling.std()
                    elif stat == "min":
                        result[stat_col_name] = rolling.min()
                    elif stat == "max":
                        result[stat_col_name] = rolling.max()
                    elif stat == "sum":
                        result[stat_col_name] = rolling.sum()

        return result


@dataclass
class StepDate:
    """
    Extract date/time features from datetime columns.

    Creates features like year, month, day, dayofweek, quarter, etc.

    Attributes:
        column: Datetime column to extract features from
        features: List of features to extract (year, month, day, dayofweek, quarter, etc.)
        keep_original_date: Whether to keep the original date column (default: False)
    """

    column: str
    features: List[str] = None
    keep_original_date: bool = False

    def __post_init__(self):
        if self.features is None:
            self.features = ["year", "month", "day", "dayofweek"]

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepDate":
        """
        Prepare date feature extraction.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepDate ready to extract date features
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data")

        return PreparedStepDate(
            column=self.column,
            features=self.features,
            keep_original_date=self.keep_original_date
        )


@dataclass
class PreparedStepDate:
    """
    Fitted date extraction step.

    Attributes:
        column: Source datetime column
        features: Features to extract
        keep_original_date: Whether to keep the original date column
    """

    column: str
    features: List[str]
    keep_original_date: bool = False

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with date features
        """
        result = data.copy()

        if self.column not in result.columns:
            return result

        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[self.column]):
            result[self.column] = pd.to_datetime(result[self.column])

        dt = result[self.column].dt

        for feature in self.features:
            feature_col_name = f"{self.column}_{feature}"

            if feature == "year":
                result[feature_col_name] = dt.year
            elif feature == "month":
                result[feature_col_name] = dt.month
            elif feature == "day":
                result[feature_col_name] = dt.day
            elif feature == "dayofweek":
                result[feature_col_name] = dt.dayofweek
            elif feature == "dayofyear":
                result[feature_col_name] = dt.dayofyear
            elif feature == "quarter":
                result[feature_col_name] = dt.quarter
            elif feature == "week":
                result[feature_col_name] = dt.isocalendar().week
            elif feature == "hour":
                result[feature_col_name] = dt.hour
            elif feature == "minute":
                result[feature_col_name] = dt.minute
            elif feature == "is_weekend":
                result[feature_col_name] = (dt.dayofweek >= 5).astype(int)
            elif feature == "is_month_start":
                result[feature_col_name] = dt.is_month_start.astype(int)
            elif feature == "is_month_end":
                result[feature_col_name] = dt.is_month_end.astype(int)
            elif feature == "is_quarter_start":
                result[feature_col_name] = dt.is_quarter_start.astype(int)
            elif feature == "is_quarter_end":
                result[feature_col_name] = dt.is_quarter_end.astype(int)
            elif feature == "is_year_start":
                result[feature_col_name] = dt.is_year_start.astype(int)
            elif feature == "is_year_end":
                result[feature_col_name] = dt.is_year_end.astype(int)

        # Remove original date column unless explicitly kept
        if not self.keep_original_date and self.column in result.columns:
            result = result.drop(columns=[self.column])

        return result
