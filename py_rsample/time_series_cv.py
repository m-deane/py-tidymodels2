"""
Time series cross-validation with rolling and expanding windows

Provides time_series_cv() for creating multiple train/test folds for time series validation.
"""

from typing import Union, List
import pandas as pd
import numpy as np

from py_rsample.split import Split, RSplit
from py_rsample.period_parser import parse_period


class TimeSeriesCV:
    """
    Time series cross-validation with rolling or expanding windows.

    Creates multiple chronological train/test splits for time series validation.
    Supports both rolling window (fixed training size) and expanding window
    (growing training size).

    Attributes:
        data: Original DataFrame
        date_column: Name of date column
        initial: Initial training period size
        assess: Assessment (test) period size
        skip: Gap between consecutive folds
        cumulative: If True, use expanding window; if False, use rolling window
        lag: Gap between train and test (forecast horizon)
        splits: List of RSplit objects

    Example:
        >>> cv = TimeSeriesCV(
        ...     data=sales_data,
        ...     date_column="date",
        ...     initial="2 years",
        ...     assess="3 months",
        ...     skip=0,
        ...     cumulative=True,  # Expanding window
        ...     lag="1 month"
        ... )
        >>> for split in cv:
        ...     train = split.training()
        ...     test = split.testing()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str,
        initial: Union[str, int, pd.Timedelta],
        assess: Union[str, int, pd.Timedelta],
        skip: Union[str, int, pd.Timedelta] = 0,
        cumulative: bool = True,
        lag: Union[str, int, pd.Timedelta] = 0,
    ):
        """
        Initialize time series cross-validation.

        Args:
            data: DataFrame with time series data
            date_column: Name of date column
            initial: Initial training period size
                - int: Number of rows
                - str: Period like "2 years"
                - Timedelta: pandas Timedelta
            assess: Assessment (test) period size
            skip: Gap between consecutive folds (default 0 = no gap)
            cumulative: If True, use expanding window; if False, use rolling window
            lag: Gap between train and test (forecast horizon)

        Raises:
            ValueError: If parameters are invalid
        """
        self.data = data
        self.date_column = date_column
        self.cumulative = cumulative

        # Validate date column
        if date_column not in data.columns:
            raise ValueError(
                f"Date column '{date_column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            raise ValueError(
                f"Column '{date_column}' must be datetime type, got {data[date_column].dtype}"
            )

        # Parse periods to row counts
        self.initial = parse_period(initial, data, date_column)
        self.assess = parse_period(assess, data, date_column)
        self.skip = parse_period(skip, data, date_column)
        self.lag = parse_period(lag, data, date_column)

        # Validate parameters
        if self.initial < 1:
            raise ValueError(f"initial must be at least 1, got {self.initial}")
        if self.assess < 1:
            raise ValueError(f"assess must be at least 1, got {self.assess}")
        if self.skip < 0:
            raise ValueError(f"skip must be non-negative, got {self.skip}")
        if self.lag < 0:
            raise ValueError(f"lag must be non-negative, got {self.lag}")

        # Create splits
        self.splits = self._create_splits()

    def _create_splits(self) -> List[RSplit]:
        """
        Create all train/test splits.

        Returns:
            List of RSplit objects
        """
        n = len(self.data)
        splits = []

        # Starting position for first training set (always 0)
        # Ending position for first training set
        train_end = self.initial

        # Counter for split IDs
        fold_num = 1

        while True:
            # Calculate test start and end
            test_start = train_end + self.lag
            test_end = test_start + self.assess

            # Check if we have enough data for the test set
            if test_end > n:
                break

            # Create indices
            if self.cumulative:
                # Expanding window: training set grows from 0 to train_end
                train_indices = np.arange(0, train_end)
            else:
                # Rolling window: fixed-size training set
                train_indices = np.arange(train_end - self.initial, train_end)

            test_indices = np.arange(test_start, test_end)

            # Create split
            split = Split(
                data=self.data,
                in_id=train_indices,
                out_id=test_indices,
                id=f"Slice{fold_num:03d}"
            )

            splits.append(RSplit(split))

            # Move to next fold
            # skip + assess moves us to the next assessment period
            train_end += self.skip + self.assess
            fold_num += 1

        if len(splits) == 0:
            raise ValueError(
                f"Not enough data to create any splits. "
                f"Need at least {self.initial + self.lag + self.assess} rows, "
                f"have {n} rows"
            )

        return splits

    def __iter__(self):
        """Iterate over splits"""
        return iter(self.splits)

    def __len__(self):
        """Number of splits"""
        return len(self.splits)

    def __getitem__(self, index):
        """Get split by index"""
        return self.splits[index]

    def __repr__(self):
        """String representation"""
        window_type = "expanding" if self.cumulative else "rolling"
        return (
            f"TimeSeriesCV({window_type} window, {len(self.splits)} splits, "
            f"initial={self.initial}, assess={self.assess}, lag={self.lag})"
        )


def time_series_cv(
    data: pd.DataFrame,
    date_column: str,
    initial: Union[str, int, pd.Timedelta],
    assess: Union[str, int, pd.Timedelta],
    skip: Union[str, int, pd.Timedelta] = 0,
    cumulative: bool = True,
    lag: Union[str, int, pd.Timedelta] = 0,
) -> TimeSeriesCV:
    """
    Create time series cross-validation splits.

    Creates multiple chronological train/test splits for time series validation.
    Supports both rolling window (fixed training size) and expanding window
    (growing training size).

    Args:
        data: DataFrame with time series data
        date_column: Name of date column
        initial: Initial training period size
            - int: Number of rows
            - str: Period like "2 years", "6 months"
            - Timedelta: pandas Timedelta
        assess: Assessment (test) period size
        skip: Gap between consecutive folds (default 0)
        cumulative: If True, use expanding window; if False, use rolling window
        lag: Gap between train and test (forecast horizon)

    Returns:
        TimeSeriesCV object (iterable of RSplit objects)

    Examples:
        >>> # Expanding window: 2-year initial, 3-month assessment
        >>> cv = time_series_cv(
        ...     data=sales_data,
        ...     date_column="date",
        ...     initial="2 years",
        ...     assess="3 months",
        ...     cumulative=True
        ... )
        >>> for split in cv:
        ...     train = split.training()
        ...     test = split.testing()

        >>> # Rolling window with 1-month lag
        >>> cv = time_series_cv(
        ...     data=sales_data,
        ...     date_column="date",
        ...     initial="1 year",
        ...     assess="1 month",
        ...     cumulative=False,  # Rolling window
        ...     lag="1 month"
        ... )

        >>> # Integer-based (row counts)
        >>> cv = time_series_cv(
        ...     data=sales_data,
        ...     date_column="date",
        ...     initial=100,  # 100 rows
        ...     assess=20,    # 20 rows
        ...     skip=10       # Skip 10 rows between folds
        ... )
    """
    return TimeSeriesCV(
        data=data,
        date_column=date_column,
        initial=initial,
        assess=assess,
        skip=skip,
        cumulative=cumulative,
        lag=lag,
    )
