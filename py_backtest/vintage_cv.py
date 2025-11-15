"""
Vintage cross-validation with data vintages for backtesting.

Provides VintageCV class for creating multiple train/test folds with
vintage-aware data selection that simulates production forecasting conditions.
"""

from typing import Union, List
import pandas as pd
import numpy as np
import warnings

from py_rsample.period_parser import parse_period
from py_backtest.vintage_split import VintageSplit, VintageRSplit
from py_backtest.vintage_utils import validate_vintage_data, select_vintage


class VintageCV:
    """
    Vintage cross-validation with point-in-time data.

    Creates multiple chronological train/test splits for backtesting with
    data vintages. Each fold uses only data available at that point in time,
    simulating production forecasting conditions where data gets revised.

    Attributes:
        data: Original vintage DataFrame
        as_of_col: Name of vintage date column
        date_col: Name of observation date column
        initial: Initial training period size
        assess: Assessment (test) period size
        skip: Gap between consecutive folds
        lag: Gap between train and test (forecast horizon)
        vintage_selection: Strategy for selecting vintages
        splits: List of VintageRSplit objects

    Example:
        >>> vintage_cv = VintageCV(
        ...     data=margins_with_vintages,
        ...     as_of_col="as_of_date",
        ...     date_col="date",
        ...     initial="2 years",
        ...     assess="3 months",
        ...     skip="1 month",
        ...     lag="1 week",
        ...     vintage_selection="latest"
        ... )
        >>> for split in vintage_cv:
        ...     train = split.training()  # Uses vintage data
        ...     test = split.testing()    # Uses final data
    """

    def __init__(
        self,
        data: pd.DataFrame,
        as_of_col: str,
        date_col: str,
        initial: Union[str, int, pd.Timedelta],
        assess: Union[str, int, pd.Timedelta],
        skip: Union[str, int, pd.Timedelta] = 0,
        lag: Union[str, int, pd.Timedelta] = 0,
        vintage_selection: str = "latest",
        slice_limit: int = None,
    ):
        """
        Initialize vintage cross-validation.

        Args:
            data: DataFrame with vintage data (must have as_of_col and date_col)
            as_of_col: Name of vintage date column (when snapshot was taken)
            date_col: Name of observation date column (what data refers to)
            initial: Initial training period size
                - int: Number of unique observation dates
                - str: Period like "2 years"
                - Timedelta: pandas Timedelta
            assess: Assessment (test) period size
            skip: Gap between consecutive folds (default 0 = no gap)
            lag: Gap between train and test (forecast horizon)
            vintage_selection: Strategy for selecting vintage
                - "latest": Use most recent vintage available at forecast time
                - "exact": Use exact vintage date match
            slice_limit: Maximum number of folds to create (default None = all folds)

        Raises:
            ValueError: If parameters are invalid or data structure is wrong
        """
        self.data = data
        self.as_of_col = as_of_col
        self.date_col = date_col
        self.vintage_selection = vintage_selection
        self.slice_limit = slice_limit

        # Validate vintage data structure
        validate_vintage_data(data, as_of_col, date_col)

        # Get unique observation dates for period parsing
        unique_dates = data[date_col].drop_duplicates().sort_values().reset_index(drop=True)
        date_data = pd.DataFrame({date_col: unique_dates})

        # Parse periods to counts of unique observation dates
        self.initial = parse_period(initial, date_data, date_col)
        self.assess = parse_period(assess, date_data, date_col)
        self.skip = parse_period(skip, date_data, date_col)
        self.lag = parse_period(lag, date_data, date_col)

        # Validate parameters
        if self.initial < 1:
            raise ValueError(f"initial must be at least 1, got {self.initial}")
        if self.assess < 1:
            raise ValueError(f"assess must be at least 1, got {self.assess}")
        if self.skip < 0:
            raise ValueError(f"skip must be non-negative, got {self.skip}")
        if self.lag < 0:
            raise ValueError(f"lag must be non-negative, got {self.lag}")

        # Store unique dates for split creation
        self.unique_dates = unique_dates.values

        # Create splits
        self.splits = self._create_splits()

    def _create_splits(self) -> List[VintageRSplit]:
        """
        Create all vintage-aware train/test splits.

        Returns:
            List of VintageRSplit objects
        """
        n_dates = len(self.unique_dates)
        splits = []

        # Starting position for first training set (always 0)
        train_end_idx = self.initial

        # Counter for split IDs
        fold_num = 1

        while True:
            # Calculate test start and end indices
            test_start_idx = train_end_idx + self.lag
            test_end_idx = test_start_idx + self.assess

            # Check if we have enough data for the test set
            if test_end_idx > n_dates:
                break

            # Get actual dates for this fold
            training_start = self.unique_dates[0]
            training_end = self.unique_dates[train_end_idx - 1]
            test_start = self.unique_dates[test_start_idx]
            test_end = self.unique_dates[test_end_idx - 1]

            # Vintage date is the training end date (last date we have data for)
            # This simulates forecasting on training_end, using data available at that time
            vintage_date = pd.Timestamp(training_end)

            # Create dummy indices (actual data selection happens in VintageRSplit)
            # We use empty arrays since VintageRSplit uses date ranges instead
            train_indices = np.array([])
            test_indices = np.array([])

            # Create vintage split
            split = VintageSplit(
                data=self.data,
                in_id=train_indices,
                out_id=test_indices,
                id=f"Vintage{fold_num:03d}",
                vintage_date=vintage_date,
                training_start=pd.Timestamp(training_start),
                training_end=pd.Timestamp(training_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                as_of_col=self.as_of_col,
                date_col=self.date_col,
                vintage_selection=self.vintage_selection
            )

            splits.append(VintageRSplit(split))

            # Move to next fold
            train_end_idx += self.skip + self.assess
            fold_num += 1

        if len(splits) == 0:
            min_required = self.initial + self.lag + self.assess
            raise ValueError(
                f"Not enough data to create any splits. "
                f"Need at least {min_required} unique observation dates, "
                f"have {n_dates} unique dates"
            )

        # Limit number of splits if requested
        if self.slice_limit is not None:
            splits = splits[:self.slice_limit]

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
        return (
            f"VintageCV({len(self.splits)} splits, "
            f"initial={self.initial} dates, assess={self.assess} dates, "
            f"lag={self.lag} dates, vintage_selection='{self.vintage_selection}')"
        )


def vintage_cv(
    data: pd.DataFrame,
    as_of_col: str,
    date_col: str,
    initial: Union[str, int, pd.Timedelta],
    assess: Union[str, int, pd.Timedelta],
    skip: Union[str, int, pd.Timedelta] = 0,
    lag: Union[str, int, pd.Timedelta] = 0,
    vintage_selection: str = "latest",
    slice_limit: int = None,
) -> VintageCV:
    """
    Create vintage cross-validation splits for backtesting.

    Convenience function that creates VintageCV object with vintage-aware
    train/test splits for point-in-time backtesting.

    Args:
        data: DataFrame with vintage data
        as_of_col: Name of vintage date column
        date_col: Name of observation date column
        initial: Initial training period size
        assess: Assessment (test) period size
        skip: Gap between consecutive folds (default 0)
        lag: Gap between train and test (forecast horizon)
        vintage_selection: Strategy for selecting vintage ("latest" or "exact")
        slice_limit: Maximum number of folds (default None = all folds)

    Returns:
        VintageCV object (iterable of VintageRSplit objects)

    Examples:
        >>> # Create vintage CV with 2-year initial, 3-month assessment
        >>> cv = vintage_cv(
        ...     data=margins_with_vintages,
        ...     as_of_col="as_of_date",
        ...     date_col="date",
        ...     initial="2 years",
        ...     assess="3 months",
        ...     lag="1 week"
        ... )
        >>> for split in cv:
        ...     train = split.training()  # Uses vintage data
        ...     test = split.testing()    # Uses final data
        ...     info = split.get_vintage_info()

        >>> # Integer-based (counts of unique dates)
        >>> cv = vintage_cv(
        ...     data=vintage_df,
        ...     as_of_col="as_of_date",
        ...     date_col="date",
        ...     initial=100,  # 100 unique dates
        ...     assess=20,    # 20 unique dates
        ...     skip=10       # Skip 10 dates between folds
        ... )
    """
    return VintageCV(
        data=data,
        as_of_col=as_of_col,
        date_col=date_col,
        initial=initial,
        assess=assess,
        skip=skip,
        lag=lag,
        vintage_selection=vintage_selection,
        slice_limit=slice_limit,
    )
