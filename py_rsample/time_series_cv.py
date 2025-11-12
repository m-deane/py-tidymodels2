"""
Time series cross-validation with rolling and expanding windows

Provides time_series_cv() for creating multiple train/test folds for time series validation.
Also provides time_series_nested_cv() for group-aware CV splits.
"""

from typing import Union, List, Dict
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
        slice_limit: int = None,
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
            slice_limit: Maximum number of folds to create (default None = all folds)

        Raises:
            ValueError: If parameters are invalid
        """
        self.data = data
        self.date_column = date_column
        self.cumulative = cumulative
        self.slice_limit = slice_limit

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
    slice_limit: int = None,
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
        slice_limit: Maximum number of folds to create (default None = all folds)

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
        slice_limit=slice_limit,
    )


def time_series_nested_cv(
    data: pd.DataFrame,
    group_col: str,
    date_column: str,
    initial: Union[str, int, pd.Timedelta],
    assess: Union[str, int, pd.Timedelta],
    skip: Union[str, int, pd.Timedelta] = 0,
    cumulative: bool = True,
    lag: Union[str, int, pd.Timedelta] = 0,
    slice_limit: int = None,
) -> Dict[str, TimeSeriesCV]:
    """
    Create time series cross-validation splits per group (group-aware CV).

    This is a convenience function that creates separate time series CV splits
    for each unique value in the group column. Each group gets its own CV splits
    based on that group's data. Designed to work seamlessly with
    WorkflowSet.fit_nested_resamples().

    Args:
        data: DataFrame with time series data
        group_col: Column name identifying groups (e.g., 'country', 'store_id')
        date_column: Name of date column
        initial: Initial training period size
            - int: Number of rows
            - str: Period like "2 years", "6 months"
            - Timedelta: pandas Timedelta
        assess: Assessment (test) period size
        skip: Gap between consecutive folds (default 0)
        cumulative: If True, use expanding window; if False, use rolling window
        lag: Gap between train and test (forecast horizon)
        slice_limit: Maximum number of folds per group (default None = all folds)

    Returns:
        Dictionary mapping group names to TimeSeriesCV objects
        Example: {'USA': TimeSeriesCV(...), 'Germany': TimeSeriesCV(...), ...}

    Examples:
        >>> # Create CV splits per country (nested modeling)
        >>> cv_by_country = time_series_nested_cv(
        ...     data=sales_data,
        ...     group_col='country',
        ...     date_column='date',
        ...     initial='18 months',
        ...     assess='3 months',
        ...     skip='2 months',
        ...     cumulative=False
        ... )
        >>>
        >>> # Use with WorkflowSet.fit_nested_resamples()
        >>> from py_workflowsets import WorkflowSet
        >>> from py_yardstick import metric_set, rmse, mae
        >>>
        >>> wf_set = WorkflowSet.from_cross(
        ...     preproc=["sales ~ price", "sales ~ price + promotion"],
        ...     models=[linear_reg()]
        ... )
        >>>
        >>> results = wf_set.fit_nested_resamples(
        ...     resamples=cv_by_country,
        ...     group_col='country',
        ...     metrics=metric_set(rmse, mae)
        ... )

    Notes:
        - Each group gets its own independent CV splits based on that group's data
        - Periods (initial, assess, skip, lag) are parsed relative to each group's date range
        - Groups with insufficient data will raise ValueError from TimeSeriesCV
        - Use this for per-group modeling (fit_nested_resamples)
        - For global modeling, use time_series_global_cv() instead
    """
    # Validate group column
    if group_col not in data.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Get unique groups
    groups = data[group_col].unique()

    # Create CV splits per group
    cv_by_group = {}
    failed_groups = []

    for group in groups:
        group_data = data[data[group_col] == group].copy()

        try:
            cv_by_group[group] = time_series_cv(
                data=group_data,
                date_column=date_column,
                initial=initial,
                assess=assess,
                skip=skip,
                cumulative=cumulative,
                lag=lag,
                slice_limit=slice_limit
            )
        except ValueError as e:
            # Group has insufficient data
            failed_groups.append((group, str(e)))

    # Raise error if any groups failed
    if failed_groups:
        error_msg = f"Failed to create CV splits for {len(failed_groups)} group(s):\n"
        for group, error in failed_groups:
            error_msg += f"  - {group}: {error}\n"
        raise ValueError(error_msg)

    # Report success
    total_splits = sum(len(cv) for cv in cv_by_group.values())
    print(f"✓ Created CV splits for {len(cv_by_group)} groups ({total_splits} total folds)")

    return cv_by_group


def time_series_global_cv(
    data: pd.DataFrame,
    group_col: str,
    date_column: str,
    initial: Union[str, int, pd.Timedelta],
    assess: Union[str, int, pd.Timedelta],
    skip: Union[str, int, pd.Timedelta] = 0,
    cumulative: bool = True,
    lag: Union[str, int, pd.Timedelta] = 0,
    slice_limit: int = None,
) -> Dict[str, TimeSeriesCV]:
    """
    Create time series cross-validation splits on full dataset for global modeling.

    This function creates CV splits on the FULL dataset (not per-group), then returns
    a dictionary where each group gets the same CV splits. This is designed to work
    with WorkflowSet.fit_global_resamples(), which fits a global model with group as
    a feature and evaluates performance per-group.

    Args:
        data: DataFrame with time series data
        group_col: Column name identifying groups (e.g., 'country', 'store_id')
        date_column: Name of date column
        initial: Initial training period size
            - int: Number of rows
            - str: Period like "2 years", "6 months"
            - Timedelta: pandas Timedelta
        assess: Assessment (test) period size
        skip: Gap between consecutive folds (default 0)
        cumulative: If True, use expanding window; if False, use rolling window
        lag: Gap between train and test (forecast horizon)
        slice_limit: Maximum number of folds (default None = all folds)

    Returns:
        Dictionary mapping group names to the same TimeSeriesCV object
        Example: {'USA': cv, 'Germany': cv, 'Japan': cv, ...}
        (all groups get the same CV splits)

    Examples:
        >>> # Create CV splits on full dataset (global modeling)
        >>> cv_by_country = time_series_global_cv(
        ...     data=sales_data,
        ...     group_col='country',
        ...     date_column='date',
        ...     initial='18 months',
        ...     assess='3 months',
        ...     skip='2 months',
        ...     cumulative=False
        ... )
        >>>
        >>> # Use with WorkflowSet.fit_global_resamples()
        >>> from py_workflowsets import WorkflowSet
        >>> from py_yardstick import metric_set, rmse, mae
        >>>
        >>> wf_set = WorkflowSet.from_cross(
        ...     preproc=["sales ~ price", "sales ~ price + promotion"],
        ...     models=[linear_reg()]
        ... )
        >>>
        >>> results = wf_set.fit_global_resamples(
        ...     data=sales_data,
        ...     resamples=cv_by_country,
        ...     group_col='country',
        ...     metrics=metric_set(rmse, mae)
        ... )

    Notes:
        - Creates CV splits on the FULL dataset (all groups combined)
        - All groups get the same CV splits (same train/test indices)
        - fit_global_resamples() fits a global model, then evaluates per-group
        - Use this for global modeling (fit_global_resamples)
        - For per-group modeling, use time_series_nested_cv() instead

    Key Difference from time_series_nested_cv():
        - time_series_nested_cv(): Each group gets its own CV splits
        - time_series_global_cv(): All groups share the same CV splits
    """
    # Validate group column
    if group_col not in data.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Get unique groups
    groups = data[group_col].unique()

    # Create CV splits on FULL dataset (not per-group)
    cv_global = time_series_cv(
        data=data,
        date_column=date_column,
        initial=initial,
        assess=assess,
        skip=skip,
        cumulative=cumulative,
        lag=lag,
        slice_limit=slice_limit
    )

    # Return same CV object for all groups
    cv_by_group = {group: cv_global for group in groups}

    # Report success
    print(f"✓ Created global CV splits for {len(cv_by_group)} groups ({len(cv_global)} folds)")

    return cv_by_group
