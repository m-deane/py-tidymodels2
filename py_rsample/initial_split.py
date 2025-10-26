"""
Initial time series train/test split

Provides initial_time_split() for creating a single chronological train/test split.
"""

from typing import Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from py_rsample.split import Split, RSplit
from py_rsample.period_parser import parse_period


def _parse_date_spec(
    date_spec: Union[str, datetime, pd.Timestamp, None],
    data: pd.DataFrame,
    date_column: str,
    reference: str = "start"
) -> Optional[pd.Timestamp]:
    """
    Parse a date specification into an actual timestamp.

    Args:
        date_spec: Date specification (absolute date, relative delta, or None)
        data: DataFrame containing the time series
        date_column: Name of the date column
        reference: Reference point ("start" or "end") for relative deltas

    Returns:
        Parsed timestamp or None

    Examples:
        - Absolute: datetime(2020, 1, 1), "2020-01-01"
        - Relative: "start + 1 year", "end - 3 months"
    """
    if date_spec is None:
        return None

    # Convert to string for processing
    if isinstance(date_spec, (datetime, pd.Timestamp)):
        return pd.Timestamp(date_spec)

    date_str = str(date_spec).strip()

    # Check if it's just "start" or "end"
    if date_str == "start":
        return pd.Timestamp(data[date_column].min())
    elif date_str == "end":
        return pd.Timestamp(data[date_column].max())

    # Check if it's a relative delta
    if "+" in date_str or ("-" in date_str and ("start" in date_str or "end" in date_str)):
        # Parse relative delta like "start + 1 year" or "end - 3 months"
        if "+" in date_str:
            parts = date_str.split("+")
            ref_str = parts[0].strip()
            delta_str = parts[1].strip()
            sign = 1
        else:
            parts = date_str.split("-", 1)
            ref_str = parts[0].strip()
            delta_str = parts[1].strip()
            sign = -1

        # Get reference date
        if ref_str == "start":
            ref_date = data[date_column].min()
        elif ref_str == "end":
            ref_date = data[date_column].max()
        else:
            raise ValueError(f"Unknown reference point: {ref_str}. Use 'start' or 'end'")

        # Parse delta
        delta_rows = parse_period(delta_str, data, date_column)

        # Calculate date by adding/subtracting rows
        date_series = data[date_column].sort_values()
        ref_idx = date_series.searchsorted(ref_date)
        target_idx = ref_idx + (sign * delta_rows)
        target_idx = max(0, min(target_idx, len(date_series) - 1))

        return pd.Timestamp(date_series.iloc[target_idx])

    # Try to parse as absolute date
    try:
        return pd.to_datetime(date_spec)
    except Exception as e:
        raise ValueError(f"Could not parse date specification '{date_spec}': {e}")


def _date_to_index(
    date: Union[pd.Timestamp, datetime, None],
    data: pd.DataFrame,
    date_column: str,
    position: str = "nearest"
) -> Optional[int]:
    """
    Convert a date to a row index in the data.

    Args:
        date: Timestamp to convert
        data: DataFrame containing the time series
        date_column: Name of the date column
        position: How to handle dates between data points
                 - "nearest": Use nearest date
                 - "before": Use last date before or equal to target
                 - "after": Use first date after or equal to target

    Returns:
        Row index or None
    """
    if date is None:
        return None

    date = pd.Timestamp(date)
    date_series = data[date_column]

    if position == "nearest":
        # Find nearest date
        idx = (date_series - date).abs().idxmin()
    elif position == "before":
        # Find last date <= target
        mask = date_series <= date
        if not mask.any():
            raise ValueError(f"No dates on or before {date}")
        idx = date_series[mask].idxmax()
    elif position == "after":
        # Find first date >= target
        mask = date_series >= date
        if not mask.any():
            raise ValueError(f"No dates on or after {date}")
        idx = date_series[mask].idxmin()
    else:
        raise ValueError(f"Unknown position: {position}. Use 'nearest', 'before', or 'after'")

    return data.index.get_loc(idx)


def initial_time_split(
    data: pd.DataFrame,
    prop: Optional[float] = None,
    lag: Union[str, int, pd.Timedelta] = 0,
    date_column: Optional[str] = None,
    train_start: Union[str, datetime, pd.Timestamp, None] = None,
    train_end: Union[str, datetime, pd.Timestamp, None] = None,
    test_start: Union[str, datetime, pd.Timestamp, None] = None,
    test_end: Union[str, datetime, pd.Timestamp, None] = None,
    **kwargs
) -> RSplit:
    """
    Create initial chronological train/test split for time series.

    Supports two modes:
    1. Proportion-based (prop/lag): Simple percentage split
    2. Explicit dates: Specify exact date ranges for train/test

    Args:
        data: DataFrame to split
        prop: Proportion for training (e.g., 0.75 for 75% train). If None,
              uses all but last assessment period
        lag: Gap between train and test (forecast horizon)
             - int: Number of rows
             - str: Period like "1 month", "7 days"
             - Timedelta: pandas Timedelta
        date_column: Name of date column (required for explicit dates or period lags)
        train_start: Start date for training data (absolute or relative)
                    - Absolute: datetime, pd.Timestamp, or "2020-01-01"
                    - Relative: "start + 1 year", "end - 2 years"
        train_end: End date for training data (absolute or relative)
        test_start: Start date for testing data (absolute or relative)
        test_end: End date for testing data (absolute or relative)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        RSplit object with training/testing methods

    Raises:
        ValueError: If parameters are invalid or date column is missing

    Examples:
        >>> # Mode 1: Proportion-based (backward compatible)
        >>> split = initial_time_split(sales_data, prop=0.75)
        >>> train = split.training()
        >>> test = split.testing()

        >>> # With lag (forecast horizon)
        >>> split = initial_time_split(
        ...     sales_data,
        ...     prop=0.8,
        ...     lag="1 month",
        ...     date_column="date"
        ... )

        >>> # Mode 2: Explicit dates - absolute dates
        >>> split = initial_time_split(
        ...     sales_data,
        ...     date_column="date",
        ...     train_start="2020-01-01",
        ...     train_end="2021-12-31",
        ...     test_start="2022-01-01",
        ...     test_end="2022-06-30"
        ... )

        >>> # Relative deltas from data boundaries
        >>> split = initial_time_split(
        ...     sales_data,
        ...     date_column="date",
        ...     train_start="start",
        ...     train_end="start + 2 years",
        ...     test_start="start + 2 years + 1 month",
        ...     test_end="end"
        ... )

        >>> # Mix of absolute and relative
        >>> split = initial_time_split(
        ...     sales_data,
        ...     date_column="date",
        ...     train_start="2020-01-01",
        ...     train_end="end - 6 months",
        ...     test_start="end - 6 months + 1 week",
        ...     test_end="end"
        ... )
    """
    if len(data) < 2:
        raise ValueError("Data must have at least 2 rows")

    n = len(data)

    # Determine which mode to use
    explicit_dates_provided = any([
        train_start is not None,
        train_end is not None,
        test_start is not None,
        test_end is not None
    ])

    if explicit_dates_provided:
        # Mode 2: Explicit date ranges
        if date_column is None:
            raise ValueError("date_column required when using explicit date ranges")

        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")

        # Parse date specifications
        train_start_ts = _parse_date_spec(train_start or "start", data, date_column)
        train_end_ts = _parse_date_spec(train_end, data, date_column) if train_end else None
        test_start_ts = _parse_date_spec(test_start, data, date_column) if test_start else None
        test_end_ts = _parse_date_spec(test_end or "end", data, date_column)

        # Convert dates to indices
        train_start_idx = _date_to_index(train_start_ts, data, date_column, "after")
        train_end_idx = _date_to_index(train_end_ts, data, date_column, "before") if train_end_ts else None
        test_start_idx = _date_to_index(test_start_ts, data, date_column, "after") if test_start_ts else None
        test_end_idx = _date_to_index(test_end_ts, data, date_column, "before")

        # Default: if train_end not specified, use data end
        if train_end_idx is None:
            train_end_idx = n - 1

        # Default: if test_start not specified, start right after train_end
        if test_start_idx is None:
            test_start_idx = train_end_idx + 1

        # Validate indices
        if train_start_idx >= train_end_idx:
            raise ValueError(
                f"train_start ({train_start}) must be before train_end ({train_end})"
            )
        if test_start_idx > test_end_idx:
            raise ValueError(
                f"test_start ({test_start}) must be before or equal to test_end ({test_end})"
            )
        if test_start_idx <= train_end_idx:
            raise ValueError(
                f"test_start ({test_start}) must be after train_end ({train_end})"
            )

        # Create indices
        in_id = np.arange(train_start_idx, train_end_idx + 1)
        out_id = np.arange(test_start_idx, test_end_idx + 1)

    else:
        # Mode 1: Proportion-based (backward compatible)
        # Parse lag to row count
        if lag != 0:
            if date_column is None and not isinstance(lag, int):
                raise ValueError(
                    "date_column required when lag is specified as string or Timedelta"
                )
            lag_rows = parse_period(lag, data, date_column) if date_column else lag
        else:
            lag_rows = 0

        # Determine training size
        if prop is None:
            # Use default: all data minus assessment (1/4 of data)
            prop = 0.75

        if not 0 < prop < 1:
            raise ValueError(f"prop must be between 0 and 1, got {prop}")

        # Calculate split point
        train_size = int(n * prop)

        if train_size < 1:
            raise ValueError(f"Training size is {train_size}, need at least 1 row")

        if train_size + lag_rows >= n:
            raise ValueError(
                f"Training size ({train_size}) + lag ({lag_rows}) >= total rows ({n}). "
                f"Not enough data for test set."
            )

        # Create indices
        # Training: rows 0 to train_size-1
        # Gap: rows train_size to train_size+lag_rows-1 (excluded from both)
        # Testing: rows train_size+lag_rows to end
        in_id = np.arange(train_size)
        out_id = np.arange(train_size + lag_rows, n)

        if len(out_id) == 0:
            raise ValueError("No rows left for testing after applying lag")

    # Create split
    split = Split(
        data=data,
        in_id=in_id,
        out_id=out_id,
        id="Split1"
    )

    return RSplit(split)
