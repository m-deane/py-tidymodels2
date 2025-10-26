"""
Period parsing utilities for time series resampling

Converts period strings like "1 year", "3 months", "14 days" to pandas Timedeltas
or row counts for use in time series cross-validation.
"""

import re
from typing import Union
import pandas as pd


def parse_period(
    period: Union[str, int, pd.Timedelta],
    data: pd.DataFrame,
    date_column: str
) -> int:
    """
    Parse a period specification to number of rows.

    Supports three formats:
    1. Integer: Direct row count (e.g., 100 means 100 rows)
    2. Timedelta: pandas Timedelta object
    3. String: Period string like "1 year", "3 months", "14 days"

    Args:
        period: Period specification (int, Timedelta, or string)
        data: DataFrame with time series data
        date_column: Name of date column in data

    Returns:
        Number of rows corresponding to the period

    Raises:
        ValueError: If period format is invalid or date column not found

    Examples:
        >>> parse_period(100, df, "date")  # 100 rows
        100

        >>> parse_period("1 year", df, "date")  # Rows spanning 1 year
        365

        >>> parse_period(pd.Timedelta(days=30), df, "date")  # 30 days worth
        30
    """
    # If already an integer, return directly
    if isinstance(period, int):
        if period < 0:
            raise ValueError(f"Period must be positive, got {period}")
        return period

    # Validate date column exists
    if date_column not in data.columns:
        raise ValueError(
            f"Date column '{date_column}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        raise ValueError(
            f"Column '{date_column}' must be datetime type, got {data[date_column].dtype}"
        )

    # Convert string to Timedelta if needed
    if isinstance(period, str):
        period_td = _parse_period_string(period)
    elif isinstance(period, pd.Timedelta):
        period_td = period
    else:
        raise ValueError(
            f"Period must be int, str, or Timedelta, got {type(period)}"
        )

    # Convert Timedelta to row count based on data
    return _timedelta_to_rows(period_td, data, date_column)


def _parse_period_string(period_str: str) -> pd.Timedelta:
    """
    Parse period string to Timedelta.

    Supported formats:
    - "N year(s)", "N month(s)", "N week(s)", "N day(s)"
    - "N hour(s)", "N minute(s)", "N second(s)"

    Args:
        period_str: Period string (e.g., "1 year", "3 months", "14 days")

    Returns:
        pandas Timedelta

    Raises:
        ValueError: If period string format is invalid

    Examples:
        >>> _parse_period_string("1 year")
        Timedelta('365 days 00:00:00')

        >>> _parse_period_string("3 months")
        Timedelta('90 days 00:00:00')

        >>> _parse_period_string("14 days")
        Timedelta('14 days 00:00:00')
    """
    # Clean and normalize the string
    period_str = period_str.strip().lower()

    # Parse pattern: <number> <unit>
    pattern = r'^(\d+)\s*(year|month|week|day|hour|minute|second)s?$'
    match = re.match(pattern, period_str)

    if not match:
        raise ValueError(
            f"Invalid period format: '{period_str}'. "
            f"Expected format: '<number> <unit>' (e.g., '1 year', '3 months', '14 days')"
        )

    value = int(match.group(1))
    unit = match.group(2)

    # Convert to Timedelta
    # Note: months and years are approximate (30 days and 365 days)
    if unit == "year":
        return pd.Timedelta(days=value * 365)
    elif unit == "month":
        return pd.Timedelta(days=value * 30)
    elif unit == "week":
        return pd.Timedelta(weeks=value)
    elif unit == "day":
        return pd.Timedelta(days=value)
    elif unit == "hour":
        return pd.Timedelta(hours=value)
    elif unit == "minute":
        return pd.Timedelta(minutes=value)
    elif unit == "second":
        return pd.Timedelta(seconds=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def _timedelta_to_rows(
    timedelta: pd.Timedelta,
    data: pd.DataFrame,
    date_column: str
) -> int:
    """
    Convert Timedelta to number of rows based on data frequency.

    Args:
        timedelta: pandas Timedelta
        data: DataFrame with time series data
        date_column: Name of date column

    Returns:
        Number of rows corresponding to the timedelta

    Raises:
        ValueError: If data has fewer than 2 rows
    """
    if len(data) < 2:
        raise ValueError("Data must have at least 2 rows to infer frequency")

    # Get date column as Series
    dates = data[date_column]

    # Calculate time span from start to end
    time_span = dates.iloc[-1] - dates.iloc[0]

    if time_span == pd.Timedelta(0):
        raise ValueError("All dates in data are identical")

    # Calculate approximate rows per timedelta
    # rows_per_unit = (n_rows - 1) / time_span
    # target_rows = timedelta * rows_per_unit
    n_rows = len(data)
    rows_per_timedelta = (n_rows - 1) / time_span.total_seconds()
    target_rows = int(timedelta.total_seconds() * rows_per_timedelta)

    # Ensure at least 1 row
    return max(1, target_rows)
