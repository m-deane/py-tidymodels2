"""
Utility functions for handling data vintages in backtesting.

Provides functions for selecting appropriate vintages, creating synthetic
vintage data, and validating vintage datasets.
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
import warnings
from py_rsample.period_parser import _parse_period_string


def validate_vintage_data(
    data: pd.DataFrame,
    as_of_col: str,
    date_col: str
) -> None:
    """
    Validate vintage dataset has required columns and correct structure.

    Args:
        data: DataFrame with vintage data
        as_of_col: Name of vintage date column (as_of_date)
        date_col: Name of observation date column (date)

    Raises:
        ValueError: If data structure is invalid

    Examples:
        >>> validate_vintage_data(df, "as_of_date", "date")
    """
    # Check required columns exist
    if as_of_col not in data.columns:
        raise ValueError(
            f"Vintage date column '{as_of_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    if date_col not in data.columns:
        raise ValueError(
            f"Observation date column '{date_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Check columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(data[as_of_col]):
        raise ValueError(
            f"Vintage date column '{as_of_col}' must be datetime type, "
            f"got {data[as_of_col].dtype}"
        )

    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        raise ValueError(
            f"Observation date column '{date_col}' must be datetime type, "
            f"got {data[date_col].dtype}"
        )

    # Check chronology: as_of_date >= date (can't have vintage before observation)
    invalid_rows = data[data[as_of_col] < data[date_col]]
    if len(invalid_rows) > 0:
        n_invalid = len(invalid_rows)
        example = invalid_rows.iloc[0]
        raise ValueError(
            f"Found {n_invalid} rows where as_of_date < date (chronology violation). "
            f"Cannot have vintage from before observation date.\n"
            f"Example: as_of_date={example[as_of_col]}, date={example[date_col]}"
        )

    # Check for at least one vintage per observation date
    n_observations = data[date_col].nunique()
    if n_observations == 0:
        raise ValueError("Data contains no unique observation dates")


def select_vintage(
    data: pd.DataFrame,
    as_of_col: str,
    date_col: str,
    vintage_date: pd.Timestamp,
    vintage_selection: str = "latest"
) -> pd.DataFrame:
    """
    Select appropriate vintage for given forecast date.

    Args:
        data: Full vintage dataset
        as_of_col: Vintage date column name
        date_col: Observation date column name
        vintage_date: Forecast date (point in time to simulate)
        vintage_selection: Strategy for selecting vintage
            - "latest": Most recent vintage available <= vintage_date
            - "exact": Exact as_of_date match (requires exact vintage)

    Returns:
        DataFrame with one row per unique observation date (selected vintage)

    Raises:
        ValueError: If vintage_selection is invalid or no data available

    Examples:
        >>> # Get latest vintage available on 2023-01-15
        >>> train_data = select_vintage(
        ...     data=vintage_df,
        ...     as_of_col="as_of_date",
        ...     date_col="date",
        ...     vintage_date=pd.Timestamp("2023-01-15"),
        ...     vintage_selection="latest"
        ... )
    """
    if vintage_selection == "latest":
        # For each observation date, get most recent vintage available <= vintage_date
        filtered = data[data[as_of_col] <= vintage_date].copy()

        if len(filtered) == 0:
            raise ValueError(
                f"No vintage data available for vintage_date={vintage_date}. "
                f"Earliest as_of_date in data: {data[as_of_col].min()}"
            )

        # Group by observation date, keep row with latest as_of_date
        idx = filtered.groupby(date_col)[as_of_col].idxmax()
        result = filtered.loc[idx].reset_index(drop=True)

        return result

    elif vintage_selection == "exact":
        # Use exact as_of_date match
        result = data[data[as_of_col] == vintage_date].copy()

        if len(result) == 0:
            raise ValueError(
                f"No vintage data with exact as_of_date={vintage_date}. "
                f"Available vintages: {sorted(data[as_of_col].unique())}"
            )

        return result.reset_index(drop=True)

    else:
        raise ValueError(
            f"Unknown vintage_selection: '{vintage_selection}'. "
            f"Supported: 'latest', 'exact'"
        )


def create_vintage_data(
    final_data: pd.DataFrame,
    date_col: str,
    n_revisions: int = 3,
    revision_std: float = 0.05,
    revision_lag: Union[str, pd.Timedelta] = "1 month",
    as_of_col: str = "as_of_date"
) -> pd.DataFrame:
    """
    Generate synthetic vintage data from final/revised data.

    Creates multiple vintages per observation by adding measurement noise
    that decreases over time (earlier vintages are noisier).

    Args:
        final_data: DataFrame with final/revised data
        date_col: Name of date column in final_data
        n_revisions: Number of vintages per observation (including final)
        revision_std: Standard deviation of revision noise as fraction of value
            (e.g., 0.05 = 5% noise)
        revision_lag: Time between revisions (e.g., "1 month")
        as_of_col: Name for vintage date column in output

    Returns:
        DataFrame with vintage data structure:
        - as_of_date: Vintage date column
        - date: Observation date column
        - All columns from final_data

    Examples:
        >>> # Create 3 vintages with 5% noise, 1 month apart
        >>> vintage_df = create_vintage_data(
        ...     final_data=df,
        ...     date_col="date",
        ...     n_revisions=3,
        ...     revision_std=0.05,
        ...     revision_lag="1 month"
        ... )
    """
    if n_revisions < 1:
        raise ValueError(f"n_revisions must be at least 1, got {n_revisions}")

    if revision_std < 0:
        raise ValueError(f"revision_std must be non-negative, got {revision_std}")

    if date_col not in final_data.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in final_data. "
            f"Available columns: {list(final_data.columns)}"
        )

    # Parse revision lag
    if isinstance(revision_lag, str):
        revision_lag_td = _parse_period_string(revision_lag)
    elif isinstance(revision_lag, pd.Timedelta):
        revision_lag_td = revision_lag
    else:
        raise ValueError(
            f"revision_lag must be str or Timedelta, got {type(revision_lag)}"
        )

    # Create vintages
    vintage_rows = []

    for _, row in final_data.iterrows():
        obs_date = row[date_col]

        # Create n_revisions vintages for this observation
        for revision_num in range(n_revisions):
            # Calculate as_of_date for this vintage
            # First vintage is right after observation date
            # Subsequent vintages are spaced by revision_lag
            as_of_date = obs_date + (revision_num * revision_lag_td)

            # Create noisy version of data (earlier vintages are noisier)
            vintage_row = row.copy()
            vintage_row[as_of_col] = as_of_date

            # Add measurement noise to numeric columns (not date columns)
            # Noise decreases with later revisions (final revision has no noise)
            if revision_num < n_revisions - 1:
                # Calculate noise factor (earlier vintages have more noise)
                noise_factor = (n_revisions - revision_num) / n_revisions

                for col in final_data.columns:
                    if col != date_col and pd.api.types.is_numeric_dtype(final_data[col]):
                        # Add proportional noise
                        value = row[col]
                        if pd.notna(value) and value != 0:
                            noise = np.random.normal(0, abs(value) * revision_std * noise_factor)
                            vintage_row[col] = value + noise

            vintage_rows.append(vintage_row)

    # Combine into DataFrame
    vintage_df = pd.DataFrame(vintage_rows)

    # Reorder columns: as_of_date, date, then rest
    other_cols = [col for col in vintage_df.columns if col not in [as_of_col, date_col]]
    vintage_df = vintage_df[[as_of_col, date_col] + other_cols]

    # Sort by date, then as_of_date
    vintage_df = vintage_df.sort_values([date_col, as_of_col]).reset_index(drop=True)

    return vintage_df
