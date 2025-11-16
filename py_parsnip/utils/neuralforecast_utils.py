"""
NeuralForecast data formatting utilities.

This module provides helper functions for converting py-tidymodels data structures
to NeuralForecast's expected format and handling deep learning-specific operations.
"""

from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import warnings


def convert_to_neuralforecast_format(
    data: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    unique_id_name: str = 'unique_id',
    ds_name: str = 'ds',
    y_name: str = 'y'
) -> pd.DataFrame:
    """
    Convert DataFrame to NeuralForecast's expected format.

    NeuralForecast requires a specific DataFrame structure:
    - 'unique_id': Identifier for each time series (for panel data)
    - 'ds': Datetime column
    - 'y': Target variable
    - Exogenous columns (optional)

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with time series data.
    date_col : str
        Name of date column. Use '__index__' for DatetimeIndex.
    target_col : str
        Name of target variable column.
    exog_cols : list of str, optional
        Names of exogenous variable columns. Default is None (no exogenous).
    group_col : str, optional
        Name of group column for panel data. If None, creates single time series
        with unique_id='series_1'.
    unique_id_name : str, default='unique_id'
        Name for the unique identifier column in output.
    ds_name : str, default='ds'
        Name for the datetime column in output.
    y_name : str, default='y'
        Name for the target variable column in output.

    Returns
    -------
    pd.DataFrame
        DataFrame in NeuralForecast format with columns:
        - unique_id: Time series identifier
        - ds: Datetime
        - y: Target variable
        - [exog_cols]: Exogenous variables (if provided)

    Raises
    ------
    ValueError
        If target_col not in data columns.
        If date_col is not '__index__' and not in data columns.
        If any exog_cols not in data columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>>
    >>> # Simple univariate time series
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=10),
    ...     'sales': range(10)
    ... })
    >>> nf_df = convert_to_neuralforecast_format(df, 'date', 'sales')
    >>> print(nf_df.columns.tolist())
    ['unique_id', 'ds', 'y']
    >>>
    >>> # With exogenous variables
    >>> df['price'] = range(100, 110)
    >>> df['promo'] = [0, 1] * 5
    >>> nf_df = convert_to_neuralforecast_format(
    ...     df, 'date', 'sales', exog_cols=['price', 'promo']
    ... )
    >>> print(nf_df.columns.tolist())
    ['unique_id', 'ds', 'y', 'price', 'promo']
    >>>
    >>> # Panel data with groups
    >>> df['store'] = ['A'] * 5 + ['B'] * 5
    >>> nf_df = convert_to_neuralforecast_format(
    ...     df, 'date', 'sales', group_col='store'
    ... )
    >>> print(nf_df['unique_id'].unique())
    ['A' 'B']
    """
    # Validate target column exists
    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Get datetime values
    if date_col == '__index__':
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "date_col is '__index__' but data does not have DatetimeIndex. "
                f"Got index type: {type(data.index).__name__}"
            )
        date_values = data.index
    else:
        if date_col not in data.columns:
            raise ValueError(
                f"Date column '{date_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        date_values = data[date_col]

    # Build base DataFrame
    result = pd.DataFrame({
        ds_name: date_values,
        y_name: data[target_col].values
    })

    # Add unique_id for group identification
    if group_col is not None:
        if group_col not in data.columns:
            raise ValueError(
                f"Group column '{group_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        result[unique_id_name] = data[group_col].values
    else:
        # Single time series - assign default unique_id
        result[unique_id_name] = 'series_1'

    # Add exogenous variables if provided
    if exog_cols:
        missing_cols = [col for col in exog_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Exogenous columns {missing_cols} not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        for col in exog_cols:
            result[col] = data[col].values

    # Reorder columns: unique_id, ds, y, [exog_cols]
    col_order = [unique_id_name, ds_name, y_name]
    if exog_cols:
        col_order.extend(exog_cols)

    return result[col_order]


def parse_formula_for_dl(formula: str, date_col: str) -> Tuple[str, List[str]]:
    """
    Parse formula for deep learning models to extract target and exogenous variables.

    Similar to _parse_ts_formula but tailored for deep learning models.
    Automatically excludes date column from exogenous variables.

    Parameters
    ----------
    formula : str
        Formula string in Patsy format (e.g., "y ~ x1 + x2 + date").
    date_col : str
        Name of the date column to exclude from exogenous variables.
        Use '__index__' if using DatetimeIndex.

    Returns
    -------
    tuple of (str, list of str)
        - target: Name of the target variable
        - exog_vars: List of exogenous variable names (excluding date column)
          Returns empty list if formula is "y ~ 1" (no exogenous variables).

    Raises
    ------
    ValueError
        If formula is invalid (missing ~, empty sides, etc.).
        If formula contains multiple outcomes (DL models support single outcome only).

    Examples
    --------
    >>> # Standard formula with exogenous variables
    >>> parse_formula_for_dl("sales ~ price + promo + date", "date")
    ('sales', ['price', 'promo'])
    >>>
    >>> # Formula with only intercept (no exogenous)
    >>> parse_formula_for_dl("sales ~ 1", "date")
    ('sales', [])
    >>>
    >>> # Formula with all predictors (.)
    >>> parse_formula_for_dl("target ~ .", "date")
    ('target', ['.'])
    >>>
    >>> # DatetimeIndex case
    >>> parse_formula_for_dl("y ~ lag1 + lag2", "__index__")
    ('y', ['lag1', 'lag2'])
    >>>
    >>> # Invalid: Multiple outcomes not supported
    >>> parse_formula_for_dl("y1 + y2 ~ x1", "date")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Deep learning models support single outcome only...
    """
    # Validate formula contains ~
    if '~' not in formula:
        raise ValueError(
            f"Invalid formula: '{formula}'. Formula must contain '~' separator."
        )

    # Split formula into outcome and predictors
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid formula: '{formula}'. Formula must have exactly one '~' separator."
        )

    outcome_str = parts[0].strip()
    predictors_str = parts[1].strip()

    # Validate outcome is not empty
    if not outcome_str:
        raise ValueError(
            f"Invalid formula: '{formula}'. Outcome (left side) cannot be empty."
        )

    # Check for multiple outcomes (not supported by DL models)
    if '+' in outcome_str:
        raise ValueError(
            f"Invalid formula: '{formula}'. Deep learning models support single outcome only. "
            f"Found: '{outcome_str}'"
        )

    # Validate predictors is not empty
    if not predictors_str:
        raise ValueError(
            f"Invalid formula: '{formula}'. Predictors (right side) cannot be empty. "
            f"Use '~ 1' for univariate model (no exogenous variables)."
        )

    # Handle intercept-only case (univariate model)
    if predictors_str == '1':
        return (outcome_str, [])

    # Handle "all columns" case (.)
    if predictors_str == '.':
        return (outcome_str, ['.'])

    # Parse predictor terms
    predictor_terms = [term.strip() for term in predictors_str.split('+')]
    predictor_terms = [term for term in predictor_terms if term]  # Remove empty

    # Filter out date column (unless it's __index__)
    if date_col != '__index__':
        exog_vars = [term for term in predictor_terms if term != date_col]
    else:
        exog_vars = predictor_terms

    return (outcome_str, exog_vars)


def infer_frequency(datetime_index: pd.DatetimeIndex) -> str:
    """
    Infer frequency from DatetimeIndex for NeuralForecast models.

    NeuralForecast models require explicit frequency specification.
    This function infers the frequency and returns it in a format
    compatible with NeuralForecast's frequency argument.

    Parameters
    ----------
    datetime_index : pd.DatetimeIndex
        DatetimeIndex from which to infer frequency.

    Returns
    -------
    str
        Inferred frequency string (e.g., 'D', 'W', 'M', 'H', etc.).
        Common values:
        - 'D': Daily
        - 'W': Weekly
        - 'M': Monthly
        - 'H': Hourly
        - 'T': Minutely
        - 'S': Secondly

    Raises
    ------
    ValueError
        If frequency cannot be inferred from the index.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Daily frequency
    >>> idx = pd.date_range('2020-01-01', periods=10, freq='D')
    >>> infer_frequency(idx)
    'D'
    >>>
    >>> # Monthly frequency
    >>> idx = pd.date_range('2020-01-01', periods=12, freq='M')
    >>> infer_frequency(idx)
    'M'
    >>>
    >>> # Hourly frequency
    >>> idx = pd.date_range('2020-01-01', periods=24, freq='H')
    >>> infer_frequency(idx)
    'H'
    """
    if not isinstance(datetime_index, pd.DatetimeIndex):
        raise ValueError(
            f"Expected pd.DatetimeIndex, got {type(datetime_index).__name__}"
        )

    # If frequency already set, return it
    if datetime_index.freq is not None:
        return datetime_index.freqstr

    # Try to infer frequency
    freq = pd.infer_freq(datetime_index)

    if freq is not None:
        return freq

    # Fallback: Calculate most common difference
    if len(datetime_index) < 2:
        raise ValueError(
            f"Cannot infer frequency from DatetimeIndex with length {len(datetime_index)}. "
            f"Need at least 2 observations."
        )

    diffs = datetime_index[1:] - datetime_index[:-1]
    unique_diffs = diffs.unique()

    if len(unique_diffs) > 1:
        # Irregular frequency - use most common difference
        warnings.warn(
            f"Irregular time series detected. Found {len(unique_diffs)} different intervals. "
            f"Using most common interval for frequency inference.",
            UserWarning
        )

    most_common_diff = diffs.value_counts().idxmax()

    # Convert timedelta to frequency string
    total_seconds = most_common_diff.total_seconds()

    # Common frequencies (in seconds)
    freq_map = {
        1: 'S',           # Second
        60: 'T',          # Minute
        3600: 'H',        # Hour
        86400: 'D',       # Day
        604800: 'W',      # Week
    }

    if total_seconds in freq_map:
        return freq_map[total_seconds]

    # For irregular frequencies, return in days
    days = total_seconds / 86400
    if days.is_integer():
        return f'{int(days)}D'

    # For sub-day frequencies, return in hours
    hours = total_seconds / 3600
    if hours.is_integer():
        return f'{int(hours)}H'

    # For sub-hour frequencies, return in minutes
    minutes = total_seconds / 60
    if minutes.is_integer():
        return f'{int(minutes)}T'

    # Last resort: return in seconds
    return f'{int(total_seconds)}S'


def create_validation_split(
    data: pd.DataFrame,
    val_proportion: float = 0.2,
    method: str = 'time_based',
    date_col: Optional[str] = None,
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split for deep learning models.

    Splits data into training and validation sets. For time series data,
    uses chronological split by default to prevent data leakage.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to split.
    val_proportion : float, default=0.2
        Proportion of data to use for validation (0 < val_proportion < 1).
    method : str, default='time_based'
        Splitting method:
        - 'time_based': Chronological split (last val_proportion% for validation)
        - 'random': Random split (shuffled)
        - 'stratified': Stratified split (requires group_col)
    date_col : str, optional
        Name of date column for time-based splitting. Use '__index__' for
        DatetimeIndex. Required if method='time_based'.
    group_col : str, optional
        Name of group column for stratified splitting. Required if method='stratified'.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - train_df: Training data
        - val_df: Validation data

    Raises
    ------
    ValueError
        If val_proportion not in (0, 1).
        If method is invalid.
        If required columns are missing for specified method.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Time-based split
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100)
    ... })
    >>> train, val = create_validation_split(df, val_proportion=0.2, date_col='date')
    >>> len(train), len(val)
    (80, 20)
    >>>
    >>> # Random split
    >>> train, val = create_validation_split(df, val_proportion=0.2, method='random')
    >>> len(train) + len(val)
    100
    >>>
    >>> # Stratified split by group
    >>> df['group'] = ['A'] * 50 + ['B'] * 50
    >>> train, val = create_validation_split(
    ...     df, val_proportion=0.2, method='stratified', group_col='group'
    ... )
    """
    if not (0 < val_proportion < 1):
        raise ValueError(
            f"val_proportion must be between 0 and 1, got {val_proportion}"
        )

    valid_methods = ['time_based', 'random', 'stratified']
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: '{method}'. Must be one of {valid_methods}."
        )

    n_total = len(data)
    n_val = int(n_total * val_proportion)
    n_train = n_total - n_val

    if method == 'time_based':
        # Chronological split (preserve time order)
        if date_col is None:
            raise ValueError(
                "date_col is required for method='time_based'. "
                "Specify column name or use '__index__' for DatetimeIndex."
            )

        # Sort by date to ensure chronological order
        if date_col == '__index__':
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    "date_col is '__index__' but data does not have DatetimeIndex"
                )
            sorted_data = data.sort_index()
        else:
            if date_col not in data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            sorted_data = data.sort_values(date_col)

        # Split: first n_train for training, last n_val for validation
        train_df = sorted_data.iloc[:n_train].copy()
        val_df = sorted_data.iloc[n_train:].copy()

    elif method == 'random':
        # Random split (shuffle data)
        shuffled = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_df = shuffled.iloc[:n_train].copy()
        val_df = shuffled.iloc[n_train:].copy()

    elif method == 'stratified':
        # Stratified split (maintain group proportions)
        if group_col is None:
            raise ValueError(
                "group_col is required for method='stratified'. "
                "Specify the column containing group labels."
            )

        if group_col not in data.columns:
            raise ValueError(
                f"Group column '{group_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            data,
            test_size=val_proportion,
            stratify=data[group_col],
            random_state=42
        )

    else:
        # Should never reach here due to validation above
        raise ValueError(f"Unknown method: {method}")

    return train_df, val_df


def expand_dot_notation_for_dl(
    exog_vars: List[str],
    data: pd.DataFrame,
    target_col: str,
    date_col: str
) -> List[str]:
    """
    Expand patsy's "." notation to all columns except target and date.

    Similar to _expand_dot_notation but tailored for deep learning models.

    Parameters
    ----------
    exog_vars : list of str
        Exogenous variable names from parse_formula_for_dl. May contain ['.'].
    data : pd.DataFrame
        The data containing all columns.
    target_col : str
        Name of the target variable to exclude.
    date_col : str
        Name of the date column to exclude. Use '__index__' for DatetimeIndex.

    Returns
    -------
    list of str
        Expanded list of column names, or original list if no '.' found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=5),
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [10, 20, 30, 40, 50],
    ...     'y': [100, 200, 300, 400, 500]
    ... })
    >>> expand_dot_notation_for_dl(['.'], df, 'y', 'date')
    ['x1', 'x2']
    >>> expand_dot_notation_for_dl(['x1'], df, 'y', 'date')
    ['x1']
    """
    if exog_vars == ['.']:
        # Expand to all columns except target and date
        return [
            col for col in data.columns
            if col != target_col and col != date_col and col != '__index__'
        ]
    return exog_vars
