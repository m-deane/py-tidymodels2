"""
Time series utility functions for date column handling and formula parsing.

This module provides helper functions for time series models to handle date columns
and parse time series formulas consistently across different engines.
"""

from typing import Optional, List, Tuple
import pandas as pd
import re


def _infer_date_column(
    data: pd.DataFrame,
    spec_date_col: Optional[str] = None,
    fit_date_col: Optional[str] = None
) -> str:
    """
    Infer the date column from a DataFrame using priority-based detection.

    Priority order:
    1. fit_date_col - Date column from fitted model (used during prediction)
    2. spec_date_col - Date column from model specification
    3. DatetimeIndex - If data has a DatetimeIndex
    4. Auto-detect - Find single datetime column in DataFrame

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing time series data.
    spec_date_col : str, optional
        Date column specified in model specification (e.g., via date_col parameter).
    fit_date_col : str, optional
        Date column stored from fitted model (used during prediction to ensure consistency).

    Returns
    -------
    str
        Name of the date column. Returns '__index__' if using DatetimeIndex.

    Raises
    ------
    ValueError
        If no datetime column is found or multiple datetime columns exist without
        explicit specification.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>>
    >>> # Case 1: Explicit date column in spec
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=10),
    ...     'value': range(10)
    ... })
    >>> _infer_date_column(df, spec_date_col='date')
    'date'
    >>>
    >>> # Case 2: DatetimeIndex
    >>> df_indexed = df.set_index('date')
    >>> _infer_date_column(df_indexed)
    '__index__'
    >>>
    >>> # Case 3: Auto-detect single datetime column
    >>> _infer_date_column(df)
    'date'
    >>>
    >>> # Case 4: fit_date_col takes priority
    >>> _infer_date_column(df, spec_date_col='value', fit_date_col='date')
    'date'
    >>>
    >>> # Case 5: Multiple datetime columns - raises error
    >>> df_multi = pd.DataFrame({
    ...     'date1': pd.date_range('2020-01-01', periods=10),
    ...     'date2': pd.date_range('2021-01-01', periods=10),
    ...     'value': range(10)
    ... })
    >>> _infer_date_column(df_multi)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Multiple datetime columns found: ['date1', 'date2'].
                 Please specify date_col explicitly.
    """
    # Priority 1: Use fit_date_col if provided (prediction phase)
    if fit_date_col is not None:
        if fit_date_col == '__index__':
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    f"fit_date_col is '__index__' but data does not have a DatetimeIndex. "
                    f"Got index type: {type(data.index).__name__}"
                )
            return '__index__'
        elif fit_date_col not in data.columns:
            raise ValueError(
                f"fit_date_col '{fit_date_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        return fit_date_col

    # Priority 2: Use spec_date_col if provided
    if spec_date_col is not None:
        if spec_date_col not in data.columns:
            raise ValueError(
                f"spec_date_col '{spec_date_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        if not pd.api.types.is_datetime64_any_dtype(data[spec_date_col]):
            raise ValueError(
                f"Column '{spec_date_col}' is not a datetime type. "
                f"Got dtype: {data[spec_date_col].dtype}"
            )
        return spec_date_col

    # Priority 3: Check for DatetimeIndex
    if isinstance(data.index, pd.DatetimeIndex):
        return '__index__'

    # Priority 4: Auto-detect datetime columns
    datetime_cols = [
        col for col in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[col])
    ]

    if len(datetime_cols) == 0:
        raise ValueError(
            f"No datetime column found in data. Please specify date_col explicitly. "
            f"Available columns: {list(data.columns)}"
        )
    elif len(datetime_cols) > 1:
        raise ValueError(
            f"Multiple datetime columns found: {datetime_cols}. "
            f"Please specify date_col explicitly."
        )

    return datetime_cols[0]


def _parse_ts_formula(formula: str, date_col: str) -> Tuple[str, List[str]]:
    """
    Parse time series formula to extract outcome and exogenous variables.

    Parses a formula string (e.g., "target ~ lag1 + lag2 + date") to extract:
    - Outcome variable (left side of ~)
    - Exogenous variables (right side of ~), EXCLUDING the date column

    The date column is automatically excluded from exogenous variables since it
    serves as the time index, not as a predictor.

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
        - outcome: Name of the outcome variable
        - exog_vars: List of exogenous variable names (excluding date column)
          Returns empty list if formula is "y ~ 1" (no exogenous variables)
          or if only the date column was on right side.

    Raises
    ------
    ValueError
        If formula is invalid (missing ~, empty sides, etc.).

    Examples
    --------
    >>> # Standard formula with exogenous variables
    >>> _parse_ts_formula("sales ~ lag1 + lag2 + date", "date")
    ('sales', ['lag1', 'lag2'])
    >>>
    >>> # Formula with only intercept (no exogenous)
    >>> _parse_ts_formula("sales ~ 1", "date")
    ('sales', [])
    >>>
    >>> # Formula with date only (becomes no exogenous)
    >>> _parse_ts_formula("sales ~ date", "date")
    ('sales', [])
    >>>
    >>> # Formula with all predictors (.)
    >>> _parse_ts_formula("target ~ .", "date")
    ('target', ['.'])
    >>>
    >>> # DatetimeIndex case
    >>> _parse_ts_formula("y ~ lag1 + lag2", "__index__")
    ('y', ['lag1', 'lag2'])
    >>>
    >>> # Multiple outcomes (e.g., for VARMAX)
    >>> _parse_ts_formula("y1 + y2 ~ x1 + x2 + date", "date")
    ('y1 + y2', ['x1', 'x2'])
    >>>
    >>> # Invalid formula - missing tilde
    >>> _parse_ts_formula("sales", "date")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Invalid formula: 'sales'. Formula must contain '~' separator.
    >>>
    >>> # Invalid formula - empty outcome
    >>> _parse_ts_formula("~ x1 + x2", "date")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Invalid formula: '~ x1 + x2'. Outcome (left side) cannot be empty.
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

    # Validate predictors is not empty
    if not predictors_str:
        raise ValueError(
            f"Invalid formula: '{formula}'. Predictors (right side) cannot be empty. "
            f"Use '~ 1' for intercept-only model."
        )

    # Handle intercept-only case
    if predictors_str == '1':
        return (outcome_str, [])

    # Handle "all columns" case (.)
    if predictors_str == '.':
        return (outcome_str, ['.'])

    # Parse predictor terms
    # Split by + and clean whitespace
    predictor_terms = [term.strip() for term in predictors_str.split('+')]

    # Remove empty terms
    predictor_terms = [term for term in predictor_terms if term]

    # Filter out date column (unless it's __index__ which isn't in columns)
    if date_col != '__index__':
        exog_vars = [term for term in predictor_terms if term != date_col]
    else:
        exog_vars = predictor_terms

    # Handle case where only date column was specified
    if len(exog_vars) == 0:
        return (outcome_str, [])

    return (outcome_str, exog_vars)


def _expand_dot_notation(exog_vars: List[str], data: pd.DataFrame, outcome_name: str, date_col: str) -> List[str]:
    """
    Expand patsy's "." notation to all columns except outcome and date.

    Parameters
    ----------
    exog_vars : list of str
        Exogenous variable names from _parse_ts_formula. May contain ['.'].
    data : pd.DataFrame
        The data containing all columns.
    outcome_name : str
        Name of the outcome variable to exclude.
    date_col : str
        Name of the date column to exclude. Use '__index__' for DatetimeIndex.

    Returns
    -------
    list of str
        Expanded list of column names, or original list if no '.' found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'date': ['2020-01-01'], 'x1': [1], 'x2': [2], 'y': [3]})
    >>> _expand_dot_notation(['.'], df, 'y', 'date')
    ['x1', 'x2']
    >>> _expand_dot_notation(['x1'], df, 'y', 'date')
    ['x1']
    """
    if exog_vars == ['.']:
        # Expand to all columns except outcome and date
        return [col for col in data.columns
                if col != outcome_name and col != date_col and col != '__index__']
    return exog_vars


def _validate_frequency(
    series: pd.Series,
    require_freq: bool = True,
    infer_freq: bool = True
) -> pd.Series:
    """
    Validate and optionally infer frequency for a time series with DatetimeIndex.

    Some time series models (e.g., skforecast, statsmodels) require explicit
    frequency on the DatetimeIndex. This function validates frequency and
    optionally infers it if missing.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    require_freq : bool, default=True
        If True, raise error if frequency cannot be determined.
        If False, return series as-is if frequency cannot be inferred.
    infer_freq : bool, default=True
        If True, attempt to infer frequency if not already set.
        If False, only validate existing frequency.

    Returns
    -------
    pd.Series
        Time series with validated/inferred frequency.

    Raises
    ------
    ValueError
        If index is not DatetimeIndex, or if frequency is required but cannot
        be determined.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Series with explicit frequency
    >>> dates = pd.date_range('2020-01-01', periods=10, freq='D')
    >>> s = pd.Series(range(10), index=dates)
    >>> validated = _validate_frequency(s)
    >>> validated.index.freq
    <Day>
    >>>
    >>> # Series without frequency - will infer
    >>> dates_no_freq = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10))
    >>> dates_no_freq.freq = None
    >>> s_no_freq = pd.Series(range(10), index=dates_no_freq)
    >>> validated = _validate_frequency(s_no_freq)
    >>> validated.index.freq is not None
    True
    >>>
    >>> # Series with irregular frequency - raises error
    >>> irregular_dates = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-08'])
    >>> s_irregular = pd.Series([1, 2, 3], index=irregular_dates)
    >>> _validate_frequency(s_irregular)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Could not infer frequency for DatetimeIndex...
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            f"Series must have DatetimeIndex. Got index type: {type(series.index).__name__}"
        )

    # If frequency already set, return as-is
    if series.index.freq is not None:
        return series

    # Try to infer frequency if requested
    if infer_freq:
        freq = pd.infer_freq(series.index)

        if freq is not None:
            # Create new series with explicit frequency
            new_index = pd.DatetimeIndex(series.index, freq=freq)
            return pd.Series(series.values, index=new_index, name=series.name)
        else:
            # Fallback: use most common difference (only if all differences are equal)
            if len(series.index) >= 2:
                diffs = series.index[1:] - series.index[:-1]
                unique_diffs = diffs.unique()

                # Only use asfreq if all differences are the same
                if len(unique_diffs) == 1:
                    most_common_diff = unique_diffs[0]
                    # Try to create regular frequency with asfreq
                    try:
                        series_with_freq = series.asfreq(most_common_diff)
                        return series_with_freq
                    except Exception:
                        pass  # Fall through to error handling

    # If we get here and frequency is required, raise error
    if require_freq:
        raise ValueError(
            f"Could not infer frequency for DatetimeIndex. "
            f"Please ensure the index has regular intervals or specify frequency explicitly. "
            f"Index range: {series.index.min()} to {series.index.max()}, "
            f"Length: {len(series.index)}"
        )

    # Otherwise return series as-is
    return series
