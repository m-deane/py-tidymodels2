"""
Utility functions for standardizing output DataFrame column ordering.

This module ensures consistent column ordering across all extract_outputs() implementations:
1. 'date' column (if present) is always FIRST
2. Group column (if present) is always SECOND
3. Core columns follow: actuals, fitted, forecast, residuals, split
4. Metadata columns last: model, model_group_name, group (if not group_col)
"""

from typing import List, Optional
import pandas as pd


def reorder_outputs_columns(
    df: pd.DataFrame,
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Reorder columns in outputs DataFrame to ensure consistent ordering.

    Ordering priority:
    1. 'date' (always first if present in columns or index)
    2. group_col (e.g., 'country', 'store_id') - second if present
    3. Core columns: actuals, fitted, forecast, residuals, split
    4. Metadata: model, model_group_name, group
    5. Any remaining columns (in original order)

    If date is in the index, it will be reset to be the first column.

    Args:
        df: Outputs DataFrame from extract_outputs()
        group_col: Name of grouping column (e.g., 'store_id', 'country')
                   If provided and exists in df, will be placed second

    Returns:
        DataFrame with reordered columns (date will be a column, not index)

    Examples:
        >>> # Without group column
        >>> outputs = reorder_outputs_columns(outputs)
        >>> # date, actuals, fitted, forecast, residuals, split, model, ...

        >>> # With group column
        >>> outputs = reorder_outputs_columns(outputs, group_col='store_id')
        >>> # date, store_id, actuals, fitted, forecast, residuals, split, model, ...
    """
    if df.empty:
        return df

    # Check if date is in the index (some engines set it there)
    has_date_index = (
        isinstance(df.index, pd.DatetimeIndex) or
        (df.index.name == 'date' and pd.api.types.is_datetime64_any_dtype(df.index))
    )

    # Reset index if date is there (to make it a column)
    if has_date_index:
        df = df.reset_index()
        # If index was unnamed DatetimeIndex, name it 'date'
        if 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
            df = df.rename(columns={'index': 'date'})

    # Build ordered column list
    ordered_cols = []

    # 1. Date column (always first if present)
    if 'date' in df.columns:
        ordered_cols.append('date')

    # 2. Group column (always second if present)
    if group_col and group_col in df.columns:
        ordered_cols.append(group_col)

    # 3. Core columns (in priority order)
    core_cols = ['actuals', 'fitted', 'forecast', 'residuals', 'split']
    for col in core_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 4. Metadata columns (in priority order)
    metadata_cols = ['model', 'model_group_name', 'group']
    for col in metadata_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 5. Any remaining columns (preserve original order)
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    # Reorder DataFrame
    return df[ordered_cols]


def reorder_coefficients_columns(
    df: pd.DataFrame,
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Reorder columns in coefficients DataFrame to ensure consistent ordering.

    Ordering priority:
    1. group_col (e.g., 'country', 'store_id') - first if present
    2. variable, coefficient, std_error
    3. t_stat, p_value
    4. conf_low, conf_high
    5. vif
    6. Metadata: model, model_group_name, group
    7. Any remaining columns

    Args:
        df: Coefficients DataFrame from extract_outputs()
        group_col: Name of grouping column (optional)

    Returns:
        DataFrame with reordered columns
    """
    if df.empty:
        return df

    ordered_cols = []

    # 1. Group column (first if present)
    if group_col and group_col in df.columns:
        ordered_cols.append(group_col)

    # 2. Core coefficient columns
    core_cols = ['variable', 'coefficient', 'std_error', 't_stat', 'p_value',
                 'conf_low', 'conf_high', 'vif']
    for col in core_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 3. Metadata columns
    metadata_cols = ['model', 'model_group_name', 'group']
    for col in metadata_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 4. Remaining columns
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    return df[ordered_cols]


def reorder_stats_columns(
    df: pd.DataFrame,
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Reorder columns in stats DataFrame to ensure consistent ordering.

    Ordering priority:
    1. group_col (e.g., 'country', 'store_id') - first if present
    2. split, metric, value
    3. Metadata: model, model_group_name, group
    4. Any remaining columns

    Args:
        df: Stats DataFrame from extract_outputs()
        group_col: Name of grouping column (optional)

    Returns:
        DataFrame with reordered columns
    """
    if df.empty:
        return df

    ordered_cols = []

    # 1. Group column (first if present)
    if group_col and group_col in df.columns:
        ordered_cols.append(group_col)

    # 2. Core stats columns
    core_cols = ['split', 'metric', 'value']
    for col in core_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 3. Metadata columns
    metadata_cols = ['model', 'model_group_name', 'group']
    for col in metadata_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # 4. Remaining columns
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    return df[ordered_cols]
