"""
Column selector functions for recipes

Provides tidyselect-style functions for selecting columns by type, role, or pattern.
Inspired by tidymodels recipes::selections.
"""

from typing import List, Callable, Optional, Union
import pandas as pd
import numpy as np
import re


# Type Selectors

def all_numeric() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all numeric columns.

    Returns a selector function that identifies numeric columns.

    Returns:
        Function that takes a DataFrame and returns list of numeric column names

    Examples:
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> selector = all_numeric()
        >>> selector(data)
        ['x']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return selector


def all_nominal() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all categorical/nominal columns.

    Returns a selector function that identifies non-numeric columns.

    Returns:
        Function that takes a DataFrame and returns list of nominal column names

    Examples:
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> selector = all_nominal()
        >>> selector(data)
        ['y']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(exclude=[np.number]).columns.tolist()
    return selector


def all_integer() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all integer columns.

    Returns a selector function that identifies integer columns.

    Returns:
        Function that takes a DataFrame and returns list of integer column names
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(include=['int64', 'int32', 'int16', 'int8']).columns.tolist()
    return selector


def all_float() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all float columns.

    Returns a selector function that identifies float columns.

    Returns:
        Function that takes a DataFrame and returns list of float column names
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(include=['float64', 'float32']).columns.tolist()
    return selector


def all_string() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all string columns.

    Returns a selector function that identifies string/object columns.

    Returns:
        Function that takes a DataFrame and returns list of string column names
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(include=['object', 'string']).columns.tolist()
    return selector


def all_datetime() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all datetime columns.

    Returns a selector function that identifies datetime columns.

    Returns:
        Function that takes a DataFrame and returns list of datetime column names
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.select_dtypes(include=['datetime64']).columns.tolist()
    return selector


# Pattern Selectors

def starts_with(prefix: str, ignore_case: bool = True) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns whose names start with a prefix.

    Args:
        prefix: String prefix to match
        ignore_case: Whether to ignore case (default: True)

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'temp_1': [1], 'temp_2': [2], 'other': [3]})
        >>> selector = starts_with('temp')
        >>> selector(data)
        ['temp_1', 'temp_2']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        if ignore_case:
            prefix_lower = prefix.lower()
            return [col for col in data.columns if col.lower().startswith(prefix_lower)]
        return [col for col in data.columns if col.startswith(prefix)]
    return selector


def ends_with(suffix: str, ignore_case: bool = True) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns whose names end with a suffix.

    Args:
        suffix: String suffix to match
        ignore_case: Whether to ignore case (default: True)

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'x_1': [1], 'y_1': [2], 'z_2': [3]})
        >>> selector = ends_with('_1')
        >>> selector(data)
        ['x_1', 'y_1']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        if ignore_case:
            suffix_lower = suffix.lower()
            return [col for col in data.columns if col.lower().endswith(suffix_lower)]
        return [col for col in data.columns if col.endswith(suffix)]
    return selector


def contains(substring: str, ignore_case: bool = True) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns whose names contain a substring.

    Args:
        substring: String to search for
        ignore_case: Whether to ignore case (default: True)

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'x_temp_1': [1], 'y_temp_2': [2], 'z_other': [3]})
        >>> selector = contains('temp')
        >>> selector(data)
        ['x_temp_1', 'y_temp_2']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        if ignore_case:
            substring_lower = substring.lower()
            return [col for col in data.columns if substring_lower in col.lower()]
        return [col for col in data.columns if substring in col]
    return selector


def matches(pattern: str, ignore_case: bool = True) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns whose names match a regular expression.

    Args:
        pattern: Regular expression pattern
        ignore_case: Whether to ignore case (default: True)

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'x1': [1], 'x2': [2], 'y1': [3]})
        >>> selector = matches(r'^x\\d+$')
        >>> selector(data)
        ['x1', 'x2']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        return [col for col in data.columns if regex.search(col)]
    return selector


# Utility Selectors

def everything() -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all columns.

    Returns:
        Function that takes a DataFrame and returns all column names

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        >>> selector = everything()
        >>> selector(data)
        ['x', 'y', 'z']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return data.columns.tolist()
    return selector


def one_of(*columns: str) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns from a list of names.

    Args:
        *columns: Column names to select

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        >>> selector = one_of('x', 'z')
        >>> selector(data)
        ['x', 'z']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return [col for col in columns if col in data.columns]
    return selector


def none_of(*columns: str) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select all columns except those specified.

    Args:
        *columns: Column names to exclude

    Returns:
        Function that takes a DataFrame and returns non-matching column names

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        >>> selector = none_of('y')
        >>> selector(data)
        ['x', 'z']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        excluded = set(columns)
        return [col for col in data.columns if col not in excluded]
    return selector


def where(predicate: Callable[[pd.Series], bool]) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns where predicate is true.

    Args:
        predicate: Function that takes a Series and returns bool

    Returns:
        Function that takes a DataFrame and returns matching column names

    Examples:
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        >>> selector = where(lambda s: s.max() > 20)
        >>> selector(data)
        ['y']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        return [col for col in data.columns if predicate(data[col])]
    return selector


# Combination Selectors

def union(*selectors: Callable[[pd.DataFrame], List[str]]) -> Callable[[pd.DataFrame], List[str]]:
    """
    Combine multiple selectors with union (OR).

    Args:
        *selectors: Selector functions to combine

    Returns:
        Function that returns union of all selector results

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': ['a']})
        >>> selector = union(starts_with('x'), all_nominal())
        >>> selector(data)
        ['x', 'y']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        result = set()
        for sel in selectors:
            result.update(sel(data))
        # Preserve original column order
        return [col for col in data.columns if col in result]
    return selector


def intersection(*selectors: Callable[[pd.DataFrame], List[str]]) -> Callable[[pd.DataFrame], List[str]]:
    """
    Combine multiple selectors with intersection (AND).

    Args:
        *selectors: Selector functions to combine

    Returns:
        Function that returns intersection of all selector results

    Examples:
        >>> data = pd.DataFrame({'x_num': [1], 'x_str': ['a'], 'y_num': [2]})
        >>> selector = intersection(starts_with('x'), all_numeric())
        >>> selector(data)
        ['x_num']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        if not selectors:
            return []

        result = set(selectors[0](data))
        for sel in selectors[1:]:
            result.intersection_update(sel(data))

        # Preserve original column order
        return [col for col in data.columns if col in result]
    return selector


def difference(
    include: Callable[[pd.DataFrame], List[str]],
    exclude: Callable[[pd.DataFrame], List[str]]
) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns from include selector but not in exclude selector.

    Args:
        include: Selector for columns to include
        exclude: Selector for columns to exclude

    Returns:
        Function that returns difference of selectors

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        >>> selector = difference(all_numeric(), one_of('y'))
        >>> selector(data)
        ['x', 'z']
    """
    def selector(data: pd.DataFrame) -> List[str]:
        include_cols = set(include(data))
        exclude_cols = set(exclude(data))
        result = include_cols - exclude_cols

        # Preserve original column order
        return [col for col in data.columns if col in result]
    return selector


# Role Selector

def has_role(role: str) -> Callable[[pd.DataFrame], List[str]]:
    """
    Select columns with a specific role.

    Note: This requires the selector to be used within a Recipe context
    where roles are defined.

    Args:
        role: Role name to filter by

    Returns:
        Function that takes a DataFrame and returns columns with role

    Examples:
        >>> # Within a recipe context
        >>> rec = recipe().update_role(['x1', 'x2'], 'predictor')
        >>> rec = rec.step_normalize(has_role('predictor'))
    """
    def selector(data: pd.DataFrame) -> List[str]:
        # This is a placeholder - actual role resolution happens in recipe context
        # For now, return all columns
        return data.columns.tolist()

    # Attach role metadata for recipe to use
    selector.role = role  # type: ignore
    return selector


# Helper function to resolve selectors

def resolve_selector(
    selector: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]],
    data: pd.DataFrame
) -> List[str]:
    """
    Resolve a selector specification to a list of column names.

    Args:
        selector: Column specification - can be:
            - None: select all columns
            - str: single column name
            - List[str]: list of column names
            - Callable: selector function
        data: DataFrame to apply selector to

    Returns:
        List of column names

    Examples:
        >>> data = pd.DataFrame({'x': [1], 'y': [2]})
        >>> resolve_selector(None, data)
        ['x', 'y']
        >>> resolve_selector('x', data)
        ['x']
        >>> resolve_selector(['x', 'y'], data)
        ['x', 'y']
        >>> resolve_selector(all_numeric(), data)
        ['x', 'y']
    """
    if selector is None:
        return data.columns.tolist()
    elif isinstance(selector, str):
        return [selector] if selector in data.columns else []
    elif isinstance(selector, list):
        return [col for col in selector if col in data.columns]
    elif callable(selector):
        return selector(data)
    else:
        raise ValueError(f"Invalid selector type: {type(selector)}")
