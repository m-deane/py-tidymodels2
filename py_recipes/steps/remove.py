"""
Column removal step for recipes.

Remove/drop specified columns from the dataset.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional, Callable
import pandas as pd


@dataclass
class StepRm:
    """
    Remove columns from the dataset.

    This step simply drops specified columns from the data. Useful for
    removing columns that shouldn't be used in modeling (like IDs, dates,
    or other non-predictive features).

    Parameters
    ----------
    columns : str, list of str, or callable
        Column(s) to remove. Can be:
        - Single column name (str)
        - List of column names
        - Selector function (e.g., all_predictors())
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>>
    >>> # Remove single column
    >>> rec = recipe().step_rm("id")
    >>>
    >>> # Remove multiple columns
    >>> rec = recipe().step_rm(["id", "date", "timestamp"])
    >>>
    >>> # Remove using selector
    >>> rec = recipe().step_rm(all_datetime())
    """
    columns: Union[str, List[str], Callable]
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _columns_to_remove: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by resolving which columns to remove."""
        if self.skip or not training:
            return self

        # Resolve columns
        if isinstance(self.columns, str):
            self._columns_to_remove = [self.columns]
        elif isinstance(self.columns, list):
            self._columns_to_remove = self.columns
        elif callable(self.columns):
            # Selector function
            self._columns_to_remove = self.columns(data)
        else:
            raise TypeError(f"columns must be str, list, or callable, got {type(self.columns)}")

        # Validate columns exist
        missing = set(self._columns_to_remove) - set(data.columns)
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        self._is_prepared = True
        return self

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove the specified columns."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        # Drop columns that exist in the data
        cols_to_drop = [col for col in self._columns_to_remove if col in data.columns]

        if not cols_to_drop:
            return data

        return data.drop(columns=cols_to_drop)


@dataclass
class StepSelect:
    """
    Select (keep) only specified columns from the dataset.

    This is the inverse of step_rm() - it keeps only the specified columns
    and removes everything else.

    Parameters
    ----------
    columns : str, list of str, or callable
        Column(s) to keep. Can be:
        - Single column name (str)
        - List of column names
        - Selector function (e.g., all_numeric())
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>>
    >>> # Keep only specific columns
    >>> rec = recipe().step_select(["feature1", "feature2", "target"])
    >>>
    >>> # Keep all numeric columns
    >>> rec = recipe().step_select(all_numeric())
    """
    columns: Union[str, List[str], Callable]
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _columns_to_keep: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by resolving which columns to keep."""
        if self.skip or not training:
            return self

        # Resolve columns
        if isinstance(self.columns, str):
            self._columns_to_keep = [self.columns]
        elif isinstance(self.columns, list):
            self._columns_to_keep = self.columns
        elif callable(self.columns):
            # Selector function
            self._columns_to_keep = self.columns(data)
        else:
            raise TypeError(f"columns must be str, list, or callable, got {type(self.columns)}")

        # Validate columns exist
        missing = set(self._columns_to_keep) - set(data.columns)
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        self._is_prepared = True
        return self

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Keep only the specified columns."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        # Keep only columns that exist in both the prepared list and current data
        cols_to_keep = [col for col in self._columns_to_keep if col in data.columns]

        if not cols_to_keep:
            raise ValueError("No columns to keep after filtering")

        return data[cols_to_keep]
