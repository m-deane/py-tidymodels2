"""
Step for removing rows with missing values

Removes rows that contain NA/NaN values in specified columns.
Useful after creating lag features or other transformations that introduce NAs.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class StepNaOmit:
    """
    Remove rows with missing values.

    Filters out rows containing NA/NaN values in specified columns or all columns.
    Commonly used after lag features which introduce NaN at the beginning of time series.

    Attributes:
        columns: Columns to check for NAs (None = check all columns)

    Examples:
        >>> # Remove rows with any NA values
        >>> step = StepNaOmit()
        >>>
        >>> # Remove rows with NA in specific columns
        >>> step = StepNaOmit(columns=['value_lag_1', 'value_lag_7'])
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepNaOmit":
        """
        Store column information (no fitting needed).

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepNaOmit with column names
        """
        if self.columns is None:
            cols = data.columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        return PreparedStepNaOmit(columns=cols)


@dataclass
class PreparedStepNaOmit:
    """
    Fitted naomit step.

    Attributes:
        columns: Columns to check for missing values
    """

    columns: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values.

        Args:
            data: Data to transform

        Returns:
            DataFrame with rows containing NAs removed
        """
        result = data.copy()

        # Get columns that exist in the current data
        cols_to_check = [col for col in self.columns if col in result.columns]

        if cols_to_check:
            # Remove rows with any NA in the specified columns
            result = result.dropna(subset=cols_to_check)

        return result
