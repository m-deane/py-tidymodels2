"""
Step for normalizing numeric columns

Wraps sklearn StandardScaler and MinMaxScaler for centering and scaling.
"""

from dataclasses import dataclass
from typing import List, Union, Optional, Any, Callable
import pandas as pd
import numpy as np


@dataclass
class StepNormalize:
    """
    Normalize numeric columns using sklearn scalers.

    Centers and scales numeric features to have standard distribution
    (zscore) or to a fixed range (minmax).

    Attributes:
        columns: Columns to normalize. Can be:
            - None: defaults to all_numeric() selector
            - str: single column name
            - List[str]: list of column names
            - Callable: selector function (e.g., all_numeric(), starts_with('temp'))
        method: "zscore" (StandardScaler) or "minmax" (MinMaxScaler)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    method: str = "zscore"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepNormalize":
        """
        Fit scaler to training data.

        Note: Datetime columns are automatically excluded from normalization.
        Use step_date() or step_timeseries_signature() to extract time features instead.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepNormalize with fitted scaler
        """
        from py_recipes.selectors import resolve_selector, all_numeric

        # Use resolve_selector with all_numeric() as default
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        # Exclude datetime columns from normalization
        # Datetime columns should be processed by step_date() or similar instead
        datetime_cols = [c for c in data.columns
                        if pd.api.types.is_datetime64_any_dtype(data[c])]
        cols = [c for c in cols if c not in datetime_cols]

        # Fit scaler
        if self.method == "zscore":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        if len(cols) > 0:
            scaler.fit(data[cols])
        else:
            scaler = None

        return PreparedStepNormalize(
            columns=cols,
            scaler=scaler,
            method=self.method
        )


@dataclass
class PreparedStepNormalize:
    """
    Fitted normalization step.

    Attributes:
        columns: Columns to normalize
        scaler: Fitted sklearn scaler
        method: Normalization method used
    """

    columns: List[str]
    scaler: Optional[Any]
    method: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scaler to new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with normalized columns
        """
        result = data.copy()

        if self.scaler is not None and len(self.columns) > 0:
            # Check which columns exist in new data
            existing_cols = [col for col in self.columns if col in result.columns]

            if len(existing_cols) > 0:
                result[existing_cols] = self.scaler.transform(result[existing_cols])

        return result
