"""
Feature filtering preprocessing steps

Provides zero variance, near-zero variance, linear combination, and missing data filters.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class StepZv:
    """
    Remove zero variance columns.

    Removes columns that have the same value for all observations,
    as they provide no predictive information.

    Attributes:
        columns: Columns to check (None = all numeric)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepZv":
        """
        Identify zero variance columns.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepZv with columns to remove
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Find columns with zero variance
        cols_to_remove = []
        for col in cols:
            if data[col].nunique() <= 1:
                cols_to_remove.append(col)

        return PreparedStepZv(columns_to_remove=cols_to_remove)


@dataclass
class PreparedStepZv:
    """
    Fitted zero variance filter.

    Attributes:
        columns_to_remove: Columns identified as zero variance
    """

    columns_to_remove: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove zero variance columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with zero variance columns removed
        """
        result = data.copy()

        # Remove columns that exist in the data
        cols_to_drop = [col for col in self.columns_to_remove if col in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result


@dataclass
class StepNzv:
    """
    Remove near-zero variance columns.

    Removes columns with very low variance based on:
    - Frequency ratio: ratio of most common to second most common value
    - Unique value percentage: percentage of unique values

    Attributes:
        columns: Columns to check (None = all numeric)
        freq_cut: Frequency ratio threshold (default: 95/5)
        unique_cut: Unique value percentage threshold (default: 10%)
    """

    columns: Optional[List[str]] = None
    freq_cut: float = 19.0  # 95/5 ratio
    unique_cut: float = 10.0  # 10% unique values

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepNzv":
        """
        Identify near-zero variance columns.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepNzv with columns to remove
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        cols_to_remove = []
        n_rows = len(data)

        for col in cols:
            value_counts = data[col].value_counts()

            if len(value_counts) == 0:
                cols_to_remove.append(col)
                continue

            # Calculate frequency ratio
            if len(value_counts) > 1:
                freq_ratio = value_counts.iloc[0] / value_counts.iloc[1]
            else:
                freq_ratio = float('inf')

            # Calculate unique value percentage
            unique_pct = (len(value_counts) / n_rows) * 100

            # Remove if both conditions met
            if freq_ratio > self.freq_cut and unique_pct < self.unique_cut:
                cols_to_remove.append(col)

        return PreparedStepNzv(columns_to_remove=cols_to_remove)


@dataclass
class PreparedStepNzv:
    """
    Fitted near-zero variance filter.

    Attributes:
        columns_to_remove: Columns identified as near-zero variance
    """

    columns_to_remove: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove near-zero variance columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with near-zero variance columns removed
        """
        result = data.copy()

        cols_to_drop = [col for col in self.columns_to_remove if col in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result


@dataclass
class StepLinComb:
    """
    Remove columns that are linear combinations of others.

    Identifies and removes columns that can be perfectly predicted
    from other columns, eliminating multicollinearity.

    Attributes:
        columns: Columns to check (None = all numeric)
        threshold: Tolerance for linear dependency detection (default: 1e-5)
    """

    columns: Optional[List[str]] = None
    threshold: float = 1e-5

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepLinComb":
        """
        Identify linearly dependent columns.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepLinComb with columns to remove
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        if len(cols) == 0:
            return PreparedStepLinComb(columns_to_remove=[])

        # Get numeric data
        X = data[cols].values

        # Handle missing values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)

        cols_to_remove = []

        # Use QR decomposition to find linear dependencies
        try:
            _, R = np.linalg.qr(X)

            # Check diagonal of R matrix
            diag = np.abs(np.diag(R))

            # Columns with near-zero diagonal are linearly dependent
            for i, val in enumerate(diag):
                if val < self.threshold and i < len(cols):
                    cols_to_remove.append(cols[i])
        except np.linalg.LinAlgError:
            # If QR fails, try correlation-based approach
            corr_matrix = np.corrcoef(X.T)

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if abs(corr_matrix[i, j]) > (1.0 - self.threshold):
                        if cols[j] not in cols_to_remove:
                            cols_to_remove.append(cols[j])

        return PreparedStepLinComb(columns_to_remove=cols_to_remove)


@dataclass
class PreparedStepLinComb:
    """
    Fitted linear combination filter.

    Attributes:
        columns_to_remove: Columns identified as linear combinations
    """

    columns_to_remove: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linearly dependent columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with linearly dependent columns removed
        """
        result = data.copy()

        cols_to_drop = [col for col in self.columns_to_remove if col in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result


@dataclass
class StepFilterMissing:
    """
    Remove columns with high proportion of missing values.

    Filters out columns where missing data exceeds a threshold,
    as they may not be useful for modeling.

    Attributes:
        columns: Columns to check (None = all columns)
        threshold: Maximum proportion of missing values (default: 0.5)
    """

    columns: Optional[List[str]] = None
    threshold: float = 0.5

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepFilterMissing":
        """
        Identify columns with excessive missing values.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepFilterMissing with columns to remove
        """
        if self.columns is None:
            cols = data.columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        cols_to_remove = []
        n_rows = len(data)

        for col in cols:
            missing_pct = data[col].isna().sum() / n_rows

            if missing_pct > self.threshold:
                cols_to_remove.append(col)

        return PreparedStepFilterMissing(columns_to_remove=cols_to_remove)


@dataclass
class PreparedStepFilterMissing:
    """
    Fitted missing data filter.

    Attributes:
        columns_to_remove: Columns with excessive missing values
    """

    columns_to_remove: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with excessive missing values.

        Args:
            data: Data to transform

        Returns:
            DataFrame with high-missing columns removed
        """
        result = data.copy()

        cols_to_drop = [col for col in self.columns_to_remove if col in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result
