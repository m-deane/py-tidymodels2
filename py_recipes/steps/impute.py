"""
Steps for imputing missing values

Provides mean, median, mode, KNN, and linear interpolation imputation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class StepImputeMean:
    """
    Impute missing values using mean.

    Replaces NA values in numeric columns with the training mean.

    Attributes:
        columns: Columns to impute (None = all numeric with NA)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMean":
        """
        Calculate means for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMean with mean values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with NA
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cols = [col for col in numeric_cols if data[col].isna().any()]
        else:
            cols = self.columns

        # Calculate means
        means = {}
        for col in cols:
            if col in data.columns:
                means[col] = data[col].mean()

        return PreparedStepImputeMean(means=means)


@dataclass
class PreparedStepImputeMean:
    """
    Fitted mean imputation step.

    Attributes:
        means: Dictionary mapping column names to mean values
    """

    means: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with means.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, mean_val in self.means.items():
            if col in result.columns:
                result[col] = result[col].fillna(mean_val)

        return result


@dataclass
class StepImputeMedian:
    """
    Impute missing values using median.

    Replaces NA values in numeric columns with the training median.

    Attributes:
        columns: Columns to impute (None = all numeric with NA)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMedian":
        """
        Calculate medians for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMedian with median values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with NA
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cols = [col for col in numeric_cols if data[col].isna().any()]
        else:
            cols = self.columns

        # Calculate medians
        medians = {}
        for col in cols:
            if col in data.columns:
                medians[col] = data[col].median()

        return PreparedStepImputeMedian(medians=medians)


@dataclass
class PreparedStepImputeMedian:
    """
    Fitted median imputation step.

    Attributes:
        medians: Dictionary mapping column names to median values
    """

    medians: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with medians.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, median_val in self.medians.items():
            if col in result.columns:
                result[col] = result[col].fillna(median_val)

        return result


@dataclass
class StepImputeMode:
    """
    Impute missing values using mode (most frequent value).

    Replaces NA values with the most common value.
    Works for both numeric and categorical columns.

    Attributes:
        columns: Columns to impute (None = all columns with NA)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMode":
        """
        Calculate mode for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMode with mode values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select columns with NA
            cols = [col for col in data.columns if data[col].isna().any()]
        else:
            cols = self.columns

        # Calculate modes
        modes = {}
        for col in cols:
            if col in data.columns:
                mode_series = data[col].mode()
                if len(mode_series) > 0:
                    modes[col] = mode_series.iloc[0]
                else:
                    modes[col] = None

        return PreparedStepImputeMode(modes=modes)


@dataclass
class PreparedStepImputeMode:
    """
    Fitted mode imputation step.

    Attributes:
        modes: Dictionary mapping column names to mode values
    """

    modes: Dict[str, Any]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with modes.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, mode_val in self.modes.items():
            if col in result.columns and mode_val is not None:
                result[col] = result[col].fillna(mode_val)

        return result


@dataclass
class StepImputeKnn:
    """
    Impute missing values using K-Nearest Neighbors.

    Uses KNN algorithm to impute missing values based on similar observations.
    Useful when missing values have patterns related to other features.

    Attributes:
        columns: Columns to impute (None = all numeric with NA)
        neighbors: Number of neighbors to use (default: 5)
        weights: Weight function ('uniform' or 'distance')
    """

    columns: Optional[List[str]] = None
    neighbors: int = 5
    weights: str = "uniform"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeKnn":
        """
        Fit KNN imputer on training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeKnn with fitted imputer
        """
        from sklearn.impute import KNNImputer

        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with NA
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cols = [col for col in numeric_cols if data[col].isna().any()]
        else:
            cols = self.columns

        if len(cols) == 0:
            return PreparedStepImputeKnn(
                columns=[],
                imputer=None,
                feature_names=[]
            )

        # Get all numeric columns for context
        all_numeric = data.select_dtypes(include=[np.number]).columns.tolist()

        # Fit imputer on numeric data
        imputer = KNNImputer(
            n_neighbors=self.neighbors,
            weights=self.weights
        )
        imputer.fit(data[all_numeric])

        return PreparedStepImputeKnn(
            columns=cols,
            imputer=imputer,
            feature_names=all_numeric
        )


@dataclass
class PreparedStepImputeKnn:
    """
    Fitted KNN imputation step.

    Attributes:
        columns: Columns to impute
        imputer: Fitted KNNImputer
        feature_names: All numeric feature names used for imputation
    """

    columns: List[str]
    imputer: Any
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using KNN.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        if self.imputer is None or len(self.columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform numeric features
        numeric_data = result[self.feature_names]
        imputed_data = self.imputer.transform(numeric_data)

        # Update result with imputed values
        result[self.feature_names] = imputed_data

        return result


@dataclass
class StepImputeLinear:
    """
    Impute missing values using linear interpolation.

    Fills missing values by linear interpolation between adjacent values.
    Particularly useful for time series data.

    Attributes:
        columns: Columns to impute (None = all numeric with NA)
        limit: Maximum number of consecutive NAs to fill (None = no limit)
        limit_direction: Direction to fill ('forward', 'backward', or 'both')
    """

    columns: Optional[List[str]] = None
    limit: Optional[int] = None
    limit_direction: str = "both"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeLinear":
        """
        Prepare linear interpolation step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeLinear ready to interpolate
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with NA
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cols = [col for col in numeric_cols if data[col].isna().any()]
        else:
            cols = self.columns

        return PreparedStepImputeLinear(
            columns=cols,
            limit=self.limit,
            limit_direction=self.limit_direction
        )


@dataclass
class PreparedStepImputeLinear:
    """
    Fitted linear interpolation step.

    Attributes:
        columns: Columns to impute
        limit: Maximum consecutive NAs to fill
        limit_direction: Direction to fill
    """

    columns: List[str]
    limit: Optional[int]
    limit_direction: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using linear interpolation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                result[col] = result[col].interpolate(
                    method='linear',
                    limit=self.limit,
                    limit_direction=self.limit_direction
                )

        return result
