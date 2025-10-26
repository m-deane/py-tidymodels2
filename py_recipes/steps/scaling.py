"""
Scaling and centering preprocessing steps

Provides center, scale, and range transformations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class StepCenter:
    """
    Center numeric columns to have mean zero.

    Attributes:
        columns: Columns to center (None = all numeric)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepCenter":
        """
        Calculate means from training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepCenter with fitted means
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Calculate means
        means = {col: data[col].mean() for col in cols}

        return PreparedStepCenter(columns=cols, means=means)


@dataclass
class PreparedStepCenter:
    """
    Fitted centering step.

    Attributes:
        columns: Columns to center
        means: Fitted mean values
    """

    columns: List[str]
    means: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply centering transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with centered columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                result[col] = result[col] - self.means[col]

        return result


@dataclass
class StepScale:
    """
    Scale numeric columns to have standard deviation of one.

    Attributes:
        columns: Columns to scale (None = all numeric)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepScale":
        """
        Calculate standard deviations from training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepScale with fitted standard deviations
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Calculate standard deviations
        stds = {col: data[col].std() for col in cols}

        return PreparedStepScale(columns=cols, stds=stds)


@dataclass
class PreparedStepScale:
    """
    Fitted scaling step.

    Attributes:
        columns: Columns to scale
        stds: Fitted standard deviation values
    """

    columns: List[str]
    stds: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with scaled columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns and self.stds[col] > 0:
                result[col] = result[col] / self.stds[col]

        return result


@dataclass
class StepRange:
    """
    Scale numeric columns to a custom range.

    Attributes:
        columns: Columns to scale (None = all numeric)
        min_val: Minimum value of scaled range (default: 0)
        max_val: Maximum value of scaled range (default: 1)
    """

    columns: Optional[List[str]] = None
    min_val: float = 0.0
    max_val: float = 1.0

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepRange":
        """
        Calculate min/max values from training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepRange with fitted ranges
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Calculate min/max for each column
        mins = {col: data[col].min() for col in cols}
        maxs = {col: data[col].max() for col in cols}

        return PreparedStepRange(
            columns=cols,
            mins=mins,
            maxs=maxs,
            min_val=self.min_val,
            max_val=self.max_val
        )


@dataclass
class PreparedStepRange:
    """
    Fitted range scaling step.

    Attributes:
        columns: Columns to scale
        mins: Fitted minimum values
        maxs: Fitted maximum values
        min_val: Target minimum
        max_val: Target maximum
    """

    columns: List[str]
    mins: Dict[str, float]
    maxs: Dict[str, float]
    min_val: float
    max_val: float

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply range scaling transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with range-scaled columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                col_min = self.mins[col]
                col_max = self.maxs[col]

                if col_max > col_min:
                    # Scale to [0, 1]
                    scaled = (result[col] - col_min) / (col_max - col_min)
                    # Scale to [min_val, max_val]
                    result[col] = scaled * (self.max_val - self.min_val) + self.min_val

        return result
