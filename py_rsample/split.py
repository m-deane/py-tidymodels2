"""
Core split classes for time series resampling

Provides Split and RSplit classes for managing train/test splits in cross-validation.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Split:
    """
    Immutable split containing train/test indices.

    A Split represents a single train/test division of data, storing:
    - The original data
    - Training indices (in_id)
    - Testing indices (out_id)
    - Split identifier

    Attributes:
        data: Original DataFrame
        in_id: Training indices (numpy array)
        out_id: Testing indices (numpy array)
        id: Split identifier string (e.g., "Slice001", "Fold01")

    Example:
        >>> split = Split(
        ...     data=df,
        ...     in_id=np.array([0, 1, 2, 3, 4]),
        ...     out_id=np.array([5, 6, 7]),
        ...     id="Slice001"
        ... )
    """

    data: pd.DataFrame
    in_id: np.ndarray  # Training indices
    out_id: np.ndarray  # Testing indices
    id: str  # Split identifier


class RSplit:
    """
    rsample split object for extracting training/testing data.

    RSplit wraps a Split to provide convenient methods for accessing
    training and testing DataFrames. This matches R's rsample interface.

    Attributes:
        split: Underlying Split object

    Methods:
        training(): Get training DataFrame
        testing(): Get testing DataFrame
        analysis(): Alias for training() (rsample compatibility)
        assessment(): Alias for testing() (rsample compatibility)

    Example:
        >>> rsplit = RSplit(split)
        >>> train_df = rsplit.training()
        >>> test_df = rsplit.testing()
    """

    def __init__(self, split: Split):
        """
        Initialize RSplit from a Split.

        Args:
            split: Split object containing data and indices
        """
        self._split = split

    @property
    def split(self) -> Split:
        """Get the underlying Split object"""
        return self._split

    def training(self) -> pd.DataFrame:
        """
        Get training (analysis) data.

        Returns:
            DataFrame with training observations
        """
        return self._split.data.iloc[self._split.in_id].reset_index(drop=True)

    def testing(self) -> pd.DataFrame:
        """
        Get testing (assessment) data.

        Returns:
            DataFrame with testing observations
        """
        return self._split.data.iloc[self._split.out_id].reset_index(drop=True)

    # rsample compatibility aliases
    def analysis(self) -> pd.DataFrame:
        """Alias for training() (rsample compatibility)"""
        return self.training()

    def assessment(self) -> pd.DataFrame:
        """Alias for testing() (rsample compatibility)"""
        return self.testing()

    def __repr__(self) -> str:
        """String representation"""
        n_train = len(self._split.in_id)
        n_test = len(self._split.out_id)
        return f"RSplit(id={self._split.id}, train={n_train}, test={n_test})"

    def __len__(self) -> int:
        """Total number of observations (train + test)"""
        return len(self._split.in_id) + len(self._split.out_id)


# Helper functions for R-like API
def training(split: RSplit) -> pd.DataFrame:
    """
    Extract training data from an RSplit object.

    Helper function that matches R's rsample API for extracting
    training data from a split object.

    Args:
        split: RSplit object

    Returns:
        DataFrame with training observations

    Example:
        >>> from py_rsample import initial_time_split, training
        >>> split = initial_time_split(data, prop=0.75)
        >>> train_df = training(split)
    """
    return split.training()


def testing(split: RSplit) -> pd.DataFrame:
    """
    Extract testing data from an RSplit object.

    Helper function that matches R's rsample API for extracting
    testing data from a split object.

    Args:
        split: RSplit object

    Returns:
        DataFrame with testing observations

    Example:
        >>> from py_rsample import initial_time_split, testing
        >>> split = initial_time_split(data, prop=0.75)
        >>> test_df = testing(split)
    """
    return split.testing()
