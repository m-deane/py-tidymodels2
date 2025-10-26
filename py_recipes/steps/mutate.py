"""
Step for creating or modifying columns using custom functions

Allows arbitrary transformations to be included in recipe pipelines.
"""

from dataclasses import dataclass
from typing import Dict, Callable
import pandas as pd


@dataclass
class StepMutate:
    """
    Create or modify columns using custom functions.

    Applies user-defined transformations to create new columns or modify
    existing ones. Functions receive the full DataFrame and return a Series.

    Attributes:
        transformations: Dict mapping column names to transformation functions

    Examples:
        >>> step = StepMutate({
        ...     "log_value": lambda df: np.log(df["value"] + 1),
        ...     "interaction": lambda df: df["x1"] * df["x2"]
        ... })
    """

    transformations: Dict[str, Callable[[pd.DataFrame], pd.Series]]

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepMutate":
        """
        Store transformations (no fitting needed).

        Args:
            data: Training data (not used, transformations are stateless)
            training: Whether this is training data

        Returns:
            PreparedStepMutate with transformations
        """
        return PreparedStepMutate(transformations=self.transformations)


@dataclass
class PreparedStepMutate:
    """
    Fitted mutate step (stateless).

    Attributes:
        transformations: Dict of transformation functions
    """

    transformations: Dict[str, Callable[[pd.DataFrame], pd.Series]]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to create/modify columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with new/modified columns
        """
        result = data.copy()

        for col_name, transform_func in self.transformations.items():
            result[col_name] = transform_func(result)

        return result
