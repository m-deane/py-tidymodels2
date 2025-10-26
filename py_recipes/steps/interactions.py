"""
Feature interaction preprocessing steps

Provides interaction terms and ratio features.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class StepInteract:
    """
    Create interaction features between columns.

    Generates multiplicative interaction terms between specified columns,
    capturing non-additive relationships.

    Attributes:
        interactions: List of column pairs to interact (e.g., [("x1", "x2"), ("x1", "x3")])
        separator: Separator for interaction names (default: "_x_")
    """

    interactions: List[tuple]
    separator: str = "_x_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepInteract":
        """
        Prepare interaction step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepInteract ready to create interactions
        """
        # Validate that columns exist
        valid_interactions = []
        feature_names = []

        for col1, col2 in self.interactions:
            if col1 in data.columns and col2 in data.columns:
                valid_interactions.append((col1, col2))
                feature_names.append(f"{col1}{self.separator}{col2}")

        return PreparedStepInteract(
            interactions=valid_interactions,
            separator=self.separator,
            feature_names=feature_names
        )


@dataclass
class PreparedStepInteract:
    """
    Fitted interaction step.

    Attributes:
        interactions: List of column pairs to interact
        separator: Separator for names
        feature_names: Names for interaction features
    """

    interactions: List[tuple]
    separator: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with interaction features added
        """
        result = data.copy()

        for (col1, col2), feature_name in zip(self.interactions, self.feature_names):
            if col1 in result.columns and col2 in result.columns:
                result[feature_name] = result[col1] * result[col2]

        return result


@dataclass
class StepRatio:
    """
    Create ratio features between columns.

    Generates ratio features by dividing one column by another,
    useful for normalization or relative comparisons.

    Attributes:
        ratios: List of (numerator, denominator) column pairs
        offset: Small value added to denominator to avoid division by zero (default: 1e-10)
        separator: Separator for ratio names (default: "_per_")
    """

    ratios: List[tuple]
    offset: float = 1e-10
    separator: str = "_per_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepRatio":
        """
        Prepare ratio step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepRatio ready to create ratios
        """
        # Validate that columns exist
        valid_ratios = []
        feature_names = []

        for num_col, den_col in self.ratios:
            if num_col in data.columns and den_col in data.columns:
                valid_ratios.append((num_col, den_col))
                feature_names.append(f"{num_col}{self.separator}{den_col}")

        return PreparedStepRatio(
            ratios=valid_ratios,
            offset=self.offset,
            separator=self.separator,
            feature_names=feature_names
        )


@dataclass
class PreparedStepRatio:
    """
    Fitted ratio step.

    Attributes:
        ratios: List of (numerator, denominator) pairs
        offset: Offset for division
        separator: Separator for names
        feature_names: Names for ratio features
    """

    ratios: List[tuple]
    offset: float
    separator: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with ratio features added
        """
        result = data.copy()

        for (num_col, den_col), feature_name in zip(self.ratios, self.feature_names):
            if num_col in result.columns and den_col in result.columns:
                # Add offset to avoid division by zero
                denominator = result[den_col] + self.offset
                result[feature_name] = result[num_col] / denominator

        return result
