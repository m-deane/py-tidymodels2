"""
Extended categorical preprocessing steps

Provides advanced categorical encoding: pooling rare levels, handling novel categories,
missing indicators, and integer encoding.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Callable
import pandas as pd
import numpy as np
from py_recipes.selectors import resolve_selector, all_nominal


@dataclass
class StepOther:
    """
    Pool infrequent categorical levels into "other".

    Combines rare factor levels that occur below a threshold
    into a single "other" category, reducing dimensionality.

    Attributes:
        columns: Column selector (None = all categorical, can be list, string, or callable)
        threshold: Minimum frequency to keep level (default: 0.05)
        other_label: Label for pooled category (default: "other")
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    threshold: float = 0.05
    other_label: str = "other"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepOther":
        """
        Identify infrequent levels to pool.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepOther with level mappings
        """
        selector = self.columns if self.columns is not None else all_nominal()
        cols = resolve_selector(selector, data)

        # Track levels to keep for each column
        levels_to_keep = {}
        n_rows = len(data)

        for col in cols:
            value_counts = data[col].value_counts()
            freq_threshold = self.threshold * n_rows

            # Keep levels that meet threshold
            keep_levels = value_counts[value_counts >= freq_threshold].index.tolist()
            levels_to_keep[col] = keep_levels

        return PreparedStepOther(
            columns=cols,
            levels_to_keep=levels_to_keep,
            other_label=self.other_label
        )


@dataclass
class PreparedStepOther:
    """
    Fitted other step for pooling rare levels.

    Attributes:
        columns: Columns to transform
        levels_to_keep: Dict of column -> list of levels to keep
        other_label: Label for pooled category
    """

    columns: List[str]
    levels_to_keep: Dict[str, List[Any]]
    other_label: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pool infrequent levels into "other".

        Args:
            data: Data to transform

        Returns:
            DataFrame with rare levels pooled
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                keep_levels = self.levels_to_keep[col]

                # Replace levels not in keep_levels with other_label
                result[col] = result[col].apply(
                    lambda x: x if x in keep_levels else self.other_label
                )

        return result


@dataclass
class StepNovel:
    """
    Handle novel categorical levels in new data.

    Assigns novel levels (not seen in training) to a designated value,
    preventing errors when encoding test data.

    Attributes:
        columns: Column selector (None = all categorical, can be list, string, or callable)
        novel_label: Label for novel levels (default: "new")
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    novel_label: str = "new"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepNovel":
        """
        Record training levels.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepNovel with known levels
        """
        selector = self.columns if self.columns is not None else all_nominal()
        cols = resolve_selector(selector, data)

        # Record all levels seen in training
        training_levels = {}
        for col in cols:
            training_levels[col] = data[col].dropna().unique().tolist()

        return PreparedStepNovel(
            columns=cols,
            training_levels=training_levels,
            novel_label=self.novel_label
        )


@dataclass
class PreparedStepNovel:
    """
    Fitted novel level handler.

    Attributes:
        columns: Columns to transform
        training_levels: Dict of column -> list of known levels
        novel_label: Label for novel levels
    """

    columns: List[str]
    training_levels: Dict[str, List[Any]]
    novel_label: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace novel levels with designated label.

        Args:
            data: Data to transform

        Returns:
            DataFrame with novel levels handled
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                known_levels = self.training_levels[col]

                # Replace novel levels
                result[col] = result[col].apply(
                    lambda x: x if pd.isna(x) or x in known_levels else self.novel_label
                )

        return result


@dataclass
class StepUnknown:
    """
    Assign missing categorical values to "unknown" level.

    Creates a dedicated factor level for missing categorical data,
    allowing models that can't handle NA to work with incomplete data.
    Essential preprocessing for categorical variables with missingness.

    Attributes:
        columns: Column selector (None = all categorical, can be list, string, or callable)
        unknown_label: Label for missing values (default: "_unknown_")
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    unknown_label: str = "_unknown_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepUnknown":
        """
        Identify categorical columns with missing values.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepUnknown ready to handle missing values
        """
        selector = self.columns if self.columns is not None else all_nominal()
        cols = resolve_selector(selector, data)

        return PreparedStepUnknown(
            columns=cols,
            unknown_label=self.unknown_label
        )


@dataclass
class PreparedStepUnknown:
    """
    Fitted unknown level handler.

    Attributes:
        columns: Columns to transform
        unknown_label: Label for missing values
    """

    columns: List[str]
    unknown_label: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing values with unknown label.

        Args:
            data: Data to transform

        Returns:
            DataFrame with NA replaced by unknown label
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                # Replace NA with unknown_label
                result[col] = result[col].fillna(self.unknown_label)

        return result


@dataclass
class StepIndicateNa:
    """
    Create indicator columns for missing values.

    Adds binary indicator columns showing where missing values occurred,
    preserving information about missingness patterns.

    Attributes:
        columns: Column selector (None = all with NA, can be list, string, or callable)
        prefix: Prefix for indicator columns (default: "na_ind")
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    prefix: str = "na_ind"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepIndicateNa":
        """
        Identify columns with missing values.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepIndicateNa ready to create indicators
        """
        if self.columns is None:
            # Find columns with any missing values
            cols = [col for col in data.columns if data[col].isna().any()]
        else:
            selector = self.columns
            cols = resolve_selector(selector, data)
            # Filter to only columns with missing values
            cols = [col for col in cols if data[col].isna().any()]

        return PreparedStepIndicateNa(columns=cols, prefix=self.prefix)


@dataclass
class PreparedStepIndicateNa:
    """
    Fitted missing indicator step.

    Attributes:
        columns: Columns to create indicators for
        prefix: Prefix for indicator columns
    """

    columns: List[str]
    prefix: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create missing value indicator columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with indicator columns added
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                indicator_name = f"{self.prefix}_{col}"
                result[indicator_name] = result[col].isna().astype(int)

        return result


@dataclass
class StepInteger:
    """
    Integer encode categorical variables.

    Converts categorical variables to integers, useful for tree-based models.
    Maintains consistent encoding between training and test data.

    Attributes:
        columns: Categorical columns to encode (None = all categorical)
        zero_based: Use zero-based indexing (default: True)
    """

    columns: Optional[List[str]] = None
    zero_based: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepInteger":
        """
        Create integer mappings for categorical levels.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepInteger with level mappings
        """
        if self.columns is None:
            cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Create mappings for each column
        mappings = {}

        for col in cols:
            unique_values = data[col].dropna().unique()

            # Sort for consistency
            unique_values = sorted(unique_values, key=str)

            # Create integer mapping
            if self.zero_based:
                mapping = {val: idx for idx, val in enumerate(unique_values)}
            else:
                mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

            mappings[col] = mapping

        return PreparedStepInteger(
            columns=cols,
            mappings=mappings,
            zero_based=self.zero_based
        )


@dataclass
class PreparedStepInteger:
    """
    Fitted integer encoding step.

    Attributes:
        columns: Columns to encode
        mappings: Dict of column -> {level: integer} mappings
        zero_based: Whether encoding is zero-based
    """

    columns: List[str]
    mappings: Dict[str, Dict[Any, int]]
    zero_based: bool

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply integer encoding.

        Args:
            data: Data to transform

        Returns:
            DataFrame with integer-encoded columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                mapping = self.mappings[col]

                # Map values, unknown values become NaN
                result[col] = result[col].map(mapping)

        return result
