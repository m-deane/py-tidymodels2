"""
Discretization preprocessing steps

Provides binning and cutting of continuous variables into categories.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class StepDiscretize:
    """
    Discretize numeric columns into bins.

    Converts continuous variables into categorical bins based on quantiles
    or equal-width intervals. Useful for capturing non-linear relationships
    in linear models.

    Attributes:
        columns: Columns to discretize (None = all numeric)
        num_breaks: Number of bins (default: 4)
        method: Binning method ('quantile' or 'width')
        labels: Custom bin labels (None = auto-generate)
    """

    columns: Optional[List[str]] = None
    num_breaks: int = 4
    method: str = "quantile"
    labels: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepDiscretize":
        """
        Calculate bin edges from training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepDiscretize with bin edges
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Calculate bin edges for each column
        bin_edges = {}

        for col in cols:
            col_data = data[col].dropna()

            if self.method == "quantile":
                # Quantile-based bins
                quantiles = np.linspace(0, 1, self.num_breaks + 1)
                edges = np.quantile(col_data, quantiles)
                # Ensure unique edges
                edges = np.unique(edges)
            else:  # width
                # Equal-width bins
                edges = np.linspace(col_data.min(), col_data.max(), self.num_breaks + 1)

            bin_edges[col] = edges

        # Generate labels if not provided
        if self.labels is None:
            labels = [f"bin_{i+1}" for i in range(self.num_breaks)]
        else:
            labels = self.labels

        return PreparedStepDiscretize(
            columns=cols,
            bin_edges=bin_edges,
            labels=labels
        )


@dataclass
class PreparedStepDiscretize:
    """
    Fitted discretization step.

    Attributes:
        columns: Columns to discretize
        bin_edges: Dict of column -> bin edge arrays
        labels: Labels for bins
    """

    columns: List[str]
    bin_edges: dict
    labels: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply discretization.

        Args:
            data: Data to transform

        Returns:
            DataFrame with discretized columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns and col in self.bin_edges:
                edges = self.bin_edges[col]

                # Cut into bins
                # Use fewer labels if we have fewer bins due to unique edges
                n_bins = len(edges) - 1
                labels = self.labels[:n_bins] if len(self.labels) >= n_bins else None

                result[col] = pd.cut(
                    result[col],
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    duplicates='drop'
                )

        return result


@dataclass
class StepCut:
    """
    Cut numeric columns at specified thresholds.

    Bins continuous variables at user-specified breakpoints,
    providing fine control over bin boundaries.

    Attributes:
        columns: Columns to cut
        breaks: Dict of column -> list of breakpoints
        labels: Dict of column -> list of labels (None = auto-generate)
        include_lowest: Include lowest value in first bin (default: True)
    """

    columns: List[str]
    breaks: dict
    labels: Optional[dict] = None
    include_lowest: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepCut":
        """
        Prepare cutting step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepCut ready to cut
        """
        cols = [col for col in self.columns if col in data.columns]

        # Validate breaks
        valid_breaks = {}
        valid_labels = {}

        for col in cols:
            if col in self.breaks:
                breaks = sorted(self.breaks[col])
                valid_breaks[col] = breaks

                # Generate or validate labels
                n_bins = len(breaks) - 1
                if self.labels and col in self.labels:
                    labels = self.labels[col]
                    if len(labels) != n_bins:
                        # Auto-generate if mismatch
                        labels = [f"bin_{i+1}" for i in range(n_bins)]
                else:
                    labels = [f"bin_{i+1}" for i in range(n_bins)]

                valid_labels[col] = labels

        return PreparedStepCut(
            columns=cols,
            breaks=valid_breaks,
            labels=valid_labels,
            include_lowest=self.include_lowest
        )


@dataclass
class PreparedStepCut:
    """
    Fitted cutting step.

    Attributes:
        columns: Columns to cut
        breaks: Dict of breakpoints
        labels: Dict of labels
        include_lowest: Include lowest value
    """

    columns: List[str]
    breaks: dict
    labels: dict
    include_lowest: bool

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cutting transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with cut columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns and col in self.breaks:
                breaks = self.breaks[col]
                labels = self.labels[col]

                result[col] = pd.cut(
                    result[col],
                    bins=breaks,
                    labels=labels,
                    include_lowest=self.include_lowest,
                    duplicates='drop'
                )

        return result


@dataclass
class StepPercentile:
    """
    Convert numeric columns to percentile ranks.

    Transforms continuous variables to their percentile ranks (0-100 scale),
    useful for normalizing distributions and creating rank-based features.

    Attributes:
        columns: Columns to convert (None = all numeric)
        num_breaks: Number of percentile bins (default: 100 for 0-100 scale)
        as_integer: Return integer percentiles (default: True)
    """

    columns: Optional[List[str]] = None
    num_breaks: int = 100
    as_integer: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPercentile":
        """
        Calculate percentile breakpoints from training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPercentile with percentile breakpoints
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Calculate percentile breakpoints for each column
        percentile_breaks = {}

        for col in cols:
            col_data = data[col].dropna()

            # Calculate percentile breakpoints
            percentiles = np.linspace(0, 100, self.num_breaks + 1)
            breaks = np.percentile(col_data, percentiles)

            # Ensure unique breaks
            breaks = np.unique(breaks)

            percentile_breaks[col] = breaks

        return PreparedStepPercentile(
            columns=cols,
            percentile_breaks=percentile_breaks,
            num_breaks=self.num_breaks,
            as_integer=self.as_integer
        )


@dataclass
class PreparedStepPercentile:
    """
    Fitted percentile conversion step.

    Attributes:
        columns: Columns to convert
        percentile_breaks: Dict of column -> percentile breakpoints
        num_breaks: Number of percentile bins
        as_integer: Return integer percentiles
    """

    columns: List[str]
    percentile_breaks: dict
    num_breaks: int
    as_integer: bool

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert values to percentile ranks.

        Args:
            data: Data to transform

        Returns:
            DataFrame with percentile-transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns and col in self.percentile_breaks:
                breaks = self.percentile_breaks[col]

                # Use searchsorted to find percentile rank
                # This gives us the bin index for each value
                ranks = np.searchsorted(breaks, result[col].values, side='right')

                # Convert to percentile (0-100 scale or 0-num_breaks scale)
                n_bins = len(breaks) - 1
                if n_bins > 0:
                    percentiles = (ranks / n_bins) * self.num_breaks
                else:
                    percentiles = np.zeros_like(ranks, dtype=float)

                # Handle values outside the range
                percentiles = np.clip(percentiles, 0, self.num_breaks)

                if self.as_integer:
                    result[col] = percentiles.astype(int)
                else:
                    result[col] = percentiles

        return result
