"""
Feature selection preprocessing steps

Provides correlation-based selection, PCA, and other dimensionality reduction methods.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class StepPCA:
    """
    Principal Component Analysis (PCA) transformation.

    Reduces dimensionality by projecting data onto principal components.

    Attributes:
        columns: Columns to apply PCA to (None = all numeric)
        num_comp: Number of components to keep
        threshold: Variance threshold (alternative to num_comp)
    """

    columns: Optional[List[str]] = None
    num_comp: Optional[int] = None
    threshold: Optional[float] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPCA":
        """
        Fit PCA to training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPCA with fitted PCA model
        """
        from sklearn.decomposition import PCA

        # Determine columns
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        if len(cols) == 0:
            return PreparedStepPCA(columns=[], pca=None, num_comp=0)

        # Determine number of components
        if self.num_comp is not None:
            n_components = min(self.num_comp, len(cols))
        elif self.threshold is not None:
            n_components = self.threshold  # Will be interpreted as variance threshold
        else:
            n_components = len(cols)  # Keep all components

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(data[cols])

        return PreparedStepPCA(
            columns=cols,
            pca=pca,
            num_comp=pca.n_components_
        )


@dataclass
class PreparedStepPCA:
    """
    Fitted PCA step.

    Attributes:
        columns: Original columns
        pca: Fitted PCA model
        num_comp: Number of components
    """

    columns: List[str]
    pca: Optional[Any]
    num_comp: int

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with PCA components
        """
        result = data.copy()

        if self.pca is None or len(self.columns) == 0:
            return result

        # Check which columns exist
        existing_cols = [col for col in self.columns if col in result.columns]

        if len(existing_cols) > 0:
            # Transform data
            pca_values = self.pca.transform(result[existing_cols])

            # Remove original columns
            result = result.drop(columns=existing_cols)

            # Add PCA components
            for i in range(self.num_comp):
                result[f"PC{i+1}"] = pca_values[:, i]

        return result


@dataclass
class StepSelectCorr:
    """
    Select features based on correlation with outcome.

    Removes features with low correlation to the outcome variable or
    high correlation with other predictors (multicollinearity).

    Attributes:
        outcome: Outcome column name
        threshold: Correlation threshold (default 0.9 for multicollinearity)
        method: 'multicollinearity' or 'outcome'
    """

    outcome: str
    threshold: float = 0.9
    method: str = "multicollinearity"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepSelectCorr":
        """
        Determine which features to keep based on correlation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepSelectCorr with selected features
        """
        # Get numeric columns (excluding outcome)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        predictor_cols = [col for col in numeric_cols if col != self.outcome]

        if len(predictor_cols) == 0:
            return PreparedStepSelectCorr(
                columns_to_keep=predictor_cols,
                outcome=self.outcome
            )

        if self.method == "multicollinearity":
            # Remove highly correlated predictors
            corr_matrix = data[predictor_cols].corr().abs()

            # Find pairs of highly correlated features
            columns_to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > self.threshold:
                        # Drop the second column in the pair
                        columns_to_drop.add(corr_matrix.columns[i])

            columns_to_keep = [col for col in predictor_cols if col not in columns_to_drop]

        elif self.method == "outcome":
            # Keep only features with sufficient correlation to outcome
            if self.outcome not in data.columns:
                raise ValueError(f"Outcome column '{self.outcome}' not found")

            correlations = data[predictor_cols + [self.outcome]].corr()[self.outcome].abs()
            columns_to_keep = correlations[correlations > self.threshold].index.tolist()
            columns_to_keep = [col for col in columns_to_keep if col != self.outcome]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return PreparedStepSelectCorr(
            columns_to_keep=columns_to_keep,
            outcome=self.outcome
        )


@dataclass
class PreparedStepSelectCorr:
    """
    Fitted correlation-based selection step.

    Attributes:
        columns_to_keep: Features to retain
        outcome: Outcome column name
    """

    columns_to_keep: List[str]
    outcome: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection.

        Args:
            data: Data to transform

        Returns:
            DataFrame with selected features
        """
        result = data.copy()

        # Keep only selected columns plus outcome and any non-numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()

        # Columns to drop: numeric columns not in keep list and not the outcome
        columns_to_drop = [
            col for col in numeric_cols
            if col not in self.columns_to_keep and col != self.outcome
        ]

        result = result.drop(columns=columns_to_drop)

        return result
