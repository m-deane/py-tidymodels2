"""
Feature selection preprocessing steps

Provides correlation-based selection, PCA, and other dimensionality reduction methods.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
from py_recipes.selectors import resolve_selector


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
        corr_method: Correlation coefficient method ('pearson', 'spearman', 'kendall')
    """

    outcome: str
    threshold: float = 0.9
    method: str = "multicollinearity"
    corr_method: str = "pearson"

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
            corr_matrix = data[predictor_cols].corr(method=self.corr_method).abs()

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

            correlations = data[predictor_cols + [self.outcome]].corr(method=self.corr_method)[self.outcome].abs()
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


@dataclass
class StepCorr:
    """
    Remove highly correlated features.

    Identifies pairs of features with correlation above the threshold and removes
    one from each pair. For each correlated pair, keeps the feature with higher
    mean absolute correlation with all other features.

    This is useful for removing multicollinearity among predictors before modeling.

    Attributes:
        threshold: Correlation threshold (default 0.9). Pairs with abs(correlation) > threshold are flagged
        columns: Columns to check (None = all numeric). Can be:
            - None: all numeric columns
            - str: single column name
            - List[str]: list of column names
            - Callable: selector function (e.g., all_numeric(), all_predictors())
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Examples:
        >>> # Remove features with correlation > 0.9 (default)
        >>> rec = Recipe().step_corr()
        >>>
        >>> # Use custom threshold
        >>> rec = Recipe().step_corr(threshold=0.8)
        >>>
        >>> # Check specific columns
        >>> rec = Recipe().step_corr(columns=['x1', 'x2', 'x3'], threshold=0.95)
        >>>
        >>> # Use selector
        >>> from py_recipes import all_numeric_predictors
        >>> rec = Recipe().step_corr(columns=all_numeric_predictors(), threshold=0.85)
        >>>
        >>> # Use Spearman correlation for non-linear relationships
        >>> rec = Recipe().step_corr(threshold=0.9, method='spearman')
    """

    threshold: float = 0.9
    columns: Union[None, str, List[str], Callable] = None
    method: str = "pearson"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepCorr":
        """
        Identify highly correlated features to remove.

        For each pair of features with correlation above threshold, removes the one
        with higher mean absolute correlation with all other features.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepCorr with list of columns to remove
        """
        # Resolve column selection
        if self.columns is None:
            # Default: all numeric columns
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Use resolve_selector to handle all selector types
            cols = resolve_selector(self.columns, data)
            # Filter to numeric only
            cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col])]

        if len(cols) < 2:
            # Need at least 2 columns for correlation
            return PreparedStepCorr(columns_to_remove=[])

        # Calculate correlation matrix
        corr_matrix = data[cols].corr(method=self.method).abs()

        # Find pairs with correlation above threshold
        columns_to_remove = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                if corr_matrix.iloc[i, j] > self.threshold:
                    # These two columns are highly correlated
                    # Remove the one with higher mean absolute correlation with all others

                    if col_i not in columns_to_remove and col_j not in columns_to_remove:
                        # Calculate mean absolute correlation for both columns
                        # Exclude self-correlation (diagonal = 1.0)
                        mean_corr_i = (corr_matrix.iloc[i].sum() - 1.0) / (len(cols) - 1)
                        mean_corr_j = (corr_matrix.iloc[j].sum() - 1.0) / (len(cols) - 1)

                        # Remove the column with higher mean correlation
                        if mean_corr_i > mean_corr_j:
                            columns_to_remove.add(col_i)
                        else:
                            columns_to_remove.add(col_j)

        return PreparedStepCorr(columns_to_remove=list(columns_to_remove))


@dataclass
class PreparedStepCorr:
    """
    Fitted correlation-based filtering step.

    Attributes:
        columns_to_remove: List of columns to remove due to high correlation
    """

    columns_to_remove: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated columns.

        Args:
            data: Data to transform

        Returns:
            DataFrame with correlated columns removed
        """
        if len(self.columns_to_remove) == 0:
            return data.copy()

        # Drop columns that exist in the data
        existing_cols = [col for col in self.columns_to_remove if col in data.columns]
        return data.drop(columns=existing_cols, errors='ignore')
