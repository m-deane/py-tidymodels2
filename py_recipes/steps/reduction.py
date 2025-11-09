"""
Advanced dimensionality reduction preprocessing steps

Provides ICA, kernel PCA, and PLS transformations.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
from py_recipes.selectors import resolve_selector


@dataclass
class StepIca:
    """
    Independent Component Analysis (ICA) transformation.

    Separates multivariate signal into independent non-Gaussian components.
    Useful for blind source separation and feature extraction.

    Attributes:
        columns: Columns to apply ICA to (None = all numeric, supports selectors)
        num_comp: Number of components to extract
        algorithm: ICA algorithm ('parallel', 'deflation')
        max_iter: Maximum iterations (default: 200)
    """

    columns: Union[None, str, List[str], Callable] = None
    num_comp: Optional[int] = None
    algorithm: str = "parallel"
    max_iter: int = 200

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepIca":
        """
        Fit ICA transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepIca with fitted transformer
        """
        from sklearn.decomposition import FastICA

        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = resolve_selector(self.columns, data)
            # Filter to numeric only
            cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col])]

        if len(cols) == 0:
            return PreparedStepIca(
                original_columns=cols,
                transformer=None,
                feature_names=[]
            )

        # Determine number of components
        n_comp = self.num_comp if self.num_comp else min(len(cols), len(data))

        # Fit ICA
        ica = FastICA(
            n_components=n_comp,
            algorithm=self.algorithm,
            max_iter=self.max_iter,
            random_state=42
        )
        ica.fit(data[cols])

        # Generate feature names
        feature_names = [f"IC{i+1}" for i in range(n_comp)]

        return PreparedStepIca(
            original_columns=cols,
            transformer=ica,
            feature_names=feature_names
        )


@dataclass
class PreparedStepIca:
    """
    Fitted ICA transformation.

    Attributes:
        original_columns: Original column names
        transformer: Fitted FastICA
        feature_names: Names for ICA components
    """

    original_columns: List[str]
    transformer: Any
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ICA transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with ICA components
        """
        if self.transformer is None or len(self.original_columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform
        components = self.transformer.transform(result[self.original_columns])

        # Add components
        for i, name in enumerate(self.feature_names):
            result[name] = components[:, i]

        # Remove original columns
        result = result.drop(columns=self.original_columns)

        return result


@dataclass
class StepKpca:
    """
    Kernel Principal Component Analysis (kernel PCA).

    Non-linear dimensionality reduction using kernel methods.
    Can capture non-linear relationships in data.

    Attributes:
        columns: Columns to apply kernel PCA to (None = all numeric, supports selectors)
        num_comp: Number of components to keep
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        gamma: Kernel coefficient (None = auto)
    """

    columns: Union[None, str, List[str], Callable] = None
    num_comp: Optional[int] = None
    kernel: str = "rbf"
    gamma: Optional[float] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepKpca":
        """
        Fit kernel PCA transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepKpca with fitted transformer
        """
        from sklearn.decomposition import KernelPCA

        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = resolve_selector(self.columns, data)
            # Filter to numeric only
            cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col])]

        if len(cols) == 0:
            return PreparedStepKpca(
                original_columns=cols,
                transformer=None,
                feature_names=[]
            )

        # Determine number of components
        n_comp = self.num_comp if self.num_comp else min(len(cols), len(data))

        # Fit kernel PCA
        kpca = KernelPCA(
            n_components=n_comp,
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=False,
            random_state=42
        )
        kpca.fit(data[cols])

        # Generate feature names
        feature_names = [f"KPC{i+1}" for i in range(n_comp)]

        return PreparedStepKpca(
            original_columns=cols,
            transformer=kpca,
            feature_names=feature_names
        )


@dataclass
class PreparedStepKpca:
    """
    Fitted kernel PCA transformation.

    Attributes:
        original_columns: Original column names
        transformer: Fitted KernelPCA
        feature_names: Names for kernel PC components
    """

    original_columns: List[str]
    transformer: Any
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply kernel PCA transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with kernel PC components
        """
        if self.transformer is None or len(self.original_columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform
        components = self.transformer.transform(result[self.original_columns])

        # Add components
        for i, name in enumerate(self.feature_names):
            result[name] = components[:, i]

        # Remove original columns
        result = result.drop(columns=self.original_columns)

        return result


@dataclass
class StepPls:
    """
    Partial Least Squares (PLS) transformation.

    Supervised dimensionality reduction that finds components
    maximally correlated with the outcome variable.

    Attributes:
        columns: Predictor columns (None = all numeric except outcome, supports selectors)
        outcome: Outcome column name
        num_comp: Number of components to extract
    """

    columns: Union[None, str, List[str], Callable] = None
    outcome: str = None
    num_comp: Optional[int] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPls":
        """
        Fit PLS transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPls with fitted transformer
        """
        from sklearn.cross_decomposition import PLSRegression

        if self.outcome is None or self.outcome not in data.columns:
            return PreparedStepPls(
                original_columns=[],
                outcome=self.outcome,
                transformer=None,
                feature_names=[]
            )

        if self.columns is None:
            # All numeric except outcome
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            cols = [col for col in numeric_cols if col != self.outcome]
        else:
            cols = resolve_selector(self.columns, data)
            # Filter to numeric only and exclude outcome
            cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col]) and col != self.outcome]

        if len(cols) == 0:
            return PreparedStepPls(
                original_columns=cols,
                outcome=self.outcome,
                transformer=None,
                feature_names=[]
            )

        # Determine number of components
        n_comp = self.num_comp if self.num_comp else min(len(cols), len(data), 10)

        # Fit PLS
        pls = PLSRegression(n_components=n_comp)
        pls.fit(data[cols], data[self.outcome])

        # Generate feature names
        feature_names = [f"PLS{i+1}" for i in range(n_comp)]

        return PreparedStepPls(
            original_columns=cols,
            outcome=self.outcome,
            transformer=pls,
            feature_names=feature_names
        )


@dataclass
class PreparedStepPls:
    """
    Fitted PLS transformation.

    Attributes:
        original_columns: Original predictor column names
        outcome: Outcome column name
        transformer: Fitted PLSRegression
        feature_names: Names for PLS components
    """

    original_columns: List[str]
    outcome: str
    transformer: Any
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PLS transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with PLS components (outcome preserved)
        """
        if self.transformer is None or len(self.original_columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform predictors
        components = self.transformer.transform(result[self.original_columns])

        # Add components
        for i, name in enumerate(self.feature_names):
            result[name] = components[:, i]

        # Remove original predictor columns
        result = result.drop(columns=self.original_columns)

        return result
