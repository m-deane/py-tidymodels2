"""
Mathematical transformation preprocessing steps

Provides log, sqrt, Box-Cox, and Yeo-Johnson transformations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class StepLog:
    """
    Apply logarithmic transformation.

    Computes log(x + offset) to handle zeros and negatives.

    Attributes:
        columns: Columns to transform (None = all numeric)
        base: Logarithm base (default: natural log)
        offset: Value added before transformation (default: 0)
        signed: If True, preserves sign (default: False)
    """

    columns: Optional[List[str]] = None
    base: float = np.e
    offset: float = 0.0
    signed: bool = False

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepLog":
        """
        Prepare log transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepLog ready to transform data
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        return PreparedStepLog(
            columns=cols,
            base=self.base,
            offset=self.offset,
            signed=self.signed
        )


@dataclass
class PreparedStepLog:
    """
    Fitted log transformation step.

    Attributes:
        columns: Columns to transform
        base: Logarithm base
        offset: Offset value
        signed: Preserve sign
    """

    columns: List[str]
    base: float
    offset: float
    signed: bool

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with log-transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                if self.signed:
                    # Preserve sign: sign(x) * log(abs(x) + offset)
                    result[col] = np.sign(result[col]) * np.log(np.abs(result[col]) + self.offset) / np.log(self.base)
                else:
                    result[col] = np.log(result[col] + self.offset) / np.log(self.base)

        return result


@dataclass
class StepSqrt:
    """
    Apply square root transformation.

    Computes sqrt(x) for non-negative values.

    Attributes:
        columns: Columns to transform (None = all numeric)
    """

    columns: Optional[List[str]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepSqrt":
        """
        Prepare sqrt transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepSqrt ready to transform data
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        return PreparedStepSqrt(columns=cols)


@dataclass
class PreparedStepSqrt:
    """
    Fitted sqrt transformation step.

    Attributes:
        columns: Columns to transform
    """

    columns: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sqrt transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with sqrt-transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                result[col] = np.sqrt(result[col])

        return result


@dataclass
class StepBoxCox:
    """
    Apply Box-Cox transformation.

    Applies power transformation to make data more normally distributed.
    Uses sklearn's PowerTransformer.

    Attributes:
        columns: Columns to transform (None = all numeric)
        lambdas: Dict of column -> lambda parameter (None = estimate from data)
        limits: Tuple of (lower, upper) limits for lambda search
    """

    columns: Optional[List[str]] = None
    lambdas: Optional[Dict[str, float]] = None
    limits: tuple = (-5, 5)

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepBoxCox":
        """
        Prepare Box-Cox transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepBoxCox with fitted transformers
        """
        from sklearn.preprocessing import PowerTransformer

        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Fit transformers
        transformers = {}
        lambdas_fitted = {}

        for col in cols:
            # Box-Cox requires positive values
            if (data[col] > 0).all():
                pt = PowerTransformer(method='box-cox', standardize=False)
                pt.fit(data[[col]])
                transformers[col] = pt
                lambdas_fitted[col] = pt.lambdas_[0]
            else:
                # Skip columns with non-positive values
                pass

        return PreparedStepBoxCox(
            columns=list(transformers.keys()),
            transformers=transformers,
            lambdas=lambdas_fitted
        )


@dataclass
class PreparedStepBoxCox:
    """
    Fitted Box-Cox transformation step.

    Attributes:
        columns: Columns to transform
        transformers: Dict of fitted PowerTransformers
        lambdas: Dict of fitted lambda values
    """

    columns: List[str]
    transformers: Dict[str, Any]
    lambdas: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Box-Cox transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with Box-Cox transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns and col in self.transformers:
                result[col] = self.transformers[col].transform(result[[col]]).ravel()

        return result


@dataclass
class StepYeoJohnson:
    """
    Apply Yeo-Johnson transformation.

    Similar to Box-Cox but handles negative values.
    Uses sklearn's PowerTransformer.

    Attributes:
        columns: Columns to transform (None = all numeric)
        lambdas: Dict of column -> lambda parameter (None = estimate from data)
        limits: Tuple of (lower, upper) limits for lambda search
    """

    columns: Optional[List[str]] = None
    lambdas: Optional[Dict[str, float]] = None
    limits: tuple = (-5, 5)

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepYeoJohnson":
        """
        Prepare Yeo-Johnson transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepYeoJohnson with fitted transformers
        """
        from sklearn.preprocessing import PowerTransformer

        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        # Fit transformers
        transformers = {}
        lambdas_fitted = {}

        for col in cols:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            pt.fit(data[[col]])
            transformers[col] = pt
            lambdas_fitted[col] = pt.lambdas_[0]

        return PreparedStepYeoJohnson(
            columns=cols,
            transformers=transformers,
            lambdas=lambdas_fitted
        )


@dataclass
class PreparedStepYeoJohnson:
    """
    Fitted Yeo-Johnson transformation step.

    Attributes:
        columns: Columns to transform
        transformers: Dict of fitted PowerTransformers
        lambdas: Dict of fitted lambda values
    """

    columns: List[str]
    transformers: Dict[str, Any]
    lambdas: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Yeo-Johnson transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with Yeo-Johnson transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                result[col] = self.transformers[col].transform(result[[col]]).ravel()

        return result


@dataclass
class StepInverse:
    """
    Apply inverse transformation (1/x).

    Transforms variables by computing their reciprocal, useful for
    modeling relationships where the effect decreases with magnitude.

    Attributes:
        columns: Columns to transform (None = all numeric)
        offset: Value added before inversion to avoid division by zero (default: 0)
    """

    columns: Optional[List[str]] = None
    offset: float = 0.0

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepInverse":
        """
        Prepare inverse transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepInverse ready to transform data
        """
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in data.columns]

        return PreparedStepInverse(
            columns=cols,
            offset=self.offset
        )


@dataclass
class PreparedStepInverse:
    """
    Fitted inverse transformation step.

    Attributes:
        columns: Columns to transform
        offset: Offset value
    """

    columns: List[str]
    offset: float

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with inverse-transformed columns
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                # Apply inverse: 1 / (x + offset)
                result[col] = 1.0 / (result[col] + self.offset)

        return result
