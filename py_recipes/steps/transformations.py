"""
Mathematical transformation preprocessing steps

Provides log, sqrt, Box-Cox, and Yeo-Johnson transformations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Callable
import pandas as pd
import numpy as np
from ..selectors import resolve_selector, all_numeric


@dataclass
class StepLog:
    """
    Apply logarithmic transformation.

    Computes log(x + offset) to handle zeros and negatives.

    Attributes:
        columns: Column selector (None = all numeric, str = single column,
                 List[str] = column list, Callable = selector function)
        base: Logarithm base (default: natural log)
        offset: Value added before transformation (default: 0)
        signed: If True, preserves sign (default: False)
        inplace: If True, replace original columns; if False, create new columns with suffix (default: True)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    base: float = np.e
    offset: float = 0.0
    signed: bool = False
    inplace: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepLog":
        """
        Prepare log transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepLog ready to transform data
        """
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        # Calculate safe offset if data has negative values and offset is provided
        safe_offset = self.offset
        if not self.signed and training:
            # Check minimum value across all selected columns
            min_val = min([data[col].min() for col in cols if col in data.columns], default=0)
            if min_val < 0:
                # Need offset to make all values positive
                required_offset = abs(min_val) + 1e-10
                if self.offset < required_offset:
                    import warnings
                    warnings.warn(
                        f"Log transform: offset={self.offset} is insufficient for negative values "
                        f"(min={min_val:.4f}). Using offset={required_offset:.4f} instead. "
                        f"Consider using signed=True for signed log transform.",
                        UserWarning
                    )
                    safe_offset = required_offset

        return PreparedStepLog(
            columns=cols,
            base=self.base,
            offset=safe_offset,
            signed=self.signed,
            inplace=self.inplace
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
        inplace: Replace original columns or create new ones
    """

    columns: List[str]
    base: float
    offset: float
    signed: bool
    inplace: bool

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
                    transformed = np.sign(result[col]) * np.log(np.abs(result[col]) + self.offset) / np.log(self.base)
                else:
                    transformed = np.log(result[col] + self.offset) / np.log(self.base)

                if self.inplace:
                    result[col] = transformed
                else:
                    result[f"{col}_log"] = transformed

        return result


@dataclass
class StepSqrt:
    """
    Apply square root transformation.

    Computes sqrt(x) for non-negative values.

    Attributes:
        columns: Column selector (None = all numeric, str = single column,
                 List[str] = column list, Callable = selector function)
        inplace: If True, replace original columns; if False, create new columns with suffix (default: True)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    inplace: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepSqrt":
        """
        Prepare sqrt transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepSqrt ready to transform data
        """
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepSqrt(columns=cols, inplace=self.inplace)


@dataclass
class PreparedStepSqrt:
    """
    Fitted sqrt transformation step.

    Attributes:
        columns: Columns to transform
        inplace: Replace original columns or create new ones
    """

    columns: List[str]
    inplace: bool

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
                transformed = np.sqrt(result[col])

                if self.inplace:
                    result[col] = transformed
                else:
                    result[f"{col}_sqrt"] = transformed

        return result


@dataclass
class StepBoxCox:
    """
    Apply Box-Cox transformation.

    Applies power transformation to make data more normally distributed.
    Uses sklearn's PowerTransformer.

    Attributes:
        columns: Column selector (None = all numeric, str = single column,
                 List[str] = column list, Callable = selector function)
        lambdas: Dict of column -> lambda parameter (None = estimate from data)
        limits: Tuple of (lower, upper) limits for lambda search
        inplace: If True, replace original columns; if False, create new columns with suffix (default: True)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    lambdas: Optional[Dict[str, float]] = None
    limits: tuple = (-5, 5)
    inplace: bool = True

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

        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

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
            lambdas=lambdas_fitted,
            inplace=self.inplace
        )


@dataclass
class PreparedStepBoxCox:
    """
    Fitted Box-Cox transformation step.

    Attributes:
        columns: Columns to transform
        transformers: Dict of fitted PowerTransformers
        lambdas: Dict of fitted lambda values
        inplace: Replace original columns or create new ones
    """

    columns: List[str]
    transformers: Dict[str, Any]
    lambdas: Dict[str, float]
    inplace: bool

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
                transformed = self.transformers[col].transform(result[[col]]).ravel()

                if self.inplace:
                    result[col] = transformed
                else:
                    result[f"{col}_boxcox"] = transformed

        return result


@dataclass
class StepYeoJohnson:
    """
    Apply Yeo-Johnson transformation.

    Similar to Box-Cox but handles negative values.
    Uses sklearn's PowerTransformer.

    Attributes:
        columns: Column selector (None = all numeric, str = single column,
                 List[str] = column list, Callable = selector function)
        lambdas: Dict of column -> lambda parameter (None = estimate from data)
        limits: Tuple of (lower, upper) limits for lambda search
        inplace: If True, replace original columns; if False, create new columns with suffix (default: True)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    lambdas: Optional[Dict[str, float]] = None
    limits: tuple = (-5, 5)
    inplace: bool = True

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

        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

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
            lambdas=lambdas_fitted,
            inplace=self.inplace
        )


@dataclass
class PreparedStepYeoJohnson:
    """
    Fitted Yeo-Johnson transformation step.

    Attributes:
        columns: Columns to transform
        transformers: Dict of fitted PowerTransformers
        lambdas: Dict of fitted lambda values
        inplace: Replace original columns or create new ones
    """

    columns: List[str]
    transformers: Dict[str, Any]
    lambdas: Dict[str, float]
    inplace: bool

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
                transformed = self.transformers[col].transform(result[[col]]).ravel()

                if self.inplace:
                    result[col] = transformed
                else:
                    result[f"{col}_yeojohnson"] = transformed

        return result


@dataclass
class StepInverse:
    """
    Apply inverse transformation (1/x).

    Transforms variables by computing their reciprocal, useful for
    modeling relationships where the effect decreases with magnitude.

    Attributes:
        columns: Column selector (None = all numeric, str = single column,
                 List[str] = column list, Callable = selector function)
        offset: Value added before inversion to avoid division by zero (default: 0)
        inplace: If True, replace original columns; if False, create new columns with suffix (default: True)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    offset: float = 0.0
    inplace: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepInverse":
        """
        Prepare inverse transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepInverse ready to transform data
        """
        selector = self.columns if self.columns is not None else all_numeric()
        cols = resolve_selector(selector, data)

        return PreparedStepInverse(
            columns=cols,
            offset=self.offset,
            inplace=self.inplace
        )


@dataclass
class PreparedStepInverse:
    """
    Fitted inverse transformation step.

    Attributes:
        columns: Columns to transform
        offset: Offset value
        inplace: Replace original columns or create new ones
    """

    columns: List[str]
    offset: float
    inplace: bool

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
                transformed = 1.0 / (result[col] + self.offset)

                if self.inplace:
                    result[col] = transformed
                else:
                    result[f"{col}_inverse"] = transformed

        return result
