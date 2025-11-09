"""
Basis function preprocessing steps

Provides B-splines, natural splines, polynomial features, and harmonic functions.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np


@dataclass
class StepBs:
    """
    Create B-spline basis functions.

    Generates B-spline (basis spline) features for smooth non-linear transformations.
    Useful for capturing non-linear relationships.

    Attributes:
        column: Column to create splines for
        degree: Degree of spline (default: 3 for cubic)
        df: Degrees of freedom (number of basis functions)
        knots: Number of internal knots (alternative to df)
    """

    column: str
    degree: int = 3
    df: Optional[int] = None
    knots: Optional[int] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepBs":
        """
        Fit B-spline transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepBs with fitted spline
        """
        from scipy.interpolate import BSpline

        if self.column not in data.columns:
            return PreparedStepBs(
                column=self.column,
                degree=self.degree,
                knot_values=np.array([]),
                feature_names=[]
            )

        x = data[self.column].dropna().values

        # Determine number of internal knots
        if self.df is not None:
            # df = degree + n_knots + 1
            n_knots = self.df - self.degree - 1
        elif self.knots is not None:
            n_knots = self.knots
        else:
            n_knots = 5  # default

        n_knots = max(0, n_knots)

        # Place knots at quantiles
        if n_knots > 0:
            quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
            internal_knots = np.quantile(x, quantiles)
        else:
            internal_knots = np.array([])

        # Full knot sequence includes boundaries
        x_min, x_max = x.min(), x.max()
        knot_values = np.concatenate([
            np.repeat(x_min, self.degree + 1),
            internal_knots,
            np.repeat(x_max, self.degree + 1)
        ])

        # Feature names
        n_features = len(internal_knots) + self.degree + 1
        feature_names = [f"{self.column}_bs_{i+1}" for i in range(n_features)]

        return PreparedStepBs(
            column=self.column,
            degree=self.degree,
            knot_values=knot_values,
            feature_names=feature_names
        )


@dataclass
class PreparedStepBs:
    """
    Fitted B-spline transformation.

    Attributes:
        column: Column to transform
        degree: Spline degree
        knot_values: Fitted knot locations
        feature_names: Names for spline features
    """

    column: str
    degree: int
    knot_values: np.ndarray
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply B-spline transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with B-spline features added
        """
        from scipy.interpolate import BSpline

        result = data.copy()

        if self.column not in result.columns or len(self.knot_values) == 0:
            return result

        x = result[self.column].values

        # Compute B-spline basis
        n_features = len(self.feature_names)
        spline_features = np.zeros((len(x), n_features))

        for i in range(n_features):
            # Create coefficients (all zeros except position i)
            c = np.zeros(n_features)
            c[i] = 1.0

            # Create B-spline
            bspl = BSpline(self.knot_values, c, self.degree, extrapolate=True)
            spline_features[:, i] = bspl(x)

        # Add features to result
        for i, name in enumerate(self.feature_names):
            result[name] = spline_features[:, i]

        # Remove original column
        result = result.drop(columns=[self.column])

        return result


@dataclass
class StepNs:
    """
    Create natural spline basis functions.

    Natural splines are cubic splines with linear behavior beyond boundary knots,
    reducing overfitting at extremes.

    Attributes:
        column: Column to create splines for
        df: Degrees of freedom (number of basis functions)
        knots: Number of internal knots (alternative to df)
    """

    column: str
    df: Optional[int] = None
    knots: Optional[int] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepNs":
        """
        Fit natural spline transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepNs with fitted spline
        """
        if self.column not in data.columns:
            return PreparedStepNs(
                column=self.column,
                knot_values=np.array([]),
                boundary_knots=(0, 1),
                feature_names=[]
            )

        x = data[self.column].dropna().values

        # Determine number of internal knots
        if self.df is not None:
            n_knots = self.df - 1
        elif self.knots is not None:
            n_knots = self.knots
        else:
            n_knots = 4  # default

        n_knots = max(0, n_knots)

        # Place knots at quantiles
        if n_knots > 0:
            quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
            internal_knots = np.quantile(x, quantiles)
        else:
            internal_knots = np.array([])

        # Boundary knots
        x_min, x_max = x.min(), x.max()
        boundary_knots = (x_min, x_max)

        # Feature names
        n_features = len(internal_knots) + 1
        feature_names = [f"{self.column}_ns_{i+1}" for i in range(n_features)]

        return PreparedStepNs(
            column=self.column,
            knot_values=internal_knots,
            boundary_knots=boundary_knots,
            feature_names=feature_names
        )


@dataclass
class PreparedStepNs:
    """
    Fitted natural spline transformation.

    Attributes:
        column: Column to transform
        knot_values: Internal knot locations
        boundary_knots: Boundary knot locations
        feature_names: Names for spline features
    """

    column: str
    knot_values: np.ndarray
    boundary_knots: tuple
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply natural spline transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with natural spline features added
        """
        result = data.copy()

        if self.column not in result.columns:
            return result

        x = result[self.column].values
        x_min, x_max = self.boundary_knots

        # Simple natural spline basis (polynomial approximation)
        n_features = len(self.feature_names)
        spline_features = np.zeros((len(x), n_features))

        # Normalize x to [0, 1]
        x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x)
        x_norm = np.clip(x_norm, 0, 1)

        # Create basis functions
        if n_features > 0:
            spline_features[:, 0] = x_norm

            for i in range(1, n_features):
                if i <= len(self.knot_values):
                    knot = (self.knot_values[i-1] - x_min) / (x_max - x_min)
                    spline_features[:, i] = np.maximum(0, x_norm - knot) ** 3

        # Add features to result
        for i, name in enumerate(self.feature_names):
            result[name] = spline_features[:, i]

        # Remove original column
        result = result.drop(columns=[self.column])

        return result


@dataclass
class StepPoly:
    """
    Create polynomial features.

    Generates polynomial features up to specified degree,
    including interaction terms if requested.

    Attributes:
        columns: Columns to create polynomials for (supports selectors)
        degree: Maximum polynomial degree (default: 2)
        include_interactions: Include cross terms (default: False)
        inplace: If True, replace original columns; if False, keep originals and add polynomial features (default: True)
    """

    columns: Union[List[str], Callable, str, None]
    degree: int = 2
    include_interactions: bool = False
    inplace: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepPoly":
        """
        Prepare polynomial transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepPoly ready to transform
        """
        from sklearn.preprocessing import PolynomialFeatures
        from py_recipes.selectors import resolve_selector

        # Resolve selector to actual column names
        cols = resolve_selector(self.columns, data)

        if len(cols) == 0:
            return PreparedStepPoly(
                columns=[],
                poly_transformer=None,
                feature_names=[]
            )

        # Determine interaction mode
        if self.include_interactions:
            # Create both polynomial and interaction terms
            poly = PolynomialFeatures(
                degree=self.degree,
                interaction_only=False,
                include_bias=False
            )
            poly.fit(data[cols])
            feature_names = poly.get_feature_names_out(cols)
        else:
            # Create ONLY pure polynomial terms (x^2, x^3), NO interactions
            # sklearn doesn't support this directly, so we create all terms then filter
            poly = PolynomialFeatures(
                degree=self.degree,
                interaction_only=False,
                include_bias=False
            )
            poly.fit(data[cols])
            all_feature_names = poly.get_feature_names_out(cols)

            # Filter to keep only single-variable polynomial terms (e.g., "x^2", not "x1 x2")
            # An interaction term contains spaces (e.g., "x1 x2"), pure polynomial doesn't
            feature_names = [name for name in all_feature_names if ' ' not in name]

            # Update poly to only keep these features
            feature_indices = [i for i, name in enumerate(all_feature_names) if ' ' not in name]
            # Store indices for filtering during transform
            poly._feature_indices = feature_indices

        # Replace spaces with underscores for formula compatibility
        # sklearn uses spaces like "x1 x2" but we need "x1_x2"
        feature_names = [name.replace(' ', '_') for name in feature_names]

        return PreparedStepPoly(
            columns=cols,
            poly_transformer=poly,
            feature_names=list(feature_names),
            inplace=self.inplace
        )


@dataclass
class PreparedStepPoly:
    """
    Fitted polynomial transformation.

    Attributes:
        columns: Original columns
        poly_transformer: Fitted PolynomialFeatures
        feature_names: Names for polynomial features
        inplace: Whether to replace original columns or keep them
    """

    columns: List[str]
    poly_transformer: Any
    feature_names: List[str]
    inplace: bool = True

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply polynomial transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with polynomial features
        """
        if self.poly_transformer is None or len(self.columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform
        poly_data = self.poly_transformer.transform(result[self.columns])

        # Check if we need to filter features (when include_interactions=False)
        if hasattr(self.poly_transformer, '_feature_indices'):
            # Only use selected feature columns
            feature_indices = self.poly_transformer._feature_indices
            for i, name in enumerate(self.feature_names):
                actual_index = feature_indices[i]
                result[name] = poly_data[:, actual_index]
        else:
            # Use all features (include_interactions=True)
            for i, name in enumerate(self.feature_names):
                result[name] = poly_data[:, i]

        # Remove original columns if inplace=True
        if self.inplace:
            result = result.drop(columns=self.columns)

        return result


@dataclass
class StepHarmonic:
    """
    Create harmonic (Fourier) basis functions.

    Generates sine and cosine features for capturing periodic patterns,
    useful for seasonal time series data.

    Attributes:
        column: Column to create harmonics for (typically time index)
        frequency: Number of harmonics/cycles to include
        period: Period of seasonality (e.g., 12 for monthly, 7 for weekly)
    """

    column: str
    frequency: int = 1
    period: float = 1.0

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepHarmonic":
        """
        Prepare harmonic transformation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepHarmonic ready to transform
        """
        if self.column not in data.columns:
            return PreparedStepHarmonic(
                column=self.column,
                frequency=self.frequency,
                period=self.period,
                feature_names=[]
            )

        # Generate feature names
        feature_names = []
        for k in range(1, self.frequency + 1):
            feature_names.append(f"{self.column}_sin_{k}")
            feature_names.append(f"{self.column}_cos_{k}")

        return PreparedStepHarmonic(
            column=self.column,
            frequency=self.frequency,
            period=self.period,
            feature_names=feature_names
        )


@dataclass
class PreparedStepHarmonic:
    """
    Fitted harmonic transformation.

    Attributes:
        column: Column to transform
        frequency: Number of harmonics
        period: Period of seasonality
        feature_names: Names for harmonic features
    """

    column: str
    frequency: int
    period: float
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply harmonic transformation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with harmonic features added
        """
        result = data.copy()

        if self.column not in result.columns:
            return result

        x = result[self.column].values

        # Create sine and cosine features
        for k in range(1, self.frequency + 1):
            # Frequency component
            freq = 2 * np.pi * k / self.period

            sin_name = f"{self.column}_sin_{k}"
            cos_name = f"{self.column}_cos_{k}"

            result[sin_name] = np.sin(freq * x)
            result[cos_name] = np.cos(freq * x)

        return result
