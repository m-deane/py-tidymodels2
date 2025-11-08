"""
Steps for imputing missing values

Provides mean, median, mode, KNN, and linear interpolation imputation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Callable
import pandas as pd
import numpy as np

from py_recipes.selectors import resolve_selector, where


@dataclass
class StepImputeMean:
    """
    Impute missing values using mean.

    Replaces NA values in numeric columns with the training mean.

    Attributes:
        columns: Columns to impute (None = all numeric with NA, or use selector)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMean":
        """
        Calculate means for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMean with mean values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with missing values
            selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        # Calculate means
        means = {}
        for col in cols:
            if col in data.columns:
                means[col] = data[col].mean()

        return PreparedStepImputeMean(means=means)


@dataclass
class PreparedStepImputeMean:
    """
    Fitted mean imputation step.

    Attributes:
        means: Dictionary mapping column names to mean values
    """

    means: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with means.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, mean_val in self.means.items():
            if col in result.columns:
                result[col] = result[col].fillna(mean_val)

        return result


@dataclass
class StepImputeMedian:
    """
    Impute missing values using median.

    Replaces NA values in numeric columns with the training median.

    Attributes:
        columns: Columns to impute (None = all numeric with NA, or use selector)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMedian":
        """
        Calculate medians for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMedian with median values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with missing values
            selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        # Calculate medians
        medians = {}
        for col in cols:
            if col in data.columns:
                medians[col] = data[col].median()

        return PreparedStepImputeMedian(medians=medians)


@dataclass
class PreparedStepImputeMedian:
    """
    Fitted median imputation step.

    Attributes:
        medians: Dictionary mapping column names to median values
    """

    medians: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with medians.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, median_val in self.medians.items():
            if col in result.columns:
                result[col] = result[col].fillna(median_val)

        return result


@dataclass
class StepImputeMode:
    """
    Impute missing values using mode (most frequent value).

    Replaces NA values with the most common value.
    Works for both numeric and categorical columns.

    Attributes:
        columns: Columns to impute (None = all columns with NA, or use selector)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeMode":
        """
        Calculate mode for training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeMode with mode values
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select all columns with missing values
            selector = where(lambda s: s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        # Calculate modes
        modes = {}
        for col in cols:
            if col in data.columns:
                mode_series = data[col].mode()
                if len(mode_series) > 0:
                    modes[col] = mode_series.iloc[0]
                else:
                    modes[col] = None

        return PreparedStepImputeMode(modes=modes)


@dataclass
class PreparedStepImputeMode:
    """
    Fitted mode imputation step.

    Attributes:
        modes: Dictionary mapping column names to mode values
    """

    modes: Dict[str, Any]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with modes.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col, mode_val in self.modes.items():
            if col in result.columns and mode_val is not None:
                result[col] = result[col].fillna(mode_val)

        return result


@dataclass
class StepImputeKnn:
    """
    Impute missing values using K-Nearest Neighbors.

    Uses KNN algorithm to impute missing values based on similar observations.
    Useful when missing values have patterns related to other features.

    Attributes:
        columns: Columns to impute (None = all numeric with NA, or use selector)
        neighbors: Number of neighbors to use (default: 5)
        weights: Weight function ('uniform' or 'distance')
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    neighbors: int = 5
    weights: str = "uniform"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeKnn":
        """
        Fit KNN imputer on training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeKnn with fitted imputer
        """
        from sklearn.impute import KNNImputer

        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with missing values
            selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        if len(cols) == 0:
            return PreparedStepImputeKnn(
                columns=[],
                imputer=None,
                feature_names=[]
            )

        # Get all numeric columns for context
        all_numeric = data.select_dtypes(include=[np.number]).columns.tolist()

        # Fit imputer on numeric data
        imputer = KNNImputer(
            n_neighbors=self.neighbors,
            weights=self.weights
        )
        imputer.fit(data[all_numeric])

        return PreparedStepImputeKnn(
            columns=cols,
            imputer=imputer,
            feature_names=all_numeric
        )


@dataclass
class PreparedStepImputeKnn:
    """
    Fitted KNN imputation step.

    Attributes:
        columns: Columns to impute
        imputer: Fitted KNNImputer
        feature_names: All numeric feature names used for imputation
    """

    columns: List[str]
    imputer: Any
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using KNN.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        if self.imputer is None or len(self.columns) == 0:
            return data.copy()

        result = data.copy()

        # Transform numeric features
        numeric_data = result[self.feature_names]
        imputed_data = self.imputer.transform(numeric_data)

        # Update result with imputed values
        result[self.feature_names] = imputed_data

        return result


@dataclass
class StepImputeLinear:
    """
    Impute missing values using linear interpolation.

    Fills missing values by linear interpolation between adjacent values.
    Particularly useful for time series data.

    Attributes:
        columns: Columns to impute (None = all numeric with NA, or use selector)
        limit: Maximum number of consecutive NAs to fill (None = no limit)
        limit_direction: Direction to fill ('forward', 'backward', or 'both')
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    limit: Optional[int] = None
    limit_direction: str = "both"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeLinear":
        """
        Prepare linear interpolation step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeLinear ready to interpolate
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with missing values
            selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        return PreparedStepImputeLinear(
            columns=cols,
            limit=self.limit,
            limit_direction=self.limit_direction
        )


@dataclass
class PreparedStepImputeLinear:
    """
    Fitted linear interpolation step.

    Attributes:
        columns: Columns to impute
        limit: Maximum consecutive NAs to fill
        limit_direction: Direction to fill
    """

    columns: List[str]
    limit: Optional[int]
    limit_direction: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using linear interpolation.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col in self.columns:
            if col in result.columns:
                result[col] = result[col].interpolate(
                    method='linear',
                    limit=self.limit,
                    limit_direction=self.limit_direction
                )

        return result


@dataclass
class StepImputeBag:
    """
    Impute missing values using bagged tree models.

    Creates bagged tree models to impute missing data. For each variable
    requiring imputation, a bagged tree is created where the variable is
    the outcome and other variables are predictors.

    Attributes:
        columns: Columns to impute (None = all columns with NA, or use selector)
        impute_with: Columns to use as predictors (None = all other columns)
        trees: Number of bagged trees to use (default: 25)
        seed_val: Random seed for reproducibility
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    impute_with: Optional[List[str]] = None
    trees: int = 25
    seed_val: int = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeBag":
        """
        Fit bagged tree models for imputation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeBag with fitted models
        """
        from sklearn.ensemble import BaggingRegressor, BaggingClassifier
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

        # Set seed if provided
        random_state = self.seed_val if self.seed_val is not None else None

        # Determine columns to impute
        if self.columns is None:
            # Auto-select all columns with missing values
            selector = where(lambda s: s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        if len(cols) == 0:
            return PreparedStepImputeBag(
                columns=[],
                models={},
                impute_with_cols=[],
                feature_types={}
            )

        # Determine predictor columns
        if self.impute_with is None:
            # Use all other columns except those being imputed
            impute_with_cols = [col for col in data.columns if col not in cols]
        else:
            # Remove columns being imputed from impute_with list
            impute_with_cols = [col for col in self.impute_with if col not in cols]

        # Fit models for each column
        models = {}
        feature_types = {}

        for col in cols:
            # Determine if target is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(data[col])
            is_integer = pd.api.types.is_integer_dtype(data[col])
            feature_types[col] = {
                'is_numeric': is_numeric,
                'is_integer': is_integer,
                'dtype': data[col].dtype
            }

            # Get rows without missing values in target
            not_missing = ~data[col].isna()

            if not_missing.sum() == 0:
                # No training data available
                models[col] = None
                continue

            # Prepare training data
            y_train = data.loc[not_missing, col]
            X_train = data.loc[not_missing, impute_with_cols]

            # Create and fit bagged model
            if is_numeric:
                # Regression for numeric
                base_estimator = DecisionTreeRegressor(max_depth=None)
                model = BaggingRegressor(
                    estimator=base_estimator,
                    n_estimators=self.trees,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                # Classification for categorical
                base_estimator = DecisionTreeClassifier(max_depth=None)
                model = BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=self.trees,
                    random_state=random_state,
                    n_jobs=-1
                )

            # Fit model (handles missing values in predictors via tree-based models)
            model.fit(X_train, y_train)
            models[col] = model

        return PreparedStepImputeBag(
            columns=cols,
            models=models,
            impute_with_cols=impute_with_cols,
            feature_types=feature_types
        )


@dataclass
class PreparedStepImputeBag:
    """
    Fitted bagged tree imputation step.

    Attributes:
        columns: Columns to impute
        models: Dictionary of fitted bagged tree models
        impute_with_cols: Predictor columns used for imputation
        feature_types: Dictionary of feature type information
    """

    columns: List[str]
    models: Dict[str, Any]
    impute_with_cols: List[str]
    feature_types: Dict[str, Dict[str, Any]]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using bagged tree models.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            model = self.models.get(col)
            if model is None:
                continue

            # Find rows with missing values
            missing = result[col].isna()

            if not missing.any():
                continue

            # Prepare predictor data
            X_pred = result.loc[missing, self.impute_with_cols]

            # Make predictions
            predictions = model.predict(X_pred)

            # Convert back to original type if needed
            feature_info = self.feature_types[col]
            if feature_info['is_integer']:
                predictions = np.round(predictions).astype(feature_info['dtype'])
            elif not feature_info['is_numeric']:
                # For categorical, ensure same dtype
                predictions = predictions.astype(feature_info['dtype'])

            # Impute missing values
            result.loc[missing, col] = predictions

        return result


@dataclass
class StepImputeRoll:
    """
    Impute numeric data using a rolling window statistic.

    Substitutes missing values of numeric variables by a measure of location
    (e.g. median) within a moving window. Particularly useful for time series data.

    Attributes:
        columns: Columns to impute (None = all numeric with NA, or use selector)
        window: Size of rolling window (must be odd integer >= 3, default: 5)
        statistic: Function to compute imputed value (default: np.nanmedian)
    """

    columns: Union[None, str, List[str], Callable[[pd.DataFrame], List[str]]] = None
    window: int = 5
    statistic: Any = None

    def __post_init__(self):
        if self.statistic is None:
            self.statistic = np.nanmedian

        # Validate window size
        if self.window < 3:
            raise ValueError("window must be at least 3")
        if self.window % 2 == 0:
            raise ValueError("window must be an odd integer")

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepImputeRoll":
        """
        Prepare rolling window imputation step.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepImputeRoll ready to impute
        """
        # Determine columns to impute
        if self.columns is None:
            # Auto-select numeric columns with missing values
            selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        else:
            selector = self.columns

        cols = resolve_selector(selector, data)

        return PreparedStepImputeRoll(
            columns=cols,
            window=self.window,
            statistic=self.statistic
        )


@dataclass
class PreparedStepImputeRoll:
    """
    Fitted rolling window imputation step.

    Attributes:
        columns: Columns to impute
        window: Window size
        statistic: Statistic function
    """

    columns: List[str]
    window: int
    statistic: Any

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using rolling window statistic.

        Args:
            data: Data to transform

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            values = result[col].values.copy()
            missing_mask = pd.isna(values)

            if not missing_mask.any():
                continue

            # Get indices of missing values
            missing_indices = np.where(missing_mask)[0]

            # Calculate half window size
            half_window = self.window // 2

            # Impute each missing value
            for idx in missing_indices:
                # Calculate window bounds
                # At edges, shift window toward center
                start = max(0, idx - half_window)
                end = min(len(values), idx + half_window + 1)

                # Adjust if window is truncated at edges
                if start == 0:
                    end = min(len(values), self.window)
                elif end == len(values):
                    start = max(0, len(values) - self.window)

                # Extract window values (excluding the current missing point)
                window_values = np.concatenate([values[start:idx], values[idx+1:end]])

                # Remove NaN values from window
                window_values_clean = window_values[~pd.isna(window_values)]

                # Compute statistic if there are non-missing values
                if len(window_values_clean) > 0:
                    values[idx] = self.statistic(window_values_clean)
                # If all values in window are missing, leave as NaN

            # Update result
            result[col] = values

        return result
