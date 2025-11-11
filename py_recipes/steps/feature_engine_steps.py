"""
Feature engineering steps using feature-engine library.

Provides decision tree-based transformations and outlier handling from feature-engine.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any
import pandas as pd
import numpy as np
from ..selectors import resolve_selector, all_numeric


@dataclass
class StepDtDiscretiser:
    """
    Discretize continuous variables using decision trees.

    Uses DecisionTreeDiscretiser from feature-engine to bin numeric variables
    into discrete intervals based on decision tree splits optimized for predicting
    the outcome variable.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    columns : selector, optional
        Which columns to discretize. If None, uses all numeric columns except outcome
    cv : int, default=3
        Cross-validation folds for decision tree
    scoring : str, default='neg_mean_squared_error'
        Scoring metric for regression ('neg_mean_squared_error', 'r2', etc.)
        or classification ('roc_auc', 'f1', 'accuracy', etc.)
    regression : bool, default=True
        If True, fits regression tree. If False, fits classification tree
    random_state : int, optional
        Random state for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_dt_discretiser
    >>>
    >>> # Discretize numeric features based on relationship with outcome
    >>> rec = recipe().step_dt_discretiser(
    ...     outcome='target',
    ...     columns=['age', 'income', 'credit_score'],
    ...     cv=5
    ... )
    >>>
    >>> # Discretize all numeric predictors
    >>> rec = recipe().step_dt_discretiser(
    ...     outcome='species',
    ...     regression=False,
    ...     scoring='roc_auc'
    ... )

    Notes
    -----
    - Creates bins optimized for predicting the outcome variable
    - Bins are determined by decision tree splits
    - Works for both regression and classification outcomes
    - Preserves bin boundaries for consistent test data transformation
    """
    outcome: str
    columns: Union[None, str, List[str], Callable] = None
    cv: int = 3
    scoring: str = 'neg_mean_squared_error'
    regression: bool = True
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _discretiser: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting the decision tree discretiser."""
        if self.skip or not training:
            return self

        from feature_engine.discretisation import DecisionTreeDiscretiser

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to discretize
        if self.columns is None:
            # Use all numeric columns except outcome
            score_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        # Remove outcome if accidentally included
        score_cols = [c for c in score_cols if c != self.outcome]

        if len(score_cols) == 0:
            raise ValueError("No columns to discretize after resolving selector")

        # Fit discretiser
        discretiser = DecisionTreeDiscretiser(
            variables=score_cols,
            cv=self.cv,
            scoring=self.scoring,
            regression=self.regression,
            random_state=self.random_state
        )

        X = data[score_cols]
        discretiser.fit(X, y)

        # Create prepared instance
        prepared = replace(self)
        prepared._discretiser = discretiser
        prepared._selected_columns = score_cols
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply discretization to new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Transform
        data = data.copy()

        # Only transform columns that exist in data
        cols_to_transform = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_transform) == 0:
            return data

        X_transformed = self._discretiser.transform(data[cols_to_transform])
        data[cols_to_transform] = X_transformed

        return data


@dataclass
class StepWinsorizer:
    """
    Cap extreme values using Winsorization.

    Uses Winsorizer from feature-engine to replace extreme values with less extreme values
    at specified percentiles. This reduces the impact of outliers without removing observations.

    Parameters
    ----------
    columns : selector, optional
        Which columns to winsorize. If None, uses all numeric columns
    capping_method : str, default='iqr'
        Method for detecting outliers:
        - 'iqr': Interquartile range method
        - 'gaussian': Mean ± N standard deviations
        - 'quantiles': Fixed quantiles
    tail : str, default='both'
        Which tail to cap: 'right', 'left', or 'both'
    fold : float, default=1.5
        For 'iqr' method, multiplier for IQR (1.5 = standard outlier detection)
        For 'gaussian' method, number of standard deviations
    quantiles : tuple, optional
        For 'quantiles' method: (lower_quantile, upper_quantile)
        Note: upper_quantile must equal (1 - lower_quantile) due to feature_engine API
        Example: (0.05, 0.95) caps at 5th and 95th percentiles
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_winsorizer
    >>>
    >>> # Winsorize using IQR method (cap at Q1 - 1.5*IQR and Q3 + 1.5*IQR)
    >>> rec = recipe().step_winsorizer(
    ...     columns=['income', 'age'],
    ...     capping_method='iqr',
    ...     fold=1.5
    ... )
    >>>
    >>> # Winsorize using quantiles (cap at 5th and 95th percentile)
    >>> rec = recipe().step_winsorizer(
    ...     capping_method='quantiles',
    ...     quantiles=(0.05, 0.95)
    ... )
    >>>
    >>> # Winsorize only upper tail using gaussian method
    >>> rec = recipe().step_winsorizer(
    ...     columns=['score'],
    ...     capping_method='gaussian',
    ...     tail='right',
    ...     fold=3.0  # Mean + 3 std
    ... )

    Notes
    -----
    - Preserves number of observations (unlike outlier removal)
    - Learns capping values from training data
    - Applies same capping to test data for consistency
    """
    columns: Union[None, str, List[str], Callable] = None
    capping_method: str = 'iqr'
    tail: str = 'both'
    fold: float = 1.5
    quantiles: Optional[tuple] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _winsorizer: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_methods = ['iqr', 'gaussian', 'quantiles']
        if self.capping_method not in valid_methods:
            raise ValueError(f"capping_method must be one of {valid_methods}, got {self.capping_method}")

        valid_tails = ['right', 'left', 'both']
        if self.tail not in valid_tails:
            raise ValueError(f"tail must be one of {valid_tails}, got {self.tail}")

        if self.capping_method == 'quantiles' and self.quantiles is None:
            raise ValueError("quantiles must be specified when capping_method='quantiles'")

        if self.capping_method == 'quantiles' and self.quantiles is not None:
            if len(self.quantiles) != 2:
                raise ValueError(f"quantiles must be a tuple of 2 values, got {len(self.quantiles)}")
            lower, upper = self.quantiles
            expected_upper = 1 - lower
            if abs(upper - expected_upper) > 1e-6:
                raise ValueError(
                    f"quantiles must be symmetric: upper must equal (1 - lower). "
                    f"Got ({lower}, {upper}), expected ({lower}, {expected_upper}). "
                    f"This is required by the feature_engine Winsorizer API."
                )

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting the winsorizer."""
        if self.skip or not training:
            return self

        from feature_engine.outliers import Winsorizer

        # Resolve columns
        if self.columns is None:
            score_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        if len(score_cols) == 0:
            raise ValueError("No columns to winsorize after resolving selector")

        # Fit winsorizer
        if self.capping_method == 'quantiles':
            # For quantiles method, fold should be a single float (lower quantile)
            # Upper quantile is automatically calculated as 1 - fold
            # E.g., fold=0.05 → caps at 5th and 95th percentiles
            winsorizer = Winsorizer(
                capping_method='quantiles',
                tail=self.tail,
                fold=self.quantiles[0],  # Use lower quantile from tuple
                variables=score_cols
            )
        else:
            winsorizer = Winsorizer(
                capping_method=self.capping_method,
                tail=self.tail,
                fold=self.fold,
                variables=score_cols
            )

        winsorizer.fit(data[score_cols])

        # Create prepared instance
        prepared = replace(self)
        prepared._winsorizer = winsorizer
        prepared._selected_columns = score_cols
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Transform
        data = data.copy()

        # Only transform columns that exist in data
        cols_to_transform = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_transform) == 0:
            return data

        X_transformed = self._winsorizer.transform(data[cols_to_transform])
        data[cols_to_transform] = X_transformed

        return data


@dataclass
class StepOutlierTrimmer:
    """
    Remove observations with outliers.

    Uses OutlierTrimmer from feature-engine to remove rows containing outliers
    in specified columns. Unlike Winsorizer, this removes entire observations.

    Parameters
    ----------
    columns : selector, optional
        Which columns to check for outliers. If None, uses all numeric columns
    capping_method : str, default='iqr'
        Method for detecting outliers:
        - 'iqr': Interquartile range method
        - 'gaussian': Mean ± N standard deviations
        - 'quantiles': Fixed quantiles
    tail : str, default='both'
        Which tail to check: 'right', 'left', or 'both'
    fold : float, default=1.5
        For 'iqr' method, multiplier for IQR
        For 'gaussian' method, number of standard deviations
    quantiles : tuple, optional
        For 'quantiles' method: (lower_quantile, upper_quantile)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_outlier_trimmer
    >>>
    >>> # Remove rows with outliers using IQR method
    >>> rec = recipe().step_outlier_trimmer(
    ...     columns=['income', 'age'],
    ...     capping_method='iqr',
    ...     fold=1.5
    ... )
    >>>
    >>> # Remove rows with extreme upper values only
    >>> rec = recipe().step_outlier_trimmer(
    ...     columns=['score'],
    ...     capping_method='gaussian',
    ...     tail='right',
    ...     fold=3.0
    ... )

    Notes
    -----
    - Removes observations, reducing sample size
    - Use with caution on test data (may remove valid observations)
    - Consider step_winsorizer as alternative that preserves observations

    Warnings
    --------
    This step modifies the number of rows in the DataFrame, which can cause
    index misalignment issues. Use early in the recipe pipeline.
    """
    columns: Union[None, str, List[str], Callable] = None
    capping_method: str = 'iqr'
    tail: str = 'both'
    fold: float = 1.5
    quantiles: Optional[tuple] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _trimmer: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        valid_methods = ['iqr', 'gaussian', 'quantiles']
        if self.capping_method not in valid_methods:
            raise ValueError(f"capping_method must be one of {valid_methods}, got {self.capping_method}")

        valid_tails = ['right', 'left', 'both']
        if self.tail not in valid_tails:
            raise ValueError(f"tail must be one of {valid_tails}, got {self.tail}")

        if self.capping_method == 'quantiles' and self.quantiles is None:
            raise ValueError("quantiles must be specified when capping_method='quantiles'")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting the outlier trimmer."""
        if self.skip or not training:
            return self

        from feature_engine.outliers import OutlierTrimmer

        # Resolve columns
        if self.columns is None:
            score_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        if len(score_cols) == 0:
            raise ValueError("No columns to check for outliers after resolving selector")

        # Fit trimmer
        if self.capping_method == 'quantiles':
            trimmer = OutlierTrimmer(
                capping_method='quantiles',
                tail=self.tail,
                fold=self.quantiles,
                variables=score_cols
            )
        else:
            trimmer = OutlierTrimmer(
                capping_method=self.capping_method,
                tail=self.tail,
                fold=self.fold,
                variables=score_cols
            )

        trimmer.fit(data[score_cols])

        # Create prepared instance
        prepared = replace(self)
        prepared._trimmer = trimmer
        prepared._selected_columns = score_cols
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Only check columns that exist in data
        cols_to_check = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_check) == 0:
            return data

        # Transform (removes rows with outliers)
        # OutlierTrimmer only works with the columns it was fitted on
        data_transformed = self._trimmer.transform(data[cols_to_check])

        # Get indices of kept rows
        kept_indices = data_transformed.index

        # Return full data with only kept rows
        return data.loc[kept_indices]


@dataclass
class StepDtFeatures:
    """
    Create new features using decision trees.

    Uses DecisionTreeFeatures from feature-engine to create engineered features
    by combining existing features through decision tree models optimized for
    predicting the outcome variable.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    columns : selector, optional
        Which columns to use for feature creation. If None, uses all numeric columns except outcome
    features_to_combine : int or None, default=None
        Number of features to combine to create new features.
        If None, creates all possible combinations (can be computationally expensive).
        If int, creates that many feature combinations.
    cv : int, default=3
        Cross-validation folds for decision tree
    scoring : str, default='neg_mean_squared_error'
        Scoring metric for regression or classification
    regression : bool, default=True
        If True, fits regression tree. If False, fits classification tree
    random_state : int, optional
        Random state for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_dt_features
    >>>
    >>> # Create decision tree features for regression
    >>> rec = recipe().step_dt_features(
    ...     outcome='price',
    ...     features_to_combine=5,
    ...     cv=5
    ... )
    >>>
    >>> # Create features for classification
    >>> rec = recipe().step_dt_features(
    ...     outcome='species',
    ...     columns=['sepal_length', 'sepal_width'],
    ...     regression=False,
    ...     features_to_combine=3
    ... )

    Notes
    -----
    - Creates new features by combining existing ones via decision trees
    - New features are binary indicators of tree leaf membership
    - Useful for capturing non-linear relationships
    - Can increase feature count significantly
    """
    outcome: str
    columns: Union[None, str, List[str], Callable] = None
    features_to_combine: Union[int, None] = None
    cv: int = 3
    scoring: str = 'neg_mean_squared_error'
    regression: bool = True
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _dt_features: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _new_feature_names: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting decision tree features."""
        if self.skip or not training:
            return self

        from feature_engine.creation import DecisionTreeFeatures

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        # Remove outcome if accidentally included
        score_cols = [c for c in score_cols if c != self.outcome]

        if len(score_cols) == 0:
            raise ValueError("No columns for feature creation after resolving selector")

        # Fit decision tree features
        dt_features = DecisionTreeFeatures(
            variables=score_cols,
            features_to_combine=self.features_to_combine,
            cv=self.cv,
            scoring=self.scoring,
            regression=self.regression,
            random_state=self.random_state
        )

        X = data[score_cols]
        dt_features.fit(X, y)

        # Get new feature names
        # DecisionTreeFeatures creates columns like 'tree(brent)', "tree(['brent', 'dubai'])", etc.
        sample_transform = dt_features.transform(X.head(1))
        new_cols = [c for c in sample_transform.columns if c not in score_cols]

        # Rename columns to avoid Patsy interpreting "tree(...)" as a function call
        # Examples:
        #   'tree(brent)' → 'dt_brent'
        #   "tree(['brent', 'dubai'])" → 'dt_brent_dubai'
        renamed_cols = []
        for col in new_cols:
            if col.startswith('tree('):
                # Extract content between parentheses
                content = col[5:-1]  # Remove 'tree(' and ')'

                # Handle list notation like "['brent', 'dubai']"
                if content.startswith('[') and content.endswith(']'):
                    # Remove brackets and quotes, split on comma
                    content = content[1:-1]  # Remove [ and ]
                    features = [f.strip().strip("'\"") for f in content.split(',')]
                    safe_name = 'dt_' + '_'.join(features)
                else:
                    # Single feature like 'brent'
                    feature = content.strip("'\"")
                    safe_name = f'dt_{feature}'

                renamed_cols.append(safe_name)
            else:
                # Fallback: just replace problematic characters
                safe_name = col.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
                safe_name = safe_name.replace(',', '_').replace(' ', '_').replace("'", '')
                renamed_cols.append(safe_name)

        # Create prepared instance
        prepared = replace(self)
        prepared._dt_features = dt_features
        prepared._selected_columns = score_cols
        prepared._new_feature_names = renamed_cols  # Store renamed versions
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create decision tree features in new data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Only transform columns that exist in data
        cols_to_use = [c for c in self._selected_columns if c in data.columns]

        if len(cols_to_use) == 0:
            return data

        # Transform - DecisionTreeFeatures creates columns with "_tree_" pattern
        X_transformed = self._dt_features.transform(data[cols_to_use])

        # Get original column names from DecisionTreeFeatures
        original_new_cols = [c for c in X_transformed.columns if c not in cols_to_use]

        # Add new features to data with renamed columns to avoid Patsy errors
        data = data.copy()
        for orig_col, renamed_col in zip(original_new_cols, self._new_feature_names):
            if orig_col in X_transformed.columns:
                data[renamed_col] = X_transformed[orig_col].values

        return data


@dataclass
class StepSelectSmartCorr:
    """
    Select features using smart correlation selection.

    Uses SmartCorrelatedSelection from feature-engine to remove correlated features
    intelligently by keeping the one most correlated with the outcome variable.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    columns : selector, optional
        Which columns to consider. If None, uses all numeric columns except outcome
    threshold : float, default=0.8
        Correlation threshold above which features are considered correlated
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    selection_method : str, default='variance'
        How to select among correlated features:
        - 'variance': Keep feature with highest variance
        - 'cardinality': For categorical, keep one with most categories
        - 'model_performance': Keep one most correlated with outcome (requires outcome)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_select_smart_corr
    >>>
    >>> # Remove highly correlated features, keeping most predictive
    >>> rec = recipe().step_select_smart_corr(
    ...     outcome='target',
    ...     threshold=0.9,
    ...     selection_method='model_performance'
    ... )
    >>>
    >>> # Remove correlated features based on variance
    >>> rec = recipe().step_select_smart_corr(
    ...     columns=['feature1', 'feature2', 'feature3'],
    ...     threshold=0.85,
    ...     method='spearman',
    ...     selection_method='variance'
    ... )

    Notes
    -----
    - Smarter than simple correlation filtering
    - Considers relationship with outcome variable
    - Reduces multicollinearity while preserving predictive power
    """
    outcome: str
    columns: Union[None, str, List[str], Callable] = None
    threshold: float = 0.8
    method: str = 'pearson'
    selection_method: str = 'variance'
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selector: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _features_to_drop: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if not (0 < self.threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

        valid_methods = ['pearson', 'spearman', 'kendall']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

        valid_selection = ['variance', 'cardinality', 'model_performance']
        if self.selection_method not in valid_selection:
            raise ValueError(f"selection_method must be one of {valid_selection}, got {self.selection_method}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by identifying correlated features to drop."""
        if self.skip or not training:
            return self

        from feature_engine.selection import SmartCorrelatedSelection

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        # Remove outcome if accidentally included
        score_cols = [c for c in score_cols if c != self.outcome]

        if len(score_cols) == 0:
            raise ValueError("No columns for correlation selection after resolving selector")

        # Fit selector
        selector = SmartCorrelatedSelection(
            variables=score_cols,
            method=self.method,
            threshold=self.threshold,
            selection_method=self.selection_method
        )

        X = data[score_cols + [self.outcome]]
        selector.fit(X)

        # Get features to drop
        features_to_drop = selector.features_to_drop_

        # Create prepared instance
        prepared = replace(self)
        prepared._selector = selector
        prepared._selected_columns = score_cols
        prepared._features_to_drop = list(features_to_drop)
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features from data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Drop correlated features
        data = data.copy()
        cols_to_drop = [c for c in self._features_to_drop if c in data.columns]

        if len(cols_to_drop) > 0:
            data = data.drop(columns=cols_to_drop)

        return data


@dataclass
class StepSelectPsi:
    """
    Remove features with high Population Stability Index (PSI).

    Uses DropHighPSIFeatures from feature-engine to identify and remove features
    with high PSI, indicating distribution shifts between training and test data.

    Parameters
    ----------
    columns : selector, optional
        Which columns to check. If None, uses all numeric columns
    threshold : float, default=0.25
        PSI threshold above which features are dropped
        - PSI < 0.1: No significant change
        - PSI 0.1-0.25: Moderate change
        - PSI > 0.25: Significant change (drop)
    bins : int, default=10
        Number of bins for discretization when calculating PSI
    strategy : str, default='equal_frequency'
        Binning strategy: 'equal_frequency' or 'equal_width'
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_select_psi
    >>>
    >>> # Remove features with significant distribution shift
    >>> rec = recipe().step_select_psi(
    ...     threshold=0.25,
    ...     bins=10
    ... )
    >>>
    >>> # More conservative threshold
    >>> rec = recipe().step_select_psi(
    ...     columns=['feature1', 'feature2'],
    ...     threshold=0.15,
    ...     strategy='equal_width'
    ... )

    Notes
    -----
    - Requires a reference dataset (typically training data stored during prep)
    - PSI measures distribution shift between train and new data
    - High PSI indicates feature may not be reliable for prediction
    - Useful for detecting data drift in production

    Warnings
    --------
    This step requires access to both training and test data during prep phase
    to calculate PSI. The implementation stores training data distributions.
    """
    columns: Union[None, str, List[str], Callable] = None
    threshold: float = 0.25
    bins: int = 10
    strategy: str = 'equal_frequency'
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _psi_selector: Any = field(default=None, init=False, repr=False)
    _selected_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _features_to_drop: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")

        valid_strategies = ['equal_frequency', 'equal_width']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {self.strategy}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by calculating PSI and identifying high-PSI features."""
        if self.skip or not training:
            return self

        from feature_engine.selection import DropHighPSIFeatures

        # Resolve columns
        if self.columns is None:
            score_cols = list(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = resolve_selector(self.columns, data)
        else:
            score_cols = list(self.columns)

        if len(score_cols) == 0:
            raise ValueError("No columns for PSI selection after resolving selector")

        # Fit PSI selector
        # Note: feature-engine's DropHighPSIFeatures requires split_frac parameter
        # We use the training data as reference
        psi_selector = DropHighPSIFeatures(
            variables=score_cols,
            split_frac=0.5,  # Split training data to estimate PSI
            threshold=self.threshold,
            bins=self.bins,
            strategy=self.strategy
        )

        psi_selector.fit(data[score_cols])

        # Get features to drop
        features_to_drop = psi_selector.features_to_drop_

        # Create prepared instance
        prepared = replace(self)
        prepared._psi_selector = psi_selector
        prepared._selected_columns = score_cols
        prepared._features_to_drop = list(features_to_drop)
        prepared._is_prepared = True

        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove high-PSI features from data."""
        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        if self.skip:
            return data

        # Drop high-PSI features
        data = data.copy()
        cols_to_drop = [c for c in self._features_to_drop if c in data.columns]

        if len(cols_to_drop) > 0:
            data = data.drop(columns=cols_to_drop)

        return data
