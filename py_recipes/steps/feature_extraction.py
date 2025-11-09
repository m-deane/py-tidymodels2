"""
SAFE: Surrogate Assisted Feature Extraction for interpretable ML models.

Implements the SAFE (Surrogate Assisted Feature Extraction) methodology from:
SAFE library: https://github.com/ModelOriented/SAFE
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

# Optional dependencies
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import ward, cut_tree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False


@dataclass
class StepSafe:
    """
    Surrogate Assisted Feature Extraction (SAFE) for interpretable model building.

    SAFE uses a complex surrogate model to guide feature transformation. It creates
    interpretable features by:
    - For numeric variables: Detecting changepoints in partial dependence plots
    - For categorical variables: Merging levels with similar model responses

    The transformed features can then be used in simpler, interpretable models while
    retaining information from the complex surrogate model.

    Parameters
    ----------
    surrogate_model : fitted model
        Pre-fitted surrogate model (e.g., GradientBoosting, RandomForest).
        Must implement predict() for regression or predict_proba() for classification.
    outcome : str
        Name of the outcome variable (required for supervised transformation)
    penalty : float, default=3.0
        Penalty for adding new changepoints. Higher values = fewer intervals.
        Typical range: 0.1-10.0
    pelt_model : {'l2', 'l1', 'rbf'}, default='l2'
        Cost function for Pelt changepoint detection algorithm
    no_changepoint_strategy : {'median', 'drop'}, default='median'
        Strategy when no changepoint detected:
        - 'median': Create one split at median value
        - 'drop': Remove feature from output
    keep_original_cols : bool, default=False
        Whether to keep original columns alongside transformed features
    top_n : int, optional
        If specified, select only top N most important transformed features.
        Feature importance based on variance explained in surrogate model predictions.
    grid_resolution : int, default=1000
        Number of points for partial dependence plot grid
    skip : bool, default=False
        Skip this step during prep/bake
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>>
    >>> # Fit surrogate model
    >>> surrogate = GradientBoostingRegressor(n_estimators=100)
    >>> surrogate.fit(train_data.drop('target', axis=1), train_data['target'])
    >>>
    >>> # Create recipe with SAFE transformation
    >>> rec = recipe(data, "target ~ .").step_safe(
    ...     surrogate_model=surrogate,
    ...     outcome='target',
    ...     penalty=3.0,
    ...     keep_original_cols=False
    ... )
    >>>
    >>> # Select top 10 most important SAFE features
    >>> rec = recipe(data, "target ~ .").step_safe(
    ...     surrogate_model=surrogate,
    ...     outcome='target',
    ...     top_n=10
    ... )

    Notes
    -----
    - Requires ruptures, scipy, and kneed packages
    - Surrogate model must be pre-fitted before creating recipe
    - Numeric features transformed via changepoint detection
    - Categorical features transformed via hierarchical clustering
    - Output is one-hot encoded with p-1 scheme
    - First level represented by zeros in all dummy columns

    References
    ----------
    SAFE library: https://github.com/ModelOriented/SAFE
    """
    surrogate_model: Any
    outcome: str
    penalty: float = 3.0
    pelt_model: Literal['l2', 'l1', 'rbf'] = 'l2'
    no_changepoint_strategy: Literal['median', 'drop'] = 'median'
    keep_original_cols: bool = False
    top_n: Optional[int] = None
    grid_resolution: int = 1000
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _variables: List[Any] = field(default_factory=list, init=False, repr=False)
    _original_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _feature_importances: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Check dependencies
        if not RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures package required for step_safe(). "
                "Install with: pip install ruptures"
            )
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy package required for step_safe(). "
                "Install with: pip install scipy"
            )
        if not KNEED_AVAILABLE:
            raise ImportError(
                "kneed package required for step_safe(). "
                "Install with: pip install kneed"
            )

        # Validate parameters
        if self.penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {self.penalty}")

        if self.pelt_model not in ['l2', 'l1', 'rbf']:
            raise ValueError(
                f"pelt_model must be 'l2', 'l1', or 'rbf', got '{self.pelt_model}'"
            )

        if self.no_changepoint_strategy not in ['median', 'drop']:
            raise ValueError(
                f"no_changepoint_strategy must be 'median' or 'drop', "
                f"got '{self.no_changepoint_strategy}'"
            )

        if self.grid_resolution < 100:
            raise ValueError(f"grid_resolution must be >= 100, got {self.grid_resolution}")

        if self.top_n is not None and self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")

        # Check if surrogate model is fitted
        if not self._is_surrogate_fitted():
            raise ValueError(
                "surrogate_model must be pre-fitted before creating step_safe(). "
                "Fit the model first: surrogate_model.fit(X, y)"
            )

    def _sanitize_threshold(self, value: float) -> str:
        """
        Convert threshold value to patsy-friendly string.

        Similar to step_splitwise sanitization for formula compatibility.

        Parameters
        ----------
        value : float
            Threshold value

        Returns
        -------
        str
            Sanitized string safe for column names and formulas
        """
        # Format to 2 decimal places
        formatted = f"{value:.2f}"

        # Replace negative sign with 'm' (minus)
        sanitized = formatted.replace('-', 'm')

        # Replace decimal point with 'p' (point)
        sanitized = sanitized.replace('.', 'p')

        return sanitized

    def _is_surrogate_fitted(self) -> bool:
        """Check if surrogate model is fitted."""
        from sklearn.exceptions import NotFittedError

        try:
            # Try to check fitted attributes (sklearn convention)
            # Most sklearn models have these when fitted
            if hasattr(self.surrogate_model, 'n_features_in_'):
                return True

            # Alternative: try predict on tiny dummy data
            # This will raise NotFittedError if not fitted
            dummy_data = np.zeros((1, 1))
            try:
                self.surrogate_model.predict(dummy_data)
                return True
            except (NotFittedError, ValueError):
                # ValueError can occur if model expects different n_features
                # But NotFittedError means definitely not fitted
                return False
        except:
            return False

    def prep(self, data: pd.DataFrame, training: bool = True):
        """
        Prepare step by learning SAFE transformations from training data.

        Parameters
        ----------
        data : DataFrame
            Training data containing predictors and outcome
        training : bool
            Whether this is training data

        Returns
        -------
        self
            Modified step with transformation metadata stored
        """
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data columns")

        # Prepare data without outcome
        X = data.drop(columns=[self.outcome]).copy()
        self._original_columns = list(X.columns)

        # Detect categorical vs numeric columns
        categorical_cols = []
        numeric_cols = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # One-hot encode categorical variables for surrogate model input
        X_for_model = X.copy()
        categorical_dummies = {}

        for col in categorical_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            categorical_dummies[col] = {
                'levels': list(X[col].unique()),
                'dummy_names': list(dummies.columns)
            }

            # Replace categorical column with dummies
            col_idx = X_for_model.columns.get_loc(col)
            X_for_model = pd.concat([
                X_for_model.iloc[:, :col_idx],
                dummies,
                X_for_model.iloc[:, col_idx+1:]
            ], axis=1)

        # Create variable transformations
        for idx, col in enumerate(self._original_columns):
            if col in numeric_cols:
                var = self._fit_numeric_variable(
                    col, idx, X_for_model, X[col]
                )
                self._variables.append(var)
            else:  # categorical
                dummy_info = categorical_dummies[col]
                var = self._fit_categorical_variable(
                    col, idx, X_for_model, X[col],
                    dummy_info['dummy_names'],
                    dummy_info['levels']
                )
                self._variables.append(var)

        # Compute feature importances for transformed features
        self._compute_feature_importances(X_for_model)

        # Select top N features if specified
        if self.top_n is not None:
            all_features = []
            for var in self._variables:
                if var['new_names']:
                    all_features.extend(var['new_names'])

            # Sort by importance
            sorted_features = sorted(
                all_features,
                key=lambda f: self._feature_importances.get(f, 0),
                reverse=True
            )

            self._selected_features = sorted_features[:self.top_n]

        self._is_prepared = True
        return self

    def _fit_numeric_variable(
        self, col_name: str, col_idx: int, X_model: pd.DataFrame, X_original: pd.Series
    ) -> Dict[str, Any]:
        """
        Fit transformation for numeric variable using changepoint detection.

        Parameters
        ----------
        col_name : str
            Variable name
        col_idx : int
            Original column index
        X_model : DataFrame
            Data for surrogate model (with categoricals one-hot encoded)
        X_original : Series
            Original numeric variable values

        Returns
        -------
        dict
            Variable transformation metadata
        """
        # Get partial dependence plot
        pdp, axes = self._get_partial_dependence_numeric(col_name, X_model, X_original)

        # Detect changepoints using Pelt algorithm
        algo = rpt.Pelt(model=self.pelt_model).fit(pdp)
        changepoint_indices = algo.predict(pen=self.penalty)

        # Convert indices to actual values
        changepoint_values = [axes[i] for i in changepoint_indices[:-1]]

        # Handle no changepoints case
        if not changepoint_values:
            if self.no_changepoint_strategy == 'median':
                changepoint_values = [float(np.median(X_original))]
            elif self.no_changepoint_strategy == 'drop':
                # No changepoints, will drop feature
                return {
                    'type': 'numeric',
                    'original_name': col_name,
                    'original_index': col_idx,
                    'changepoint_values': [],
                    'new_names': []
                }

        # Create interval names (patsy-compatible)
        # Sanitize threshold values for use in column names
        changepoint_names = [self._sanitize_threshold(v) for v in changepoint_values] + ['Inf']
        new_names = [
            f"{col_name}_{changepoint_names[i]}_to_{changepoint_names[i+1]}"
            for i in range(len(changepoint_names) - 1)
        ]

        return {
            'type': 'numeric',
            'original_name': col_name,
            'original_index': col_idx,
            'changepoint_values': changepoint_values,
            'new_names': new_names
        }

    def _fit_categorical_variable(
        self, col_name: str, col_idx: int, X_model: pd.DataFrame,
        X_original: pd.Series, dummy_names: List[str], levels: List[str]
    ) -> Dict[str, Any]:
        """
        Fit transformation for categorical variable using hierarchical clustering.

        Parameters
        ----------
        col_name : str
            Variable name
        col_idx : int
            Original column index
        X_model : DataFrame
            Data for surrogate model
        X_original : Series
            Original categorical variable values
        dummy_names : list
            One-hot encoded dummy column names
        levels : list
            Unique categorical levels

        Returns
        -------
        dict
            Variable transformation metadata
        """
        # Get partial dependence for each category level
        pdp, axes = self._get_partial_dependence_categorical(
            col_name, X_model, dummy_names
        )

        # Hierarchical clustering with Ward linkage
        if pdp.ndim == 1:
            pdp_array = pdp.reshape(len(pdp), 1)
        else:
            pdp_array = pdp

        Z = ward(pdp_array)

        # Determine optimal number of clusters
        clusters = None
        new_names = []

        if pdp.shape[0] == 3:
            # For 3 levels, use first linkage
            clusters = cut_tree(Z, height=Z[0, 2] - np.finfo(float).eps)
            new_names = self._create_categorical_names(
                col_name, clusters, dummy_names, levels
            )
        elif pdp.shape[0] > 3:
            # Use KneeLocator for optimal clusters
            kneed = KneeLocator(
                range(Z.shape[0]), Z[:, 2],
                direction='increasing', curve='convex'
            )

            if kneed.knee is not None:
                clusters = cut_tree(Z, height=Z[kneed.knee + 1, 2] - np.finfo(float).eps)
                new_names = self._create_categorical_names(
                    col_name, clusters, dummy_names, levels
                )

        return {
            'type': 'categorical',
            'original_name': col_name,
            'original_index': col_idx,
            'dummy_names': dummy_names,
            'levels': sorted(levels),
            'clusters': clusters,
            'Z': Z,
            'new_names': new_names[1:] if new_names else []  # Drop first (base level)
        }

    def _get_partial_dependence_numeric(
        self, col_name: str, X: pd.DataFrame, X_original: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence plot for numeric variable.

        Parameters
        ----------
        col_name : str
            Variable name
        X : DataFrame
            Full feature matrix for surrogate model
        X_original : Series
            Original numeric variable (before one-hot encoding)

        Returns
        -------
        pdp : array
            Partial dependence values
        axes : array
            Grid points
        """
        # Create grid from min to max
        points = np.linspace(
            X_original.min(),
            X_original.max(),
            self.grid_resolution
        )

        pdp = []

        for point in points:
            # Create a copy and set all rows to this value
            X_copy = X.copy()
            if col_name in X_copy.columns:
                X_copy[col_name] = point

            # Ensure column order matches what model expects (sklearn requirement)
            if hasattr(self.surrogate_model, 'feature_names_in_'):
                expected_cols = self.surrogate_model.feature_names_in_
                # Reorder columns to match training order
                X_copy = X_copy[expected_cols]

            # Get predictions
            if hasattr(self.surrogate_model, 'predict_proba'):
                preds = self.surrogate_model.predict_proba(X_copy)
            else:
                preds = self.surrogate_model.predict(X_copy)

            # Average predictions
            pdp.append(np.mean(preds, axis=0))

        return np.array(pdp), points

    def _get_partial_dependence_categorical(
        self, col_name: str, X: pd.DataFrame, dummy_names: List[str]
    ) -> tuple[np.ndarray, List[str]]:
        """
        Compute partial dependence for categorical variable.

        Parameters
        ----------
        col_name : str
            Variable name
        X : DataFrame
            Full feature matrix
        dummy_names : list
            One-hot encoded dummy column names

        Returns
        -------
        pdp : array
            Partial dependence for each level
        axes : list
            Level names
        """
        pdp = []
        axes = ['base']

        # Base level (all dummies = 0)
        X_copy = X.copy()
        X_copy[dummy_names] = 0

        # Ensure column order matches what model expects
        if hasattr(self.surrogate_model, 'feature_names_in_'):
            expected_cols = self.surrogate_model.feature_names_in_
            X_copy = X_copy[expected_cols]

        if hasattr(self.surrogate_model, 'predict_proba'):
            preds = self.surrogate_model.predict_proba(X_copy)
        else:
            preds = self.surrogate_model.predict(X_copy)
        pdp.append(np.mean(preds, axis=0))

        # Each dummy level
        for dummy in dummy_names:
            axes.append(dummy)
            X_copy = X.copy()
            X_copy[dummy_names] = 0
            X_copy[dummy] = 1

            # Ensure column order matches
            if hasattr(self.surrogate_model, 'feature_names_in_'):
                X_copy = X_copy[expected_cols]

            if hasattr(self.surrogate_model, 'predict_proba'):
                preds = self.surrogate_model.predict_proba(X_copy)
            else:
                preds = self.surrogate_model.predict(X_copy)
            pdp.append(np.mean(preds, axis=0))

        return np.array(pdp), axes

    def _create_categorical_names(
        self, col_name: str, clusters: np.ndarray,
        dummy_names: List[str], levels: List[str]
    ) -> List[str]:
        """Create new categorical variable names from clusters."""
        new_names = []

        for cluster_id in range(len(np.unique(clusters))):
            names = []
            for idx, c_val in enumerate(clusters):
                if c_val == cluster_id:
                    if idx == 0:
                        names.append('base')
                    else:
                        # Extract level name from dummy
                        level_name = dummy_names[idx - 1][len(col_name) + 1:]
                        names.append(level_name)

            # Create patsy-safe names (replace spaces, special chars)
            safe_name = f"{col_name}_{'_'.join(names)}"
            safe_name = safe_name.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            new_names.append(safe_name)

        return new_names

    def _compute_feature_importances(self, X: pd.DataFrame):
        """
        Compute feature importances for transformed features.

        Uses variance explained in surrogate model predictions as proxy
        for feature importance.
        """
        # Get all transformed feature names
        all_features = []
        for var in self._variables:
            if var['new_names']:
                all_features.extend(var['new_names'])

        # Initialize importances
        for feat in all_features:
            self._feature_importances[feat] = 0.0

        # If surrogate has feature_importances_ attribute, use it
        if hasattr(self.surrogate_model, 'feature_importances_'):
            # Map original features to transformed features
            # This is an approximation
            importances = self.surrogate_model.feature_importances_

            # Distribute importance across transformed features
            for var in self._variables:
                if var['new_names']:
                    # Equal distribution for simplicity
                    importance_per_feature = 1.0 / len(var['new_names'])
                    for feat in var['new_names']:
                        self._feature_importances[feat] = importance_per_feature
        else:
            # Fallback: uniform importances
            for feat in all_features:
                self._feature_importances[feat] = 1.0 / len(all_features)

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SAFE transformations to new data.

        Parameters
        ----------
        data : DataFrame
            Data to transform (train or test)

        Returns
        -------
        DataFrame
            Transformed data with SAFE features
        """
        if self.skip or not self._is_prepared:
            return data.copy()

        # Extract original columns from data
        X = data[[col for col in self._original_columns if col in data.columns]].copy()

        # Transform each variable
        transformed_dfs = []

        for var in self._variables:
            col_name = var['original_name']

            if col_name not in X.columns:
                continue

            if var['type'] == 'numeric':
                transformed = self._transform_numeric_variable(var, X[col_name])
            else:  # categorical
                transformed = self._transform_categorical_variable(var, X[col_name])

            if transformed is not None and not transformed.empty:
                transformed_dfs.append(transformed)

        # Combine all transformed features
        if transformed_dfs:
            result = pd.concat(transformed_dfs, axis=1).reset_index(drop=True)
        else:
            result = pd.DataFrame(index=range(len(data)))

        # Filter to selected features if top_n specified
        if self.top_n is not None and self._selected_features:
            available_features = [f for f in self._selected_features if f in result.columns]
            result = result[available_features]

        # Always preserve outcome column if present (needed for workflows)
        if self.outcome in data.columns:
            result[self.outcome] = data[self.outcome].reset_index(drop=True)

        # Keep original predictor columns if requested
        if self.keep_original_cols:
            # Add original columns (excluding outcome which we already added)
            original_predictors = [col for col in data.columns if col != self.outcome]
            for col in original_predictors:
                if col not in result.columns and col in data.columns:
                    result[col] = data[col].reset_index(drop=True)

        return result

    def _transform_numeric_variable(
        self, var: Dict[str, Any], X_col: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Transform numeric variable based on changepoints."""
        if not var['changepoint_values'] or not var['new_names']:
            return None

        # Determine which interval each observation belongs to
        changepoints = var['changepoint_values']
        new_names = var['new_names']

        # Create interval assignments
        # Count how many changepoints each value exceeds
        interval_assignments = []
        for x in X_col:
            interval_idx = sum(1 for cp in changepoints if x >= cp)
            interval_assignments.append(interval_idx)

        # One-hot encode (p-1 encoding)
        # First interval (value < all changepoints) is the base (all zeros)
        # So we create len(changepoints) columns (one per changepoint)
        n_rows = len(interval_assignments)
        n_cols = len(changepoints)

        transformed = np.zeros((n_rows, n_cols))

        for row_idx, interval_idx in enumerate(interval_assignments):
            # interval_idx = 0 means first interval (base) → all zeros
            # interval_idx = 1 means second interval → col 0 = 1
            # interval_idx = 2 means third interval → col 1 = 1
            if interval_idx > 0:
                transformed[row_idx, interval_idx - 1] = 1

        # new_names already excludes the base interval
        return pd.DataFrame(transformed, columns=new_names)

    def _transform_categorical_variable(
        self, var: Dict[str, Any], X_col: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Transform categorical variable based on clusters."""
        if var['clusters'] is None or not var['new_names']:
            # No clustering performed, return one-hot encoded
            dummies = pd.get_dummies(X_col, prefix=var['original_name'], drop_first=True)
            return dummies

        # Apply cluster-based transformation
        clusters = var['clusters']
        levels = var['levels']
        new_names = var['new_names']

        # Create one-hot encoded version first
        dummies = pd.get_dummies(X_col, prefix=var['original_name'], drop_first=True)

        # Map to clusters
        n_rows = len(X_col)
        n_clusters = len(np.unique(clusters)) - 1  # Exclude base cluster

        transformed = np.zeros((n_rows, n_clusters))

        for row_idx in range(n_rows):
            # Check if any dummy is 1
            if row_idx < len(dummies) and dummies.iloc[row_idx].sum() > 0:
                # Find which dummy is 1
                dummy_idx = np.argmax(dummies.iloc[row_idx].values)
                # Map to cluster (offset by 1 for base level)
                cluster_id = clusters[dummy_idx + 1]
                if cluster_id > 0:
                    transformed[row_idx, cluster_id - 1] = 1

        return pd.DataFrame(transformed, columns=new_names)

    def get_transformations(self) -> Dict[str, Any]:
        """
        Get transformation metadata for all variables.

        Returns
        -------
        dict
            Transformation information for each variable
        """
        if not self._is_prepared:
            raise ValueError("Step must be prepared before accessing transformations")

        transformations = {}
        for var in self._variables:
            info = {
                'type': var['type'],
                'original_name': var['original_name']
            }

            if var['type'] == 'numeric':
                info['changepoints'] = var['changepoint_values']
                info['intervals'] = var['new_names']
            else:
                info['levels'] = var['levels']
                info['merged_levels'] = var['new_names']

            transformations[var['original_name']] = info

        return transformations

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances for transformed features.

        Returns
        -------
        DataFrame
            Feature importances sorted by importance
        """
        if not self._is_prepared:
            raise ValueError("Step must be prepared before accessing feature importances")

        importances_df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in self._feature_importances.items()
        ])

        return importances_df.sort_values('importance', ascending=False).reset_index(drop=True)
