"""
SAFE: Surrogate Assisted Feature Extraction for interpretable ML models.

Implements the SAFE (Surrogate Assisted Feature Extraction) methodology from:
SAFE library: https://github.com/ModelOriented/SAFE
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import pandas as pd
import numpy as np
import warnings
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
    feature_type : {'dummies', 'interactions', 'both'}, default='dummies'
        Type of features to create:
        - 'dummies': Binary dummy variables only (default)
        - 'interactions': Binary dummy * original feature interactions only
        - 'both': Both dummies and interactions
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
    feature_type: Literal['dummies', 'interactions', 'both'] = 'dummies'
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

        if self.feature_type not in ['dummies', 'interactions', 'both']:
            raise ValueError(
                f"feature_type must be 'dummies', 'interactions', or 'both', "
                f"got '{self.feature_type}'"
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
        # Create transformed dataset for importance calculation
        X_transformed = self._create_transformed_dataset(X)
        outcome_series = data[self.outcome]
        self._compute_feature_importances(X_transformed, outcome_series)

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

            # If feature_type includes interactions, also add interaction column names
            # Interactions have pattern: "dummy_name_x_original_name"
            if self.feature_type in ['interactions', 'both']:
                expanded_features = []
                for feat in self._selected_features:
                    # Add the base feature (dummy or interaction)
                    expanded_features.append(feat)

                    # For 'interactions' mode, replace dummy with interaction
                    # For 'both' mode, add both dummy and interaction
                    if self.feature_type == 'both':
                        # Find the original variable name for this feature
                        # Feature format: "varname_<interval>" or "varname_cp_N"
                        for var in self._variables:
                            if feat in var['new_names']:
                                original_name = var['original_name']
                                interaction_name = f"{feat}_x_{original_name}"
                                expanded_features.append(interaction_name)
                                break
                    elif self.feature_type == 'interactions':
                        # Remove dummy, add interaction instead
                        expanded_features.remove(feat)
                        for var in self._variables:
                            if feat in var['new_names']:
                                original_name = var['original_name']
                                interaction_name = f"{feat}_x_{original_name}"
                                expanded_features.append(interaction_name)
                                break

                self._selected_features = expanded_features

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

    def _compute_feature_importances(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """
        Compute feature importances using various methods on transformed features.

        Parameters
        ----------
        X_transformed : DataFrame
            Transformed features (SAFE binary indicators or interactions)
        outcome : Series
            Target variable for supervised importance calculation

        Notes
        -----
        Supports multiple methods: lasso, ridge, permutation, hybrid.
        Importances are NOT normalized per variable group - raw scores are used.
        """
        # Skip if no transformed features
        if X_transformed.empty or len(X_transformed.columns) == 0:
            self._use_uniform_importance()
            return

        try:
            if self.importance_method == 'lasso':
                self._compute_lasso_importance(X_transformed, outcome)
            elif self.importance_method == 'ridge':
                self._compute_ridge_importance(X_transformed, outcome)
            elif self.importance_method == 'permutation':
                self._compute_permutation_importance(X_transformed, outcome)
            elif self.importance_method == 'hybrid':
                self._compute_hybrid_importance(X_transformed, outcome)
            else:
                raise ValueError(f"Unknown importance method: {self.importance_method}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            warnings.warn(
                f"Feature importance calculation failed: {e}\n"
                f"Method: {self.importance_method}\n"
                f"Error details: {error_details}\n"
                f"Using uniform distribution as fallback.",
                UserWarning,
                stacklevel=2
            )
            self._use_uniform_importance()

    def _compute_lasso_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using Lasso coefficients."""
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        is_regression = self._is_regression_task(outcome)

        if is_regression:
            model = LassoCV(cv=5, random_state=42, max_iter=5000)
        else:
            model = LogisticRegressionCV(
                cv=5, penalty='l1', solver='liblinear',
                random_state=42, max_iter=5000
            )

        model.fit(X_transformed, outcome)

        # Get absolute coefficients as importance
        if is_regression:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, importances))

    def _compute_ridge_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using Ridge coefficients."""
        from sklearn.linear_model import RidgeCV, LogisticRegressionCV

        is_regression = self._is_regression_task(outcome)

        if is_regression:
            model = RidgeCV(cv=5)
        else:
            model = LogisticRegressionCV(
                cv=5, penalty='l2', solver='lbfgs',
                random_state=42, max_iter=5000
            )

        model.fit(X_transformed, outcome)

        # Get absolute coefficients as importance
        if is_regression:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, importances))

    def _compute_permutation_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using permutation importance."""
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import Ridge, LogisticRegression

        is_regression = self._is_regression_task(outcome)

        # Fit a simple model for permutation
        if is_regression:
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_transformed, outcome)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_transformed, outcome,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        # Get mean importances
        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, perm_importance.importances_mean))

    def _compute_hybrid_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using hybrid approach: Lasso + Mutual Information."""
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

        is_regression = self._is_regression_task(outcome)

        # Method 1: Lasso coefficients
        if is_regression:
            lasso = LassoCV(cv=3, random_state=42, max_iter=2000)
        else:
            lasso = LogisticRegressionCV(
                cv=3, penalty='l1', solver='liblinear',
                random_state=42, max_iter=2000
            )
        lasso.fit(X_transformed, outcome)
        lasso_imp = np.abs(lasso.coef_ if is_regression else lasso.coef_[0])

        # Method 2: Mutual Information
        if is_regression:
            mi_scores = mutual_info_regression(X_transformed, outcome, random_state=42)
        else:
            mi_scores = mutual_info_classif(X_transformed, outcome, random_state=42)

        # Normalize both to [0, 1]
        lasso_imp = lasso_imp / (lasso_imp.max() + 1e-10)
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)

        # Average the two
        combined_importance = (lasso_imp + mi_scores) / 2.0

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, combined_importance))

    def _use_uniform_importance(self):
        """Fallback: uniform importance distribution (current behavior)."""
        for var in self._variables:
            if var['new_names']:
                importance_per_feature = 1.0 / len(var['new_names'])
                for feat in var['new_names']:
                    self._feature_importances[feat] = importance_per_feature

    def _is_regression_task(self, outcome: pd.Series) -> bool:
        """Determine if outcome is regression or classification."""
        # Numeric with >10 unique values = regression
        # Categorical or few unique values = classification
        if pd.api.types.is_numeric_dtype(outcome):
            n_unique = outcome.nunique()
            return n_unique > 10
        else:
            return False

    def _create_transformed_dataset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset with transformed SAFE features for importance calculation.

        Applies all transformations (numeric and categorical) to get binary features.
        """
        transformed_dfs = []

        for var in self._variables:
            col_name = var['original_name']
            if col_name not in X.columns:
                continue

            if var['type'] == 'numeric':
                transformed = self._transform_numeric_variable(var, X[col_name])
            else:
                transformed = self._transform_categorical_variable(var, X[col_name])

            if transformed is not None and not transformed.empty:
                transformed_dfs.append(transformed)

        if transformed_dfs:
            result = pd.concat(transformed_dfs, axis=1)

            # Deduplicate columns to prevent LightGBM errors
            if result.columns.duplicated().any():
                result = result.loc[:, ~result.columns.duplicated()]
        else:
            result = pd.DataFrame()

        return result

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

            # Deduplicate columns immediately after concat (prevents duplicate column names)
            # This can happen if same feature appears multiple times in transformations
            if result.columns.duplicated().any():
                result = result.loc[:, ~result.columns.duplicated()]
        else:
            result = pd.DataFrame(index=range(len(data)))

        # Filter to selected features if top_n specified
        if self.top_n is not None and self._selected_features:
            # Deduplicate while preserving order (prevents duplicate column names)
            available_features = []
            seen = set()
            for f in self._selected_features:
                if f in result.columns and f not in seen:
                    available_features.append(f)
                    seen.add(f)
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

        # Create DataFrame with dummies
        dummies_df = pd.DataFrame(transformed, columns=new_names)

        # Handle feature_type
        if self.feature_type == 'dummies':
            return dummies_df
        elif self.feature_type == 'interactions':
            # Create interactions: dummy * original_value
            interactions_df = pd.DataFrame()
            original_values = X_col.values
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interactions_df[interaction_name] = dummies_df[col] * original_values
            return interactions_df
        else:  # 'both'
            # Return both dummies and interactions
            result_df = dummies_df.copy()
            original_values = X_col.values
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                result_df[interaction_name] = dummies_df[col] * original_values
            return result_df

    def _transform_categorical_variable(
        self, var: Dict[str, Any], X_col: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Transform categorical variable based on clusters."""
        if var['clusters'] is None or not var['new_names']:
            # No clustering performed, return one-hot encoded
            dummies = pd.get_dummies(X_col, prefix=var['original_name'], drop_first=True)

            # Handle feature_type for simple one-hot encoding
            if self.feature_type == 'dummies':
                return dummies
            elif self.feature_type == 'interactions':
                # For categorical, use label encoding for interactions
                label_encoded = pd.factorize(X_col)[0]
                interactions_df = pd.DataFrame()
                for col in dummies.columns:
                    interaction_name = f"{col}_x_{var['original_name']}"
                    interactions_df[interaction_name] = dummies[col] * label_encoded
                return interactions_df
            else:  # 'both'
                result_df = dummies.copy()
                label_encoded = pd.factorize(X_col)[0]
                for col in dummies.columns:
                    interaction_name = f"{col}_x_{var['original_name']}"
                    result_df[interaction_name] = dummies[col] * label_encoded
                return result_df

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

        # Create DataFrame with dummies
        dummies_df = pd.DataFrame(transformed, columns=new_names)

        # Handle feature_type
        if self.feature_type == 'dummies':
            return dummies_df
        elif self.feature_type == 'interactions':
            # For categorical, use label encoding for interactions
            label_encoded = pd.factorize(X_col)[0]
            interactions_df = pd.DataFrame()
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interactions_df[interaction_name] = dummies_df[col] * label_encoded
            return interactions_df
        else:  # 'both'
            result_df = dummies_df.copy()
            label_encoded = pd.factorize(X_col)[0]
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                result_df[interaction_name] = dummies_df[col] * label_encoded
            return result_df

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


@dataclass
class StepSafeV2:
    """
    SAFE v2: Surrogate Assisted Feature Extraction with UNFITTED model.

    This version accepts an UNFITTED surrogate model and fits it during prep().
    It also adds max_thresholds parameter to control threshold quantity and
    sanitizes feature names for LightGBM compatibility.

    Key differences from StepSafe:
    - Accepts UNFITTED surrogate model (fitted during prep)
    - Adds max_thresholds parameter (default=5)
    - Sanitizes feature names for compatibility
    - Recalculates importances on TRANSFORMED features using multiple methods

    Parameters
    ----------
    surrogate_model : unfitted model
        UNFITTED sklearn-compatible model (will be fitted during prep)
    outcome : str
        Name of outcome variable
    penalty : float, default=10.0
        Changepoint penalty (higher = fewer thresholds)
    top_n : int, optional
        Select top N most important TRANSFORMED features
    max_thresholds : int, default=5
        Maximum number of thresholds per numeric feature
    keep_original_cols : bool, default=True
        Keep original features alongside transformations
    grid_resolution : int, default=100
        PDP grid points
    feature_type : str, default='both'
        Which variable types to process: 'numeric', 'categorical', or 'both'
    output_mode : str, default='dummies'
        Type of features to create:
        - 'dummies': Binary dummy variables only (default)
        - 'interactions': Binary dummy * original feature interactions only
        - 'both': Both dummies and interactions
    importance_method : str, default='lasso'
        Method for calculating feature importance:
        - 'lasso': Lasso regression coefficients (best for linear models)
        - 'ridge': Ridge regression coefficients (more stable, keeps all features)
        - 'permutation': Permutation importance (most reliable, slower)
        - 'hybrid': Average of Lasso + Mutual Information (robust)
    columns : selector, optional
        Which columns to transform (None = all except outcome)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> # UNFITTED model
    >>> surrogate = GradientBoostingRegressor(n_estimators=100)
    >>> rec = recipe(data, "y ~ .").step_safe_v2(
    ...     surrogate_model=surrogate,
    ...     outcome='y',
    ...     penalty=10.0,
    ...     max_thresholds=5
    ... )
    """
    surrogate_model: Any
    outcome: str
    penalty: float = 10.0
    top_n: Optional[int] = None
    max_thresholds: int = 5
    keep_original_cols: bool = True
    grid_resolution: int = 100
    feature_type: Literal['numeric', 'categorical', 'both'] = 'both'
    output_mode: Literal['dummies', 'interactions', 'both'] = 'dummies'
    importance_method: Literal['lasso', 'ridge', 'permutation', 'hybrid'] = 'lasso'
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _variables: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _original_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _feature_importances: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _fitted_model: Any = field(default=None, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Check dependencies
        if not RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures package required for step_safe_v2(). "
                "Install with: pip install ruptures"
            )

        # Validate parameters
        if self.penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {self.penalty}")

        if self.max_thresholds < 1:
            raise ValueError(f"max_thresholds must be >= 1, got {self.max_thresholds}")

        if self.grid_resolution < 10:
            raise ValueError(f"grid_resolution must be >= 10, got {self.grid_resolution}")

        if self.top_n is not None and self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")

        if self.feature_type not in ['numeric', 'categorical', 'both']:
            raise ValueError(
                f"feature_type must be 'numeric', 'categorical', or 'both', "
                f"got '{self.feature_type}'"
            )

        if self.output_mode not in ['dummies', 'interactions', 'both']:
            raise ValueError(
                f"output_mode must be 'dummies', 'interactions', or 'both', "
                f"got '{self.output_mode}'"
            )

        if self.importance_method not in ['lasso', 'ridge', 'permutation', 'hybrid']:
            raise ValueError(
                f"importance_method must be 'lasso', 'ridge', 'permutation', or 'hybrid', "
                f"got '{self.importance_method}'"
            )

        # Validate model is UNFITTED (should not have n_features_in_)
        if hasattr(self.surrogate_model, 'n_features_in_'):
            warnings.warn(
                "surrogate_model appears to be already fitted. "
                "step_safe_v2 expects UNFITTED model which will be fitted during prep().",
                UserWarning
            )

    def _sanitize_feature_name(self, name: str) -> str:
        """
        Sanitize feature names for LightGBM compatibility.

        Removes special characters and replaces with underscores.
        Pattern from step_select_shap for LightGBM compatibility.

        Parameters
        ----------
        name : str
            Original feature name

        Returns
        -------
        str
            Sanitized feature name safe for LightGBM
        """
        import re
        # Replace special characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    def prep(self, data: pd.DataFrame, training: bool = True):
        """
        Prepare step by fitting surrogate model and learning transformations.

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

        y = data[self.outcome]

        # Resolve columns to transform
        if self.columns is None:
            transform_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            transform_cols = [self.columns]
        elif callable(self.columns):
            transform_cols = self.columns(data)
        else:
            transform_cols = list(self.columns)

        # Exclude outcome and datetime columns
        transform_cols = [
            c for c in transform_cols
            if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
        ]

        if len(transform_cols) == 0:
            warnings.warn("No columns to transform after filtering", UserWarning)
            self._is_prepared = True
            return self

        X = data[transform_cols].copy()
        self._original_columns = list(X.columns)

        # Detect categorical vs numeric columns
        categorical_cols = []
        numeric_cols = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if self.feature_type in ['numeric', 'both']:
                    numeric_cols.append(col)
            else:
                if self.feature_type in ['categorical', 'both']:
                    categorical_cols.append(col)

        # Prepare data for surrogate model (one-hot encode ALL categoricals)
        # Even if we're not transforming them, we need them encoded for model fitting
        X_for_model = X.copy()
        categorical_dummies = {}

        # Get ALL categorical columns (not just ones we're transforming)
        all_cat_cols = [col for col in X.columns
                       if not pd.api.types.is_numeric_dtype(X[col])]

        for col in all_cat_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            # Sanitize dummy column names for LightGBM
            dummies.columns = [self._sanitize_feature_name(c) for c in dummies.columns]

            # Store info only for columns we're transforming
            if col in categorical_cols:
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

        # Sanitize all numeric column names too
        X_for_model.columns = [self._sanitize_feature_name(c) for c in X_for_model.columns]

        # FIT SURROGATE MODEL during prep()
        try:
            self._fitted_model = self.surrogate_model.fit(X_for_model, y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit surrogate model during prep(): {e}")

        # Create variable transformations
        for idx, col in enumerate(self._original_columns):
            if col in numeric_cols:
                var = self._fit_numeric_variable(
                    col, idx, X_for_model, X[col]
                )
                self._variables.append(var)
            elif col in categorical_cols:
                dummy_info = categorical_dummies[col]
                var = self._fit_categorical_variable(
                    col, idx, X_for_model, X[col],
                    dummy_info['dummy_names'],
                    dummy_info['levels']
                )
                self._variables.append(var)

        # Compute feature importances on TRANSFORMED features
        X_transformed = self._create_transformed_dataset(X)
        self._compute_feature_importances(X_transformed, y)

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
        Fit transformation for numeric variable using PDP changepoint detection.

        Uses max_thresholds to limit the number of thresholds created.

        Parameters
        ----------
        col_name : str
            Variable name
        col_idx : int
            Original column index
        X_model : DataFrame
            Data for surrogate model (sanitized names)
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
        algo = rpt.Pelt(model='l2').fit(pdp)
        changepoint_indices = algo.predict(pen=self.penalty)

        # Convert indices to actual values
        changepoint_values = [axes[i] for i in changepoint_indices[:-1]]

        # Apply max_thresholds limit
        if len(changepoint_values) > self.max_thresholds:
            # Keep most important changepoints (largest PDP jumps)
            pdp_diffs = []
            for i in changepoint_indices[:-1]:
                if i > 0 and i < len(pdp):
                    diff = abs(pdp[i] - pdp[i-1])
                    pdp_diffs.append((axes[i], diff))

            # Sort by diff magnitude and keep top max_thresholds
            pdp_diffs.sort(key=lambda x: x[1], reverse=True)
            changepoint_values = sorted([v for v, _ in pdp_diffs[:self.max_thresholds]])

        # Handle no changepoints case
        if not changepoint_values:
            # Always create at least one split at median
            changepoint_values = [float(np.median(X_original))]

        # Create threshold features (binary indicators: feature > threshold)
        # Sanitize feature names
        new_names = []
        for threshold_val in changepoint_values:
            sanitized_col = self._sanitize_feature_name(col_name)
            sanitized_thresh = self._sanitize_feature_name(f"{threshold_val:.2f}")
            new_names.append(f"{sanitized_col}_gt_{sanitized_thresh}")

        return {
            'type': 'numeric',
            'original_name': col_name,
            'original_index': col_idx,
            'thresholds': changepoint_values,
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
            Data for surrogate model (sanitized names)
        X_original : Series
            Original categorical variable values
        dummy_names : list
            One-hot encoded dummy column names (already sanitized)
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
        if not SCIPY_AVAILABLE:
            # No clustering - just one-hot encode
            return {
                'type': 'categorical',
                'original_name': col_name,
                'original_index': col_idx,
                'dummy_names': dummy_names,
                'levels': sorted(levels),
                'clusters': None,
                'new_names': dummy_names  # Use as-is
            }

        if pdp.ndim == 1:
            pdp_array = pdp.reshape(len(pdp), 1)
        else:
            pdp_array = pdp

        Z = ward(pdp_array)

        # Determine optimal number of clusters
        clusters = None
        new_names = []

        if pdp.shape[0] == 3 and SCIPY_AVAILABLE:
            # For 3 levels, use first linkage
            clusters = cut_tree(Z, height=Z[0, 2] - np.finfo(float).eps)
            new_names = self._create_categorical_names(
                col_name, clusters, dummy_names, levels
            )
        elif pdp.shape[0] > 3 and KNEED_AVAILABLE and SCIPY_AVAILABLE:
            # Use KneeLocator for optimal clusters
            try:
                kneed = KneeLocator(
                    range(Z.shape[0]), Z[:, 2],
                    direction='increasing', curve='convex'
                )

                if kneed.knee is not None:
                    clusters = cut_tree(Z, height=Z[kneed.knee + 1, 2] - np.finfo(float).eps)
                    new_names = self._create_categorical_names(
                        col_name, clusters, dummy_names, levels
                    )
            except:
                pass

        return {
            'type': 'categorical',
            'original_name': col_name,
            'original_index': col_idx,
            'dummy_names': dummy_names,
            'levels': sorted(levels),
            'clusters': clusters,
            'new_names': new_names[1:] if new_names else []  # Drop first (base level)
        }

    def _get_partial_dependence_numeric(
        self, col_name: str, X: pd.DataFrame, X_original: pd.Series
    ) -> tuple:
        """
        Compute partial dependence plot for numeric variable.

        Uses the FITTED surrogate model.

        Parameters
        ----------
        col_name : str
            Variable name
        X : DataFrame
            Full feature matrix for surrogate model (sanitized names)
        X_original : Series
            Original numeric variable (before sanitization)

        Returns
        -------
        tuple
            (pdp values, grid points)
        """
        # Create grid from min to max
        points = np.linspace(
            X_original.min(),
            X_original.max(),
            self.grid_resolution
        )

        pdp = []
        sanitized_col = self._sanitize_feature_name(col_name)

        for point in points:
            # Create a copy and set all rows to this value
            X_copy = X.copy()
            if sanitized_col in X_copy.columns:
                X_copy[sanitized_col] = point

            # Get predictions from FITTED model
            if hasattr(self._fitted_model, 'predict_proba'):
                preds = self._fitted_model.predict_proba(X_copy)
            else:
                preds = self._fitted_model.predict(X_copy)

            # Average predictions
            pdp.append(np.mean(preds, axis=0))

        return np.array(pdp), points

    def _get_partial_dependence_categorical(
        self, col_name: str, X: pd.DataFrame, dummy_names: List[str]
    ) -> tuple:
        """
        Compute partial dependence for categorical variable.

        Uses the FITTED surrogate model.

        Parameters
        ----------
        col_name : str
            Variable name
        X : DataFrame
            Full feature matrix (sanitized names)
        dummy_names : list
            One-hot encoded dummy column names (already sanitized)

        Returns
        -------
        tuple
            (pdp values, level names)
        """
        pdp = []
        axes = ['base']

        # Base level (all dummies = 0)
        X_copy = X.copy()
        X_copy[dummy_names] = 0

        if hasattr(self._fitted_model, 'predict_proba'):
            preds = self._fitted_model.predict_proba(X_copy)
        else:
            preds = self._fitted_model.predict(X_copy)
        pdp.append(np.mean(preds, axis=0))

        # Each dummy level
        for dummy in dummy_names:
            axes.append(dummy)
            X_copy = X.copy()
            X_copy[dummy_names] = 0
            X_copy[dummy] = 1

            if hasattr(self._fitted_model, 'predict_proba'):
                preds = self._fitted_model.predict_proba(X_copy)
            else:
                preds = self._fitted_model.predict(X_copy)
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
                        level_name = dummy_names[idx - 1].replace(f"{col_name}_", "")
                        names.append(level_name)

            # Create sanitized names
            safe_name = f"{col_name}_{'_'.join(names)}"
            safe_name = self._sanitize_feature_name(safe_name)
            new_names.append(safe_name)

        return new_names

    def _compute_feature_importances(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """
        Compute feature importances using various methods on transformed features.

        Parameters
        ----------
        X_transformed : DataFrame
            Transformed features (SAFE binary indicators or interactions)
        outcome : Series
            Target variable for supervised importance calculation

        Notes
        -----
        Supports multiple methods: lasso, ridge, permutation, hybrid.
        Importances are NOT normalized per variable group - raw scores are used.
        """
        # Skip if no transformed features
        if X_transformed.empty or len(X_transformed.columns) == 0:
            self._use_uniform_importance()
            return

        try:
            if self.importance_method == 'lasso':
                self._compute_lasso_importance(X_transformed, outcome)
            elif self.importance_method == 'ridge':
                self._compute_ridge_importance(X_transformed, outcome)
            elif self.importance_method == 'permutation':
                self._compute_permutation_importance(X_transformed, outcome)
            elif self.importance_method == 'hybrid':
                self._compute_hybrid_importance(X_transformed, outcome)
            else:
                raise ValueError(f"Unknown importance method: {self.importance_method}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            warnings.warn(
                f"Feature importance calculation failed: {e}\n"
                f"Method: {self.importance_method}\n"
                f"Error details: {error_details}\n"
                f"Using uniform distribution as fallback.",
                UserWarning,
                stacklevel=2
            )
            self._use_uniform_importance()

    def _compute_lasso_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using Lasso coefficients."""
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        is_regression = self._is_regression_task(outcome)

        if is_regression:
            model = LassoCV(cv=5, random_state=42, max_iter=5000)
        else:
            model = LogisticRegressionCV(
                cv=5, penalty='l1', solver='liblinear',
                random_state=42, max_iter=5000
            )

        model.fit(X_transformed, outcome)

        # Get absolute coefficients as importance
        if is_regression:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, importances))

    def _compute_ridge_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using Ridge coefficients."""
        from sklearn.linear_model import RidgeCV, LogisticRegressionCV

        is_regression = self._is_regression_task(outcome)

        if is_regression:
            model = RidgeCV(cv=5)
        else:
            model = LogisticRegressionCV(
                cv=5, penalty='l2', solver='lbfgs',
                random_state=42, max_iter=5000
            )

        model.fit(X_transformed, outcome)

        # Get absolute coefficients as importance
        if is_regression:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, importances))

    def _compute_permutation_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using permutation importance."""
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import Ridge, LogisticRegression

        is_regression = self._is_regression_task(outcome)

        # Fit a simple model for permutation
        if is_regression:
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_transformed, outcome)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_transformed, outcome,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        # Get mean importances
        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, perm_importance.importances_mean))

    def _compute_hybrid_importance(self, X_transformed: pd.DataFrame, outcome: pd.Series):
        """Compute importance using hybrid approach: Lasso + Mutual Information."""
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

        is_regression = self._is_regression_task(outcome)

        # Method 1: Lasso coefficients
        if is_regression:
            lasso = LassoCV(cv=3, random_state=42, max_iter=2000)
        else:
            lasso = LogisticRegressionCV(
                cv=3, penalty='l1', solver='liblinear',
                random_state=42, max_iter=2000
            )
        lasso.fit(X_transformed, outcome)
        lasso_imp = np.abs(lasso.coef_ if is_regression else lasso.coef_[0])

        # Method 2: Mutual Information
        if is_regression:
            mi_scores = mutual_info_regression(X_transformed, outcome, random_state=42)
        else:
            mi_scores = mutual_info_classif(X_transformed, outcome, random_state=42)

        # Normalize both to [0, 1]
        lasso_imp = lasso_imp / (lasso_imp.max() + 1e-10)
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)

        # Average the two
        combined_importance = (lasso_imp + mi_scores) / 2.0

        feature_names = X_transformed.columns
        self._feature_importances = dict(zip(feature_names, combined_importance))

    def _use_uniform_importance(self):
        """Fallback: uniform importance distribution."""
        for var in self._variables:
            if var['new_names']:
                importance_per_feature = 1.0 / len(var['new_names'])
                for feat in var['new_names']:
                    self._feature_importances[feat] = importance_per_feature

    def _is_regression_task(self, outcome: pd.Series) -> bool:
        """Determine if outcome is regression or classification."""
        if pd.api.types.is_numeric_dtype(outcome):
            n_unique = outcome.nunique()
            return n_unique > 10
        else:
            return False

    def _create_transformed_dataset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset with transformed SAFE features for importance calculation.

        Applies all transformations (numeric and categorical) to get binary features.
        """
        transformed_dfs = []

        for var in self._variables:
            col_name = var['original_name']
            if col_name not in X.columns:
                continue

            if var['type'] == 'numeric':
                transformed = self._transform_numeric_variable(var, X[col_name])
            else:
                transformed = self._transform_categorical_variable(var, X[col_name])

            if transformed is not None and not transformed.empty:
                transformed_dfs.append(transformed)

        if transformed_dfs:
            result = pd.concat(transformed_dfs, axis=1)

            # Deduplicate columns to prevent LightGBM errors
            if result.columns.duplicated().any():
                result = result.loc[:, ~result.columns.duplicated()]
        else:
            result = pd.DataFrame()

        return result

    def _transform_numeric_variable(
        self, var: Dict[str, Any], X_col: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Transform numeric variable into binary threshold indicators and/or interactions."""
        if not var.get('thresholds') or not var['new_names']:
            return None

        thresholds = var['thresholds']
        new_names = var['new_names']

        # Create binary indicators (feature > threshold)
        dummies_df = pd.DataFrame()
        for threshold_val, feat_name in zip(thresholds, new_names):
            dummies_df[feat_name] = (X_col > threshold_val).astype(int)

        # Handle output_mode
        if self.output_mode == 'dummies':
            return dummies_df
        elif self.output_mode == 'interactions':
            # Create interactions: dummy * original_value
            interactions_df = pd.DataFrame()
            original_values = X_col.values
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interaction_name = self._sanitize_feature_name(interaction_name)
                interactions_df[interaction_name] = dummies_df[col] * original_values
            return interactions_df
        else:  # 'both'
            # Return both dummies and interactions
            result_df = dummies_df.copy()
            original_values = X_col.values
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interaction_name = self._sanitize_feature_name(interaction_name)
                result_df[interaction_name] = dummies_df[col] * original_values
            return result_df

    def _transform_categorical_variable(
        self, var: Dict[str, Any], X_col: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Transform categorical variable based on clusters and/or interactions."""
        if var['clusters'] is None or not var['new_names']:
            # No clustering - simple one-hot encode
            dummies = pd.get_dummies(X_col, prefix=var['original_name'], drop_first=True)
            # Sanitize column names
            dummies.columns = [self._sanitize_feature_name(c) for c in dummies.columns]

            # Handle output_mode for simple one-hot encoding
            if self.output_mode == 'dummies':
                return dummies
            elif self.output_mode == 'interactions':
                # For categorical, use label encoding for interactions
                label_encoded = pd.factorize(X_col)[0]
                interactions_df = pd.DataFrame()
                for col in dummies.columns:
                    interaction_name = f"{col}_x_{var['original_name']}"
                    interaction_name = self._sanitize_feature_name(interaction_name)
                    interactions_df[interaction_name] = dummies[col] * label_encoded
                return interactions_df
            else:  # 'both'
                result_df = dummies.copy()
                label_encoded = pd.factorize(X_col)[0]
                for col in dummies.columns:
                    interaction_name = f"{col}_x_{var['original_name']}"
                    interaction_name = self._sanitize_feature_name(interaction_name)
                    result_df[interaction_name] = dummies[col] * label_encoded
                return result_df

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

        # Create DataFrame with dummies
        dummies_df = pd.DataFrame(transformed, columns=new_names)

        # Handle output_mode
        if self.output_mode == 'dummies':
            return dummies_df
        elif self.output_mode == 'interactions':
            # For categorical, use label encoding for interactions
            label_encoded = pd.factorize(X_col)[0]
            interactions_df = pd.DataFrame()
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interaction_name = self._sanitize_feature_name(interaction_name)
                interactions_df[interaction_name] = dummies_df[col] * label_encoded
            return interactions_df
        else:  # 'both'
            result_df = dummies_df.copy()
            label_encoded = pd.factorize(X_col)[0]
            for col in dummies_df.columns:
                interaction_name = f"{col}_x_{var['original_name']}"
                interaction_name = self._sanitize_feature_name(interaction_name)
                result_df[interaction_name] = dummies_df[col] * label_encoded
            return result_df

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SAFE v2 transformations to new data.

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

            # Deduplicate columns
            if result.columns.duplicated().any():
                result = result.loc[:, ~result.columns.duplicated()]
        else:
            result = pd.DataFrame(index=range(len(data)))

        # Filter to selected features if top_n specified
        if self.top_n is not None and self._selected_features:
            # Deduplicate while preserving order
            available_features = []
            seen = set()
            for f in self._selected_features:
                if f in result.columns and f not in seen:
                    available_features.append(f)
                    seen.add(f)
            result = result[available_features]

        # Always preserve outcome column if present
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

    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances for transformed features."""
        if not self._is_prepared:
            raise ValueError("Step must be prepared before accessing feature importances")

        importances_df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in self._feature_importances.items()
        ])

        return importances_df.sort_values('importance', ascending=False).reset_index(drop=True)

    def get_transformations(self) -> Dict[str, Any]:
        """
        Get transformation metadata for all variables.

        Returns
        -------
        dict
            Transformation information for each variable including:
            - For numeric: type, original_name, changepoints (alias for thresholds), thresholds, intervals (alias for new_names), new_names
            - For categorical: type, original_name, levels, merged_levels (alias for new_names), new_names

        Note: Provides both old and new naming for backward compatibility.
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
                # New naming
                info['thresholds'] = var['thresholds']
                info['new_names'] = var['new_names']
                # Old naming (backward compatibility)
                info['changepoints'] = var['thresholds']
                info['intervals'] = var['new_names']
            else:  # categorical
                # New naming
                info['levels'] = var['levels']
                info['new_names'] = var['new_names']
                # Old naming (backward compatibility)
                info['merged_levels'] = var['new_names']

            transformations[var['original_name']] = info

        return transformations


def step_safe(recipe, surrogate_model: Any, outcome: str, penalty: float = 3.0,
              pelt_model: str = 'l2', no_changepoint_strategy: str = 'median',
              feature_type: str = 'dummies', keep_original_cols: bool = False,
              top_n: Optional[int] = None, grid_resolution: int = 1000,
              skip: bool = False, id: Optional[str] = None):
    """
    Add SAFE (Surrogate Assisted Feature Extraction) step to recipe.

    Creates interpretable features by using a complex surrogate model to guide
    feature transformations. For numeric variables, detects changepoints in
    partial dependence plots. For categorical variables, merges similar levels.

    Parameters
    ----------
    recipe : Recipe
        Recipe to add step to
    surrogate_model : object
        Pre-fitted surrogate model (e.g., GradientBoosting, RandomForest)
    outcome : str
        Name of outcome variable
    penalty : float, default=3.0
        Changepoint detection penalty (higher = fewer changepoints)
    pelt_model : str, default='l2'
        Cost function for Pelt algorithm: 'l2', 'l1', or 'rbf'
    no_changepoint_strategy : str, default='median'
        Strategy when no changepoint detected: 'median' or 'drop'
    feature_type : str, default='dummies'
        Type of features: 'dummies', 'interactions', or 'both'
    keep_original_cols : bool, default=False
        Whether to keep original columns
    top_n : int, optional
        Select top N most important transformed features
    grid_resolution : int, default=1000
        Number of points for partial dependence grid
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Returns
    -------
    Recipe
        Recipe with step added

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> surrogate = GradientBoostingRegressor()
    >>> surrogate.fit(X_train, y_train)
    >>> rec = recipe(data, "y ~ .").step_safe(
    ...     surrogate_model=surrogate,
    ...     outcome='y',
    ...     penalty=3.0
    ... )
    """
    step = StepSafe(
        surrogate_model=surrogate_model,
        outcome=outcome,
        penalty=penalty,
        pelt_model=pelt_model,
        no_changepoint_strategy=no_changepoint_strategy,
        feature_type=feature_type,
        keep_original_cols=keep_original_cols,
        top_n=top_n,
        grid_resolution=grid_resolution,
        skip=skip,
        id=id
    )
    recipe.steps.append(step)
    return recipe


def step_safe_v2(recipe, surrogate_model: Any, outcome: str, penalty: float = 10.0,
                 top_n: Optional[int] = None, max_thresholds: int = 5,
                 keep_original_cols: bool = True, grid_resolution: int = 100,
                 feature_type: str = 'both', columns=None,
                 skip: bool = False, id: Optional[str] = None):
    """
    Add SAFE v2 (UNFITTED model version) step to recipe.

    This version accepts an UNFITTED surrogate model and fits it during prep().
    It also sanitizes feature names for LightGBM compatibility and recalculates
    importances on transformed features.

    Parameters
    ----------
    recipe : Recipe
        Recipe to add step to
    surrogate_model : object
        UNFITTED sklearn-compatible model (will be fitted during prep)
    outcome : str
        Name of outcome variable
    penalty : float, default=10.0
        Changepoint penalty (higher = fewer thresholds)
    top_n : int, optional
        Select top N most important TRANSFORMED features
    max_thresholds : int, default=5
        Maximum number of thresholds per numeric feature
    keep_original_cols : bool, default=True
        Keep original features alongside transformations
    grid_resolution : int, default=100
        PDP grid points
    feature_type : str, default='both'
        'numeric', 'categorical', or 'both'
    columns : selector, optional
        Which columns to transform
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Returns
    -------
    Recipe
        Recipe with step added

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> # UNFITTED model
    >>> surrogate = GradientBoostingRegressor(n_estimators=100)
    >>> rec = recipe(data, "y ~ .").step_safe_v2(
    ...     surrogate_model=surrogate,
    ...     outcome='y',
    ...     penalty=10.0,
    ...     max_thresholds=5
    ... )
    """
    step = StepSafeV2(
        surrogate_model=surrogate_model,
        outcome=outcome,
        penalty=penalty,
        top_n=top_n,
        max_thresholds=max_thresholds,
        keep_original_cols=keep_original_cols,
        grid_resolution=grid_resolution,
        feature_type=feature_type,
        columns=columns,
        skip=skip,
        id=id
    )
    recipe.steps.append(step)
    return recipe
