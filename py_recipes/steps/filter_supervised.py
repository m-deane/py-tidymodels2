"""
Supervised filter-based feature selection steps.

These steps use statistical tests and machine learning techniques to score and rank
features based on their relationship with the outcome variable. Based on R's filtro package.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score


@dataclass
class StepFilterAnova:
    """
    Filter features using ANOVA F-test.

    Tests if groups (for categorical outcome) have different means, or if
    categorical predictors have different distributions (for numeric outcome).

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    threshold : float, optional
        Minimum score to keep feature. If None, uses top_n or top_p
    top_n : int, optional
        Keep top N features by score
    top_p : float, optional
        Keep top proportion of features (e.g., 0.2 for top 20%)
    use_pvalue : bool, default=True
        If True, use -log10(p-value) as score. If False, use F-statistic
    columns : selector, optional
        Which columns to score. If None, uses all numeric columns
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_filter_anova
    >>>
    >>> # Keep top 20% of features by ANOVA F-test
    >>> rec = recipe(data, "species ~ .").step_filter_anova(
    ...     outcome='species', top_p=0.2
    ... )
    >>>
    >>> # Keep features with p-value < 0.05 (score > 1.3)
    >>> rec = recipe(data, "price ~ .").step_filter_anova(
    ...     outcome='price', threshold=1.3, use_pvalue=True
    ... )
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    use_pvalue: bool = True
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Validate selection mode
        modes = sum([self.threshold is not None, self.top_n is not None, self.top_p is not None])
        if modes == 0:
            raise ValueError("Must specify one of: threshold, top_n, or top_p")
        if modes > 1:
            raise ValueError("Can only specify one of: threshold, top_n, or top_p")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by computing scores and selecting features."""
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to score
        if self.columns is None:
            # Use all columns except outcome
            score_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = self.columns(data)
        else:
            score_cols = list(self.columns)

        # Remove outcome if accidentally included
        score_cols = [c for c in score_cols if c != self.outcome]

        if len(score_cols) == 0:
            raise ValueError("No columns to score after resolving selector")

        # Compute scores and create a new prepared instance (not mutate self)
        prepared = replace(self)
        prepared._scores = self._compute_anova_scores(data[score_cols], y)

        # Select features based on threshold/top_n/top_p
        prepared._selected_features = self._select_features(prepared._scores)

        prepared._is_prepared = True
        return prepared

    def _compute_anova_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute ANOVA F-test scores for each feature."""
        scores = {}
        y_is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)

        for col in X.columns:
            try:
                x = X[col]
                x_is_categorical = pd.api.types.is_categorical_dtype(x) or pd.api.types.is_object_dtype(x)

                if y_is_categorical and not x_is_categorical:
                    # Numeric predictor -> Categorical outcome: standard ANOVA
                    groups = [x[y == level].dropna() for level in y.unique()]
                    # Remove empty groups
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) < 2:
                        scores[col] = 0.0
                        continue

                    f_stat, p_val = stats.f_oneway(*groups)

                elif not y_is_categorical and x_is_categorical:
                    # Categorical predictor -> Numeric outcome: ANOVA via linear model
                    groups = [y[x == level].dropna() for level in x.unique()]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) < 2:
                        scores[col] = 0.0
                        continue

                    f_stat, p_val = stats.f_oneway(*groups)

                elif not y_is_categorical and not x_is_categorical:
                    # Both numeric: use linear regression F-test
                    mask = ~(x.isna() | y.isna())
                    x_clean = x[mask].values.reshape(-1, 1)
                    y_clean = y[mask].values

                    if len(x_clean) < 3:
                        scores[col] = 0.0
                        continue

                    # Fit y ~ x and extract F-statistic
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(x_clean, y_clean)

                    y_pred = model.predict(x_clean)
                    ssr = np.sum((y_pred - y_clean.mean()) ** 2)  # Regression sum of squares
                    sse = np.sum((y_clean - y_pred) ** 2)  # Error sum of squares

                    df_reg = 1  # 1 predictor
                    df_res = len(y_clean) - 2  # n - 2

                    if df_res <= 0 or sse == 0:
                        scores[col] = 0.0
                        continue

                    msr = ssr / df_reg
                    mse = sse / df_res
                    f_stat = msr / mse if mse > 0 else 0.0

                    # Calculate p-value
                    p_val = 1 - stats.f.cdf(f_stat, df_reg, df_res)

                else:
                    # Both categorical: use chi-squared instead (will score as 0 here)
                    scores[col] = 0.0
                    continue

                # Store score
                if self.use_pvalue:
                    # Transform to -log10(p-value), capping at infinity fallback
                    if p_val == 0 or np.isnan(p_val):
                        scores[col] = np.inf
                    else:
                        scores[col] = -np.log10(p_val)
                else:
                    scores[col] = f_stat if not np.isnan(f_stat) else 0.0

            except Exception:
                # Fallback value for failed tests
                scores[col] = np.inf if self.use_pvalue else 0.0

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on threshold/top_n/top_p."""
        if len(scores) == 0:
            return []

        # Sort by score (descending)
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            # Keep features above threshold
            selected = [feat for feat, score in sorted_features if score >= self.threshold]
        elif self.top_n is not None:
            # Keep top N
            selected = [feat for feat, _ in sorted_features[:self.top_n]]
        else:  # top_p
            # Keep top proportion
            n_keep = max(1, int(len(sorted_features) * self.top_p))
            selected = [feat for feat, _ in sorted_features[:n_keep]]

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to new data."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        # Keep selected features + outcome
        keep_cols = self._selected_features + [self.outcome]
        keep_cols = [c for c in keep_cols if c in data.columns]

        return data[keep_cols]


@dataclass
class StepFilterRfImportance:
    """
    Filter features using Random Forest feature importance.

    Uses permutation-based feature importance from Random Forest to rank features.
    Works with both classification and regression outcomes, and handles non-linear
    relationships and feature interactions.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable
    threshold : float, optional
        Minimum importance score to keep feature
    top_n : int, optional
        Keep top N features by importance
    top_p : float, optional
        Keep top proportion of features
    trees : int, default=100
        Number of trees in random forest
    mtry : int, optional
        Number of features to sample per split. If None, uses sqrt(n_features)
    min_n : int, default=2
        Minimum samples per leaf
    random_state : int, optional
        Random seed for reproducibility
    columns : selector, optional
        Which columns to score
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Examples
    --------
    >>> # Keep top 10 features by RF importance
    >>> rec = recipe(data, "y ~ .").step_filter_rf_importance(
    ...     outcome='y', top_n=10, trees=200
    ... )
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    trees: int = 100
    mtry: Optional[int] = None
    min_n: int = 2
    random_state: Optional[int] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        modes = sum([self.threshold is not None, self.top_n is not None, self.top_p is not None])
        if modes == 0:
            raise ValueError("Must specify one of: threshold, top_n, or top_p")
        if modes > 1:
            raise ValueError("Can only specify one of: threshold, top_n, or top_p")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare step by computing RF importance scores."""
        if self.skip or not training:
            return self

        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = self.columns(data)
        else:
            score_cols = list(self.columns)

        # Exclude outcome and datetime columns
        score_cols = [
            c for c in score_cols
            if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
        ]

        if len(score_cols) == 0:
            raise ValueError("No columns to score")

        # Compute RF importance scores and create a new prepared instance
        prepared = replace(self)
        prepared._scores = self._compute_rf_importance(data[score_cols], y)
        prepared._selected_features = self._select_features(prepared._scores)
        prepared._is_prepared = True
        return prepared

    def _compute_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute Random Forest feature importance."""
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove zero-variance numeric columns
        non_zv_cols = []
        if len(numeric_cols) > 0:
            variances = X[numeric_cols].var()
            non_zv_numeric = variances[variances > 0].index.tolist()
            non_zv_cols.extend(non_zv_numeric)

        # Keep all categorical columns (variance check doesn't apply)
        non_zv_cols.extend(cat_cols)

        if len(non_zv_cols) == 0:
            return {col: 0.0 for col in X.columns}

        X_clean = X[non_zv_cols].copy()

        # Handle categorical columns (one-hot encode)
        cat_cols_clean = X_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols_clean) > 0:
            X_clean = pd.get_dummies(X_clean, columns=cat_cols_clean, drop_first=True)

        # Drop rows with missing values
        mask = ~(X_clean.isna().any(axis=1) | y.isna())
        X_clean = X_clean[mask]
        y_clean = y[mask]

        if len(X_clean) < 10:
            return {col: 0.0 for col in X.columns}

        # Determine task type
        y_is_categorical = pd.api.types.is_categorical_dtype(y_clean) or pd.api.types.is_object_dtype(y_clean)

        # Set mtry
        mtry = self.mtry if self.mtry is not None else max(1, int(np.sqrt(X_clean.shape[1])))
        mtry = min(mtry, X_clean.shape[1])

        # Fit Random Forest
        if y_is_categorical:
            rf = RandomForestClassifier(
                n_estimators=self.trees,
                max_features=mtry,
                min_samples_leaf=self.min_n,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=self.trees,
                max_features=mtry,
                min_samples_leaf=self.min_n,
                random_state=self.random_state,
                n_jobs=-1
            )

        rf.fit(X_clean, y_clean)

        # Extract feature importances
        importances = rf.feature_importances_

        # Map back to original column names
        scores = {}
        feature_names = X_clean.columns.tolist()

        for orig_col in X.columns:
            if orig_col in non_zv_cols:
                # Find matching features (handles one-hot encoded)
                matching_indices = [i for i, name in enumerate(feature_names) if name.startswith(orig_col)]
                if matching_indices:
                    # Sum importance across one-hot encoded features
                    scores[orig_col] = sum(importances[i] for i in matching_indices)
                else:
                    scores[orig_col] = 0.0
            else:
                scores[orig_col] = 0.0

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on threshold/top_n/top_p."""
        if len(scores) == 0:
            return []

        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            selected = [feat for feat, score in sorted_features if score >= self.threshold]
        elif self.top_n is not None:
            selected = [feat for feat, _ in sorted_features[:self.top_n]]
        else:
            n_keep = max(1, int(len(sorted_features) * self.top_p))
            selected = [feat for feat, _ in sorted_features[:n_keep]]

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to new data."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        keep_cols = self._selected_features + [self.outcome]
        keep_cols = [c for c in keep_cols if c in data.columns]
        return data[keep_cols]


@dataclass
class StepFilterMutualInfo:
    """
    Filter features using mutual information (information gain).

    Measures the mutual dependence between features and outcome using entropy.
    Works for both classification and regression, capturing non-linear relationships.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    threshold : float, optional
        Minimum MI score to keep feature
    top_n : int, optional
        Keep top N features
    top_p : float, optional
        Keep top proportion
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random seed
    columns : selector, optional
        Which columns to score
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Examples
    --------
    >>> # Keep features with MI > 0.1
    >>> rec = recipe(data, "y ~ .").step_filter_mutual_info(
    ...     outcome='y', threshold=0.1
    ... )
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    n_neighbors: int = 3
    random_state: Optional[int] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        modes = sum([self.threshold is not None, self.top_n is not None, self.top_p is not None])
        if modes == 0:
            raise ValueError("Must specify one of: threshold, top_n, or top_p")
        if modes > 1:
            raise ValueError("Can only specify one of: threshold, top_n, or top_p")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare step by computing mutual information scores."""
        if self.skip or not training:
            return self

        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = self.columns(data)
        else:
            score_cols = list(self.columns)

        # Exclude outcome and datetime columns
        score_cols = [
            c for c in score_cols
            if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
        ]

        if len(score_cols) == 0:
            raise ValueError("No columns to score")

        # Compute MI scores and create a new prepared instance
        prepared = replace(self)
        prepared._scores = self._compute_mutual_info(data[score_cols], y)
        prepared._selected_features = self._select_features(prepared._scores)
        prepared._is_prepared = True
        return prepared

    def _compute_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute mutual information scores."""
        # Handle categorical columns
        X_clean = X.copy()
        cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            X_clean = pd.get_dummies(X_clean, columns=cat_cols, drop_first=True)

        # Drop rows with missing
        mask = ~(X_clean.isna().any(axis=1) | y.isna())
        X_clean = X_clean[mask]
        y_clean = y[mask]

        if len(X_clean) < 5:
            return {col: 0.0 for col in X.columns}

        # Determine task type
        y_is_categorical = pd.api.types.is_categorical_dtype(y_clean) or pd.api.types.is_object_dtype(y_clean)

        # Compute MI
        if y_is_categorical:
            mi_scores = mutual_info_classif(
                X_clean, y_clean,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )
        else:
            mi_scores = mutual_info_regression(
                X_clean, y_clean,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )

        # Map back to original columns
        scores = {}
        feature_names = X_clean.columns.tolist()

        for orig_col in X.columns:
            matching_indices = [i for i, name in enumerate(feature_names) if name.startswith(orig_col)]
            if matching_indices:
                scores[orig_col] = sum(mi_scores[i] for i in matching_indices)
            else:
                scores[orig_col] = 0.0

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on threshold/top_n/top_p."""
        if len(scores) == 0:
            return []

        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            selected = [feat for feat, score in sorted_features if score >= self.threshold]
        elif self.top_n is not None:
            selected = [feat for feat, _ in sorted_features[:self.top_n]]
        else:
            n_keep = max(1, int(len(sorted_features) * self.top_p))
            selected = [feat for feat, _ in sorted_features[:n_keep]]

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to new data."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        keep_cols = self._selected_features + [self.outcome]
        keep_cols = [c for c in keep_cols if c in data.columns]
        return data[keep_cols]


@dataclass
class StepFilterRocAuc:
    """
    Filter features using ROC AUC scores.

    Computes area under ROC curve for each feature individually against the outcome.
    Works for binary and multiclass classification. For regression, swaps roles to
    treat numeric outcome as predictor.

    Parameters
    ----------
    outcome : str
        Name of outcome variable (must be categorical)
    threshold : float, optional
        Minimum AUC to keep feature (range: 0.5-1.0)
    top_n : int, optional
        Keep top N features
    top_p : float, optional
        Keep top proportion
    multiclass_strategy : str, default='ovr'
        For multiclass: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)
    columns : selector, optional
        Which columns to score
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Examples
    --------
    >>> # Keep features with AUC > 0.7
    >>> rec = recipe(data, "species ~ .").step_filter_roc_auc(
    ...     outcome='species', threshold=0.7
    ... )
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    multiclass_strategy: str = 'ovr'
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        modes = sum([self.threshold is not None, self.top_n is not None, self.top_p is not None])
        if modes == 0:
            raise ValueError("Must specify one of: threshold, top_n, or top_p")
        if modes > 1:
            raise ValueError("Can only specify one of: threshold, top_n, or top_p")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.multiclass_strategy not in ['ovr', 'ovo']:
            raise ValueError(f"multiclass_strategy must be 'ovr' or 'ovo', got {self.multiclass_strategy}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare step by computing ROC AUC scores."""
        if self.skip or not training:
            return self

        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = self.columns(data)
        else:
            score_cols = list(self.columns)

        # Exclude outcome and datetime columns
        score_cols = [
            c for c in score_cols
            if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
        ]

        if len(score_cols) == 0:
            raise ValueError("No columns to score")

        # Compute ROC AUC scores and create a new prepared instance
        prepared = replace(self)
        prepared._scores = self._compute_roc_auc(data[score_cols], y)
        prepared._selected_features = self._select_features(prepared._scores)
        prepared._is_prepared = True
        return prepared

    def _compute_roc_auc(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute ROC AUC scores for each feature."""
        scores = {}

        # Check if outcome is categorical
        y_is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)

        if not y_is_categorical:
            # Outcome is numeric - swap roles (treat y as predictor for each X)
            for col in X.columns:
                x = X[col]
                x_is_categorical = pd.api.types.is_categorical_dtype(x) or pd.api.types.is_object_dtype(x)

                if not x_is_categorical:
                    scores[col] = 0.5  # Both numeric - ROC AUC not applicable
                    continue

                # Categorical X, numeric Y - compute AUC
                mask = ~(x.isna() | y.isna())
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean.unique()) < 2:
                    scores[col] = 0.5
                    continue

                try:
                    auc = roc_auc_score(x_clean, y_clean, multi_class=self.multiclass_strategy)
                    # Transform: max(auc, 1-auc) to account for reverse direction
                    scores[col] = max(auc, 1 - auc)
                except Exception:
                    scores[col] = 0.5
        else:
            # Outcome is categorical - standard ROC AUC
            for col in X.columns:
                x = X[col]

                # Drop missing
                mask = ~(x.isna() | y.isna())
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 10:
                    scores[col] = 0.5
                    continue

                try:
                    # For categorical X, use one-hot encoding
                    x_is_categorical = pd.api.types.is_categorical_dtype(x_clean) or pd.api.types.is_object_dtype(x_clean)
                    if x_is_categorical:
                        # Create binary indicators and average their AUCs
                        aucs = []
                        for level in x_clean.unique():
                            x_binary = (x_clean == level).astype(int)
                            try:
                                auc = roc_auc_score(y_clean, x_binary, multi_class=self.multiclass_strategy)
                                aucs.append(max(auc, 1 - auc))
                            except:
                                pass
                        scores[col] = np.mean(aucs) if aucs else 0.5
                    else:
                        # Numeric predictor
                        auc = roc_auc_score(y_clean, x_clean, multi_class=self.multiclass_strategy)
                        scores[col] = max(auc, 1 - auc)
                except Exception:
                    scores[col] = 0.5

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on threshold/top_n/top_p."""
        if len(scores) == 0:
            return []

        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            selected = [feat for feat, score in sorted_features if score >= self.threshold]
        elif self.top_n is not None:
            selected = [feat for feat, _ in sorted_features[:self.top_n]]
        else:
            n_keep = max(1, int(len(sorted_features) * self.top_p))
            selected = [feat for feat, _ in sorted_features[:n_keep]]

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to new data."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        keep_cols = self._selected_features + [self.outcome]
        keep_cols = [c for c in keep_cols if c in data.columns]
        return data[keep_cols]


@dataclass
class StepFilterChisq:
    """
    Filter features using Chi-squared or Fisher exact test.

    Tests independence between categorical variables using contingency tables.
    Works for categorical predictors and categorical outcomes.

    Parameters
    ----------
    outcome : str
        Name of outcome variable (must be categorical)
    threshold : float, optional
        Minimum score (-log10 p-value) to keep feature
    top_n : int, optional
        Keep top N features
    top_p : float, optional
        Keep top proportion
    method : str, default='chisq'
        Test method: 'chisq' (chi-squared) or 'fisher' (Fisher exact)
    use_pvalue : bool, default=True
        If True, use -log10(p-value). If False, use test statistic
    columns : selector, optional
        Which columns to score
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier

    Examples
    --------
    >>> # Keep categorical features with p < 0.05
    >>> rec = recipe(data, "species ~ .").step_filter_chisq(
    ...     outcome='species', threshold=1.3  # -log10(0.05) ≈ 1.3
    ... )
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    method: str = 'chisq'
    use_pvalue: bool = True
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        modes = sum([self.threshold is not None, self.top_n is not None, self.top_p is not None])
        if modes == 0:
            raise ValueError("Must specify one of: threshold, top_n, or top_p")
        if modes > 1:
            raise ValueError("Can only specify one of: threshold, top_n, or top_p")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.method not in ['chisq', 'fisher']:
            raise ValueError(f"method must be 'chisq' or 'fisher', got {self.method}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare step by computing chi-squared/Fisher test scores."""
        if self.skip or not training:
            return self

        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns
        if self.columns is None:
            score_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            score_cols = [self.columns]
        elif callable(self.columns):
            score_cols = self.columns(data)
        else:
            score_cols = list(self.columns)

        # Exclude outcome and datetime columns
        score_cols = [
            c for c in score_cols
            if c != self.outcome and not pd.api.types.is_datetime64_any_dtype(data[c])
        ]

        if len(score_cols) == 0:
            raise ValueError("No columns to score")

        # Compute chi-squared scores and create a new prepared instance
        prepared = replace(self)
        prepared._scores = self._compute_chisq_scores(data[score_cols], y)
        prepared._selected_features = self._select_features(prepared._scores)
        prepared._is_prepared = True
        return prepared

    def _compute_chisq_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute chi-squared or Fisher exact test scores."""
        scores = {}

        for col in X.columns:
            x = X[col]

            # Both must be categorical
            x_is_cat = pd.api.types.is_categorical_dtype(x) or pd.api.types.is_object_dtype(x)
            y_is_cat = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)

            if not (x_is_cat and y_is_cat):
                scores[col] = 0.0 if not self.use_pvalue else np.inf
                continue

            # Create contingency table
            try:
                mask = ~(x.isna() | y.isna())
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 5:
                    scores[col] = 0.0 if not self.use_pvalue else np.inf
                    continue

                contingency = pd.crosstab(x_clean, y_clean)

                if self.method == 'chisq':
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                    stat = chi2
                else:  # fisher
                    # Fisher exact only works for 2x2 tables
                    if contingency.shape == (2, 2):
                        oddsratio, p_val = stats.fisher_exact(contingency)
                        stat = -np.log(oddsratio) if oddsratio > 0 else 0.0
                    else:
                        # Fall back to chi-squared for larger tables
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        stat = chi2

                if self.use_pvalue:
                    if p_val == 0 or np.isnan(p_val):
                        scores[col] = np.inf
                    else:
                        scores[col] = -np.log10(p_val)
                else:
                    scores[col] = stat if not np.isnan(stat) else 0.0

            except Exception:
                scores[col] = 0.0 if not self.use_pvalue else np.inf

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on threshold/top_n/top_p."""
        if len(scores) == 0:
            return []

        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            selected = [feat for feat, score in sorted_features if score >= self.threshold]
        elif self.top_n is not None:
            selected = [feat for feat, _ in sorted_features[:self.top_n]]
        else:
            n_keep = max(1, int(len(sorted_features) * self.top_p))
            selected = [feat for feat, _ in sorted_features[:n_keep]]

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to new data."""
        if self.skip:
            return data

        if not self._is_prepared:
            raise ValueError("Step must be prepped before baking")

        keep_cols = self._selected_features + [self.outcome]
        keep_cols = [c for c in keep_cols if c in data.columns]
        return data[keep_cols]


# Helper functions for function-style API
def step_filter_anova(recipe, outcome: str, threshold: Optional[float] = None,
                      top_n: Optional[int] = None, top_p: Optional[float] = None,
                      use_pvalue: bool = True, columns=None, skip: bool = False,
                      id: Optional[str] = None):
    """Add ANOVA filter step to recipe."""
    step = StepFilterAnova(
        outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
        use_pvalue=use_pvalue, columns=columns, skip=skip, id=id
    )
    recipe.steps.append(step)
    return recipe


def step_filter_rf_importance(recipe, outcome: str, threshold: Optional[float] = None,
                               top_n: Optional[int] = None, top_p: Optional[float] = None,
                               trees: int = 100, mtry: Optional[int] = None,
                               min_n: int = 2, random_state: Optional[int] = None,
                               columns=None, skip: bool = False, id: Optional[str] = None):
    """Add Random Forest importance filter step to recipe."""
    step = StepFilterRfImportance(
        outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
        trees=trees, mtry=mtry, min_n=min_n, random_state=random_state,
        columns=columns, skip=skip, id=id
    )
    recipe.steps.append(step)
    return recipe


def step_filter_mutual_info(recipe, outcome: str, threshold: Optional[float] = None,
                             top_n: Optional[int] = None, top_p: Optional[float] = None,
                             n_neighbors: int = 3, random_state: Optional[int] = None,
                             columns=None, skip: bool = False, id: Optional[str] = None):
    """Add mutual information filter step to recipe."""
    step = StepFilterMutualInfo(
        outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
        n_neighbors=n_neighbors, random_state=random_state,
        columns=columns, skip=skip, id=id
    )
    recipe.steps.append(step)
    return recipe


def step_filter_roc_auc(recipe, outcome: str, threshold: Optional[float] = None,
                        top_n: Optional[int] = None, top_p: Optional[float] = None,
                        multiclass_strategy: str = 'ovr', columns=None,
                        skip: bool = False, id: Optional[str] = None):
    """Add ROC AUC filter step to recipe."""
    step = StepFilterRocAuc(
        outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
        multiclass_strategy=multiclass_strategy, columns=columns,
        skip=skip, id=id
    )
    recipe.steps.append(step)
    return recipe


def step_filter_chisq(recipe, outcome: str, threshold: Optional[float] = None,
                      top_n: Optional[int] = None, top_p: Optional[float] = None,
                      method: str = 'chisq', use_pvalue: bool = True,
                      columns=None, skip: bool = False, id: Optional[str] = None):
    """Add chi-squared/Fisher exact test filter step to recipe."""
    step = StepFilterChisq(
        outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
        method=method, use_pvalue=use_pvalue, columns=columns,
        skip=skip, id=id
    )
    recipe.steps.append(step)
    return recipe
