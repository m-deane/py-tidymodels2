"""
Advanced feature selection steps for py-recipes (Phase 3).

These steps provide sophisticated feature selection techniques including:
- VIF-based multicollinearity removal
- P-value based selection from statistical models
- Stability selection
- Leave-One-Feature-Out importance
- Granger causality for feature selection
- Stepwise regression
- Probe feature selection
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings


@dataclass
class StepVif:
    """
    Remove features with high Variance Inflation Factor (VIF).

    VIF measures multicollinearity by quantifying how much the variance of a
    coefficient is inflated due to collinearity with other features. Features
    with VIF > threshold are iteratively removed until all remaining features
    have VIF <= threshold.

    Parameters
    ----------
    threshold : float, default=10.0
        VIF threshold. Features with VIF > threshold are removed.
        Common guidelines: VIF < 5 (stringent), VIF < 10 (moderate)
    columns : selector, optional
        Which columns to check for VIF. If None, uses all numeric columns
    outcome : str, optional
        Name of outcome variable (excluded from VIF calculation)
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_vif(threshold=10.0, outcome='target')
    >>> rec = recipe().step_vif(threshold=5.0)  # Stricter threshold

    Notes
    -----
    VIF for feature i is calculated as: VIF_i = 1 / (1 - R²_i)
    where R²_i is from regressing feature i on all other features.

    VIF = 1: No correlation with other features
    VIF = 5: R² = 0.8 (moderate multicollinearity)
    VIF = 10: R² = 0.9 (high multicollinearity)
    """
    threshold: float = 10.0
    columns: Union[None, str, List[str], Callable] = None
    outcome: Optional[str] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _vif_scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _removed_features: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by calculating VIF and removing high VIF features."""
        if self.skip or not training:
            return self

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Resolve columns to check
        if self.columns is None:
            # Use all numeric columns except outcome
            check_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                         if self.outcome is None or c != self.outcome]
        elif isinstance(self.columns, str):
            check_cols = [self.columns]
        elif callable(self.columns):
            check_cols = self.columns(data)
        else:
            check_cols = self.columns

        if len(check_cols) == 0:
            warnings.warn("No columns to check for VIF")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        # Get data for VIF calculation
        X = data[check_cols].copy()

        # Remove any non-numeric columns
        X = X.select_dtypes(include=[np.number])

        # Handle missing values (drop rows with any NaN)
        X = X.dropna()

        if X.shape[0] < 2 or X.shape[1] < 2:
            warnings.warn("Insufficient data for VIF calculation")
            prepared = replace(self)
            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Iteratively remove features with highest VIF
        remaining_features = list(X.columns)
        removed_features = []
        vif_history = {}

        while len(remaining_features) > 1:
            # Calculate VIF for all remaining features
            X_subset = X[remaining_features].values

            # Check for constant columns or perfect multicollinearity
            if np.linalg.matrix_rank(X_subset) < X_subset.shape[1]:
                # Remove feature with lowest variance
                variances = X[remaining_features].var()
                remove_feature = variances.idxmin()
                vif_history[remove_feature] = float('inf')
                remaining_features.remove(remove_feature)
                removed_features.append(remove_feature)
                continue

            vifs = {}
            max_vif = 0
            max_vif_feature = None

            for i, feature in enumerate(remaining_features):
                try:
                    vif = variance_inflation_factor(X_subset, i)
                    vifs[feature] = vif
                    if vif > max_vif:
                        max_vif = vif
                        max_vif_feature = feature
                except Exception as e:
                    # If VIF calculation fails, assume infinite VIF
                    vifs[feature] = float('inf')
                    max_vif = float('inf')
                    max_vif_feature = feature

            # Store VIF scores
            vif_history.update(vifs)

            # Check if all VIFs are below threshold
            if max_vif <= self.threshold:
                break

            # Remove feature with highest VIF
            if max_vif_feature is not None:
                remaining_features.remove(max_vif_feature)
                removed_features.append(max_vif_feature)

        prepared = replace(self)
        prepared._selected_features = remaining_features
        prepared._vif_scores = vif_history
        prepared._removed_features = removed_features
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VIF-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome (use set to avoid duplicates)
        cols_to_keep = set(self._selected_features)

        # Add outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Keep all columns that weren't checked for VIF
        all_original_cols = set(data.columns)
        checked_cols = set(self._selected_features + self._removed_features)
        unchecked_cols = all_original_cols - checked_cols
        cols_to_keep.update(unchecked_cols)

        # Select columns (preserve order from original data)
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]
        result = result[cols_to_keep_list]

        return result


@dataclass
class StepPvalue:
    """
    Select features based on p-values from statistical models.

    Fits a linear or logistic regression model and selects features with
    p-values below the threshold. Uses proper statistical inference via
    statsmodels for accurate p-value calculation.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    threshold : float, default=0.05
        P-value threshold. Features with p-value < threshold are selected
    model_type : {'auto', 'linear', 'logistic'}, default='auto'
        Type of model to fit. 'auto' detects based on outcome
    columns : selector, optional
        Which columns to test. If None, uses all numeric columns
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_pvalue(outcome='target', threshold=0.05)
    >>> rec = recipe().step_pvalue(outcome='species', model_type='logistic', threshold=0.01)

    Notes
    -----
    This step requires the outcome to be present in the data during prep.
    P-values test the null hypothesis that a feature's coefficient is zero.
    """
    outcome: str
    threshold: float = 0.05
    model_type: Literal['auto', 'linear', 'logistic'] = 'auto'
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _pvalues: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not (0 < self.threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by fitting model and extracting p-values."""
        if self.skip or not training:
            return self

        import statsmodels.api as sm

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to test
        if self.columns is None:
            test_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c != self.outcome]
        elif isinstance(self.columns, str):
            test_cols = [self.columns]
        elif callable(self.columns):
            test_cols = self.columns(data)
        else:
            test_cols = self.columns

        if len(test_cols) == 0:
            warnings.warn("No columns to test for p-values")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        # Get predictor data
        X = data[test_cols].copy()

        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if X_clean.shape[0] < 2:
            warnings.warn("Insufficient data for p-value calculation")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Add intercept
        X_clean = sm.add_constant(X_clean)

        # Determine model type
        if self.model_type == 'auto':
            # Detect based on outcome
            if y_clean.dtype == 'object' or len(y_clean.unique()) < 10:
                model_type = 'logistic'
            else:
                model_type = 'linear'
        else:
            model_type = self.model_type

        # Fit model
        try:
            if model_type == 'logistic':
                model = sm.Logit(y_clean, X_clean)
            else:
                model = sm.OLS(y_clean, X_clean)

            result = model.fit(disp=False)

            # Extract p-values (skip intercept)
            pvalues = {}
            for col in test_cols:
                if col in result.pvalues.index:
                    pvalues[col] = result.pvalues[col]
                else:
                    pvalues[col] = 1.0  # Not in model

            # Select features with p-value < threshold
            selected_features = [col for col, pval in pvalues.items()
                               if pval < self.threshold]

            if len(selected_features) == 0:
                warnings.warn(f"No features have p-value < {self.threshold}. "
                            f"Consider increasing threshold.")
                selected_features = list(X.columns)

            prepared = replace(self)


            prepared._selected_features = selected_features


            prepared._pvalues = pvalues


            prepared._is_prepared = True


            return prepared

        except Exception as e:
            warnings.warn(f"Model fitting failed: {e}. Keeping all features.")
            prepared = replace(self)
            prepared._selected_features = list(X.columns)
            prepared._pvalues = {}
            prepared._is_prepared = True
            return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply p-value based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        # Select columns
        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result


@dataclass
class StepSelectStability:
    """
    Stability selection for robust feature selection.

    Performs feature selection on multiple bootstrap samples of the data and
    selects features that appear frequently across samples. This provides more
    robust feature selection than single-sample methods.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    threshold : float, default=0.8
        Selection frequency threshold (0-1). Features selected in at least
        this fraction of bootstrap samples are kept
    n_bootstrap : int, default=100
        Number of bootstrap samples
    sample_fraction : float, default=0.5
        Fraction of data to use in each bootstrap sample
    estimator : estimator, optional
        Sklearn estimator with feature_importances_ or coef_ attribute.
        If None, uses RandomForest (auto-detects regression/classification)
    n_features_per_bootstrap : int, optional
        Number of features to select in each bootstrap. If None, uses half
    columns : selector, optional
        Which columns to test. If None, uses all numeric columns
    random_state : int, optional
        Random seed for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_select_stability(
    ...     outcome='target', threshold=0.8, n_bootstrap=100
    ... )
    >>> rec = recipe().step_select_stability(
    ...     outcome='target', threshold=0.6, sample_fraction=0.7
    ... )

    Notes
    -----
    Stability selection provides theoretical guarantees on false discovery rate
    when combined with appropriate thresholds. See Meinshausen & Bühlmann (2010).
    """
    outcome: str
    threshold: float = 0.8
    n_bootstrap: int = 100
    sample_fraction: float = 0.5
    estimator: Optional[Any] = None
    n_features_per_bootstrap: Optional[int] = None
    columns: Union[None, str, List[str], Callable] = None
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _selection_frequencies: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not (0 < self.threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if not (0 < self.sample_fraction <= 1):
            raise ValueError(f"sample_fraction must be in (0, 1], got {self.sample_fraction}")
        if self.n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {self.n_bootstrap}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by performing stability selection."""
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to test
        if self.columns is None:
            test_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c != self.outcome]
        elif isinstance(self.columns, str):
            test_cols = [self.columns]
        elif callable(self.columns):
            test_cols = self.columns(data)
        else:
            test_cols = self.columns

        if len(test_cols) == 0:
            warnings.warn("No columns for stability selection")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        X = data[test_cols].copy()
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if X_clean.shape[0] < 2:
            warnings.warn("Insufficient data for stability selection")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Determine estimator
        if self.estimator is None:
            # Auto-detect regression vs classification
            if y_clean.dtype == 'object' or len(y_clean.unique()) < 10:
                estimator = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        else:
            estimator = self.estimator

        # Determine number of features to select per bootstrap
        if self.n_features_per_bootstrap is None:
            n_select = max(1, len(test_cols) // 2)
        else:
            n_select = min(self.n_features_per_bootstrap, len(test_cols))

        # Perform bootstrap stability selection
        np.random.seed(self.random_state)
        selection_counts = {col: 0 for col in test_cols}

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            n_samples = int(X_clean.shape[0] * self.sample_fraction)
            sample_idx = np.random.choice(X_clean.shape[0], size=n_samples, replace=True)

            X_boot = X_clean.iloc[sample_idx]
            y_boot = y_clean.iloc[sample_idx]

            try:
                # Fit model
                est_clone = clone(estimator)
                est_clone.fit(X_boot, y_boot)

                # Get feature importances/coefficients
                if hasattr(est_clone, 'feature_importances_'):
                    importances = est_clone.feature_importances_
                elif hasattr(est_clone, 'coef_'):
                    importances = np.abs(est_clone.coef_).flatten()
                else:
                    continue

                # Select top n_select features
                top_indices = np.argsort(importances)[-n_select:]
                selected_in_boot = [test_cols[idx] for idx in top_indices]

                # Increment counts
                for col in selected_in_boot:
                    selection_counts[col] += 1

            except Exception:
                continue

        # Calculate selection frequencies
        selection_frequencies = {col: count / self.n_bootstrap
                                for col, count in selection_counts.items()}

        # Select features with frequency >= threshold
        selected_features = [col for col, freq in selection_frequencies.items()
                           if freq >= self.threshold]

        if len(selected_features) == 0:
            warnings.warn(f"No features selected with threshold {self.threshold}. "
                        f"Consider lowering threshold or increasing n_bootstrap.")
            selected_features = list(X.columns)

        prepared = replace(self)


        prepared._selected_features = selected_features


        prepared._selection_frequencies = selection_frequencies


        prepared._is_prepared = True


        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply stability-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result


@dataclass
class StepSelectLofo:
    """
    Leave-One-Feature-Out (LOFO) importance for feature selection.

    Measures each feature's importance by comparing model performance with
    and without that feature. More robust than single feature importances
    as it captures interactions and dependencies between features.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    threshold : float, optional
        Minimum importance to keep feature. If None, uses top_n or top_p
    top_n : int, optional
        Keep top N features by importance
    top_p : float, optional
        Keep top proportion of features (e.g., 0.2 for top 20%)
    estimator : estimator, optional
        Sklearn estimator for evaluation. If None, uses RandomForest
    cv : int, default=3
        Number of cross-validation folds for evaluation
    scoring : str, optional
        Scoring metric for sklearn. If None, uses default for estimator
    columns : selector, optional
        Which columns to test. If None, uses all numeric columns
    random_state : int, optional
        Random seed for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_select_lofo(outcome='target', top_n=10)
    >>> rec = recipe().step_select_lofo(
    ...     outcome='target', threshold=0.01, cv=5
    ... )

    Notes
    -----
    LOFO importance for feature i is:
        importance_i = score(all features) - score(all features except i)

    Features with positive importance improve model performance.
    """
    outcome: str
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    estimator: Optional[Any] = None
    cv: int = 3
    scoring: Optional[str] = None
    columns: Union[None, str, List[str], Callable] = None
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _importances: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
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
        """Prepare the step by calculating LOFO importances."""
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to test
        if self.columns is None:
            test_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c != self.outcome]
        elif isinstance(self.columns, str):
            test_cols = [self.columns]
        elif callable(self.columns):
            test_cols = self.columns(data)
        else:
            test_cols = self.columns

        if len(test_cols) == 0:
            warnings.warn("No columns for LOFO importance")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        X = data[test_cols].copy()
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx].values
        y_clean = y[valid_idx].values

        if X_clean.shape[0] < 2:
            warnings.warn("Insufficient data for LOFO importance")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Determine estimator
        if self.estimator is None:
            if y.dtype == 'object' or len(np.unique(y_clean)) < 10:
                estimator = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        else:
            estimator = self.estimator

        # Baseline score with all features
        try:
            baseline_score = cross_val_score(
                clone(estimator), X_clean, y_clean,
                cv=self.cv, scoring=self.scoring, n_jobs=-1
            ).mean()
        except Exception as e:
            warnings.warn(f"Baseline evaluation failed: {e}. Keeping all features.")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._importances = {}
            prepared._is_prepared = True
            return prepared

        # Calculate LOFO importance for each feature
        importances = {}
        for i, col in enumerate(test_cols):
            # Create dataset without this feature
            mask = np.ones(X_clean.shape[1], dtype=bool)
            mask[i] = False
            X_lofo = X_clean[:, mask]

            try:
                lofo_score = cross_val_score(
                    clone(estimator), X_lofo, y_clean,
                    cv=self.cv, scoring=self.scoring, n_jobs=-1
                ).mean()

                # Importance = performance drop when feature removed
                importances[col] = baseline_score - lofo_score
            except Exception:
                importances[col] = 0.0

        # Select features based on importance
        if self.threshold is not None:
            selected_features = [col for col, imp in importances.items()
                               if imp >= self.threshold]
        elif self.top_n is not None:
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [col for col, _ in sorted_features[:self.top_n]]
        else:  # top_p
            n_select = max(1, int(len(importances) * self.top_p))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [col for col, _ in sorted_features[:n_select]]

        if len(selected_features) == 0:
            warnings.warn("No features selected by LOFO. Keeping all features.")
            selected_features = list(X.columns)

        prepared = replace(self)


        prepared._selected_features = selected_features


        prepared._importances = importances


        prepared._is_prepared = True


        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply LOFO-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result


@dataclass
class StepSelectGranger:
    """
    Select features using Granger causality test.

    Tests if past values of each feature help predict the outcome, beyond what
    the outcome's own history provides. Features that Granger-cause the outcome
    (p-value < alpha) are selected.

    Parameters
    ----------
    outcome : str
        Name of outcome variable (must be time series)
    max_lag : int, default=5
        Maximum number of lags to test
    test : str, default='ssr_ftest'
        Test type: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
    alpha : float, default=0.05
        Significance level for Granger causality test
    columns : selector, optional
        Which columns to test. If None, uses all numeric columns
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_select_granger(
    ...     outcome='sales', max_lag=5, alpha=0.05
    ... )
    >>> rec = recipe().step_select_granger(
    ...     outcome='price', max_lag=10, test='ssr_chi2test'
    ... )

    Notes
    -----
    Granger causality tests if X helps predict Y beyond Y's own history.
    It does NOT imply true causality, only predictive precedence.

    Data should be stationary for valid Granger causality tests.
    """
    outcome: str
    max_lag: int = 5
    test: str = 'ssr_ftest'
    alpha: float = 0.05
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _pvalues: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")
        if not (0 < self.alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        valid_tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
        if self.test not in valid_tests:
            raise ValueError(f"test must be one of {valid_tests}, got {self.test}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by running Granger causality tests."""
        if self.skip or not training:
            return self

        from statsmodels.tsa.stattools import grangercausalitytests

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to test
        if self.columns is None:
            test_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c != self.outcome]
        elif isinstance(self.columns, str):
            test_cols = [self.columns]
        elif callable(self.columns):
            test_cols = self.columns(data)
        else:
            test_cols = self.columns

        if len(test_cols) == 0:
            warnings.warn("No columns for Granger causality test")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        X = data[test_cols].copy()
        X = X.select_dtypes(include=[np.number])

        # Test each feature
        pvalues = {}
        selected_features = []

        for col in X.columns:
            # Create test data [outcome, feature]
            test_data = pd.DataFrame({
                self.outcome: y,
                col: X[col]
            }).dropna()

            if len(test_data) < 2 * self.max_lag + 1:
                warnings.warn(f"Insufficient data for Granger test on {col}")
                pvalues[col] = 1.0
                continue

            try:
                # Run Granger causality test
                result = grangercausalitytests(
                    test_data,
                    maxlag=self.max_lag,
                    verbose=False
                )

                # Get minimum p-value across all lags
                min_pvalue = 1.0
                for lag in range(1, self.max_lag + 1):
                    pval = result[lag][0][self.test][1]
                    if pval < min_pvalue:
                        min_pvalue = pval

                pvalues[col] = min_pvalue

                # Select if any lag is significant
                if min_pvalue < self.alpha:
                    selected_features.append(col)

            except Exception as e:
                warnings.warn(f"Granger test failed for {col}: {e}")
                pvalues[col] = 1.0

        if len(selected_features) == 0:
            warnings.warn(f"No features Granger-cause {self.outcome}. Keeping all.")
            selected_features = list(X.columns)

        prepared = replace(self)


        prepared._selected_features = selected_features


        prepared._pvalues = pvalues


        prepared._is_prepared = True


        return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Granger-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result


@dataclass
class StepSelectStepwise:
    """
    Stepwise feature selection based on AIC/BIC.

    Performs forward, backward, or bidirectional stepwise selection by
    iteratively adding/removing features based on information criteria (AIC/BIC).

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    direction : {'forward', 'backward', 'both'}, default='both'
        Direction of stepwise selection
    criterion : {'aic', 'bic'}, default='aic'
        Information criterion for model comparison
    max_steps : int, optional
        Maximum number of steps. If None, continues until no improvement
    model_type : {'auto', 'linear', 'logistic'}, default='auto'
        Type of model to fit
    columns : selector, optional
        Which columns to consider. If None, uses all numeric columns
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_select_stepwise(
    ...     outcome='target', direction='forward', criterion='aic'
    ... )
    >>> rec = recipe().step_select_stepwise(
    ...     outcome='target', direction='both', criterion='bic'
    ... )

    Notes
    -----
    Stepwise selection can be unstable and may overfit. Consider using
    stability selection or cross-validation instead for more robust results.

    AIC = 2k - 2ln(L) where k is number of parameters, L is likelihood
    BIC = k*ln(n) - 2ln(L) where n is sample size
    """
    outcome: str
    direction: Literal['forward', 'backward', 'both'] = 'both'
    criterion: Literal['aic', 'bic'] = 'aic'
    max_steps: Optional[int] = None
    model_type: Literal['auto', 'linear', 'logistic'] = 'auto'
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _final_criterion: float = field(default=float('inf'), init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by performing stepwise selection."""
        if self.skip or not training:
            return self

        import statsmodels.api as sm

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to consider
        if self.columns is None:
            candidate_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                             if c != self.outcome]
        elif isinstance(self.columns, str):
            candidate_cols = [self.columns]
        elif callable(self.columns):
            candidate_cols = self.columns(data)
        else:
            candidate_cols = self.columns

        if len(candidate_cols) == 0:
            warnings.warn("No columns for stepwise selection")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        X = data[candidate_cols].copy()
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if X_clean.shape[0] < 2:
            warnings.warn("Insufficient data for stepwise selection")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Determine model type
        if self.model_type == 'auto':
            if y_clean.dtype == 'object' or len(y_clean.unique()) < 10:
                model_type = 'logistic'
            else:
                model_type = 'linear'
        else:
            model_type = self.model_type

        def get_criterion(features):
            """Fit model with given features and return AIC/BIC."""
            if len(features) == 0:
                return float('inf')

            X_subset = sm.add_constant(X_clean[features])

            try:
                if model_type == 'logistic':
                    model = sm.Logit(y_clean, X_subset)
                else:
                    model = sm.OLS(y_clean, X_subset)

                result = model.fit(disp=False)
                return result.bic if self.criterion == 'bic' else result.aic
            except Exception:
                return float('inf')

        # Perform stepwise selection
        if self.direction == 'forward':
            selected = self._forward_selection(candidate_cols, get_criterion)
        elif self.direction == 'backward':
            selected = self._backward_elimination(candidate_cols, get_criterion)
        else:  # both
            selected = self._bidirectional_selection(candidate_cols, get_criterion)

        if len(selected) == 0:
            warnings.warn("No features selected by stepwise. Keeping all.")
            selected = list(X.columns)

        prepared = replace(self)


        prepared._selected_features = selected
        prepared._final_criterion = get_criterion(selected)
        prepared._is_prepared = True
        return prepared

    def _forward_selection(self, candidates, get_criterion):
        """Forward stepwise selection."""
        selected = []
        remaining = list(candidates)
        current_criterion = float('inf')

        steps = 0
        while remaining and (self.max_steps is None or steps < self.max_steps):
            best_feature = None
            best_criterion = current_criterion

            for feature in remaining:
                test_features = selected + [feature]
                criterion = get_criterion(test_features)

                if criterion < best_criterion:
                    best_criterion = criterion
                    best_feature = feature

            if best_feature is None:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)
            current_criterion = best_criterion
            steps += 1

        return selected

    def _backward_elimination(self, candidates, get_criterion):
        """Backward stepwise elimination."""
        selected = list(candidates)
        current_criterion = get_criterion(selected)

        steps = 0
        while len(selected) > 0 and (self.max_steps is None or steps < self.max_steps):
            worst_feature = None
            best_criterion = current_criterion

            for feature in selected:
                test_features = [f for f in selected if f != feature]
                criterion = get_criterion(test_features)

                if criterion < best_criterion:
                    best_criterion = criterion
                    worst_feature = feature

            if worst_feature is None:
                break

            selected.remove(worst_feature)
            current_criterion = best_criterion
            steps += 1

        return selected

    def _bidirectional_selection(self, candidates, get_criterion):
        """Bidirectional stepwise selection."""
        selected = []
        remaining = list(candidates)
        current_criterion = float('inf')

        steps = 0
        while (self.max_steps is None or steps < self.max_steps):
            # Try adding a feature
            best_add = None
            best_add_criterion = current_criterion

            for feature in remaining:
                test_features = selected + [feature]
                criterion = get_criterion(test_features)

                if criterion < best_add_criterion:
                    best_add_criterion = criterion
                    best_add = feature

            # Try removing a feature
            best_remove = None
            best_remove_criterion = current_criterion

            for feature in selected:
                test_features = [f for f in selected if f != feature]
                criterion = get_criterion(test_features)

                if criterion < best_remove_criterion:
                    best_remove_criterion = criterion
                    best_remove = feature

            # Choose best action
            if best_add_criterion < best_remove_criterion:
                if best_add is not None:
                    selected.append(best_add)
                    remaining.remove(best_add)
                    current_criterion = best_add_criterion
                    steps += 1
                else:
                    break
            elif best_remove is not None:
                selected.remove(best_remove)
                remaining.append(best_remove)
                current_criterion = best_remove_criterion
                steps += 1
            else:
                break

        return selected

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply stepwise-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result


@dataclass
class StepSelectProbe:
    """
    Probe feature selection using random features.

    Creates random "probe" features and compares real feature importances to
    probe importances. Selects only real features with importance greater than
    the maximum probe importance. This ensures selected features are better
    than random.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    n_probes : int, default=10
        Number of random probe features to create
    estimator : estimator, optional
        Sklearn estimator with feature_importances_ or coef_ attribute.
        If None, uses RandomForest
    threshold_percentile : float, default=100
        Percentile of probe importances to use as threshold (0-100).
        100 = maximum probe importance, 95 = 95th percentile, etc.
    columns : selector, optional
        Which columns to test. If None, uses all numeric columns
    random_state : int, optional
        Random seed for reproducibility
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> rec = recipe().step_select_probe(
    ...     outcome='target', n_probes=10
    ... )
    >>> rec = recipe().step_select_probe(
    ...     outcome='target', n_probes=20, threshold_percentile=95
    ... )

    Notes
    -----
    Probe features are random permutations of existing features or random noise.
    Real features must have importance exceeding the threshold to be selected.

    This method provides a data-driven threshold rather than arbitrary cutoffs.
    """
    outcome: str
    n_probes: int = 10
    estimator: Optional[Any] = None
    threshold_percentile: float = 100
    columns: Union[None, str, List[str], Callable] = None
    random_state: Optional[int] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _importances: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _probe_threshold: float = field(default=0.0, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.n_probes < 1:
            raise ValueError(f"n_probes must be >= 1, got {self.n_probes}")
        if not (0 <= self.threshold_percentile <= 100):
            raise ValueError(f"threshold_percentile must be in [0, 100], got {self.threshold_percentile}")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare the step by creating probes and selecting features."""
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        y = data[self.outcome]

        # Resolve columns to test
        if self.columns is None:
            test_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c != self.outcome]
        elif isinstance(self.columns, str):
            test_cols = [self.columns]
        elif callable(self.columns):
            test_cols = self.columns(data)
        else:
            test_cols = self.columns

        if len(test_cols) == 0:
            warnings.warn("No columns for probe selection")
            prepared = replace(self)

            prepared._selected_features = []

            prepared._is_prepared = True

            return prepared

        X = data[test_cols].copy()
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if X_clean.shape[0] < 2:
            warnings.warn("Insufficient data for probe selection")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._is_prepared = True
            return prepared

        # Create probe features (random permutations)
        np.random.seed(self.random_state)
        probe_features = {}

        for i in range(self.n_probes):
            # Randomly select a real feature to permute
            base_feature = np.random.choice(X_clean.columns)
            probe_name = f"__probe_{i}__"
            # Permute values to destroy relationship with outcome
            probe_features[probe_name] = np.random.permutation(X_clean[base_feature].values)

        # Combine real features with probes
        X_with_probes = X_clean.copy()
        for name, values in probe_features.items():
            X_with_probes[name] = values

        # Determine estimator
        if self.estimator is None:
            if y_clean.dtype == 'object' or len(y_clean.unique()) < 10:
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        else:
            estimator = self.estimator

        # Fit model
        try:
            estimator.fit(X_with_probes, y_clean)

            # Get feature importances
            if hasattr(estimator, 'feature_importances_'):
                importances_array = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                importances_array = np.abs(estimator.coef_).flatten()
            else:
                warnings.warn("Estimator has no feature_importances_ or coef_")
                prepared = replace(self)
                prepared._selected_features = list(X.columns)
                prepared._is_prepared = True
                return prepared

            # Map importances to feature names
            all_features = list(X_with_probes.columns)
            importances_dict = dict(zip(all_features, importances_array))

            # Separate real and probe importances
            real_importances = {k: v for k, v in importances_dict.items()
                              if not k.startswith('__probe_')}
            probe_importances = [v for k, v in importances_dict.items()
                               if k.startswith('__probe_')]

            # Calculate threshold from probes
            if len(probe_importances) > 0:
                threshold = np.percentile(probe_importances, self.threshold_percentile)
            else:
                threshold = 0.0

            # Select features with importance > threshold
            selected_features = [col for col, imp in real_importances.items()
                               if imp > threshold]

            if len(selected_features) == 0:
                warnings.warn(f"No features exceed probe threshold {threshold:.4f}. Keeping all.")
                selected_features = list(X.columns)

            prepared = replace(self)


            prepared._selected_features = selected_features


            prepared._importances = real_importances


            prepared._probe_threshold = threshold


            prepared._is_prepared = True


            return prepared

        except Exception as e:
            warnings.warn(f"Probe selection failed: {e}. Keeping all features.")
            prepared = replace(self)

            prepared._selected_features = list(X.columns)
            prepared._importances = {}
            prepared._probe_threshold = 0.0
            prepared._is_prepared = True
            return prepared

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply probe-based feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data.copy()

        result = data.copy()

        # Keep selected features and outcome
        cols_to_keep = self._selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        result = result[[c for c in cols_to_keep if c in result.columns]]

        return result
