"""
Practical Selection Steps (Phase 4)

Regularization-based, statistical, and time series selection methods
using standard dependencies (sklearn, statsmodels, scipy).
"""

from dataclasses import dataclass, field, replace
from typing import List, Union, Optional, Callable, Literal
import pandas as pd
import numpy as np
import warnings


@dataclass
class StepSelectLasso:
    """
    Select features using Lasso (L1 regularization) regression.

    Features with non-zero coefficients after L1 regularization are kept.
    Lasso performs automatic feature selection by shrinking coefficients to zero.

    Parameters
    ----------
    outcome : str
        Outcome variable name (required)
    alpha : float
        Regularization strength (default: 1.0). Larger values = more regularization
    threshold : float or None
        Keep features with |coefficient| >= threshold (default: None = all non-zero)
    top_n : int or None
        Keep top N features by |coefficient| (default: None)
    normalize : bool
        Normalize features before fitting (default: True)
    max_iter : int
        Maximum iterations for solver (default: 1000)
    columns : list, str, callable, or None
        Columns to consider (default: None = all numeric predictors)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    outcome: str
    alpha: float = 1.0
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    normalize: bool = True
    max_iter: int = 1000
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _coefficients: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by fitting Lasso and selecting features."""
        from sklearn.linear_model import Lasso

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome in columns:
                columns.remove(self.outcome)
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_lasso")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Get data
        X = data[columns].copy()
        y = data[self.outcome].copy()

        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            warnings.warn("No valid data after removing NaN in step_select_lasso")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Fit Lasso (normalize parameter removed in sklearn 1.2+)
        # Note: If normalization is needed, use step_normalize() before this step
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        lasso.fit(X_clean, y_clean)

        # Get coefficients
        coefficients = dict(zip(columns, lasso.coef_))

        # Select features
        if self.top_n is not None:
            # Top N by absolute coefficient
            sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            selected_features = [feat for feat, coef in sorted_features[:self.top_n] if coef != 0]
        elif self.threshold is not None:
            # Threshold on absolute coefficient
            selected_features = [feat for feat, coef in coefficients.items()
                               if abs(coef) >= self.threshold]
        else:
            # All non-zero coefficients
            selected_features = [feat for feat, coef in coefficients.items()
                               if coef != 0]

        if not selected_features:
            warnings.warn("No features selected by Lasso - all coefficients were zero")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._coefficients = coefficients
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features
        cols_to_keep = set(self._selected_features)

        # Always keep outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectRidge:
    """
    Select features using Ridge (L2 regularization) regression.

    Features ranked by absolute Ridge coefficients. Ridge doesn't zero out
    coefficients but provides stable importance estimates.

    Parameters
    ----------
    outcome : str
        Outcome variable name (required)
    alpha : float
        Regularization strength (default: 1.0). Larger values = more regularization
    threshold : float or None
        Keep features with |coefficient| >= threshold (default: None)
    top_n : int or None
        Keep top N features by |coefficient| (default: None)
    normalize : bool
        Normalize features before fitting (default: True)
    columns : list, str, callable, or None
        Columns to consider (default: None = all numeric predictors)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    outcome: str
    alpha: float = 1.0
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    normalize: bool = True
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _coefficients: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by fitting Ridge and selecting features."""
        from sklearn.linear_model import Ridge

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome in columns:
                columns.remove(self.outcome)
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_ridge")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Get data
        X = data[columns].copy()
        y = data[self.outcome].copy()

        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            warnings.warn("No valid data after removing NaN in step_select_ridge")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Fit Ridge
        # normalize parameter removed in sklearn 1.2+
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(X_clean, y_clean)

        # Get coefficients
        coefficients = dict(zip(columns, ridge.coef_))

        # Select features
        if self.top_n is not None:
            # Top N by absolute coefficient
            sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            selected_features = [feat for feat, coef in sorted_features[:self.top_n]]
        elif self.threshold is not None:
            # Threshold on absolute coefficient
            selected_features = [feat for feat, coef in coefficients.items()
                               if abs(coef) >= self.threshold]
        else:
            # All features (Ridge doesn't zero out)
            selected_features = columns

        if not selected_features:
            warnings.warn("No features selected by Ridge threshold")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._coefficients = coefficients
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features
        cols_to_keep = set(self._selected_features)

        # Always keep outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectElasticNet:
    """
    Select features using Elastic Net (L1 + L2 regularization).

    Combines Lasso and Ridge - can zero out coefficients (like Lasso)
    while maintaining stability (like Ridge).

    Parameters
    ----------
    outcome : str
        Outcome variable name (required)
    alpha : float
        Overall regularization strength (default: 1.0)
    l1_ratio : float
        L1 vs L2 balance (default: 0.5). 1.0 = pure Lasso, 0.0 = pure Ridge
    threshold : float or None
        Keep features with |coefficient| >= threshold (default: None = all non-zero)
    top_n : int or None
        Keep top N features by |coefficient| (default: None)
    normalize : bool
        Normalize features before fitting (default: True)
    max_iter : int
        Maximum iterations for solver (default: 1000)
    columns : list, str, callable, or None
        Columns to consider (default: None = all numeric predictors)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    outcome: str
    alpha: float = 1.0
    l1_ratio: float = 0.5
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    normalize: bool = True
    max_iter: int = 1000
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _coefficients: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by fitting Elastic Net and selecting features."""
        from sklearn.linear_model import ElasticNet

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome in columns:
                columns.remove(self.outcome)
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_elastic_net")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Get data
        X = data[columns].copy()
        y = data[self.outcome].copy()

        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            warnings.warn("No valid data after removing NaN in step_select_elastic_net")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Fit Elastic Net (normalize parameter removed in sklearn 1.2+)
        enet = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter
        )
        enet.fit(X_clean, y_clean)

        # Get coefficients
        coefficients = dict(zip(columns, enet.coef_))

        # Select features
        if self.top_n is not None:
            # Top N by absolute coefficient
            sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            selected_features = [feat for feat, coef in sorted_features[:self.top_n] if coef != 0]
        elif self.threshold is not None:
            # Threshold on absolute coefficient
            selected_features = [feat for feat, coef in coefficients.items()
                               if abs(coef) >= self.threshold]
        else:
            # All non-zero coefficients
            selected_features = [feat for feat, coef in coefficients.items()
                               if coef != 0]

        if not selected_features:
            warnings.warn("No features selected by Elastic Net - all coefficients were zero")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._coefficients = coefficients
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features
        cols_to_keep = set(self._selected_features)

        # Always keep outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectUnivariate:
    """
    Select features using univariate statistical tests.

    Tests each feature independently against the outcome using
    f_classif, f_regression, chi2, or mutual_info.

    Parameters
    ----------
    outcome : str
        Outcome variable name (required)
    score_func : str
        Scoring function: 'f_classif', 'f_regression', 'chi2', 'mutual_info_classif', 'mutual_info_regression'
        (default: 'f_regression')
    threshold : float or None
        Keep features with score >= threshold (default: None)
    top_n : int or None
        Keep top N features by score (default: None)
    top_p : float or None
        Keep top proportion of features (e.g., 0.5 for top 50%) (default: None)
    columns : list, str, callable, or None
        Columns to consider (default: None = all numeric predictors)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    outcome: str
    score_func: str = 'f_regression'
    threshold: Optional[float] = None
    top_n: Optional[int] = None
    top_p: Optional[float] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _scores: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by computing univariate scores."""
        from sklearn.feature_selection import (
            f_classif, f_regression, chi2,
            mutual_info_classif, mutual_info_regression
        )

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome in columns:
                columns.remove(self.outcome)
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_univariate")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Get data
        X = data[columns].copy()
        y = data[self.outcome].copy()

        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            warnings.warn("No valid data after removing NaN in step_select_univariate")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Select score function
        score_funcs = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'chi2': chi2,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression
        }

        if self.score_func not in score_funcs:
            raise ValueError(f"Unknown score_func: {self.score_func}")

        func = score_funcs[self.score_func]

        # Compute scores
        if self.score_func == 'chi2':
            # chi2 requires non-negative features
            X_clean = X_clean - X_clean.min() + 1e-10

        # Some scoring functions return (scores, pvalues), others just scores
        result = func(X_clean, y_clean)
        if isinstance(result, tuple):
            scores, _ = result
        else:
            scores = result

        # Handle NaN scores
        scores = np.nan_to_num(scores, nan=0.0)

        scores_dict = dict(zip(columns, scores))

        # Select features
        if self.top_n is not None:
            # Top N by score
            sorted_features = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in sorted_features[:self.top_n]]
        elif self.top_p is not None:
            # Top proportion
            n_select = max(1, int(len(columns) * self.top_p))
            sorted_features = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in sorted_features[:n_select]]
        elif self.threshold is not None:
            # Threshold on score
            selected_features = [feat for feat, score in scores_dict.items()
                               if score >= self.threshold]
        else:
            # All features
            selected_features = columns

        if not selected_features:
            warnings.warn("No features selected by univariate test")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._scores = scores_dict
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features
        cols_to_keep = set(self._selected_features)

        # Always keep outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectVarianceThreshold:
    """
    Remove low-variance features.

    Features with variance below threshold are removed. Useful for
    removing near-constant features.

    Parameters
    ----------
    threshold : float
        Variance threshold (default: 0.0 = remove only constants)
    columns : list, str, callable, or None
        Columns to consider (default: None = all numeric)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    threshold: float = 0.0
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _variances: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by computing variances."""
        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_variance_threshold")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Compute variances
        variances = {}
        selected_features = []

        for col in columns:
            if col in data.columns:
                var = data[col].var()
                variances[col] = var
                if var > self.threshold:
                    selected_features.append(col)

        if not selected_features:
            warnings.warn("No features passed variance threshold")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._variances = variances
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features (preserve all non-numeric columns)
        cols_to_keep = set(self._selected_features)

        # Keep all non-numeric columns
        for col in data.columns:
            if col not in self._variances:
                cols_to_keep.add(col)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectStationary:
    """
    Select stationary time series features using ADF test.

    Tests each feature for stationarity using Augmented Dickey-Fuller test.
    Only stationary features (p-value < alpha) are kept.

    Parameters
    ----------
    alpha : float
        Significance level for ADF test (default: 0.05)
    max_lag : int or None
        Maximum lag for ADF test (default: None = auto)
    columns : list, str, callable, or None
        Columns to test (default: None = all numeric)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    alpha: float = 0.05
    max_lag: Optional[int] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _pvalues: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by testing stationarity."""
        from statsmodels.tsa.stattools import adfuller

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_stationary")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Test each feature
        pvalues = {}
        selected_features = []

        for col in columns:
            if col in data.columns:
                series = data[col].dropna()

                if len(series) < 12:
                    warnings.warn(f"Insufficient data for ADF test on {col}")
                    continue

                try:
                    result = adfuller(series, maxlag=self.max_lag)
                    pvalue = result[1]
                    pvalues[col] = pvalue

                    # Stationary if p-value < alpha
                    if pvalue < self.alpha:
                        selected_features.append(col)
                except Exception as e:
                    warnings.warn(f"ADF test failed for {col}: {e}")
                    continue

        if not selected_features:
            warnings.warn("No stationary features found")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._pvalues = pvalues
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features (preserve all non-tested columns)
        cols_to_keep = set(self._selected_features)

        # Keep all columns not tested
        for col in data.columns:
            if col not in self._pvalues:
                cols_to_keep.add(col)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectCointegration:
    """
    Select features cointegrated with outcome (time series).

    Tests for cointegration between each feature and the outcome using
    Engle-Granger test. Features with p-value < alpha are kept.

    Parameters
    ----------
    outcome : str
        Outcome variable name (required)
    alpha : float
        Significance level (default: 0.05)
    max_lag : int or None
        Maximum lag for cointegration test (default: None = auto)
    columns : list, str, callable, or None
        Columns to test (default: None = all numeric predictors)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    outcome: str
    alpha: float = 0.05
    max_lag: Optional[int] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _pvalues: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by testing cointegration."""
        from statsmodels.tsa.stattools import coint

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome in columns:
                columns.remove(self.outcome)
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_cointegration")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Get outcome
        y = data[self.outcome].dropna()

        if len(y) < 12:
            warnings.warn("Insufficient data for cointegration test")
            prepared = replace(self)
            prepared._selected_features = columns[:1] if columns else []
            prepared._is_prepared = True
            return prepared

        # Test each feature
        pvalues = {}
        selected_features = []

        for col in columns:
            if col in data.columns:
                x = data[col].dropna()

                # Align series
                common_index = y.index.intersection(x.index)
                if len(common_index) < 12:
                    warnings.warn(f"Insufficient overlapping data for {col}")
                    continue

                y_aligned = y.loc[common_index]
                x_aligned = x.loc[common_index]

                try:
                    # Cointegration test
                    _, pvalue, _ = coint(y_aligned, x_aligned, maxlag=self.max_lag)
                    pvalues[col] = pvalue

                    # Cointegrated if p-value < alpha
                    if pvalue < self.alpha:
                        selected_features.append(col)
                except Exception as e:
                    warnings.warn(f"Cointegration test failed for {col}: {e}")
                    continue

        if not selected_features:
            warnings.warn("No cointegrated features found")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._pvalues = pvalues
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features
        cols_to_keep = set(self._selected_features)

        # Always keep outcome if present
        if self.outcome and self.outcome in result.columns:
            cols_to_keep.add(self.outcome)

        # Keep all columns not tested
        for col in data.columns:
            if col not in self._pvalues and col != self.outcome:
                if col in data.select_dtypes(exclude=[np.number]).columns:
                    cols_to_keep.add(col)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]


@dataclass
class StepSelectSeasonal:
    """
    Select features with significant seasonal patterns.

    Tests for seasonality using FFT (Fast Fourier Transform) to detect
    periodic patterns. Features with strong seasonality are kept.

    Parameters
    ----------
    period : int or None
        Expected seasonal period (e.g., 12 for monthly, 7 for daily)
        If None, automatically detects dominant period
    threshold : float
        Minimum spectral power ratio to consider seasonal (default: 0.1)
    columns : list, str, callable, or None
        Columns to test (default: None = all numeric)
    skip : bool
        Skip this step during bake (default: False)
    id : str or None
        Unique identifier for this step
    """
    period: Optional[int] = None
    threshold: float = 0.1
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _seasonal_strength: dict = field(default_factory=dict, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def prep(self, data: pd.DataFrame, training: bool = True):
        """Prepare by testing seasonality."""
        from scipy.fft import fft, fftfreq

        if not training:
            return self

        # Get columns to process
        if self.columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        elif callable(self.columns):
            columns = self.columns(data)
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns

        if not columns:
            warnings.warn("No columns to process in step_select_seasonal")
            prepared = replace(self)
            prepared._selected_features = []
            prepared._is_prepared = True
            return prepared

        # Test each feature
        seasonal_strength = {}
        selected_features = []

        for col in columns:
            if col in data.columns:
                series = data[col].dropna()

                if len(series) < 20:
                    warnings.warn(f"Insufficient data for seasonality test on {col}")
                    continue

                try:
                    # FFT
                    n = len(series)
                    yf = fft(series.values)
                    xf = fftfreq(n, 1)[:n//2]
                    power = 2.0/n * np.abs(yf[0:n//2])

                    # Find dominant frequency
                    if self.period is not None:
                        # Check specific period
                        target_freq = 1.0 / self.period
                        idx = np.argmin(np.abs(xf - target_freq))
                        strength = power[idx] / np.mean(power)
                    else:
                        # Strongest frequency (excluding DC component)
                        strength = np.max(power[1:]) / np.mean(power[1:])

                    seasonal_strength[col] = strength

                    if strength > self.threshold:
                        selected_features.append(col)
                except Exception as e:
                    warnings.warn(f"Seasonality test failed for {col}: {e}")
                    continue

        if not selected_features:
            warnings.warn("No seasonal features found")
            selected_features = columns[:1] if columns else []

        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._seasonal_strength = seasonal_strength
        prepared._is_prepared = True
        return prepared

    def bake(self, data: pd.DataFrame, training: bool = False):
        """Apply feature selection."""
        if not self._is_prepared:
            raise RuntimeError("Step must be prepped before baking")

        if self.skip:
            return data

        result = data.copy()

        # Select features (preserve all non-tested columns)
        cols_to_keep = set(self._selected_features)

        # Keep all columns not tested
        for col in data.columns:
            if col not in self._seasonal_strength:
                cols_to_keep.add(col)

        # Preserve column order
        cols_to_keep_list = [c for c in data.columns if c in cols_to_keep]

        return result[cols_to_keep_list]
