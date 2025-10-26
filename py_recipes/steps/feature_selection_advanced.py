"""Advanced feature selection steps for py-recipes"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression


@dataclass
class StepVip:
    """Variable Importance in Projection (VIP) feature selection.

    Calculates VIP scores from a PLS model and selects features based on threshold.
    VIP scores measure the importance of each variable in the projection used in a PLS model.

    Args:
        threshold: VIP threshold for feature selection (default 1.0)
            Variables with VIP > threshold are kept
        num_comp: Number of PLS components to use (default 2)
        outcome: Name of outcome column (required for supervised selection)

    Example:
        >>> rec = recipe().step_vip(outcome='y', threshold=1.0, num_comp=2)
    """
    threshold: float = 1.0
    num_comp: int = 2
    outcome: Optional[str] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepVip":
        """Prepare VIP feature selection."""
        from sklearn.cross_decomposition import PLSRegression

        if self.outcome is None:
            raise ValueError("outcome must be specified for VIP feature selection")

        # Separate predictors and outcome
        X = data.drop(columns=[self.outcome])
        y = data[self.outcome]

        # Fit PLS model
        pls = PLSRegression(n_components=self.num_comp)
        pls.fit(X, y)

        # Calculate VIP scores
        # VIP_j = sqrt(p * sum((w_j^a)^2 * SSY_a) / sum(SSY_a))
        # where p is number of variables, w is weight, SSY is sum of squares explained

        W = pls.x_weights_  # Weight matrix (p x num_comp)
        T = pls.x_scores_   # Score matrix (n x num_comp)

        # Calculate sum of squares for each component
        ss = np.sum(T ** 2, axis=0)

        # Calculate VIP scores
        p = X.shape[1]
        vip_scores = np.sqrt(p * np.sum((W ** 2) * ss, axis=1) / np.sum(ss))

        # Select features based on threshold
        selected_features = [col for col, score in zip(X.columns, vip_scores)
                           if score > self.threshold]

        # Store VIP scores
        vip_dict = dict(zip(X.columns, vip_scores))

        if len(selected_features) == 0:
            raise ValueError(f"No features have VIP > {self.threshold}. "
                           f"Consider lowering threshold or changing num_comp.")

        return PreparedStepVip(
            threshold=self.threshold,
            num_comp=self.num_comp,
            outcome=self.outcome,
            selected_features=selected_features,
            vip_scores=vip_dict
        )


@dataclass
class PreparedStepVip:
    """Prepared VIP feature selection step."""
    threshold: float
    num_comp: int
    outcome: str
    selected_features: List[str]
    vip_scores: Dict[str, float]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VIP feature selection."""
        result = data.copy()

        # Keep outcome if present
        cols_to_keep = self.selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        # Select only the features with VIP > threshold
        result = result[[col for col in cols_to_keep if col in result.columns]]

        return result


@dataclass
class StepBoruta:
    """Boruta all-relevant feature selection.

    Uses Boruta algorithm to identify all features that are statistically relevant
    to the outcome. Creates shadow features and compares real feature importance
    to maximum shadow importance.

    Args:
        outcome: Name of outcome column (required)
        max_iter: Maximum number of iterations (default 100)
        random_state: Random seed for reproducibility (default None)
        perc: Percentile of shadow feature importance to compare against (default 100)
        alpha: P-value threshold for feature importance test (default 0.05)

    Example:
        >>> rec = recipe().step_boruta(outcome='y', max_iter=100)
    """
    outcome: str
    max_iter: int = 100
    random_state: Optional[int] = None
    perc: int = 100
    alpha: float = 0.05

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepBoruta":
        """Prepare Boruta feature selection."""
        try:
            from boruta import BorutaPy
            use_boruta_py = True
        except ImportError:
            use_boruta_py = False

        # Separate predictors and outcome
        X = data.drop(columns=[self.outcome])
        y = data[self.outcome]

        if use_boruta_py:
            # Use BorutaPy package if available
            # Determine if classification or regression
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                estimator = RandomForestClassifier(
                    n_jobs=-1,
                    max_depth=5,
                    random_state=self.random_state
                )
            else:
                estimator = RandomForestRegressor(
                    n_jobs=-1,
                    max_depth=5,
                    random_state=self.random_state
                )

            # Run Boruta
            boruta_selector = BorutaPy(
                estimator=estimator,
                n_estimators='auto',
                max_iter=self.max_iter,
                perc=self.perc,
                alpha=self.alpha,
                random_state=self.random_state
            )
            boruta_selector.fit(X.values, y.values)

            # Get selected features
            selected_mask = boruta_selector.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature rankings
            feature_ranks = dict(zip(X.columns, boruta_selector.ranking_))

        else:
            # Manual implementation using Random Forest importance
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )

            # Fit and get importances
            rf.fit(X.values, y.values)
            importances = rf.feature_importances_

            # Simple threshold: keep features with importance > mean
            threshold = np.mean(importances)
            selected_features = X.columns[importances > threshold].tolist()

            # Create rankings
            ranks = np.argsort(np.argsort(-importances)) + 1
            feature_ranks = dict(zip(X.columns, ranks))

        if len(selected_features) == 0:
            raise ValueError("No features selected by Boruta. "
                           "Consider increasing max_iter or changing alpha.")

        return PreparedStepBoruta(
            outcome=self.outcome,
            selected_features=selected_features,
            feature_ranks=feature_ranks
        )


@dataclass
class PreparedStepBoruta:
    """Prepared Boruta feature selection step."""
    outcome: str
    selected_features: List[str]
    feature_ranks: Dict[str, int]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Boruta feature selection."""
        result = data.copy()

        # Keep outcome if present
        cols_to_keep = self.selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        # Select only Boruta-selected features
        result = result[[col for col in cols_to_keep if col in result.columns]]

        return result


@dataclass
class StepRfe:
    """Recursive Feature Elimination (RFE) feature selection.

    Recursively removes features and builds models to find the optimal subset.
    Uses model coefficients or feature importances to rank features.

    Args:
        outcome: Name of outcome column (required)
        n_features: Number of features to select (default None = select half)
        step: Number of features to remove at each iteration (default 1)
        estimator: Sklearn estimator to use (default None = LogisticRegression/LinearRegression)

    Example:
        >>> rec = recipe().step_rfe(outcome='y', n_features=10)
    """
    outcome: str
    n_features: Optional[int] = None
    step: int = 1
    estimator: Optional[Any] = None

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepRfe":
        """Prepare RFE feature selection."""
        # Separate predictors and outcome
        X = data.drop(columns=[self.outcome])
        y = data[self.outcome]

        # Determine number of features to select
        if self.n_features is None:
            n_features_to_select = max(1, X.shape[1] // 2)
        else:
            n_features_to_select = min(self.n_features, X.shape[1])

        # Choose estimator if not provided
        if self.estimator is None:
            # Determine if classification or regression
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            else:
                estimator = LinearRegression()
        else:
            estimator = self.estimator

        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=self.step
        )
        rfe.fit(X.values, y.values)

        # Get selected features
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()

        # Get feature rankings
        feature_ranks = dict(zip(X.columns, rfe.ranking_))

        return PreparedStepRfe(
            outcome=self.outcome,
            n_features=n_features_to_select,
            selected_features=selected_features,
            feature_ranks=feature_ranks
        )


@dataclass
class PreparedStepRfe:
    """Prepared RFE feature selection step."""
    outcome: str
    n_features: int
    selected_features: List[str]
    feature_ranks: Dict[str, int]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply RFE feature selection."""
        result = data.copy()

        # Keep outcome if present
        cols_to_keep = self.selected_features.copy()
        if self.outcome in result.columns and self.outcome not in cols_to_keep:
            cols_to_keep.append(self.outcome)

        # Select only RFE-selected features
        result = result[[col for col in cols_to_keep if col in result.columns]]

        return result
