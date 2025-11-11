"""
SplitWise: Adaptive dummy encoding for numeric predictors.

Implements the SplitWise regression methodology from:
Kurbucz, Marcell T.; Tzivanakis, Nikolaos; Aslam, Nilufer Sari; Sykulski, Adam M. (2025).
SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding.
arXiv preprint https://arxiv.org/abs/2505.15423
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class StepSplitwise:
    """
    Adaptive dummy encoding for numeric predictors using shallow decision trees.

    SplitWise automatically transforms numeric predictors into either:
    - Binary dummy variables (with 1 or 2 split points)
    - Linear predictors (unchanged)

    The transformation decision is data-driven, based on AIC/BIC improvement.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable (required for supervised transformation)
    transformation_mode : {'univariate', 'iterative'}, default='univariate'
        How to determine transformations:
        - 'univariate': Each predictor evaluated independently
        - 'iterative': Adaptive with partial residuals (not yet implemented)
    min_support : float, default=0.1
        Minimum fraction of observations required in each dummy group.
        Range: (0, 0.5). Prevents highly imbalanced splits.
    min_improvement : float, default=3.0
        Minimum AIC/BIC improvement required to prefer dummy over linear.
        Higher values = more conservative (fewer transformations).
    criterion : {'AIC', 'BIC'}, default='AIC'
        Model selection criterion for comparing transformations
    feature_type : {'dummies', 'interactions', 'both'}, default='dummies'
        Type of features to create:
        - 'dummies': Binary dummy variables only (default)
        - 'interactions': Binary dummy * original feature interactions only
        - 'both': Both dummies and interactions
    exclude_vars : list of str, optional
        Variables forced to stay linear (no transformation)
    columns : selector, optional
        Which columns to consider for transformation. If None, uses all numeric
        predictors except outcome. Supports selector functions.
    skip : bool, default=False
        Skip this step during prep/bake
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.selectors import all_numeric_predictors
    >>>
    >>> # Basic usage with univariate mode
    >>> rec = recipe(data, "price ~ .").step_splitwise(
    ...     outcome='price',
    ...     min_support=0.15,
    ...     min_improvement=2.0
    ... )
    >>>
    >>> # Exclude specific variables from transformation
    >>> rec = recipe(data, "sales ~ .").step_splitwise(
    ...     outcome='sales',
    ...     exclude_vars=['year', 'month']
    ... )

    Notes
    -----
    - Only numeric predictors can be transformed
    - Categorical variables must be pre-encoded (e.g., with step_dummy)
    - This is a supervised step (requires outcome during prep)
    - Shallow trees (max_depth=2) prevent overfitting
    - Support constraint ensures balanced dummy groups

    References
    ----------
    Kurbucz et al. (2025). SplitWise Regression: Stepwise Modeling with
    Adaptive Dummy Encoding. arXiv:2505.15423
    """
    outcome: str
    transformation_mode: Literal['univariate', 'iterative'] = 'univariate'
    min_support: float = 0.1
    min_improvement: float = 3.0
    criterion: Literal['AIC', 'BIC'] = 'AIC'
    feature_type: Literal['dummies', 'interactions', 'both'] = 'dummies'
    exclude_vars: Optional[List[str]] = None
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None

    # Prepared state (stored after prep)
    _decisions: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _cutoffs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _original_columns: List[str] = field(default_factory=list, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Validate parameters
        if not (0 < self.min_support < 0.5):
            raise ValueError(f"min_support must be in (0, 0.5), got {self.min_support}")

        if self.min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {self.min_improvement}")

        if self.transformation_mode not in ['univariate', 'iterative']:
            raise ValueError(
                f"transformation_mode must be 'univariate' or 'iterative', "
                f"got '{self.transformation_mode}'"
            )

        if self.transformation_mode == 'iterative':
            raise NotImplementedError(
                "Iterative mode is not yet implemented. Use transformation_mode='univariate'"
            )

        if self.feature_type not in ['dummies', 'interactions', 'both']:
            raise ValueError(
                f"feature_type must be 'dummies', 'interactions', or 'both', "
                f"got '{self.feature_type}'"
            )

        if self.criterion not in ['AIC', 'BIC']:
            raise ValueError(f"criterion must be 'AIC' or 'BIC', got '{self.criterion}'")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """
        Prepare step by determining optimal transformations for each predictor.

        Parameters
        ----------
        data : DataFrame
            Training data containing predictors and outcome
        training : bool
            Whether this is training data

        Returns
        -------
        self
            Modified step with transformation decisions stored
        """
        if self.skip or not training:
            return self

        # Get outcome
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data columns")

        y = data[self.outcome].values

        # Resolve columns to consider
        from py_recipes.selectors import resolve_selector

        if self.columns is None:
            # All numeric columns except outcome
            candidate_cols = [
                c for c in data.columns
                if c != self.outcome and pd.api.types.is_numeric_dtype(data[c])
            ]
        else:
            candidate_cols = resolve_selector(self.columns, data)
            # Filter to numeric only and exclude outcome
            candidate_cols = [
                c for c in candidate_cols
                if c != self.outcome and pd.api.types.is_numeric_dtype(data[c])
            ]

        if len(candidate_cols) == 0:
            raise ValueError("No numeric predictor columns found for transformation")

        # Build state in local variables (avoid self mutation)
        original_columns = candidate_cols
        decisions = {}
        cutoffs = {}

        # Apply exclusions
        exclude = self.exclude_vars if self.exclude_vars else []
        cols_to_transform = [c for c in candidate_cols if c not in exclude]

        # For excluded vars, force linear
        for col in exclude:
            if col in candidate_cols:
                decisions[col] = 'linear'
                cutoffs[col] = {}

        # Univariate transformation mode
        if self.transformation_mode == 'univariate':
            for col in cols_to_transform:
                decision, cutoff = self._decide_transformation_univariate(
                    data[col].values, y, col
                )
                decisions[col] = decision
                cutoffs[col] = cutoff

        # Create new prepared instance with computed state
        prepared = replace(self)
        prepared._original_columns = original_columns
        prepared._decisions = decisions
        prepared._cutoffs = cutoffs
        prepared._is_prepared = True

        return prepared

    def _decide_transformation_univariate(
        self, x: np.ndarray, y: np.ndarray, var_name: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Decide transformation for a single variable using univariate approach.

        Fits shallow decision tree and compares:
        1. Linear: y ~ x (no transformation)
        2. Single-split dummy: y ~ I(x >= threshold)
        3. Double-split dummy: y ~ I(lower < x < upper)

        Parameters
        ----------
        x : array
            Predictor values
        y : array
            Outcome values
        var_name : str
            Variable name (for reporting)

        Returns
        -------
        decision : str
            One of: 'linear', 'single_split', 'double_split'
        cutoffs : dict
            Cutoff values if dummy selected, empty dict if linear
        """
        # Remove missing values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 10:  # Need minimum observations
            return 'linear', {}

        # Fit shallow decision tree to find potential split points
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(5, int(len(x_clean) * 0.05)))
        tree.fit(x_clean.reshape(-1, 1), y_clean)

        # Extract split points from tree
        split_points = self._extract_split_points(tree, x_clean)

        if len(split_points) == 0:
            # No valid splits found
            return 'linear', {}

        # Compare transformations
        aic_linear = self._compute_aic(x_clean, y_clean, transformation='linear')

        best_aic = aic_linear
        best_decision = 'linear'
        best_cutoffs = {}

        # Try single-split transformations
        for threshold in split_points:
            dummy = (x_clean >= threshold).astype(float)

            # Check support constraint
            support = np.mean(dummy)
            if support < self.min_support or support > (1 - self.min_support):
                continue  # Skip this split due to support violation

            aic = self._compute_aic(dummy, y_clean, transformation='dummy')

            if aic < best_aic - self.min_improvement:
                best_aic = aic
                best_decision = 'single_split'
                best_cutoffs = {'threshold': threshold, 'type': 'greater_equal'}

        # Try double-split (middle region)
        if len(split_points) >= 2:
            for i, lower in enumerate(split_points[:-1]):
                for upper in split_points[i+1:]:
                    dummy = ((x_clean > lower) & (x_clean < upper)).astype(float)

                    # Check support constraint
                    support = np.mean(dummy)
                    if support < self.min_support or support > (1 - self.min_support):
                        continue

                    aic = self._compute_aic(dummy, y_clean, transformation='dummy')

                    if aic < best_aic - self.min_improvement:
                        best_aic = aic
                        best_decision = 'double_split'
                        best_cutoffs = {
                            'lower': lower,
                            'upper': upper,
                            'type': 'between'
                        }

        return best_decision, best_cutoffs

    def _extract_split_points(self, tree: DecisionTreeRegressor, x: np.ndarray) -> List[float]:
        """Extract split thresholds from fitted decision tree."""
        split_points = []

        # Get tree structure
        tree_struct = tree.tree_

        for node_id in range(tree_struct.node_count):
            # Check if this is a split node (not leaf)
            if tree_struct.feature[node_id] != -2:  # -2 indicates leaf
                threshold = tree_struct.threshold[node_id]
                if not np.isnan(threshold):
                    split_points.append(threshold)

        # Remove duplicates and sort
        split_points = sorted(set(split_points))

        # Filter to points within data range
        x_min, x_max = x.min(), x.max()
        split_points = [s for s in split_points if x_min < s < x_max]

        return split_points

    def _compute_aic(
        self, x: np.ndarray, y: np.ndarray, transformation: str = 'linear'
    ) -> float:
        """
        Compute AIC or BIC for linear regression y ~ x.

        Parameters
        ----------
        x : array
            Predictor (can be original numeric or dummy)
        y : array
            Outcome
        transformation : str
            Type of transformation ('linear' or 'dummy')

        Returns
        -------
        float
            AIC or BIC value
        """
        from sklearn.linear_model import LinearRegression

        n = len(y)

        # Fit linear regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)

        # Compute residuals and RSS
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)

        # Number of parameters (intercept + slope)
        k = 2

        # Compute log-likelihood (assuming normal errors)
        sigma2 = rss / n
        if sigma2 <= 0:
            return np.inf

        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        # Compute AIC or BIC
        if self.criterion == 'AIC':
            ic = -2 * log_likelihood + 2 * k
        else:  # BIC
            ic = -2 * log_likelihood + k * np.log(n)

        return ic

    def _sanitize_threshold(self, value: float) -> str:
        """
        Convert threshold value to patsy-friendly string.

        Parameters
        ----------
        value : float
            Threshold value to sanitize

        Returns
        -------
        str
            Sanitized string safe for column names and patsy formulas
        """
        # Format to 4 decimal places
        formatted = f"{value:.4f}"

        # Replace negative sign with 'm' (for minus)
        sanitized = formatted.replace('-', 'm')

        # Replace decimal point with 'p' (for point)
        sanitized = sanitized.replace('.', 'p')

        return sanitized

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to new data.

        Parameters
        ----------
        data : DataFrame
            Data to transform (train or test)

        Returns
        -------
        DataFrame
            Transformed data with dummy variables and/or interactions
        """
        if self.skip or not self._is_prepared:
            return data.copy()

        result = data.copy()

        # Apply transformations
        for col in self._original_columns:
            if col not in result.columns:
                continue

            decision = self._decisions.get(col, 'linear')

            if decision == 'linear':
                # Keep as-is
                continue

            elif decision == 'single_split':
                cutoffs = self._cutoffs[col]
                threshold = cutoffs['threshold']

                # Save original values for potential interaction
                original_values = result[col].copy()

                # Create dummy variable
                dummy = (result[col] >= threshold).astype(int)

                # Sanitize threshold for patsy-friendly column name
                threshold_str = self._sanitize_threshold(threshold)
                dummy_name = f"{col}_ge_{threshold_str}"

                # Add features based on feature_type
                if self.feature_type == 'dummies':
                    result[dummy_name] = dummy
                    result = result.drop(columns=[col])
                elif self.feature_type == 'interactions':
                    interaction_name = f"{dummy_name}_x_{col}"
                    result[interaction_name] = dummy * original_values
                    result = result.drop(columns=[col])
                else:  # 'both'
                    result[dummy_name] = dummy
                    interaction_name = f"{dummy_name}_x_{col}"
                    result[interaction_name] = dummy * original_values
                    result = result.drop(columns=[col])

            elif decision == 'double_split':
                cutoffs = self._cutoffs[col]
                lower = cutoffs['lower']
                upper = cutoffs['upper']

                # Save original values for potential interaction
                original_values = result[col].copy()

                # Create dummy variable (middle region)
                dummy = ((result[col] > lower) & (result[col] < upper)).astype(int)

                # Sanitize thresholds for patsy-friendly column name
                lower_str = self._sanitize_threshold(lower)
                upper_str = self._sanitize_threshold(upper)
                dummy_name = f"{col}_between_{lower_str}_{upper_str}"

                # Add features based on feature_type
                if self.feature_type == 'dummies':
                    result[dummy_name] = dummy
                    result = result.drop(columns=[col])
                elif self.feature_type == 'interactions':
                    interaction_name = f"{dummy_name}_x_{col}"
                    result[interaction_name] = dummy * original_values
                    result = result.drop(columns=[col])
                else:  # 'both'
                    result[dummy_name] = dummy
                    interaction_name = f"{dummy_name}_x_{col}"
                    result[interaction_name] = dummy * original_values
                    result = result.drop(columns=[col])

        return result

    def get_decisions(self) -> Dict[str, Any]:
        """
        Get transformation decisions for all variables.

        Returns
        -------
        dict
            Dictionary mapping variable names to transformation info
        """
        if not self._is_prepared:
            raise ValueError("Step must be prepared before accessing decisions")

        decisions_info = {}
        for col in self._original_columns:
            decisions_info[col] = {
                'decision': self._decisions.get(col, 'linear'),
                'cutoffs': self._cutoffs.get(col, {})
            }

        return decisions_info
