"""
Statistical constraints for genetic algorithm feature selection.

This module provides penalty functions for various statistical constraints
that can be applied during feature selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, spearmanr, kendalltau


class ConstraintEvaluator:
    """
    Evaluate statistical constraints on feature subsets.

    This class computes penalty scores when feature subsets violate
    statistical constraints (p-values, stability, multicollinearity, etc.).

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    outcome_col : str
        Outcome column name
    model_spec : ModelSpec
        Model specification for constraint evaluation
    """

    def __init__(self, data: pd.DataFrame, outcome_col: str, model_spec):
        self.data = data
        self.outcome_col = outcome_col
        self.model_spec = model_spec

    def evaluate_constraints(
        self,
        selected_features: List[str],
        constraints: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate all constraints and return total penalty.

        Parameters
        ----------
        selected_features : List[str]
            Names of selected features
        constraints : Dict
            Dictionary of constraint specifications

        Returns
        -------
        penalty : float
            Total penalty (0 if no violations, positive otherwise)
        """
        if len(selected_features) == 0:
            return np.inf  # Empty feature set is invalid

        total_penalty = 0.0

        # P-value constraint
        if "p_value" in constraints:
            total_penalty += self.p_value_penalty(
                selected_features,
                constraints["p_value"]
            )

        # Coefficient stability constraint
        if "coef_stability" in constraints:
            total_penalty += self.stability_penalty(
                selected_features,
                constraints["coef_stability"]
            )

        # VIF constraint
        if "vif" in constraints:
            total_penalty += self.vif_penalty(
                selected_features,
                constraints["vif"]
            )

        # Effect size constraint
        if "effect_size" in constraints:
            total_penalty += self.effect_size_penalty(
                selected_features,
                constraints["effect_size"]
            )

        # Outcome correlation constraint
        if "outcome_correlation" in constraints:
            total_penalty += self.outcome_correlation_penalty(
                selected_features,
                constraints["outcome_correlation"]
            )

        return total_penalty

    def p_value_penalty(
        self,
        selected_features: List[str],
        constraint: Dict[str, Any]
    ) -> float:
        """
        Penalize if features have p-value > threshold.

        Parameters
        ----------
        selected_features : List[str]
            Selected feature names
        constraint : Dict
            Constraint specification with keys:
            - max: Maximum allowed p-value
            - method: Multiple testing correction ('bonferroni', 'fdr_bh', 'none')

        Returns
        -------
        penalty : float
            Penalty score
        """
        try:
            # Build formula and fit model
            formula = f"{self.outcome_col} ~ {' + '.join(selected_features)}"
            fit = self.model_spec.fit(self.data, formula)

            # Extract coefficients
            _, coeffs, _ = fit.extract_outputs()

            if coeffs is None or len(coeffs) == 0:
                return 0.0  # Model doesn't provide p-values

            # Check if p-values are available
            if "p_value" not in coeffs.columns:
                return 0.0  # Model doesn't support p-values

            # Get p-values (exclude intercept)
            p_values = coeffs[coeffs["term"] != "Intercept"]["p_value"].values

            # Apply multiple testing correction
            method = constraint.get("method", "none")
            alpha = constraint["max"]

            if method == "bonferroni":
                alpha_corrected = alpha / len(p_values)
            elif method == "fdr_bh":
                # Benjamini-Hochberg FDR correction
                reject, pvals_corrected, _, _ = multipletests(
                    p_values, alpha=alpha, method='fdr_bh'
                )
                # Use corrected p-values for penalty
                p_values = pvals_corrected
                alpha_corrected = alpha
            else:  # 'none'
                alpha_corrected = alpha

            # Count violations
            violations = np.sum(p_values > alpha_corrected)

            # Heavy penalty for non-significant features
            penalty = violations * 10.0

            return penalty

        except Exception:
            # If model fitting or extraction fails, return no penalty
            return 0.0

    def stability_penalty(
        self,
        selected_features: List[str],
        constraint: Dict[str, Any]
    ) -> float:
        """
        Penalize if coefficients are unstable across CV folds.

        Parameters
        ----------
        selected_features : List[str]
            Selected feature names
        constraint : Dict
            Constraint specification with keys:
            - min: Minimum stability score [0, 1]
            - method: 'correlation' (Pearson r across folds) or 'cv' (coefficient of variation)
            - cv_folds: Number of CV folds (default: 5)

        Returns
        -------
        penalty : float
            Penalty score
        """
        try:
            cv_folds = constraint.get("cv_folds", 5)
            method = constraint.get("method", "correlation")
            min_stability = constraint["min"]

            formula = f"{self.outcome_col} ~ {' + '.join(selected_features)}"

            # Collect coefficients across folds
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            coef_matrix = []  # rows=folds, cols=features

            for train_idx, _ in kf.split(self.data):
                train_data = self.data.iloc[train_idx]

                # Fit model on fold
                fit = self.model_spec.fit(train_data, formula)

                # Extract coefficients
                _, coeffs, _ = fit.extract_outputs()

                if coeffs is None or len(coeffs) == 0:
                    return 0.0  # Model doesn't provide coefficients

                # Get coefficient values (exclude intercept)
                coef_values = coeffs[coeffs["term"] != "Intercept"]["estimate"].values
                coef_matrix.append(coef_values)

            coef_matrix = np.array(coef_matrix)  # shape: (cv_folds, n_features)

            # Compute stability
            if method == "correlation":
                # Pearson correlation between each pair of folds, then average
                correlations = []
                for i in range(cv_folds):
                    for j in range(i + 1, cv_folds):
                        # Handle case where coefficients are constant
                        if np.std(coef_matrix[i]) == 0 or np.std(coef_matrix[j]) == 0:
                            correlations.append(0.0)
                        else:
                            r = np.corrcoef(coef_matrix[i], coef_matrix[j])[0, 1]
                            correlations.append(r)

                stability = np.mean(correlations) if correlations else 0.0

            elif method == "cv":
                # Coefficient of variation (std/mean) per feature, then average
                # Lower CV = more stable
                cv_scores = np.std(coef_matrix, axis=0) / (np.abs(np.mean(coef_matrix, axis=0)) + 1e-10)
                # Invert so higher = more stable
                stability = 1.0 / (1.0 + np.mean(cv_scores))

            else:
                raise ValueError(f"Unknown stability method: {method}")

            # Penalize if below threshold
            if stability < min_stability:
                penalty = (min_stability - stability) * 20.0
            else:
                penalty = 0.0

            return penalty

        except Exception:
            # If evaluation fails, return no penalty
            return 0.0

    def vif_penalty(
        self,
        selected_features: List[str],
        constraint: Dict[str, Any]
    ) -> float:
        """
        Penalize if VIF (multicollinearity) exceeds threshold.

        Parameters
        ----------
        selected_features : List[str]
            Selected feature names
        constraint : Dict
            Constraint specification with keys:
            - max: Maximum allowed VIF
            - exclude_if_exceeded: If True, use very heavy penalty (effectively exclude)

        Returns
        -------
        penalty : float
            Penalty score
        """
        try:
            if len(selected_features) < 2:
                return 0.0  # VIF requires at least 2 features

            # Get feature matrix
            X = self.data[selected_features].values

            # Check for constant columns or non-numeric data
            if not np.isfinite(X).all():
                return 0.0

            # Compute VIF for each feature
            vif_scores = []
            for i in range(len(selected_features)):
                try:
                    vif = variance_inflation_factor(X, i)
                    if np.isfinite(vif):
                        vif_scores.append(vif)
                except Exception:
                    # Skip features where VIF computation fails
                    continue

            if len(vif_scores) == 0:
                return 0.0

            max_vif = np.max(vif_scores)
            threshold = constraint["max"]

            # Check penalty type
            if constraint.get("exclude_if_exceeded", False):
                # Very heavy penalty (effectively exclude)
                if max_vif > threshold:
                    return 1000.0
                else:
                    return 0.0
            else:
                # Soft penalty proportional to VIF excess
                if max_vif > threshold:
                    return (max_vif - threshold) * 5.0
                else:
                    return 0.0

        except Exception:
            return 0.0

    def effect_size_penalty(
        self,
        selected_features: List[str],
        constraint: Dict[str, Any]
    ) -> float:
        """
        Penalize if effect size is below threshold.

        Parameters
        ----------
        selected_features : List[str]
            Selected feature names
        constraint : Dict
            Constraint specification with keys:
            - min: Minimum effect size
            - method: 'cohens_f2' (for regression)

        Returns
        -------
        penalty : float
            Penalty score
        """
        try:
            method = constraint.get("method", "cohens_f2")

            if method != "cohens_f2":
                raise ValueError(f"Effect size method '{method}' not implemented")

            # Fit full model
            formula_full = f"{self.outcome_col} ~ {' + '.join(selected_features)}"
            fit_full = self.model_spec.fit(self.data, formula_full)

            # Get R² for full model
            _, _, stats_full = fit_full.extract_outputs()
            if stats_full is None or "r_squared" not in stats_full.columns:
                return 0.0  # Can't compute effect size without R²

            r2_full = stats_full["r_squared"].iloc[0]

            # Fit null model (intercept only)
            formula_null = f"{self.outcome_col} ~ 1"
            fit_null = self.model_spec.fit(self.data, formula_null)
            _, _, stats_null = fit_null.extract_outputs()

            r2_null = stats_null["r_squared"].iloc[0] if stats_null is not None else 0.0

            # Cohen's f² = (R²_full - R²_null) / (1 - R²_full)
            if r2_full >= 1.0:
                f2 = np.inf  # Perfect fit
            else:
                f2 = (r2_full - r2_null) / (1.0 - r2_full)

            min_effect = constraint["min"]

            # Penalize if below threshold
            if f2 < min_effect:
                penalty = (min_effect - f2) * 15.0
            else:
                penalty = 0.0

            return penalty

        except Exception:
            return 0.0

    def outcome_correlation_penalty(
        self,
        selected_features: List[str],
        constraint: Dict[str, Any]
    ) -> float:
        """
        Penalize if features have low correlation with outcome.

        Parameters
        ----------
        selected_features : List[str]
            Selected feature names
        constraint : Dict
            Constraint specification with keys:
            - min: Minimum absolute correlation
            - method: 'pearson', 'spearman', or 'kendall'

        Returns
        -------
        penalty : float
            Penalty score
        """
        try:
            method = constraint.get("method", "pearson")
            min_corr = constraint["min"]

            y = self.data[self.outcome_col].values
            violations = 0

            for feature in selected_features:
                x = self.data[feature].values

                # Compute correlation
                if method == "pearson":
                    corr, _ = pearsonr(x, y)
                elif method == "spearman":
                    corr, _ = spearmanr(x, y)
                elif method == "kendall":
                    corr, _ = kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown correlation method: {method}")

                # Check if below threshold
                if abs(corr) < min_corr:
                    violations += 1

            # Penalize violations
            penalty = violations * 5.0

            return penalty

        except Exception:
            return 0.0


def create_constrained_fitness_function(
    data: pd.DataFrame,
    outcome_col: str,
    feature_names: List[str],
    model_spec,
    base_fitness_fn,
    constraints: Dict[str, Dict[str, Any]]
) -> callable:
    """
    Create fitness function that combines performance + constraint penalties.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    outcome_col : str
        Outcome column name
    feature_names : List[str]
        All candidate feature names
    model_spec : ModelSpec
        Model specification
    base_fitness_fn : Callable
        Base fitness function (performance-based)
    constraints : Dict
        Constraint specifications

    Returns
    -------
    fitness_fn : Callable
        Constrained fitness function

    Examples
    --------
    >>> from py_recipes.utils.model_fitness import create_model_fitness_evaluator
    >>> base_fitness = create_model_fitness_evaluator(data, 'y', linear_reg(), 'rmse')
    >>> constraints = {'p_value': {'max': 0.05, 'method': 'bonferroni'}}
    >>> fitness_fn = create_constrained_fitness_function(
    ...     data, 'y', feature_names, linear_reg(), base_fitness, constraints
    ... )
    """
    constraint_evaluator = ConstraintEvaluator(data, outcome_col, model_spec)

    def constrained_fitness(chromosome: np.ndarray) -> float:
        # Get base fitness (performance)
        base_fitness = base_fitness_fn(chromosome)

        if not np.isfinite(base_fitness) or base_fitness <= 0:
            return base_fitness  # Invalid or very poor fitness

        # Get selected features
        selected_features = [
            feature_names[i]
            for i in range(len(chromosome))
            if chromosome[i] == 1
        ]

        # Evaluate constraints
        penalty = constraint_evaluator.evaluate_constraints(
            selected_features,
            constraints
        )

        # Return fitness - penalty
        return base_fitness - penalty

    return constrained_fitness
