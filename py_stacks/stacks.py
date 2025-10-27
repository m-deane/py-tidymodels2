"""
Model stacking/ensembling via meta-learning

Combines predictions from multiple base models using elastic net regularization.
"""

from typing import Optional, Any, Tuple, List, Dict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


def stacks() -> "Stacks":
    """Create empty stacking ensemble.

    Returns
    -------
    Stacks
        Empty stacks object ready to receive candidate models

    Examples
    --------
    >>> from py_stacks import stacks
    >>> from py_workflowsets import workflow_set
    >>> from py_tune import tune_grid
    >>>
    >>> # Create workflow set and tune
    >>> wf_set = workflow_set(...)
    >>> results = wf_set.fit_resamples(cv_splits)
    >>>
    >>> # Create stack
    >>> stack = (
    ...     stacks()
    ...     .add_candidates(results)
    ...     .blend_predictions()
    ... )
    >>>
    >>> # Get model weights
    >>> weights = stack.get_model_weights()
    """
    return Stacks()


@dataclass
class Stacks:
    """Model stacking/ensembling container.

    Collects predictions from base models (candidates) and learns optimal
    weights via meta-learning (elastic net regularization).

    Attributes
    ----------
    candidates : list
        List of candidate prediction DataFrames from workflow sets
    meta_learner : sklearn model, optional
        Fitted elastic net meta-learner
    blend_fit : BlendedStack, optional
        Fitted blended ensemble
    """

    candidates: List[pd.DataFrame] = field(default_factory=list)
    candidate_names: List[str] = field(default_factory=list)
    meta_learner: Optional[Any] = None
    blend_fit: Optional["BlendedStack"] = None

    def add_candidates(
        self,
        results,  # WorkflowSetResults or similar
        name: Optional[str] = None
    ) -> "Stacks":
        """Add base model predictions as candidates for stacking.

        Parameters
        ----------
        results : WorkflowSetResults
            Results from workflow_set.fit_resamples() or tune_grid()
            containing cross-validated predictions
        name : str, optional
            Name for this set of candidates. If None, auto-generates.

        Returns
        -------
        Stacks
            Self for method chaining

        Examples
        --------
        >>> from py_stacks import stacks
        >>>
        >>> # Add candidates from multiple workflow sets
        >>> stack = (
        ...     stacks()
        ...     .add_candidates(linear_results, name="linear_models")
        ...     .add_candidates(tree_results, name="tree_models")
        ... )
        """
        # Extract predictions from results
        # This depends on WorkflowSetResults structure
        try:
            predictions = results.collect_predictions()
        except AttributeError:
            # If results doesn't have collect_predictions, assume it's already a DataFrame
            predictions = results

        # Auto-generate name if not provided
        if name is None:
            name = f"candidates_{len(self.candidates) + 1}"

        self.candidates.append(predictions)
        self.candidate_names.append(name)

        return self

    def blend_predictions(
        self,
        penalty: float = 0.01,
        mixture: float = 1.0,
        non_negative: bool = True,
        metric: Optional[Any] = None
    ) -> "BlendedStack":
        """Fit meta-learner to learn optimal blend weights.

        Uses elastic net regularization with optional non-negative constraint
        to learn interpretable weights for combining base model predictions.

        Parameters
        ----------
        penalty : float, default=0.01
            Regularization strength (alpha in sklearn).
            Higher values = more regularization.
        mixture : float, default=1.0
            Elastic net mixing parameter (l1_ratio in sklearn).
            - 1.0 = Lasso (L1 regularization)
            - 0.0 = Ridge (L2 regularization)
            - Between 0 and 1 = Elastic net mix
        non_negative : bool, default=True
            If True, constrains weights to be non-negative.
            Makes interpretation easier (each model contributes positively).
        metric : metric function, optional
            Metric to optimize. If None, uses RMSE for regression.

        Returns
        -------
        BlendedStack
            Fitted ensemble with meta-learner

        Examples
        --------
        >>> stack = (
        ...     stacks()
        ...     .add_candidates(results)
        ...     .blend_predictions(penalty=0.01, mixture=1.0)
        ... )
        >>>
        >>> # Get model weights
        >>> weights = stack.get_model_weights()
        >>> print(weights)
        """
        if len(self.candidates) == 0:
            raise ValueError("No candidates added. Use add_candidates() first.")

        # Prepare meta-features and target
        meta_X, meta_y, feature_names = self._prepare_meta_features()

        # Fit meta-learner
        from sklearn.linear_model import ElasticNet

        meta_learner = ElasticNet(
            alpha=penalty,
            l1_ratio=mixture,
            positive=non_negative,
            fit_intercept=True,
            max_iter=10000,
            random_state=42
        )

        meta_learner.fit(meta_X, meta_y)

        # Store meta-learner
        self.meta_learner = meta_learner

        # Create BlendedStack
        self.blend_fit = BlendedStack(
            stacks=self,
            meta_learner=meta_learner,
            feature_names=feature_names
        )

        return self.blend_fit

    def _prepare_meta_features(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare meta-features from candidate predictions.

        Returns
        -------
        meta_X : DataFrame
            Meta-features (one column per base model prediction)
        meta_y : Series
            Actual values (target for meta-learner)
        feature_names : list of str
            Names of features (model identifiers)
        """
        if len(self.candidates) == 0:
            raise ValueError("No candidates to prepare")

        # Collect all predictions into meta-features
        meta_features = []
        feature_names = []

        for i, (candidate_df, candidate_name) in enumerate(zip(self.candidates, self.candidate_names)):
            # Assuming candidate_df has columns: [.pred, actual, .config, etc.]
            # Each unique .config represents a different model

            if ".config" in candidate_df.columns:
                # Multiple models in this candidate set
                configs = candidate_df[".config"].unique()

                for config in configs:
                    config_preds = candidate_df[candidate_df[".config"] == config]
                    meta_features.append(config_preds[".pred"].values)
                    feature_names.append(f"{candidate_name}_{config}")
            else:
                # Single model in this candidate set
                meta_features.append(candidate_df[".pred"].values)
                feature_names.append(candidate_name)

        # Stack into DataFrame
        meta_X = pd.DataFrame(
            np.column_stack(meta_features),
            columns=feature_names
        )

        # Extract actual values (should be the same across all candidates)
        # Use first candidate's actuals
        if "actual" in self.candidates[0].columns:
            meta_y = pd.Series(self.candidates[0]["actual"].values)
        elif "actuals" in self.candidates[0].columns:
            meta_y = pd.Series(self.candidates[0]["actuals"].values)
        else:
            # Try to find outcome column
            possible_cols = [col for col in self.candidates[0].columns if col not in [".pred", ".config", "split"]]
            if len(possible_cols) > 0:
                meta_y = pd.Series(self.candidates[0][possible_cols[0]].values)
            else:
                raise ValueError("Could not find actual values column in candidates")

        return meta_X, meta_y, feature_names


@dataclass
class BlendedStack:
    """Fitted stacked ensemble.

    Represents a complete ensemble with trained meta-learner that can
    make predictions and report model weights.

    Attributes
    ----------
    stacks : Stacks
        Parent Stacks object with candidate models
    meta_learner : sklearn model
        Fitted elastic net meta-learner
    feature_names : list of str
        Names of base models
    """

    stacks: Stacks
    meta_learner: Any
    feature_names: List[str]

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble predictions.

        Creates meta-features from base model predictions, then predicts
        using the fitted meta-learner.

        Parameters
        ----------
        new_data : DataFrame
            New data to make predictions on

        Returns
        -------
        DataFrame
            Predictions with column ".pred"

        Examples
        --------
        >>> predictions = stack.predict(test_data)
        >>> print(predictions)
        """
        # This is a simplified implementation
        # In practice, we'd need to:
        # 1. Generate predictions from all base models
        # 2. Stack predictions as meta-features
        # 3. Predict with meta-learner

        raise NotImplementedError(
            "predict() requires storing fitted base models. "
            "This is a future enhancement. "
            "Currently, stacks are primarily for analyzing CV results."
        )

    def get_model_weights(self) -> pd.DataFrame:
        """Extract and interpret meta-learner weights.

        Shows contribution of each base model to the ensemble prediction.

        Returns
        -------
        DataFrame
            Model weights with columns: model, weight
            Sorted by weight (descending)

        Examples
        --------
        >>> weights = stack.get_model_weights()
        >>> print(weights)
           model                    weight
        0  linear_models_config_1   0.45
        1  tree_models_config_3     0.35
        2  tree_models_config_1     0.15
        3  linear_models_config_2   0.05
        """
        weights = pd.DataFrame({
            "model": self.feature_names,
            "weight": self.meta_learner.coef_
        })

        # Add intercept info
        intercept_row = pd.DataFrame({
            "model": ["(Intercept)"],
            "weight": [self.meta_learner.intercept_]
        })

        weights = pd.concat([weights, intercept_row], ignore_index=True)

        # Sort by absolute weight (descending)
        weights = weights.iloc[:-1].sort_values("weight", ascending=False, key=abs)

        # Add intercept back at end
        weights = pd.concat([weights, intercept_row], ignore_index=True)

        # Add percentage contribution (excluding intercept)
        total_weight = weights.iloc[:-1]["weight"].abs().sum()
        if total_weight > 0:
            weights.loc[weights.index[:-1], "contribution_pct"] = (
                weights.iloc[:-1]["weight"].abs() / total_weight * 100
            )
        else:
            weights.loc[weights.index[:-1], "contribution_pct"] = 0.0

        weights.loc[weights.index[-1], "contribution_pct"] = np.nan

        return weights

    def get_metrics(self) -> pd.DataFrame:
        """Calculate ensemble performance metrics.

        Returns
        -------
        DataFrame
            Performance metrics (RMSE, MAE, R²) on training/CV data

        Examples
        --------
        >>> metrics = stack.get_metrics()
        >>> print(metrics)
        """
        # Extract meta-features and actuals
        meta_X, meta_y, _ = self.stacks._prepare_meta_features()

        # Generate predictions
        ensemble_preds = self.meta_learner.predict(meta_X)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = pd.DataFrame({
            "metric": ["rmse", "mae", "r_squared"],
            "value": [
                np.sqrt(mean_squared_error(meta_y, ensemble_preds)),
                mean_absolute_error(meta_y, ensemble_preds),
                r2_score(meta_y, ensemble_preds)
            ]
        })

        return metrics

    def compare_to_candidates(self) -> pd.DataFrame:
        """Compare ensemble performance to individual base models.

        Returns
        -------
        DataFrame
            Metrics for ensemble vs each candidate model

        Examples
        --------
        >>> comparison = stack.compare_to_candidates()
        >>> print(comparison)
        """
        # Get ensemble metrics
        ensemble_metrics = self.get_metrics()

        # Calculate metrics for each candidate
        meta_X, meta_y, feature_names = self.stacks._prepare_meta_features()

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        all_metrics = []

        # Ensemble
        ensemble_preds = self.meta_learner.predict(meta_X)
        all_metrics.append({
            "model": "Ensemble",
            "rmse": np.sqrt(mean_squared_error(meta_y, ensemble_preds)),
            "mae": mean_absolute_error(meta_y, ensemble_preds),
            "r_squared": r2_score(meta_y, ensemble_preds)
        })

        # Individual candidates
        for i, feature_name in enumerate(feature_names):
            candidate_preds = meta_X.iloc[:, i]
            all_metrics.append({
                "model": feature_name,
                "rmse": np.sqrt(mean_squared_error(meta_y, candidate_preds)),
                "mae": mean_absolute_error(meta_y, candidate_preds),
                "r_squared": r2_score(meta_y, candidate_preds)
            })

        comparison = pd.DataFrame(all_metrics)
        comparison = comparison.sort_values("rmse")

        return comparison
