"""
Model fitness evaluation utilities for genetic algorithm feature selection.

This module provides functions to evaluate feature subsets using parsnip models
and cross-validation for robust performance estimation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Callable
from sklearn.model_selection import KFold
from py_yardstick import (
    rmse, mae, mape, r_squared,
    accuracy, precision, recall, f_meas
)


class ModelFitnessEvaluator:
    """
    Evaluate fitness of feature subsets using machine learning models.

    This class handles:
    - Model training and prediction
    - Cross-validation for robust estimates
    - Multiple evaluation metrics
    - Regression and classification tasks

    Parameters
    ----------
    data : pd.DataFrame
        Training data with outcome and predictors
    outcome_col : str
        Name of outcome column
    model_spec : ModelSpec
        Parsnip model specification to use for evaluation
    metric : str
        Metric to optimize ('rmse', 'mae', 'r_squared', 'accuracy', etc.)
    maximize : bool
        Whether to maximize metric (True) or minimize (False)
    cv_folds : int, default=5
        Number of cross-validation folds. If 1, uses train-only evaluation
    random_state : Optional[int]
        Random seed for CV splits

    Attributes
    ----------
    feature_names_ : List[str]
        Names of all candidate features
    n_features_ : int
        Number of candidate features
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        model_spec,
        metric: str = "rmse",
        maximize: bool = False,
        cv_folds: int = 5,
        random_state: Optional[int] = None
    ):
        self.data = data
        self.outcome_col = outcome_col
        self.model_spec = model_spec
        self.metric = metric.lower()
        self.maximize = maximize
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Extract feature names (all columns except outcome)
        self.feature_names_ = [col for col in data.columns if col != outcome_col]
        self.n_features_ = len(self.feature_names_)

        # Validate metric
        self._validate_metric()

    def _validate_metric(self):
        """Validate that metric is supported."""
        supported_metrics = {
            # Regression metrics
            'rmse', 'mae', 'mape', 'r_squared', 'rsq', 'r2',
            # Classification metrics
            'accuracy', 'precision', 'recall', 'f_meas', 'f1'
        }

        if self.metric not in supported_metrics:
            raise ValueError(
                f"Unsupported metric '{self.metric}'. "
                f"Supported: {sorted(supported_metrics)}"
            )

    def _compute_metric(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Compute specified metric given true and predicted values.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns
        -------
        score : float
            Metric score
        """
        # Convert to pandas Series if needed
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_true.index)

        # Compute metric
        if self.metric in ['rmse']:
            result = rmse(y_true, y_pred)
        elif self.metric in ['mae']:
            result = mae(y_true, y_pred)
        elif self.metric in ['mape']:
            result = mape(y_true, y_pred)
        elif self.metric in ['r_squared', 'rsq', 'r2']:
            result = r_squared(y_true, y_pred)
        elif self.metric in ['accuracy']:
            result = accuracy(y_true, y_pred)
        elif self.metric in ['precision']:
            result = precision(y_true, y_pred)
        elif self.metric in ['recall']:
            result = recall(y_true, y_pred)
        elif self.metric in ['f_meas', 'f1']:
            result = f_meas(y_true, y_pred)
        else:
            raise ValueError(f"Metric '{self.metric}' not implemented")

        # Extract scalar value from result DataFrame
        return result.iloc[0]["value"]

    def evaluate_features(
        self,
        feature_indices: np.ndarray
    ) -> float:
        """
        Evaluate a feature subset using the model and metric.

        Parameters
        ----------
        feature_indices : np.ndarray
            Binary array indicating which features to include (1) or exclude (0)

        Returns
        -------
        fitness : float
            Fitness score (higher is better, regardless of metric direction)
        """
        # Convert binary array to feature names
        selected_features = [
            self.feature_names_[i]
            for i in range(len(feature_indices))
            if feature_indices[i] == 1
        ]

        if len(selected_features) == 0:
            # No features selected - return worst possible fitness
            return -np.inf

        # Build formula
        formula = f"{self.outcome_col} ~ {' + '.join(selected_features)}"

        try:
            if self.cv_folds <= 1:
                # No CV - train and evaluate on same data
                score = self._evaluate_no_cv(formula)
            else:
                # Cross-validation
                score = self._evaluate_with_cv(formula)

            # Convert to fitness (higher is better)
            if self.maximize:
                fitness = score
            else:
                # For metrics to minimize (RMSE, MAE), invert and scale
                # Use 1 / (1 + score) to ensure positive fitness
                fitness = 1.0 / (1.0 + abs(score))

            return fitness

        except Exception as e:
            # Model fitting failed - return very low fitness
            return -np.inf

    def _evaluate_no_cv(self, formula: str) -> float:
        """Evaluate on training data without CV (for debugging/speed)."""
        fit = self.model_spec.fit(self.data, formula)
        predictions = fit.predict(self.data)

        y_true = self.data[self.outcome_col]
        y_pred = predictions[".pred"]

        return self._compute_metric(y_true, y_pred)

    def _evaluate_with_cv(self, formula: str) -> float:
        """Evaluate using cross-validation."""
        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scores = []
        for train_idx, val_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx].copy()
            val_data = self.data.iloc[val_idx].copy()

            # Fit model on fold
            fit = self.model_spec.fit(train_data, formula)

            # Predict on validation fold
            predictions = fit.predict(val_data)

            y_true = val_data[self.outcome_col]
            y_pred = predictions[".pred"]

            score = self._compute_metric(y_true, y_pred)
            scores.append(score)

        # Return mean score across folds
        return np.mean(scores)

    def create_fitness_function(self) -> Callable[[np.ndarray], float]:
        """
        Create a fitness function for the genetic algorithm.

        Returns
        -------
        fitness_fn : Callable
            Function that takes a binary chromosome and returns fitness
        """
        def fitness_fn(chromosome: np.ndarray) -> float:
            return self.evaluate_features(chromosome)

        return fitness_fn


def create_model_fitness_evaluator(
    data: pd.DataFrame,
    outcome_col: str,
    model_spec,
    metric: str = "rmse",
    maximize: bool = False,
    cv_folds: int = 5,
    random_state: Optional[int] = None
) -> Callable[[np.ndarray], float]:
    """
    Convenience function to create a fitness evaluator.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    outcome_col : str
        Outcome column name
    model_spec : ModelSpec
        Parsnip model specification
    metric : str
        Metric to optimize
    maximize : bool
        Whether to maximize metric
    cv_folds : int
        CV folds
    random_state : Optional[int]
        Random seed

    Returns
    -------
    fitness_fn : Callable
        Fitness function for GA

    Examples
    --------
    >>> from py_parsnip import linear_reg
    >>> fitness_fn = create_model_fitness_evaluator(
    ...     data=train_data,
    ...     outcome_col='y',
    ...     model_spec=linear_reg(),
    ...     metric='rmse',
    ...     cv_folds=5
    ... )
    >>> # Use with GeneticAlgorithm
    >>> ga = GeneticAlgorithm(n_features=10, fitness_function=fitness_fn)
    >>> best_features, best_fitness, history = ga.evolve()
    """
    evaluator = ModelFitnessEvaluator(
        data=data,
        outcome_col=outcome_col,
        model_spec=model_spec,
        metric=metric,
        maximize=maximize,
        cv_folds=cv_folds,
        random_state=random_state
    )
    return evaluator.create_fitness_function()


def create_importance_based_seeds(
    data: pd.DataFrame,
    outcome_col: str,
    feature_names: List[str],
    n_seeds: int = 5,
    top_n: Optional[int] = None
) -> np.ndarray:
    """
    Create seed chromosomes based on univariate feature importance.

    Ranks features by absolute correlation with outcome and creates
    chromosomes with top K features for different values of K.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    outcome_col : str
        Name of outcome column
    feature_names : List[str]
        List of feature names to consider
    n_seeds : int, default=5
        Number of seed chromosomes to create
    top_n : int, optional
        Maximum features per chromosome (uses n_features//2 if None)

    Returns
    -------
    seed_chromosomes : np.ndarray
        Binary array of shape (n_seeds, n_features)

    Examples
    --------
    >>> seeds = create_importance_based_seeds(
    ...     data=train_data,
    ...     outcome_col='y',
    ...     feature_names=['x1', 'x2', 'x3'],
    ...     n_seeds=3,
    ...     top_n=2
    ... )
    >>> seeds.shape
    (3, 3)
    """
    from scipy.stats import pearsonr
    
    # Calculate correlations
    correlations = {}
    y = data[outcome_col].values
    
    for feature in feature_names:
        x = data[feature].values
        if len(np.unique(x)) > 1:  # Skip constant features
            corr, _ = pearsonr(x, y)
            correlations[feature] = abs(corr)
        else:
            correlations[feature] = 0.0
    
    # Rank features by importance
    ranked_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Create seed chromosomes
    n_features = len(feature_names)
    if top_n is None:
        top_n = max(1, n_features // 2)
    
    seeds = []
    # Create chromosomes with top K features for K in range
    k_values = np.linspace(1, min(top_n, n_features), n_seeds, dtype=int)
    
    for k in k_values:
        chromosome = np.zeros(n_features, dtype=int)
        # Select top k features
        top_k_features = [f for f, _ in ranked_features[:k]]
        for i, feature in enumerate(feature_names):
            if feature in top_k_features:
                chromosome[i] = 1
        seeds.append(chromosome)
    
    return np.array(seeds)


def create_low_correlation_seeds(
    data: pd.DataFrame,
    feature_names: List[str],
    n_seeds: int = 5,
    max_corr_threshold: float = 0.7
) -> np.ndarray:
    """
    Create seed chromosomes with low-correlation feature sets.

    Selects features greedily to minimize inter-feature correlation,
    reducing multicollinearity.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    feature_names : List[str]
        List of feature names to consider
    n_seeds : int, default=5
        Number of seed chromosomes to create
    max_corr_threshold : float, default=0.7
        Maximum allowable correlation between selected features

    Returns
    -------
    seed_chromosomes : np.ndarray
        Binary array of shape (n_seeds, n_features)

    Examples
    --------
    >>> seeds = create_low_correlation_seeds(
    ...     data=train_data,
    ...     feature_names=['x1', 'x2', 'x3'],
    ...     n_seeds=2,
    ...     max_corr_threshold=0.5
    ... )
    """
    from scipy.stats import pearsonr
    
    n_features = len(feature_names)
    
    # Calculate pairwise correlations
    corr_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            x1 = data[feature_names[i]].values
            x2 = data[feature_names[j]].values
            if len(np.unique(x1)) > 1 and len(np.unique(x2)) > 1:
                corr, _ = pearsonr(x1, x2)
                corr_matrix[i, j] = abs(corr)
                corr_matrix[j, i] = abs(corr)
    
    seeds = []
    # Create multiple seeds with different starting features
    start_indices = np.linspace(0, n_features - 1, n_seeds, dtype=int)
    
    for start_idx in start_indices:
        chromosome = np.zeros(n_features, dtype=int)
        selected = []
        
        # Start with a random feature
        selected.append(start_idx)
        chromosome[start_idx] = 1
        
        # Greedily add features with low correlation to selected ones
        while len(selected) < n_features:
            best_feature = None
            min_max_corr = float('inf')
            
            for candidate in range(n_features):
                if candidate in selected:
                    continue
                
                # Check max correlation with already selected features
                max_corr = max([corr_matrix[candidate, s] for s in selected])
                
                if max_corr < min_max_corr:
                    min_max_corr = max_corr
                    best_feature = candidate
            
            # Add if correlation is below threshold
            if best_feature is not None and min_max_corr < max_corr_threshold:
                selected.append(best_feature)
                chromosome[best_feature] = 1
            else:
                break  # No more features meet criteria
        
        seeds.append(chromosome)
    
    return np.array(seeds)
